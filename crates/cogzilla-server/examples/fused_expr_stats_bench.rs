//! End-to-end benchmark & exact parity verification:
//! 2-band Lanczos3 NDVI `(b2 - b1) / (b2 + b1)` -> TiTiler Band Statistics + Histogram + Percentiles
//!
//! Compares:
//!   1. CPU (Lanczos3 2-band NDVI + `compute_f32_band_stats` with `sort_unstable_by`)
//!   2. 2-Pass GPU (`CoalescedJob::Expr { RawF32 }` -> 4-67 MB readback -> 4-67 MB re-upload -> `GpuStatsReducer::reduce_f32`)
//!   3. 1-Pass Fused GPU (`CoalescedJob::WithStats`: ExprVM + 3-stage `stats.wgsl` in ONE submit,
//!      ZERO f32 readback, ZERO f32 re-upload, 12 KB total readback -> `decode_stats_readback`)

use std::sync::Arc;
use std::time::Instant;

use cogzilla_gpu::{
    CoalescedJob, CoalescedOutput, GpuCoalescer, GpuColormapMode, GpuContext, GpuExprRenderer,
    GpuInstr, GpuOptions, GpuStatsReducer, OwnedCoordPlan, decode_stats_readback, opcodes,
};
use cogzilla_server::titiler::{StatsOptions, compute_f32_band_stats};

fn make_nir_red_bands(w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut red = Vec::with_capacity(n);
    let mut nir = Vec::with_capacity(n);
    for y in 0..h {
        let fy = y as f32 / h as f32;
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let veg = 0.5 + 0.45 * (fx * 17.0).sin() * (fy * 13.0).cos()
                + 0.15 * ((fx + fy) * 37.0).sin();
            red.push(600.0 + 400.0 * (1.0 - veg));
            nir.push(1800.0 + 1600.0 * veg);
        }
    }
    (red, nir)
}

fn lanczos3_weight(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 1e-6 {
        return 1.0;
    }
    if ax >= 3.0 {
        return 0.0;
    }
    let pi_x = std::f32::consts::PI * ax;
    let pi_x3 = pi_x / 3.0;
    (pi_x.sin() / pi_x) * (pi_x3.sin() / pi_x3)
}

fn cpu_ndvi_lanczos_stats(
    red: &[f32],
    nir: &[f32],
    src_w: usize,
    src_h: usize,
    u: &[f32],
    v: &[f32],
    opts: &StatsOptions,
) -> serde_json::Value {
    let dst_w = u.len();
    let dst_h = v.len();
    let mut ndvi = vec![0.0f32; dst_w * dst_h];
    for (dy, &sy) in v.iter().enumerate() {
        let y0 = sy.floor() as isize;
        let mut wy = [0.0f32; 6];
        let mut wysum = 0.0f32;
        for k in 0..6 {
            let w = lanczos3_weight(sy - (y0 - 2 + k as isize) as f32);
            wy[k] = w;
            wysum += w;
        }
        for (dx, &sx) in u.iter().enumerate() {
            let x0 = sx.floor() as isize;
            let mut wx = [0.0f32; 6];
            let mut wxsum = 0.0f32;
            for k in 0..6 {
                let w = lanczos3_weight(sx - (x0 - 2 + k as isize) as f32);
                wx[k] = w;
                wxsum += w;
            }
            let norm = 1.0 / (wxsum * wysum);
            let mut r_acc = 0.0f32;
            let mut n_acc = 0.0f32;
            for ky in 0..6 {
                let yy = (y0 - 2 + ky as isize).clamp(0, (src_h - 1) as isize) as usize;
                let row = yy * src_w;
                let mut rr = 0.0f32;
                let mut nn = 0.0f32;
                for kx in 0..6 {
                    let xx = (x0 - 2 + kx as isize).clamp(0, (src_w - 1) as isize) as usize;
                    rr += red[row + xx] * wx[kx];
                    nn += nir[row + xx] * wx[kx];
                }
                r_acc += rr * wy[ky];
                n_acc += nn * wy[ky];
            }
            let r = r_acc * norm;
            let n = n_acc * norm;
            ndvi[dy * dst_w + dx] = (n - r) / (n + r);
        }
    }
    compute_f32_band_stats(&ndvi, "(b2-b1)/(b2+b1)", opts)
}

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let ctx = GpuContext::new(GpuOptions::default()).expect("GPU adapter");
    println!(
        "Adapter: {} ({:?}, unified={})",
        ctx.info.name, ctx.info.backend, ctx.info.unified_memory
    );
    let expr = GpuExprRenderer::new(&ctx);
    let coalescer = GpuCoalescer::new(&ctx, 16, 100);
    let mut reducer = GpuStatsReducer::new(&ctx);

    let src_w = 1024usize;
    let src_h = 1024usize;
    let (red, nir) = make_nir_red_bands(src_w, src_h);
    let slab = Arc::new(
        expr.upload_bands(&[&red, &nir], src_w as u32, src_h as u32, src_w as u32)
            .expect("upload 2-band slab"),
    );

    // Bytecode for (b1 - b0) / (b1 + b0) = (NIR - RED) / (NIR + RED)
    let instrs = vec![
        GpuInstr::load_band(1),
        GpuInstr::load_band(0),
        GpuInstr::op(opcodes::SUB),
        GpuInstr::load_band(1),
        GpuInstr::load_band(0),
        GpuInstr::op(opcodes::ADD),
        GpuInstr::op(opcodes::DIV),
    ];
    let opts = StatsOptions {
        percentiles: vec![2.0, 98.0],
        histogram_bins: 10,
        ..StatsOptions::default()
    };

    println!();
    println!(
        "{:<10} {:>10} {:>11} {:>11} {:>10} {:>11} {:>10}",
        "Size", "CPU ms", "2-Pass ms", "Fused ms", "vs CPU", "vs 2-Pass", "p50 err%"
    );
    println!("{}", "-".repeat(80));

    for &(size, iters) in &[(1024usize, 7usize), (2048, 5), (4096, 3)] {
        let u: Vec<f32> = (0..size)
            .map(|i| 4.0 + (i as f32 / size as f32) * (src_w as f32 - 8.0))
            .collect();
        let v: Vec<f32> = (0..size)
            .map(|i| 4.0 + (i as f32 / size as f32) * (src_h as f32 - 8.0))
            .collect();
        let mk_job = || CoalescedJob::Expr {
            slab: Arc::clone(&slab),
            plan: OwnedCoordPlan::Separable1D {
                u: u.clone(),
                v: v.clone(),
            },
            mode: GpuColormapMode::Lanczos3,
            instrs: instrs.clone(),
            rescale_min: -1.0,
            rescale_max: 1.0,
            lut_u32: [0u32; 256],
            output: CoalescedOutput::RawF32,
        };

        // Warmup + exact parity check
        let cpu_json = cpu_ndvi_lanczos_stats(&red, &nir, src_w, src_h, &u, &v, &opts);
        let two_raw = coalescer.submit(mk_job()).expect("2-pass expr");
        let two_f32: Vec<f32> = two_raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let two_stats = reducer
            .reduce_f32(&two_f32, None, 10, None, &[2.0, 98.0])
            .expect("2-pass stats");

        let fused_raw = coalescer
            .submit(CoalescedJob::WithStats {
                inner: Box::new(mk_job()),
                nodata: None,
                num_bins: 10,
                hist_range: None,
            })
            .expect("fused expr+stats");
        let fused_stats = decode_stats_readback(&fused_raw, size * size, 10, None, &[2.0, 98.0]);

        assert_eq!(two_stats.valid_pixels, fused_stats.valid_pixels);
        assert_eq!(two_stats.hist_counts, fused_stats.hist_counts);
        assert!((two_stats.min.unwrap() - fused_stats.min.unwrap()).abs() < 1e-6);
        assert!((two_stats.max.unwrap() - fused_stats.max.unwrap()).abs() < 1e-6);
        assert!((two_stats.median.unwrap() - fused_stats.median.unwrap()).abs() < 1e-6);

        let cpu_min = cpu_json["min"].as_f64().unwrap();
        let cpu_max = cpu_json["max"].as_f64().unwrap();
        let cpu_p50 = cpu_json["median"].as_f64().unwrap();
        let span = (cpu_max - cpu_min).max(1e-9);
        let p50_err_pct = (fused_stats.median.unwrap() - cpu_p50).abs() / span * 100.0;
        assert!(p50_err_pct < 0.25, "p50 error too large: {p50_err_pct:.4}%");

        let mut cpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = cpu_ndvi_lanczos_stats(&red, &nir, src_w, src_h, &u, &v, &opts);
            cpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut two_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let raw = coalescer.submit(mk_job()).expect("2-pass");
            let vals: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let _ = reducer
                .reduce_f32(&vals, None, 10, None, &[2.0, 98.0])
                .expect("stats");
            two_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut fused_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let raw = coalescer
                .submit(CoalescedJob::WithStats {
                    inner: Box::new(mk_job()),
                    nodata: None,
                    num_bins: 10,
                    hist_range: None,
                })
                .expect("fused");
            let _ = decode_stats_readback(&raw, size * size, 10, None, &[2.0, 98.0]);
            fused_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let c = median_ms(cpu_times);
        let t2 = median_ms(two_times);
        let f1 = median_ms(fused_times);
        println!(
            "{:<10} {:>10.2} {:>11.2} {:>11.2} {:>9.2}x {:>10.2}x {:>9.4}%",
            format!("{size}x{size}"),
            c,
            t2,
            f1,
            c / f1,
            t2 / f1,
            p50_err_pct
        );
    }
}
