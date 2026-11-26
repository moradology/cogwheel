//! Benchmark & bit-exact parity check:
//!   1. `ResampleMode::Nearest` single-band Colormap PNG (`GpuBackend::try_render_bbox_single_band`)
//!      vs CPU `sample_general_scaled(..., ResampleMode::Nearest)`
//!   2. 1-pass fused `GpuBackend::try_render_bbox_single_band_stats` (Nearest) &
//!      `try_render_bbox_expr_stats` (NDVI Nearest) vs CPU `compute_f32_band_stats`.

use std::time::Instant;

use cogzilla_codec::{BandData, BlockStats, DecodedBand};
use cogzilla_kernel::{BandSource, ExprProgram, ResampleMode, sample_general_scaled};
use cogzilla_server::gpu_backend::{BboxCoordSource, GpuBackend};
use cogzilla_server::titiler::{StatsOptions, compute_f32_band_stats};

fn make_band(sw: usize, sh: usize, bias: f32, scale: f32, phase: f32) -> DecodedBand {
    let mut vals = Vec::with_capacity(sw * sh);
    for y in 0..sh {
        let fy = y as f32 / sh as f32;
        for x in 0..sw {
            let fx = x as f32 / sw as f32;
            let v = bias + scale * (0.5 + 0.45 * (fx * 17.0 + phase).sin() * (fy * 13.0).cos());
            vals.push(v.round().clamp(0.0, 65535.0) as u16);
        }
    }
    DecodedBand {
        width: sw as u32,
        height: sh as u32,
        data: BandData::U16 {
            data: vals.into_boxed_slice(),
            nodata: None,
        },
        stats: BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: bias,
            max: bias + scale,
        },
    }
}

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let gpu = GpuBackend::try_init(256 * 1024 * 1024).expect("GPU adapter");
    println!(
        "Adapter: {} ({:?}, unified={})",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.ctx.info.unified_memory
    );

    let sw = 1024usize;
    let sh = 1024usize;
    let b0 = make_band(sw, sh, 1600.0, 1800.0, 1.3); // NIR -> b1 (slot 0)
    let b1 = make_band(sw, sh, 400.0, 600.0, 0.0); // RED -> b2 (slot 1)
    let g0 = [Some(&b0)];
    let g1 = [Some(&b1)];
    let srcs = [
        BandSource::new(sw as u32, sh as u32, sw as u32, sh as u32, 0, 0, 1, 1, &g0),
        BandSource::new(sw as u32, sh as u32, sw as u32, sh as u32, 0, 0, 1, 1, &g1),
    ];
    let prog = ExprProgram::compile("(b1-b2)/(b1+b2)").expect("compile ndvi");
    let opts = StatsOptions {
        percentiles: vec![2.0, 98.0],
        histogram_bins: 10,
        ..StatsOptions::default()
    };

    println!();
    println!(
        "{:<24} {:>10} {:>10} {:>10} {:>10}",
        "Workload (Nearest)", "CPU ms", "GPU ms", "Speedup", "p50 err%"
    );
    println!("{}", "-".repeat(68));

    for &(size, iters) in &[(1024usize, 7usize), (2048, 5), (4096, 3)] {
        let u: Vec<f32> = (0..size)
            .map(|i| (i as f32 / size as f32) * (sw as f32 - 1.0))
            .collect();
        let v: Vec<f32> = (0..size)
            .map(|i| (i as f32 / size as f32) * (sh as f32 - 1.0))
            .collect();
        let coords = BboxCoordSource::Separable { u: &u, v: &v };

        // 1. Single-band Nearest + Stats (1-pass fused GPU)
        let cpu_single = || {
            let mut px = Vec::with_capacity(size * size);
            for &sy in &v {
                for &sx in &u {
                    px.push(sample_general_scaled(&srcs[0], sx, sy, 1.0, 1.0, ResampleMode::Nearest));
                }
            }
            compute_f32_band_stats(&px, "b1", &opts)
        };
        let cpu_j = cpu_single();
        let gpu_s = gpu
            .try_render_bbox_single_band_stats(
                42,
                0,
                0,
                coords,
                &srcs[0],
                ResampleMode::Nearest,
                10,
                None,
                &[2.0, 98.0],
            )
            .expect("gpu single stats");
        let span = (cpu_j["max"].as_f64().unwrap() - cpu_j["min"].as_f64().unwrap()).max(1e-9);
        let p50_err =
            (gpu_s.median.unwrap() - cpu_j["median"].as_f64().unwrap()).abs() / span * 100.0;
        assert!(p50_err < 0.25);

        let mut ct = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = cpu_single();
            ct.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = gpu
                .try_render_bbox_single_band_stats(
                    42,
                    0,
                    0,
                    BboxCoordSource::Separable { u: &u, v: &v },
                    &srcs[0],
                    ResampleMode::Nearest,
                    10,
                    None,
                    &[2.0, 98.0],
                )
                .expect("gpu single stats");
            gt.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let c = median_ms(ct);
        let g = median_ms(gt);
        println!(
            "{:<24} {:>10.2} {:>10.2} {:>9.2}x {:>9.4}%",
            format!("Single {size}x{size}"),
            c,
            g,
            c / g,
            p50_err
        );

        // 2. NDVI Expr Nearest + Stats (1-pass fused GPU)
        let cpu_ndvi = || {
            let mut px = Vec::with_capacity(size * size);
            for &sy in &v {
                for &sx in &u {
                    let n = sample_general_scaled(&srcs[0], sx, sy, 1.0, 1.0, ResampleMode::Nearest);
                    let r = sample_general_scaled(&srcs[1], sx, sy, 1.0, 1.0, ResampleMode::Nearest);
                    px.push((n - r) / (n + r));
                }
            }
            compute_f32_band_stats(&px, "ndvi", &opts)
        };
        let cpu_nj = cpu_ndvi();
        let gpu_ns = gpu
            .try_render_bbox_expr_stats(
                None,
                BboxCoordSource::Separable { u: &u, v: &v },
                &srcs,
                &prog,
                ResampleMode::Nearest,
                10,
                None,
                &[2.0, 98.0],
            )
            .expect("gpu ndvi stats");
        let nspan = (cpu_nj["max"].as_f64().unwrap() - cpu_nj["min"].as_f64().unwrap()).max(1e-9);
        let np50_err =
            (gpu_ns.median.unwrap() - cpu_nj["median"].as_f64().unwrap()).abs() / nspan * 100.0;
        assert!(np50_err < 0.25);

        let mut ct2 = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = cpu_ndvi();
            ct2.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt2 = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = gpu
                .try_render_bbox_expr_stats(
                    None,
                    BboxCoordSource::Separable { u: &u, v: &v },
                    &srcs,
                    &prog,
                    ResampleMode::Nearest,
                    10,
                    None,
                    &[2.0, 98.0],
                )
                .expect("gpu ndvi stats");
            gt2.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let c2 = median_ms(ct2);
        let g2 = median_ms(gt2);
        println!(
            "{:<24} {:>10.2} {:>10.2} {:>9.2}x {:>9.4}%",
            format!("NDVI   {size}x{size}"),
            c2,
            g2,
            c2 / g2,
            np50_err
        );
    }
}
