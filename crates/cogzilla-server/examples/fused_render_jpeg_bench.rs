//! End-to-end benchmark & bit-exact parity verification:
//! 3-band RGB Lanczos3 + ColorFormula (Gamma + Sigmoidal + Saturation) -> Baseline JFIF `.jpg`
//!
//! Compares:
//!   1. CPU (`render_rgb_tile` / separable Lanczos3 + ColorFormula + `jpeg-encoder`)
//!   2. 2-Pass GPU (Coalesced Render -> RGBA8 readback -> RGBA8 re-upload -> `GpuJpegEncoder`)
//!   3. 1-Pass Fused GPU (`CoalescedJob::WithJpeg`: Render + chained `jpeg_dct.wgsl` in 1 submit,
//!      zero RGBA8 readback, zero RGBA8 re-upload, lock-free parallel `DRI`/`RSTm` Huffman)

use std::sync::Arc;
use std::time::Instant;

use cogzilla_gpu::{
    CoalescedJob, GpuCoalescer, GpuColorFormula, GpuColormapMode, GpuContext, GpuExprRenderer,
    GpuJpegEncoder, GpuOptions, OwnedCoordPlan, huffman_encode_coeffs,
};
use cogzilla_kernel::jpeg_fast::encode_jpeg_rgba;
use cogzilla_kernel::ColorFormula;

fn make_3band_slab(w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * w * h];
    for y in 0..h {
        let fy = y as f32 / h as f32;
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let terrain = (fx * 19.0).sin() * (fy * 17.0).cos()
                + 0.4 * ((fx + fy) * 43.0).sin()
                + 0.2 * ((fx * 97.0).cos() * (fy * 89.0).sin());
            let idx = y * w + x;
            out[idx] = 1200.0 + 900.0 * terrain;
            out[w * h + idx] = 1400.0 + 800.0 * (terrain + 0.3 * (fy * 11.0).sin());
            out[2 * w * h + idx] = 1000.0 + 850.0 * (terrain - 0.4 * (fx * 13.0).cos());
        }
    }
    out
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

fn cpu_render_rgb_formula_jpeg(
    bands: &[f32],
    src_w: usize,
    src_h: usize,
    u: &[f32],
    v: &[f32],
    rescale_min: [f32; 3],
    rescale_max: [f32; 3],
    formula: &ColorFormula,
    quality: u8,
) -> Vec<u8> {
    let dst_w = u.len();
    let dst_h = v.len();
    let plane = src_w * src_h;
    let mut rgba = vec![0u8; dst_w * dst_h * 4];
    let inv = [
        255.0 / (rescale_max[0] - rescale_min[0]),
        255.0 / (rescale_max[1] - rescale_min[1]),
        255.0 / (rescale_max[2] - rescale_min[2]),
    ];
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
            let mut rgb = [0.0f32; 3];
            for b in 0..3 {
                let base = b * plane;
                let mut acc = 0.0f32;
                for ky in 0..6 {
                    let yy = (y0 - 2 + ky as isize).clamp(0, (src_h - 1) as isize) as usize;
                    let row = base + yy * src_w;
                    let mut row_acc = 0.0f32;
                    for kx in 0..6 {
                        let xx = (x0 - 2 + kx as isize).clamp(0, (src_w - 1) as isize) as usize;
                        row_acc += bands[row + xx] * wx[kx];
                    }
                    acc += row_acc * wy[ky];
                }
                rgb[b] = ((acc * norm - rescale_min[b]) * inv[b]).clamp(0.0, 255.0);
            }
            let o = (dy * dst_w + dx) * 4;
            rgba[o] = rgb[0].round() as u8;
            rgba[o + 1] = rgb[1].round() as u8;
            rgba[o + 2] = rgb[2].round() as u8;
            rgba[o + 3] = 255;
        }
    }
    formula.apply_buffer_in_place(&mut rgba, 4);
    encode_jpeg_rgba(dst_w as u32, dst_h as u32, &rgba, quality).to_vec()
}

fn decode_psnr(jpg_a: &[u8], jpg_b: &[u8]) -> f64 {
    let mut da = zune_jpeg::JpegDecoder::new(jpg_a);
    let mut db = zune_jpeg::JpegDecoder::new(jpg_b);
    let ra = da.decode().expect("decode a");
    let rb = db.decode().expect("decode b");
    assert_eq!(ra.len(), rb.len());
    let mut mse = 0.0f64;
    for (&a, &b) in ra.iter().zip(rb.iter()) {
        let d = a as f64 - b as f64;
        mse += d * d;
    }
    mse /= ra.len() as f64;
    10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
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
    let mut jpeg_enc = GpuJpegEncoder::new(&ctx);

    let src_w = 1024usize;
    let src_h = 1024usize;
    let bands = make_3band_slab(src_w, src_h);
    let plane = src_w * src_h;
    let slab = Arc::new(
        expr.upload_bands(
            &[&bands[..plane], &bands[plane..2 * plane], &bands[2 * plane..]],
            src_w as u32,
            src_h as u32,
            src_w as u32,
        )
        .expect("upload 3-band slab"),
    );

    let rescale_min = [200.0f32, 200.0, 200.0];
    let rescale_max = [2600.0f32, 2600.0, 2600.0];
    let formula = ColorFormula::parse("Gamma RGB 1.15 Sigmoidal RGB 6 0.45 Saturation 1.25")
        .expect("parse formula");
    let (lut_opt, saturation) = formula.as_lut_then_saturation().expect("gpu formula");
    let gpu_formula = GpuColorFormula {
        rgb_lut: lut_opt.map(|(r, g, b)| GpuColorFormula::pack_rgb_luts(r, g, b)),
        saturation,
    };
    let q = 85u8;

    println!();
    println!(
        "{:<10} {:>10} {:>11} {:>11} {:>10} {:>11} {:>10}",
        "Size", "CPU ms", "2-Pass ms", "Fused ms", "vs CPU", "vs 2-Pass", "CPU PSNR"
    );
    println!("{}", "-".repeat(80));

    for &(size, iters) in &[(1024usize, 7usize), (2048, 5), (4096, 3)] {
        let u: Vec<f32> = (0..size)
            .map(|i| 4.0 + (i as f32 / size as f32) * (src_w as f32 - 8.0))
            .collect();
        let v: Vec<f32> = (0..size)
            .map(|i| 4.0 + (i as f32 / size as f32) * (src_h as f32 - 8.0))
            .collect();
        let mk_job = || CoalescedJob::Rgb {
            slab: Arc::clone(&slab),
            plan: OwnedCoordPlan::Separable1D {
                u: u.clone(),
                v: v.clone(),
            },
            mode: GpuColormapMode::Lanczos3,
            rescale_min,
            rescale_max,
            formula: gpu_formula.clone(),
        };

        // Warmup + parity check
        let cpu_jpg = cpu_render_rgb_formula_jpeg(
            &bands,
            src_w,
            src_h,
            &u,
            &v,
            rescale_min,
            rescale_max,
            &formula,
            q,
        );
        let two_pass_rgba = coalescer.submit(mk_job()).expect("2-pass render");
        let two_pass_jpg = jpeg_enc
            .encode_rgba8(&two_pass_rgba, size as u32, size as u32, q)
            .expect("2-pass jpeg");
        let fused_coeffs = coalescer
            .submit(CoalescedJob::WithJpeg {
                inner: Box::new(mk_job()),
                quality: q,
            })
            .expect("fused render+jpeg");
        let fused_jpg = huffman_encode_coeffs(&fused_coeffs, size as u32, size as u32, q);

        assert_eq!(
            two_pass_jpg, fused_jpg,
            "1-pass fused Render+JPEG must be bit-for-bit identical to 2-pass GPU!"
        );
        let psnr = decode_psnr(&cpu_jpg, &fused_jpg);
        assert!(psnr > 40.0, "Fused GPU vs CPU JPEG PSNR too low: {psnr:.2} dB");

        let mut cpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = cpu_render_rgb_formula_jpeg(
                &bands,
                src_w,
                src_h,
                &u,
                &v,
                rescale_min,
                rescale_max,
                &formula,
                q,
            );
            cpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut two_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let rgba = coalescer.submit(mk_job()).expect("2-pass");
            let _ = jpeg_enc
                .encode_rgba8(&rgba, size as u32, size as u32, q)
                .expect("jpeg");
            two_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut fused_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let coeffs = coalescer
                .submit(CoalescedJob::WithJpeg {
                    inner: Box::new(mk_job()),
                    quality: q,
                })
                .expect("fused");
            let _ = huffman_encode_coeffs(&coeffs, size as u32, size as u32, q);
            fused_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let c = median_ms(cpu_times);
        let t2 = median_ms(two_times);
        let f1 = median_ms(fused_times);
        println!(
            "{:<10} {:>10.2} {:>11.2} {:>11.2} {:>9.2}x {:>10.2}x {:>9.2}dB",
            format!("{size}x{size}"),
            c,
            t2,
            f1,
            c / f1,
            t2 / f1,
            psnr
        );
    }
}
