//! End-to-end benchmark for zero-copy parallel `GpuMosaicCompositor::upload_scenes`,
//! 1-submit batched multi-band `try_composite_mosaic_f32_bands`, and 1-pass fused
//! Mosaic -> GPU JPEG (`try_composite_mosaic_styled` with `OutputFormat::Jpeg`).
//!
//! Run:
//!   cargo run --release -p cogzilla-server --example e2e_mosaic_fused_bench

use std::time::Instant;

use cogzilla_gpu::GpuPixelSelection;
use cogzilla_kernel::jpeg_fast::encode_jpeg_rgba;
use cogzilla_kernel::{ColormapName, OutputFormat};
use cogzilla_server::gpu_backend::GpuBackend;

fn make_scenes(w: u32, h: u32, n_scenes: usize, seed: usize) -> Vec<Vec<f32>> {
    let n = (w * h) as usize;
    (0..n_scenes)
        .map(|s| {
            let mut p = vec![0.0f32; n];
            for (i, dst) in p.iter_mut().enumerate() {
                if (i + s * 37 + seed * 13) % 8 == 0 {
                    *dst = f32::NAN;
                } else {
                    *dst = 50.0 + ((i * 11 + s * 73 + seed * 29) % 200) as f32;
                }
            }
            p
        })
        .collect()
}

fn cpu_median_f32(scenes: &[Vec<f32>]) -> Vec<f32> {
    let n = scenes[0].len();
    let mut out = vec![f32::NAN; n];
    let mut scratch = Vec::with_capacity(scenes.len());
    for i in 0..n {
        scratch.clear();
        for s in scenes {
            let v = s[i];
            if v.is_finite() {
                scratch.push(v);
            }
        }
        let k = scratch.len();
        if k > 0 {
            scratch.sort_unstable_by(|a, b| a.total_cmp(b));
            out[i] = if k % 2 == 1 {
                scratch[k / 2]
            } else {
                0.5 * (scratch[k / 2 - 1] + scratch[k / 2])
            };
        }
    }
    out
}

fn cpu_median_jpeg(scenes: &[Vec<f32>], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let med = cpu_median_f32(scenes);
    let lut = ColormapName::Viridis.lut();
    let inv = 255.0f32 / 200.0;
    let mut rgba = vec![0u8; med.len() * 4];
    for (i, &v) in med.iter().enumerate() {
        let off = i * 4;
        if v.is_finite() {
            let idx = ((v - 50.0) * inv).clamp(0.0, 255.0) as usize;
            let rgb = lut.rgb[idx];
            rgba[off] = rgb[0];
            rgba[off + 1] = rgb[1];
            rgba[off + 2] = rgb[2];
            rgba[off + 3] = 255;
        }
    }
    encode_jpeg_rgba(w, h, &rgba, quality).to_vec()
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn main() {
    let gpu = GpuBackend::try_init(256 * 1024 * 1024).expect("GPU adapter required");
    println!(
        "GPU: {} ({:?}, unified_memory={})",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.ctx.info.unified_memory
    );
    println!();

    // 1. Batched multi-band median (1 submit vs N submits vs CPU)
    println!("=== Multi-Band Median Mosaic (1-submit batched vs per-band submits vs CPU) ===");
    for &(dim, n_scenes, n_bands, iters) in &[
        (512u32, 12usize, 3usize, 10usize),
        (1024, 12, 3, 7),
        (1024, 12, 4, 7),
    ] {
        let bands: Vec<Vec<Vec<f32>>> = (0..n_bands)
            .map(|b| make_scenes(dim, dim, n_scenes, b))
            .collect();
        let band_refs: Vec<Vec<&[f32]>> = bands
            .iter()
            .map(|b| b.iter().map(Vec::as_slice).collect())
            .collect();

        // Warmup
        let _ = gpu.try_composite_mosaic_f32_bands(
            dim,
            dim,
            &band_refs,
            GpuPixelSelection::Median,
        );

        let mut cpu_times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut sum = 0usize;
            for b in &bands {
                sum += cpu_median_f32(b).len();
            }
            std::hint::black_box(sum);
            cpu_times.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        let mut seq_times = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let mut sum = 0usize;
            for refs in &band_refs {
                let v = gpu
                    .try_composite_mosaic_f32(dim, dim, refs, GpuPixelSelection::Median)
                    .expect("seq GPU dispatch");
                sum += v.len();
            }
            std::hint::black_box(sum);
            seq_times.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        let mut batch_times = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let out = gpu
                .try_composite_mosaic_f32_bands(
                    dim,
                    dim,
                    &band_refs,
                    GpuPixelSelection::Median,
                )
                .expect("batched GPU dispatch");
            std::hint::black_box(out.len());
            batch_times.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        let cpu_ms = median_ms(cpu_times);
        let seq_ms = median_ms(seq_times);
        let bat_ms = median_ms(batch_times);
        println!(
            "{dim}x{dim} x {n_scenes} scenes x {n_bands} bands: CPU={cpu_ms:7.2} ms | GPU seq({n_bands} submits)={seq_ms:6.2} ms | GPU batched(1 submit)={bat_ms:6.2} ms ({:.2}x vs seq, {:.1}x vs CPU)",
            seq_ms / bat_ms,
            cpu_ms / bat_ms
        );
    }

    println!();
    println!("=== Styled Median Mosaic -> JPEG (1-pass fused GPU JPEG vs 2-pass GPU RGBA8+CPU JPEG vs CPU) ===");
    for &(dim, n_scenes, iters) in &[(512u32, 12usize, 10usize), (1024, 12, 7), (2048, 8, 5)] {
        let scenes = make_scenes(dim, dim, n_scenes, 0);
        let refs: Vec<&[f32]> = scenes.iter().map(Vec::as_slice).collect();

        // Warmup
        let _ = gpu.try_composite_mosaic_styled(
            dim,
            dim,
            &refs,
            GpuPixelSelection::Median,
            (50.0, 250.0),
            ColormapName::Viridis,
            OutputFormat::Jpeg { quality: 85 },
            false,
        );

        let mut cpu_times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let jpg = cpu_median_jpeg(&scenes, dim, dim, 85);
            std::hint::black_box(jpg.len());
            cpu_times.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        // Old 2-pass baseline: GPU Mosaic -> RGBA8 readback -> CPU JPEG encode
        let mut two_pass_times = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let rgba = gpu
                .try_composite_mosaic_styled(
                    dim,
                    dim,
                    &refs,
                    GpuPixelSelection::Median,
                    (50.0, 250.0),
                    ColormapName::Viridis,
                    OutputFormat::RawRgba,
                    false,
                )
                .expect("GPU RawRgba dispatch");
            let jpg = encode_jpeg_rgba(dim, dim, &rgba, 85);
            std::hint::black_box(jpg.len());
            two_pass_times.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        // New 1-pass fused: GPU Mosaic -> GPU JPEG FDCT in 1 submit
        let mut fused_times = Vec::new();
        let mut jpg_bytes = 0usize;
        for _ in 0..iters {
            let t0 = Instant::now();
            let jpg = gpu
                .try_composite_mosaic_styled(
                    dim,
                    dim,
                    &refs,
                    GpuPixelSelection::Median,
                    (50.0, 250.0),
                    ColormapName::Viridis,
                    OutputFormat::Jpeg { quality: 85 },
                    false,
                )
                .expect("GPU fused JPEG dispatch");
            jpg_bytes = jpg.len();
            std::hint::black_box(jpg_bytes);
            fused_times.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        let cpu_ms = median_ms(cpu_times);
        let two_ms = median_ms(two_pass_times);
        let fus_ms = median_ms(fused_times);
        println!(
            "{dim}x{dim} x {n_scenes} scenes -> JPEG ({} KB): CPU={cpu_ms:7.2} ms | GPU 2-pass(RGBA8+CPU JPEG)={two_ms:6.2} ms | GPU 1-pass fused={fus_ms:6.2} ms ({:.2}x vs 2-pass, {:.1}x vs CPU)",
            jpg_bytes / 1024,
            two_ms / fus_ms,
            cpu_ms / fus_ms
        );
    }

    println!();
    println!("=== 3-Band RGB Median Mosaic -> JPEG (1-pass fused RGB->GPU JPEG vs 2-pass 3xf32+CPU rescale+CPU JPEG vs CPU) ===");
    for &(dim, n_scenes, iters) in &[(512u32, 12usize, 10usize), (1024, 12, 7), (2048, 8, 5)] {
        let r_scenes = make_scenes(dim, dim, n_scenes, 0);
        let g_scenes = make_scenes(dim, dim, n_scenes, 1);
        let b_scenes = make_scenes(dim, dim, n_scenes, 2);
        let r_refs: Vec<&[f32]> = r_scenes.iter().map(Vec::as_slice).collect();
        let g_refs: Vec<&[f32]> = g_scenes.iter().map(Vec::as_slice).collect();
        let b_refs: Vec<&[f32]> = b_scenes.iter().map(Vec::as_slice).collect();
        let rgb_planes: [&[&[f32]]; 3] = [&r_refs, &g_refs, &b_refs];
        let band_vecs: Vec<Vec<&[f32]>> = vec![r_refs.clone(), g_refs.clone(), b_refs.clone()];
        let rmin = [50.0f32; 3];
        let rmax = [250.0f32; 3];
        let inv = 255.0f32 / 200.0;
        let n_px = (dim * dim) as usize;

        // Warmup
        let _ = gpu.try_composite_mosaic_rgb_styled(
            dim,
            dim,
            &rgb_planes,
            GpuPixelSelection::Median,
            rmin,
            rmax,
            OutputFormat::Jpeg { quality: 85 },
            false,
        );

        let mut cpu_times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let mr = cpu_median_f32(&r_scenes);
            let mg = cpu_median_f32(&g_scenes);
            let mb = cpu_median_f32(&b_scenes);
            let mut rgba = vec![0u8; n_px * 4];
            for i in 0..n_px {
                if mr[i].is_finite() && mg[i].is_finite() && mb[i].is_finite() {
                    rgba[i * 4] = ((mr[i] - 50.0) * inv).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 1] = ((mg[i] - 50.0) * inv).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 2] = ((mb[i] - 50.0) * inv).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 3] = 255;
                }
            }
            let jpg = encode_jpeg_rgba(dim, dim, &rgba, 85);
            std::hint::black_box(jpg.len());
            cpu_times.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        // Old 2-pass: GPU batched 3xf32 readback + CPU RGB rescale + CPU JPEG
        let mut two_pass_times = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let bands = gpu
                .try_composite_mosaic_f32_bands(dim, dim, &band_vecs, GpuPixelSelection::Median)
                .expect("batched f32 dispatch");
            let mut rgba = vec![0u8; n_px * 4];
            for i in 0..n_px {
                let r = bands[0][i];
                let g = bands[1][i];
                let b = bands[2][i];
                if r.is_finite() && g.is_finite() && b.is_finite() {
                    rgba[i * 4] = ((r - 50.0) * inv).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 1] = ((g - 50.0) * inv).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 2] = ((b - 50.0) * inv).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 3] = 255;
                }
            }
            let jpg = encode_jpeg_rgba(dim, dim, &rgba, 85);
            std::hint::black_box(jpg.len());
            two_pass_times.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        // New 1-pass fused: GPU RGB Mosaic -> GPU JPEG FDCT in 1 submit
        let mut fused_times = Vec::new();
        let mut jpg_bytes = 0usize;
        for _ in 0..iters {
            let t0 = Instant::now();
            let jpg = gpu
                .try_composite_mosaic_rgb_styled(
                    dim,
                    dim,
                    &rgb_planes,
                    GpuPixelSelection::Median,
                    rmin,
                    rmax,
                    OutputFormat::Jpeg { quality: 85 },
                    false,
                )
                .expect("GPU fused RGB JPEG dispatch");
            jpg_bytes = jpg.len();
            std::hint::black_box(jpg_bytes);
            fused_times.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        let cpu_ms = median_ms(cpu_times);
        let two_ms = median_ms(two_pass_times);
        let fus_ms = median_ms(fused_times);
        println!(
            "{dim}x{dim} x {n_scenes} scenes x RGB -> JPEG ({} KB): CPU={cpu_ms:7.2} ms | GPU 2-pass(3xf32+CPU JPEG)={two_ms:6.2} ms | GPU 1-pass fused={fus_ms:6.2} ms ({:.2}x vs 2-pass, {:.1}x vs CPU)",
            jpg_bytes / 1024,
            two_ms / fus_ms,
            cpu_ms / fus_ms
        );
    }
}
