//! End-to-end server mosaic pixel-selection (`Median`, `Stdev`, `Mean`) CPU vs GPU+PNG benchmark.
//!
//! Benchmarks `GpuBackend::try_composite_mosaic_f32` and `try_composite_mosaic_styled` (including
//! scene stack upload, WGSL reduction + colormap, readback, and indexed PNG encoding) against the
//! exact CPU pixel-selection loops in `cogzilla-server::http::render_mosaic_tile_pixel_selection`.

use std::time::Instant;

use cogzilla_gpu::GpuPixelSelection;
use cogzilla_kernel::{ColormapName, OutputFormat};
use cogzilla_server::gpu_backend::GpuBackend;

fn cpu_median_f32(scenes: &[Vec<f32>], n_px: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; n_px];
    let mut scratch = Vec::with_capacity(scenes.len());
    for i in 0..n_px {
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

fn cpu_stdev_f32(scenes: &[Vec<f32>], n_px: usize) -> Vec<f32> {
    let mut mean = vec![0.0f64; n_px];
    let mut m2 = vec![0.0f64; n_px];
    let mut cnt = vec![0u32; n_px];
    for s in scenes {
        for i in 0..n_px {
            let v = s[i];
            if v.is_finite() {
                cnt[i] += 1;
                let vf = v as f64;
                let d = vf - mean[i];
                mean[i] += d / (cnt[i] as f64);
                let d2 = vf - mean[i];
                m2[i] += d * d2;
            }
        }
    }
    (0..n_px)
        .map(|i| {
            if cnt[i] > 0 {
                (m2[i] / (cnt[i] as f64)).sqrt() as f32
            } else {
                f32::NAN
            }
        })
        .collect()
}

fn cpu_style_and_png(vals: &[f32], w: u32, h: u32, rmin: f32, rmax: f32, cmap: ColormapName) -> usize {
    let n_px = (w * h) as usize;
    let lut = cmap.lut();
    let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rmin, rmax);
    let mut rgba = vec![0u8; n_px * 4];
    for i in 0..n_px {
        let v = vals[i];
        if v.is_finite() {
            let idx = ((v - rmin) * inv).clamp(0.0, 255.0) as usize;
            let px = lut.rgba_u32[idx].to_le_bytes();
            rgba[i * 4..i * 4 + 4].copy_from_slice(&px);
        }
    }
    let png = cogzilla_kernel::encode_rgba_tile(&rgba, w, false, OutputFormat::Png);
    png.len()
}

fn make_scenes(n_scenes: usize, w: u32, h: u32) -> Vec<Vec<f32>> {
    let n_px = (w * h) as usize;
    (0..n_scenes)
        .map(|s| {
            let mut plane = vec![f32::NAN; n_px];
            for i in 0..n_px {
                if (i + s * 37) % 7 != 0 {
                    plane[i] = 100.0
                        + 50.0 * ((i as f32) * 0.001 + (s as f32) * 0.7).sin()
                        + (s as f32) * 8.0;
                }
            }
            plane
        })
        .collect()
}

fn main() {
    let Some(gpu) = GpuBackend::try_init(256 * 1024 * 1024) else {
        eprintln!("No GPU adapter available");
        return;
    };
    println!(
        "Mosaic Server GPU backend: {} ({:?}, floor={:.1} us)\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );
    println!("| Mosaic Pixel Selection (End-to-End) | Scenes | Size | CPU (ms) | GPU (ms) | Speedup |");
    println!("| :--- | ---: | :--- | ---: | ---: | ---: |");

    let cases: [(&str, usize, u32, GpuPixelSelection, bool); 6] = [
        ("Median f32", 8, 512, GpuPixelSelection::Median, false),
        ("Median fused PNG", 8, 512, GpuPixelSelection::Median, true),
        ("Median f32", 16, 512, GpuPixelSelection::Median, false),
        ("Median fused PNG", 16, 512, GpuPixelSelection::Median, true),
        ("Median fused PNG", 16, 1024, GpuPixelSelection::Median, true),
        ("Stdev fused PNG", 16, 1024, GpuPixelSelection::Stdev, true),
    ];

    let iters = 5usize;
    for (name, n_scenes, dim, method, styled) in cases {
        let scenes = make_scenes(n_scenes, dim, dim);
        let refs: Vec<&[f32]> = scenes.iter().map(Vec::as_slice).collect();
        let n_px = (dim * dim) as usize;

        // Warmup
        let _ = if styled {
            gpu.try_composite_mosaic_styled(
                dim, dim, &refs, method, (50.0, 250.0), ColormapName::Viridis, OutputFormat::Png, true,
            )
        } else {
            gpu.try_composite_mosaic_f32(dim, dim, &refs, method).map(|v| bytes::Bytes::from(vec![0u8; v.len()]))
        };

        let t0 = Instant::now();
        for _ in 0..iters {
            let vals = match method {
                GpuPixelSelection::Median => cpu_median_f32(&scenes, n_px),
                GpuPixelSelection::Stdev => cpu_stdev_f32(&scenes, n_px),
                _ => unreachable!(),
            };
            if styled {
                let len = cpu_style_and_png(&vals, dim, dim, 50.0, 250.0, ColormapName::Viridis);
                assert!(len > 8);
            } else {
                assert_eq!(vals.len(), n_px);
            }
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let t1 = Instant::now();
        let mut used_gpu = true;
        for _ in 0..iters {
            if styled {
                match gpu.try_composite_mosaic_styled(
                    dim, dim, &refs, method, (50.0, 250.0), ColormapName::Viridis, OutputFormat::Png, true,
                ) {
                    Some(png) => assert!(png.starts_with(b"\x89PNG")),
                    None => {
                        used_gpu = false;
                        break;
                    }
                }
            } else {
                match gpu.try_composite_mosaic_f32(dim, dim, &refs, method) {
                    Some(v) => assert_eq!(v.len(), n_px),
                    None => {
                        used_gpu = false;
                        break;
                    }
                }
            }
        }
        let gpu_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        if used_gpu {
            println!(
                "| {:<35} | {:>6} | {}x{} | {:>8.2} | {:>8.2} | {:>6.2}x |",
                name, n_scenes, dim, dim, cpu_ms, gpu_ms, cpu_ms / gpu_ms
            );
        } else {
            println!(
                "| {:<35} | {:>6} | {}x{} | {:>8.2} | CPU (pol)|    —    |",
                name, n_scenes, dim, dim, cpu_ms
            );
        }
    }
}
