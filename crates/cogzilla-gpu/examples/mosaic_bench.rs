use std::time::{Duration, Instant};

use cogzilla_gpu::{
    DispatchPolicy, GpuColormapRenderer, GpuContext, GpuMosaicCompositor, GpuPixelOutput,
    GpuPixelSelection,
};
use cogzilla_kernel::ColormapName;

fn median_us(mut samples: Vec<Duration>) -> f64 {
    samples.sort();
    samples[samples.len() / 2].as_secs_f64() * 1e6
}

fn cpu_composite_indexed(
    planes: &[Vec<f32>],
    method: GpuPixelSelection,
    rmin: f32,
    rmax: f32,
    iters: usize,
) -> f64 {
    let n = planes[0].len();
    let inv254 = 254.0 / (rmax - rmin);
    let mut out = vec![0u8; n];
    let mut scratch = [0.0f32; 32];
    let mut times = Vec::with_capacity(iters);

    for _ in 0..iters {
        let t0 = Instant::now();
        for i in 0..n {
            let val = match method {
                GpuPixelSelection::Mean => {
                    let mut sum = 0.0f32;
                    let mut c = 0u32;
                    for p in planes {
                        let v = p[i];
                        if v.is_finite() {
                            sum += v;
                            c += 1;
                        }
                    }
                    if c > 0 { sum / c as f32 } else { f32::NAN }
                }
                GpuPixelSelection::Median => {
                    let mut c = 0usize;
                    for p in planes {
                        let v = p[i];
                        if v.is_finite() {
                            let mut j = c;
                            while j > 0 && scratch[j - 1] > v {
                                scratch[j] = scratch[j - 1];
                                j -= 1;
                            }
                            scratch[j] = v;
                            c += 1;
                        }
                    }
                    if c == 0 {
                        f32::NAN
                    } else if c % 2 == 1 {
                        scratch[c / 2]
                    } else {
                        0.5 * (scratch[c / 2 - 1] + scratch[c / 2])
                    }
                }
                GpuPixelSelection::Stdev => {
                    let mut mean = 0.0f32;
                    let mut m2 = 0.0f32;
                    let mut c = 0u32;
                    for p in planes {
                        let v = p[i];
                        if v.is_finite() {
                            c += 1;
                            let d = v - mean;
                            mean += d / c as f32;
                            m2 += d * (v - mean);
                        }
                    }
                    if c > 0 { (m2 / c as f32).max(0.0).sqrt() } else { f32::NAN }
                }
                _ => f32::NAN,
            };
            out[i] = if val.is_finite() {
                1 + ((val - rmin) * inv254).clamp(0.0, 254.0) as u8
            } else {
                0
            };
        }
        std::hint::black_box(out[0]);
        times.push(t0.elapsed());
    }
    median_us(times)
}

fn main() {
    let ctx = GpuContext::new_default().expect("failed to initialize GPU context");
    let policy = DispatchPolicy::calibrate(&ctx);
    println!(
        "Selected GPU [{}]: {} ({:?}, floor={:.1} us, up={:.2} GB/s, down={:.2} GB/s)\n",
        ctx.info.index,
        ctx.info.name,
        ctx.info.backend,
        policy.profile.dispatch_floor_us,
        policy.profile.upload_gb_s,
        policy.profile.readback_gb_s
    );

    let mut comp = GpuMosaicCompositor::new(&ctx);
    let lut = ColormapName::Viridis.lut();
    let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);

    println!("| Method | Scenes | Output | CPU (us) | GPU Warm IdxU8 (us) | Speedup |");
    println!("| :--- | ---: | :--- | ---: | ---: | ---: |");

    for &n in &[512u32, 1024, 2048] {
        let pixels = (n * n) as usize;
        for &num_scenes in &[8usize, 16, 32] {
            let mut planes = Vec::with_capacity(num_scenes);
            for s in 0..num_scenes {
                let mut p = vec![0.0f32; pixels];
                for i in 0..pixels {
                    if (i + s * 31) % 7 == 0 {
                        p[i] = f32::NAN;
                    } else {
                        p[i] = 100.0 + ((i.wrapping_mul(17) ^ s.wrapping_mul(131)) & 1023) as f32;
                    }
                }
                planes.push(p);
            }
            let refs: Vec<&[f32]> = planes.iter().map(Vec::as_slice).collect();
            let stack = comp.upload_scenes(&refs, n, n).unwrap();

            for &(name, method) in &[
                ("Mean", GpuPixelSelection::Mean),
                ("Median", GpuPixelSelection::Median),
                ("Stdev", GpuPixelSelection::Stdev),
            ] {
                let iters = if n <= 512 { 7 } else if n <= 1024 { 5 } else { 3 };
                let cpu_us = cpu_composite_indexed(&planes, method, 100.0, 1200.0, iters);

                // Warmup
                let _ = comp
                    .composite_styled(&stack, method, 100.0, 1200.0, &lut_u32, GpuPixelOutput::IndexedU8)
                    .unwrap();
                let mut g_times = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let out = comp
                        .composite_styled(&stack, method, 100.0, 1200.0, &lut_u32, GpuPixelOutput::IndexedU8)
                        .unwrap();
                    std::hint::black_box(out[0]);
                    g_times.push(t0.elapsed());
                }
                let gpu_us = median_us(g_times);
                println!(
                    "| {:<6} | {:>6} | {:>4}x{:<4} | {:>8.1} | {:>19.1} | {:>6.2}x |",
                    name,
                    num_scenes,
                    n,
                    n,
                    cpu_us,
                    gpu_us,
                    cpu_us / gpu_us
                );
            }
        }
    }
}
