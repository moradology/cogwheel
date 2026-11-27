use std::time::{Duration, Instant};

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_gpu::{
    DispatchPolicy, DispatchRequest, GpuColormapMode, GpuColormapRenderer, GpuContext,
    GpuPixelOutput, GpuResampler,
};
use cogzilla_kernel::{BandSource, ColormapName, ResampleMode, sample_cubic, sample_general};

fn make_dem(w: u32, h: u32) -> Vec<f32> {
    let mut data = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        let fy = y as f32 * 0.013;
        for x in 0..w {
            let fx = x as f32 * 0.017;
            data[(y * w + x) as usize] = 1200.0 + 350.0 * fx.sin() * fy.cos() + 40.0 * (fx + fy).sin();
        }
    }
    data
}

fn median_us(mut samples: Vec<Duration>) -> f64 {
    samples.sort();
    samples[samples.len() / 2].as_secs_f64() * 1e6
}

fn bench_cpu_indexed(
    bs: &BandSource<'_>,
    u_coords: &[f32],
    v_coords: &[f32],
    mode: ResampleMode,
    rmin: f32,
    rmax: f32,
    iters: usize,
) -> f64 {
    let dst_w = u_coords.len();
    let dst_h = v_coords.len();
    let inv254 = 254.0 / (rmax - rmin);
    let mut out = vec![0u8; dst_w * dst_h];
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for (py, &v) in v_coords.iter().enumerate() {
            let row = &mut out[py * dst_w..(py + 1) * dst_w];
            for (px, &u) in u_coords.iter().enumerate() {
                let val = match mode {
                    ResampleMode::Cubic => sample_cubic(bs, u, v),
                    other => sample_general(bs, u, v, other),
                };
                row[px] = if val.is_finite() {
                    1 + ((val - rmin) * inv254).clamp(0.0, 254.0) as u8
                } else {
                    0
                };
            }
        }
        std::hint::black_box(out[0]);
        times.push(t0.elapsed());
    }
    median_us(times)
}

fn main() {
    let ctx = GpuContext::new_default().expect("failed to initialize GPU context");
    println!(
        "Selected GPU [{}]: {} ({:?}, unified={})",
        ctx.info.index, ctx.info.name, ctx.info.backend, ctx.info.unified_memory
    );

    let t_cal = Instant::now();
    let policy = DispatchPolicy::calibrate(&ctx);
    println!(
        "Calibrated GpuProfile in {:.1} ms: dispatch_floor={:.1} us, upload={:.2} GB/s, readback={:.2} GB/s\n",
        t_cal.elapsed().as_secs_f64() * 1e3,
        policy.profile.dispatch_floor_us,
        policy.profile.upload_gb_s,
        policy.profile.readback_gb_s
    );

    let resampler = GpuResampler::new(&ctx);
    let mut renderer = GpuColormapRenderer::new(&ctx);

    let src_w = 2048u32;
    let src_h = 2048u32;
    let src = make_dem(src_w, src_h);

    let mut slab_cpu = AlignedF32Slab::new_uninit((src_w * src_h) as usize);
    slab_cpu.as_mut_slice().copy_from_slice(&src);
    let band = DecodedBand {
        data: BandData::F32(slab_cpu),
        width: src_w,
        height: src_h,
        stats: BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 800.0,
            max: 1600.0,
        },
    };
    let blocks = [Some(&band)];
    let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
    let gpu_slab = resampler.upload_slab(&src, src_w, src_h, src_w).unwrap();

    let lut = ColormapName::Viridis.lut();
    let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
    let rmin = 800.0f32;
    let rmax = 1600.0f32;

    println!(
        "| Mode | Output | CPU (us) | GPU RGBA8 Warm (us) | GPU IdxU8 Warm (us) | IdxU8 Speedup | Policy (Warm IdxU8) |"
    );
    println!("| :--- | :--- | ---: | ---: | ---: | ---: | :--- |");

    for &(mode_name, cpu_mode, gpu_mode) in &[
        ("Bilinear", ResampleMode::Bilinear, GpuColormapMode::Bilinear),
        ("Cubic", ResampleMode::Cubic, GpuColormapMode::Cubic),
        ("Lanczos3", ResampleMode::Lanczos, GpuColormapMode::Lanczos3),
    ] {
        for &n in &[256usize, 512, 1024, 2048, 4096] {
            let u_coords: Vec<f32> = (0..n)
                .map(|i| 4.0 + (i as f32) * ((src_w as f32 - 8.0) / n as f32))
                .collect();
            let v_coords: Vec<f32> = (0..n)
                .map(|i| 4.0 + (i as f32) * ((src_h as f32 - 8.0) / n as f32))
                .collect();

            let iters = if n <= 512 { 15 } else if n <= 1024 { 7 } else { 3 };

            // Warmup
            let _ = renderer
                .render_slab(
                    &gpu_slab,
                    &u_coords,
                    &v_coords,
                    gpu_mode,
                    rmin,
                    rmax,
                    &lut_u32,
                    GpuPixelOutput::Rgba8,
                )
                .unwrap();
            let _ = renderer
                .render_slab(
                    &gpu_slab,
                    &u_coords,
                    &v_coords,
                    gpu_mode,
                    rmin,
                    rmax,
                    &lut_u32,
                    GpuPixelOutput::IndexedU8,
                )
                .unwrap();

            let cpu_us = bench_cpu_indexed(&bs, &u_coords, &v_coords, cpu_mode, rmin, rmax, iters);

            let mut rgba_times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let out = renderer
                    .render_slab(
                        &gpu_slab,
                        &u_coords,
                        &v_coords,
                        gpu_mode,
                        rmin,
                        rmax,
                        &lut_u32,
                        GpuPixelOutput::Rgba8,
                    )
                    .unwrap();
                std::hint::black_box(out[0]);
                rgba_times.push(t0.elapsed());
            }
            let rgba_us = median_us(rgba_times);

            let mut idx_times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let out = renderer
                    .render_slab(
                        &gpu_slab,
                        &u_coords,
                        &v_coords,
                        gpu_mode,
                        rmin,
                        rmax,
                        &lut_u32,
                        GpuPixelOutput::IndexedU8,
                    )
                    .unwrap();
                std::hint::black_box(out[0]);
                idx_times.push(t0.elapsed());
            }
            let idx_us = median_us(idx_times);

            let decision = policy.decide(&DispatchRequest {
                op: gpu_mode.to_policy_op(),
                dst_pixels: n * n,
                upload_bytes: 0,
                readback_bytes: n * n,
                batch_size: 1,
                in_flight: 0,
            });
            let pred = if decision.use_gpu { "GPU" } else { "CPU" };

            println!(
                "| {:<8} | {:>4}x{:<4} | {:>8.1} | {:>19.1} | {:>19.1} | {:>12.2}x | {} ({:.0} vs {:.0} us) |",
                mode_name,
                n,
                n,
                cpu_us,
                rgba_us,
                idx_us,
                cpu_us / idx_us,
                pred,
                decision.est_gpu_us,
                decision.est_cpu_us
            );
        }
    }
}
