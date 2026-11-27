use std::time::{Duration, Instant};

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_gpu::{GpuContext, GpuResampleMode, GpuResampler};
use cogzilla_kernel::{BandSource, ResampleMode, sample_cubic, sample_general};

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

fn bench_cpu(
    bs: &BandSource<'_>,
    u_coords: &[f32],
    v_coords: &[f32],
    mode: ResampleMode,
    iters: usize,
) -> (f64, Vec<f32>) {
    let dst_w = u_coords.len();
    let dst_h = v_coords.len();
    let mut out = vec![0.0f32; dst_w * dst_h];
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for (py, &v) in v_coords.iter().enumerate() {
            let row = &mut out[py * dst_w..(py + 1) * dst_w];
            match mode {
                ResampleMode::Cubic => {
                    for (px, &u) in u_coords.iter().enumerate() {
                        row[px] = sample_cubic(bs, u, v);
                    }
                }
                other => {
                    for (px, &u) in u_coords.iter().enumerate() {
                        row[px] = sample_general(bs, u, v, other);
                    }
                }
            }
        }
        times.push(t0.elapsed());
    }
    (median_us(times), out)
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let adapters = GpuContext::list_adapters(false);
    println!("Available hardware GPU adapters:");
    for a in &adapters {
        println!(
            "  [{}] {} ({:?}, {:?}, unified={})",
            a.index, a.name, a.backend, a.device_type, a.unified_memory
        );
    }

    let ctx = GpuContext::new_default().expect("failed to initialize GPU context");
    println!(
        "\nSelected GPU [{}]: {} ({:?}, driver: {} {})",
        ctx.info.index, ctx.info.name, ctx.info.backend, ctx.info.driver, ctx.info.driver_info
    );

    let mut resampler = GpuResampler::new(&ctx);

    // Warmup + measure noop dispatch floor
    for _ in 0..5 {
        resampler.noop_dispatch_sync();
    }
    let noop_times: Vec<Duration> = (0..25)
        .map(|_| {
            let t0 = Instant::now();
            resampler.noop_dispatch_sync();
            t0.elapsed()
        })
        .collect();
    println!("Noop dispatch + sync floor: {:.1} us (median of 25)\n", median_us(noop_times));

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

    println!(
        "| Mode | Output | CPU (us) | GPU Warm (us) | Warm Speedup | GPU Cold (us) | Cold Speedup | Max Err |"
    );
    println!("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |");

    for &(mode_name, cpu_mode, gpu_mode) in &[
        ("Cubic", ResampleMode::Cubic, GpuResampleMode::Cubic),
        ("Lanczos3", ResampleMode::Lanczos, GpuResampleMode::Lanczos3),
    ] {
        for &n in &[256usize, 512, 1024, 2048, 4096] {
            let u_coords: Vec<f32> = (0..n)
                .map(|i| 4.0 + (i as f32) * ((src_w as f32 - 8.0) / n as f32))
                .collect();
            let v_coords: Vec<f32> = (0..n)
                .map(|i| 4.0 + (i as f32) * ((src_h as f32 - 8.0) / n as f32))
                .collect();

            let iters = if n <= 512 { 15 } else if n <= 1024 { 7 } else { 3 };

            // Warmup GPU
            let _ = resampler.resample_slab(&gpu_slab, &u_coords, &v_coords, gpu_mode).unwrap();

            let (cpu_us, cpu_out) = bench_cpu(&bs, &u_coords, &v_coords, cpu_mode, iters);

            let mut warm_times = Vec::with_capacity(iters);
            let mut gpu_out = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                gpu_out = resampler
                    .resample_slab(&gpu_slab, &u_coords, &v_coords, gpu_mode)
                    .unwrap();
                warm_times.push(t0.elapsed());
            }
            let warm_us = median_us(warm_times);

            let mut cold_times = Vec::with_capacity(iters.min(5));
            for _ in 0..iters.min(5) {
                let (_, timings) = resampler
                    .resample_cold_timed(&src, src_w, src_h, src_w, &u_coords, &v_coords, gpu_mode)
                    .unwrap();
                cold_times.push(timings.total);
            }
            let cold_us = median_us(cold_times);

            let err = max_abs_err(&cpu_out, &gpu_out);
            println!(
                "| {:<8} | {:>4}x{:<4} | {:>8.1} | {:>13.1} | {:>11.2}x | {:>13.1} | {:>11.2}x | {:.2e} |",
                mode_name,
                n,
                n,
                cpu_us,
                warm_us,
                cpu_us / warm_us,
                cold_us,
                cpu_us / cold_us,
                err
            );
        }
    }
}
