use std::time::{Duration, Instant};

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_gpu::{
    DispatchPolicy, DispatchRequest, GpuColormapMode, GpuColormapRenderer, GpuContext,
    GpuDemAlgorithm, GpuOp, GpuPixelOutput, GpuResampler, GpuTerrainRenderer,
};
use cogzilla_kernel::{
    BandSource, ColormapName, DemAlgorithm, ResampleMode, encode_terrarium_rgb,
    eval_horn_3x3_aniso, sample_cubic, sample_general,
};

fn make_dem(w: u32, h: u32) -> Vec<f32> {
    let mut data = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        let fy = y as f32 * 0.015;
        for x in 0..w {
            let fx = x as f32 * 0.018;
            data[(y * w + x) as usize] = 1500.0 + 900.0 * fx.sin() * fy.cos() + 80.0 * (fx + fy).sin();
        }
    }
    data
}

fn median_us(mut samples: Vec<Duration>) -> f64 {
    samples.sort();
    samples[samples.len() / 2].as_secs_f64() * 1e6
}

fn bench_cpu_hillshade(
    bs: &BandSource<'_>,
    u_coords: &[f32],
    v_coords: &[f32],
    mode: ResampleMode,
    iters: usize,
) -> f64 {
    let n = u_coords.len();
    let pad = n + 2;
    let mut elev = vec![0.0f32; pad * pad];
    let mut out = vec![0u8; n * n];
    let alg = DemAlgorithm::Hillshade {
        azimuth_deg: 315.0,
        altitude_deg: 45.0,
        z_factor: 1.5,
    };
    let inv_8d = 1.0 / (8.0 * 30.0);
    let inv254 = 254.0 / 255.0;

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for py in 0..pad {
            let iy = py as i32 - 1;
            let v = if iy < 0 {
                2.0 * v_coords[0] - v_coords[1]
            } else if iy >= n as i32 {
                2.0 * v_coords[n - 1] - v_coords[n - 2]
            } else {
                v_coords[iy as usize]
            };
            let row = &mut elev[py * pad..(py + 1) * pad];
            for px in 0..pad {
                let ix = px as i32 - 1;
                let u = if ix < 0 {
                    2.0 * u_coords[0] - u_coords[1]
                } else if ix >= n as i32 {
                    2.0 * u_coords[n - 1] - u_coords[n - 2]
                } else {
                    u_coords[ix as usize]
                };
                row[px] = match mode {
                    ResampleMode::Cubic => sample_cubic(bs, u, v),
                    other => sample_general(bs, u, v, other),
                };
            }
        }
        for py in 0..n {
            for px in 0..n {
                let r0 = py * pad + px;
                let r1 = (py + 1) * pad + px;
                let r2 = (py + 2) * pad + px;
                let val = eval_horn_3x3_aniso(
                    elev[r0],
                    elev[r0 + 1],
                    elev[r0 + 2],
                    elev[r1],
                    elev[r1 + 1],
                    elev[r1 + 2],
                    elev[r2],
                    elev[r2 + 1],
                    elev[r2 + 2],
                    inv_8d,
                    inv_8d,
                    alg,
                );
                out[py * n + px] = if val.is_finite() {
                    1 + (val * inv254).clamp(0.0, 254.0) as u8
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

fn bench_cpu_terrarium(
    bs: &BandSource<'_>,
    u_coords: &[f32],
    v_coords: &[f32],
    iters: usize,
) -> f64 {
    let n = u_coords.len();
    let mut out = vec![0u8; n * n * 4];
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for (py, &v) in v_coords.iter().enumerate() {
            for (px, &u) in u_coords.iter().enumerate() {
                let z = sample_cubic(bs, u, v);
                let off = (py * n + px) * 4;
                if z.is_finite() {
                    let rgb = encode_terrarium_rgb(z);
                    out[off] = rgb[0];
                    out[off + 1] = rgb[1];
                    out[off + 2] = rgb[2];
                    out[off + 3] = 255;
                } else {
                    out[off..off + 4].fill(0);
                }
            }
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

    let resampler = GpuResampler::new(&ctx);
    let mut terrain = GpuTerrainRenderer::new(&ctx);

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
            min: 500.0,
            max: 2500.0,
        },
    };
    let blocks = [Some(&band)];
    let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
    let gpu_slab = resampler.upload_slab(&src, src_w, src_h, src_w).unwrap();

    let lut = ColormapName::Viridis.lut();
    let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);

    println!("| Workload | Output | CPU (us) | GPU Warm (us) | Speedup | Policy |");
    println!("| :--- | :--- | ---: | ---: | ---: | :--- |");

    let hs_alg = GpuDemAlgorithm::Hillshade {
        azimuth_deg: 315.0,
        altitude_deg: 45.0,
        z_factor: 1.5,
    };

    for &n in &[256usize, 512, 1024, 2048, 4096] {
        let u_coords: Vec<f32> = (0..n)
            .map(|i| 4.0 + (i as f32) * ((src_w as f32 - 8.0) / n as f32))
            .collect();
        let v_coords: Vec<f32> = (0..n)
            .map(|i| 4.0 + (i as f32) * ((src_h as f32 - 8.0) / n as f32))
            .collect();
        let iters = if n <= 512 { 11 } else if n <= 1024 { 5 } else { 3 };

        // 1. Cubic Hillshade (IndexedU8)
        let _ = terrain
            .render_slab(
                &gpu_slab,
                &u_coords,
                &v_coords,
                GpuColormapMode::Cubic,
                hs_alg,
                30.0,
                30.0,
                0.0,
                255.0,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();
        let cpu_us = bench_cpu_hillshade(&bs, &u_coords, &v_coords, ResampleMode::Cubic, iters);
        let mut g_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let out = terrain
                .render_slab(
                    &gpu_slab,
                    &u_coords,
                    &v_coords,
                    GpuColormapMode::Cubic,
                    hs_alg,
                    30.0,
                    30.0,
                    0.0,
                    255.0,
                    &lut_u32,
                    GpuPixelOutput::IndexedU8,
                )
                .unwrap();
            std::hint::black_box(out[0]);
            g_times.push(t0.elapsed());
        }
        let gpu_us = median_us(g_times);
        let dec = policy.decide(&DispatchRequest {
            op: GpuOp::DemHornCubic,
            dst_pixels: n * n,
            upload_bytes: 0,
            readback_bytes: n * n,
            batch_size: 1,
            in_flight: 0,
        });
        println!(
            "| {:<20} | {:>4}x{:<4} | {:>8.1} | {:>13.1} | {:>6.2}x | {} |",
            "Hillshade Cubic U8",
            n,
            n,
            cpu_us,
            gpu_us,
            cpu_us / gpu_us,
            if dec.use_gpu { "GPU" } else { "CPU" }
        );

        // 2. Lanczos3 Hillshade (IndexedU8)
        let _ = terrain
            .render_slab(
                &gpu_slab,
                &u_coords,
                &v_coords,
                GpuColormapMode::Lanczos3,
                hs_alg,
                30.0,
                30.0,
                0.0,
                255.0,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();
        let cpu_l_us =
            bench_cpu_hillshade(&bs, &u_coords, &v_coords, ResampleMode::Lanczos, iters.min(3));
        let mut gl_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let out = terrain
                .render_slab(
                    &gpu_slab,
                    &u_coords,
                    &v_coords,
                    GpuColormapMode::Lanczos3,
                    hs_alg,
                    30.0,
                    30.0,
                    0.0,
                    255.0,
                    &lut_u32,
                    GpuPixelOutput::IndexedU8,
                )
                .unwrap();
            std::hint::black_box(out[0]);
            gl_times.push(t0.elapsed());
        }
        let gpu_l_us = median_us(gl_times);
        let dec_l = policy.decide(&DispatchRequest {
            op: GpuOp::DemHornLanczos3,
            dst_pixels: n * n,
            upload_bytes: 0,
            readback_bytes: n * n,
            batch_size: 1,
            in_flight: 0,
        });
        println!(
            "| {:<20} | {:>4}x{:<4} | {:>8.1} | {:>13.1} | {:>6.2}x | {} |",
            "Hillshade Lanczos U8",
            n,
            n,
            cpu_l_us,
            gpu_l_us,
            cpu_l_us / gpu_l_us,
            if dec_l.use_gpu { "GPU" } else { "CPU" }
        );

        // 3. Cubic Terrarium RGB (RGBA8)
        let _ = terrain
            .render_slab(
                &gpu_slab,
                &u_coords,
                &v_coords,
                GpuColormapMode::Cubic,
                GpuDemAlgorithm::Terrarium,
                30.0,
                30.0,
                0.0,
                255.0,
                &lut_u32,
                GpuPixelOutput::Rgba8,
            )
            .unwrap();
        let cpu_t_us = bench_cpu_terrarium(&bs, &u_coords, &v_coords, iters);
        let mut gt_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let out = terrain
                .render_slab(
                    &gpu_slab,
                    &u_coords,
                    &v_coords,
                    GpuColormapMode::Cubic,
                    GpuDemAlgorithm::Terrarium,
                    30.0,
                    30.0,
                    0.0,
                    255.0,
                    &lut_u32,
                    GpuPixelOutput::Rgba8,
                )
                .unwrap();
            std::hint::black_box(out[0]);
            gt_times.push(t0.elapsed());
        }
        let gpu_t_us = median_us(gt_times);
        let dec_t = policy.decide(&DispatchRequest {
            op: GpuOp::DemTerrainRgbCubic,
            dst_pixels: n * n,
            upload_bytes: 0,
            readback_bytes: n * n * 4,
            batch_size: 1,
            in_flight: 0,
        });
        println!(
            "| {:<20} | {:>4}x{:<4} | {:>8.1} | {:>13.1} | {:>6.2}x | {} |",
            "Terrarium Cubic RGBA",
            n,
            n,
            cpu_t_us,
            gpu_t_us,
            cpu_t_us / gpu_t_us,
            if dec_t.use_gpu { "GPU" } else { "CPU" }
        );
    }
}
