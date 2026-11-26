use std::time::{Duration, Instant};

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_gpu::{
    DispatchPolicy, GpuColormapMode, GpuColormapRenderer, GpuContext, GpuCoordPlan,
    GpuDemAlgorithm, GpuPixelOutput, GpuResampler, GpuTerrainRenderer,
};
use cogzilla_kernel::{
    BandSource, ColormapName, DemAlgorithm, ResampleMode, eval_horn_3x3_aniso, sample_cubic,
    sample_general,
};

fn median_us(mut v: Vec<Duration>) -> f64 {
    v.sort();
    v[v.len() / 2].as_secs_f64() * 1e6
}

fn main() {
    let ctx = GpuContext::new_default().expect("GPU adapter required");
    let policy = DispatchPolicy::calibrate(&ctx);
    println!(
        "Selected GPU [{}]: {} ({:?}, floor={:.1} us)\n",
        ctx.info.index, ctx.info.name, ctx.info.backend, policy.profile.dispatch_floor_us
    );

    let resampler = GpuResampler::new(&ctx);
    let mut terrain = GpuTerrainRenderer::new(&ctx);

    let src_w = 2048u32;
    let src_h = 2048u32;
    let mut src = vec![0.0f32; (src_w * src_h) as usize];
    for y in 0..src_h {
        let fy = y as f32 * 0.015;
        for x in 0..src_w {
            let fx = x as f32 * 0.017;
            src[(y * src_w + x) as usize] = 1500.0 + 800.0 * fx.sin() * fy.cos();
        }
    }
    let mut slab_cpu = AlignedF32Slab::new_uninit(src.len());
    slab_cpu.as_mut_slice().copy_from_slice(&src);
    let band = DecodedBand {
        data: BandData::F32(slab_cpu),
        width: src_w,
        height: src_h,
        stats: BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 600.0,
            max: 2400.0,
        },
    };
    let blocks = [Some(&band)];
    let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
    let gpu_slab = resampler.upload_slab(&src, src_w, src_h, src_w).unwrap();

    let mut grid_u = [0.0f32; 289];
    let mut grid_v = [0.0f32; 289];
    let cos_a = 0.9659f32;
    let sin_a = 0.2588f32;
    for gy in 0..17 {
        for gx in 0..17 {
            let nx = (gx as f32 / 16.0) - 0.5;
            let ny = (gy as f32 / 16.0) - 0.5;
            grid_u[gy * 17 + gx] = 1024.0 + 1300.0 * (nx * cos_a - ny * sin_a);
            grid_v[gy * 17 + gx] = 1024.0 + 1300.0 * (nx * sin_a + ny * cos_a) + 25.0 * nx * nx;
        }
    }

    let lut = ColormapName::Viridis.lut();
    let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
    let hs_gpu = GpuDemAlgorithm::Hillshade {
        azimuth_deg: 315.0,
        altitude_deg: 45.0,
        z_factor: 1.5,
    };
    let hs_cpu = DemAlgorithm::Hillshade {
        azimuth_deg: 315.0,
        altitude_deg: 45.0,
        z_factor: 1.5,
    };
    let inv_8d = 1.0 / (8.0 * 30.0);
    let inv254 = 254.0 / 255.0;

    println!("| Curved 17x17 DEM Hillshade | Output | CPU (us) | GPU Warm IdxU8 (us) | Speedup |");
    println!("| :--- | :--- | ---: | ---: | ---: |");

    for &(name, cpu_mode, gpu_mode) in &[
        ("Hillshade Cubic U8", ResampleMode::Cubic, GpuColormapMode::Cubic),
        ("Hillshade Lanczos3 U8", ResampleMode::Lanczos, GpuColormapMode::Lanczos3),
    ] {
        for &n in &[256usize, 512, 1024, 2048] {
            let iters = if n <= 512 { 7 } else if n <= 1024 { 5 } else { 3 };
            let cell_px = n / 16;
            let inv_cell = 1.0 / cell_px as f32;
            let mut elev = vec![0.0f32; n * n];
            let mut out = vec![0u8; n * n];

            let mut ct = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                for py in 0..n {
                    let gy = (py / cell_px).min(15);
                    let ty = (py - gy * cell_px) as f32 * inv_cell;
                    let r0 = gy * 17;
                    let r1 = (gy + 1) * 17;
                    for gx in 0..16 {
                        let u_l = grid_u[r0 + gx] + (grid_u[r1 + gx] - grid_u[r0 + gx]) * ty;
                        let u_r = grid_u[r0 + gx + 1] + (grid_u[r1 + gx + 1] - grid_u[r0 + gx + 1]) * ty;
                        let v_l = grid_v[r0 + gx] + (grid_v[r1 + gx] - grid_v[r0 + gx]) * ty;
                        let v_r = grid_v[r0 + gx + 1] + (grid_v[r1 + gx + 1] - grid_v[r0 + gx + 1]) * ty;
                        let du = u_r - u_l;
                        let dv = v_r - v_l;
                        let bx = gx * cell_px;
                        for k in 0..cell_px {
                            let tx = k as f32 * inv_cell;
                            elev[py * n + bx + k] = match cpu_mode {
                                ResampleMode::Cubic => sample_cubic(&bs, u_l + du * tx, v_l + dv * tx),
                                other => sample_general(&bs, u_l + du * tx, v_l + dv * tx, other),
                            };
                        }
                    }
                }
                for py in 1..(n - 1) {
                    for px in 1..(n - 1) {
                        let r0 = (py - 1) * n + (px - 1);
                        let r1 = py * n + (px - 1);
                        let r2 = (py + 1) * n + (px - 1);
                        let v = eval_horn_3x3_aniso(
                            elev[r0], elev[r0 + 1], elev[r0 + 2],
                            elev[r1], elev[r1 + 1], elev[r1 + 2],
                            elev[r2], elev[r2 + 1], elev[r2 + 2],
                            inv_8d, inv_8d, hs_cpu,
                        );
                        out[py * n + px] = 1 + (v * inv254).clamp(0.0, 254.0) as u8;
                    }
                }
                std::hint::black_box(out[n + 1]);
                ct.push(t0.elapsed());
            }
            let c_us = median_us(ct);

            let _ = terrain
                .render_slab_with_plan(
                    &gpu_slab,
                    GpuCoordPlan::Grid2D { dst_w: n, dst_h: n, grid_u: &grid_u, grid_v: &grid_v },
                    gpu_mode,
                    hs_gpu,
                    30.0,
                    30.0,
                    0.0,
                    255.0,
                    &lut_u32,
                    GpuPixelOutput::IndexedU8,
                )
                .unwrap();
            let mut gt = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let res = terrain
                    .render_slab_with_plan(
                        &gpu_slab,
                        GpuCoordPlan::Grid2D { dst_w: n, dst_h: n, grid_u: &grid_u, grid_v: &grid_v },
                        gpu_mode,
                        hs_gpu,
                        30.0,
                        30.0,
                        0.0,
                        255.0,
                        &lut_u32,
                        GpuPixelOutput::IndexedU8,
                    )
                    .unwrap();
                std::hint::black_box(res[0]);
                gt.push(t0.elapsed());
            }
            let g_us = median_us(gt);
            println!(
                "| {:<26} | {:>4}x{:<4} | {:>8.1} | {:>19.1} | {:>6.2}x |",
                name,
                n,
                n,
                c_us,
                g_us,
                c_us / g_us
            );
        }
    }
}
