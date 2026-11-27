use std::time::{Duration, Instant};

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_gpu::{
    DispatchPolicy, GpuColormapMode, GpuColormapRenderer, GpuContext, GpuCoordPlan,
    GpuPixelOutput, GpuResampler,
};
use cogzilla_kernel::{BandSource, ColormapName, ResampleMode, sample_cubic, sample_general};

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
    let mut renderer = GpuColormapRenderer::new(&ctx);

    let src_w = 2048u32;
    let src_h = 2048u32;
    let mut src = vec![0.0f32; (src_w * src_h) as usize];
    for y in 0..src_h {
        let fy = y as f32 * 0.015;
        for x in 0..src_w {
            let fx = x as f32 * 0.017;
            src[(y * src_w + x) as usize] = 1000.0 + 500.0 * fx.sin() * fy.cos();
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
            min: 400.0,
            max: 1600.0,
        },
    };
    let blocks = [Some(&band)];
    let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
    let gpu_slab = resampler.upload_slab(&src, src_w, src_h, src_w).unwrap();

    // Rotated + curved 17x17 UTM-style control grid
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
    let rmin = 400.0f32;
    let rmax = 1600.0f32;
    let inv254 = 254.0 / (rmax - rmin);

    println!("| Curved 17x17 Grid2D (UTM) | Output | CPU (us) | GPU Warm IdxU8 (us) | Speedup |");
    println!("| :--- | :--- | ---: | ---: | ---: |");

    for &(name, cpu_mode, gpu_mode) in &[
        ("Bilinear", ResampleMode::Bilinear, GpuColormapMode::Bilinear),
        ("Cubic", ResampleMode::Cubic, GpuColormapMode::Cubic),
        ("Lanczos3", ResampleMode::Lanczos, GpuColormapMode::Lanczos3),
    ] {
        for &n in &[256usize, 512, 1024, 2048] {
            let iters = if n <= 512 { 11 } else if n <= 1024 { 5 } else { 3 };
            let cell_px = n / 16;
            let inv_cell = 1.0 / cell_px as f32;
            let mut cpu_out = vec![0u8; n * n];
            let mut ct = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                for py in 0..n {
                    let gy = (py / cell_px).min(15);
                    let ty = (py - gy * cell_px) as f32 * inv_cell;
                    let r0 = gy * 17;
                    let r1 = (gy + 1) * 17;
                    let row = &mut cpu_out[py * n..(py + 1) * n];
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
                            let val = match cpu_mode {
                                ResampleMode::Cubic => sample_cubic(&bs, u_l + du * tx, v_l + dv * tx),
                                other => sample_general(&bs, u_l + du * tx, v_l + dv * tx, other),
                            };
                            row[bx + k] = if val.is_finite() {
                                1 + ((val - rmin) * inv254).clamp(0.0, 254.0) as u8
                            } else {
                                0
                            };
                        }
                    }
                }
                std::hint::black_box(cpu_out[0]);
                ct.push(t0.elapsed());
            }
            let c_us = median_us(ct);

            let _ = renderer
                .render_slab_with_plan(
                    &gpu_slab,
                    GpuCoordPlan::Grid2D { dst_w: n, dst_h: n, grid_u: &grid_u, grid_v: &grid_v },
                    gpu_mode,
                    rmin,
                    rmax,
                    &lut_u32,
                    GpuPixelOutput::IndexedU8,
                )
                .unwrap();
            let mut gt = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let out = renderer
                    .render_slab_with_plan(
                        &gpu_slab,
                        GpuCoordPlan::Grid2D { dst_w: n, dst_h: n, grid_u: &grid_u, grid_v: &grid_v },
                        gpu_mode,
                        rmin,
                        rmax,
                        &lut_u32,
                        GpuPixelOutput::IndexedU8,
                    )
                    .unwrap();
                std::hint::black_box(out[0]);
                gt.push(t0.elapsed());
            }
            let g_us = median_us(gt);
            println!(
                "| {:<25} | {:>4}x{:<4} | {:>8.1} | {:>19.1} | {:>6.2}x |",
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
