use std::time::{Duration, Instant};

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_kernel::png_fast::{encode_indexed_png, filter_sub_u8_row};
use cogzilla_kernel::{
    BandSource, ColormapName, DemAlgorithm, OutputFormat, ResampleMode, SingleBandStyle,
    eval_horn_3x3_aniso, sample_cubic, sample_general,
};
use cogzilla_server::gpu_backend::GpuBackend;

fn median_us(mut v: Vec<Duration>) -> f64 {
    v.sort();
    v[v.len() / 2].as_secs_f64() * 1e6
}

fn cpu_crop_png(
    src: &BandSource<'_>,
    u_coords: &[f32],
    v_coords: &[f32],
    mode: ResampleMode,
    rmin: f32,
    rmax: f32,
) -> usize {
    let w = u_coords.len();
    let h = v_coords.len();
    let inv254 = 254.0 / (rmax - rmin);
    let mut row_idx = vec![0u8; w];
    let mut filtered = vec![0u8; h * (1 + w)];
    for (py, &v) in v_coords.iter().enumerate() {
        for (px, &u) in u_coords.iter().enumerate() {
            let val = match mode {
                ResampleMode::Cubic => sample_cubic(src, u, v),
                other => sample_general(src, u, v, other),
            };
            row_idx[px] = if val.is_finite() {
                1 + ((val - rmin) * inv254).clamp(0.0, 254.0) as u8
            } else {
                0
            };
        }
        filter_sub_u8_row(&row_idx, &mut filtered[py * (1 + w)..(py + 1) * (1 + w)]);
    }
    let lut = ColormapName::Viridis.lut();
    let mut plte = [0u8; 768];
    for i in 1..256 {
        let src_i = ((i - 1) * 255 + 127) / 254;
        plte[i * 3] = lut.rgb[src_i][0];
        plte[i * 3 + 1] = lut.rgb[src_i][1];
        plte[i * 3 + 2] = lut.rgb[src_i][2];
    }
    encode_indexed_png(w as u32, h as u32, &plte, Some(&[0]), &filtered).len()
}

fn cpu_dem_crop_png(
    src: &BandSource<'_>,
    u_coords: &[f32],
    v_coords: &[f32],
    mode: ResampleMode,
) -> usize {
    let w = u_coords.len();
    let h = v_coords.len();
    let pad_w = w + 2;
    let pad_h = h + 2;
    let mut elev = vec![0.0f32; pad_w * pad_h];
    for py in 0..pad_h {
        let iy = py as i32 - 1;
        let v = if iy < 0 {
            2.0 * v_coords[0] - v_coords[1]
        } else if iy >= h as i32 {
            2.0 * v_coords[h - 1] - v_coords[h - 2]
        } else {
            v_coords[iy as usize]
        };
        for px in 0..pad_w {
            let ix = px as i32 - 1;
            let u = if ix < 0 {
                2.0 * u_coords[0] - u_coords[1]
            } else if ix >= w as i32 {
                2.0 * u_coords[w - 1] - u_coords[w - 2]
            } else {
                u_coords[ix as usize]
            };
            elev[py * pad_w + px] = match mode {
                ResampleMode::Cubic => sample_cubic(src, u, v),
                other => sample_general(src, u, v, other),
            };
        }
    }
    let alg = DemAlgorithm::Hillshade {
        azimuth_deg: 315.0,
        altitude_deg: 45.0,
        z_factor: 1.5,
    };
    let inv_8d = 1.0 / (8.0 * 30.0);
    let inv254 = 254.0 / 255.0;
    let mut row_idx = vec![0u8; w];
    let mut filtered = vec![0u8; h * (1 + w)];
    for py in 0..h {
        for px in 0..w {
            let r0 = py * pad_w + px;
            let r1 = (py + 1) * pad_w + px;
            let r2 = (py + 2) * pad_w + px;
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
            row_idx[px] = if val.is_finite() {
                1 + (val * inv254).clamp(0.0, 254.0) as u8
            } else {
                0
            };
        }
        filter_sub_u8_row(&row_idx, &mut filtered[py * (1 + w)..(py + 1) * (1 + w)]);
    }
    let lut = ColormapName::Viridis.lut();
    let mut plte = [0u8; 768];
    for i in 1..256 {
        let src_i = ((i - 1) * 255 + 127) / 254;
        plte[i * 3] = lut.rgb[src_i][0];
        plte[i * 3 + 1] = lut.rgb[src_i][1];
        plte[i * 3 + 2] = lut.rgb[src_i][2];
    }
    encode_indexed_png(w as u32, h as u32, &plte, Some(&[0]), &filtered).len()
}

fn main() {
    let gpu = GpuBackend::try_init(256 * 1024 * 1024).expect("GPU backend required");
    println!(
        "GPU Backend: {} ({:?}, floor={:.1} us)\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );

    // 4x4 grid of 512x512 f32 blocks = 2048x2048 multi-block COG neighborhood
    let tw = 512u32;
    let th = 512u32;
    let mut bands = Vec::with_capacity(16);
    for ty in 0..4u32 {
        for tx in 0..4u32 {
            let mut slab = AlignedF32Slab::new_uninit((tw * th) as usize);
            let dst = slab.as_mut_slice();
            for y in 0..th {
                let fy = (ty * th + y) as f32 * 0.015;
                for x in 0..tw {
                    let fx = (tx * tw + x) as f32 * 0.017;
                    dst[(y * tw + x) as usize] = 1200.0 + 500.0 * fx.sin() * fy.cos();
                }
            }
            bands.push(DecodedBand {
                data: BandData::F32(slab),
                width: tw,
                height: th,
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 600.0,
                    max: 1800.0,
                },
            });
        }
    }
    let refs: Vec<Option<&DecodedBand>> = bands.iter().map(Some).collect();
    let src = BandSource::new(2048, 2048, tw, th, 0, 0, 4, 4, &refs);

    println!("| Multi-Block Crop PNG (4x4 512² COG) | Output | CPU+PNG (ms) | GPU+PNG Warm (ms) | Speedup |");
    println!("| :--- | :--- | ---: | ---: | ---: |");

    let hs = DemAlgorithm::Hillshade {
        azimuth_deg: 315.0,
        altitude_deg: 45.0,
        z_factor: 1.5,
    };

    for &n in &[1024usize, 2048, 4096] {
        let u_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (2040.0 / n as f32)).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (2040.0 / n as f32)).collect();
        let iters = if n <= 1024 { 5 } else { 3 };

        for &(label, resample) in &[
            ("Single Cubic PNG", ResampleMode::Cubic),
            ("Single Lanczos3 PNG", ResampleMode::Lanczos),
        ] {
            let style = SingleBandStyle {
                rescale_min: 600.0,
                rescale_max: 1800.0,
                colormap: ColormapName::Viridis,
                resample,
                format: OutputFormat::Png,
            };
            // Warmup GPU (may return None if cold upload exceeds CPU cost on 1024^2 Cubic)
            let warm_ok = gpu
                .try_render_crop_single_band(7, 0, 0, &u_coords, &v_coords, &src, &style)
                .is_some();

            let mut ct = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                std::hint::black_box(cpu_crop_png(&src, &u_coords, &v_coords, resample, 600.0, 1800.0));
                ct.push(t0.elapsed());
            }
            let c_ms = median_us(ct) / 1000.0;
            if !warm_ok {
                println!(
                    "| {:<35} | {:>4}x{:<4} | {:>12.2} | {:>17} | {:>7} |",
                    label, n, n, c_ms, "CPU (policy)", "1.00x"
                );
                continue;
            }
            let mut gt = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let b = gpu
                    .try_render_crop_single_band(7, 0, 0, &u_coords, &v_coords, &src, &style)
                    .expect("expected GPU dispatch");
                std::hint::black_box(b.len());
                gt.push(t0.elapsed());
            }
            let g_ms = median_us(gt) / 1000.0;
            println!(
                "| {:<35} | {:>4}x{:<4} | {:>12.2} | {:>17.2} | {:>6.2}x |",
                label,
                n,
                n,
                c_ms,
                g_ms,
                c_ms / g_ms
            );
        }

        let hs_style = SingleBandStyle {
            rescale_min: 0.0,
            rescale_max: 255.0,
            colormap: ColormapName::Viridis,
            resample: ResampleMode::Lanczos,
            format: OutputFormat::Png,
        };
        let _ = gpu.try_render_crop_dem(7, 0, 0, &u_coords, &v_coords, &src, hs, 30.0, 30.0, &hs_style);
        let mut ct = Vec::with_capacity(iters.min(3));
        for _ in 0..iters.min(3) {
            let t0 = Instant::now();
            std::hint::black_box(cpu_dem_crop_png(&src, &u_coords, &v_coords, ResampleMode::Lanczos));
            ct.push(t0.elapsed());
        }
        let mut gt = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let b = gpu
                .try_render_crop_dem(7, 0, 0, &u_coords, &v_coords, &src, hs, 30.0, 30.0, &hs_style)
                .expect("expected GPU dispatch for large DEM crop");
            std::hint::black_box(b.len());
            gt.push(t0.elapsed());
        }
        let c_ms = median_us(ct) / 1000.0;
        let g_ms = median_us(gt) / 1000.0;
        println!(
            "| {:<35} | {:>4}x{:<4} | {:>12.2} | {:>17.2} | {:>6.2}x |",
            "Hillshade Lanczos3 PNG",
            n,
            n,
            c_ms,
            g_ms,
            c_ms / g_ms
        );
    }
}
