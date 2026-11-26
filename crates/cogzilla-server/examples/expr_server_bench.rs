use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, ExprProgram, OutputFormat, ResampleMode};
use cogzilla_proj::TileCoord;
use cogzilla_server::{BandRequest, TileEngine, TileRequest};

/// Write a self-contained 512×512 3-band Planar Float32 WebMercator (EPSG:3857) COG.
fn write_synthetic_3band_cog(path: &std::path::Path) {
    let w = 512u32;
    let h = 512u32;
    let band_bytes = (w * h * 4) as u32;
    let mut band_data = Vec::with_capacity((band_bytes * 3) as usize);
    for b in 0..3u32 {
        for y in 0..h {
            let fy = y as f32 * 0.025;
            for x in 0..w {
                let fx = x as f32 * 0.022;
                let v = match b {
                    0 => 1500.0 + 1000.0 * fx.sin().abs(),
                    1 => 800.0 + 600.0 * fy.cos().abs(),
                    _ => 400.0 + 500.0 * (fx + fy).sin().abs(),
                };
                band_data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());

    let tile_span = 2.0 * 20037508.342789244 / 1024.0;
    let px_m = tile_span / (w as f64);
    let pixel_scale = [px_m, px_m, 0.0];
    let tiepoint: [f64; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let geokeys: [u16; 12] = [1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 3857];
    let bps: [u16; 3] = [32, 32, 32];
    let sf: [u16; 3] = [3, 3, 3];

    let mut entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, w),
        (257, 4, 1, h),
        (258, 3, 3, 0),    // BitsPerSample [32,32,32]
        (259, 3, 1, 1),    // Compression = None
        (262, 3, 1, 2),    // Photometric = RGB
        (277, 3, 1, 3),    // SamplesPerPixel = 3
        (284, 3, 1, 2),    // PlanarConfiguration = 2 (Planar)
        (322, 4, 1, w),    // TileWidth
        (323, 4, 1, h),    // TileLength
        (324, 4, 3, 0),    // TileOffsets [3]
        (325, 4, 3, 0),    // TileByteCounts [3]
        (339, 3, 3, 0),    // SampleFormat [3,3,3]
        (33550, 12, 3, 0), // ModelPixelScaleTag
        (33922, 12, 6, 0), // ModelTiepointTag
        (34735, 3, 12, 0), // GeoKeyDirectoryTag
    ];
    entries.sort_by_key(|e| e.0);
    let num_entries = entries.len() as u16;
    let ifd_size = 2 + (num_entries as usize) * 12 + 4;
    let bps_off = (8 + ifd_size) as u32;
    let sf_off = bps_off + 6;
    let toff_off = sf_off + 6;
    let tcnt_off = toff_off + 12;
    let scale_off = tcnt_off + 12;
    let tie_off = scale_off + 24;
    let gk_off = tie_off + 48;
    let data_off = gk_off + 24;

    let tile_offsets: [u32; 3] = [data_off, data_off + band_bytes, data_off + 2 * band_bytes];
    let tile_counts: [u32; 3] = [band_bytes, band_bytes, band_bytes];

    for e in &mut entries {
        match e.0 {
            258 => e.3 = bps_off,
            339 => e.3 = sf_off,
            324 => e.3 = toff_off,
            325 => e.3 = tcnt_off,
            33550 => e.3 = scale_off,
            33922 => e.3 = tie_off,
            34735 => e.3 = gk_off,
            _ => {}
        }
    }

    buf.extend_from_slice(&num_entries.to_le_bytes());
    for (tag, typ, cnt, val) in entries {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&typ.to_le_bytes());
        buf.extend_from_slice(&cnt.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf.extend_from_slice(&0u32.to_le_bytes());

    for v in bps { buf.extend_from_slice(&v.to_le_bytes()); }
    for v in sf { buf.extend_from_slice(&v.to_le_bytes()); }
    for v in tile_offsets { buf.extend_from_slice(&v.to_le_bytes()); }
    for v in tile_counts { buf.extend_from_slice(&v.to_le_bytes()); }
    for v in pixel_scale { buf.extend_from_slice(&v.to_le_bytes()); }
    for v in tiepoint { buf.extend_from_slice(&v.to_le_bytes()); }
    for v in geokeys { buf.extend_from_slice(&v.to_le_bytes()); }
    buf.extend_from_slice(&band_data);
    std::fs::write(path, &buf).expect("write 3-band COG");
}

fn median_us(mut v: Vec<Duration>) -> f64 {
    v.sort();
    v[v.len() / 2].as_secs_f64() * 1e6
}

#[tokio::main]
async fn main() {
    let tmp_path = std::env::temp_dir().join("cogzilla_expr_server_bench.tif");
    write_synthetic_3band_cog(&tmp_path);

    let store_cpu = Arc::new(CogStore::new(64 * 1024 * 1024));
    let cog_cpu = store_cpu.register_local("s2", &tmp_path).expect("open COG");
    let mut engine_cpu = TileEngine::with_store_and_tile_cache(store_cpu, 0);
    engine_cpu.gpu = None;

    let store_gpu = Arc::new(CogStore::new(64 * 1024 * 1024));
    let cog_gpu = store_gpu.register_local("s2", &tmp_path).expect("open COG");
    let engine_gpu = TileEngine::with_store_and_tile_cache(store_gpu, 0);

    let gpu = engine_gpu.gpu.as_ref().expect("GPU backend required");
    println!(
        "TileEngine GPU backend: {} ({:?}, floor={:.1} us)\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );

    let ndvi = ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap();
    let sar = ExprProgram::compile("db(b1)-db(b2)+sqrt(b3)").unwrap();
    let tile = TileCoord { z: 10, x: 512, y: 512 };

    println!("| End-to-End Multi-Band PNG Route | Size | CPU (us) | GPU+PNG (us) | Speedup | GPU Dispatches |");
    println!("| :--- | :--- | ---: | ---: | ---: | ---: |");

    let cases = [
        (
            "Expr NDVI Lanczos3 PNG",
            BandRequest::CustomExpr {
                prog: ndvi,
                rescale: (-0.2, 0.8),
                colormap: ColormapName::Viridis,
            },
            ResampleMode::Lanczos,
        ),
        (
            "Expr dB+sqrt Lanczos3 PNG",
            BandRequest::CustomExpr {
                prog: sar,
                rescale: (0.0, 60.0),
                colormap: ColormapName::Viridis,
            },
            ResampleMode::Lanczos,
        ),
        (
            "RGB 3-band Lanczos3 PNG",
            BandRequest::Rgb {
                bands_1based: [1, 2, 3],
                rescale_min: [0.0; 3],
                rescale_max: [2500.0, 1500.0, 1000.0],
            },
            ResampleMode::Lanczos,
        ),
    ];

    for (label, bands, resample) in cases {
        for &tile_size in &[256u32, 512u32] {
            let req = TileRequest {
                tile,
                tile_size,
                bands: bands.clone(),
                resample,
                format: OutputFormat::Png,
                unscale: false,
            };

            let cpu_png = engine_cpu.render_tile(&cog_cpu, &req).await.unwrap();
            let before = engine_gpu.stats.gpu_dispatches.load(Ordering::Relaxed);
            let gpu_png = engine_gpu.render_tile(&cog_gpu, &req).await.unwrap();
            assert!(cpu_png.starts_with(b"\x89PNG"));
            assert!(gpu_png.starts_with(b"\x89PNG"));

            let iters = 7usize;
            let mut ct = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let b = engine_cpu.render_tile(&cog_cpu, &req).await.unwrap();
                std::hint::black_box(b.len());
                ct.push(t0.elapsed());
            }
            let mut gt = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let b = engine_gpu.render_tile(&cog_gpu, &req).await.unwrap();
                std::hint::black_box(b.len());
                gt.push(t0.elapsed());
            }
            let after = engine_gpu.stats.gpu_dispatches.load(Ordering::Relaxed);
            let c_us = median_us(ct);
            let g_us = median_us(gt);
            println!(
                "| {:<29} | {:>3}x{:<3} | {:>8.1} | {:>12.1} | {:>6.2}x | {:>14} |",
                label,
                tile_size,
                tile_size,
                c_us,
                g_us,
                c_us / g_us,
                after - before
            );
        }
    }

    let _ = std::fs::remove_file(&tmp_path);
}
