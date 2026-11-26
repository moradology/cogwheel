use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, DemAlgorithm, OutputFormat, ResampleMode};
use cogzilla_proj::TileCoord;
use cogzilla_server::{BandRequest, TileEngine, TileRequest};

/// Write a valid, self-contained 512×512 Float32 WebMercator (EPSG:3857) tiled GeoTIFF
/// so the benchmark runs identically on M5 and `thales` without external fixture files.
fn write_synthetic_dem_cog(path: &std::path::Path) {
    let w = 512u32;
    let h = 512u32;
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let fy = y as f32 * 0.025;
        for x in 0..w {
            let fx = x as f32 * 0.022;
            let elev = 1200.0 + 600.0 * fx.sin() * fy.cos() + 50.0 * (fx * 2.0 + fy).sin();
            pixels.extend_from_slice(&elev.to_le_bytes());
        }
    }

    // Minimal Classic TIFF (little-endian) with 1 tiled 512x512 float32 block + EPSG:3857 GeoKeys
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes()); // IFD at offset 8

    // Place in WebMercator centered near (0, 0) covering z=10 tile (512, 512)
    let tile_span = 2.0 * 20037508.342789244 / 1024.0; // z=10 tile width in meters (~39135 m)
    let origin_x = 0.0f64;
    let origin_y = 0.0f64;
    let tiepoint: [f64; 6] = [0.0, 0.0, 0.0, origin_x, origin_y, 0.0];

    // GeoKeyDirectory: version=1, rev=1.0, num_keys=2: GTModelTypeGeoKey(1024)=1 (Projected), ProjectedCSTypeGeoKey(3072)=3857
    let geokeys: [u16; 12] = [1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 3857];

    let data_len = pixels.len() as u32;

    // Scale pixel_scale so 512 pixels span exactly 1 z=10 WebMercator tile
    let px_m = tile_span / (w as f64);
    let pixel_scale = [px_m, px_m, 0.0];

    let mut entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, w),          // ImageWidth
        (257, 4, 1, h),          // ImageLength
        (258, 3, 1, 32),         // BitsPerSample = 32
        (259, 3, 1, 1),          // Compression = None
        (262, 3, 1, 1),          // Photometric = MinIsBlack
        (277, 3, 1, 1),          // SamplesPerPixel = 1
        (322, 4, 1, w),          // TileWidth
        (323, 4, 1, h),          // TileLength
        (324, 4, 1, 0),          // TileOffsets
        (325, 4, 1, data_len),   // TileByteCounts
        (339, 3, 1, 3),          // SampleFormat = IEEEFP
        (33550, 12, 3, 0),       // ModelPixelScaleTag
        (33922, 12, 6, 0),       // ModelTiepointTag
        (34735, 3, 12, 0),       // GeoKeyDirectoryTag
    ];
    entries.sort_by_key(|e| e.0);
    // Patch num_entries and offsets for 14 entries
    let num_entries = entries.len() as u16;
    let ifd_size = 2 + (num_entries as usize) * 12 + 4;
    let scale_off = (8 + ifd_size) as u32;
    let tie_off = scale_off + 24;
    let gk_off = tie_off + 48;
    let data_off = gk_off + 24;
    for e in &mut entries {
        match e.0 {
            324 => e.3 = data_off,
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
    buf.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0

    for v in pixel_scale {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in tiepoint {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in geokeys {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&pixels);
    std::fs::write(path, &buf).expect("write synthetic COG");
}

fn median_us(mut v: Vec<Duration>) -> f64 {
    v.sort();
    v[v.len() / 2].as_secs_f64() * 1e6
}

#[tokio::main]
async fn main() {
    let tmp_path = std::env::temp_dir().join("cogzilla_server_gpu_bench.tif");
    write_synthetic_dem_cog(&tmp_path);

    let store_cpu = Arc::new(CogStore::new(64 * 1024 * 1024));
    let cog_cpu = store_cpu.register_local("dem", &tmp_path).expect("open COG");
    // Construct CPU-only engine (disable GPU via PolicyOverride or clearing gpu field)
    let mut engine_cpu = TileEngine::with_store_and_tile_cache(store_cpu, 0);
    engine_cpu.gpu = None;

    let store_gpu = Arc::new(CogStore::new(64 * 1024 * 1024));
    let cog_gpu = store_gpu.register_local("dem", &tmp_path).expect("open COG");
    let engine_gpu = TileEngine::with_store_and_tile_cache(store_gpu, 0);

    let gpu_info = engine_gpu
        .gpu
        .as_ref()
        .expect("GPU backend must initialize for server_gpu_bench");
    println!(
        "TileEngine GPU backend: {} ({:?}, floor={:.1} us)\n",
        gpu_info.ctx.info.name, gpu_info.ctx.info.backend, gpu_info.policy.profile.dispatch_floor_us
    );

    println!("| End-to-End PNG Route | Size | CPU (us) | GPU+PNG (us) | Speedup | GPU Dispatches |");
    println!("| :--- | :--- | ---: | ---: | ---: | ---: |");

    let tile = TileCoord { z: 10, x: 512, y: 512 };
    let cases = [
        (
            "Single Lanczos3 PNG",
            BandRequest::Single {
                band_1based: 1,
                rescale: (500.0, 2000.0),
                colormap: ColormapName::Viridis,
            },
            ResampleMode::Lanczos,
        ),
        (
            "Hillshade Lanczos3 PNG",
            BandRequest::Dem {
                band_1based: 1,
                algorithm: DemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 1.5,
                },
                rescale: (0.0, 255.0),
                colormap: ColormapName::Viridis,
            },
            ResampleMode::Lanczos,
        ),
        (
            "Hillshade Cubic PNG",
            BandRequest::Dem {
                band_1based: 1,
                algorithm: DemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 1.5,
                },
                rescale: (0.0, 255.0),
                colormap: ColormapName::Viridis,
            },
            ResampleMode::Cubic,
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

            // Warmup both
            let cpu_png = engine_cpu.render_tile(&cog_cpu, &req).await.unwrap();
            let before = engine_gpu.stats.gpu_dispatches.load(Ordering::Relaxed);
            let gpu_png = engine_gpu.render_tile(&cog_gpu, &req).await.unwrap();
            assert!(cpu_png.starts_with(b"\x89PNG"));
            assert!(gpu_png.starts_with(b"\x89PNG"));

            let iters = 9usize;
            let mut cpu_t = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let b = engine_cpu.render_tile(&cog_cpu, &req).await.unwrap();
                std::hint::black_box(b.len());
                cpu_t.push(t0.elapsed());
            }

            let mut gpu_t = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let b = engine_gpu.render_tile(&cog_gpu, &req).await.unwrap();
                std::hint::black_box(b.len());
                gpu_t.push(t0.elapsed());
            }
            let after = engine_gpu.stats.gpu_dispatches.load(Ordering::Relaxed);

            let c_us = median_us(cpu_t);
            let g_us = median_us(gpu_t);
            println!(
                "| {:<22} | {:>3}x{:<3} | {:>8.1} | {:>12.1} | {:>6.2}x | {:>14} |",
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
