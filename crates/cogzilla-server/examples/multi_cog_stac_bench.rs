//! End-to-end Multi-COG STAC (`TileEngine::render_multi_cog_tile`) GPU parity & benchmark.
//!
//! Simulates a Sentinel-2 L2A / Sentinel-1 STAC scene where each band lives in a separate
//! single-band COG (`B04.tif`, `B03.tif`, `B02.tif`, `B08.tif`) and compares CPU vs GPU
//! (both 512x512 single-tile and 32x 256x256 concurrent coalesced bursts) for True-Color RGB,
//! NDVI (`Spectral2`), EVI (`Spectral3`), and Dual-Pol SAR RGB.

use std::sync::Arc;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, OutputFormat, ResampleMode, SpectralExpr};
use cogzilla_proj::TileCoord;
use cogzilla_server::engine::{BandRequest, TileEngine, TileRequest};

fn build_512_single_band_cog(seed: f32, bias: f32) -> Vec<u8> {
    let w: u32 = 512;
    let h: u32 = 512;
    let tile_bytes = (w * h * 4) as usize;
    let mut tile_data = Vec::with_capacity(tile_bytes);
    for y in 0..h {
        for x in 0..w {
            let v = bias
                + 0.14 * (x as f32 * 0.025 + seed).sin().abs()
                + 0.10 * (y as f32 * 0.03 + seed * 1.3).cos().abs();
            tile_data.extend_from_slice(&v.to_le_bytes());
        }
    }

    let world_half = 20_037_508.342_789_244_f64;
    let tile_span_z10 = (2.0 * world_half) / 1024.0;
    let span = tile_span_z10 * 8.0;

    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    let num_entries: u16 = 16;
    buf.extend_from_slice(&num_entries.to_le_bytes());
    let entries_start = buf.len();
    buf.resize(entries_start + (num_entries as usize) * 12 + 4, 0);

    let write_extra = |buf: &mut Vec<u8>, bytes: &[u8]| -> u32 {
        let off = buf.len() as u32;
        buf.extend_from_slice(bytes);
        if buf.len() % 2 == 1 {
            buf.push(0);
        }
        off
    };

    let mut scale_bytes = Vec::with_capacity(24);
    scale_bytes.extend_from_slice(&(span / w as f64).to_le_bytes());
    scale_bytes.extend_from_slice(&(span / h as f64).to_le_bytes());
    scale_bytes.extend_from_slice(&0.0f64.to_le_bytes());
    let scale_off = write_extra(&mut buf, &scale_bytes);

    let mut tie_bytes = Vec::with_capacity(48);
    for v in [0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0] {
        tie_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let tie_off = write_extra(&mut buf, &tie_bytes);

    let gkd: [u16; 16] = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3857];
    let mut gkd_bytes = Vec::with_capacity(32);
    for v in gkd {
        gkd_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let gkd_off = write_extra(&mut buf, &gkd_bytes);

    let tile_off = buf.len() as u32;
    buf.extend_from_slice(&tile_data);

    let entries: [(u16, u16, u32, u32); 16] = [
        (256, 4, 1, w),
        (257, 4, 1, h),
        (258, 3, 1, 32),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (277, 3, 1, 1),
        (284, 3, 1, 1),
        (322, 4, 1, w),
        (323, 4, 1, h),
        (324, 4, 1, tile_off),
        (325, 4, 1, tile_bytes as u32),
        (339, 3, 1, 3),
        (33550, 12, 3, scale_off),
        (33922, 12, 6, tie_off),
        (34735, 3, gkd.len() as u32, gkd_off),
        (42113, 2, 4, u32::from_le_bytes(*b"nan\0")),
    ];
    for (i, (tag, typ, cnt, val)) in entries.iter().enumerate() {
        let p = entries_start + i * 12;
        buf[p..p + 2].copy_from_slice(&tag.to_le_bytes());
        buf[p + 2..p + 4].copy_from_slice(&typ.to_le_bytes());
        buf[p + 4..p + 8].copy_from_slice(&cnt.to_le_bytes());
        buf[p + 8..p + 12].copy_from_slice(&val.to_le_bytes());
    }
    buf
}

#[tokio::main]
async fn main() {
    let assets = [
        ("B04_red", 0.2f32, 0.12f32),
        ("B03_green", 0.9f32, 0.10f32),
        ("B02_blue", 1.7f32, 0.08f32),
        ("B08_nir", 2.4f32, 0.30f32),
    ];
    let mut paths = Vec::new();
    for (name, seed, bias) in assets {
        let p = std::env::temp_dir().join(format!("cogzilla_multicog_{name}.tif"));
        std::fs::write(&p, build_512_single_band_cog(seed, bias)).expect("write asset COG");
        paths.push((name, p));
    }

    let cpu_store = Arc::new(CogStore::new(64 * 1024 * 1024));
    let cpu_cogs: Vec<_> = paths
        .iter()
        .map(|(n, p)| cpu_store.register_local(*n, p).expect("open cpu"))
        .collect();
    let mut cpu_engine = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_engine.gpu = None;
    let cpu_engine = Arc::new(cpu_engine);

    let gpu_store = Arc::new(CogStore::new(64 * 1024 * 1024));
    let gpu_cogs: Vec<_> = paths
        .iter()
        .map(|(n, p)| gpu_store.register_local(*n, p).expect("open gpu"))
        .collect();
    let gpu_engine = Arc::new(TileEngine::with_store_and_tile_cache(gpu_store, 0));
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };

    let workloads: Vec<(&str, BandRequest)> = vec![
        (
            "Multi-COG RGB (B04/B03/B02)",
            BandRequest::Rgb {
                bands_1based: [1, 2, 3],
                rescale_min: [0.0, 0.0, 0.0],
                rescale_max: [0.5, 0.5, 0.5],
            },
        ),
        (
            "Multi-COG NDVI (B08/B04)",
            BandRequest::Spectral2 {
                expr: SpectralExpr::NormalizedDiff,
                bands_1based: [4, 1],
                rescale: (-1.0, 1.0),
                colormap: ColormapName::Viridis,
            },
        ),
        (
            "Multi-COG EVI (B08/B04/B02)",
            BandRequest::Spectral3 {
                expr: SpectralExpr::Evi,
                bands_1based: [4, 1, 3],
                rescale: (-1.0, 1.0),
                colormap: ColormapName::Magma,
            },
        ),
        (
            "Multi-COG Dual-Pol SAR RGB",
            BandRequest::SarRgb {
                bands_1based: [1, 2],
                rescale_min: [-20.0, -25.0, 0.0],
                rescale_max: [0.0, 0.0, 10.0],
            },
        ),
    ];

    // 1. CPU-vs-GPU pixel parity check on 512x512 Lanczos3 RawRgba
    for (name, bands) in &workloads {
        let req = TileRequest {
            tile: TileCoord { z: 10, x: 513, y: 513 },
            tile_size: 512,
            bands: bands.clone(),
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
            unscale: false,
        };
        let cpu_out = cpu_engine.render_multi_cog_tile(&cpu_cogs, &req).await.unwrap();
        let gpu_out = gpu_engine.render_multi_cog_tile(&gpu_cogs, &req).await.unwrap();
        assert_eq!(cpu_out.len(), gpu_out.len());
        let mut max_diff = 0i32;
        for (&a, &b) in cpu_out.iter().zip(gpu_out.iter()) {
            max_diff = max_diff.max((a as i32 - b as i32).abs());
        }
        assert!(
            max_diff <= 6,
            "{name} multi-COG CPU/GPU max channel diff {max_diff} > 6"
        );
    }

    println!(
        "Multi-COG STAC GPU backend: {} ({:?}, floor={:.1} us) [parity PASS]\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );
    println!("| Multi-COG STAC Workload (4 separate COGs, Lanczos3 PNG) | Mode | CPU (ms) | GPU (ms) | Speedup | GPU Tiles/s |");
    println!("| :--- | :--- | ---: | ---: | ---: | ---: |");

    let iters = 5usize;
    for (name, bands) in &workloads {
        // Single 512x512 tile (warm VRAM multi-band slab)
        let req512 = TileRequest {
            tile: TileCoord { z: 10, x: 513, y: 513 },
            tile_size: 512,
            bands: bands.clone(),
            resample: ResampleMode::Lanczos,
            format: OutputFormat::Png,
            unscale: false,
        };
        let _ = cpu_engine.render_multi_cog_tile(&cpu_cogs, &req512).await.unwrap();
        let _ = gpu_engine.render_multi_cog_tile(&gpu_cogs, &req512).await.unwrap();

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = cpu_engine.render_multi_cog_tile(&cpu_cogs, &req512).await.unwrap();
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = gpu_engine.render_multi_cog_tile(&gpu_cogs, &req512).await.unwrap();
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!(
            "| {name} | 1x 512x512 | {cpu_ms:.2} | {gpu_ms:.2} | {:.2}x | {:.0} |",
            cpu_ms / gpu_ms,
            1000.0 / gpu_ms
        );

        // 32-tile concurrent 256x256 burst via GpuCoalescer
        let k = 32usize;
        let reqs: Vec<TileRequest> = (0..k)
            .map(|i| TileRequest {
                tile: TileCoord {
                    z: 10,
                    x: 512 + (i % 8) as u32,
                    y: 512 + (i / 8) as u32,
                },
                tile_size: 256,
                bands: bands.clone(),
                resample: ResampleMode::Lanczos,
                format: OutputFormat::Png,
                unscale: false,
            })
            .collect();
        let _ = cpu_engine.render_multi_cog_tile(&cpu_cogs, &reqs[0]).await.unwrap();
        let _ = gpu_engine.render_multi_cog_tile(&gpu_cogs, &reqs[0]).await.unwrap();

        let t0 = Instant::now();
        for _ in 0..iters {
            let mut hs = Vec::with_capacity(k);
            for r in &reqs {
                let eng = Arc::clone(&cpu_engine);
                let cs = cpu_cogs.clone();
                let r = r.clone();
                hs.push(tokio::spawn(async move {
                    eng.render_multi_cog_tile(&cs, &r).await.unwrap()
                }));
            }
            for h in hs {
                let _ = h.await.unwrap();
            }
        }
        let cpu_burst_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            let mut hs = Vec::with_capacity(k);
            for r in &reqs {
                let eng = Arc::clone(&gpu_engine);
                let cs = gpu_cogs.clone();
                let r = r.clone();
                hs.push(tokio::spawn(async move {
                    eng.render_multi_cog_tile(&cs, &r).await.unwrap()
                }));
            }
            for h in hs {
                let _ = h.await.unwrap();
            }
        }
        let gpu_burst_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!(
            "| {name} | 32x 256x256 conc | {cpu_burst_ms:.2} | {gpu_burst_ms:.2} | {:.2}x | {:.0} |",
            cpu_burst_ms / gpu_burst_ms,
            (k as f64) / (gpu_burst_ms / 1000.0)
        );
    }

    for (_, p) in paths {
        let _ = std::fs::remove_file(p);
    }
}
