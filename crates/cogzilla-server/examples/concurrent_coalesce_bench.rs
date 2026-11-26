//! End-to-end concurrent heterogeneous HTTP-style micro-batching benchmark (`GpuCoalescer`).
//!
//! Spawns K=16 and K=32 concurrent Tokio tasks calling the normal per-request
//! `TileEngine::render_tile(...)` API across **4 distinct 3-band COGs** and **6 distinct styles**
//! spanning **all 5 GPU shader families** (Viridis Single-band, DEM Horn Hillshade, Terrarium,
//! NDVI `Spectral2`, 3-band `Rgb`, and Dual-Pol `SarRgb`) — verifying that `GpuCoalescer`
//! automatically packs arbitrary concurrent work into ~1 command buffer with zero batch API.

use std::sync::Arc;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{
    ColormapName, DemAlgorithm, OutputFormat, ResampleMode, SpectralExpr, TerrainEncoding,
};
use cogzilla_proj::TileCoord;
use cogzilla_server::engine::{BandRequest, TileEngine, TileRequest};

/// Build a 512x512 3-band chunky Float32 EPSG:3857 COG with positive values suitable for
/// single-band, DEM, NDVI (`(b1-b2)/(b1+b2)`), 3-band RGB, and Dual-Pol SAR dB RGB.
fn build_512_3band_mercator_cog_bytes(seed: f32) -> Vec<u8> {
    let w: u32 = 512;
    let h: u32 = 512;
    let spp: u32 = 3;
    let tile_bytes = (w * h * spp * 4) as usize;
    let mut tile_data = Vec::with_capacity(tile_bytes);
    for y in 0..h {
        for x in 0..w {
            let b0 = 0.20 + 0.15 * (x as f32 * 0.025 + seed).sin().abs() + 0.10 * (y as f32 * 0.02).cos().abs();
            let b1 = 0.08 + 0.07 * (y as f32 * 0.03 + seed * 1.7).cos().abs();
            let b2 = 0.12 + 0.09 * ((x + y) as f32 * 0.015 + seed).sin().abs();
            tile_data.extend_from_slice(&b0.to_le_bytes());
            tile_data.extend_from_slice(&b1.to_le_bytes());
            tile_data.extend_from_slice(&b2.to_le_bytes());
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

    let bps_off = write_extra(&mut buf, &[32, 0, 32, 0, 32, 0]);
    let sf_off = write_extra(&mut buf, &[3, 0, 3, 0, 3, 0]);

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
        (258, 3, spp, bps_off),
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (277, 3, 1, spp),
        (284, 3, 1, 1),
        (322, 4, 1, w),
        (323, 4, 1, h),
        (324, 4, 1, tile_off),
        (325, 4, 1, tile_bytes as u32),
        (339, 3, spp, sf_off),
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

fn build_mixed_workload(
    k: usize,
    resample: ResampleMode,
    format: OutputFormat,
) -> Vec<(usize, TileRequest)> {
    (0..k)
        .map(|i| {
            let cog_idx = i % 4;
            let dx = (i % 8) as u32;
            let dy = ((i / 8) % 8) as u32;
            let bands = match i % 6 {
                0 => BandRequest::Single {
                    band_1based: 1,
                    rescale: (0.0, 0.5),
                    colormap: ColormapName::Viridis,
                },
                1 => BandRequest::Dem {
                    band_1based: 1,
                    algorithm: DemAlgorithm::Hillshade {
                        azimuth_deg: 315.0,
                        altitude_deg: 45.0,
                        z_factor: 100.0,
                    },
                    colormap: ColormapName::Grayscale,
                    rescale: (0.0, 255.0),
                },
                2 => BandRequest::Terrain {
                    band_1based: 1,
                    encoding: TerrainEncoding::Terrarium,
                },
                3 => BandRequest::Spectral2 {
                    expr: SpectralExpr::NormalizedDiff,
                    bands_1based: [1, 2],
                    rescale: (-1.0, 1.0),
                    colormap: ColormapName::Magma,
                },
                4 => BandRequest::Rgb {
                    bands_1based: [1, 2, 3],
                    rescale_min: [0.0, 0.0, 0.0],
                    rescale_max: [0.5, 0.5, 0.5],
                },
                _ => BandRequest::SarRgb {
                    bands_1based: [1, 2],
                    rescale_min: [-20.0, -25.0, 0.0],
                    rescale_max: [0.0, 0.0, 10.0],
                },
            };
            (
                cog_idx,
                TileRequest {
                    tile: TileCoord {
                        z: 10,
                        x: 512 + dx,
                        y: 512 + dy,
                    },
                    tile_size: 256,
                    bands,
                    resample,
                    format,
                    unscale: false,
                },
            )
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let mut paths = Vec::with_capacity(4);
    for c in 0..4 {
        let p = std::env::temp_dir().join(format!("cogzilla_coalesce_5fam_bench_{c}.tif"));
        std::fs::write(&p, build_512_3band_mercator_cog_bytes(c as f32 * 0.9)).expect("write COG");
        paths.push(p);
    }

    let cpu_store = Arc::new(CogStore::new(64 * 1024 * 1024));
    let cpu_cogs: Vec<_> = paths
        .iter()
        .enumerate()
        .map(|(i, p)| cpu_store.register_local(&format!("c{i}"), p).expect("open"))
        .collect();
    let mut cpu_engine = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_engine.gpu = None;
    let cpu_engine = Arc::new(cpu_engine);

    let gpu_store = Arc::new(CogStore::new(64 * 1024 * 1024));
    let gpu_cogs: Vec<_> = paths
        .iter()
        .enumerate()
        .map(|(i, p)| gpu_store.register_local(&format!("g{i}"), p).expect("open"))
        .collect();
    let gpu_engine = Arc::new(TileEngine::with_store_and_tile_cache(gpu_store, 0));
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };

    println!(
        "5-Family Heterogeneous GpuCoalescer backend: {} ({:?}, floor={:.1} us)\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );
    println!("| 5-Family Mixed Burst (4 COGs x 6 styles, 256x256) | K | Conc CPU (ms) | Serial GPU (ms) | Coalesced GPU (ms) | vs Conc CPU | vs Serial GPU | Tiles/s | Batches (avg K) |");
    println!("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");

    let cases = [
        ("6-Style Mixed Cubic PNG", 32usize, ResampleMode::Cubic, OutputFormat::Png),
        ("6-Style Mixed Lanczos3 PNG", 16usize, ResampleMode::Lanczos, OutputFormat::Png),
        ("6-Style Mixed Lanczos3 PNG", 32usize, ResampleMode::Lanczos, OutputFormat::Png),
        ("6-Style Mixed Lanczos3 RawRgba", 32usize, ResampleMode::Lanczos, OutputFormat::RawRgba),
    ];

    let iters = 5usize;
    for (name, k, resample, format) in cases {
        let workload = build_mixed_workload(k, resample, format);

        // Warmup slabs + multi-band slabs + decode cache on both engines
        for (ci, req) in &workload {
            let _ = cpu_engine.render_tile(&cpu_cogs[*ci], req).await.unwrap();
            let _ = gpu_engine.render_tile(&gpu_cogs[*ci], req).await.unwrap();
        }

        // 1. Concurrent CPU across Tokio worker pool
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut handles = Vec::with_capacity(k);
            for (ci, req) in &workload {
                let eng = Arc::clone(&cpu_engine);
                let cog = Arc::clone(&cpu_cogs[*ci]);
                let req = req.clone();
                handles.push(tokio::spawn(async move {
                    eng.render_tile(&cog, &req).await.unwrap()
                }));
            }
            for h in handles {
                let _ = h.await.unwrap();
            }
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // 2. Serial GPU (K=1 sequential)
        let t0 = Instant::now();
        for _ in 0..iters {
            for (ci, req) in &workload {
                let _ = gpu_engine.render_tile(&gpu_cogs[*ci], req).await.unwrap();
            }
        }
        let serial_gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // 3. Concurrent GPU via GpuCoalescer (all 5 shader families packed into ~1 submit!)
        let stats_before = gpu.coalescer.stats();
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut handles = Vec::with_capacity(k);
            for (ci, req) in &workload {
                let eng = Arc::clone(&gpu_engine);
                let cog = Arc::clone(&gpu_cogs[*ci]);
                let req = req.clone();
                handles.push(tokio::spawn(async move {
                    eng.render_tile(&cog, &req).await.unwrap()
                }));
            }
            for h in handles {
                let _ = h.await.unwrap();
            }
        }
        let coal_gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let stats_after = gpu.coalescer.stats();
        let batches = stats_after.batches_submitted - stats_before.batches_submitted;
        let jobs = stats_after.jobs_executed - stats_before.jobs_executed;
        let avg_k = if batches > 0 {
            jobs as f64 / batches as f64
        } else {
            0.0
        };
        let batches_per_burst = batches as f64 / iters as f64;
        let tps = (k as f64) / (coal_gpu_ms / 1000.0);

        println!(
            "| {name} | {k} | {cpu_ms:.2} | {serial_gpu_ms:.2} | {coal_gpu_ms:.2} | {:.2}x | {:.2}x | {:.0} | {:.1} ({:.1}) |",
            cpu_ms / coal_gpu_ms,
            serial_gpu_ms / coal_gpu_ms,
            tps,
            batches_per_burst,
            avg_k,
        );
    }

    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}
