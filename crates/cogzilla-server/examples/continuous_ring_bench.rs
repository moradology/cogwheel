//! Sustained continuous-load A/B benchmark for `GpuCoalescer` ring depth
//! (`COGZILLA_GPU_RING_DEPTH=1` synchronous vs `2` double-buffered vs `3` triple-buffered).
//!
//! Streams 256 heterogeneous requests (4 distinct 3-band COGs x 6 styles spanning all 5 GPU
//! shader families) through a fixed concurrency semaphore (32 or 64 in-flight HTTP clients)
//! so new requests arrive continuously while earlier batches execute and PNG-encode.

use std::sync::Arc;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{
    ColormapName, DemAlgorithm, OutputFormat, ResampleMode, SpectralExpr, TerrainEncoding,
};
use cogzilla_proj::TileCoord;
use cogzilla_server::engine::{BandRequest, TileEngine, TileRequest};
use tokio::sync::Semaphore;

fn build_512_3band_mercator_cog_bytes(seed: f32) -> Vec<u8> {
    let w: u32 = 512;
    let h: u32 = 512;
    let spp: u32 = 3;
    let tile_bytes = (w * h * spp * 4) as usize;
    let mut tile_data = Vec::with_capacity(tile_bytes);
    for y in 0..h {
        for x in 0..w {
            let b0 = 0.20
                + 0.15 * (x as f32 * 0.025 + seed).sin().abs()
                + 0.10 * (y as f32 * 0.02).cos().abs();
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

fn build_stream(total: usize, resample: ResampleMode, format: OutputFormat) -> Vec<(usize, TileRequest)> {
    (0..total)
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

async fn run_stream(
    engine: &Arc<TileEngine>,
    cogs: &[Arc<cogzilla_io::OpenedCog>],
    stream: &[(usize, TileRequest)],
    concurrency: usize,
) -> (f64, f64, f64) {
    let sem = Arc::new(Semaphore::new(concurrency));
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(stream.len());
    for (ci, req) in stream {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let eng = Arc::clone(engine);
        let cog = Arc::clone(&cogs[*ci]);
        let req = req.clone();
        handles.push(tokio::spawn(async move {
            let s = Instant::now();
            let _ = eng.render_tile(&cog, &req).await.unwrap();
            let lat_ms = s.elapsed().as_secs_f64() * 1000.0;
            drop(permit);
            lat_ms
        }));
    }
    let mut lats = Vec::with_capacity(handles.len());
    for h in handles {
        lats.push(h.await.unwrap());
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = lats[lats.len() / 2];
    let p99 = lats[(lats.len() * 99) / 100];
    (wall_ms, p50, p99)
}

#[tokio::main]
async fn main() {
    let depth_label = std::env::var("COGZILLA_GPU_RING_DEPTH").unwrap_or_else(|_| "3".into());
    let mut paths = Vec::with_capacity(4);
    for c in 0..4 {
        let p = std::env::temp_dir().join(format!("cogzilla_ring_bench_{c}.tif"));
        std::fs::write(&p, build_512_3band_mercator_cog_bytes(c as f32 * 0.9)).expect("write COG");
        paths.push(p);
    }

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

    let total = 256usize;
    println!(
        "Continuous Ring Bench [ring_depth={}] backend: {} ({:?}, floor={:.1} us)",
        depth_label, gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );
    println!("| Workload (256 tiles, 4 COGs x 6 styles) | Conc | Wall (ms) | Tiles/s | p50 (ms) | p99 (ms) | Batches (avg K) |");
    println!("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |");

    let cases = [
        ("Lanczos3 PNG", 32usize, ResampleMode::Lanczos, OutputFormat::Png),
        ("Lanczos3 PNG", 64usize, ResampleMode::Lanczos, OutputFormat::Png),
        ("Lanczos3 RawRgba", 32usize, ResampleMode::Lanczos, OutputFormat::RawRgba),
        ("Lanczos3 RawRgba", 64usize, ResampleMode::Lanczos, OutputFormat::RawRgba),
    ];

    // Warmup all slabs + multi-band slabs
    let warm = build_stream(24, ResampleMode::Lanczos, OutputFormat::Png);
    for (ci, req) in &warm {
        let _ = gpu_engine.render_tile(&gpu_cogs[*ci], req).await.unwrap();
    }

    let iters = 3usize;
    for (name, conc, resample, format) in cases {
        let stream = build_stream(total, resample, format);
        let st0 = gpu.coalescer.stats();
        let mut wall_sum = 0.0;
        let mut p50_sum = 0.0;
        let mut p99_sum = 0.0;
        for _ in 0..iters {
            let (w, p50, p99) = run_stream(&gpu_engine, &gpu_cogs, &stream, conc).await;
            wall_sum += w;
            p50_sum += p50;
            p99_sum += p99;
        }
        let st1 = gpu.coalescer.stats();
        let wall = wall_sum / iters as f64;
        let p50 = p50_sum / iters as f64;
        let p99 = p99_sum / iters as f64;
        let batches = st1.batches_submitted - st0.batches_submitted;
        let jobs = st1.jobs_executed - st0.jobs_executed;
        let avg_k = if batches > 0 {
            jobs as f64 / batches as f64
        } else {
            0.0
        };
        let tps = (total as f64) / (wall / 1000.0);
        println!(
            "| {name} | {conc} | {wall:.2} | {tps:.0} | {p50:.2} | {p99:.2} | {:.1} ({avg_k:.1}) |",
            batches as f64 / iters as f64,
        );
    }

    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}
