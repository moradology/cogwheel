//! Long-running realistic map-server workload benchmark (CPU vs GPU continuous batching).
//!
//! Exercises the real `TileEngine` (tiled 2048×2048 3-band F32 GeoTIFF COG on disk, zero tile
//! cache so every request actually renders) across 4 realistic traffic patterns:
//!   1. Staggered Pan & Zoom (16 clients, 0–6 ms jitter, 8s sustained, mixed RGB/DEM/NDVI/Single)
//!   2. High-Concurrency Tile Flood (32 clients, zero think time, 8s sustained)
//!   3. Mixed Live Tiles + Heavy Analytics Contention (12 live tile clients + 2 heavy 2048²
//!      bbox/stats/zonal clients, 8s sustained) — measures tail latency isolation
//!   4. Concurrency Scaling Sweep (1, 4, 16, 32 clients × 4s each on 512² RGB Lanczos3 JPEG)
//!
//! Run:
//!   cargo run --release -p cogzilla-server --example e2e_realistic_workload_bench

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, DemAlgorithm, ExprProgram, OutputFormat, ResampleMode};
use cogzilla_proj::TileCoord;
use cogzilla_server::engine::{BandRequest, TileEngine, TileRequest};
use cogzilla_server::titiler::{
    PolygonMask, StatsOptions, render_wgs84_bbox, render_wgs84_bbox_stats,
};
use cogzilla_server::zonal::compute_native_exact_polygon_stats;

/// Build a valid 2048×2048 3-band F32 tiled GeoTIFF (EPSG:4326, `[-10.0, 35.0, -8.0, 37.0]`,
/// 512×512 tiles, planar configuration = 2).
fn build_3band_2048_cog_bytes() -> Vec<u8> {
    let w: u32 = 2048;
    let h: u32 = 2048;
    let tw: u32 = 512;
    let th: u32 = 512;
    let tiles_x = w / tw;
    let tiles_y = h / th;
    let tiles_per_band = (tiles_x * tiles_y) as usize;
    let total_tiles = tiles_per_band * 3;
    let tile_bytes = (tw * th * 4) as usize;

    let mut tile_data: Vec<Vec<u8>> = Vec::with_capacity(total_tiles);
    for band in 0..3u32 {
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let mut raw = Vec::with_capacity(tile_bytes);
                for py in 0..th {
                    let gy = ty * th + py;
                    for px in 0..tw {
                        let gx = tx * tw + px;
                        let fx = gx as f32 / w as f32;
                        let fy = gy as f32 / h as f32;
                        let v = match band {
                            0 => 1200.0 + 900.0 * (fx * 14.0).sin() + 400.0 * (fy * 11.0).cos(),
                            1 => 650.0 + 450.0 * (fx * 9.0 + 1.0).cos() + 250.0 * (fy * 13.0).sin(),
                            _ => 400.0 + 300.0 * ((fx + fy) * 8.0).sin(),
                        };
                        raw.extend_from_slice(&v.to_le_bytes());
                    }
                }
                tile_data.push(raw);
            }
        }
    }

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
    let dx = 2.0f64 / w as f64;
    let dy = 2.0f64 / h as f64;
    scale_bytes.extend_from_slice(&dx.to_le_bytes());
    scale_bytes.extend_from_slice(&dy.to_le_bytes());
    scale_bytes.extend_from_slice(&0.0f64.to_le_bytes());
    let scale_off = write_extra(&mut buf, &scale_bytes);

    let mut tie_bytes = Vec::with_capacity(48);
    for v in [0.0f64, 0.0, 0.0, -10.0, 37.0, 0.0] {
        tie_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let tie_off = write_extra(&mut buf, &tie_bytes);

    let gkd: [u16; 16] = [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326];
    let mut gkd_bytes = Vec::with_capacity(32);
    for v in gkd {
        gkd_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let gkd_off = write_extra(&mut buf, &gkd_bytes);

    let offsets_pos = buf.len() as u32;
    buf.resize(buf.len() + total_tiles * 4, 0);
    let counts_pos = buf.len() as u32;
    buf.resize(buf.len() + total_tiles * 4, 0);

    let mut offsets = Vec::with_capacity(total_tiles);
    let mut counts = Vec::with_capacity(total_tiles);
    for td in &tile_data {
        offsets.push(buf.len() as u32);
        counts.push(td.len() as u32);
        buf.extend_from_slice(td);
    }

    for (i, (&off, &cnt)) in offsets.iter().zip(&counts).enumerate() {
        let po = offsets_pos as usize + i * 4;
        buf[po..po + 4].copy_from_slice(&off.to_le_bytes());
        let pc = counts_pos as usize + i * 4;
        buf[pc..pc + 4].copy_from_slice(&cnt.to_le_bytes());
    }

    let entries: [(u16, u16, u32, u32); 16] = [
        (256, 4, 1, w),
        (257, 4, 1, h),
        (258, 3, 3, bps_off),
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (277, 3, 1, 3),
        (284, 3, 1, 2),
        (322, 4, 1, tw),
        (323, 4, 1, th),
        (324, 4, total_tiles as u32, offsets_pos),
        (325, 4, total_tiles as u32, counts_pos),
        (339, 3, 3, sf_off),
        (33550, 12, 3, scale_off),
        (33922, 12, 6, tie_off),
        (34735, 3, 16, gkd_off),
        (42113, 2, 1, 0),
    ];
    for (idx, &(tag, typ, cnt, val)) in entries.iter().enumerate() {
        let p = entries_start + idx * 12;
        buf[p..p + 2].copy_from_slice(&tag.to_le_bytes());
        buf[p + 2..p + 4].copy_from_slice(&typ.to_le_bytes());
        buf[p + 4..p + 8].copy_from_slice(&cnt.to_le_bytes());
        buf[p + 8..p + 12].copy_from_slice(&val.to_le_bytes());
    }
    buf
}

fn tile_xy_for_lonlat(z: u8, lon: f64, lat: f64) -> TileCoord {
    let n = (1u64 << z) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor().clamp(0.0, n - 1.0) as u32;
    let lat_rad = lat.to_radians();
    let y = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0 * n)
        .floor()
        .clamp(0.0, n - 1.0) as u32;
    TileCoord { z, x, y }
}

fn build_tile_pool() -> Vec<TileCoord> {
    let mut pool = Vec::new();
    for z in 9u8..=12u8 {
        let t0 = tile_xy_for_lonlat(z, -9.85, 36.85);
        let t1 = tile_xy_for_lonlat(z, -8.15, 35.15);
        for y in t0.y..=t1.y {
            for x in t0.x..=t1.x {
                pool.push(TileCoord { z, x, y });
            }
        }
    }
    pool
}

#[derive(Clone)]
struct LatencyStats {
    rps: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

fn summarize(mut lats_ms: Vec<f64>, elapsed: Duration) -> LatencyStats {
    lats_ms.sort_by(|a, b| a.total_cmp(b));
    let n = lats_ms.len();
    let pct = |p: f64| -> f64 {
        if n == 0 {
            return 0.0;
        }
        let idx = ((p / 100.0) * (n - 1) as f64).round() as usize;
        lats_ms[idx.min(n - 1)]
    };
    let elapsed_s = elapsed.as_secs_f64();
    LatencyStats {
        rps: n as f64 / elapsed_s.max(1e-9),
        p50: pct(50.0),
        p95: pct(95.0),
        p99: pct(99.0),
    }
}

fn make_mixed_request(tile: TileCoord, seed: u64, ndvi_prog: &ExprProgram) -> TileRequest {
    match seed % 20 {
        // 40%: 512x512 RGB Lanczos3 JPEG
        0..=7 => TileRequest {
            tile,
            tile_size: 512,
            bands: BandRequest::Rgb {
                bands_1based: [1, 2, 3],
                rescale_min: [0.0, 0.0, 0.0],
                rescale_max: [2500.0, 1500.0, 1000.0],
            },
            resample: ResampleMode::Lanczos,
            format: OutputFormat::Jpeg { quality: 85 },
            unscale: false,
        },
        // 25%: 256x256 DEM Horn Hillshade Lanczos3 PNG
        8..=12 => TileRequest {
            tile,
            tile_size: 256,
            bands: BandRequest::Dem {
                band_1based: 1,
                algorithm: DemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 1.0,
                },
                rescale: (0.0, 255.0),
                colormap: ColormapName::Grayscale,
            },
            resample: ResampleMode::Lanczos,
            format: OutputFormat::Png,
            unscale: false,
        },
        // 20%: 512x512 NDVI CustomExpr Cubic + Viridis PNG
        13..=16 => TileRequest {
            tile,
            tile_size: 512,
            bands: BandRequest::CustomExpr {
                prog: ndvi_prog.clone(),
                rescale: (-1.0, 1.0),
                colormap: ColormapName::Viridis,
            },
            resample: ResampleMode::Cubic,
            format: OutputFormat::Png,
            unscale: false,
        },
        // 15%: 256x256 Single-band Lanczos3 Viridis PNG
        _ => TileRequest {
            tile,
            tile_size: 256,
            bands: BandRequest::Single {
                band_1based: 1,
                rescale: (0.0, 2500.0),
                colormap: ColormapName::Viridis,
            },
            resample: ResampleMode::Lanczos,
            format: OutputFormat::Png,
            unscale: false,
        },
    }
}

async fn run_tile_soak(
    engine: Arc<TileEngine>,
    cog: Arc<cogzilla_io::OpenedCog>,
    pool: Arc<Vec<TileCoord>>,
    ndvi_prog: ExprProgram,
    concurrency: usize,
    duration: Duration,
    max_jitter_ms: u64,
    rgb_only: bool,
) -> LatencyStats {
    let deadline = Instant::now() + duration;
    let start = Instant::now();
    let mut handles = Vec::with_capacity(concurrency);
    for worker in 0..concurrency {
        let eng = Arc::clone(&engine);
        let cog = Arc::clone(&cog);
        let pool = Arc::clone(&pool);
        let prog = ndvi_prog.clone();
        handles.push(tokio::spawn(async move {
            let mut lats = Vec::with_capacity(256);
            let mut rng: u64 = 0x9e37_79b9_7f4a_7c15 ^ ((worker as u64 + 1) * 0xbf58_476d_1ce4_e5b9);
            let mut next_u64 = || {
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
            };
            // Stagger initial client starts so they don't synchronize in lockstep
            if max_jitter_ms > 0 {
                let init_ms = next_u64() % (max_jitter_ms + 1);
                tokio::time::sleep(Duration::from_millis(init_ms)).await;
            }
            while Instant::now() < deadline {
                let s = next_u64();
                let tile = pool[(s as usize) % pool.len()];
                let req = if rgb_only {
                    TileRequest {
                        tile,
                        tile_size: 512,
                        bands: BandRequest::Rgb {
                            bands_1based: [1, 2, 3],
                            rescale_min: [0.0, 0.0, 0.0],
                            rescale_max: [2500.0, 1500.0, 1000.0],
                        },
                        resample: ResampleMode::Lanczos,
                        format: OutputFormat::Jpeg { quality: 85 },
                        unscale: false,
                    }
                } else {
                    make_mixed_request(tile, s >> 16, &prog)
                };
                let t0 = Instant::now();
                let res = eng.render_tile(&cog, &req).await.expect("render_tile ok");
                std::hint::black_box(res.len());
                lats.push(t0.elapsed().as_secs_f64() * 1e3);
                if max_jitter_ms > 0 {
                    let think_us = next_u64() % (max_jitter_ms * 1000);
                    if think_us >= 200 {
                        tokio::time::sleep(Duration::from_micros(think_us)).await;
                    }
                }
            }
            lats
        }));
    }
    let mut all = Vec::new();
    for h in handles {
        all.extend(h.await.unwrap());
    }
    summarize(all, start.elapsed())
}

async fn run_mixed_contention_soak(
    engine: Arc<TileEngine>,
    cog: Arc<cogzilla_io::OpenedCog>,
    pool: Arc<Vec<TileCoord>>,
    ndvi_prog: ExprProgram,
    duration: Duration,
) -> (LatencyStats, LatencyStats) {
    let deadline = Instant::now() + duration;
    let start = Instant::now();

    // 12 live tile clients (2 ms pacing)
    let mut tile_handles = Vec::with_capacity(12);
    for worker in 0..12usize {
        let eng = Arc::clone(&engine);
        let cog = Arc::clone(&cog);
        let pool = Arc::clone(&pool);
        let prog = ndvi_prog.clone();
        tile_handles.push(tokio::spawn(async move {
            let mut lats = Vec::new();
            let mut rng: u64 = 0x1234_5678_9abc_def0 ^ ((worker as u64 + 1) * 0x94d0_49bb_1331_11eb);
            let mut next_u64 = || {
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
            };
            while Instant::now() < deadline {
                let s = next_u64();
                let tile = pool[(s as usize) % pool.len()];
                let req = make_mixed_request(tile, s >> 16, &prog);
                let t0 = Instant::now();
                let res = eng.render_tile(&cog, &req).await.expect("live tile ok");
                std::hint::black_box(res.len());
                lats.push(t0.elapsed().as_secs_f64() * 1e3);
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            lats
        }));
    }

    // 2 heavy analytics clients running 2048² RGB bbox JPEG, 2048² NDVI bbox stats, and 1024² zonal stats
    let diamond = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[
            [-9.0, 36.8],
            [-8.2, 36.0],
            [-9.0, 35.2],
            [-9.8, 36.0],
            [-9.0, 36.8]
        ]]
    });
    let (pm, _) = PolygonMask::from_geojson(&diamond).expect("valid polygon");
    let pm = Arc::new(pm);
    let crs = Arc::new(engine.get_or_build_crs(&cog).unwrap());

    let mut heavy_handles = Vec::with_capacity(2);
    for worker in 0..2usize {
        let eng = Arc::clone(&engine);
        let cog = Arc::clone(&cog);
        let prog = ndvi_prog.clone();
        let pm = Arc::clone(&pm);
        let crs = Arc::clone(&crs);
        heavy_handles.push(tokio::spawn(async move {
            let mut lats = Vec::new();
            let mut iter = worker;
            let cogs = [Arc::clone(&cog)];
            let sub_bbox = [-9.7, 35.3, -8.3, 36.7];
            let mut stats_opts = StatsOptions::default();
            stats_opts.percentiles = vec![2.0, 98.0];
            stats_opts.histogram_bins = 10;
            stats_opts.area_weighted = true;
            while Instant::now() < deadline {
                let t0 = Instant::now();
                match iter % 3 {
                    0 => {
                        let b = render_wgs84_bbox(
                            &eng,
                            &cogs,
                            sub_bbox,
                            2048,
                            2048,
                            None,
                            &BandRequest::Rgb {
                                bands_1based: [1, 2, 3],
                                rescale_min: [0.0, 0.0, 0.0],
                                rescale_max: [2500.0, 1500.0, 1000.0],
                            },
                            ResampleMode::Lanczos,
                            OutputFormat::Jpeg { quality: 85 },
                            None,
                            None,
                            None,
                            false,
                            false,
                        )
                        .await
                        .expect("bbox ok");
                        std::hint::black_box(b.len());
                    }
                    1 => {
                        let j = render_wgs84_bbox_stats(
                            &eng,
                            &cogs,
                            sub_bbox,
                            2048,
                            2048,
                            None,
                            &BandRequest::CustomExpr {
                                prog: prog.clone(),
                                rescale: (-1.0, 1.0),
                                colormap: ColormapName::Viridis,
                            },
                            ResampleMode::Lanczos,
                            None,
                            false,
                            "ndvi",
                            &stats_opts,
                        )
                        .await
                        .expect("bbox stats ok");
                        std::hint::black_box(j.is_object());
                    }
                    _ => {
                        let j = compute_native_exact_polygon_stats(
                            &eng,
                            &cog,
                            &crs,
                            &pm,
                            1024,
                            false,
                            None,
                            &stats_opts,
                        )
                        .await
                        .expect("zonal ok");
                        std::hint::black_box(j.is_object());
                    }
                }
                lats.push(t0.elapsed().as_secs_f64() * 1e3);
                iter += 1;
            }
            lats
        }));
    }

    let mut tile_lats = Vec::new();
    for h in tile_handles {
        tile_lats.extend(h.await.unwrap());
    }
    let mut heavy_lats = Vec::new();
    for h in heavy_handles {
        heavy_lats.extend(h.await.unwrap());
    }
    let elapsed = start.elapsed();
    (summarize(tile_lats, elapsed), summarize(heavy_lats, elapsed))
}

fn print_row(label: &str, cpu: &LatencyStats, gpu: &LatencyStats) {
    println!(
        "| {label} | {:.0} req/s (p50={:.1}, p95={:.1}, p99={:.1} ms) | {:.0} req/s (p50={:.1}, p95={:.1}, p99={:.1} ms) | **{:.1}×** |",
        cpu.rps,
        cpu.p50,
        cpu.p95,
        cpu.p99,
        gpu.rps,
        gpu.p50,
        gpu.p95,
        gpu.p99,
        gpu.rps / cpu.rps.max(1e-9),
    );
}

#[tokio::main]
async fn main() {
    let tmp = std::env::temp_dir().join("cogzilla_e2e_realistic_workload_bench.tif");
    std::fs::write(&tmp, build_3band_2048_cog_bytes()).expect("write 2048^2 3-band COG");

    let cpu_store = Arc::new(CogStore::new(512 * 1024 * 1024));
    let cpu_cog = cpu_store.register_local("s2", &tmp).expect("open cpu cog");
    let mut cpu_eng = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_eng.gpu = None;
    let cpu_engine = Arc::new(cpu_eng);

    let gpu_store = Arc::new(CogStore::new(512 * 1024 * 1024));
    let gpu_cog = gpu_store.register_local("s2", &tmp).expect("open gpu cog");
    let gpu_engine = Arc::new(TileEngine::with_store_and_tile_cache(gpu_store, 0));
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };

    let pool = Arc::new(build_tile_pool());
    let ndvi_prog = ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap();

    println!(
        "Realistic Sustained Map-Server Soak Benchmark: {} ({:?}, unified={}, floor={:.1} us, tile_pool={})\n",
        gpu.ctx.info.name,
        gpu.ctx.info.backend,
        gpu.ctx.info.unified_memory,
        gpu.policy.profile.dispatch_floor_us,
        pool.len()
    );

    // Warm up block cache + GPU slabs on both engines
    let _ = run_tile_soak(
        Arc::clone(&cpu_engine),
        Arc::clone(&cpu_cog),
        Arc::clone(&pool),
        ndvi_prog.clone(),
        8,
        Duration::from_secs(1),
        0,
        false,
    )
    .await;
    let _ = run_tile_soak(
        Arc::clone(&gpu_engine),
        Arc::clone(&gpu_cog),
        Arc::clone(&pool),
        ndvi_prog.clone(),
        8,
        Duration::from_secs(1),
        0,
        false,
    )
    .await;

    let c0 = gpu.coalescer.stats();
    let d0 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);

    println!("### 1–2. Sustained Mixed Map Traffic (8s each, zero tile cache)");
    println!("| Workload | CPU-Only | GPU Continuous Batching | Throughput Speedup |");
    println!("| :--- | :--- | :--- | ---: |");

    // Scenario 1: 16 clients staggered pan & zoom (0-6 ms jitter), 8s
    let cpu_s1 = run_tile_soak(
        Arc::clone(&cpu_engine),
        Arc::clone(&cpu_cog),
        Arc::clone(&pool),
        ndvi_prog.clone(),
        16,
        Duration::from_secs(8),
        6,
        false,
    )
    .await;
    let gpu_s1 = run_tile_soak(
        Arc::clone(&gpu_engine),
        Arc::clone(&gpu_cog),
        Arc::clone(&pool),
        ndvi_prog.clone(),
        16,
        Duration::from_secs(8),
        6,
        false,
    )
    .await;
    print_row("Pan & Zoom (16 clients, 0–6ms jitter, mixed)", &cpu_s1, &gpu_s1);

    // Scenario 2: 32 clients flood (0 ms think time), 8s
    let cpu_s2 = run_tile_soak(
        Arc::clone(&cpu_engine),
        Arc::clone(&cpu_cog),
        Arc::clone(&pool),
        ndvi_prog.clone(),
        32,
        Duration::from_secs(8),
        0,
        false,
    )
    .await;
    let gpu_s2 = run_tile_soak(
        Arc::clone(&gpu_engine),
        Arc::clone(&gpu_cog),
        Arc::clone(&pool),
        ndvi_prog.clone(),
        32,
        Duration::from_secs(8),
        0,
        false,
    )
    .await;
    print_row("Mixed Tile Flood (32 clients, 0ms think)", &cpu_s2, &gpu_s2);

    // Scenario 3: Live tiles + Heavy 2048² analytics contention, 8s
    println!("\n### 3. Live Tiles + Heavy Analytics Contention (12 live tile clients + 2 heavy 2048² clients, 8s)");
    println!("| Stream | CPU-Only | GPU Continuous Batching | Speedup |");
    println!("| :--- | :--- | :--- | ---: |");
    let (cpu_live, cpu_heavy) = run_mixed_contention_soak(
        Arc::clone(&cpu_engine),
        Arc::clone(&cpu_cog),
        Arc::clone(&pool),
        ndvi_prog.clone(),
        Duration::from_secs(8),
    )
    .await;
    let (gpu_live, gpu_heavy) = run_mixed_contention_soak(
        Arc::clone(&gpu_engine),
        Arc::clone(&gpu_cog),
        Arc::clone(&pool),
        ndvi_prog.clone(),
        Duration::from_secs(8),
    )
    .await;
    print_row("Live Tiles (12 clients, 2ms pacing)", &cpu_live, &gpu_live);
    print_row("Heavy Analytics (2048² bbox/stats + 1024² zonal)", &cpu_heavy, &gpu_heavy);

    // Scenario 4: Concurrency scaling sweep (1, 4, 16, 32 clients × 4s each on 512² RGB Lanczos3 JPEG)
    println!("\n### 4. Concurrency Scaling (`512×512` RGB Lanczos3 JPEG, 4s each)");
    println!("| Concurrency | CPU-Only | GPU Continuous Batching | Throughput Speedup |");
    println!("| :--- | :--- | :--- | ---: |");
    for &conc in &[1usize, 4, 16, 32] {
        let cpu_c = run_tile_soak(
            Arc::clone(&cpu_engine),
            Arc::clone(&cpu_cog),
            Arc::clone(&pool),
            ndvi_prog.clone(),
            conc,
            Duration::from_secs(4),
            0,
            true,
        )
        .await;
        let gpu_c = run_tile_soak(
            Arc::clone(&gpu_engine),
            Arc::clone(&gpu_cog),
            Arc::clone(&pool),
            ndvi_prog.clone(),
            conc,
            Duration::from_secs(4),
            0,
            true,
        )
        .await;
        print_row(&format!("{conc} client(s)"), &cpu_c, &gpu_c);
    }

    let c1 = gpu.coalescer.stats();
    let d1 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
    let batches = c1.batches_submitted.saturating_sub(c0.batches_submitted);
    let jobs = c1.jobs_executed.saturating_sub(c0.jobs_executed);
    let avg_batch = if batches > 0 {
        jobs as f64 / batches as f64
    } else {
        0.0
    };
    let s1 = gpu.slab_cache.stats();
    let ms1 = gpu.multi_slab_cache.stats();
    println!(
        "\n### GPU Coalescer & Cache Telemetry (across all 4 scenarios)\n- Total GPU dispatches: {}\n- Coalesced jobs: {} across {} batches (avg batch = {:.2}, max batch = {})\n- Single-band slab cache: hits={}, misses={} ({:.1}% hit rate)\n- Multi-band slab cache: hits={}, misses={} ({:.1}% hit rate)",
        d1 - d0,
        jobs,
        batches,
        avg_batch,
        c1.max_batch_seen,
        s1.hits,
        s1.misses,
        100.0 * s1.hits as f64 / (s1.hits + s1.misses).max(1) as f64,
        ms1.hits,
        ms1.misses,
        100.0 * ms1.hits as f64 / (ms1.hits + ms1.misses).max(1) as f64,
    );

    let _ = std::fs::remove_file(&tmp);
}
