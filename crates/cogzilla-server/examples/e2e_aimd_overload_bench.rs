//! End-to-end live HTTP overload benchmark comparing CPU-only vs GPU servers with AIMD
//! concurrency limiting OFF vs ON.
//!
//! Spins up real `cogzilla_server::http::serve_listener` servers on `127.0.0.1:0` backed by a
//! 2048×2048 3-band F32 tiled GeoTIFF (zero tile cache so every HTTP request exercises the full
//! render + encode pipeline across thousands of distinct tiles), then drives sustained concurrent
//! HTTP traffic (`reqwest`) while probing `/healthz` every 100 ms and scraping `/metrics`.
//!
//! Run (GPU + CPU-only comparison):
//!   cargo run --release -p cogzilla-server --example e2e_aimd_overload_bench
//!
//! Run (CPU-only build without GPU feature):
//!   cargo run --release -p cogzilla-server --no-default-features --example e2e_aimd_overload_bench

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cogzilla_io::CogStore;
use cogzilla_proj::TileCoord;
use cogzilla_server::aimd::AimdLimiter;
use cogzilla_server::engine::TileEngine;
use tokio::net::TcpListener;

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

/// Build >2,000 distinct overlapping tiles across z=11..=14 so singleflight coalescing does not
/// collapse concurrent HTTP clients onto the same tile key.
fn build_tile_pool() -> Vec<TileCoord> {
    let mut pool = Vec::new();
    for z in 11u8..=14u8 {
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

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

async fn run_http_storm(
    label: &str,
    engine: Arc<TileEngine>,
    tiles: &[TileCoord],
    concurrency: usize,
    duration: Duration,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");

    let server_engine = engine.clone();
    let server_handle = tokio::spawn(async move {
        let _ = cogzilla_server::http::serve_listener(server_engine, listener).await;
    });

    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(concurrency + 32)
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap();

    // Warmup 1 request to verify 200 OK + real JPEG bytes and warm COG block cache / GPU slab
    let t0 = tiles[0];
    let warm_url = format!(
        "{base}/tiles/bench/{}/{}/{}@2x.jpg?bidx=1&bidx=2&bidx=3&rescale=0,2500&resampling=lanczos&reproject=lanczos",
        t0.z, t0.x, t0.y
    );
    let warm_resp = client.get(&warm_url).send().await.unwrap();
    assert_eq!(
        warm_resp.status(),
        reqwest::StatusCode::OK,
        "warmup failed for {label}"
    );
    let warm_bytes = warm_resp.bytes().await.unwrap();
    assert!(
        warm_bytes.len() > 2000 && &warm_bytes[0..2] == b"\xff\xd8",
        "expected non-empty JPEG tile, got {} bytes",
        warm_bytes.len()
    );

    let stop = Arc::new(AtomicBool::new(false));
    let healthz_ok = Arc::new(AtomicU64::new(0));
    let healthz_fail = Arc::new(AtomicU64::new(0));

    // Concurrent /healthz prober every 100ms
    let probe_stop = stop.clone();
    let probe_client = client.clone();
    let probe_url = format!("{base}/healthz");
    let h_ok = healthz_ok.clone();
    let h_fail = healthz_fail.clone();
    let healthz_task = tokio::spawn(async move {
        let mut lats = Vec::new();
        while !probe_stop.load(Ordering::Relaxed) {
            let s = Instant::now();
            match probe_client.get(&probe_url).send().await {
                Ok(r) if r.status() == reqwest::StatusCode::OK => {
                    let _ = r.bytes().await;
                    h_ok.fetch_add(1, Ordering::Relaxed);
                    lats.push(s.elapsed().as_secs_f64() * 1000.0);
                }
                _ => {
                    h_fail.fetch_add(1, Ordering::Relaxed);
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        lats
    });

    let start = Instant::now();
    let tiles = Arc::new(tiles.to_vec());
    let stride = (tiles.len() / concurrency.max(1)).max(1);
    let mut workers = Vec::with_capacity(concurrency);
    for wid in 0..concurrency {
        // Gentle ramp-up (32 connections/batch, 5ms pause) so macOS somaxconn=128 never drops SYNs
        if wid > 0 && wid % 32 == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let c = client.clone();
        let s = stop.clone();
        let b = base.clone();
        let tpool = tiles.clone();
        workers.push(tokio::spawn(async move {
            let mut ok_lats = Vec::new();
            let mut shed_lats = Vec::new();
            let mut errs = 0u64;
            let mut idx = wid * stride;
            while !s.load(Ordering::Relaxed) {
                let tile = tpool[idx % tpool.len()];
                idx = idx.wrapping_add(1);
                let url = format!(
                    "{b}/tiles/bench/{}/{}/{}@2x.jpg?bidx=1&bidx=2&bidx=3&rescale=0,2500&resampling=lanczos&reproject=lanczos",
                    tile.z, tile.x, tile.y
                );
                let req_start = Instant::now();
                match c.get(&url).send().await {
                    Ok(resp) => {
                        let status = resp.status();
                        let _ = resp.bytes().await;
                        let ms = req_start.elapsed().as_secs_f64() * 1000.0;
                        if status == reqwest::StatusCode::OK {
                            ok_lats.push(ms);
                        } else if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                            shed_lats.push(ms);
                            tokio::time::sleep(Duration::from_millis(15)).await;
                        } else {
                            errs += 1;
                        }
                    }
                    Err(_) => {
                        errs += 1;
                    }
                }
            }
            (ok_lats, shed_lats, errs)
        }));
    }

    tokio::time::sleep(duration).await;
    stop.store(true, Ordering::Relaxed);

    let mut all_ok = Vec::new();
    let mut all_shed = Vec::new();
    let mut other_err = 0u64;
    for w in workers {
        let (ok, shed, err) = w.await.unwrap();
        all_ok.extend(ok);
        all_shed.extend(shed);
        other_err += err;
    }
    let elapsed = start.elapsed();
    let mut h_lats = healthz_task.await.unwrap();
    h_lats.sort_by(|a, b| a.total_cmp(b));

    let metrics_text = client
        .get(format!("{base}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics_text.contains("cogzilla_aimd_limit "));
    assert!(metrics_text.contains("cogzilla_aimd_rejected_429_total "));

    let snap = engine.aimd.snapshot();
    let gpu_dispatches = engine.stats.gpu_dispatches.load(Ordering::Relaxed);
    let coalesced = engine.tile_cache.coalesced.load(Ordering::Relaxed);
    server_handle.abort();

    all_ok.sort_by(|a, b| a.total_cmp(b));
    all_shed.sort_by(|a, b| a.total_cmp(b));

    println!(
        "  {label:<32} | 200={:>5} ({:>6.1} r/s) | 429={:>5} (p50 {:>4.1}ms) | err={:>2} | 200 p50={:>6.1}ms p95={:>6.1}ms p99={:>6.1}ms | hz={}/{} (p99 {:>5.1}ms) | lim={:>3} (gpu={gpu_dispatches} coal={coalesced})",
        all_ok.len(),
        all_ok.len() as f64 / elapsed.as_secs_f64(),
        all_shed.len(),
        pct(&all_shed, 50.0),
        other_err,
        pct(&all_ok, 50.0),
        pct(&all_ok, 95.0),
        pct(&all_ok, 99.0),
        healthz_ok.load(Ordering::Relaxed),
        healthz_ok.load(Ordering::Relaxed) + healthz_fail.load(Ordering::Relaxed),
        pct(&h_lats, 99.0),
        snap.limit,
    );
}

fn make_engine(cog_path: &std::path::Path, use_gpu: bool, aimd: AimdLimiter) -> Arc<TileEngine> {
    let store = Arc::new(CogStore::new(512 * 1024 * 1024));
    #[allow(unused_mut)]
    let mut engine = TileEngine::with_store_and_tile_cache(store, 0).with_aimd(aimd);
    #[cfg(feature = "gpu")]
    if !use_gpu {
        engine.gpu = None;
    }
    #[cfg(not(feature = "gpu"))]
    let _ = use_gpu;
    engine.store.register_local("bench", cog_path).unwrap();
    Arc::new(engine)
}

#[tokio::main]
async fn main() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_aimd_bench_{}.tif", std::process::id()));
    std::fs::write(&tmp, build_3band_2048_cog_bytes()).unwrap();
    let tiles = build_tile_pool();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let dur = Duration::from_secs(6);

    println!(
        "=== Live HTTP Overload Benchmark (512x512 RGB Lanczos3 JPEG, {} unique tiles, {cores} CPU cores, {}s/case) ===",
        tiles.len(),
        dur.as_secs()
    );

    // 1. CPU-only overload (128 concurrent HTTP clients, target=100ms)
    println!("\n--- CPU-Only Server (gpu=None, 128 concurrent HTTP clients, target=100ms) ---");
    let cpu_off = make_engine(&tmp, false, AimdLimiter::new(false, 64, 4, 4096, 100));
    run_http_storm("CPU-only | AIMD OFF (128 cli)", cpu_off, &tiles, 128, dur).await;

    let cpu_on = make_engine(
        &tmp,
        false,
        AimdLimiter::new(true, (cores * 4).clamp(16, 128), cores.clamp(4, 32), 4096, 100),
    );
    run_http_storm("CPU-only | AIMD ON  (128 cli)", cpu_on, &tiles, 128, dur).await;

    // 2. GPU overload (256 concurrent HTTP clients, target=100ms)
    #[cfg(feature = "gpu")]
    {
        let probe = TileEngine::with_store_and_tile_cache(Arc::new(CogStore::new(64 * 1024 * 1024)), 0);
        if probe.gpu.is_some() {
            drop(probe);
            println!("\n--- GPU Server (256 concurrent HTTP clients, target=100ms) ---");
            let gpu_off = make_engine(&tmp, true, AimdLimiter::new(false, 256, 32, 4096, 100));
            run_http_storm("GPU      | AIMD OFF (256 cli)", gpu_off, &tiles, 256, dur).await;

            let gpu_on = make_engine(&tmp, true, AimdLimiter::new(true, 128, 32, 4096, 100));
            run_http_storm("GPU      | AIMD ON  (256 cli)", gpu_on, &tiles, 256, dur).await;
        } else {
            println!("\n(No GPU adapter available; skipping GPU HTTP overload section)");
        }
    }

    let _ = std::fs::remove_file(&tmp);
}
