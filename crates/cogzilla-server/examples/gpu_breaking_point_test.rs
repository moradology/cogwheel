//! Breaking-point failure & limit discovery test for `cogzilla-gpu`.
//!
//! Deliberately targets the hardest production failure modes:
//!   1. **Coalesced staging-buffer explosion**: 16 concurrent 4096×4096 PNG/RawRgba bboxes
//!      (16 × 64 MiB = 1 GiB in a single coalesced frame > 512 MiB Metal / Vulkan limit!).
//!   2. **Single-request oversized output**: 8192×8192 and 12288×12288 bbox renders (256–576 MiB).
//!   3. **Pathological expressions**: deep stack (20+ nested ops), div-by-zero, Inf/NaN overflow,
//!      `log(-1)`, `sqrt(-1)`, `pow(-2, 0.5)`, `where(NaN, 1, 2)`.
//!   4. **2,048-client concurrency wall** (exceeds `max_in_flight=32 * 32 = 1024`, forcing CPU
//!      fallback shedding — measures whether Tokio async workers starve!).
//!   5. **Multi-GB VRAM / Slab-cache pressure**: 40 distinct 2048² 3-band COGs (~1.9 GiB working
//!      set) thrashed by 64 concurrent clients — monitors whether VRAM stays bounded or OOMs.
//!
//! Each probe is isolated with `catch_unwind` / `JoinHandle` so we see EVERY failure, not just
//! the first panic.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, ExprProgram, OutputFormat, ResampleMode};
use cogzilla_proj::TileCoord;
use cogzilla_server::engine::{BandRequest, TileEngine, TileRequest};
use cogzilla_server::titiler::render_wgs84_bbox;

fn build_cog_bytes(w: u32, h: u32, seed: f32) -> Vec<u8> {
    let tw: u32 = 512.min(w);
    let th: u32 = 512.min(h);
    let tiles_x = w / tw;
    let tiles_y = h / th;
    let total_tiles = (tiles_x * tiles_y * 3) as usize;
    let tile_bytes = (tw * th * 4) as usize;

    let mut tile_data = Vec::with_capacity(total_tiles);
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
                            0 => 1000.0 + 800.0 * (fx * 12.0 + seed).sin(),
                            1 => 600.0 + 400.0 * (fy * 9.0 + seed).cos(),
                            _ => 400.0 + 250.0 * ((fx + fy) * 7.0 + seed).sin(),
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
    scale_bytes.extend_from_slice(&(2.0f64 / w as f64).to_le_bytes());
    scale_bytes.extend_from_slice(&(2.0f64 / h as f64).to_le_bytes());
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

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[tokio::main]
async fn main() {
    let tmp_dir = std::env::temp_dir().join("cogzilla_gpu_breaking_point");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let main_tif = tmp_dir.join("main_2048.tif");
    std::fs::write(&main_tif, build_cog_bytes(2048, 2048, 0.0)).unwrap();

    let store = Arc::new(CogStore::new(512 * 1024 * 1024));
    let main_cog = store.register_local("main", &main_tif).unwrap();
    let engine = Arc::new(TileEngine::with_store_and_tile_cache(Arc::clone(&store), 0));
    let Some(ref gpu) = engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };
    let limits = gpu.ctx.device.limits();
    println!(
        "BREAKING-POINT TEST on {} ({:?}, unified={}, floor={:.1} us, max_buf={} MiB, max_storage_binding={} MiB)\n",
        gpu.ctx.info.name,
        gpu.ctx.info.backend,
        gpu.ctx.info.unified_memory,
        gpu.policy.profile.dispatch_floor_us,
        limits.max_buffer_size / (1024 * 1024),
        limits.max_storage_buffer_binding_size / (1024 * 1024),
    );

    let sub_bbox = [-9.7, 35.3, -8.3, 36.7];
    let pool = Arc::new(build_tile_pool());

    // =========================================================================
    // PROBE 1: Pathological expressions (div-by-zero, Inf, NaN, log(-1), deep nesting)
    // =========================================================================
    print!("[1/5] Pathological expressions (div/0, Inf, log(-1), sqrt(-1), deep nesting)... ");
    {
        let mut cpu_eng = TileEngine::with_store_and_tile_cache(Arc::clone(&store), 0);
        cpu_eng.gpu = None;
        let exprs = [
            "(b1 - b1) / (b2 - b2)",                     // 0/0 -> NaN everywhere
            "1.0 / (b1 - b1)",                           // 1/0 -> NaN/Inf
            "log(-abs(b1) - 1.0)",                       // log(negative) -> NaN
            "sqrt(-abs(b1) - 1.0)",                      // sqrt(negative) -> NaN
            "where((b1 - b1)/(b1 - b1), b2, b3)",        // where(NaN, ...)
            "(b1 * 1e38) * 1e38",                        // Inf overflow
            "(((((((((b1+b2)+b3)+b1)+b2)+b3)+b1)+b2)+b3)+b1)", // 10-deep nesting
        ];
        let tile = pool[pool.len() / 2];
        for expr_str in exprs {
            let prog = ExprProgram::compile(expr_str).expect("compile expr");
            let req = TileRequest {
                tile,
                tile_size: 512,
                bands: BandRequest::CustomExpr {
                    prog,
                    rescale: (-10.0, 10.0),
                    colormap: ColormapName::Viridis,
                },
                resample: ResampleMode::Lanczos,
                format: OutputFormat::RawRgba,
                unscale: false,
            };
            let cpu = cpu_eng.render_tile(&main_cog, &req).await.unwrap();
            let gpu_out = engine.render_tile(&main_cog, &req).await.unwrap();
            assert_eq!(cpu.len(), gpu_out.len(), "len mismatch for {expr_str}");
            let mut mask_diff = 0usize;
            for (c, g) in cpu.chunks_exact(4).zip(gpu_out.chunks_exact(4)) {
                if (c[3] == 0) != (g[3] == 0) {
                    mask_diff += 1;
                }
            }
            if mask_diff > 0 {
                println!(
                    "\n  [DIVERGENCE] expr `{expr_str}`: {mask_diff}/{} alpha-mask diffs vs CPU!",
                    cpu.len() / 4
                );
            }
        }
    }
    println!("DONE!");

    // =========================================================================
    // PROBE 2: 2,048-client concurrency wall (exceeds max_in_flight=32 * 32 = 1024!)
    // =========================================================================
    print!("[2/5] 2,048-client concurrency wall (exceeds 1,024 max_in_flight, 5s)... ");
    {
        let deadline = Instant::now() + Duration::from_secs(5);
        let start = Instant::now();
        let mut hs = Vec::with_capacity(2048);
        for w in 0..2048usize {
            let eng = Arc::clone(&engine);
            let cog = Arc::clone(&main_cog);
            let pool = Arc::clone(&pool);
            hs.push(tokio::spawn(async move {
                let mut lats = Vec::new();
                let mut i = w;
                while Instant::now() < deadline {
                    let req = TileRequest {
                        tile: pool[(i * 19) % pool.len()],
                        tile_size: 256,
                        bands: BandRequest::Rgb {
                            bands_1based: [1, 2, 3],
                            rescale_min: [0.0; 3],
                            rescale_max: [2000.0; 3],
                        },
                        resample: ResampleMode::Lanczos,
                        format: OutputFormat::Jpeg { quality: 85 },
                        unscale: false,
                    };
                    let t0 = Instant::now();
                    eng.render_tile(&cog, &req).await.expect("2048-wall tile ok");
                    lats.push(t0.elapsed().as_secs_f64() * 1e3);
                    i += 1;
                }
                lats
            }));
        }
        let mut all = Vec::new();
        for h in hs {
            all.extend(h.await.unwrap());
        }
        let elapsed = start.elapsed().as_secs_f64();
        all.sort_by(|a, b| a.total_cmp(b));
        println!(
            "{:.0} req/s (n={}, p50={:.1}ms, p95={:.1}ms, p99={:.1}ms, max={:.1}ms)",
            all.len() as f64 / elapsed,
            all.len(),
            pct(&all, 50.0),
            pct(&all, 95.0),
            pct(&all, 99.0),
            all.last().copied().unwrap_or(0.0),
        );
    }

    // =========================================================================
    // PROBE 3: Multi-GB Slab Pressure (24 distinct 2048² 3-band COGs = 1.15 GiB)
    // =========================================================================
    print!("[3/5] Multi-GB slab pressure (24 distinct 2048² 3-band COGs, 32 tasks, 5s)... ");
    {
        let mut cogs = Vec::new();
        for c in 0..24usize {
            let p = tmp_dir.join(format!("big_{c}.tif"));
            std::fs::write(&p, build_cog_bytes(2048, 2048, c as f32 * 0.2)).unwrap();
            cogs.push(store.register_local(&format!("big_{c}"), &p).unwrap());
        }
        let cogs = Arc::new(cogs);
        let deadline = Instant::now() + Duration::from_secs(5);
        let ok = Arc::new(AtomicUsize::new(0));
        let mut hs = Vec::new();
        for w in 0..32usize {
            let eng = Arc::clone(&engine);
            let cogs = Arc::clone(&cogs);
            let pool = Arc::clone(&pool);
            let ok = Arc::clone(&ok);
            hs.push(tokio::spawn(async move {
                let mut i = w;
                while Instant::now() < deadline {
                    let cog = &cogs[i % cogs.len()];
                    let req = TileRequest {
                        tile: pool[(i * 11) % pool.len()],
                        tile_size: 512,
                        bands: BandRequest::Rgb {
                            bands_1based: [1, 2, 3],
                            rescale_min: [0.0; 3],
                            rescale_max: [2000.0; 3],
                        },
                        resample: ResampleMode::Lanczos,
                        format: OutputFormat::Jpeg { quality: 85 },
                        unscale: false,
                    };
                    eng.render_tile(cog, &req).await.expect("multi-gb tile ok");
                    ok.fetch_add(1, Ordering::Relaxed);
                    i += 1;
                }
            }));
        }
        for h in hs {
            h.await.unwrap();
        }
        let ms = gpu.multi_slab_cache.stats();
        println!(
            "{} tiles (multi_slab hits={}, misses={}, evictions={})",
            ok.load(Ordering::Relaxed),
            ms.hits,
            ms.misses,
            ms.evictions
        );
    }

    // =========================================================================
    // PROBE 4: Coalesced Staging-Buffer Explosion (16 concurrent 4096² RawRgba = 1 GiB!)
    // =========================================================================
    print!("[4/5] Coalesced staging explosion (16 concurrent 4096² RawRgba = 1 GiB batch!)... ");
    {
        let mut hs = Vec::with_capacity(16);
        let t0 = Instant::now();
        for _ in 0..16usize {
            let eng = Arc::clone(&engine);
            let cog = Arc::clone(&main_cog);
            hs.push(tokio::spawn(async move {
                let cogs = [cog];
                let b = render_wgs84_bbox(
                    &eng,
                    &cogs,
                    sub_bbox,
                    4096,
                    4096,
                    None,
                    &BandRequest::Rgb {
                        bands_1based: [1, 2, 3],
                        rescale_min: [0.0; 3],
                        rescale_max: [2000.0; 3],
                    },
                    ResampleMode::Lanczos,
                    OutputFormat::RawRgba,
                    None,
                    None,
                    None,
                    true,
                    false,
                )
                .await?;
                Ok::<usize, cogzilla_server::engine::EngineError>(b.len())
            }));
        }
        let mut ok_n = 0usize;
        let mut err_n = 0usize;
        for h in hs {
            match h.await {
                Ok(Ok(len)) => {
                    assert_eq!(len, 4096 * 4096 * 4);
                    ok_n += 1;
                }
                Ok(Err(e)) => {
                    eprintln!("\n  [ERR] 4096² RawRgba: {e}");
                    err_n += 1;
                }
                Err(e) => {
                    eprintln!("\n  [PANIC] 4096² RawRgba: {e}");
                    err_n += 1;
                }
            }
        }
        println!(
            "ok={ok_n}/16, err={err_n}, elapsed={:.1}ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }

    // =========================================================================
    // PROBE 5: Single-Request Giant Output (8192×8192 & 12288×12288 RawRgba / JPEG)
    // =========================================================================
    for &(dim, fmt_name, fmt) in &[
        (8192u32, "JPEG", OutputFormat::Jpeg { quality: 85 }),
        (8192u32, "RawRgba", OutputFormat::RawRgba),
        (12288u32, "RawRgba (576 MiB > 512 MiB!)", OutputFormat::RawRgba),
    ] {
        print!("[5/5] Giant single bbox {dim}×{dim} {fmt_name}... ");
        let eng = Arc::clone(&engine);
        let cog = Arc::clone(&main_cog);
        let t0 = Instant::now();
        let h = tokio::spawn(async move {
            let cogs = [cog];
            render_wgs84_bbox(
                &eng,
                &cogs,
                sub_bbox,
                dim,
                dim,
                None,
                &BandRequest::Rgb {
                    bands_1based: [1, 2, 3],
                    rescale_min: [0.0; 3],
                    rescale_max: [2000.0; 3],
                },
                ResampleMode::Bilinear,
                fmt,
                None,
                None,
                None,
                true,
                false,
            )
            .await
        });
        match h.await {
            Ok(Ok(b)) => println!(
                "OK ({:.1} MiB in {:.1}ms)",
                b.len() as f64 / (1024.0 * 1024.0),
                t0.elapsed().as_secs_f64() * 1e3
            ),
            Ok(Err(e)) => println!("ERR: {e}"),
            Err(e) => println!("PANICKED: {e}"),
        }
    }

    // Verify coalescer is STILL ALIVE after Probes 1–5!
    print!("\n[HEALTH CHECK] Coalescer still alive after Probes 1–5?... ");
    let d_before = gpu.coalescer.stats().jobs_executed;
    let req = TileRequest {
        tile: pool[0],
        tile_size: 512,
        bands: BandRequest::Rgb {
            bands_1based: [1, 2, 3],
            rescale_min: [0.0; 3],
            rescale_max: [2000.0; 3],
        },
        resample: ResampleMode::Lanczos,
        format: OutputFormat::Jpeg { quality: 85 },
        unscale: false,
    };
    let _ = engine.render_tile(&main_cog, &req).await.unwrap();
    let d_after = gpu.coalescer.stats().jobs_executed;
    if d_after > d_before {
        println!("YES (coalescer alive, total jobs={d_after})");
    } else {
        println!("DEAD!! (coalescer worker died and fell back to CPU! jobs stuck at {d_after})");
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
