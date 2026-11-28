//! Adversarial GPU torture & chaos test — tries to make `cogzilla-gpu` fall over via:
//!   1. Edge/degenerate geometry & values (1x1, 1x4096, 4096x1, 333x777, [0,0] rescale, all-NaN,
//!      oversized >512 MiB RGB mosaic stack).
//!   2. Cancellation / mid-flight drop chaos (64 tasks with 1–5 ms timeouts dropping futures mid-GPU).
//!   3. Slab-cache thrashing (16 distinct COGs vs tight 32 MiB slab cache, 64 concurrent clients).
//!   4. Extreme concurrency flood (256 & 512 concurrent clients hammering `GpuCoalescer`).
//!   5. Full cross-subsystem chaos (Coalescer + 4096² bboxes + WeightedZonal + Mosaic RGB JPEG +
//!      standalone GPU JPEG all at once from 96 tasks) — hunts for deadlocks, VRAM blowup, or panics.
//!
//! Run:
//!   cargo run --release -p cogzilla-server --example gpu_torture_test

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cogzilla_gpu::GpuPixelSelection;
use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, DemAlgorithm, ExprProgram, OutputFormat, ResampleMode};
use cogzilla_proj::TileCoord;
use cogzilla_server::engine::{BandRequest, TileEngine, TileRequest};
use cogzilla_server::stats::{
    PolygonMask, StatsOptions, render_wgs84_bbox, render_wgs84_bbox_stats,
};
use cogzilla_server::zonal::compute_native_exact_polygon_stats;

fn build_cog_bytes(w: u32, h: u32, all_nan: bool, seed: f32) -> Vec<u8> {
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
                        let v = if all_nan {
                            f32::NAN
                        } else {
                            let fx = gx as f32 / w as f32;
                            let fy = gy as f32 / h as f32;
                            match band {
                                0 => 1000.0 + 800.0 * (fx * 12.0 + seed).sin(),
                                1 => 600.0 + 400.0 * (fy * 9.0 + seed).cos(),
                                _ => 400.0 + 250.0 * ((fx + fy) * 7.0 + seed).sin(),
                            }
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
    let tmp_dir = std::env::temp_dir().join("cogzilla_gpu_torture");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let main_tif = tmp_dir.join("main_2048.tif");
    let nan_tif = tmp_dir.join("all_nan_1024.tif");
    std::fs::write(&main_tif, build_cog_bytes(2048, 2048, false, 0.0)).unwrap();
    std::fs::write(&nan_tif, build_cog_bytes(1024, 1024, true, 0.0)).unwrap();

    let store = Arc::new(CogStore::new(512 * 1024 * 1024));
    let main_cog = store.register_local("main", &main_tif).unwrap();
    let nan_cog = store.register_local("nan", &nan_tif).unwrap();
    let engine = Arc::new(TileEngine::with_store_and_tile_cache(Arc::clone(&store), 0));
    let Some(ref gpu) = engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };
    println!(
        "GPU TORTURE TEST on {} ({:?}, unified={}, floor={:.1} us)\n",
        gpu.ctx.info.name,
        gpu.ctx.info.backend,
        gpu.ctx.info.unified_memory,
        gpu.policy.profile.dispatch_floor_us
    );

    let pool = Arc::new(build_tile_pool());
    let ndvi_prog = ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap();
    let sub_bbox = [-9.7, 35.3, -8.3, 36.7];

    // =========================================================================
    // TEST 1: Edge / Degenerate Geometry, All-NaN, Zero Rescale, Oversized Stack
    // =========================================================================
    print!("[1/5] Edge / degenerate geometry, all-NaN, [0,0] rescale, oversized 768 MiB mosaic... ");
    {
        let main_cogs = [Arc::clone(&main_cog)];
        let nan_cogs = [Arc::clone(&nan_cog)];
        let opts = StatsOptions {
            percentiles: vec![2.0, 98.0],
            histogram_bins: 10,
            ..StatsOptions::default()
        };

        for &(ew, eh) in &[(1u32, 1u32), (1, 4096), (4096, 1), (333, 777), (17, 19)] {
            let b = render_wgs84_bbox(
                &engine,
                &main_cogs,
                sub_bbox,
                ew,
                eh,
                None,
                &BandRequest::Rgb {
                    bands_1based: [1, 2, 3],
                    rescale_min: [0.0, 0.0, 0.0],
                    rescale_max: [0.0, 0.0, 0.0], // degenerate zero range!
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
            .expect("degenerate bbox JPEG ok");
            assert!(b.starts_with(&[0xff, 0xd8]), "valid JFIF for {ew}x{eh}");

            let j = render_wgs84_bbox_stats(
                &engine,
                &nan_cogs,
                sub_bbox,
                ew,
                eh,
                None,
                &BandRequest::Single {
                    band_1based: 1,
                    rescale: (0.0, 1.0),
                    colormap: ColormapName::Viridis,
                },
                ResampleMode::Lanczos,
                None,
                false,
                "all_nan",
                &opts,
            )
            .await
            .expect("all-NaN bbox stats ok");
            assert!(j.is_object());
        }

        // Oversized 2048x2048 x 16 scenes x RGB = 768 MiB (> 512 MiB wgpu max_buffer_size)
        // Must cleanly return None (CPU fallback) without panicking!
        let big_plane = vec![100.0f32; 2048 * 2048];
        let refs: Vec<&[f32]> = (0..16).map(|_| big_plane.as_slice()).collect();
        let rgb_planes: [&[&[f32]]; 3] = [&refs, &refs, &refs];
        let oversized = gpu.try_composite_mosaic(&cogzilla_server::gpu_backend::MosaicCompositeRequest {
            width: 2048,
            height: 2048,
            input: cogzilla_server::gpu_backend::MosaicCompositeInput::Rgb(&rgb_planes),
            method: GpuPixelSelection::Median,
            output: cogzilla_server::gpu_backend::MosaicCompositeOutput::RgbStyled {
                rescale_min: [0.0; 3],
                rescale_max: [255.0; 3],
                format: OutputFormat::Jpeg { quality: 85 },
                return_mask: false,
            },
        });
        assert!(
            oversized.is_none(),
            "768 MiB stack should cleanly fall back to None instead of panicking"
        );
    }
    println!("SURVIVED!");

    // =========================================================================
    // TEST 2: Mid-Flight Cancellation / Dropped Future Chaos (64 tasks, 4s)
    // =========================================================================
    print!("[2/5] Mid-flight cancellation / timeout drop chaos (64 tasks, 4s)... ");
    {
        let deadline = Instant::now() + Duration::from_secs(4);
        let ok_cnt = Arc::new(AtomicUsize::new(0));
        let drop_cnt = Arc::new(AtomicUsize::new(0));
        let mut hs = Vec::new();
        for w in 0..64usize {
            let eng = Arc::clone(&engine);
            let cog = Arc::clone(&main_cog);
            let pool = Arc::clone(&pool);
            let ok_cnt = Arc::clone(&ok_cnt);
            let drop_cnt = Arc::clone(&drop_cnt);
            hs.push(tokio::spawn(async move {
                let mut i = w;
                while Instant::now() < deadline {
                    let tile = pool[i % pool.len()];
                    let req = TileRequest {
                        tile,
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
                    // Half the workers cancel aggressively at 1-3ms, half run to completion
                    if w % 2 == 0 {
                        match tokio::time::timeout(
                            Duration::from_millis(1 + (i as u64 % 3)),
                            eng.render_tile(&cog, &req),
                        )
                        .await
                        {
                            Ok(Ok(_)) => {
                                ok_cnt.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(Err(e)) => panic!("unexpected render error under cancel: {e}"),
                            Err(_) => {
                                drop_cnt.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    } else {
                        eng.render_tile(&cog, &req).await.expect("survivor tile ok");
                        ok_cnt.fetch_add(1, Ordering::Relaxed);
                    }
                    i += 1;
                }
            }));
        }
        for h in hs {
            h.await.unwrap();
        }
        println!(
            "SURVIVED! (completed={}, dropped_mid_flight={})",
            ok_cnt.load(Ordering::Relaxed),
            drop_cnt.load(Ordering::Relaxed)
        );
    }

    // =========================================================================
    // TEST 3: Slab-Cache Thrashing (12 distinct 1024² COGs, 64 concurrent tasks, 5s)
    // =========================================================================
    print!("[3/5] Slab-cache eviction thrashing (12 distinct COGs, 64 tasks, 5s)... ");
    {
        let mut cogs = Vec::new();
        for c in 0..12usize {
            let p = tmp_dir.join(format!("thrash_{c}.tif"));
            std::fs::write(&p, build_cog_bytes(1024, 1024, false, c as f32 * 0.3)).unwrap();
            cogs.push(store.register_local(format!("thrash_{c}"), &p).unwrap());
        }
        let cogs = Arc::new(cogs);
        let deadline = Instant::now() + Duration::from_secs(5);
        let total = Arc::new(AtomicUsize::new(0));
        let mut hs = Vec::new();
        for w in 0..64usize {
            let eng = Arc::clone(&engine);
            let cogs = Arc::clone(&cogs);
            let pool = Arc::clone(&pool);
            let total = Arc::clone(&total);
            hs.push(tokio::spawn(async move {
                let mut i = w;
                while Instant::now() < deadline {
                    let cog = &cogs[i % cogs.len()];
                    let tile = pool[(i * 7) % pool.len()];
                    let req = TileRequest {
                        tile,
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
                    eng.render_tile(cog, &req).await.expect("thrash render ok");
                    total.fetch_add(1, Ordering::Relaxed);
                    i += 1;
                }
            }));
        }
        for h in hs {
            h.await.unwrap();
        }
        let ms = gpu.multi_slab_cache.stats();
        println!(
            "SURVIVED! ({} tiles, multi_slab hits={}, misses={}, evictions={})",
            total.load(Ordering::Relaxed),
            ms.hits,
            ms.misses,
            ms.evictions
        );
    }

    // =========================================================================
    // TEST 4: Extreme Concurrency Flood (256 & 512 concurrent clients, 5s each)
    // =========================================================================
    for &conc in &[256usize, 512usize] {
        print!("[4/5] Extreme concurrency flood ({conc} clients, 5s)... ");
        let deadline = Instant::now() + Duration::from_secs(5);
        let start = Instant::now();
        let mut hs = Vec::with_capacity(conc);
        for w in 0..conc {
            let eng = Arc::clone(&engine);
            let cog = Arc::clone(&main_cog);
            let pool = Arc::clone(&pool);
            let prog = ndvi_prog.clone();
            hs.push(tokio::spawn(async move {
                let mut lats = Vec::new();
                let mut i = w;
                while Instant::now() < deadline {
                    let tile = pool[(i * 13) % pool.len()];
                    let req = match i % 3 {
                        0 => TileRequest {
                            tile,
                            tile_size: 512,
                            bands: BandRequest::Rgb {
                                bands_1based: [1, 2, 3],
                                rescale_min: [0.0; 3],
                                rescale_max: [2000.0; 3],
                            },
                            resample: ResampleMode::Lanczos,
                            format: OutputFormat::Jpeg { quality: 85 },
                            unscale: false,
                        },
                        1 => TileRequest {
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
                                math: None,
                            },
                            resample: ResampleMode::Lanczos,
                            format: OutputFormat::Png,
                            unscale: false,
                        },
                        _ => TileRequest {
                            tile,
                            tile_size: 512,
                            bands: BandRequest::CustomExpr {
                                prog: prog.clone(),
                                rescale: (-1.0, 1.0),
                                colormap: ColormapName::Viridis,
                            },
                            resample: ResampleMode::Cubic,
                            format: OutputFormat::Png,
                            unscale: false,
                        },
                    };
                    let t0 = Instant::now();
                    eng.render_tile(&cog, &req).await.expect("flood tile ok");
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
            "SURVIVED! {:.0} req/s (n={}, p50={:.1}ms, p95={:.1}ms, p99={:.1}ms, max={:.1}ms)",
            all.len() as f64 / elapsed,
            all.len(),
            pct(&all, 50.0),
            pct(&all, 95.0),
            pct(&all, 99.0),
            all.last().copied().unwrap_or(0.0),
        );
    }

    // =========================================================================
    // TEST 5: Full Cross-Subsystem Chaos (Coalescer + 4096² Bbox + Zonal + Mosaic RGB JPEG)
    // =========================================================================
    print!("[5/5] Full cross-subsystem chaos (64 tile + 4x 4096² bbox + 4x zonal + 4x mosaic RGB JPEG, 8s)... ");
    {
        let deadline = Instant::now() + Duration::from_secs(8);
        let tile_n = Arc::new(AtomicUsize::new(0));
        let bbox4k_n = Arc::new(AtomicUsize::new(0));
        let zonal_n = Arc::new(AtomicUsize::new(0));
        let mosaic_n = Arc::new(AtomicUsize::new(0));
        let mosaic_fb = Arc::new(AtomicUsize::new(0));
        let mut hs = Vec::new();

        // 64 tile workers
        for w in 0..64usize {
            let eng = Arc::clone(&engine);
            let cog = Arc::clone(&main_cog);
            let pool = Arc::clone(&pool);
            let tile_n = Arc::clone(&tile_n);
            hs.push(tokio::spawn(async move {
                let mut i = w;
                while Instant::now() < deadline {
                    let req = TileRequest {
                        tile: pool[i % pool.len()],
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
                    eng.render_tile(&cog, &req).await.expect("chaos tile ok");
                    tile_n.fetch_add(1, Ordering::Relaxed);
                    i += 1;
                }
            }));
        }

        // 4 concurrent 4096x4096 RGB Lanczos3 JPEG bbox workers
        for _ in 0..4usize {
            let eng = Arc::clone(&engine);
            let cog = Arc::clone(&main_cog);
            let bbox4k_n = Arc::clone(&bbox4k_n);
            hs.push(tokio::spawn(async move {
                let cogs = [cog];
                while Instant::now() < deadline {
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
                        OutputFormat::Jpeg { quality: 85 },
                        None,
                        None,
                        None,
                        false,
                        false,
                    )
                    .await
                    .expect("chaos 4096^2 bbox ok");
                    assert!(b.starts_with(&[0xff, 0xd8]));
                    bbox4k_n.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        // 4 concurrent 2048² exact weighted polygon zonal workers (with 2 expressions)
        let diamond = serde_json::json!({
            "type": "Polygon",
            "coordinates": [[
                [-9.0, 36.8], [-8.2, 36.0], [-9.0, 35.2], [-9.8, 36.0], [-9.0, 36.8]
            ]]
        });
        let (pm, _) = PolygonMask::from_geojson(&diamond).unwrap();
        let pm = Arc::new(pm);
        let crs = Arc::new(engine.get_or_build_crs(&main_cog).unwrap());
        for _ in 0..4usize {
            let eng = Arc::clone(&engine);
            let cog = Arc::clone(&main_cog);
            let pm = Arc::clone(&pm);
            let crs = Arc::clone(&crs);
            let zonal_n = Arc::clone(&zonal_n);
            hs.push(tokio::spawn(async move {
                let opts = StatsOptions {
                    area_weighted: true,
                    expressions: Some(vec![
                        ("(b1-b2)/(b1+b2)".into(), ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap()),
                        ("(b2-b3)/(b2+b3)".into(), ExprProgram::compile("(b2-b3)/(b2+b3)").unwrap()),
                    ]),
                    ..StatsOptions::default()
                };
                while Instant::now() < deadline {
                    let j = compute_native_exact_polygon_stats(
                        &eng, &cog, &crs, &pm, 2048, false, None, &opts,
                    )
                    .await
                    .expect("chaos zonal ok");
                    assert!(j.is_object());
                    zonal_n.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        // 4 concurrent 1024² x 12 scenes x RGB fused Mosaic -> GPU JPEG workers
        let mosaic_scenes: Arc<Vec<Vec<f32>>> = Arc::new(
            (0..12)
                .map(|s| vec![50.0 + s as f32 * 10.0; 1024 * 1024])
                .collect(),
        );
        for _ in 0..4usize {
            let eng = Arc::clone(&engine);
            let scenes = Arc::clone(&mosaic_scenes);
            let mosaic_n = Arc::clone(&mosaic_n);
            let mosaic_fb = Arc::clone(&mosaic_fb);
            hs.push(tokio::spawn(async move {
                while Instant::now() < deadline {
                    let eng2 = Arc::clone(&eng);
                    let scenes2 = Arc::clone(&scenes);
                    let used_gpu = tokio::task::spawn_blocking(move || {
                        let refs: Vec<&[f32]> = scenes2.iter().map(Vec::as_slice).collect();
                        let rgb: [&[&[f32]]; 3] = [&refs, &refs, &refs];
                        if let Some(jpg) = eng2
                            .gpu
                            .as_ref()
                            .unwrap()
                            .try_composite_mosaic(
                                &cogzilla_server::gpu_backend::MosaicCompositeRequest {
                                    width: 1024,
                                    height: 1024,
                                    input: cogzilla_server::gpu_backend::MosaicCompositeInput::Rgb(
                                        &rgb,
                                    ),
                                    method: GpuPixelSelection::Median,
                                    output: cogzilla_server::gpu_backend::MosaicCompositeOutput::RgbStyled {
                                        rescale_min: [0.0; 3],
                                        rescale_max: [255.0; 3],
                                        format: OutputFormat::Jpeg { quality: 85 },
                                        return_mask: false,
                                    },
                                },
                            )
                            .and_then(cogzilla_server::gpu_backend::MosaicCompositeResult::into_bytes)
                        {
                            assert!(jpg.starts_with(&[0xff, 0xd8]));
                            true
                        } else {
                            false
                        }
                    })
                    .await
                    .unwrap();
                    if used_gpu {
                        mosaic_n.fetch_add(1, Ordering::Relaxed);
                    } else {
                        mosaic_fb.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }

        for h in hs {
            h.await.unwrap();
        }
        println!(
            "SURVIVED! (tiles={}, 4096^2_bboxes={}, zonal_2048^2={}, mosaic_gpu={}, mosaic_cpu_fallback={})",
            tile_n.load(Ordering::Relaxed),
            bbox4k_n.load(Ordering::Relaxed),
            zonal_n.load(Ordering::Relaxed),
            mosaic_n.load(Ordering::Relaxed),
            mosaic_fb.load(Ordering::Relaxed),
        );
    }

    // =========================================================================
    // TEST 6: 1024-Client Flood + 8x Concurrent 4096² Bbox Hammer (6s)
    // =========================================================================
    print!("[6/6] 1024-client tile flood + 8x concurrent 4096² RGB Lanczos3 JPEG (6s)... ");
    {
        let deadline = Instant::now() + Duration::from_secs(6);
        let start = Instant::now();
        let bbox4k_ok = Arc::new(AtomicUsize::new(0));
        let mut tile_hs = Vec::with_capacity(1024);
        for w in 0..1024usize {
            let eng = Arc::clone(&engine);
            let cog = Arc::clone(&main_cog);
            let pool = Arc::clone(&pool);
            tile_hs.push(tokio::spawn(async move {
                let mut lats = Vec::new();
                let mut i = w;
                while Instant::now() < deadline {
                    let req = TileRequest {
                        tile: pool[(i * 17) % pool.len()],
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
                    eng.render_tile(&cog, &req).await.expect("1024-flood tile ok");
                    lats.push(t0.elapsed().as_secs_f64() * 1e3);
                    i += 1;
                }
                lats
            }));
        }
        let mut big_hs = Vec::with_capacity(8);
        for _ in 0..8usize {
            let eng = Arc::clone(&engine);
            let cog = Arc::clone(&main_cog);
            let bbox4k_ok = Arc::clone(&bbox4k_ok);
            big_hs.push(tokio::spawn(async move {
                let cogs = [cog];
                while Instant::now() < deadline {
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
                        OutputFormat::Jpeg { quality: 85 },
                        None,
                        None,
                        None,
                        false,
                        false,
                    )
                    .await
                    .expect("8x 4096^2 bbox ok");
                    assert!(b.starts_with(&[0xff, 0xd8]));
                    bbox4k_ok.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }
        let mut all = Vec::new();
        for h in tile_hs {
            all.extend(h.await.unwrap());
        }
        for h in big_hs {
            h.await.unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64();
        all.sort_by(|a, b| a.total_cmp(b));
        println!(
            "SURVIVED! tiles={:.0} req/s (n={}, p50={:.1}ms, p95={:.1}ms, p99={:.1}ms, max={:.1}ms), 4096^2_bboxes={}",
            all.len() as f64 / elapsed,
            all.len(),
            pct(&all, 50.0),
            pct(&all, 95.0),
            pct(&all, 99.0),
            all.last().copied().unwrap_or(0.0),
            bbox4k_ok.load(Ordering::Relaxed),
        );
    }

    let c = gpu.coalescer.stats();
    println!(
        "\nALL 6 TORTURE TESTS SURVIVED! Coalescer: jobs={}, batches={}, max_batch={}",
        c.jobs_executed, c.batches_submitted, c.max_batch_seen
    );
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
