//! End-to-end TiTiler / STAC 1-pass fused GPU statistics parity & benchmark.
//!
//! Builds a real 2-band Planar Float32 GeoTIFF COG (2x2 grid of 512x512 tiles = 1024x1024),
//! opens it via `TileEngine`, and compares:
//!   1. CPU baseline (`render_native_preview(RawF32)` on CPU engine + `compute_f32_band_stats`)
//!   2. 2-pass GPU (previous production path: `render_native_preview(RawF32)` + `compute_f32_band_stats_with_engine`)
//!   3. 1-pass Fused GPU (new production path: `render_native_preview_stats` / `render_wgs84_bbox_stats`)
//!
//! Asserts full TiTiler JSON schema + numerical parity across Single-band, CustomExpr NDVI,
//! and GeoJSON `PolygonMask`-masked WGS84 bbox statistics.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, ExprProgram, OutputFormat, ResampleMode};
use cogzilla_server::engine::{BandRequest, TileEngine};
use cogzilla_server::titiler::{
    PolygonMask, StatsOptions, compute_f32_band_stats, compute_f32_band_stats_with_engine,
    render_native_preview, render_native_preview_stats, render_wgs84_bbox, render_wgs84_bbox_stats,
};

fn build_2band_1024_cog_bytes() -> Vec<u8> {
    let w: u32 = 1024;
    let h: u32 = 1024;
    let tw: u32 = 512;
    let th: u32 = 512;
    let tiles_x = w / tw;
    let tiles_y = h / th;
    let tiles_per_band = (tiles_x * tiles_y) as usize;
    let total_tiles = tiles_per_band * 2;
    let tile_bytes = (tw * th * 4) as usize;

    let mut tile_data: Vec<Vec<u8>> = Vec::with_capacity(total_tiles);
    for band in 0..2u32 {
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let mut raw = Vec::with_capacity(tile_bytes);
                for py in 0..th {
                    let gy = ty * th + py;
                    for px in 0..tw {
                        let gx = tx * tw + px;
                        // Inject ~4% NaN nodata stripe to exercise masked_pixels / valid_percent
                        if (gx + gy * 7) % 29 == 0 {
                            raw.extend_from_slice(&f32::NAN.to_le_bytes());
                            continue;
                        }
                        let fx = gx as f32 / w as f32;
                        let fy = gy as f32 / h as f32;
                        let v = match band {
                            0 => 1200.0 + 700.0 * (fx * 11.0).sin() + 350.0 * (fy * 8.0).cos(),
                            _ => 1850.0 + 900.0 * (fx * 7.0 + 0.5).cos() + 500.0 * (fy * 10.0).sin(),
                        };
                        raw.extend_from_slice(&v.to_le_bytes());
                    }
                }
                tile_data.push(raw);
            }
        }
    }

    // Classic GeoTIFF (EPSG:4326 covering [-10.0, 35.0, -8.0, 37.0])
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    let ifd_offset: u32 = 8;
    buf.extend_from_slice(&ifd_offset.to_le_bytes());

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

    let bps_val: u32 = 32 | (32 << 16);
    let sf_val: u32 = 3 | (3 << 16);

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

    let gkd: [u16; 16] = [
        1, 1, 0, 3,
        1024, 0, 1, 2,
        1025, 0, 1, 1,
        2048, 0, 1, 4326,
    ];
    let mut gkd_bytes = Vec::with_capacity(32);
    for v in gkd {
        gkd_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let gkd_off = write_extra(&mut buf, &gkd_bytes);

    let offsets_pos = buf.len() as u32;
    buf.resize(buf.len() + total_tiles * 4, 0);
    let counts_pos = buf.len() as u32;
    buf.resize(buf.len() + total_tiles * 4, 0);

    let mut tile_offsets = Vec::with_capacity(total_tiles);
    for td in &tile_data {
        tile_offsets.push(buf.len() as u32);
        buf.extend_from_slice(td);
    }

    for (i, &off) in tile_offsets.iter().enumerate() {
        let p = offsets_pos as usize + i * 4;
        buf[p..p + 4].copy_from_slice(&off.to_le_bytes());
        let cp = counts_pos as usize + i * 4;
        buf[cp..cp + 4].copy_from_slice(&(tile_bytes as u32).to_le_bytes());
    }

    let entries: [(u16, u16, u32, u32); 16] = [
        (256, 4, 1, w),
        (257, 4, 1, h),
        (258, 3, 2, bps_val),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (277, 3, 1, 2), // SamplesPerPixel = 2
        (284, 3, 1, 2), // PlanarConfiguration = Separate
        (322, 4, 1, tw),
        (323, 4, 1, th),
        (324, 4, total_tiles as u32, offsets_pos),
        (325, 4, total_tiles as u32, counts_pos),
        (339, 3, 2, sf_val),
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

fn assert_json_parity(label: &str, cpu: &serde_json::Value, gpu: &serde_json::Value) {
    let co = cpu.as_object().expect("cpu object");
    let go = gpu.as_object().expect("gpu object");

    let c_valid = co["valid_pixels"].as_u64().unwrap();
    let g_valid = go["valid_pixels"].as_u64().unwrap();
    let valid_diff = c_valid.abs_diff(g_valid);
    assert!(
        valid_diff <= (c_valid / 5000).max(4),
        "{label}: valid_pixels mismatch cpu={c_valid} gpu={g_valid}"
    );
    assert_eq!(co["description"], go["description"], "{label}: description mismatch");

    let c_min = co["min"].as_f64().unwrap();
    let g_min = go["min"].as_f64().unwrap();
    let c_max = co["max"].as_f64().unwrap();
    let g_max = go["max"].as_f64().unwrap();
    let span = (c_max - c_min).abs().max(1.0);

    let rel = |a: f64, b: f64| (a - b).abs() / span;
    assert!(rel(c_min, g_min) < 1e-3, "{label}: min cpu={c_min} gpu={g_min}");
    assert!(rel(c_max, g_max) < 1e-3, "{label}: max cpu={c_max} gpu={g_max}");
    assert!(
        rel(co["mean"].as_f64().unwrap(), go["mean"].as_f64().unwrap()) < 1e-3,
        "{label}: mean cpu={} gpu={}",
        co["mean"],
        go["mean"]
    );
    assert!(
        rel(co["std"].as_f64().unwrap(), go["std"].as_f64().unwrap()) < 1e-3,
        "{label}: std cpu={} gpu={}",
        co["std"],
        go["std"]
    );
    for key in ["median", "percentile_2", "percentile_98"] {
        let cv = co[key].as_f64().unwrap();
        let gv = go[key].as_f64().unwrap();
        assert!(
            rel(cv, gv) < 5e-3,
            "{label}: {key} cpu={cv} gpu={gv} (rel={:.5})",
            rel(cv, gv)
        );
    }
    let ch = co["histogram"].as_array().unwrap();
    let gh = go["histogram"].as_array().unwrap();
    assert_eq!(ch[0].as_array().unwrap().len(), gh[0].as_array().unwrap().len());
    assert_eq!(ch[1].as_array().unwrap().len(), gh[1].as_array().unwrap().len());
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

#[tokio::main]
async fn main() {
    let tmp = std::env::temp_dir().join("cogzilla_e2e_fused_stats_bench.tif");
    std::fs::write(&tmp, build_2band_1024_cog_bytes()).expect("write temp COG");

    let cpu_store = Arc::new(CogStore::new(256 * 1024 * 1024));
    let cpu_cog = cpu_store.register_local("s2", &tmp).expect("open cpu cog");
    let mut cpu_engine = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_engine.gpu = None;

    let gpu_store = Arc::new(CogStore::new(256 * 1024 * 1024));
    let gpu_cog = gpu_store.register_local("s2", &tmp).expect("open gpu cog");
    let gpu_engine = TileEngine::with_store_and_tile_cache(gpu_store, 0);
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };
    println!(
        "E2E TiTiler/STAC 1-Pass Fused GPU Stats: {} ({:?}, unified={}, floor={:.1} us)\n",
        gpu.ctx.info.name,
        gpu.ctx.info.backend,
        gpu.ctx.info.unified_memory,
        gpu.policy.profile.dispatch_floor_us
    );

    let cpu_cogs = [Arc::clone(&cpu_cog)];
    let gpu_cogs = [Arc::clone(&gpu_cog)];
    let full_bbox = [-10.0, 35.0, -8.0, 37.0];

    let diamond_geojson = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[
            [-9.0, 36.8],
            [-8.2, 36.0],
            [-9.0, 35.2],
            [-9.8, 36.0],
            [-9.0, 36.8]
        ]]
    });
    let (poly_mask, _) = PolygonMask::from_geojson(&diamond_geojson).expect("valid polygon");

    let mut stats_opts = StatsOptions::default();
    stats_opts.percentiles = vec![2.0, 98.0];
    stats_opts.histogram_bins = 10;

    let single_req = BandRequest::Single {
        band_1based: 1,
        rescale: (0.0, 1.0),
        colormap: ColormapName::Grayscale,
    };
    let ndvi_prog = ExprProgram::compile("(b2-b1)/(b2+b1)").unwrap();
    let ndvi_req = BandRequest::CustomExpr {
        prog: ndvi_prog,
        rescale: (0.0, 1.0),
        colormap: ColormapName::Grayscale,
    };

    struct Case<'a> {
        name: &'a str,
        poly: bool,
        w: u32,
        h: u32,
        bands: &'a BandRequest,
        resample: ResampleMode,
        desc: &'a str,
    }

    let cases = [
        Case {
            name: "Preview Single Nearest",
            poly: false,
            w: 1024,
            h: 1024,
            bands: &single_req,
            resample: ResampleMode::Nearest,
            desc: "b1",
        },
        Case {
            name: "Preview Single Bilinear",
            poly: false,
            w: 2048,
            h: 2048,
            bands: &single_req,
            resample: ResampleMode::Bilinear,
            desc: "b1",
        },
        Case {
            name: "Preview Single Bilinear",
            poly: false,
            w: 4096,
            h: 4096,
            bands: &single_req,
            resample: ResampleMode::Bilinear,
            desc: "b1",
        },
        Case {
            name: "STAC Expr NDVI Bilinear",
            poly: false,
            w: 1024,
            h: 1024,
            bands: &ndvi_req,
            resample: ResampleMode::Bilinear,
            desc: "(b2-b1)/(b2+b1)",
        },
        Case {
            name: "STAC Expr NDVI Bilinear",
            poly: false,
            w: 2048,
            h: 2048,
            bands: &ndvi_req,
            resample: ResampleMode::Bilinear,
            desc: "(b2-b1)/(b2+b1)",
        },
        Case {
            name: "STAC Expr NDVI Lanczos3",
            poly: false,
            w: 4096,
            h: 4096,
            bands: &ndvi_req,
            resample: ResampleMode::Lanczos,
            desc: "(b2-b1)/(b2+b1)",
        },
        Case {
            name: "Polygon-Masked NDVI Nearest",
            poly: true,
            w: 2048,
            h: 2048,
            bands: &ndvi_req,
            resample: ResampleMode::Nearest,
            desc: "(b2-b1)/(b2+b1)",
        },
        Case {
            name: "Polygon-Masked Single Nearest",
            poly: true,
            w: 4096,
            h: 4096,
            bands: &single_req,
            resample: ResampleMode::Nearest,
            desc: "b1",
        },
    ];

    println!("| Endpoint / Workload | Size | CPU (ms) | 2-Pass GPU (ms) | 1-Pass Fused (ms) | vs 2-Pass | vs CPU | GPU Dispatches |");
    println!("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |");

    for c in &cases {
        let pm_opt = if c.poly { Some(&poly_mask) } else { None };
        let bbox = if c.poly { poly_mask.bbox } else { full_bbox };

        // 1. CPU reference (also verifies parity)
        let run_cpu = || async {
            let raw = if c.poly {
                render_wgs84_bbox(
                    &cpu_engine,
                    &cpu_cogs,
                    bbox,
                    c.w,
                    c.h,
                    pm_opt,
                    c.bands,
                    c.resample,
                    OutputFormat::RawF32,
                    None,
                    None,
                    None,
                    true,
                    false,
                )
                .await
                .unwrap()
            } else {
                render_native_preview(
                    &cpu_engine,
                    &cpu_cogs,
                    bbox,
                    c.w,
                    c.h,
                    c.bands,
                    c.resample,
                    OutputFormat::RawF32,
                    None,
                    None,
                    None,
                    true,
                    false,
                )
                .await
                .unwrap()
            };
            let pixels: Vec<f32> = raw
                .chunks_exact(4)
                .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
                .collect();
            compute_f32_band_stats(&pixels, c.desc, &stats_opts)
        };

        // 2. Old 2-pass GPU (RawF32 readback + compute_f32_band_stats_with_engine re-upload)
        let run_2pass = || async {
            let raw = if c.poly {
                render_wgs84_bbox(
                    &gpu_engine,
                    &gpu_cogs,
                    bbox,
                    c.w,
                    c.h,
                    pm_opt,
                    c.bands,
                    c.resample,
                    OutputFormat::RawF32,
                    None,
                    None,
                    None,
                    true,
                    false,
                )
                .await
                .unwrap()
            } else {
                render_native_preview(
                    &gpu_engine,
                    &gpu_cogs,
                    bbox,
                    c.w,
                    c.h,
                    c.bands,
                    c.resample,
                    OutputFormat::RawF32,
                    None,
                    None,
                    None,
                    true,
                    false,
                )
                .await
                .unwrap()
            };
            let pixels: Vec<f32> = raw
                .chunks_exact(4)
                .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
                .collect();
            compute_f32_band_stats_with_engine(&gpu_engine, &pixels, c.desc, &stats_opts)
        };

        // 3. New 1-pass Fused GPU
        let run_1pass = || async {
            if c.poly {
                render_wgs84_bbox_stats(
                    &gpu_engine,
                    &gpu_cogs,
                    bbox,
                    c.w,
                    c.h,
                    pm_opt,
                    c.bands,
                    c.resample,
                    None,
                    false,
                    c.desc,
                    &stats_opts,
                )
                .await
                .unwrap()
            } else {
                render_native_preview_stats(
                    &gpu_engine,
                    &gpu_cogs,
                    bbox,
                    c.w,
                    c.h,
                    c.bands,
                    c.resample,
                    None,
                    false,
                    c.desc,
                    &stats_opts,
                )
                .await
                .unwrap()
            }
        };

        // Warmup & parity verification
        let cpu_json = run_cpu().await;
        let _ = run_2pass().await;
        let d0 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        let gpu_json = run_1pass().await;
        let d1 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        assert!(
            d1 > d0,
            "{} {}x{}: expected 1-pass fused GPU dispatch!",
            c.name,
            c.w,
            c.h
        );
        assert_json_parity(&format!("{} {}x{}", c.name, c.w, c.h), &cpu_json, &gpu_json);

        let iters = if c.w >= 4096 { 3 } else { 5 };
        let mut cpu_t = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = run_cpu().await;
            cpu_t.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut g2_t = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = run_2pass().await;
            g2_t.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut g1_t = Vec::with_capacity(iters);
        let d_start = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = run_1pass().await;
            g1_t.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let d_used = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - d_start;

        let cm = median_ms(cpu_t);
        let m2 = median_ms(g2_t);
        let m1 = median_ms(g1_t);
        println!(
            "| {} | {}×{} | {:.2} | {:.2} | {:.2} | {:.2}× | {:.1}× | {} |",
            c.name,
            c.w,
            c.h,
            cm,
            m2,
            m1,
            m2 / m1,
            cm / m1,
            d_used
        );
    }

    let _ = std::fs::remove_file(&tmp);
    println!("\nALL E2E TITILER/STAC FUSED STATS PARITY CHECKS PASSED!");
}
