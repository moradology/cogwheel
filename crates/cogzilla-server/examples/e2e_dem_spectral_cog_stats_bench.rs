//! End-to-end 1-pass fused GPU statistics parity & benchmark for:
//!   1. DEM Horn 3×3 (`Hillshade`, `Slope`) via `render_native_preview_stats`
//!   2. Built-in `Spectral2` (`NormalizedDiff`) via `render_native_preview_stats`
//!   3. Production `GET /cog/statistics?expression=...` (`compute_statistics_with_options`)
//!      with the new 1-pass fused GPU path inside `compute_expression_statistics_cpu`.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{
    ColormapName, DemAlgorithm, ExprProgram, OutputFormat, ResampleMode, SpectralExpr,
};
use cogzilla_server::engine::{BandRequest, TileEngine};
use cogzilla_server::titiler::{
    StatsOptions, compute_f32_band_stats, compute_f32_band_stats_with_engine,
    compute_statistics_with_options, render_native_preview, render_native_preview_stats,
};

fn build_2band_cog_bytes(w: u32, h: u32) -> Vec<u8> {
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
        (277, 3, 1, 2),
        (284, 3, 1, 2),
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

fn assert_band_parity(label: &str, cpu: &serde_json::Value, gpu: &serde_json::Value) {
    let co = cpu.as_object().expect("cpu object");
    let go = gpu.as_object().expect("gpu object");

    let c_valid = co["valid_pixels"].as_u64().unwrap();
    let g_valid = go["valid_pixels"].as_u64().unwrap();
    assert!(
        c_valid.abs_diff(g_valid) <= (c_valid / 2000).max(4),
        "{label}: valid_pixels cpu={c_valid} gpu={g_valid}"
    );
    let c_min = co["min"].as_f64().unwrap();
    let g_min = go["min"].as_f64().unwrap();
    let c_max = co["max"].as_f64().unwrap();
    let g_max = go["max"].as_f64().unwrap();
    let span = (c_max - c_min).abs().max(1.0);
    let rel = |a: f64, b: f64| (a - b).abs() / span;

    assert!(rel(c_min, g_min) < 5e-3, "{label}: min cpu={c_min} gpu={g_min}");
    assert!(rel(c_max, g_max) < 5e-3, "{label}: max cpu={c_max} gpu={g_max}");
    assert!(
        rel(co["mean"].as_f64().unwrap(), go["mean"].as_f64().unwrap()) < 5e-3,
        "{label}: mean cpu={} gpu={}",
        co["mean"],
        go["mean"]
    );
    assert!(
        rel(co["std"].as_f64().unwrap(), go["std"].as_f64().unwrap()) < 5e-3,
        "{label}: std cpu={} gpu={}",
        co["std"],
        go["std"]
    );
    for key in ["median", "percentile_2", "percentile_98"] {
        let cv = co[key].as_f64().unwrap();
        let gv = go[key].as_f64().unwrap();
        assert!(
            rel(cv, gv) < 1e-2,
            "{label}: {key} cpu={cv} gpu={gv} (rel={:.5})",
            rel(cv, gv)
        );
    }
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

#[tokio::main]
async fn main() {
    let tmp = std::env::temp_dir().join("cogzilla_e2e_dem_spectral_cog_stats_bench.tif");
    std::fs::write(&tmp, build_2band_cog_bytes(2048, 2048)).expect("write temp COG");

    let cpu_store = Arc::new(CogStore::new(512 * 1024 * 1024));
    let cpu_cog = cpu_store.register_local("dem_s2", &tmp).expect("open cpu cog");
    let mut cpu_engine = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_engine.gpu = None;

    let gpu_store = Arc::new(CogStore::new(512 * 1024 * 1024));
    let gpu_cog = gpu_store.register_local("dem_s2", &tmp).expect("open gpu cog");
    let gpu_engine = TileEngine::with_store_and_tile_cache(gpu_store, 0);
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };
    println!(
        "E2E DEM / Spectral / /cog/statistics?expression 1-Pass Fused GPU Stats: {} ({:?}, unified={}, floor={:.1} us)\n",
        gpu.ctx.info.name,
        gpu.ctx.info.backend,
        gpu.ctx.info.unified_memory,
        gpu.policy.profile.dispatch_floor_us
    );

    let cpu_cogs = [Arc::clone(&cpu_cog)];
    let gpu_cogs = [Arc::clone(&gpu_cog)];
    let full_bbox = [-10.0, 35.0, -8.0, 37.0];

    let mut stats_opts = StatsOptions::default();
    stats_opts.percentiles = vec![2.0, 98.0];
    stats_opts.histogram_bins = 10;

    let hillshade_req = BandRequest::Dem {
        band_1based: 1,
        algorithm: DemAlgorithm::Hillshade {
            azimuth_deg: 315.0,
            altitude_deg: 45.0,
            z_factor: 1.0,
        },
        rescale: (0.0, 255.0),
        colormap: ColormapName::Grayscale,
    };
    let slope_req = BandRequest::Dem {
        band_1based: 1,
        algorithm: DemAlgorithm::Slope { z_factor: 1.0 },
        rescale: (0.0, 90.0),
        colormap: ColormapName::Grayscale,
    };
    let spectral_ndvi_req = BandRequest::Spectral2 {
        expr: SpectralExpr::NormalizedDiff,
        bands_1based: [2, 1],
        rescale: (-1.0, 1.0),
        colormap: ColormapName::Grayscale,
    };

    struct PreviewCase<'a> {
        name: &'a str,
        w: u32,
        h: u32,
        bands: &'a BandRequest,
        resample: ResampleMode,
        desc: &'a str,
    }

    let preview_cases = [
        PreviewCase {
            name: "DEM Hillshade Bilinear",
            w: 1024,
            h: 1024,
            bands: &hillshade_req,
            resample: ResampleMode::Bilinear,
            desc: "hillshade",
        },
        PreviewCase {
            name: "DEM Hillshade Cubic",
            w: 2048,
            h: 2048,
            bands: &hillshade_req,
            resample: ResampleMode::Cubic,
            desc: "hillshade",
        },
        PreviewCase {
            name: "DEM Slope Lanczos3",
            w: 4096,
            h: 4096,
            bands: &slope_req,
            resample: ResampleMode::Lanczos,
            desc: "slope",
        },
        PreviewCase {
            name: "Spectral2 NDVI Bilinear",
            w: 1024,
            h: 1024,
            bands: &spectral_ndvi_req,
            resample: ResampleMode::Bilinear,
            desc: "ndvi",
        },
        PreviewCase {
            name: "Spectral2 NDVI Lanczos3",
            w: 4096,
            h: 4096,
            bands: &spectral_ndvi_req,
            resample: ResampleMode::Lanczos,
            desc: "ndvi",
        },
    ];

    println!("### Part 1: DEM Horn 3×3 & Built-in Spectral 1-Pass Fused Stats (`render_native_preview_stats`)");
    println!("| Endpoint / Workload | Size | CPU (ms) | 2-Pass GPU (ms) | 1-Pass Fused (ms) | vs 2-Pass | vs CPU | GPU Dispatches |");
    println!("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |");

    for c in &preview_cases {
        let run_cpu = || async {
            let raw = render_native_preview(
                &cpu_engine,
                &cpu_cogs,
                full_bbox,
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
            .unwrap();
            let pixels: Vec<f32> = raw
                .chunks_exact(4)
                .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
                .collect();
            compute_f32_band_stats(&pixels, c.desc, &stats_opts)
        };
        let run_2pass = || async {
            let raw = render_native_preview(
                &gpu_engine,
                &gpu_cogs,
                full_bbox,
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
            .unwrap();
            let pixels: Vec<f32> = raw
                .chunks_exact(4)
                .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
                .collect();
            compute_f32_band_stats_with_engine(&gpu_engine, &pixels, c.desc, &stats_opts)
        };
        let run_1pass = || async {
            render_native_preview_stats(
                &gpu_engine,
                &gpu_cogs,
                full_bbox,
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
        };

        let cpu_json = run_cpu().await;
        let _ = run_2pass().await;
        let d0 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        let gpu_json = run_1pass().await;
        let d1 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        assert!(d1 > d0, "{} {}x{}: expected GPU dispatch", c.name, c.w, c.h);
        assert_band_parity(&format!("{} {}x{}", c.name, c.w, c.h), &cpu_json, &gpu_json);

        let iters = if c.w >= 4096 { 3 } else { 5 };
        let mut ct = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = run_cpu().await;
            ct.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut g2t = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = run_2pass().await;
            g2t.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut g1t = Vec::with_capacity(iters);
        let ds = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = run_1pass().await;
            g1t.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let du = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - ds;
        let cm = median_ms(ct);
        let m2 = median_ms(g2t);
        let m1 = median_ms(g1t);
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
            du
        );
    }

    println!("\n### Part 2: Production `GET /cog/statistics?expression=(b2-b1)/(b2+b1)` (`compute_statistics_with_options`)");
    println!("| Endpoint / Workload | Max Size | CPU (ms) | 1-Pass Fused GPU (ms) | Speedup vs CPU |");
    println!("| :--- | :--- | ---: | ---: | ---: |");

    let expr_str = "(b2-b1)/(b2+b1)".to_string();
    let prog = ExprProgram::compile(&expr_str).unwrap();
    for max_size in [1024u32, 2048u32] {
        let mut opts = stats_opts.clone();
        opts.expressions = Some(vec![(expr_str.clone(), prog.clone())]);

        let cpu_json =
            compute_statistics_with_options(&cpu_engine, &cpu_cog, max_size, None, false, None, &opts)
                .await
                .unwrap();
        let gpu_json =
            compute_statistics_with_options(&gpu_engine, &gpu_cog, max_size, None, false, None, &opts)
                .await
                .unwrap();
        assert_band_parity(
            &format!("/cog/statistics?expression max_size={max_size}"),
            &cpu_json["b1"],
            &gpu_json["b1"],
        );

        let iters = 5;
        let mut ct = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = compute_statistics_with_options(
                &cpu_engine, &cpu_cog, max_size, None, false, None, &opts,
            )
            .await
            .unwrap();
            ct.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = compute_statistics_with_options(
                &gpu_engine, &gpu_cog, max_size, None, false, None, &opts,
            )
            .await
            .unwrap();
            gt.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let cm = median_ms(ct);
        let gm = median_ms(gt);
        println!(
            "| `/cog/statistics?expression=NDVI` | {}×{} | {:.2} | {:.2} | {:.1}× |",
            max_size,
            max_size,
            cm,
            gm,
            cm / gm
        );
    }

    let _ = std::fs::remove_file(&tmp);
    println!("\nALL DEM / SPECTRAL / COG EXPRESSION STATS PARITY CHECKS PASSED!");
}
