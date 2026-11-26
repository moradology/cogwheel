//! End-to-end 1-pass coalesced GPU `/cog/statistics` benchmark & parity suite.
//!
//! Builds a real 4-band Planar Float32 GeoTIFF COG (4x4 grid of 512x512 tiles = 2048x2048),
//! opens it via `TileEngine`, and benchmarks production `compute_statistics_with_options`:
//!   1. `GET /cog/statistics` (all 4 bands coalesced concurrently into 1 GPU submit)
//!   2. `GET /cog/statistics?bidx=1` (single Float32 band)
//!   3. `GET /cog/statistics?expression=NDVI;NDWI;RATIO` (3 semicolon-separated expressions
//!      coalesced concurrently into 1 GPU submit)
//!
//! Compares CPU-only `TileEngine` vs GPU `TileEngine` at `max_size=1024` and `max_size=2048`,
//! asserting full TiTiler JSON parity across every band and expression.

use std::sync::Arc;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::ExprProgram;
use cogzilla_server::engine::TileEngine;
use cogzilla_server::titiler::{StatsOptions, compute_statistics_with_options};

fn build_4band_cog_bytes(w: u32, h: u32) -> Vec<u8> {
    let tw: u32 = 512;
    let th: u32 = 512;
    let tiles_x = w / tw;
    let tiles_y = h / th;
    let tiles_per_band = (tiles_x * tiles_y) as usize;
    let bands: u32 = 4;
    let total_tiles = tiles_per_band * bands as usize;
    let tile_bytes = (tw * th * 4) as usize;

    let mut tile_data: Vec<Vec<u8>> = Vec::with_capacity(total_tiles);
    for band in 0..bands {
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let mut raw = Vec::with_capacity(tile_bytes);
                for py in 0..th {
                    let gy = ty * th + py;
                    for px in 0..tw {
                        let gx = tx * tw + px;
                        if (gx + gy * 5) % 37 == 0 {
                            raw.extend_from_slice(&f32::NAN.to_le_bytes());
                            continue;
                        }
                        let fx = gx as f32 / w as f32;
                        let fy = gy as f32 / h as f32;
                        let v = match band {
                            0 => 1100.0 + 600.0 * (fx * 11.0).sin() + 300.0 * (fy * 8.0).cos(),
                            1 => 1750.0 + 850.0 * (fx * 7.0 + 0.5).cos() + 450.0 * (fy * 10.0).sin(),
                            2 => 900.0 + 500.0 * ((fx + fy) * 9.0).sin() + 200.0 * (fx * 15.0).cos(),
                            _ => 2200.0 + 950.0 * (fx * 6.0 - fy * 4.0).cos() + 350.0 * (fy * 13.0).sin(),
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

    let bps_off = write_extra(&mut buf, &[32, 0, 32, 0, 32, 0, 32, 0]);
    let sf_off = write_extra(&mut buf, &[3, 0, 3, 0, 3, 0, 3, 0]);

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
        (258, 3, bands, bps_off),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (277, 3, 1, bands),
        (284, 3, 1, 2), // Planar
        (322, 4, 1, tw),
        (323, 4, 1, th),
        (324, 4, total_tiles as u32, offsets_pos),
        (325, 4, total_tiles as u32, counts_pos),
        (339, 3, bands, sf_off),
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

fn assert_all_bands_parity(label: &str, cpu: &serde_json::Value, gpu: &serde_json::Value) {
    let co = cpu.as_object().expect("cpu map");
    let go = gpu.as_object().expect("gpu map");
    assert_eq!(co.len(), go.len(), "{label}: band count mismatch");
    for (k, cv) in co {
        let gv = go.get(k).unwrap_or_else(|| panic!("{label}: missing key {k}"));
        let c = cv.as_object().unwrap();
        let g = gv.as_object().unwrap();
        assert_eq!(c["valid_pixels"], g["valid_pixels"], "{label}.{k}: valid_pixels");
        assert_eq!(c["masked_pixels"], g["masked_pixels"], "{label}.{k}: masked_pixels");
        assert_eq!(c["description"], g["description"], "{label}.{k}: description");
        let cmin = c["min"].as_f64().unwrap();
        let cmax = c["max"].as_f64().unwrap();
        let span = (cmax - cmin).abs().max(1.0);
        let rel = |a: f64, b: f64| (a - b).abs() / span;
        assert!(rel(cmin, g["min"].as_f64().unwrap()) < 1e-4, "{label}.{k}: min");
        assert!(rel(cmax, g["max"].as_f64().unwrap()) < 1e-4, "{label}.{k}: max");
        assert!(
            rel(c["mean"].as_f64().unwrap(), g["mean"].as_f64().unwrap()) < 1e-4,
            "{label}.{k}: mean"
        );
        assert!(
            rel(c["std"].as_f64().unwrap(), g["std"].as_f64().unwrap()) < 1e-4,
            "{label}.{k}: std"
        );
        for pkey in ["median", "percentile_2", "percentile_98"] {
            let err = rel(c[pkey].as_f64().unwrap(), g[pkey].as_f64().unwrap());
            assert!(err < 5e-3, "{label}.{k}.{pkey}: err={err:.5}");
        }
    }
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

#[tokio::main]
async fn main() {
    let tmp = std::env::temp_dir().join("cogzilla_e2e_coalesced_cog_stats_bench.tif");
    std::fs::write(&tmp, build_4band_cog_bytes(2048, 2048)).expect("write temp COG");

    let cpu_store = Arc::new(CogStore::new(512 * 1024 * 1024));
    let cpu_cog = cpu_store.register_local("s2_4band", &tmp).expect("open cpu cog");
    let mut cpu_engine = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_engine.gpu = None;

    let gpu_store = Arc::new(CogStore::new(512 * 1024 * 1024));
    let gpu_cog = gpu_store.register_local("s2_4band", &tmp).expect("open gpu cog");
    let gpu_engine = TileEngine::with_store_and_tile_cache(gpu_store, 0);
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };
    println!(
        "E2E Coalesced Multi-Band / Multi-Expr `/cog/statistics`: {} ({:?}, unified={}, floor={:.1} us)\n",
        gpu.ctx.info.name,
        gpu.ctx.info.backend,
        gpu.ctx.info.unified_memory,
        gpu.policy.profile.dispatch_floor_us
    );

    let mut base_opts = StatsOptions::default();
    base_opts.percentiles = vec![2.0, 98.0];
    base_opts.histogram_bins = 10;

    let mut single_band_opts = base_opts.clone();
    single_band_opts.bidx = Some(vec![1]);

    let exprs_3 = vec![
        ("(b2-b1)/(b2+b1)".to_string(), ExprProgram::compile("(b2-b1)/(b2+b1)").unwrap()),
        ("(b3-b2)/(b3+b2)".to_string(), ExprProgram::compile("(b3-b2)/(b3+b2)").unwrap()),
        ("(b4-b1)/(b4+b1)".to_string(), ExprProgram::compile("(b4-b1)/(b4+b1)").unwrap()),
    ];
    let mut multi_expr_opts = base_opts.clone();
    multi_expr_opts.expressions = Some(exprs_3);

    let cases: [(&str, u32, &StatsOptions); 5] = [
        ("`/cog/statistics` (4-band F32 coalesced)", 1024, &base_opts),
        ("`/cog/statistics` (4-band F32 coalesced)", 2048, &base_opts),
        ("`/cog/statistics?bidx=1` (1-band F32)", 2048, &single_band_opts),
        ("`/cog/statistics?expression=3×expr` (coalesced)", 1024, &multi_expr_opts),
        ("`/cog/statistics?expression=3×expr` (coalesced)", 2048, &multi_expr_opts),
    ];

    println!("| Endpoint / Workload | Max Size | CPU (ms) | Coalesced GPU (ms) | Speedup vs CPU |");
    println!("| :--- | :--- | ---: | ---: | ---: |");

    for (name, max_size, opts) in cases {
        let cpu_json =
            compute_statistics_with_options(&cpu_engine, &cpu_cog, max_size, None, false, None, opts)
                .await
                .unwrap();
        let gpu_json =
            compute_statistics_with_options(&gpu_engine, &gpu_cog, max_size, None, false, None, opts)
                .await
                .unwrap();
        assert_all_bands_parity(&format!("{name} {max_size}"), &cpu_json, &gpu_json);

        let iters = 5;
        let mut ct = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = compute_statistics_with_options(
                &cpu_engine, &cpu_cog, max_size, None, false, None, opts,
            )
            .await
            .unwrap();
            ct.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = compute_statistics_with_options(
                &gpu_engine, &gpu_cog, max_size, None, false, None, opts,
            )
            .await
            .unwrap();
            gt.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let cm = median_ms(ct);
        let gm = median_ms(gt);
        println!(
            "| {} | {}×{} | {:.2} | {:.2} | {:.1}× |",
            name,
            max_size,
            max_size,
            cm,
            gm,
            cm / gm
        );
    }

    let _ = std::fs::remove_file(&tmp);
    println!("\nALL COALESCED /cog/statistics PARITY CHECKS PASSED!");
}
