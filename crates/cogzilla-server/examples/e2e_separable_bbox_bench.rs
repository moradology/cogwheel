//! End-to-end benchmark & parity suite for lazy `BboxCoordMap::Separable` across
//! `/cog/bbox` (`render_wgs84_bbox`), `/cog/preview` (`render_native_preview`),
//! `WebMercatorQuad` (`render_mercator_bbox`), and `render_wgs84_bbox_stats`.
//!
//! Eliminates the 128 MB 2D `u_map`/`v_map` allocation, 16.7M-pixel CPU broadcast loop,
//! and 128 MB PCIe `Exact2D` upload at 4096×4096 in favor of a 32 KB 1D `Separable` upload.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, ExprProgram, OutputFormat, ResampleMode};
use cogzilla_server::engine::{BandRequest, TileEngine};
use cogzilla_server::titiler::{
    StatsOptions, render_mercator_bbox, render_native_preview, render_native_preview_stats,
    render_wgs84_bbox, render_wgs84_bbox_stats,
};

fn build_3band_1024_cog_bytes() -> Vec<u8> {
    let w: u32 = 1024;
    let h: u32 = 1024;
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
                            0 => 1200.0 + 900.0 * (fx * 12.0).sin() + 400.0 * (fy * 9.0).cos(),
                            1 => 650.0 + 450.0 * (fx * 8.0 + 1.0).cos() + 250.0 * (fy * 11.0).sin(),
                            _ => 400.0 + 300.0 * ((fx + fy) * 7.0).sin(),
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
        (258, 3, 3, bps_off),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (277, 3, 1, 3),
        (284, 3, 1, 2),
        (322, 4, 1, tw),
        (323, 4, 1, th),
        (324, 4, total_tiles as u32, offsets_pos),
        (325, 4, total_tiles as u32, counts_pos),
        (339, 3, 3, sf_off),
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

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn assert_rgba_parity(label: &str, cpu: &[u8], gpu: &[u8]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: byte len mismatch");
    let mut max_diff = 0u8;
    let mut over1 = 0usize;
    for (&a, &b) in cpu.iter().zip(gpu.iter()) {
        let d = a.abs_diff(b);
        if d > max_diff {
            max_diff = d;
        }
        if d > 1 {
            over1 += 1;
        }
    }
    let frac = over1 as f64 / cpu.len().max(1) as f64;
    assert!(
        max_diff <= 3 && frac < 0.005,
        "{label}: RGBA parity failed (max_diff={max_diff}, >1 frac={frac:.5})"
    );
}

#[tokio::main]
async fn main() {
    let tmp = std::env::temp_dir().join("cogzilla_e2e_separable_bbox_bench.tif");
    std::fs::write(&tmp, build_3band_1024_cog_bytes()).expect("write temp COG");

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
        "E2E Lazy Separable `BboxCoordMap` Benchmark: {} ({:?}, unified={}, floor={:.1} us)\n",
        gpu.ctx.info.name,
        gpu.ctx.info.backend,
        gpu.ctx.info.unified_memory,
        gpu.policy.profile.dispatch_floor_us
    );

    let cpu_cogs = [Arc::clone(&cpu_cog)];
    let gpu_cogs = [Arc::clone(&gpu_cog)];
    let sub_bbox = [-9.7, 35.3, -8.3, 36.7];
    let full_bbox = [-10.0, 35.0, -8.0, 37.0];
    // Web Mercator bbox corresponding to [-9.7, 35.3, -8.3, 36.7]
    let (mx0, my0) = cogzilla_proj::CrsModel::Epsg3857.from_lonlat_deg(-9.7, 35.3);
    let (mx1, my1) = cogzilla_proj::CrsModel::Epsg3857.from_lonlat_deg(-8.3, 36.7);
    let merc_bbox = [mx0, my0, mx1, my1];

    let rgb_req = BandRequest::Rgb {
        bands_1based: [1, 2, 3],
        rescale_min: [0.0, 0.0, 0.0],
        rescale_max: [2500.0, 1500.0, 1000.0],
    };
    let single_req = BandRequest::Single {
        band_1based: 1,
        rescale: (0.0, 2500.0),
        colormap: ColormapName::Viridis,
    };
    let ndvi_req = BandRequest::CustomExpr {
        prog: ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap(),
        rescale: (-1.0, 1.0),
        colormap: ColormapName::Viridis,
    };
    let mut stats_opts = StatsOptions::default();
    stats_opts.percentiles = vec![2.0, 98.0];

    println!("| Endpoint / Workload | Size | CPU (ms) | GPU Separable (ms) | Speedup vs CPU | GPU Dispatches |");
    println!("| :--- | :--- | ---: | ---: | ---: | ---: |");

    // 1. /cog/bbox WGS84 RGB Cubic RawRgba (2048 & 4096)
    for sz in [2048u32, 4096u32] {
        let cpu_raw = render_wgs84_bbox(
            &cpu_engine, &cpu_cogs, sub_bbox, sz, sz, None, &rgb_req,
            ResampleMode::Cubic, OutputFormat::RawRgba, None, None, None, true, false,
        ).await.unwrap();
        let d0 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        let gpu_raw = render_wgs84_bbox(
            &gpu_engine, &gpu_cogs, sub_bbox, sz, sz, None, &rgb_req,
            ResampleMode::Cubic, OutputFormat::RawRgba, None, None, None, true, false,
        ).await.unwrap();
        assert!(gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) > d0);
        assert_rgba_parity(&format!("/cog/bbox RGB Cubic {sz}"), &cpu_raw, &gpu_raw);

        let iters = if sz >= 4096 { 3 } else { 5 };
        let mut ct = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = render_wgs84_bbox(
                &cpu_engine, &cpu_cogs, sub_bbox, sz, sz, None, &rgb_req,
                ResampleMode::Cubic, OutputFormat::RawRgba, None, None, None, true, false,
            ).await.unwrap();
            ct.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt = Vec::new();
        let ds = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = render_wgs84_bbox(
                &gpu_engine, &gpu_cogs, sub_bbox, sz, sz, None, &rgb_req,
                ResampleMode::Cubic, OutputFormat::RawRgba, None, None, None, true, false,
            ).await.unwrap();
            gt.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let du = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - ds;
        let cm = median_ms(ct);
        let gm = median_ms(gt);
        println!("| `/cog/bbox` WGS84 RGB Cubic RawRgba | {sz}×{sz} | {cm:.2} | {gm:.2} | {:.1}× | {du} |", cm / gm);
    }

    // 2. /cog/bbox WGS84 RGB Lanczos3 JPEG (2048 & 4096)
    for sz in [2048u32, 4096u32] {
        let _ = render_wgs84_bbox(
            &gpu_engine, &gpu_cogs, sub_bbox, sz, sz, None, &rgb_req,
            ResampleMode::Lanczos, OutputFormat::Jpeg { quality: 85 }, None, None, None, true, false,
        ).await.unwrap();
        let iters = if sz >= 4096 { 3 } else { 5 };
        let mut ct = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = render_wgs84_bbox(
                &cpu_engine, &cpu_cogs, sub_bbox, sz, sz, None, &rgb_req,
                ResampleMode::Lanczos, OutputFormat::Jpeg { quality: 85 }, None, None, None, true, false,
            ).await.unwrap();
            ct.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt = Vec::new();
        let ds = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = render_wgs84_bbox(
                &gpu_engine, &gpu_cogs, sub_bbox, sz, sz, None, &rgb_req,
                ResampleMode::Lanczos, OutputFormat::Jpeg { quality: 85 }, None, None, None, true, false,
            ).await.unwrap();
            gt.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let du = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - ds;
        let cm = median_ms(ct);
        let gm = median_ms(gt);
        println!("| `/cog/bbox` WGS84 RGB Lanczos3 JPEG | {sz}×{sz} | {cm:.2} | {gm:.2} | {:.1}× | {du} |", cm / gm);
    }

    // 3. WebMercatorQuad (render_mercator_bbox) Single Viridis Cubic RawRgba 2048x2048
    {
        let sz = 2048u32;
        let cpu_raw = render_mercator_bbox(
            &cpu_engine, &cpu_cogs, merc_bbox, sz, sz, &single_req,
            ResampleMode::Cubic, OutputFormat::RawRgba, None, None, None, true, false,
        ).await.unwrap();
        let gpu_raw = render_mercator_bbox(
            &gpu_engine, &gpu_cogs, merc_bbox, sz, sz, &single_req,
            ResampleMode::Cubic, OutputFormat::RawRgba, None, None, None, true, false,
        ).await.unwrap();
        assert_rgba_parity("WebMercator Single Cubic 2048", &cpu_raw, &gpu_raw);
        let mut ct = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = render_mercator_bbox(
                &cpu_engine, &cpu_cogs, merc_bbox, sz, sz, &single_req,
                ResampleMode::Cubic, OutputFormat::RawRgba, None, None, None, true, false,
            ).await.unwrap();
            ct.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt = Vec::new();
        let ds = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = render_mercator_bbox(
                &gpu_engine, &gpu_cogs, merc_bbox, sz, sz, &single_req,
                ResampleMode::Cubic, OutputFormat::RawRgba, None, None, None, true, false,
            ).await.unwrap();
            gt.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let du = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - ds;
        let cm = median_ms(ct);
        let gm = median_ms(gt);
        println!("| `render_mercator_bbox` Single Cubic RawRgba | {sz}×{sz} | {cm:.2} | {gm:.2} | {:.1}× | {du} |", cm / gm);
    }

    // 4. /cog/bbox WGS84 NDVI Fused Stats 4096x4096
    {
        let sz = 4096u32;
        let _ = render_wgs84_bbox_stats(
            &gpu_engine, &gpu_cogs, sub_bbox, sz, sz, None, &ndvi_req,
            ResampleMode::Bilinear, None, false, "ndvi", &stats_opts,
        ).await.unwrap();
        let mut ct = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = render_wgs84_bbox_stats(
                &cpu_engine, &cpu_cogs, sub_bbox, sz, sz, None, &ndvi_req,
                ResampleMode::Bilinear, None, false, "ndvi", &stats_opts,
            ).await.unwrap();
            ct.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt = Vec::new();
        let ds = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = render_wgs84_bbox_stats(
                &gpu_engine, &gpu_cogs, sub_bbox, sz, sz, None, &ndvi_req,
                ResampleMode::Bilinear, None, false, "ndvi", &stats_opts,
            ).await.unwrap();
            gt.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let du = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - ds;
        let cm = median_ms(ct);
        let gm = median_ms(gt);
        println!("| `render_wgs84_bbox_stats` NDVI Bilinear | {sz}×{sz} | {cm:.2} | {gm:.2} | {:.1}× | {du} |", cm / gm);
    }

    // 5. /cog/preview Single Bilinear Fused Stats 4096x4096 (was 16.80 ms RTX / 7.34 ms M5 with 128 MB u_map)
    {
        let sz = 4096u32;
        let _ = render_native_preview_stats(
            &gpu_engine, &gpu_cogs, full_bbox, sz, sz, &single_req,
            ResampleMode::Bilinear, None, false, "b1", &stats_opts,
        ).await.unwrap();
        let mut ct = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = render_native_preview_stats(
                &cpu_engine, &cpu_cogs, full_bbox, sz, sz, &single_req,
                ResampleMode::Bilinear, None, false, "b1", &stats_opts,
            ).await.unwrap();
            ct.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt = Vec::new();
        let ds = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = render_native_preview_stats(
                &gpu_engine, &gpu_cogs, full_bbox, sz, sz, &single_req,
                ResampleMode::Bilinear, None, false, "b1", &stats_opts,
            ).await.unwrap();
            gt.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let du = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - ds;
        let cm = median_ms(ct);
        let gm = median_ms(gt);
        println!("| `render_native_preview_stats` Single Bilinear | {sz}×{sz} | {cm:.2} | {gm:.2} | {:.1}× | {du} |", cm / gm);
    }

    // 6. /cog/preview RGB Bilinear RawRgba 4096x4096
    {
        let sz = 4096u32;
        let cpu_raw = render_native_preview(
            &cpu_engine, &cpu_cogs, full_bbox, sz, sz, &rgb_req,
            ResampleMode::Bilinear, OutputFormat::RawRgba, None, None, None, true, false,
        ).await.unwrap();
        let gpu_raw = render_native_preview(
            &gpu_engine, &gpu_cogs, full_bbox, sz, sz, &rgb_req,
            ResampleMode::Bilinear, OutputFormat::RawRgba, None, None, None, true, false,
        ).await.unwrap();
        assert_rgba_parity("/cog/preview RGB Bilinear 4096", &cpu_raw, &gpu_raw);
        let mut ct = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = render_native_preview(
                &cpu_engine, &cpu_cogs, full_bbox, sz, sz, &rgb_req,
                ResampleMode::Bilinear, OutputFormat::RawRgba, None, None, None, true, false,
            ).await.unwrap();
            ct.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gt = Vec::new();
        let ds = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = render_native_preview(
                &gpu_engine, &gpu_cogs, full_bbox, sz, sz, &rgb_req,
                ResampleMode::Bilinear, OutputFormat::RawRgba, None, None, None, true, false,
            ).await.unwrap();
            gt.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let du = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - ds;
        let cm = median_ms(ct);
        let gm = median_ms(gt);
        println!("| `/cog/preview` RGB Bilinear RawRgba | {sz}×{sz} | {cm:.2} | {gm:.2} | {:.1}× | {du} |", cm / gm);
    }

    let _ = std::fs::remove_file(&tmp);
    println!("\nALL LAZY SEPARABLE BBOX PARITY CHECKS PASSED!");
}
