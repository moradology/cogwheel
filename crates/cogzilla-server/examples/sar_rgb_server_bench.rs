//! End-to-end Sentinel-1 Dual-Pol `SarRgb` (`[VV_dB, VH_dB, VV_dB - VH_dB]`) GPU vs CPU benchmark.
//!
//! Benchmarks `titiler::render_native_preview` and `titiler::render_wgs84_bbox` on a 2-band
//! 1024x1024 Planar Float32 SAR COG (with and without fused `ColorFormula`: Gamma+Sigmoidal+LCH Sat).

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColorFormula, OutputFormat, ResampleMode};
use cogzilla_server::engine::{BandRequest, TileEngine};
use cogzilla_server::titiler::{render_native_preview, render_wgs84_bbox};

fn build_2band_sar_1024_cog_bytes() -> Vec<u8> {
    let w: u32 = 1024;
    let h: u32 = 1024;
    let tw: u32 = 512;
    let th: u32 = 512;
    let total_tiles = ((w / tw) * (h / th) * 2) as usize;
    let tile_bytes = (tw * th * 4) as usize;

    let mut tile_data: Vec<Vec<u8>> = Vec::with_capacity(total_tiles);
    for band in 0..2u32 {
        for ty in 0..(h / th) {
            for tx in 0..(w / tw) {
                let mut raw = Vec::with_capacity(tile_bytes);
                for py in 0..th {
                    let gy = ty * th + py;
                    for px in 0..tw {
                        let gx = tx * tw + px;
                        let fx = gx as f32 / w as f32;
                        let fy = gy as f32 / h as f32;
                        let v = if band == 0 {
                            1000.0 + 600.0 * (fx * 11.0).sin() + 300.0 * (fy * 8.0).cos()
                        } else {
                            250.0 + 150.0 * (fx * 7.0 + 0.5).cos() + 80.0 * (fy * 9.0).sin()
                        };
                        raw.extend_from_slice(&v.max(1.0).to_le_bytes());
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
        (258, 3, 2, (32u32) | (32u32 << 16)),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (277, 3, 1, 2),
        (284, 3, 1, 2),
        (322, 4, 1, tw),
        (323, 4, 1, th),
        (324, 4, total_tiles as u32, offsets_pos),
        (325, 4, total_tiles as u32, counts_pos),
        (339, 3, 2, (3u32) | (3u32 << 16)),
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

#[tokio::main]
async fn main() {
    let tmp = std::env::temp_dir().join("cogzilla_sar_rgb_server_bench.tif");
    std::fs::write(&tmp, build_2band_sar_1024_cog_bytes()).expect("write temp COG");

    let cpu_store = Arc::new(CogStore::new(256 * 1024 * 1024));
    let cpu_cog = cpu_store.register_local("sar", &tmp).expect("open cpu cog");
    let mut cpu_engine = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_engine.gpu = None;

    let gpu_store = Arc::new(CogStore::new(256 * 1024 * 1024));
    let gpu_cog = gpu_store.register_local("sar", &tmp).expect("open gpu cog");
    let gpu_engine = TileEngine::with_store_and_tile_cache(gpu_store, 0);
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };
    println!(
        "Sentinel-1 Dual-Pol SAR RGB GPU backend: {} ({:?}, floor={:.1} us)\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );

    let cpu_cogs = [Arc::clone(&cpu_cog)];
    let gpu_cogs = [Arc::clone(&gpu_cog)];
    let full_bbox = [-10.0, 35.0, -8.0, 37.0];
    let sub_bbox = [-9.7, 35.3, -8.3, 36.7];
    let sar_bands = BandRequest::SarRgb {
        bands_1based: [1, 2],
        rescale_min: [15.0, 10.0, 0.0],
        rescale_max: [35.0, 30.0, 15.0],
    };
    let cf = ColorFormula::parse("Gamma RGB 1.4 Sigmoidal RGB 12 0.4 Saturation 1.2").unwrap();

    println!("| Dual-Pol SAR RGB Route (1024x1024 End-to-End PNG) | CPU (ms) | GPU (ms) | Speedup | GPU Dispatches |");
    println!("| :--- | ---: | ---: | ---: | ---: |");

    let cases: [(&str, bool, [f64; 4], Option<&ColorFormula>); 3] = [
        ("/cog/preview SarRgb Lanczos3 PNG", true, full_bbox, None),
        ("/cog/bbox (Exact2D) SarRgb Lanczos3 PNG", false, sub_bbox, None),
        ("/cog/preview SarRgb+ColorFormula Lanczos3", true, full_bbox, Some(&cf)),
    ];

    let iters = 4usize;
    for (name, native_preview, bbox, color_formula) in cases {
        if native_preview {
            let _ = render_native_preview(
                &cpu_engine, &cpu_cogs, bbox, 1024, 1024, &sar_bands, ResampleMode::Lanczos,
                OutputFormat::Png, None, None, color_formula, true, false,
            ).await.expect("cpu warmup");
            let _ = render_native_preview(
                &gpu_engine, &gpu_cogs, bbox, 1024, 1024, &sar_bands, ResampleMode::Lanczos,
                OutputFormat::Png, None, None, color_formula, true, false,
            ).await.expect("gpu warmup");
        } else {
            let _ = render_wgs84_bbox(
                &cpu_engine, &cpu_cogs, bbox, 1024, 1024, None, &sar_bands, ResampleMode::Lanczos,
                OutputFormat::Png, None, None, color_formula, true, false,
            ).await.expect("cpu warmup");
            let _ = render_wgs84_bbox(
                &gpu_engine, &gpu_cogs, bbox, 1024, 1024, None, &sar_bands, ResampleMode::Lanczos,
                OutputFormat::Png, None, None, color_formula, true, false,
            ).await.expect("gpu warmup");
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let out = if native_preview {
                render_native_preview(
                    &cpu_engine, &cpu_cogs, bbox, 1024, 1024, &sar_bands, ResampleMode::Lanczos,
                    OutputFormat::Png, None, None, color_formula, true, false,
                ).await.expect("cpu render")
            } else {
                render_wgs84_bbox(
                    &cpu_engine, &cpu_cogs, bbox, 1024, 1024, None, &sar_bands, ResampleMode::Lanczos,
                    OutputFormat::Png, None, None, color_formula, true, false,
                ).await.expect("cpu render")
            };
            assert!(out.starts_with(b"\x89PNG"));
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let d0 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        let t1 = Instant::now();
        for _ in 0..iters {
            let out = if native_preview {
                render_native_preview(
                    &gpu_engine, &gpu_cogs, bbox, 1024, 1024, &sar_bands, ResampleMode::Lanczos,
                    OutputFormat::Png, None, None, color_formula, true, false,
                ).await.expect("gpu render")
            } else {
                render_wgs84_bbox(
                    &gpu_engine, &gpu_cogs, bbox, 1024, 1024, None, &sar_bands, ResampleMode::Lanczos,
                    OutputFormat::Png, None, None, color_formula, true, false,
                ).await.expect("gpu render")
            };
            assert!(out.starts_with(b"\x89PNG"));
        }
        let gpu_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let dispatches = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - d0;

        println!(
            "| {:<49} | {:>8.2} | {:>8.2} | {:>6.2}x | {:>14} |",
            name, cpu_ms, gpu_ms, cpu_ms / gpu_ms, dispatches
        );
    }

    let _ = std::fs::remove_file(&tmp);
}
