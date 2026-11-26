//! End-to-end `TileEngine::render_tiles_batch` 3D GPU micro-batching benchmark.
//!
//! Compares CPU `render_tile` vs GPU `render_tile` (K=1 per dispatch) vs GPU `render_tiles_batch`
//! (1 single 3D GPU dispatch for K=16 or K=32 viewport tiles sharing a COG overview block).

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, OutputFormat, ResampleMode};
use cogzilla_proj::TileCoord;
use cogzilla_server::engine::{BandRequest, TileEngine, TileRequest};

/// Build a 512x512 single-block Float32 EPSG:3857 COG covering WebMercator z=8 x=128..132 y=128..132
/// so a 4x4 (16-tile) or 8x4 (32-tile) z=10/z=11 viewport burst maps into the same 512x512 slab.
fn build_512_mercator_cog_bytes() -> Vec<u8> {
    let w: u32 = 512;
    let h: u32 = 512;
    let tile_bytes = (w * h * 4) as usize;
    let mut tile_data = Vec::with_capacity(tile_bytes);
    for y in 0..h {
        for x in 0..w {
            let v = 1000.0
                + 600.0 * (x as f32 * 0.025).sin()
                + 450.0 * (y as f32 * 0.03).cos();
            tile_data.extend_from_slice(&v.to_le_bytes());
        }
    }

    // Cover a 4x4 block of z=10 tiles starting at x=512, y=512 (WebMercator origin = 0, 0)
    let world_half = 20_037_508.342_789_244_f64;
    let tile_span_z10 = (2.0 * world_half) / 1024.0;
    let span = tile_span_z10 * 8.0; // covers x=512..520, y=512..520 at z=10 (64 tiles!)

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
    scale_bytes.extend_from_slice(&(span / w as f64).to_le_bytes());
    scale_bytes.extend_from_slice(&(span / h as f64).to_le_bytes());
    scale_bytes.extend_from_slice(&0.0f64.to_le_bytes());
    let scale_off = write_extra(&mut buf, &scale_bytes);

    let mut tie_bytes = Vec::with_capacity(48);
    // Top-left at WebMercator (0.0, 0.0) -> tile z=10 x=512, y=512
    for v in [0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0] {
        tie_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let tie_off = write_extra(&mut buf, &tie_bytes);

    // EPSG:3857 Projected
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
        (258, 3, 1, 32),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (277, 3, 1, 1),
        (284, 3, 1, 1),
        (322, 4, 1, w),
        (323, 4, 1, h),
        (324, 4, 1, tile_off),
        (325, 4, 1, tile_bytes as u32),
        (339, 3, 1, 3),
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
    let tmp = std::env::temp_dir().join("cogzilla_batch_server_bench.tif");
    std::fs::write(&tmp, build_512_mercator_cog_bytes()).expect("write temp COG");

    let cpu_store = Arc::new(CogStore::new(64 * 1024 * 1024));
    let cpu_cog = cpu_store.register_local("b", &tmp).expect("open cpu cog");
    let mut cpu_engine = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_engine.gpu = None;

    let gpu_store = Arc::new(CogStore::new(64 * 1024 * 1024));
    let gpu_cog = gpu_store.register_local("b", &tmp).expect("open gpu cog");
    let gpu_engine = TileEngine::with_store_and_tile_cache(gpu_store, 0);
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };
    println!(
        "TileEngine Batch GPU backend: {} ({:?}, floor={:.1} us)\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );
    println!("| Viewport Batch (256x256 tiles) | K | CPU (ms) | GPU K=1 (ms) | GPU Batch (ms) | Batch vs CPU | Batch Tiles/s | GPU Dispatches |");
    println!("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");

    let cases: [(&str, usize, ResampleMode, OutputFormat); 4] = [
        ("Cubic RawRgba", 32, ResampleMode::Cubic, OutputFormat::RawRgba),
        ("Cubic PNG", 32, ResampleMode::Cubic, OutputFormat::Png),
        ("Lanczos3 RawRgba", 32, ResampleMode::Lanczos, OutputFormat::RawRgba),
        ("Lanczos3 PNG", 32, ResampleMode::Lanczos, OutputFormat::Png),
    ];

    let iters = 5usize;
    for (name, k, resample, format) in cases {
        let mut reqs = Vec::with_capacity(k);
        for i in 0..k {
            let dx = (i % 8) as u32;
            let dy = (i / 8) as u32;
            reqs.push(TileRequest {
                tile: TileCoord {
                    z: 10,
                    x: 512 + dx,
                    y: 512 + dy,
                },
                tile_size: 256,
                bands: BandRequest::Single {
                    band_1based: 1,
                    rescale: (0.0, 2000.0),
                    colormap: ColormapName::Viridis,
                },
                resample,
                format,
                unscale: false,
            });
        }

        // Warmup
        let _ = cpu_engine.render_tile(&cpu_cog, &reqs[0]).await.unwrap();
        let _ = gpu_engine.render_tiles_batch(&gpu_cog, &reqs).await;

        let t0 = Instant::now();
        for _ in 0..iters {
            for req in &reqs {
                let b = cpu_engine.render_tile(&cpu_cog, req).await.unwrap();
                assert!(b.len() > 16);
            }
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let t1 = Instant::now();
        for _ in 0..iters {
            for req in &reqs {
                let b = gpu_engine.render_tile(&gpu_cog, req).await.unwrap();
                assert!(b.len() > 16);
            }
        }
        let gpu_k1_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let d0 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        let t2 = Instant::now();
        for _ in 0..iters {
            let out = gpu_engine.render_tiles_batch(&gpu_cog, &reqs).await;
            assert_eq!(out.len(), k);
            for r in out {
                assert!(r.unwrap().len() > 16);
            }
        }
        let gpu_batch_ms = t2.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let dispatches = (gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - d0) / iters as u64;
        let tiles_per_s = (k as f64) / (gpu_batch_ms * 1e-3);

        println!(
            "| {:<30} | {:>2} | {:>8.2} | {:>12.2} | {:>14.2} | {:>11.2}x | {:>13.0} | {:>14} |",
            name, k, cpu_ms, gpu_k1_ms, gpu_batch_ms, cpu_ms / gpu_batch_ms, tiles_per_s, dispatches
        );
    }

    let _ = std::fs::remove_file(&tmp);
}
