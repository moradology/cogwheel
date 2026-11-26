//! Benchmark for GPU-accelerated scientific multi-band `.npy` / RawF32 ML tensor cubes
//! (`BandRequest::MultiBand` 4-band & 6-band, `BandRequest::Rgb`, and `BandRequest::CustomExprRgb`).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{ExprProgram, OutputFormat, ResampleMode};
use cogzilla_server::engine::{BandRequest, TileEngine};
use cogzilla_server::titiler::render_native_preview;

fn build_512_6band_cog_bytes() -> Vec<u8> {
    let w: u32 = 512;
    let h: u32 = 512;
    let spp: u32 = 6;
    let tile_bytes = (w * h * spp * 4) as usize;
    let mut tile_data = Vec::with_capacity(tile_bytes);
    for y in 0..h {
        for x in 0..w {
            for b in 0..spp {
                let v = 0.10
                    + 0.04 * (b as f32)
                    + 0.12 * (x as f32 * 0.02 + b as f32 * 0.7).sin().abs()
                    + 0.08 * (y as f32 * 0.03 + b as f32 * 0.4).cos().abs();
                tile_data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    let world_half = 20_037_508.342_789_244_f64;
    let span = (2.0 * world_half) / 128.0;

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
    let bps: Vec<u8> = (0..spp).flat_map(|_| 32u16.to_le_bytes()).collect();
    let sf: Vec<u8> = (0..spp).flat_map(|_| 3u16.to_le_bytes()).collect();
    let bps_off = write_extra(&mut buf, &bps);
    let sf_off = write_extra(&mut buf, &sf);

    let mut scale_bytes = Vec::with_capacity(24);
    scale_bytes.extend_from_slice(&(span / w as f64).to_le_bytes());
    scale_bytes.extend_from_slice(&(span / h as f64).to_le_bytes());
    scale_bytes.extend_from_slice(&0.0f64.to_le_bytes());
    let scale_off = write_extra(&mut buf, &scale_bytes);

    let mut tie_bytes = Vec::with_capacity(48);
    for v in [0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0] {
        tie_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let tie_off = write_extra(&mut buf, &tie_bytes);

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
        (258, 3, spp, bps_off),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (277, 3, 1, spp),
        (284, 3, 1, 1),
        (322, 4, 1, w),
        (323, 4, 1, h),
        (324, 4, 1, tile_off),
        (325, 4, 1, tile_bytes as u32),
        (339, 3, spp, sf_off),
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
    let tmp = std::env::temp_dir().join("cogzilla_multiband_npy_bench.tif");
    std::fs::write(&tmp, build_512_6band_cog_bytes()).expect("write COG");

    let cpu_store = Arc::new(CogStore::new(128 * 1024 * 1024));
    let cpu_cog = cpu_store.register_local("mb", &tmp).expect("open cpu");
    let mut cpu_engine = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_engine.gpu = None;

    let gpu_store = Arc::new(CogStore::new(128 * 1024 * 1024));
    let gpu_cog = gpu_store.register_local("mb", &tmp).expect("open gpu");
    let gpu_engine = TileEngine::with_store_and_tile_cache(gpu_store, 0);
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };

    let mb4 = BandRequest::MultiBand {
        bands_1based: vec![1, 2, 3, 4],
    };
    let mb6 = BandRequest::MultiBand {
        bands_1based: vec![1, 2, 3, 4, 5, 6],
    };
    let rgb = BandRequest::Rgb {
        bands_1based: [1, 2, 3],
        rescale_min: [0.0; 3],
        rescale_max: [1.0; 3],
    };
    let expr_rgb = BandRequest::CustomExprRgb {
        progs: [
            ExprProgram::compile("(b4-b1)/(b4+b1)").unwrap(),
            ExprProgram::compile("b2*2.5").unwrap(),
            ExprProgram::compile("sqrt(b3)").unwrap(),
        ],
        rescale_min: [-1.0, 0.0, 0.0],
        rescale_max: [1.0, 1.0, 1.0],
    };

    println!(
        "Multi-Band Scientific .npy ML Tensor GPU backend: {} ({:?}, floor={:.1} us)\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );
    println!("| Scientific .npy Tensor Cube (1024x1024 Lanczos3) | Shape | CPU (ms) | GPU (ms) | Speedup |");
    println!("| :--- | :--- | ---: | ---: | ---: |");

    let cases: Vec<(&str, &str, &BandRequest)> = vec![
        ("Rgb (B04/B03/B02) .npy", "(3, 1024, 1024)", &rgb),
        ("MultiBand 4-ch (RGBN) .npy", "(4, 1024, 1024)", &mb4),
        ("MultiBand 6-ch (S2 ML) .npy", "(6, 1024, 1024)", &mb6),
        ("CustomExprRgb (NDVI/G/sqrtB) .npy", "(3, 1024, 1024)", &expr_rgb),
    ];

    let world_half = 20_037_508.342_789_244_f64;
    let span = (2.0 * world_half) / 128.0;
    let bbox = [0.0, -span, span, 0.0];
    let cpu_cogs = [Arc::clone(&cpu_cog)];
    let gpu_cogs = [Arc::clone(&gpu_cog)];
    let iters = 3usize;

    for (name, shape, bands) in cases {
        let _ = render_native_preview(
            &cpu_engine, &cpu_cogs, bbox, 1024, 1024, bands, ResampleMode::Lanczos,
            OutputFormat::Npy, None, None, None, true, false,
        ).await.unwrap();
        let _ = render_native_preview(
            &gpu_engine, &gpu_cogs, bbox, 1024, 1024, bands, ResampleMode::Lanczos,
            OutputFormat::Npy, None, None, None, true, false,
        ).await.unwrap();

        let t0 = Instant::now();
        for _ in 0..iters {
            let out = render_native_preview(
                &cpu_engine, &cpu_cogs, bbox, 1024, 1024, bands, ResampleMode::Lanczos,
                OutputFormat::Npy, None, None, None, true, false,
            ).await.unwrap();
            assert!(out.starts_with(b"\x93NUMPY"));
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let d0 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        let t0 = Instant::now();
        for _ in 0..iters {
            let out = render_native_preview(
                &gpu_engine, &gpu_cogs, bbox, 1024, 1024, bands, ResampleMode::Lanczos,
                OutputFormat::Npy, None, None, None, true, false,
            ).await.unwrap();
            assert!(out.starts_with(b"\x93NUMPY"));
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let dispatches = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - d0;
        assert_eq!(dispatches, iters as u64, "{name} should dispatch to GPU");

        println!("| {name} | {shape} | {cpu_ms:.2} | {gpu_ms:.2} | {:.2}x |", cpu_ms / gpu_ms);
    }

    let _ = std::fs::remove_file(tmp);
}
