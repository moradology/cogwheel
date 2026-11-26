//! Benchmark for fused GPU 3-expression `BandRequest::CustomExprRgb` + `ColorFormula`
//! on TiTiler `/cog/preview` crops (1024x1024 and 2048x2048 Lanczos3 PNG).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use cogzilla_io::CogStore;
use cogzilla_kernel::{ColorFormula, ExprProgram, OutputFormat, ResampleMode};
use cogzilla_server::engine::{BandRequest, TileEngine};
use cogzilla_server::titiler::render_native_preview;

fn build_512_3band_cog_bytes() -> Vec<u8> {
    let w: u32 = 512;
    let h: u32 = 512;
    let spp: u32 = 3;
    let tile_bytes = (w * h * spp * 4) as usize;
    let mut tile_data = Vec::with_capacity(tile_bytes);
    for y in 0..h {
        for x in 0..w {
            let b0 = 0.25 + 0.15 * (x as f32 * 0.025).sin().abs();
            let b1 = 0.15 + 0.10 * (y as f32 * 0.03).cos().abs();
            let b2 = 0.10 + 0.08 * ((x + y) as f32 * 0.02).sin().abs();
            tile_data.extend_from_slice(&b0.to_le_bytes());
            tile_data.extend_from_slice(&b1.to_le_bytes());
            tile_data.extend_from_slice(&b2.to_le_bytes());
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
    let bps_off = write_extra(&mut buf, &[32, 0, 32, 0, 32, 0]);
    let sf_off = write_extra(&mut buf, &[3, 0, 3, 0, 3, 0]);

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
        (262, 3, 1, 2),
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
    let tmp = std::env::temp_dir().join("cogzilla_expr_rgb_bench.tif");
    std::fs::write(&tmp, build_512_3band_cog_bytes()).expect("write COG");

    let cpu_store = Arc::new(CogStore::new(64 * 1024 * 1024));
    let cpu_cog = cpu_store.register_local("er", &tmp).expect("open cpu");
    let mut cpu_engine = TileEngine::with_store_and_tile_cache(cpu_store, 0);
    cpu_engine.gpu = None;

    let gpu_store = Arc::new(CogStore::new(64 * 1024 * 1024));
    let gpu_cog = gpu_store.register_local("er", &tmp).expect("open gpu");
    let gpu_engine = TileEngine::with_store_and_tile_cache(gpu_store, 0);
    let Some(ref gpu) = gpu_engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };

    let cf = ColorFormula::parse("Gamma RGB 1.5 Sigmoidal RGB 6 0.5 Saturation 1.3").unwrap();
    let boost = BandRequest::CustomExprRgb {
        progs: [
            ExprProgram::compile("b1*2.5").unwrap(),
            ExprProgram::compile("b2*2.5").unwrap(),
            ExprProgram::compile("b3*2.5").unwrap(),
        ],
        rescale_min: [0.0, 0.0, 0.0],
        rescale_max: [1.0, 1.0, 1.0],
    };
    let false_color = BandRequest::CustomExprRgb {
        progs: [
            ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap(),
            ExprProgram::compile("b2*2.5").unwrap(),
            ExprProgram::compile("sqrt(b3)").unwrap(),
        ],
        rescale_min: [-1.0, 0.0, 0.0],
        rescale_max: [1.0, 1.0, 1.0],
    };

    println!(
        "CustomExprRgb + ColorFormula GPU backend: {} ({:?}, floor={:.1} us)\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.policy.profile.dispatch_floor_us
    );
    println!("| TiTiler CustomExprRgb /cog/preview (Lanczos3 PNG) | Size | CPU (ms) | GPU (ms) | Speedup |");
    println!("| :--- | :--- | ---: | ---: | ---: |");

    let cases: Vec<(&str, u32, &BandRequest, Option<&ColorFormula>)> = vec![
        ("True-Color Boost (b1*2.5;b2*2.5;b3*2.5)", 1024, &boost, None),
        ("False-Color (NDVI;b2*2.5;sqrt(b3))", 1024, &false_color, None),
        ("False-Color + ColorFormula", 1024, &false_color, Some(&cf)),
        ("False-Color + ColorFormula", 2048, &false_color, Some(&cf)),
    ];

    let world_half = 20_037_508.342_789_244_f64;
    let span = (2.0 * world_half) / 128.0;
    let bbox = [0.0, -span, span, 0.0];
    let cpu_cogs = [Arc::clone(&cpu_cog)];
    let gpu_cogs = [Arc::clone(&gpu_cog)];
    let iters = 3usize;

    for (name, sz, bands, formula) in cases {
        let _ = render_native_preview(
            &cpu_engine, &cpu_cogs, bbox, sz, sz, bands, ResampleMode::Lanczos,
            OutputFormat::Png, None, None, formula, true, false,
        ).await.unwrap();
        let _ = render_native_preview(
            &gpu_engine, &gpu_cogs, bbox, sz, sz, bands, ResampleMode::Lanczos,
            OutputFormat::Png, None, None, formula, true, false,
        ).await.unwrap();

        let t0 = Instant::now();
        for _ in 0..iters {
            let out = render_native_preview(
                &cpu_engine, &cpu_cogs, bbox, sz, sz, bands, ResampleMode::Lanczos,
                OutputFormat::Png, None, None, formula, true, false,
            ).await.unwrap();
            assert!(out.starts_with(b"\x89PNG"));
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let d0 = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed);
        let t0 = Instant::now();
        for _ in 0..iters {
            let out = render_native_preview(
                &gpu_engine, &gpu_cogs, bbox, sz, sz, bands, ResampleMode::Lanczos,
                OutputFormat::Png, None, None, formula, true, false,
            ).await.unwrap();
            assert!(out.starts_with(b"\x89PNG"));
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let dispatches = gpu_engine.stats.gpu_dispatches.load(Ordering::Relaxed) - d0;
        assert_eq!(dispatches, iters as u64, "{name} should dispatch to GPU");

        println!("| {name} | {sz}x{sz} | {cpu_ms:.2} | {gpu_ms:.2} | {:.2}x |", cpu_ms / gpu_ms);
    }

    let _ = std::fs::remove_file(tmp);
}
