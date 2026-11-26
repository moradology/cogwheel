//! End-to-end TiTiler JSON statistics benchmark & full schema/value parity test:
//! `compute_f32_band_stats` (CPU) vs `compute_f32_band_stats_with_engine` (TileEngine + GpuStatsReducer).

use std::time::Instant;

use cogzilla_server::engine::TileEngine;
use cogzilla_server::titiler::{
    StatsOptions, compute_f32_band_stats, compute_f32_band_stats_with_engine,
};

fn make_ndvi_band(w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i % 23 == 0 {
            out.push(f32::NAN);
        } else {
            let x = (i % w) as f32 / w as f32;
            let y = (i / w) as f32 / h as f32;
            let ndvi = 0.25
                + 0.45 * (x * 17.0).sin() * (y * 13.0).cos()
                + 0.15 * ((x + y) * 31.0).cos();
            out.push(ndvi.clamp(-1.0, 1.0));
        }
    }
    out
}

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[tokio::main]
async fn main() {
    let engine = TileEngine::new(64 * 1024 * 1024);
    let gpu = engine.gpu.as_ref().expect("GPU adapter");
    println!(
        "Adapter: {} ({:?}, unified={})",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.ctx.info.unified_memory
    );

    let opts = StatsOptions {
        percentiles: vec![2.0, 98.0],
        histogram_bins: 10,
        ..StatsOptions::default()
    };

    println!();
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Size", "CPU ms", "GPU ms", "Speedup", "p2 err%", "p50 err%", "p98 err%"
    );
    println!("{}", "-".repeat(78));

    for &(w, h, iters) in &[
        (1024usize, 1024usize, 9usize),
        (2048, 2048, 5),
        (4096, 4096, 3),
        (8192, 4096, 3),
    ] {
        let data = make_ndvi_band(w, h);

        // Warmup + full TiTiler JSON schema & value parity check
        let cpu_val = compute_f32_band_stats(&data, "(b4-b3)/(b4+b3)", &opts);
        let gpu_val = compute_f32_band_stats_with_engine(&engine, &data, "(b4-b3)/(b4+b3)", &opts);

        let co = cpu_val.as_object().expect("cpu json obj");
        let go = gpu_val.as_object().expect("gpu json obj");
        assert_eq!(
            co.keys().collect::<Vec<_>>(),
            go.keys().collect::<Vec<_>>(),
            "TiTiler JSON key schema mismatch!"
        );
        assert_eq!(co["valid_pixels"], go["valid_pixels"]);
        assert_eq!(co["masked_pixels"], go["masked_pixels"]);
        assert_eq!(co["valid_percent"], go["valid_percent"]);
        assert_eq!(co["count"], go["count"]);
        assert_eq!(co["description"], go["description"]);

        let span = (co["max"].as_f64().unwrap() - co["min"].as_f64().unwrap()).max(1e-9);
        assert!((co["min"].as_f64().unwrap() - go["min"].as_f64().unwrap()).abs() < 1e-4);
        assert!((co["max"].as_f64().unwrap() - go["max"].as_f64().unwrap()).abs() < 1e-4);
        assert!((co["mean"].as_f64().unwrap() - go["mean"].as_f64().unwrap()).abs() < 1e-3);
        assert!((co["std"].as_f64().unwrap() - go["std"].as_f64().unwrap()).abs() < 1e-3);

        let p2_err = (co["percentile_2"].as_f64().unwrap() - go["percentile_2"].as_f64().unwrap()).abs() / span * 100.0;
        let p50_err = (co["median"].as_f64().unwrap() - go["median"].as_f64().unwrap()).abs() / span * 100.0;
        let p98_err = (co["percentile_98"].as_f64().unwrap() - go["percentile_98"].as_f64().unwrap()).abs() / span * 100.0;
        assert!(p2_err < 0.25 && p50_err < 0.25 && p98_err < 0.25);

        let mut cpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = compute_f32_band_stats(&data, "(b4-b3)/(b4+b3)", &opts);
            cpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = compute_f32_band_stats_with_engine(&engine, &data, "(b4-b3)/(b4+b3)", &opts);
            gpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let c = median_ms(cpu_times);
        let g = median_ms(gpu_times);
        println!(
            "{:<12} {:>10.2} {:>10.2} {:>9.2}x {:>9.4}% {:>9.4}% {:>9.4}%",
            format!("{w}x{h}"),
            c,
            g,
            c / g,
            p2_err,
            p50_err,
            p98_err
        );
    }
}
