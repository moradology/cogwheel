//! Benchmark & parity verification: GPU WGSL two-pass statistics + 1024-bin shared-memory
//! atomic histogram + percentile CDF (`GpuStatsReducer`) vs CPU TiTiler
//! `compute_f32_band_stats` (`sort_unstable_by` + multi-pass stats/histogram).

use std::time::Instant;

use cogzilla_gpu::{GpuContext, GpuOptions, GpuResampler, GpuStatsReducer};
use cogzilla_server::titiler::{StatsOptions, compute_f32_band_stats};

fn make_dem_band(w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i % 31 == 0 {
            out.push(f32::NAN);
        } else {
            let x = (i % w) as f32 / w as f32;
            let y = (i / w) as f32 / h as f32;
            let elev = 1200.0
                + 950.0 * (x * 13.0).sin() * (y * 11.0).cos()
                + 320.0 * ((x + y) * 29.0).sin()
                + 85.0 * (x * 71.0).cos();
            out.push(elev);
        }
    }
    out
}

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let ctx = GpuContext::new(GpuOptions::default()).expect("GPU adapter");
    println!(
        "Adapter: {} ({:?}, unified={})",
        ctx.info.name, ctx.info.backend, ctx.info.unified_memory
    );
    let resampler = GpuResampler::new(&ctx);
    let mut reducer = GpuStatsReducer::new(&ctx);
    let opts = StatsOptions {
        percentiles: vec![2.0, 98.0],
        histogram_bins: 10,
        ..StatsOptions::default()
    };

    println!();
    println!(
        "{:<12} {:>10} {:>10} {:>9} {:>10} {:>9} {:>10} {:>10}",
        "Size", "CPU ms", "Host ms", "HostSpd", "Slab ms", "SlabSpd", "p50 err%", "std err%"
    );
    println!("{}", "-".repeat(88));

    for &(w, h, iters) in &[
        (1024usize, 1024usize, 9usize),
        (2048, 2048, 5),
        (4096, 4096, 3),
        (8192, 4096, 3),
    ] {
        let data = make_dem_band(w, h);
        let slab = resampler
            .upload_slab(&data, w as u32, h as u32, w as u32)
            .expect("upload slab");

        // Warmup + parity check
        let cpu_json = compute_f32_band_stats(&data, "b1", &opts);
        let gpu_host = reducer
            .reduce_f32(&data, None, 10, None, &[2.0, 98.0])
            .expect("gpu host stats");
        let gpu_slab = reducer
            .reduce_slab(&slab, None, 10, None, &[2.0, 98.0])
            .expect("gpu slab stats");

        assert_eq!(gpu_host.valid_pixels, gpu_slab.valid_pixels);
        assert_eq!(gpu_host.hist_counts, gpu_slab.hist_counts);
        let cpu_valid = cpu_json["valid_pixels"].as_u64().unwrap() as usize;
        let cpu_min = cpu_json["min"].as_f64().unwrap();
        let cpu_max = cpu_json["max"].as_f64().unwrap();
        let cpu_std = cpu_json["std"].as_f64().unwrap();
        let cpu_p50 = cpu_json["median"].as_f64().unwrap();
        assert_eq!(gpu_host.valid_pixels, cpu_valid);
        assert!((gpu_host.min.unwrap() - cpu_min).abs() < 1e-3);
        assert!((gpu_host.max.unwrap() - cpu_max).abs() < 1e-3);
        let span = (cpu_max - cpu_min).max(1e-9);
        let p50_err_pct = (gpu_host.median.unwrap() - cpu_p50).abs() / span * 100.0;
        let std_err_pct = (gpu_host.std.unwrap() - cpu_std).abs() / cpu_std.max(1e-9) * 100.0;
        assert!(p50_err_pct < 0.25, "p50 error too large: {p50_err_pct:.4}%");
        assert!(std_err_pct < 0.10, "std error too large: {std_err_pct:.4}%");

        let mut cpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = compute_f32_band_stats(&data, "b1", &opts);
            cpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut host_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = reducer
                .reduce_f32(&data, None, 10, None, &[2.0, 98.0])
                .expect("gpu host");
            host_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut slab_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = reducer
                .reduce_slab(&slab, None, 10, None, &[2.0, 98.0])
                .expect("gpu slab");
            slab_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let c = median_ms(cpu_times);
        let gh = median_ms(host_times);
        let gs = median_ms(slab_times);
        println!(
            "{:<12} {:>10.2} {:>10.2} {:>8.2}x {:>10.2} {:>8.2}x {:>9.4}% {:>9.4}%",
            format!("{w}x{h}"),
            c,
            gh,
            c / gh,
            gs,
            c / gs,
            p50_err_pct,
            std_err_pct
        );
    }
}
