//! Benchmark & PSNR parity check: GPU WGSL 4:2:0 FDCT + Annex K Huffman (`GpuJpegEncoder`)
//! vs CPU `cogzilla_kernel::jpeg_fast::encode_jpeg_rgba` (`jpeg-encoder`).

use std::time::Instant;

use cogzilla_gpu::{GpuContext, GpuJpegEncoder, GpuOptions, wgpu};
use cogzilla_kernel::jpeg_fast::encode_jpeg_rgba;

fn make_satellite_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let fy = y as f32 / h as f32;
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let terrain = (fx * 19.0).sin() * (fy * 17.0).cos()
                + 0.4 * ((fx + fy) * 43.0).sin()
                + 0.2 * ((fx * 97.0).cos() * (fy * 89.0).sin());
            let r = (110.0 + 75.0 * terrain).clamp(0.0, 255.0) as u8;
            let g = (130.0 + 65.0 * (terrain + 0.3 * (fy * 11.0).sin())).clamp(0.0, 255.0) as u8;
            let b = (95.0 + 70.0 * (terrain - 0.4 * (fx * 13.0).cos())).clamp(0.0, 255.0) as u8;
            out.extend_from_slice(&[r, g, b, 255]);
        }
    }
    out
}

fn decode_psnr(jpg: &[u8], rgba: &[u8], w: u32, h: u32) -> f64 {
    let mut dec = zune_jpeg::JpegDecoder::new(jpg);
    let rgb = dec.decode().expect("zune-jpeg decode");
    let info = dec.info().expect("jpeg info");
    assert_eq!((info.width as u32, info.height as u32), (w, h));
    let mut mse = 0.0f64;
    for i in 0..(w * h) as usize {
        let dr = rgb[i * 3] as f64 - rgba[i * 4] as f64;
        let dg = rgb[i * 3 + 1] as f64 - rgba[i * 4 + 1] as f64;
        let db = rgb[i * 3 + 2] as f64 - rgba[i * 4 + 2] as f64;
        mse += dr * dr + dg * dg + db * db;
    }
    mse /= w as f64 * h as f64 * 3.0;
    10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn main() {
    let ctx = GpuContext::new(GpuOptions::default()).expect("GPU adapter");
    println!(
        "Adapter: {} ({:?}, unified={})",
        ctx.info.name, ctx.info.backend, ctx.info.unified_memory
    );
    let mut gpu = GpuJpegEncoder::new(&ctx);

    println!();
    println!(
        "{:<10} {:>9} {:>9} {:>8} {:>9} {:>8} {:>9} {:>9} {:>9} {:>9}",
        "Size",
        "CPU ms",
        "GPU ms",
        "Speedup",
        "Dev ms",
        "DevSpd",
        "CPU KB",
        "GPU KB",
        "CPU PSNR",
        "GPU PSNR"
    );
    println!("{}", "-".repeat(102));

    for &(size, iters) in &[(512u32, 15usize), (1024, 9), (2048, 5), (4096, 3)] {
        let rgba = make_satellite_rgba(size, size);
        let q = 85u8;

        // Pre-resident GPU RGBA8 buffer (simulates output of a prior GPU render pass)
        let dev_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench-device-rgba8"),
            size: rgba.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&dev_buf, 0, &rgba);

        // Warmup
        let cpu_jpg = encode_jpeg_rgba(size, size, &rgba, q);
        let gpu_jpg = gpu.encode_rgba8(&rgba, size, size, q).expect("gpu jpeg");
        let dev_jpg = gpu
            .encode_rgba8_from_buffer(&dev_buf, size, size, q)
            .expect("gpu dev jpeg");
        assert_eq!(gpu_jpg, dev_jpg, "host vs device-resident GPU JPEG mismatch");

        let cpu_psnr = decode_psnr(&cpu_jpg, &rgba, size, size);
        let gpu_psnr = decode_psnr(&gpu_jpg, &rgba, size, size);
        assert!(gpu_psnr > 35.0, "GPU JPEG PSNR too low: {gpu_psnr:.2}");

        let mut cpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = encode_jpeg_rgba(size, size, &rgba, q);
            cpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut gpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = gpu.encode_rgba8(&rgba, size, size, q).expect("gpu jpeg");
            gpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut dev_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = gpu
                .encode_rgba8_from_buffer(&dev_buf, size, size, q)
                .expect("gpu dev jpeg");
            dev_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let c = median_ms(cpu_times);
        let g = median_ms(gpu_times);
        let d = median_ms(dev_times);
        println!(
            "{:<10} {:>9.2} {:>9.2} {:>7.2}x {:>9.2} {:>7.2}x {:>9.1} {:>9.1} {:>8.2}dB {:>8.2}dB",
            format!("{size}x{size}"),
            c,
            g,
            c / g,
            d,
            c / d,
            cpu_jpg.len() as f64 / 1024.0,
            gpu_jpg.len() as f64 / 1024.0,
            cpu_psnr,
            gpu_psnr
        );
    }
}
