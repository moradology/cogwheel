use std::time::{Duration, Instant};

use cogzilla_codec::predictor::reconstruct_predictor3;
use cogzilla_gpu::{GpuContext, GpuPredictorDecoder, GpuPredictorKind};

fn encode_pred3_block(w: usize, h: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        let mut row = vec![0u8; w * 4];
        for x in 0..w {
            let v = 1000.0 + (x as f32) * 0.5 + (y as f32) * 0.25;
            let b = v.to_be_bytes();
            row[x] = b[0];
            row[w + x] = b[1];
            row[2 * w + x] = b[2];
            row[3 * w + x] = b[3];
        }
        for i in (1..row.len()).rev() {
            row[i] = row[i].wrapping_sub(row[i - 1]);
        }
        raw.extend_from_slice(&row);
    }
    raw
}

fn median_us(mut v: Vec<Duration>) -> f64 {
    v.sort();
    v[v.len() / 2].as_secs_f64() * 1e6
}

fn main() {
    let ctx = GpuContext::new_default().expect("failed to initialize GPU context");
    println!(
        "Selected GPU [{}]: {} ({:?}, unified={})\n",
        ctx.info.index, ctx.info.name, ctx.info.backend, ctx.info.unified_memory
    );

    let mut dec = GpuPredictorDecoder::new(&ctx);
    let w = 512u32;
    let h = 512u32;
    let one_block = encode_pred3_block(w as usize, h as usize);

    println!(
        "| Batch (512x512 f32) | Data (MiB) | CPU SIMD (us) | GPU Upload+Unshuf (us) | GPU VRAM-Only (us) | VRAM GB/s | VRAM Speedup |"
    );
    println!("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |");

    for &batch in &[1u32, 4, 16, 32, 64] {
        let mut raw_batch = Vec::with_capacity(one_block.len() * batch as usize);
        for _ in 0..batch {
            raw_batch.extend_from_slice(&one_block);
        }
        let mib = (raw_batch.len() as f64) / (1024.0 * 1024.0);
        let iters = if batch <= 16 { 9 } else { 5 };

        // CPU SIMD baseline (sequential 1-core reconstruct_predictor3 per block)
        let mut cpu_work = raw_batch.clone();
        let mut cpu_t = Vec::with_capacity(iters);
        for _ in 0..iters {
            cpu_work.copy_from_slice(&raw_batch);
            let t0 = Instant::now();
            for b in 0..batch as usize {
                let start = b * one_block.len();
                let end = start + one_block.len();
                let bands = reconstruct_predictor3(&mut cpu_work[start..end], w, h, 1, None);
                std::hint::black_box(bands.len());
            }
            cpu_t.push(t0.elapsed());
        }
        let cpu_us = median_us(cpu_t);

        // GPU Upload + Unshuffle
        dec.decode_batch_resident(&raw_batch, w, h, batch, GpuPredictorKind::Predictor3, None)
            .unwrap();
        let mut gpu_up_t = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            dec.decode_batch_resident(&raw_batch, w, h, batch, GpuPredictorKind::Predictor3, None)
                .unwrap();
            gpu_up_t.push(t0.elapsed());
        }
        let gpu_up_us = median_us(gpu_up_t);

        // GPU VRAM-Only (post-nvCOMP scenario: raw bytes already in VRAM)
        let mut gpu_vram_t = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            dec.dispatch_uploaded_only(w, h, batch, GpuPredictorKind::Predictor3);
            gpu_vram_t.push(t0.elapsed());
        }
        let gpu_vram_us = median_us(gpu_vram_t);
        let vram_gb_s = (raw_batch.len() as f64 / 1e9) / (gpu_vram_us / 1e6);

        println!(
            "| {:>2} blocks           | {:>10.1} | {:>13.1} | {:>22.1} | {:>18.1} | {:>9.2} | {:>11.2}x |",
            batch,
            mib,
            cpu_us,
            gpu_up_us,
            gpu_vram_us,
            vram_gb_s,
            cpu_us / gpu_vram_us
        );
    }
}
