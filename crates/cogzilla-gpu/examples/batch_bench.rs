use std::time::{Duration, Instant};

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_gpu::{
    DispatchPolicy, DispatchRequest, GpuColormapMode, GpuColormapRenderer, GpuContext,
    GpuPixelOutput, GpuResampler, GpuSlabCache, GpuTileBatcher, SlabKey, TileCoordSlice,
};
use cogzilla_kernel::{BandSource, ColormapName, ResampleMode, sample_cubic, sample_general};

fn make_dem(w: u32, h: u32) -> Vec<f32> {
    let mut data = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        let fy = y as f32 * 0.013;
        for x in 0..w {
            let fx = x as f32 * 0.017;
            data[(y * w + x) as usize] = 1200.0 + 350.0 * fx.sin() * fy.cos() + 40.0 * (fx + fy).sin();
        }
    }
    data
}

fn median_us(mut samples: Vec<Duration>) -> f64 {
    samples.sort();
    samples[samples.len() / 2].as_secs_f64() * 1e6
}

fn bench_cpu_tile(
    bs: &BandSource<'_>,
    u_coords: &[f32],
    v_coords: &[f32],
    mode: ResampleMode,
    rmin: f32,
    rmax: f32,
    iters: usize,
) -> f64 {
    let n = u_coords.len();
    let inv254 = 254.0 / (rmax - rmin);
    let mut out = vec![0u8; n * n];
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for (py, &v) in v_coords.iter().enumerate() {
            let row = &mut out[py * n..(py + 1) * n];
            for (px, &u) in u_coords.iter().enumerate() {
                let val = match mode {
                    ResampleMode::Cubic => sample_cubic(bs, u, v),
                    other => sample_general(bs, u, v, other),
                };
                row[px] = if val.is_finite() {
                    1 + ((val - rmin) * inv254).clamp(0.0, 254.0) as u8
                } else {
                    0
                };
            }
        }
        std::hint::black_box(out[0]);
        times.push(t0.elapsed());
    }
    median_us(times)
}

fn main() {
    let ctx = GpuContext::new_default().expect("failed to initialize GPU context");
    let policy = DispatchPolicy::calibrate(&ctx);
    println!(
        "Selected GPU [{}]: {} ({:?}, floor={:.1} us, up={:.2} GB/s, down={:.2} GB/s)\n",
        ctx.info.index,
        ctx.info.name,
        ctx.info.backend,
        policy.profile.dispatch_floor_us,
        policy.profile.upload_gb_s,
        policy.profile.readback_gb_s
    );

    let resampler = GpuResampler::new(&ctx);
    let cache = GpuSlabCache::new(256 * 1024 * 1024);
    let mut batcher = GpuTileBatcher::new(&ctx);

    let src_w = 2048u32;
    let src_h = 2048u32;
    let src = make_dem(src_w, src_h);

    let mut slab_cpu = AlignedF32Slab::new_uninit((src_w * src_h) as usize);
    slab_cpu.as_mut_slice().copy_from_slice(&src);
    let band = DecodedBand {
        data: BandData::F32(slab_cpu),
        width: src_w,
        height: src_h,
        stats: BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 800.0,
            max: 1600.0,
        },
    };
    let blocks = [Some(&band)];
    let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);

    let key = SlabKey {
        dataset_id: 42,
        ifd_idx: 0,
        band: 0,
        tx: 0,
        ty: 0,
    };
    let gpu_slab = cache
        .get_or_upload(key, &src, src_w, src_h, src_w, &resampler)
        .unwrap();

    let lut = ColormapName::Viridis.lut();
    let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
    let rmin = 800.0f32;
    let rmax = 1600.0f32;

    println!(
        "| Mode | Tile | Batch K | CPU 1-core (us/tile) | GPU Batch (us/tile) | GPU Tiles/s | Speedup vs 1-core | Policy |"
    );
    println!("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |");

    for &tile_n in &[256usize, 512] {
        // Prepare 64 distinct tile coordinate windows covering the 2048x2048 slab
        let mut u_vecs = Vec::with_capacity(64);
        let mut v_vecs = Vec::with_capacity(64);
        for ty in 0..8 {
            for tx in 0..8 {
                let u0 = 8.0 + (tx as f32) * 220.0;
                let v0 = 8.0 + (ty as f32) * 220.0;
                let step = 220.0 / (tile_n as f32);
                u_vecs.push((0..tile_n).map(|i| u0 + i as f32 * step).collect::<Vec<f32>>());
                v_vecs.push((0..tile_n).map(|i| v0 + i as f32 * step).collect::<Vec<f32>>());
            }
        }

        for &(mode_name, cpu_mode, gpu_mode) in &[
            ("Cubic", ResampleMode::Cubic, GpuColormapMode::Cubic),
            ("Lanczos3", ResampleMode::Lanczos, GpuColormapMode::Lanczos3),
        ] {
            let cpu_us = bench_cpu_tile(&bs, &u_vecs[0], &v_vecs[0], cpu_mode, rmin, rmax, 7);

            for &k in &[1usize, 4, 16, 32, 64] {
                let slices: Vec<TileCoordSlice<'_>> = u_vecs[..k]
                    .iter()
                    .zip(v_vecs[..k].iter())
                    .map(|(u, v)| TileCoordSlice {
                        u_coords: u,
                        v_coords: v,
                    })
                    .collect();

                // Warmup
                let _ = batcher
                    .render_batch(
                        &gpu_slab,
                        &slices,
                        gpu_mode,
                        rmin,
                        rmax,
                        &lut_u32,
                        GpuPixelOutput::IndexedU8,
                    )
                    .unwrap();

                let iters = if k <= 16 { 9 } else { 5 };
                let mut times = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let out = batcher
                        .render_batch(
                            &gpu_slab,
                            &slices,
                            gpu_mode,
                            rmin,
                            rmax,
                            &lut_u32,
                            GpuPixelOutput::IndexedU8,
                        )
                        .unwrap();
                    std::hint::black_box(out[0]);
                    times.push(t0.elapsed());
                }
                let batch_us = median_us(times);
                let per_tile_us = batch_us / (k as f64);
                let tiles_per_sec = 1e6 / per_tile_us;

                let dec = policy.decide(&DispatchRequest {
                    op: gpu_mode.to_policy_op(),
                    dst_pixels: tile_n * tile_n,
                    upload_bytes: 0,
                    readback_bytes: tile_n * tile_n,
                    batch_size: k,
                    in_flight: 0,
                });

                println!(
                    "| {:<8} | {:>3}x{:<3} | {:>7} | {:>20.1} | {:>19.1} | {:>11.0} | {:>16.2}x | {} |",
                    mode_name,
                    tile_n,
                    tile_n,
                    k,
                    cpu_us,
                    per_tile_us,
                    tiles_per_sec,
                    cpu_us / per_tile_us,
                    if dec.use_gpu { "GPU" } else { "CPU" }
                );
            }
        }
    }
}
