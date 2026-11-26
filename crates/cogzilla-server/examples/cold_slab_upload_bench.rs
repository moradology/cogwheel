//! Benchmark & bit-exact parity verification:
//! Legacy scalar `sample_inbounds` + `Vec<f32>` + `GpuResampler::upload_slab`
//! vs Zero-copy parallel block-row SIMD `stitch_band_source_into_parallel` -> `GpuSlab::from_mapped_fill`
//! across multi-block `2×2` (1024²), `4×4` (2048²), and `8×8` (4096²) `U16` / `U8` / `F32` grids.

use std::time::Instant;

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_gpu::{GpuContext, GpuOptions, GpuResampler, GpuSlab};
use cogzilla_kernel::BandSource;
use cogzilla_server::gpu_backend::stitch_band_source_into_parallel;

fn legacy_stitch_and_upload(resampler: &GpuResampler, src: &BandSource<'_>) -> GpuSlab {
    let base_col = src.tx0 * src.tile_w;
    let base_row = src.ty0 * src.tile_h;
    let w = src.grid_w * src.tile_w;
    let h = src.grid_h * src.tile_h;
    let mut out = vec![f32::NAN; (w as usize) * (h as usize)];
    for row in 0..h {
        let dst_row = &mut out[(row as usize) * (w as usize)..((row + 1) as usize) * (w as usize)];
        let abs_y = base_row + row;
        for col in 0..w {
            dst_row[col as usize] = src.sample_inbounds(base_col + col, abs_y);
        }
    }
    resampler.upload_slab(&out, w, h, w).expect("legacy upload")
}

fn fast_zero_copy_upload(ctx: &GpuContext, src: &BandSource<'_>) -> GpuSlab {
    let w = src.grid_w * src.tile_w;
    let h = src.grid_h * src.tile_h;
    GpuSlab::from_mapped_fill(&ctx.device, w, h, w, |dst| {
        stitch_band_source_into_parallel(src, w, h, dst);
        Ok(())
    })
    .expect("zero-copy upload")
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

    println!();
    println!(
        "{:<24} {:>11} {:>11} {:>10} {:>12}",
        "Multi-Block Grid", "Legacy ms", "ZeroCopy ms", "Speedup", "Eff GB/s"
    );
    println!("{}", "-".repeat(72));

    let tile_dim = 512u32;
    let n_blk = (tile_dim * tile_dim) as usize;
    let u16_blk = DecodedBand {
        width: tile_dim,
        height: tile_dim,
        data: BandData::U16 {
            data: (0..n_blk).map(|i| ((i * 7) % 10000) as u16).collect::<Vec<_>>().into_boxed_slice(),
            nodata: None,
        },
        stats: BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 0.0,
            max: 9999.0,
        },
    };
    let u8_blk = DecodedBand {
        width: tile_dim,
        height: tile_dim,
        data: BandData::U8 {
            data: (0..n_blk).map(|i| (i & 255) as u8).collect::<Vec<_>>().into_boxed_slice(),
            nodata: None,
        },
        stats: BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 0.0,
            max: 255.0,
        },
    };
    let mut f32_slab = AlignedF32Slab::filled(n_blk, 0.0);
    for (i, v) in f32_slab.as_mut_slice().iter_mut().enumerate() {
        *v = (i % 4096) as f32 * 0.5;
    }
    let f32_blk = DecodedBand {
        width: tile_dim,
        height: tile_dim,
        data: BandData::F32(f32_slab),
        stats: BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 0.0,
            max: 2048.0,
        },
    };

    for &(grid_dim, iters) in &[(2u32, 9usize), (4, 5), (8, 3)] {
        let w = grid_dim * tile_dim;
        let h = grid_dim * tile_dim;
        let n_tiles = (grid_dim * grid_dim) as usize;

        for (label, blk) in [("U16", &u16_blk), ("U8 ", &u8_blk), ("F32", &f32_blk)] {
            let blocks: Vec<Option<&DecodedBand>> = vec![Some(blk); n_tiles];
            let src = BandSource::new(w, h, tile_dim, tile_dim, 0, 0, grid_dim, grid_dim, &blocks);

            // Bit-exact parity verification
            let mut fast_buf = vec![0.0f32; (w as usize) * (h as usize)];
            stitch_band_source_into_parallel(&src, w, h, &mut fast_buf);
            for row in (0..h).step_by(67) {
                for col in (0..w).step_by(71) {
                    let expected = src.sample_inbounds(col, row);
                    let got = fast_buf[(row as usize) * (w as usize) + (col as usize)];
                    assert_eq!(got.to_bits(), expected.to_bits(), "bit mismatch at ({col},{row})");
                }
            }

            // Warmup
            let _ = legacy_stitch_and_upload(&resampler, &src);
            let _ = fast_zero_copy_upload(&ctx, &src);

            let mut lt = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = legacy_stitch_and_upload(&resampler, &src);
                lt.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let mut ft = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = fast_zero_copy_upload(&ctx, &src);
                ft.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let l = median_ms(lt);
            let f = median_ms(ft);
            let out_gb = ((w as f64) * (h as f64) * 4.0) / 1e9;
            println!(
                "{:<24} {:>11.2} {:>11.2} {:>9.2}x {:>11.2} GB/s",
                format!("{label} {w}x{h} ({grid_dim}x{grid_dim})"),
                l,
                f,
                l / f,
                out_gb / (f / 1000.0)
            );
        }
    }
}
