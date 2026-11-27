use std::time::{Duration, Instant};

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_gpu::{
    DispatchPolicy, GpuColorFormula, GpuColormapMode, GpuColormapRenderer, GpuContext,
    GpuExprRenderer, GpuInstr, GpuPixelOutput, opcodes,
};
use cogzilla_kernel::{BandSource, ColorFormula, ColormapName, ExprProgram, sample_cubic};

fn make_bands(w: u32, h: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = (w * h) as usize;
    let mut b0 = vec![0.0f32; n];
    let mut b1 = vec![0.0f32; n];
    let mut b2 = vec![0.0f32; n];
    for y in 0..h {
        let fy = y as f32 * 0.015;
        for x in 0..w {
            let fx = x as f32 * 0.017;
            let i = (y * w + x) as usize;
            b0[i] = 1500.0 + 1000.0 * fx.sin().abs();
            b1[i] = 800.0 + 600.0 * fy.cos().abs();
            b2[i] = 400.0 + 500.0 * (fx + fy).sin().abs();
        }
    }
    (b0, b1, b2)
}

fn median_us(mut samples: Vec<Duration>) -> f64 {
    samples.sort();
    samples[samples.len() / 2].as_secs_f64() * 1e6
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

    let mut renderer = GpuExprRenderer::new(&ctx);

    let src_w = 2048u32;
    let src_h = 2048u32;
    let (b0, b1, b2) = make_bands(src_w, src_h);

    let make_db = |v: &[f32]| {
        let mut s = AlignedF32Slab::new_uninit(v.len());
        s.as_mut_slice().copy_from_slice(v);
        DecodedBand {
            data: BandData::F32(s),
            width: src_w,
            height: src_h,
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 0.0,
                max: 3000.0,
            },
        }
    };
    let db0 = make_db(&b0);
    let db1 = make_db(&b1);
    let db2 = make_db(&b2);
    let blk0 = [Some(&db0)];
    let blk1 = [Some(&db1)];
    let blk2 = [Some(&db2)];
    let bs0 = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blk0);
    let bs1 = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blk1);
    let bs2 = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blk2);

    let gpu_slab = renderer
        .upload_bands(&[&b0, &b1, &b2], src_w, src_h, src_w)
        .unwrap();

    // 1. NDVI expression: (b1-b2)/(b1+b2)
    let ndvi_prog = ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap();
    let ndvi_instrs = [
        GpuInstr::load_band(0),
        GpuInstr::load_band(1),
        GpuInstr::op(opcodes::SUB),
        GpuInstr::load_band(0),
        GpuInstr::load_band(1),
        GpuInstr::op(opcodes::ADD),
        GpuInstr::op(opcodes::DIV),
    ];

    // 2. Transcendental SAR/spectral expression: 10*log10(b1) - 10*log10(b2) + sqrt(b3)
    let sar_prog = ExprProgram::compile("db(b1)-db(b2)+sqrt(b3)").unwrap();
    let sar_instrs = [
        GpuInstr::load_band(0),
        GpuInstr::op(opcodes::DB),
        GpuInstr::load_band(1),
        GpuInstr::op(opcodes::DB),
        GpuInstr::op(opcodes::SUB),
        GpuInstr::load_band(2),
        GpuInstr::op(opcodes::SQRT),
        GpuInstr::op(opcodes::ADD),
    ];

    // 3. 3-band RGB + Gamma 1.5 + Sigmoidal 10 0.5 + LCH Saturation 1.35
    let cpu_cf = ColorFormula::parse("Gamma RGB 1.5 Sigmoidal RGB 10 0.5 Saturation 1.35").unwrap();
    let gpu_cf = GpuColorFormula::from_gamma_sigmoidal_sat(Some(1.5), Some((10.0, 0.5)), Some(1.35));

    let lut = ColormapName::Viridis.lut();
    let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);

    println!("| Workload (Cubic) | Output | CPU (us) | GPU Warm (us) | Speedup |");
    println!("| :--- | :--- | ---: | ---: | ---: |");

    for &n in &[256usize, 512, 1024, 2048, 4096] {
        let u_coords: Vec<f32> = (0..n)
            .map(|i| 4.0 + (i as f32) * ((src_w as f32 - 8.0) / n as f32))
            .collect();
        let v_coords: Vec<f32> = (0..n)
            .map(|i| 4.0 + (i as f32) * ((src_h as f32 - 8.0) / n as f32))
            .collect();
        let iters = if n <= 512 { 9 } else if n <= 1024 { 5 } else { 3 };

        // 1. NDVI IndexedU8
        let mut cpu_out = vec![0u8; n * n];
        let mut cpu_t = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            for (py, &v) in v_coords.iter().enumerate() {
                for (px, &u) in u_coords.iter().enumerate() {
                    let s0 = sample_cubic(&bs0, u, v);
                    let s1 = sample_cubic(&bs1, u, v);
                    let val = ndvi_prog.eval(&[s0, s1]);
                    cpu_out[py * n + px] = if val.is_finite() {
                        1 + ((val + 0.2) * 254.0).clamp(0.0, 254.0) as u8
                    } else {
                        0
                    };
                }
            }
            std::hint::black_box(cpu_out[0]);
            cpu_t.push(t0.elapsed());
        }
        let cpu_ndvi = median_us(cpu_t);

        let _ = renderer
            .render_expr(&gpu_slab, &u_coords, &v_coords, GpuColormapMode::Cubic, &ndvi_instrs, -0.2, 0.8, &lut_u32, GpuPixelOutput::IndexedU8)
            .unwrap();
        let mut gpu_t = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let out = renderer
                .render_expr(&gpu_slab, &u_coords, &v_coords, GpuColormapMode::Cubic, &ndvi_instrs, -0.2, 0.8, &lut_u32, GpuPixelOutput::IndexedU8)
                .unwrap();
            std::hint::black_box(out[0]);
            gpu_t.push(t0.elapsed());
        }
        let gpu_ndvi = median_us(gpu_t);
        println!(
            "| {:<22} | {:>4}x{:<4} | {:>8.1} | {:>13.1} | {:>6.2}x |",
            "NDVI 2-band IdxU8",
            n,
            n,
            cpu_ndvi,
            gpu_ndvi,
            cpu_ndvi / gpu_ndvi
        );

        // 2. Transcendental 3-band dB+sqrt IndexedU8
        let mut cpu_st = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            for (py, &v) in v_coords.iter().enumerate() {
                for (px, &u) in u_coords.iter().enumerate() {
                    let s0 = sample_cubic(&bs0, u, v);
                    let s1 = sample_cubic(&bs1, u, v);
                    let s2 = sample_cubic(&bs2, u, v);
                    let val = sar_prog.eval(&[s0, s1, s2]);
                    cpu_out[py * n + px] = if val.is_finite() {
                        1 + (val * 4.0).clamp(0.0, 254.0) as u8
                    } else {
                        0
                    };
                }
            }
            std::hint::black_box(cpu_out[0]);
            cpu_st.push(t0.elapsed());
        }
        let cpu_sar = median_us(cpu_st);

        let _ = renderer
            .render_expr(&gpu_slab, &u_coords, &v_coords, GpuColormapMode::Cubic, &sar_instrs, 0.0, 60.0, &lut_u32, GpuPixelOutput::IndexedU8)
            .unwrap();
        let mut gpu_st = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let out = renderer
                .render_expr(&gpu_slab, &u_coords, &v_coords, GpuColormapMode::Cubic, &sar_instrs, 0.0, 60.0, &lut_u32, GpuPixelOutput::IndexedU8)
                .unwrap();
            std::hint::black_box(out[0]);
            gpu_st.push(t0.elapsed());
        }
        let gpu_sar = median_us(gpu_st);
        println!(
            "| {:<22} | {:>4}x{:<4} | {:>8.1} | {:>13.1} | {:>6.2}x |",
            "dB+sqrt 3-band IdxU8",
            n,
            n,
            cpu_sar,
            gpu_sar,
            cpu_sar / gpu_sar
        );

        // 3. 3-band RGB + Gamma + Sigmoidal + LCH Saturation RGBA8
        let mut cpu_rgba = vec![0u8; n * n * 4];
        let mut cpu_ct = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            for (py, &v) in v_coords.iter().enumerate() {
                for (px, &u) in u_coords.iter().enumerate() {
                    let sr = sample_cubic(&bs0, u, v);
                    let sg = sample_cubic(&bs1, u, v);
                    let sb = sample_cubic(&bs2, u, v);
                    let r = (sr * (255.0 / 2500.0)).clamp(0.0, 255.0) as u8;
                    let g = (sg * (255.0 / 1500.0)).clamp(0.0, 255.0) as u8;
                    let b = (sb * (255.0 / 1000.0)).clamp(0.0, 255.0) as u8;
                    let (cr, cg, cb) = cpu_cf.apply_rgb(r, g, b);
                    let off = (py * n + px) * 4;
                    cpu_rgba[off] = cr;
                    cpu_rgba[off + 1] = cg;
                    cpu_rgba[off + 2] = cb;
                    cpu_rgba[off + 3] = 255;
                }
            }
            std::hint::black_box(cpu_rgba[0]);
            cpu_ct.push(t0.elapsed());
        }
        let cpu_cf_us = median_us(cpu_ct);

        let _ = renderer
            .render_rgb_formula(&gpu_slab, &u_coords, &v_coords, GpuColormapMode::Cubic, [0.0; 3], [2500.0, 1500.0, 1000.0], &gpu_cf)
            .unwrap();
        let mut gpu_ct = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let out = renderer
                .render_rgb_formula(&gpu_slab, &u_coords, &v_coords, GpuColormapMode::Cubic, [0.0; 3], [2500.0, 1500.0, 1000.0], &gpu_cf)
                .unwrap();
            std::hint::black_box(out[0]);
            gpu_ct.push(t0.elapsed());
        }
        let gpu_cf_us = median_us(gpu_ct);
        println!(
            "| {:<22} | {:>4}x{:<4} | {:>8.1} | {:>13.1} | {:>6.2}x |",
            "RGB+Gamma+Sig+LCHSat",
            n,
            n,
            cpu_cf_us,
            gpu_cf_us,
            cpu_cf_us / gpu_cf_us
        );
    }
}
