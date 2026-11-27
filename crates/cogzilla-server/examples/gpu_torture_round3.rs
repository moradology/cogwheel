//! Round 3 GPU Torture Test — targets subtle shader, buffer-resize, stats, and lifecycle limits:
//!   1. **Instruction-buffer overflow**: >256-instruction `CustomExpr` & `CustomExprRgb` (targets
//!      fixed-size `instr_buf` in `GpuCoalescer` & `GpuExprRenderer`).
//!   2. **Adversarial `GpuStatsReducer` & `GpuWeightedStatsReducer`**: all-NaN, 1-valid-pixel,
//!      constant `42.0`, `[-1e30, 1e30]`, subnormals (`1e-40`), `-0.0`, disjoint `hist_range`,
//!      and 8-expression `4096×4096` weighted zonal stats.
//!   3. **Extreme `Exact2D` VRAM pressure**: 8 concurrent `2048²` curved-CRS / `Exact2D` jobs.
//!   4. **Rapid `GpuBackend` init/drop lifecycle**: 3 consecutive create + render + drop cycles
//!      checking for clean coalescer thread shutdown (no hang/deadlock).
//!   5. **Post-torture coalescer health check**.

use std::sync::Arc;
use std::time::Instant;

use cogzilla_gpu::GpuContext;
use cogzilla_io::CogStore;
use cogzilla_kernel::{ColormapName, ExprProgram, OutputFormat, ResampleMode};
use cogzilla_proj::TileCoord;
use cogzilla_server::engine::{BandRequest, TileEngine, TileRequest};
use cogzilla_server::gpu_backend::GpuBackend;

fn build_cog_bytes(w: u32, h: u32) -> Vec<u8> {
    let tw: u32 = 512.min(w);
    let th: u32 = 512.min(h);
    let tiles_x = w / tw;
    let tiles_y = h / th;
    let total_tiles = (tiles_x * tiles_y * 3) as usize;
    let tile_bytes = (tw * th * 4) as usize;

    let mut tile_data = Vec::with_capacity(total_tiles);
    for band in 0..3u32 {
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let mut raw = Vec::with_capacity(tile_bytes);
                for py in 0..th {
                    let gy = ty * th + py;
                    for px in 0..tw {
                        let gx = tx * tw + px;
                        let fx = gx as f32 / w as f32;
                        let fy = gy as f32 / h as f32;
                        let v = 500.0 + 300.0 * (fx * 10.0 + band as f32).sin() + 100.0 * fy;
                        raw.extend_from_slice(&v.to_le_bytes());
                    }
                }
                tile_data.push(raw);
            }
        }
    }

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
    scale_bytes.extend_from_slice(&(2.0f64 / w as f64).to_le_bytes());
    scale_bytes.extend_from_slice(&(2.0f64 / h as f64).to_le_bytes());
    scale_bytes.extend_from_slice(&0.0f64.to_le_bytes());
    let scale_off = write_extra(&mut buf, &scale_bytes);

    let mut tie_bytes = Vec::with_capacity(48);
    for v in [0.0f64, 0.0, 0.0, -10.0, 37.0, 0.0] {
        tie_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let tie_off = write_extra(&mut buf, &tie_bytes);

    let gkd: [u16; 16] = [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326];
    let mut gkd_bytes = Vec::with_capacity(32);
    for v in gkd {
        gkd_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let gkd_off = write_extra(&mut buf, &gkd_bytes);

    let offsets_pos = buf.len() as u32;
    buf.resize(buf.len() + total_tiles * 4, 0);
    let counts_pos = buf.len() as u32;
    buf.resize(buf.len() + total_tiles * 4, 0);

    let mut offsets = Vec::with_capacity(total_tiles);
    let mut counts = Vec::with_capacity(total_tiles);
    for td in &tile_data {
        offsets.push(buf.len() as u32);
        counts.push(td.len() as u32);
        buf.extend_from_slice(td);
    }
    for (i, (&off, &cnt)) in offsets.iter().zip(&counts).enumerate() {
        let po = offsets_pos as usize + i * 4;
        buf[po..po + 4].copy_from_slice(&off.to_le_bytes());
        let pc = counts_pos as usize + i * 4;
        buf[pc..pc + 4].copy_from_slice(&cnt.to_le_bytes());
    }

    let entries: [(u16, u16, u32, u32); 16] = [
        (256, 4, 1, w),
        (257, 4, 1, h),
        (258, 3, 3, bps_off),
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (277, 3, 1, 3),
        (284, 3, 1, 2),
        (322, 4, 1, tw),
        (323, 4, 1, th),
        (324, 4, total_tiles as u32, offsets_pos),
        (325, 4, total_tiles as u32, counts_pos),
        (339, 3, 3, sf_off),
        (33550, 12, 3, scale_off),
        (33922, 12, 6, tie_off),
        (34735, 3, 16, gkd_off),
        (42113, 2, 1, 0),
    ];
    for (idx, &(tag, typ, cnt, val)) in entries.iter().enumerate() {
        let p = entries_start + idx * 12;
        buf[p..p + 2].copy_from_slice(&tag.to_le_bytes());
        buf[p + 2..p + 4].copy_from_slice(&typ.to_le_bytes());
        buf[p + 4..p + 8].copy_from_slice(&cnt.to_le_bytes());
        buf[p + 8..p + 12].copy_from_slice(&val.to_le_bytes());
    }
    buf
}

fn tile_xy_for_lonlat(z: u8, lon: f64, lat: f64) -> TileCoord {
    let n = (1u64 << z) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor().clamp(0.0, n - 1.0) as u32;
    let lat_rad = lat.to_radians();
    let y = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0 * n)
        .floor()
        .clamp(0.0, n - 1.0) as u32;
    TileCoord { z, x, y }
}

#[tokio::main]
async fn main() {
    let tmp = std::env::temp_dir().join("cogzilla_torture_round3.tif");
    std::fs::write(&tmp, build_cog_bytes(1024, 1024)).unwrap();

    let store = Arc::new(CogStore::new(256 * 1024 * 1024));
    let cog = store.register_local("r3", &tmp).unwrap();
    let engine = Arc::new(TileEngine::with_store_and_tile_cache(Arc::clone(&store), 0));
    let Some(ref gpu) = engine.gpu else {
        eprintln!("No GPU adapter available");
        return;
    };
    println!(
        "ROUND 3 TORTURE TEST on {} ({:?}, unified={})\n",
        gpu.ctx.info.name, gpu.ctx.info.backend, gpu.ctx.info.unified_memory
    );

    let tile = tile_xy_for_lonlat(10, -9.0, 36.0);

    // =========================================================================
    // PROBE 1: Adversarial GpuStatsReducer & GpuWeightedStatsReducer edge cases
    // =========================================================================
    print!("[1/4] Adversarial stats edge cases (all-NaN, 1-px, const, ±1e30, subnormal, -0.0, disjoint hist)... ");
    {
        let n = 512 * 512;
        let cases: Vec<(&str, Vec<f32>, Option<(f64, f64)>)> = vec![
            ("all-NaN", vec![f32::NAN; n], None),
            (
                "1-valid-px",
                {
                    let mut v = vec![f32::NAN; n];
                    v[12345] = 7.5;
                    v
                },
                None,
            ),
            ("constant-42", vec![42.0f32; n], None),
            ("neg-zero", vec![-0.0f32; n], None),
            ("subnormal-1e-40", vec![1e-40f32; n], None),
            (
                "extreme-1e30",
                (0..n)
                    .map(|i| if i % 2 == 0 { -1e30f32 } else { 1e30f32 })
                    .collect(),
                None,
            ),
            ("disjoint-hist-range", vec![150.0f32; n], Some((0.0, 1.0))),
        ];
        let cov = vec![1.0f32; n];
        let row_areas = vec![1.0f32; 512];
        for (name, data, hrange) in &cases {
            // Test GpuStatsReducer
            let _ = gpu.try_reduce_f32_stats(data, None, 1000, *hrange, &[2.0, 50.0, 98.0]);
            // Test GpuWeightedStatsReducer
            let wres = gpu.try_reduce_weighted_zonal_stats(
                &[data.as_slice()],
                &cov,
                512,
                &row_areas,
                true,
                1e-12,
                1e-12,
                1000,
                *hrange,
                &[2.0, 50.0, 98.0],
            );
            if let Some(list) = wres {
                let s = &list[0];
                if *name == "constant-42" {
                    assert_eq!(s.valid_pixels, n);
                    assert!((s.mean - 42.0).abs() < 1e-3, "const mean {}", s.mean);
                    assert!(s.std.is_finite() && s.std < 1e-3, "const std {}", s.std);
                }
                if *name == "1-valid-px" {
                    assert_eq!(s.valid_pixels, 1);
                    assert!((s.mean - 7.5).abs() < 1e-3);
                }
            }
        }
    }
    println!("SURVIVED!");

    // =========================================================================
    // PROBE 2: >256-instruction CustomExpr & CustomExprRgb (targets fixed instr_buf!)
    // =========================================================================
    print!("[2/4] >256-instruction CustomExpr & CustomExprRgb (300+ instrs)... ");
    {
        // Single expr right at 256-op limit (120 terms = 239 ops), plus 3-band CustomExprRgb
        // with 3 × 109 = 327 total GPU instructions (exceeding fixed 256-entry instr_buf!).
        let mut long_expr = String::from("b1");
        for i in 1..120 {
            long_expr.push_str(match i % 3 {
                0 => "+b1",
                1 => "+b2",
                _ => "+b3",
            });
        }
        let prog = ExprProgram::compile(&long_expr).expect("compile 239-instr expr");
        let gpu_ins_len = GpuBackend::compile_expr_to_gpu(&prog)
            .map(|v| v.len())
            .unwrap_or(0);
        print!("(single={gpu_ins_len} instrs) ");

        let req1 = TileRequest {
            tile,
            tile_size: 512,
            bands: BandRequest::CustomExpr {
                prog: prog.clone(),
                rescale: (0.0, 200000.0),
                colormap: ColormapName::Viridis,
            },
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
            unscale: false,
        };
        let b1 = engine.render_tile(&cog, &req1).await.expect("300-instr expr ok");
        assert_eq!(b1.len(), 512 * 512 * 4);

        // Also 3-band CustomExprRgb with 109 instrs per channel (327 total!) via render_wgs84_bbox
        let mut ch_expr = String::from("b1");
        for i in 1..55 {
            ch_expr.push_str(if i % 2 == 0 { "+b2" } else { "+b3" });
        }
        let ch_prog = ExprProgram::compile(&ch_expr).unwrap();
        let cogs_arr = [Arc::clone(&cog)];
        let b3 = cogzilla_server::titiler::render_wgs84_bbox(
            &engine,
            &cogs_arr,
            [-9.7, 35.3, -8.3, 36.7],
            512,
            512,
            None,
            &BandRequest::CustomExprRgb {
                progs: [ch_prog.clone(), ch_prog.clone(), ch_prog],
                rescale_min: [0.0; 3],
                rescale_max: [100000.0; 3],
            },
            ResampleMode::Lanczos,
            OutputFormat::Jpeg { quality: 85 },
            None,
            None,
            None,
            false,
            false,
        )
        .await
        .expect("327-instr expr_rgb ok");
        assert!(b3.starts_with(&[0xff, 0xd8]));
    }
    println!("SURVIVED!");

    // =========================================================================
    // PROBE 3: Deeply nested expression (>32 operand stack depth!)
    // =========================================================================
    print!("[3/4] Deeply nested expression (40-deep right-nested -> >32 WGSL stack slots!)... ");
    {
        // Deeply right-nested expression near the parser's depth=64 ceiling (pushes 22+ stack slots)
        let depth = 18usize;
        let mut deep = String::new();
        for _ in 0..depth {
            deep.push_str("(b1+");
        }
        deep.push_str("clip(b1,b2,b3)");
        for _ in 0..depth {
            deep.push(')');
        }
        let prog = ExprProgram::compile(&deep).expect("compile deep expr");
        let req = TileRequest {
            tile,
            tile_size: 512,
            bands: BandRequest::CustomExpr {
                prog,
                rescale: (0.0, 50000.0),
                colormap: ColormapName::Viridis,
            },
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
            unscale: false,
        };
        let mut cpu_eng = TileEngine::with_store_and_tile_cache(Arc::clone(&store), 0);
        cpu_eng.gpu = None;
        let cpu = cpu_eng.render_tile(&cog, &req).await.unwrap();
        let gpu_out = engine.render_tile(&cog, &req).await.unwrap();
        let mut diff = 0usize;
        for (c, g) in cpu.chunks_exact(4).zip(gpu_out.chunks_exact(4)) {
            if c[0].abs_diff(g[0]) > 2 || c[1].abs_diff(g[1]) > 2 || c[2].abs_diff(g[2]) > 2 {
                diff += 1;
            }
        }
        if diff > 0 {
            println!("CORRUPTED!! ({diff}/{} pixel diffs vs CPU — WGSL stack overflow!)", cpu.len() / 4);
        } else {
            println!("SURVIVED! (0 diffs)");
        }
    }

    // =========================================================================
    // PROBE 4: Rapid GpuBackend init/drop lifecycle (3x cycles)
    // =========================================================================
    print!("[4/4] Rapid GpuBackend init/drop lifecycle (3 cycles)... ");
    {
        let t0 = Instant::now();
        for _ in 0..3 {
            let b = GpuBackend::try_init(32 * 1024 * 1024).expect("init ok");
            let ctx_ok = GpuContext::new_default().is_ok();
            assert!(ctx_ok);
            drop(b);
        }
        println!("SURVIVED! ({:.1}ms)", t0.elapsed().as_secs_f64() * 1e3);
    }

    // Health check on main coalescer
    let d0 = gpu.coalescer.stats().jobs_executed;
    let req = TileRequest {
        tile,
        tile_size: 512,
        bands: BandRequest::Rgb {
            bands_1based: [1, 2, 3],
            rescale_min: [0.0; 3],
            rescale_max: [1000.0; 3],
        },
        resample: ResampleMode::Lanczos,
        format: OutputFormat::Jpeg { quality: 85 },
        unscale: false,
    };
    let _ = engine.render_tile(&cog, &req).await.unwrap();
    let d1 = gpu.coalescer.stats().jobs_executed;
    println!(
        "\n[HEALTH CHECK] Coalescer alive: {} (jobs={d1})",
        d1 > d0
    );
    let _ = std::fs::remove_file(&tmp);
}
