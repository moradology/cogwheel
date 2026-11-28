//! Fused spectral band-math, SAR RGB, and expression-VM tile renderers.

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_proj::CoordPlan;

use crate::bandmath::{SpectralExpr, fast_db_from_power};
use crate::colormap::{ColormapLut, nodata_index};
use crate::expr_vm::ExprProgram;
use crate::jpeg_fast::encode_jpeg_rgba;
use crate::png_fast::{
    encode_filtered_png, encode_indexed_png, filter_sub_ga_row, filter_sub_rgba_row,
    filter_sub_u8_row,
};
use crate::source::BandSource;
use crate::webp_fast::encode_webp_rgba;

use super::{
    OutputFormat, ResampleMode, RgbStyle, SCRATCH, SingleBandStyle, SingleBlockRef, WidenedSampler,
    build_plte_nodata, build_plte_opaque, coord_plan_scale, empty_tile_bytes,
    for_each_pixel_in_row, index_plan, plan_separable_conv_scale, plan_widened_scale, plane_at, rescale_u8,
    safe_rescale_denom, sample_bilinear, sample_cubic, sample_general_unwidened, sample_nearest,
    widened_plane,
};

// ============================================================================
// Fused Satellite Spectral Band Math Kernels (NDVI, NDWI, NBR, SAVI, EVI)
// Zero intermediate float tile allocation — computed directly in scanline loop!
// ============================================================================

pub(super) fn render_eval_fn<const N: usize, F: Fn(f32, f32) -> f32>(
    plan: &CoordPlan<N>,
    style: &SingleBandStyle,
    eval_uv: F,
) -> Bytes {
    let range = safe_rescale_denom(style.rescale_min, style.rescale_max);
    let inv_range = 255.0 / range;
    let min_val = style.rescale_min;

    match style.format {
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
            let mut out = BytesMut::with_capacity(N * N * 4);
            for py in 0..N {
                for_each_pixel_in_row(plan, py, |_px, u, v| {
                    out.put_slice(&eval_uv(u, v).to_le_bytes());
                });
            }
            out.freeze()
        }
        OutputFormat::Png => {
            if matches!(style.colormap, crate::colormap::ColormapName::Grayscale) {
                SCRATCH.with(|cell| {
                    let mut guard = cell.borrow_mut();
                    let sc = &mut *guard;
                    let ga_stride = 1 + N * 2;
                    let ga_total = N * ga_stride;
                    if sc.img_buf.len() < ga_total {
                        sc.img_buf.resize(ga_total, 0);
                    }
                    if sc.row_buf.len() < N * 2 {
                        sc.row_buf.resize(N * 2, 0);
                    }
                    for py in 0..N {
                        let row_ga = &mut sc.row_buf[..N * 2];
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = eval_uv(u, v);
                            let off = px * 2;
                            if val.is_nan() {
                                row_ga[off] = 0;
                                row_ga[off + 1] = 0;
                            } else {
                                row_ga[off] = rescale_u8(val, inv_range, min_val);
                                row_ga[off + 1] = 255;
                            }
                        });
                        filter_sub_ga_row(
                            row_ga,
                            &mut sc.img_buf[py * ga_stride..(py + 1) * ga_stride],
                        );
                    }
                    encode_filtered_png(N as u32, N as u32, 4, &sc.img_buf[..ga_total])
                })
            } else {
                let lut: &ColormapLut = style.colormap.lut();
                let (plte, merged) = build_plte_nodata(lut);
                SCRATCH.with(|cell| {
                    let mut guard = cell.borrow_mut();
                    let sc = &mut *guard;
                    let stride = 1 + N;
                    let total = N * stride;
                    if sc.img_buf.len() < total {
                        sc.img_buf.resize(total, 0);
                    }
                    if sc.row_buf.len() < N {
                        sc.row_buf.resize(N, 0);
                    }
                    for py in 0..N {
                        let row_idx = &mut sc.row_buf[..N];
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = eval_uv(u, v);
                            row_idx[px] = if val.is_nan() {
                                0
                            } else {
                                nodata_index(rescale_u8(val, inv_range, min_val), merged)
                            };
                        });
                        filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                    }
                    encode_indexed_png(
                        N as u32,
                        N as u32,
                        plte,
                        Some(&[0u8]),
                        &sc.img_buf[..total],
                    )
                })
            }
        }
        OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
            let lut: &ColormapLut = style.colormap.lut();
            SCRATCH.with(|cell| {
                let mut guard = cell.borrow_mut();
                let sc = &mut *guard;
                let total = N * N * 4;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                let buf = &mut sc.img_buf[..total];
                for py in 0..N {
                    let row_off = py * N * 4;
                    for_each_pixel_in_row(plan, py, |px, u, v| {
                        let val = eval_uv(u, v);
                        let px_u32 = if val.is_nan() {
                            0
                        } else {
                            let idx = rescale_u8(val, inv_range, min_val) as usize;
                            unsafe { *lut.rgba_u32.get_unchecked(idx) }
                        };
                        let off = row_off + px * 4;
                        buf[off..off + 4].copy_from_slice(&px_u32.to_ne_bytes());
                    });
                }
                match style.format {
                    OutputFormat::Jpeg { quality } => {
                        encode_jpeg_rgba(N as u32, N as u32, buf, quality)
                    }
                    OutputFormat::Webp { quality } => {
                        encode_webp_rgba(N as u32, N as u32, buf, quality)
                    }
                    OutputFormat::RawRgba => Bytes::copy_from_slice(buf),
                    _ => unreachable!(),
                }
            })
        }
    }
}

/// Encode a precomputed single-band `N x N` plane (`NaN` = no data, see [`widened_plane`]). A
/// fully valid PNG is written opaque (grey or indexed without transparency), as the clean
/// specialised samplers do; everything else goes through [`render_eval_fn`].
pub(super) fn render_plane_single<const N: usize>(plane: &[f32], style: &SingleBandStyle) -> Bytes {
    if matches!(style.format, OutputFormat::Png) && !plane.iter().any(|v| v.is_nan()) {
        let inv_range = 255.0 / safe_rescale_denom(style.rescale_min, style.rescale_max);
        let min_val = style.rescale_min;
        let lut: &ColormapLut = style.colormap.lut();
        let plte = build_plte_opaque(lut);
        return SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            let stride = 1 + N;
            let total = N * stride;
            if sc.img_buf.len() < total {
                sc.img_buf.resize(total, 0);
            }
            if sc.row_buf.len() < N {
                sc.row_buf.resize(N, 0);
            }
            for py in 0..N {
                let row_idx = &mut sc.row_buf[..N];
                for (o, &val) in row_idx.iter_mut().zip(&plane[py * N..(py + 1) * N]) {
                    *o = rescale_u8(val, inv_range, min_val);
                }
                filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
            }
            if matches!(style.colormap, crate::colormap::ColormapName::Grayscale) {
                encode_filtered_png(N as u32, N as u32, 0, &sc.img_buf[..total])
            } else {
                encode_indexed_png(N as u32, N as u32, plte, None, &sc.img_buf[..total])
            }
        });
    }
    render_eval_fn(&index_plan::<N>(), style, |u, v| plane_at::<N>(plane, u, v))
}

/// Shared const-generic N-band spectral evaluation dispatcher across all `ResampleMode`s,
/// centralizing nodata short-circuit, `clean` in-bounds detection, and separable `widened_plane`.
fn render_nband_eval<const N: usize, const B: usize>(
    plan: &CoordPlan<N>,
    srcs: [&BandSource<'_>; B],
    style: &SingleBandStyle,
    eval: impl Fn([f32; B]) -> f32,
) -> Bytes {
    if srcs.iter().any(|s| s.all_nodata) {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = plan.all_in_bounds() && srcs.iter().all(|s| s.is_clean_inbounds(c0, c1, r0, r1));
    let eval_checked = |vals: [f32; B]| {
        if vals.iter().any(|v| v.is_nan()) {
            f32::NAN
        } else {
            eval(vals)
        }
    };

    match (style.resample, clean) {
        (
            mode @ (ResampleMode::Bilinear | ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Lanczos),
            _,
        ) if plan_separable_conv_scale(plan, mode).is_some() => {
            let (sx, sy) = coord_plan_scale(plan);
            let planes: [Vec<f32>; B] =
                std::array::from_fn(|i| widened_plane(plan, srcs[i], mode, sx, sy));
            render_eval_fn(&index_plan::<N>(), style, |u, v| {
                let vals = std::array::from_fn(|i| plane_at::<N>(&planes[i], u, v));
                eval_checked(vals)
            })
        }
        (ResampleMode::Bilinear, true) => render_eval_fn(plan, style, |u, v| {
            eval(std::array::from_fn(|i| sample_bilinear::<true>(srcs[i], u, v)))
        }),
        (ResampleMode::Bilinear, false) => render_eval_fn(plan, style, |u, v| {
            eval_checked(std::array::from_fn(|i| sample_bilinear::<false>(srcs[i], u, v)))
        }),
        (ResampleMode::Nearest, true) => render_eval_fn(plan, style, |u, v| {
            eval(std::array::from_fn(|i| sample_nearest::<true>(srcs[i], u, v)))
        }),
        (ResampleMode::Nearest, false) => render_eval_fn(plan, style, |u, v| {
            eval_checked(std::array::from_fn(|i| sample_nearest::<false>(srcs[i], u, v)))
        }),
        (ResampleMode::Cubic, _) => render_eval_fn(plan, style, |u, v| {
            eval_checked(std::array::from_fn(|i| sample_cubic(srcs[i], u, v)))
        }),
        (other, _) => {
            let (scale_x, scale_y) = coord_plan_scale(plan);
            render_eval_fn(plan, style, |u, v| {
                eval_checked(std::array::from_fn(|i| {
                    sample_general_unwidened(srcs[i], u, v, scale_x, scale_y, other)
                }))
            })
        }
    }
}

/// Fused 1-band spectral transform kernel (e.g. SAR linear power -> dB `10*log10(x)`).
pub fn render_bandmath1<const N: usize>(
    plan: &CoordPlan<N>,
    src: &BandSource<'_>,
    expr: SpectralExpr,
    style: &SingleBandStyle,
) -> Bytes {
    render_nband_eval(plan, [src], style, |[a]| expr.eval1(a))
}

/// Fused Dual-Pol SAR False-Color RGB kernel (`[VV, VH]` linear power -> `[VV_dB, VH_dB, VV_dB - VH_dB]`).
pub fn render_sar_rgb<const N: usize>(
    plan: &CoordPlan<N>,
    srcs: &[BandSource<'_>; 2],
    style: &RgbStyle,
) -> Bytes {
    if srcs[0].all_nodata || srcs[1].all_nodata {
        return empty_tile_bytes(N as u32, style.format, 3);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = plan.all_in_bounds()
        && srcs[0].is_clean_inbounds(c0, c1, r0, r1)
        && srcs[1].is_clean_inbounds(c0, c1, r0, r1);

    let inv = [
        255.0 / safe_rescale_denom(style.rescale_min[0], style.rescale_max[0]),
        255.0 / safe_rescale_denom(style.rescale_min[1], style.rescale_max[1]),
        255.0 / safe_rescale_denom(style.rescale_min[2], style.rescale_max[2]),
    ];
    let mins = style.rescale_min;
    let (scale_x, scale_y) = coord_plan_scale(plan);
    let widen = style.resample != ResampleMode::Nearest && plan_widened_scale(plan, style.resample).is_some();
    let wide = widen.then(|| [
        WidenedSampler::new(style.resample, scale_x, scale_y),
        WidenedSampler::new(style.resample, scale_x, scale_y),
    ]);

    let sample_pair = |u: f32, v: f32| -> (f32, f32, f32) {
        let (vv, vh) = match (style.resample, clean) {
            (
                ResampleMode::Bilinear | ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Lanczos,
                _,
            ) if widen => match &wide {
                Some(w) => (w[0].sample(&srcs[0], u, v), w[1].sample(&srcs[1], u, v)),
                None => (f32::NAN, f32::NAN),
            },
            (ResampleMode::Cubic, _) => (
                sample_cubic(&srcs[0], u, v),
                sample_cubic(&srcs[1], u, v),
            ),
            (ResampleMode::Bilinear, true) => (
                sample_bilinear::<true>(&srcs[0], u, v),
                sample_bilinear::<true>(&srcs[1], u, v),
            ),
            (ResampleMode::Bilinear, false) => (
                sample_bilinear::<false>(&srcs[0], u, v),
                sample_bilinear::<false>(&srcs[1], u, v),
            ),
            (ResampleMode::Nearest, true) => (
                sample_nearest::<true>(&srcs[0], u, v),
                sample_nearest::<true>(&srcs[1], u, v),
            ),
            (ResampleMode::Nearest, false) => (
                sample_nearest::<false>(&srcs[0], u, v),
                sample_nearest::<false>(&srcs[1], u, v),
            ),
            (other, _) => (
                sample_general_unwidened(&srcs[0], u, v, scale_x, scale_y, other),
                sample_general_unwidened(&srcs[1], u, v, scale_x, scale_y, other),
            ),
        };
        let vv_db = fast_db_from_power(vv);
        let vh_db = fast_db_from_power(vh);
        (vv_db, vh_db, vv_db - vh_db)
    };

    match style.format {
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
            let plane_bytes = N * N * 4;
            let mut out = vec![0u8; plane_bytes * 3];
            let (r_plane, rest) = out.split_at_mut(plane_bytes);
            let (g_plane, b_plane) = rest.split_at_mut(plane_bytes);
            for py in 0..N {
                let row_off = py * N;
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    let (r, g, b) = sample_pair(u, v);
                    let off = (row_off + px) * 4;
                    r_plane[off..off + 4].copy_from_slice(&r.to_le_bytes());
                    g_plane[off..off + 4].copy_from_slice(&g.to_le_bytes());
                    b_plane[off..off + 4].copy_from_slice(&b.to_le_bytes());
                });
            }
            Bytes::from(out)
        }
        OutputFormat::Png => SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            let stride = 1 + N * 4;
            let total = N * stride;
            if sc.img_buf.len() < total {
                sc.img_buf.resize(total, 0);
            }
            if sc.row_buf.len() < N * 4 {
                sc.row_buf.resize(N * 4, 0);
            }
            for py in 0..N {
                let row_rgba = &mut sc.row_buf[..N * 4];
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    let (r, g, b) = sample_pair(u, v);
                    let off = px * 4;
                    if r.is_nan() || g.is_nan() {
                        row_rgba[off] = 0;
                        row_rgba[off + 1] = 0;
                        row_rgba[off + 2] = 0;
                        row_rgba[off + 3] = 0;
                    } else {
                        row_rgba[off] = rescale_u8(r, inv[0], mins[0]);
                        row_rgba[off + 1] = rescale_u8(g, inv[1], mins[1]);
                        row_rgba[off + 2] = rescale_u8(b, inv[2], mins[2]);
                        row_rgba[off + 3] = 255;
                    }
                });
                filter_sub_rgba_row(row_rgba, &mut sc.img_buf[py * stride..(py + 1) * stride]);
            }
            encode_filtered_png(N as u32, N as u32, 6, &sc.img_buf[..total])
        }),
        OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            let total = N * N * 4;
            if sc.img_buf.len() < total {
                sc.img_buf.resize(total, 0);
            }
            let buf = &mut sc.img_buf[..total];
            for py in 0..N {
                let row_off = py * N * 4;
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    let (r, g, b) = sample_pair(u, v);
                    let off = row_off + px * 4;
                    if r.is_nan() || g.is_nan() {
                        buf[off] = 0;
                        buf[off + 1] = 0;
                        buf[off + 2] = 0;
                        buf[off + 3] = 0;
                    } else {
                        buf[off] = rescale_u8(r, inv[0], mins[0]);
                        buf[off + 1] = rescale_u8(g, inv[1], mins[1]);
                        buf[off + 2] = rescale_u8(b, inv[2], mins[2]);
                        buf[off + 3] = 255;
                    }
                });
            }
            match style.format {
                OutputFormat::Jpeg { quality } => {
                    encode_jpeg_rgba(N as u32, N as u32, buf, quality)
                }
                OutputFormat::Webp { quality } => {
                    encode_webp_rgba(N as u32, N as u32, buf, quality)
                }
                OutputFormat::RawRgba => Bytes::copy_from_slice(buf),
                _ => unreachable!(),
            }
        }),
    }
}

/// Fused 2-band spectral index kernel (NDVI, NDWI, NBR, SAVI, Ratio, Diff).
pub fn render_bandmath2<const N: usize>(
    plan: &CoordPlan<N>,
    srcs: &[BandSource<'_>; 2],
    expr: SpectralExpr,
    style: &SingleBandStyle,
) -> Bytes {
    render_nband_eval(plan, [&srcs[0], &srcs[1]], style, |[a, b]| expr.eval2(a, b))
}

/// Fused 3-band spectral index kernel (EVI).
pub fn render_bandmath3<const N: usize>(
    plan: &CoordPlan<N>,
    srcs: &[BandSource<'_>; 3],
    expr: SpectralExpr,
    style: &SingleBandStyle,
) -> Bytes {
    render_nband_eval(plan, [&srcs[0], &srcs[1], &srcs[2]], style, |[a, b, c]| {
        expr.eval3(a, b, c)
    })
}

/// Evaluate an arbitrary compiled TiTiler `?expression=` bytecode program across any number of
/// bands using the scanline-vectorized [`ExprProgram::eval_row`] engine.
pub fn render_expr_vm<const N: usize>(
    plan: &CoordPlan<N>,
    srcs: &[BandSource<'_>],
    prog: &ExprProgram,
    style: &SingleBandStyle,
    ifd_dims: (u32, u32),
) -> Bytes {
    if !srcs.is_empty()
        && !prog.ops.contains(&crate::expr_vm::Op::Where)
        && srcs.iter().any(|s| s.all_nodata)
    {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let nb = srcs.len().min(16);
    let mask_oob =
        !plan.all_in_bounds() && (nb == 0 || prog.ops.contains(&crate::expr_vm::Op::Where));
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let all_clean =
        plan.all_in_bounds() && srcs[..nb].iter().all(|s| s.is_clean_inbounds(c0, c1, r0, r1));
    let range = safe_rescale_denom(style.rescale_min, style.rescale_max);
    let inv_range = 255.0 / range;
    let min_val = style.rescale_min;

    let single_bands: Vec<Option<SingleBlockRef<'_>>> =
        if all_clean && matches!(style.resample, ResampleMode::Bilinear | ResampleMode::Nearest) {
            srcs[..nb].iter().map(SingleBlockRef::from_source).collect()
        } else {
            Vec::new()
        };
    let all_single = !single_bands.is_empty() && single_bands.iter().all(|s| s.is_some());
    let nearest_x_idxs: Vec<[usize; N]> = if all_single && style.resample == ResampleMode::Nearest {
        if let CoordPlan::Separable1D { u, .. } = plan {
            single_bands
                .iter()
                .map(|b| b.unwrap().precompute_nearest_x(u))
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let widened_planes: Vec<Vec<f32>> = match plan_separable_conv_scale(plan, style.resample) {
        Some((sx, sy)) => srcs[..nb]
            .iter()
            .map(|s| widened_plane(plan, s, style.resample, sx, sy))
            .collect(),
        None => Vec::new(),
    };

    let sample_row = |py: usize, band_rows: &mut [[f32; N]]| {
        match (style.resample, all_clean) {
            (
                ResampleMode::Bilinear | ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Lanczos,
                _,
            ) if !widened_planes.is_empty() => {
                for b in 0..nb {
                    band_rows[b].copy_from_slice(&widened_planes[b][py * N..(py + 1) * N]);
                }
            }
            (ResampleMode::Bilinear, true) => {
                if let CoordPlan::Separable1D { u, v, .. } = plan
                    && all_single
                {
                    let vy = v[py];
                    for b in 0..nb {
                        single_bands[b]
                            .unwrap()
                            .sample_bilinear_row_f32(u, vy, &mut band_rows[b]);
                    }
                    return;
                }
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    for b in 0..nb {
                        band_rows[b][px] = sample_bilinear::<true>(&srcs[b], u, v);
                    }
                });
            }
            (ResampleMode::Bilinear, false) => {
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    for b in 0..nb {
                        band_rows[b][px] = sample_bilinear::<false>(&srcs[b], u, v);
                    }
                });
            }
            (ResampleMode::Nearest, true) => {
                if let CoordPlan::Separable1D { v, .. } = plan
                    && !nearest_x_idxs.is_empty()
                {
                    let vy = v[py];
                    for b in 0..nb {
                        single_bands[b].unwrap().sample_nearest_row_f32(
                            &nearest_x_idxs[b],
                            vy,
                            &mut band_rows[b],
                        );
                    }
                    return;
                }
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    for b in 0..nb {
                        band_rows[b][px] = sample_nearest::<true>(&srcs[b], u, v);
                    }
                });
            }
            (ResampleMode::Nearest, false) => {
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    for b in 0..nb {
                        band_rows[b][px] = sample_nearest::<false>(&srcs[b], u, v);
                    }
                });
            }
            (ResampleMode::Cubic, _) => {
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    for b in 0..nb {
                        band_rows[b][px] = sample_cubic(&srcs[b], u, v);
                    }
                });
            }
            (other, _) => {
                let (scale_x, scale_y) = coord_plan_scale(plan);
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    for b in 0..nb {
                        band_rows[b][px] =
                            sample_general_unwidened(&srcs[b], u, v, scale_x, scale_y, other);
                    }
                });
            }
        }
    };

    let mut band_rows = vec![[0.0f32; N]; nb];
    let mut expr_scratch: Vec<[f32; N]> = Vec::new();
    let mut out_row = [0.0f32; N];
    let eval_masked_row = |py: usize,
                           band_rows: &[[f32; N]],
                           expr_scratch: &mut Vec<[f32; N]>,
                           out_row: &mut [f32; N]| {
        prog.eval_row_with_scratch::<N>(band_rows, expr_scratch, out_row);
        if mask_oob {
            for_each_pixel_in_row(plan, py, |px, u, v| {
                if !BandSource::dims_contain_sample(ifd_dims.0 as i32, ifd_dims.1 as i32, u, v) {
                    out_row[px] = f32::NAN;
                }
            });
        }
    };

    match style.format {
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
            let mut out = BytesMut::with_capacity(N * N * 4);
            for py in 0..N {
                sample_row(py, &mut band_rows);
                eval_masked_row(py, &band_rows, &mut expr_scratch, &mut out_row);
                for &v in &out_row {
                    out.put_slice(&v.to_le_bytes());
                }
            }
            out.freeze()
        }
        OutputFormat::Png => {
            if matches!(style.colormap, crate::colormap::ColormapName::Grayscale) {
                SCRATCH.with(|cell| {
                    let mut guard = cell.borrow_mut();
                    let sc = &mut *guard;
                    let ga_stride = 1 + N * 2;
                    let ga_total = N * ga_stride;
                    if sc.img_buf.len() < ga_total {
                        sc.img_buf.resize(ga_total, 0);
                    }
                    if sc.row_buf.len() < N * 2 {
                        sc.row_buf.resize(N * 2, 0);
                    }
                    for py in 0..N {
                        sample_row(py, &mut band_rows);
                        eval_masked_row(py, &band_rows, &mut expr_scratch, &mut out_row);
                        let row_ga = &mut sc.row_buf[..N * 2];
                        for px in 0..N {
                            let val = out_row[px];
                            let off = px * 2;
                            if val.is_nan() {
                                row_ga[off] = 0;
                                row_ga[off + 1] = 0;
                            } else {
                                row_ga[off] = rescale_u8(val, inv_range, min_val);
                                row_ga[off + 1] = 255;
                            }
                        }
                        filter_sub_ga_row(
                            row_ga,
                            &mut sc.img_buf[py * ga_stride..(py + 1) * ga_stride],
                        );
                    }
                    encode_filtered_png(N as u32, N as u32, 4, &sc.img_buf[..ga_total])
                })
            } else {
                let lut: &ColormapLut = style.colormap.lut();
                let (plte, merged) = build_plte_nodata(lut);
                SCRATCH.with(|cell| {
                    let mut guard = cell.borrow_mut();
                    let sc = &mut *guard;
                    let stride = 1 + N;
                    let total = N * stride;
                    if sc.img_buf.len() < total {
                        sc.img_buf.resize(total, 0);
                    }
                    if sc.row_buf.len() < N {
                        sc.row_buf.resize(N, 0);
                    }
                    for py in 0..N {
                        sample_row(py, &mut band_rows);
                        eval_masked_row(py, &band_rows, &mut expr_scratch, &mut out_row);
                        let row_idx = &mut sc.row_buf[..N];
                        for px in 0..N {
                            let val = out_row[px];
                            row_idx[px] = if val.is_nan() {
                                0
                            } else {
                                nodata_index(rescale_u8(val, inv_range, min_val), merged)
                            };
                        }
                        filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                    }
                    encode_indexed_png(
                        N as u32,
                        N as u32,
                        plte,
                        Some(&[0u8]),
                        &sc.img_buf[..total],
                    )
                })
            }
        }
        OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
            let lut: &ColormapLut = style.colormap.lut();
            SCRATCH.with(|cell| {
                let mut guard = cell.borrow_mut();
                let sc = &mut *guard;
                let total = N * N * 4;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                let buf = &mut sc.img_buf[..total];
                for py in 0..N {
                    sample_row(py, &mut band_rows);
                    eval_masked_row(py, &band_rows, &mut expr_scratch, &mut out_row);
                    let row_off = py * N * 4;
                    for px in 0..N {
                        let val = out_row[px];
                        let px_u32 = if val.is_nan() {
                            0
                        } else {
                            let idx = rescale_u8(val, inv_range, min_val) as usize;
                            unsafe { *lut.rgba_u32.get_unchecked(idx) }
                        };
                        let off = row_off + px * 4;
                        buf[off..off + 4].copy_from_slice(&px_u32.to_ne_bytes());
                    }
                }
                match style.format {
                    OutputFormat::Jpeg { quality } => {
                        encode_jpeg_rgba(N as u32, N as u32, buf, quality)
                    }
                    OutputFormat::Webp { quality } => {
                        encode_webp_rgba(N as u32, N as u32, buf, quality)
                    }
                    OutputFormat::RawRgba => Bytes::copy_from_slice(buf),
                    _ => unreachable!(),
                }
            })
        }
    }
}

