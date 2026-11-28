//! Monomorphized single-band fused resampling and tile rendering kernels.

use std::cell::RefCell;

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_proj::CoordPlan;

use crate::colormap::{ColormapLut, nodata_index};
use crate::jpeg_fast::{encode_jpeg_rgb, encode_jpeg_rgba};
use crate::png_fast::{
    encode_filtered_png, encode_indexed_png, filter_sub_ga_row, filter_sub_u8_row,
};
use crate::source::BandSource;
use crate::webp_fast::{encode_webp_rgb, encode_webp_rgba};

use super::{
    AxisTapTable, OutputFormat, ResampleMode, SCRATCH, SingleBandStyle, SingleBlockRef,
    build_plte_nodata, build_plte_opaque, coord_plan_scale, empty_tile_bytes, fast_floor_i32,
    for_each_pixel_in_row, plan_separable_conv_scale, render_eval_fn, render_plane_single,
    rescale_u8, rgb_to_rgba_bytes, safe_rescale_denom, sample_cubic, sample_general_unwidened,
    sample_val, widened_plane, widened_plane_separable_into_with_tables,
};

thread_local! {
    static SINGLE_PLANE_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

// ============================================================================
// Monomorphized Single-Band Fused Kernels (with SIMD Flat-Block Fast Path!)
// ============================================================================

pub(super) fn render_single_specialized<const N: usize, const BILINEAR: bool, const CLEAN: bool>(
    plan: &CoordPlan<N>,
    src: &BandSource<'_>,
    style: &SingleBandStyle,
) -> Bytes {
    let range = safe_rescale_denom(style.rescale_min, style.rescale_max);
    let inv_range = 255.0 / range;
    let min_val = style.rescale_min;

    // Ultra-hot Nearest path: Separable1D + Nearest + CleanFastPath + Single-Block (F32/U16/U8)!
    // Hoists `[usize; N]` X-indices once outside `for py in 0..N` for direct flat slice indexing.
    if !BILINEAR && CLEAN
        && let (CoordPlan::Separable1D { u, v, .. }, Some(blk)) =
            (plan, SingleBlockRef::from_source(src))
        {
            let x_idx = blk.precompute_nearest_x(u);
            let max_y = (blk.height() as i32 - 1).max(0);
            let br_f = blk.base_row() as f32;
            match style.format {
                OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
                    let mut out = BytesMut::with_capacity(N * N * 4);
                    let mut row_f32 = [0.0f32; N];
                    for py in 0..N {
                        blk.sample_nearest_row_f32(&x_idx, v[py], &mut row_f32);
                        for &val in &row_f32 {
                            out.put_slice(&val.to_le_bytes());
                        }
                    }
                    return out.freeze();
                }
                OutputFormat::Png => {
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
                        match blk {
                            SingleBlockRef::F32(flat, _, _, row_stride) => {
                                for py in 0..N {
                                    let y0 = fast_floor_i32(v[py] + 0.5 - br_f)
                                        .0
                                        .clamp(0, max_y) as usize;
                                    let row = &flat[y0 * row_stride..];
                                    let row_idx = &mut sc.row_buf[..N];
                                    for px in 0..N {
                                        row_idx[px] =
                                            rescale_u8(row[x_idx[px]], inv_range, min_val);
                                    }
                                    filter_sub_u8_row(
                                        row_idx,
                                        &mut sc.img_buf[py * stride..(py + 1) * stride],
                                    );
                                }
                            }
                            SingleBlockRef::U16(flat, _, _, row_stride) => {
                                for py in 0..N {
                                    let y0 = fast_floor_i32(v[py] + 0.5 - br_f)
                                        .0
                                        .clamp(0, max_y) as usize;
                                    let row = &flat[y0 * row_stride..];
                                    let row_idx = &mut sc.row_buf[..N];
                                    for px in 0..N {
                                        row_idx[px] =
                                            rescale_u8(row[x_idx[px]] as f32, inv_range, min_val);
                                    }
                                    filter_sub_u8_row(
                                        row_idx,
                                        &mut sc.img_buf[py * stride..(py + 1) * stride],
                                    );
                                }
                            }
                            SingleBlockRef::U8(flat, _, _, row_stride) => {
                                let mut u8_lut = [0u8; 256];
                                for i in 0..256 {
                                    u8_lut[i] = rescale_u8(i as f32, inv_range, min_val);
                                }
                                for py in 0..N {
                                    let y0 = fast_floor_i32(v[py] + 0.5 - br_f)
                                        .0
                                        .clamp(0, max_y) as usize;
                                    let row = &flat[y0 * row_stride..];
                                    let row_idx = &mut sc.row_buf[..N];
                                    for px in 0..N {
                                        row_idx[px] = u8_lut[row[x_idx[px]] as usize];
                                    }
                                    filter_sub_u8_row(
                                        row_idx,
                                        &mut sc.img_buf[py * stride..(py + 1) * stride],
                                    );
                                }
                            }
                        }
                        if matches!(style.colormap, crate::colormap::ColormapName::Grayscale) {
                            encode_filtered_png(N as u32, N as u32, 0, &sc.img_buf[..total])
                        } else {
                            encode_indexed_png(N as u32, N as u32, plte, None, &sc.img_buf[..total])
                        }
                    });
                }
                OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
                    let lut: &ColormapLut = style.colormap.lut();
                    return SCRATCH.with(|cell| {
                        let mut guard = cell.borrow_mut();
                        let sc = &mut *guard;
                        let total = N * N * 3;
                        if sc.img_buf.len() < total {
                            sc.img_buf.resize(total, 0);
                        }
                        let buf = &mut sc.img_buf[..total];
                        match blk {
                            SingleBlockRef::F32(flat, _, _, row_stride) => {
                                for py in 0..N {
                                    let y0 = fast_floor_i32(v[py] + 0.5 - br_f)
                                        .0
                                        .clamp(0, max_y) as usize;
                                    let row = &flat[y0 * row_stride..];
                                    let row_off = py * N * 3;
                                    for px in 0..N {
                                        let idx =
                                            rescale_u8(row[x_idx[px]], inv_range, min_val) as usize;
                                        let c = lut.rgb[idx];
                                        let off = row_off + px * 3;
                                        buf[off] = c[0];
                                        buf[off + 1] = c[1];
                                        buf[off + 2] = c[2];
                                    }
                                }
                            }
                            SingleBlockRef::U16(flat, _, _, row_stride) => {
                                for py in 0..N {
                                    let y0 = fast_floor_i32(v[py] + 0.5 - br_f)
                                        .0
                                        .clamp(0, max_y) as usize;
                                    let row = &flat[y0 * row_stride..];
                                    let row_off = py * N * 3;
                                    for px in 0..N {
                                        let idx = rescale_u8(
                                            row[x_idx[px]] as f32,
                                            inv_range,
                                            min_val,
                                        ) as usize;
                                        let c = lut.rgb[idx];
                                        let off = row_off + px * 3;
                                        buf[off] = c[0];
                                        buf[off + 1] = c[1];
                                        buf[off + 2] = c[2];
                                    }
                                }
                            }
                            SingleBlockRef::U8(flat, _, _, row_stride) => {
                                let mut rgb_lut = [[0u8; 3]; 256];
                                for i in 0..256 {
                                    rgb_lut[i] =
                                        lut.rgb[rescale_u8(i as f32, inv_range, min_val) as usize];
                                }
                                for py in 0..N {
                                    let y0 = fast_floor_i32(v[py] + 0.5 - br_f)
                                        .0
                                        .clamp(0, max_y) as usize;
                                    let row = &flat[y0 * row_stride..];
                                    let row_off = py * N * 3;
                                    for px in 0..N {
                                        let c = rgb_lut[row[x_idx[px]] as usize];
                                        let off = row_off + px * 3;
                                        buf[off] = c[0];
                                        buf[off + 1] = c[1];
                                        buf[off + 2] = c[2];
                                    }
                                }
                            }
                        }
                        match style.format {
                            OutputFormat::Jpeg { quality } => {
                                encode_jpeg_rgb(N as u32, N as u32, buf, quality)
                            }
                            OutputFormat::Webp { quality } => {
                                encode_webp_rgb(N as u32, N as u32, buf, quality)
                            }
                            OutputFormat::RawRgba => rgb_to_rgba_bytes(buf),
                            _ => unreachable!(),
                        }
                    });
                }
            }
        }

    // Ultra-hot SIMD path: Separable1D + Bilinear + CleanFastPath + Single-Block (F32/U16/U8)!
    if BILINEAR && CLEAN
        && let (CoordPlan::Separable1D { u, v, .. }, Some(blk)) =
            (plan, SingleBlockRef::from_source(src))
        {
            match style.format {
                OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
                    let mut out = BytesMut::with_capacity(N * N * 4);
                    let mut row_f32 = [0.0f32; N];
                    for py in 0..N {
                        blk.sample_bilinear_row_f32(u, v[py], &mut row_f32);
                        for &val in &row_f32 {
                            out.put_slice(&val.to_le_bytes());
                        }
                    }
                    return out.freeze();
                }
                OutputFormat::Png => {
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
                        if sc.f32_row.len() < N {
                            sc.f32_row.resize(N, 0.0);
                        }
                        for py in 0..N {
                            blk.sample_bilinear_row_f32(u, v[py], &mut sc.f32_row[..N]);
                            let row_idx = &mut sc.row_buf[..N];
                            for px in 0..N {
                                row_idx[px] = rescale_u8(sc.f32_row[px], inv_range, min_val);
                            }
                            filter_sub_u8_row(
                                row_idx,
                                &mut sc.img_buf[py * stride..(py + 1) * stride],
                            );
                        }
                        if matches!(style.colormap, crate::colormap::ColormapName::Grayscale) {
                            encode_filtered_png(N as u32, N as u32, 0, &sc.img_buf[..total])
                        } else {
                            encode_indexed_png(N as u32, N as u32, plte, None, &sc.img_buf[..total])
                        }
                    });
                }
                OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
                    let lut: &ColormapLut = style.colormap.lut();
                    return SCRATCH.with(|cell| {
                        let mut guard = cell.borrow_mut();
                        let sc = &mut *guard;
                        let total = N * N * 3;
                        if sc.img_buf.len() < total {
                            sc.img_buf.resize(total, 0);
                        }
                        if sc.f32_row.len() < N {
                            sc.f32_row.resize(N, 0.0);
                        }
                        for py in 0..N {
                            blk.sample_bilinear_row_f32(u, v[py], &mut sc.f32_row[..N]);
                            let row_off = py * N * 3;
                            for px in 0..N {
                                let idx = rescale_u8(sc.f32_row[px], inv_range, min_val) as usize;
                                let rgb = unsafe { *lut.rgb.get_unchecked(idx) };
                                let off = row_off + px * 3;
                                sc.img_buf[off] = rgb[0];
                                sc.img_buf[off + 1] = rgb[1];
                                sc.img_buf[off + 2] = rgb[2];
                            }
                        }
                        match style.format {
                            OutputFormat::Jpeg { quality: jq } => {
                                encode_jpeg_rgb(N as u32, N as u32, &sc.img_buf[..total], jq)
                            }
                            OutputFormat::Webp { quality: wq } => {
                                encode_webp_rgb(N as u32, N as u32, &sc.img_buf[..total], wq)
                            }
                            OutputFormat::RawRgba => rgb_to_rgba_bytes(&sc.img_buf[..total]),
                            _ => unreachable!(),
                        }
                    });
                }
            }
        }

    match style.format {
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
            let mut out = BytesMut::with_capacity(N * N * 4);
            for py in 0..N {
                for_each_pixel_in_row(plan, py, |_px, u, v| {
                    let val = sample_val::<BILINEAR, CLEAN>(src, u, v);
                    out.put_slice(&val.to_le_bytes());
                });
            }
            out.freeze()
        }
        OutputFormat::Png => {
            let lut: &ColormapLut = style.colormap.lut();
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
                if CLEAN {
                    for py in 0..N {
                        let row_idx = &mut sc.row_buf[..N];
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = sample_val::<BILINEAR, true>(src, u, v);
                            row_idx[px] = rescale_u8(val, inv_range, min_val);
                        });
                        filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                    }
                    if matches!(style.colormap, crate::colormap::ColormapName::Grayscale) {
                        encode_filtered_png(N as u32, N as u32, 0, &sc.img_buf[..total])
                    } else {
                        let plte = build_plte_opaque(lut);
                        encode_indexed_png(N as u32, N as u32, plte, None, &sc.img_buf[..total])
                    }
                } else if matches!(style.colormap, crate::colormap::ColormapName::Grayscale) {
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
                            let val = sample_val::<BILINEAR, false>(src, u, v);
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
                } else {
                    let (plte, merged) = build_plte_nodata(lut);
                    for py in 0..N {
                        let row_idx = &mut sc.row_buf[..N];
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = sample_val::<BILINEAR, false>(src, u, v);
                            if val.is_nan() {
                                row_idx[px] = 0;
                            } else {
                                row_idx[px] = nodata_index(rescale_u8(val, inv_range, min_val), merged);
                            }
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
                }
            })
        }
        OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
            let lut: &ColormapLut = style.colormap.lut();
            SCRATCH.with(|cell| {
                let mut guard = cell.borrow_mut();
                let sc = &mut *guard;
                if CLEAN {
                    let total = N * N * 3;
                    if sc.img_buf.len() < total {
                        sc.img_buf.resize(total, 0);
                    }
                    let buf = &mut sc.img_buf[..total];
                    for py in 0..N {
                        let row_off = py * N * 3;
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = sample_val::<BILINEAR, true>(src, u, v);
                            let idx = rescale_u8(val, inv_range, min_val) as usize;
                            let rgb = unsafe { *lut.rgb.get_unchecked(idx) };
                            let off = row_off + px * 3;
                            buf[off] = rgb[0];
                            buf[off + 1] = rgb[1];
                            buf[off + 2] = rgb[2];
                        });
                    }
                    match style.format {
                        OutputFormat::Jpeg { quality } => {
                            encode_jpeg_rgb(N as u32, N as u32, buf, quality)
                        }
                        OutputFormat::Webp { quality } => {
                            encode_webp_rgb(N as u32, N as u32, buf, quality)
                        }
                        OutputFormat::RawRgba => rgb_to_rgba_bytes(buf),
                        _ => unreachable!(),
                    }
                } else {
                    let total = N * N * 4;
                    if sc.img_buf.len() < total {
                        sc.img_buf.resize(total, 0);
                    }
                    let buf = &mut sc.img_buf[..total];
                    for py in 0..N {
                        let row_off = py * N * 4;
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = sample_val::<BILINEAR, false>(src, u, v);
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
                }
            })
        }
    }
}

pub fn render_single_band<const N: usize>(
    plan: &CoordPlan<N>,
    src: &BandSource<'_>,
    style: &SingleBandStyle,
) -> Bytes {
    if src.all_nodata {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = plan.all_in_bounds() && src.is_clean_inbounds(c0, c1, r0, r1);

    match (style.resample, clean) {
        (ResampleMode::Nearest, true) => render_single_specialized::<N, false, true>(plan, src, style),
        (ResampleMode::Nearest, false) => render_single_specialized::<N, false, false>(plan, src, style),
        (mode, _) if plan_separable_conv_scale(plan, mode).is_some() => {
            let (sx, sy) = coord_plan_scale(plan);
            if let CoordPlan::Separable1D { u, v, .. } = plan {
                let cols = AxisTapTable::build(mode, &u[..], sx, src.ifd_w);
                let rows = AxisTapTable::build(mode, &v[..], sy, src.ifd_h);
                SINGLE_PLANE_SCRATCH.with(|cell| {
                    let mut plane = cell.borrow_mut();
                    let len = N * N;
                    if plane.len() < len {
                        plane.resize(len, 0.0);
                    }
                    widened_plane_separable_into_with_tables(
                        &u[..], &v[..], src, mode, sx, sy, &cols, &rows, &mut plane[..len],
                    );
                    render_plane_single::<N>(&plane[..len], style)
                })
            } else {
                render_plane_single::<N>(&widened_plane(plan, src, mode, sx, sy), style)
            }
        }
        (ResampleMode::Bilinear, true) => render_single_specialized::<N, true, true>(plan, src, style),
        (ResampleMode::Bilinear, false) => render_single_specialized::<N, true, false>(plan, src, style),
        (ResampleMode::Cubic, _) => render_eval_fn(plan, style, |u, v| sample_cubic(src, u, v)),
        (other, _) => {
            let (sx, sy) = coord_plan_scale(plan);
            render_eval_fn(plan, style, |u, v| sample_general_unwidened(src, u, v, sx, sy, other))
        }
    }
}
