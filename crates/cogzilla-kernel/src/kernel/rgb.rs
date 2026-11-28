//! Monomorphized 3-band satellite RGB fused resampling and tile rendering kernels.

use std::cell::RefCell;

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_proj::CoordPlan;

use crate::jpeg_fast::{encode_jpeg_rgb, encode_jpeg_rgba};
use crate::png_fast::{encode_filtered_png, filter_sub_rgb_row, filter_sub_rgba_row};
use crate::source::BandSource;
use crate::webp_fast::{encode_webp_rgb, encode_webp_rgba};

use super::{
    AxisTapTable, OutputFormat, ResampleMode, RgbStyle, SCRATCH, coord_plan_scale,
    empty_tile_bytes, fast_floor_i32, fill_uv_row, for_each_pixel_in_row,
    plan_separable_conv_scale, rescale_u8, rgb_to_rgba_bytes, safe_rescale_denom,
    sample_bilinear, sample_cubic, sample_general_unwidened, sample_nearest, sample_val,
    widened_plane, widened_plane_separable_into_with_tables, widened_rgb_separable_stream_clean,
};

thread_local! {
    static RGB_PLANE_SCRATCH: RefCell<[Vec<f32>; 3]> = const {
        RefCell::new([Vec::new(), Vec::new(), Vec::new()])
    };
}

/// Fused 3-band bilinear + rescale for a single-block raw typed slice (`u8`, `u16`, or `f32`).
/// Computes floor/frac and `(p00, p01)` once per pixel across all 3 bands, with NEON 4-wide /
/// AVX2 8-wide FMA bilinear + rescale.
#[inline(always)]
pub(super) fn bilinear_rgb_single_block_row<const N: usize, T: Copy, F>(
    u_row: &[f32; N],
    v_row: &[f32; N],
    r_slice: &[T],
    g_slice: &[T],
    b_slice: &[T],
    base_col: u32,
    base_row: u32,
    stride: usize,
    inv: [f32; 3],
    mins: [f32; 3],
    row_rgb: &mut [u8],
    to_f32: F,
) where
    F: Fn(T) -> f32,
{
    let bc = base_col as f32;
    let br = base_row as f32;
    let r_ptr = r_slice.as_ptr();
    let g_ptr = g_slice.as_ptr();
    let b_ptr = b_slice.as_ptr();
    #[allow(unused_assignments, unused_mut)]
    let mut start_px = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let v_bc = vdupq_n_f32(bc);
        let v_br = vdupq_n_f32(br);
        let v_zero = vdupq_n_f32(0.0);
        let v_255 = vdupq_n_f32(255.0);
        let v_inv0 = vdupq_n_f32(inv[0]);
        let v_inv1 = vdupq_n_f32(inv[1]);
        let v_inv2 = vdupq_n_f32(inv[2]);
        let v_min0 = vdupq_n_f32(mins[0]);
        let v_min1 = vdupq_n_f32(mins[1]);
        let v_min2 = vdupq_n_f32(mins[2]);

        let mut xs = [0i32; 4];
        let mut ys = [0i32; 4];
        let mut out_r = [0u32; 4];
        let mut out_g = [0u32; 4];
        let max_x = (stride as i32 - 2).max(0);
        let max_y = ((r_slice.len().min(g_slice.len()).min(b_slice.len()) / stride) as i32 - 2).max(0);
        let v_zero_i = vdupq_n_s32(0);
        let v_max_x = vdupq_n_s32(max_x);
        let v_max_y = vdupq_n_s32(max_y);
        let mut out_b = [0u32; 4];

        let mut px = 0usize;
        while px + 4 <= N {
            let u = vsubq_f32(vld1q_f32(u_row.as_ptr().add(px)), v_bc);
            let v = vsubq_f32(vld1q_f32(v_row.as_ptr().add(px)), v_br);
            let x0_f = vrndmq_f32(u);
            let y0_f = vrndmq_f32(v);
            let fx = vsubq_f32(u, x0_f);
            let fy = vsubq_f32(v, y0_f);
            let vx = vminq_s32(vmaxq_s32(vcvtq_s32_f32(x0_f), v_zero_i), v_max_x);
            let vy = vminq_s32(vmaxq_s32(vcvtq_s32_f32(y0_f), v_zero_i), v_max_y);
            vst1q_s32(xs.as_mut_ptr(), vx);
            vst1q_s32(ys.as_mut_ptr(), vy);

            let p0 = (ys[0] as usize) * stride + (xs[0] as usize);
            let p1 = (ys[1] as usize) * stride + (xs[1] as usize);
            let p2 = (ys[2] as usize) * stride + (xs[2] as usize);
            let p3 = (ys[3] as usize) * stride + (xs[3] as usize);

            macro_rules! eval_band_neon {
                ($ptr:expr, $v_inv:expr, $v_min:expr, $out:expr) => {{
                    let a00 = [
                        to_f32(*$ptr.add(p0)),
                        to_f32(*$ptr.add(p1)),
                        to_f32(*$ptr.add(p2)),
                        to_f32(*$ptr.add(p3)),
                    ];
                    let a10 = [
                        to_f32(*$ptr.add(p0 + 1)),
                        to_f32(*$ptr.add(p1 + 1)),
                        to_f32(*$ptr.add(p2 + 1)),
                        to_f32(*$ptr.add(p3 + 1)),
                    ];
                    let a01 = [
                        to_f32(*$ptr.add(p0 + stride)),
                        to_f32(*$ptr.add(p1 + stride)),
                        to_f32(*$ptr.add(p2 + stride)),
                        to_f32(*$ptr.add(p3 + stride)),
                    ];
                    let a11 = [
                        to_f32(*$ptr.add(p0 + stride + 1)),
                        to_f32(*$ptr.add(p1 + stride + 1)),
                        to_f32(*$ptr.add(p2 + stride + 1)),
                        to_f32(*$ptr.add(p3 + stride + 1)),
                    ];
                    let v00 = vld1q_f32(a00.as_ptr());
                    let v10 = vld1q_f32(a10.as_ptr());
                    let v01 = vld1q_f32(a01.as_ptr());
                    let v11 = vld1q_f32(a11.as_ptr());
                    let top = vfmaq_f32(v00, vsubq_f32(v10, v00), fx);
                    let bot = vfmaq_f32(v01, vsubq_f32(v11, v01), fx);
                    let val = vfmaq_f32(top, vsubq_f32(bot, top), fy);
                    let scaled = vmulq_f32(vsubq_f32(val, $v_min), $v_inv);
                    let clamped = vminq_f32(vmaxq_f32(scaled, v_zero), v_255);
                    vst1q_u32($out.as_mut_ptr(), vcvtq_u32_f32(clamped));
                }};
            }

            eval_band_neon!(r_ptr, v_inv0, v_min0, out_r);
            eval_band_neon!(g_ptr, v_inv1, v_min1, out_g);
            eval_band_neon!(b_ptr, v_inv2, v_min2, out_b);

            let base = px * 3;
            for k in 0..4 {
                *row_rgb.get_unchecked_mut(base + k * 3) = out_r[k] as u8;
                *row_rgb.get_unchecked_mut(base + k * 3 + 1) = out_g[k] as u8;
                *row_rgb.get_unchecked_mut(base + k * 3 + 2) = out_b[k] as u8;
            }
            px += 4;
        }
        start_px = px;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let max_y =
                ((r_slice.len().min(g_slice.len()).min(b_slice.len()) / stride) as i32 - 2).max(0);
            unsafe {
                bilinear_rgb_single_block_row_avx2::<N, T, F>(
                    u_row, v_row, r_ptr, g_ptr, b_ptr, bc, br, stride, max_y, inv, mins, row_rgb,
                    &to_f32,
                );
            }
            start_px = (N / 8) * 8;
        }
    }

    if start_px < N {
        let max_x = (stride as i32 - 2).max(0);
        let max_y = ((r_slice.len().min(g_slice.len()).min(b_slice.len()) / stride) as i32 - 2).max(0);
        for px in start_px..N {
        let (x0_raw, fx) = fast_floor_i32(u_row[px] - bc);
        let (y0_raw, fy) = fast_floor_i32(v_row[px] - br);
        let x0 = x0_raw.clamp(0, max_x) as usize;
        let y0 = y0_raw.clamp(0, max_y) as usize;
        let p00 = y0 * stride + x0;
        let p01 = p00 + stride;
        unsafe {
            let r00 = to_f32(*r_ptr.add(p00));
            let r10 = to_f32(*r_ptr.add(p00 + 1));
            let r01 = to_f32(*r_ptr.add(p01));
            let r11 = to_f32(*r_ptr.add(p01 + 1));
            let rt = r00 + (r10 - r00) * fx;
            let rb = r01 + (r11 - r01) * fx;
            let r = rt + (rb - rt) * fy;

            let g00 = to_f32(*g_ptr.add(p00));
            let g10 = to_f32(*g_ptr.add(p00 + 1));
            let g01 = to_f32(*g_ptr.add(p01));
            let g11 = to_f32(*g_ptr.add(p01 + 1));
            let gt = g00 + (g10 - g00) * fx;
            let gb = g01 + (g11 - g01) * fx;
            let g = gt + (gb - gt) * fy;

            let b00 = to_f32(*b_ptr.add(p00));
            let b10 = to_f32(*b_ptr.add(p00 + 1));
            let b01 = to_f32(*b_ptr.add(p01));
            let b11 = to_f32(*b_ptr.add(p01 + 1));
            let bt = b00 + (b10 - b00) * fx;
            let bb = b01 + (b11 - b01) * fx;
            let b = bt + (bb - bt) * fy;

            let off = px * 3;
            *row_rgb.get_unchecked_mut(off) = rescale_u8(r, inv[0], mins[0]);
            *row_rgb.get_unchecked_mut(off + 1) = rescale_u8(g, inv[1], mins[1]);
            *row_rgb.get_unchecked_mut(off + 2) = rescale_u8(b, inv[2], mins[2]);
        }
        }
    }
}

// SAFETY: Caller checks AVX2+FMA and guarantees `stride >= 2`, `max_y >= 0`, `r_ptr`/`g_ptr`/`b_ptr` cover `(max_y + 2) * stride` elements, and `row_rgb.len() >= N * 3`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn bilinear_rgb_single_block_row_avx2<const N: usize, T: Copy, F: Fn(T) -> f32>(
    u_row: &[f32; N],
    v_row: &[f32; N],
    r_ptr: *const T,
    g_ptr: *const T,
    b_ptr: *const T,
    bc: f32,
    br: f32,
    stride: usize,
    max_y: i32,
    inv: [f32; 3],
    mins: [f32; 3],
    row_rgb: &mut [u8],
    to_f32: &F,
) {
    use std::arch::x86_64::*;
    let v_bc = _mm256_set1_ps(bc);
    let v_br = _mm256_set1_ps(br);
    let v_zero = _mm256_setzero_ps();
    let v_255 = _mm256_set1_ps(255.0);
    let v_inv0 = _mm256_set1_ps(inv[0]);
    let v_inv1 = _mm256_set1_ps(inv[1]);
    let v_inv2 = _mm256_set1_ps(inv[2]);
    let v_min0 = _mm256_set1_ps(mins[0]);
    let v_min1 = _mm256_set1_ps(mins[1]);
    let v_min2 = _mm256_set1_ps(mins[2]);

    let mut xs = [0i32; 8];
    let mut ys = [0i32; 8];
    let mut out_r = [0i32; 8];
    let mut out_g = [0i32; 8];
    let max_x = (stride as i32 - 2).max(0);
    let v_zero_i = _mm256_setzero_si256();
    let v_max_x = _mm256_set1_epi32(max_x);
    let v_max_y = _mm256_set1_epi32(max_y);
    let mut out_b = [0i32; 8];

    let mut px = 0usize;
    while px + 8 <= N {
        // SAFETY: `px + 8 <= N` guarantees 8 readable `f32`s in `u_row`/`v_row`.
        let (u, v) = unsafe {
            (
                _mm256_sub_ps(_mm256_loadu_ps(u_row.as_ptr().add(px)), v_bc),
                _mm256_sub_ps(_mm256_loadu_ps(v_row.as_ptr().add(px)), v_br),
            )
        };
        let x0_f = _mm256_floor_ps(u);
        let y0_f = _mm256_floor_ps(v);
        let fx = _mm256_sub_ps(u, x0_f);
        let fy = _mm256_sub_ps(v, y0_f);
        let vx = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(x0_f), v_zero_i), v_max_x);
        let vy = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(y0_f), v_zero_i), v_max_y);
        // SAFETY: `xs` and `ys` are 8-element `i32` stack arrays (32 bytes each).
        unsafe {
            _mm256_storeu_si256(xs.as_mut_ptr() as *mut __m256i, vx);
            _mm256_storeu_si256(ys.as_mut_ptr() as *mut __m256i, vy);
        }

        let mut p = [0usize; 8];
        for k in 0..8 {
            p[k] = (ys[k] as usize) * stride + (xs[k] as usize);
        }

        macro_rules! eval_band_avx2 {
            ($ptr:expr, $v_inv:expr, $v_min:expr, $out:expr) => {{
                let mut a00 = [0.0f32; 8];
                let mut a10 = [0.0f32; 8];
                let mut a01 = [0.0f32; 8];
                let mut a11 = [0.0f32; 8];
                for k in 0..8 {
                    let pk = p[k];
                    // SAFETY: `xs[k] <= max_x` and `ys[k] <= max_y` keep the 2x2 neighborhood in-bounds.
                    unsafe {
                        a00[k] = to_f32(*$ptr.add(pk));
                        a10[k] = to_f32(*$ptr.add(pk + 1));
                        a01[k] = to_f32(*$ptr.add(pk + stride));
                        a11[k] = to_f32(*$ptr.add(pk + stride + 1));
                    }
                }
                // SAFETY: `a00..a11` are 8-element `f32` stack arrays (32 bytes each).
                let (v00, v10, v01, v11) = unsafe {
                    (
                        _mm256_loadu_ps(a00.as_ptr()),
                        _mm256_loadu_ps(a10.as_ptr()),
                        _mm256_loadu_ps(a01.as_ptr()),
                        _mm256_loadu_ps(a11.as_ptr()),
                    )
                };
                let top = _mm256_fmadd_ps(_mm256_sub_ps(v10, v00), fx, v00);
                let bot = _mm256_fmadd_ps(_mm256_sub_ps(v11, v01), fx, v01);
                let val = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), fy, top);
                let scaled = _mm256_mul_ps(_mm256_sub_ps(val, $v_min), $v_inv);
                let clamped = _mm256_min_ps(_mm256_max_ps(scaled, v_zero), v_255);
                // SAFETY: `$out` is an 8-element `i32` stack array (32 bytes).
                unsafe {
                    _mm256_storeu_si256(
                        $out.as_mut_ptr() as *mut __m256i,
                        _mm256_cvttps_epi32(clamped),
                    );
                }
            }};
        }

        eval_band_avx2!(r_ptr, v_inv0, v_min0, out_r);
        eval_band_avx2!(g_ptr, v_inv1, v_min1, out_g);
        eval_band_avx2!(b_ptr, v_inv2, v_min2, out_b);

        let base = px * 3;
        // SAFETY: `px + 8 <= N` guarantees `base + 24 <= N * 3 <= row_rgb.len()`.
        unsafe {
            for k in 0..8 {
                *row_rgb.get_unchecked_mut(base + k * 3) = out_r[k] as u8;
                *row_rgb.get_unchecked_mut(base + k * 3 + 1) = out_g[k] as u8;
                *row_rgb.get_unchecked_mut(base + k * 3 + 2) = out_b[k] as u8;
            }
        }
        px += 8;
    }
}

/// Fused clean RGB nearest-neighbor scanline sampler (TiTiler default resampling):
/// - Fast path 0: shared interleaved `RgbU8` slab (single 3-byte load per pixel, direct copy on identity).
/// - Fast path 1: single-block `U8`, `U16`, or `F32` (zero block-table lookups per pixel).
/// - Fast path 2: multi-block aligned grid (computes `(x, y)` and block index once per pixel instead of 3×).
/// - Fallback: per-band `sample_nearest::<true>` for heterogeneous grids.
pub(super) fn sample_rgb_nearest_row_clean<const N: usize>(
    plan: &CoordPlan<N>,
    py: usize,
    srcs: &[BandSource<'_>; 3],
    inv: [f32; 3],
    mins: [f32; 3],
    row_rgb: &mut [u8],
) {
    let identity = mins == [0.0, 0.0, 0.0]
        && (inv[0] - 1.0).abs() < 1e-6
        && (inv[1] - 1.0).abs() < 1e-6
        && (inv[2] - 1.0).abs() < 1e-6;

    // Fast path 0: all 3 bands share ONE interleaved RGB8 slab (Sentinel-2 TCI, NAIP, JPEG/WebP RGB).
    if let (Some((rgb0, 0, bc, br, st)), Some((rgb1, 1, _, _, _)), Some((rgb2, 2, _, _, _))) = (
        srcs[0].single_block_rgb_u8(),
        srcs[1].single_block_rgb_u8(),
        srcs[2].single_block_rgb_u8(),
    )
        && std::ptr::eq(rgb0.as_ptr(), rgb1.as_ptr())
            && std::ptr::eq(rgb0.as_ptr(), rgb2.as_ptr())
            && srcs[0].same_grid_as(&srcs[1])
            && srcs[0].same_grid_as(&srcs[2])
        {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            let bc_f = bc as f32;
            let br_f = br as f32;
            let stride3 = st * 3;
            let max_x = (st as i32 - 1).max(0);
            let max_y = ((rgb0.len() / stride3) as i32 - 1).max(0);
            if identity {
                for px in 0..N {
                    let x = fast_floor_i32(u_row[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
                    let y = fast_floor_i32(v_row[px] + 0.5 - br_f).0.clamp(0, max_y) as usize;
                    let o = y * stride3 + x * 3;
                    let d = px * 3;
                    row_rgb[d] = rgb0[o];
                    row_rgb[d + 1] = rgb0[o + 1];
                    row_rgb[d + 2] = rgb0[o + 2];
                }
            } else {
                for px in 0..N {
                    let x = fast_floor_i32(u_row[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
                    let y = fast_floor_i32(v_row[px] + 0.5 - br_f).0.clamp(0, max_y) as usize;
                    let o = y * stride3 + x * 3;
                    let d = px * 3;
                    row_rgb[d] = rescale_u8(rgb0[o] as f32, inv[0], mins[0]);
                    row_rgb[d + 1] = rescale_u8(rgb0[o + 1] as f32, inv[1], mins[1]);
                    row_rgb[d + 2] = rescale_u8(rgb0[o + 2] as f32, inv[2], mins[2]);
                }
            }
            return;
        }

    // Fast path 1a: single-block planar U8 across all 3 bands.
    if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
        srcs[0].single_block_u8(),
        srcs[1].single_block_u8(),
        srcs[2].single_block_u8(),
    )
        && bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            let bc_f = bc as f32;
            let br_f = br as f32;
            let min_len = r.len().min(g.len()).min(b.len());
            let max_x = (st as i32 - 1).max(0);
            let max_y = ((min_len / st) as i32 - 1).max(0);
            if identity {
                for px in 0..N {
                    let x = fast_floor_i32(u_row[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
                    let y = fast_floor_i32(v_row[px] + 0.5 - br_f).0.clamp(0, max_y) as usize;
                    let o = y * st + x;
                    let d = px * 3;
                    row_rgb[d] = r[o];
                    row_rgb[d + 1] = g[o];
                    row_rgb[d + 2] = b[o];
                }
            } else {
                for px in 0..N {
                    let x = fast_floor_i32(u_row[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
                    let y = fast_floor_i32(v_row[px] + 0.5 - br_f).0.clamp(0, max_y) as usize;
                    let o = y * st + x;
                    let d = px * 3;
                    row_rgb[d] = rescale_u8(r[o] as f32, inv[0], mins[0]);
                    row_rgb[d + 1] = rescale_u8(g[o] as f32, inv[1], mins[1]);
                    row_rgb[d + 2] = rescale_u8(b[o] as f32, inv[2], mins[2]);
                }
            }
            return;
        }

    // Fast path 1b: single-block planar U16 across all 3 bands (Sentinel-2 / Landsat).
    if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
        srcs[0].single_block_u16(),
        srcs[1].single_block_u16(),
        srcs[2].single_block_u16(),
    )
        && bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            let bc_f = bc as f32;
            let br_f = br as f32;
            let min_len = r.len().min(g.len()).min(b.len());
            let max_x = (st as i32 - 1).max(0);
            let max_y = ((min_len / st) as i32 - 1).max(0);
            for px in 0..N {
                let x = fast_floor_i32(u_row[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
                let y = fast_floor_i32(v_row[px] + 0.5 - br_f).0.clamp(0, max_y) as usize;
                let o = y * st + x;
                let d = px * 3;
                row_rgb[d] = rescale_u8(r[o] as f32, inv[0], mins[0]);
                row_rgb[d + 1] = rescale_u8(g[o] as f32, inv[1], mins[1]);
                row_rgb[d + 2] = rescale_u8(b[o] as f32, inv[2], mins[2]);
            }
            return;
        }

    // Fast path 1c: single-block planar F32 across all 3 bands.
    if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
        srcs[0].single_block_flat(),
        srcs[1].single_block_flat(),
        srcs[2].single_block_flat(),
    )
        && bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            let bc_f = bc as f32;
            let br_f = br as f32;
            let min_len = r.len().min(g.len()).min(b.len());
            let max_x = (st as i32 - 1).max(0);
            let max_y = ((min_len / st) as i32 - 1).max(0);
            for px in 0..N {
                let x = fast_floor_i32(u_row[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
                let y = fast_floor_i32(v_row[px] + 0.5 - br_f).0.clamp(0, max_y) as usize;
                let o = y * st + x;
                let d = px * 3;
                row_rgb[d] = rescale_u8(r[o], inv[0], mins[0]);
                row_rgb[d + 1] = rescale_u8(g[o], inv[1], mins[1]);
                row_rgb[d + 2] = rescale_u8(b[o], inv[2], mins[2]);
            }
            return;
        }

    // Fast path 2: multi-block aligned grid across all 3 bands.
    if srcs[0].same_grid_as(&srcs[1]) && srcs[0].same_grid_as(&srcs[2]) {
        let s0 = &srcs[0];
        let s1 = &srcs[1];
        let s2 = &srcs[2];
        for_each_pixel_in_row(plan, py, |px, u, v| {
            let x = fast_floor_i32(u + 0.5).0 as u32;
            let y = fast_floor_i32(v + 0.5).0 as u32;
            let (r, g, b) = if let Some((bidx, pidx)) = s0.pixel_loc(x, y) {
                unsafe {
                    if let (Some(b0), Some(b1), Some(b2)) = (
                        *s0.blocks.get_unchecked(bidx),
                        *s1.blocks.get_unchecked(bidx),
                        *s2.blocks.get_unchecked(bidx),
                    ) && b1.width == b0.width
                        && b1.height == b0.height
                        && b2.width == b0.width
                        && b2.height == b0.height
                    {
                        (
                            b0.data.get_f32_unchecked(pidx),
                            b1.data.get_f32_unchecked(pidx),
                            b2.data.get_f32_unchecked(pidx),
                        )
                    } else {
                        (
                            s0.sample_inbounds(x, y),
                            s1.sample_inbounds(x, y),
                            s2.sample_inbounds(x, y),
                        )
                    }
                }
            } else {
                (
                    s0.sample_inbounds(x, y),
                    s1.sample_inbounds(x, y),
                    s2.sample_inbounds(x, y),
                )
            };
            let d = px * 3;
            row_rgb[d] = rescale_u8(r, inv[0], mins[0]);
            row_rgb[d + 1] = rescale_u8(g, inv[1], mins[1]);
            row_rgb[d + 2] = rescale_u8(b, inv[2], mins[2]);
        });
        return;
    }

    // Fallback: heterogeneous grids across bands.
    for_each_pixel_in_row(plan, py, |px, u, v| {
        let r = sample_nearest::<true>(&srcs[0], u, v);
        let g = sample_nearest::<true>(&srcs[1], u, v);
        let b = sample_nearest::<true>(&srcs[2], u, v);
        let d = px * 3;
        row_rgb[d] = rescale_u8(r, inv[0], mins[0]);
        row_rgb[d + 1] = rescale_u8(g, inv[1], mins[1]);
        row_rgb[d + 2] = rescale_u8(b, inv[2], mins[2]);
    });
}

/// ARM NEON 4-wide bilinear row sampler for interleaved `RgbU8Channel` on `Separable1D`
/// (hoisted `y0`/`fy` once per scanline, 1 cache-line load per sample pair).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn bilinear_row_rgb_u8_separable_neon<const N: usize>(
    u_coords: &[f32; N],
    bc: f32,
    r0_ptr: *const u8,
    r1_ptr: *const u8,
    max_x: i32,
    fy: f32,
    inv: [f32; 3],
    mins: [f32; 3],
    identity: bool,
    row_rgb: &mut [u8],
) {
    use std::arch::aarch64::*;
    let (
        v_bc,
        v_fy,
        v_zero_i,
        v_max_x,
        v_zero,
        v_255,
        v_half,
        v_inv0,
        v_inv1,
        v_inv2,
        v_min0,
        v_min1,
        v_min2,
    ) = unsafe {
        (
            vdupq_n_f32(bc),
            vdupq_n_f32(fy),
            vdupq_n_s32(0),
            vdupq_n_s32(max_x),
            vdupq_n_f32(0.0),
            vdupq_n_f32(255.0),
            vdupq_n_f32(0.5),
            vdupq_n_f32(inv[0]),
            vdupq_n_f32(inv[1]),
            vdupq_n_f32(inv[2]),
            vdupq_n_f32(mins[0]),
            vdupq_n_f32(mins[1]),
            vdupq_n_f32(mins[2]),
        )
    };
    let mut xs = [0i32; 4];
    let mut out_r = [0i32; 4];
    let mut out_g = [0i32; 4];
    let mut out_b = [0i32; 4];
    let mut px = 0usize;
    while px + 4 <= N {
        unsafe {
            let u = vsubq_f32(vld1q_f32(u_coords.as_ptr().add(px)), v_bc);
            let x0_f = vrndmq_f32(u);
            let fx = vsubq_f32(u, x0_f);
            let vx = vminq_s32(vmaxq_s32(vcvtq_s32_f32(x0_f), v_zero_i), v_max_x);
            vst1q_s32(xs.as_mut_ptr(), vx);

            let o0 = (xs[0] as usize) * 3;
            let o1 = (xs[1] as usize) * 3;
            let o2 = (xs[2] as usize) * 3;
            let o3 = (xs[3] as usize) * 3;

            macro_rules! eval_ch {
                ($ch:expr, $v_inv:expr, $v_min:expr, $out:expr) => {{
                    let a00 = [
                        *r0_ptr.add(o0 + $ch) as f32,
                        *r0_ptr.add(o1 + $ch) as f32,
                        *r0_ptr.add(o2 + $ch) as f32,
                        *r0_ptr.add(o3 + $ch) as f32,
                    ];
                    let a10 = [
                        *r0_ptr.add(o0 + 3 + $ch) as f32,
                        *r0_ptr.add(o1 + 3 + $ch) as f32,
                        *r0_ptr.add(o2 + 3 + $ch) as f32,
                        *r0_ptr.add(o3 + 3 + $ch) as f32,
                    ];
                    let a01 = [
                        *r1_ptr.add(o0 + $ch) as f32,
                        *r1_ptr.add(o1 + $ch) as f32,
                        *r1_ptr.add(o2 + $ch) as f32,
                        *r1_ptr.add(o3 + $ch) as f32,
                    ];
                    let a11 = [
                        *r1_ptr.add(o0 + 3 + $ch) as f32,
                        *r1_ptr.add(o1 + 3 + $ch) as f32,
                        *r1_ptr.add(o2 + 3 + $ch) as f32,
                        *r1_ptr.add(o3 + 3 + $ch) as f32,
                    ];
                    let v00 = vld1q_f32(a00.as_ptr());
                    let v10 = vld1q_f32(a10.as_ptr());
                    let v01 = vld1q_f32(a01.as_ptr());
                    let v11 = vld1q_f32(a11.as_ptr());
                    let top = vfmaq_f32(v00, vsubq_f32(v10, v00), fx);
                    let bot = vfmaq_f32(v01, vsubq_f32(v11, v01), fx);
                    let val = vfmaq_f32(top, vsubq_f32(bot, top), v_fy);
                    let rounded = if identity {
                        vaddq_f32(val, v_half)
                    } else {
                        let scaled = vmulq_f32(vsubq_f32(val, $v_min), $v_inv);
                        vminq_f32(vmaxq_f32(scaled, v_zero), v_255)
                    };
                    vst1q_s32($out.as_mut_ptr(), vcvtq_s32_f32(rounded));
                }};
            }

            eval_ch!(0, v_inv0, v_min0, out_r);
            eval_ch!(1, v_inv1, v_min1, out_g);
            eval_ch!(2, v_inv2, v_min2, out_b);

            let base = px * 3;
            for k in 0..4 {
                *row_rgb.get_unchecked_mut(base + k * 3) = out_r[k] as u8;
                *row_rgb.get_unchecked_mut(base + k * 3 + 1) = out_g[k] as u8;
                *row_rgb.get_unchecked_mut(base + k * 3 + 2) = out_b[k] as u8;
            }
        }
        px += 4;
    }
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc);
        let o = (x0_raw.clamp(0, max_x) as usize) * 3;
        unsafe {
            let off = px * 3;
            for ch in 0..3 {
                let v00 = *r0_ptr.add(o + ch) as f32;
                let v10 = *r0_ptr.add(o + 3 + ch) as f32;
                let v01 = *r1_ptr.add(o + ch) as f32;
                let v11 = *r1_ptr.add(o + 3 + ch) as f32;
                let top = v00 + (v10 - v00) * fx;
                let bot = v01 + (v11 - v01) * fx;
                let val = top + (bot - top) * fy;
                *row_rgb.get_unchecked_mut(off + ch) = if identity {
                    (val + 0.5) as u8
                } else {
                    rescale_u8(val, inv[ch], mins[ch])
                };
            }
        }
        px += 1;
    }
}

/// x86_64 AVX2+FMA 8-wide bilinear row sampler for interleaved `RgbU8Channel` on `Separable1D`.
// SAFETY: Caller checks AVX2+FMA and guarantees `r0_ptr`/`r1_ptr` have `>= (max_x + 2) * 3` bytes and `row_rgb.len() >= N * 3`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilinear_row_rgb_u8_separable_avx2<const N: usize>(
    u_coords: &[f32; N],
    bc: f32,
    r0_ptr: *const u8,
    r1_ptr: *const u8,
    max_x: i32,
    fy: f32,
    inv: [f32; 3],
    mins: [f32; 3],
    identity: bool,
    row_rgb: &mut [u8],
) {
    use std::arch::x86_64::*;
    let v_bc = _mm256_set1_ps(bc);
    let v_fy = _mm256_set1_ps(fy);
    let v_zero_i = _mm256_setzero_si256();
    let v_max_x = _mm256_set1_epi32(max_x);
    let v_zero = _mm256_setzero_ps();
    let v_255 = _mm256_set1_ps(255.0);
    let v_half = _mm256_set1_ps(0.5);
    let v_inv0 = _mm256_set1_ps(inv[0]);
    let v_inv1 = _mm256_set1_ps(inv[1]);
    let v_inv2 = _mm256_set1_ps(inv[2]);
    let v_min0 = _mm256_set1_ps(mins[0]);
    let v_min1 = _mm256_set1_ps(mins[1]);
    let v_min2 = _mm256_set1_ps(mins[2]);
    let mut xs = [0i32; 8];
    let mut out_r = [0i32; 8];
    let mut out_g = [0i32; 8];
    let mut out_b = [0i32; 8];
    let mut px = 0usize;
    while px + 8 <= N {
        // SAFETY: `px + 8 <= N` guarantees 8 readable `f32`s in `u_coords` and `xs` holds 8 `i32`s.
        let u = _mm256_sub_ps(unsafe { _mm256_loadu_ps(u_coords.as_ptr().add(px)) }, v_bc);
        let x0_f = _mm256_floor_ps(u);
        let fx = _mm256_sub_ps(u, x0_f);
        let vx = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(x0_f), v_zero_i), v_max_x);
        unsafe { _mm256_storeu_si256(xs.as_mut_ptr() as *mut __m256i, vx) };

        let mut os = [0usize; 8];
        for k in 0..8 {
            os[k] = (xs[k] as usize) * 3;
        }

        macro_rules! eval_ch {
            ($ch:expr, $v_inv:expr, $v_min:expr, $out:expr) => {{
                let mut a00 = [0.0f32; 8];
                let mut a10 = [0.0f32; 8];
                let mut a01 = [0.0f32; 8];
                let mut a11 = [0.0f32; 8];
                for k in 0..8 {
                    let o = os[k];
                    // SAFETY: `xs[k] <= max_x` guarantees `o + 3 + $ch < (max_x + 2) * 3`.
                    unsafe {
                        a00[k] = *r0_ptr.add(o + $ch) as f32;
                        a10[k] = *r0_ptr.add(o + 3 + $ch) as f32;
                        a01[k] = *r1_ptr.add(o + $ch) as f32;
                        a11[k] = *r1_ptr.add(o + 3 + $ch) as f32;
                    }
                }
                // SAFETY: `a00..a11` are 8-element `f32` stack arrays (32 bytes each).
                let (v00, v10, v01, v11) = unsafe {
                    (
                        _mm256_loadu_ps(a00.as_ptr()),
                        _mm256_loadu_ps(a10.as_ptr()),
                        _mm256_loadu_ps(a01.as_ptr()),
                        _mm256_loadu_ps(a11.as_ptr()),
                    )
                };
                let top = _mm256_fmadd_ps(_mm256_sub_ps(v10, v00), fx, v00);
                let bot = _mm256_fmadd_ps(_mm256_sub_ps(v11, v01), fx, v01);
                let val = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), v_fy, top);
                let rounded = if identity {
                    _mm256_add_ps(val, v_half)
                } else {
                    let scaled = _mm256_mul_ps(_mm256_sub_ps(val, $v_min), $v_inv);
                    _mm256_min_ps(_mm256_max_ps(scaled, v_zero), v_255)
                };
                // SAFETY: `$out` is an 8-element `i32` stack array (32 bytes).
                unsafe {
                    _mm256_storeu_si256($out.as_mut_ptr() as *mut __m256i, _mm256_cvttps_epi32(rounded));
                }
            }};
        }

        eval_ch!(0, v_inv0, v_min0, out_r);
        eval_ch!(1, v_inv1, v_min1, out_g);
        eval_ch!(2, v_inv2, v_min2, out_b);

        let base = px * 3;
        // SAFETY: `px + 8 <= N` guarantees `base + 24 <= N * 3 <= row_rgb.len()`.
        unsafe {
            for k in 0..8 {
                *row_rgb.get_unchecked_mut(base + k * 3) = out_r[k] as u8;
                *row_rgb.get_unchecked_mut(base + k * 3 + 1) = out_g[k] as u8;
                *row_rgb.get_unchecked_mut(base + k * 3 + 2) = out_b[k] as u8;
            }
        }
        px += 8;
    }
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc);
        let o = (x0_raw.clamp(0, max_x) as usize) * 3;
        let off = px * 3;
        for ch in 0..3 {
            // SAFETY: `o <= max_x * 3` and `off + ch < N * 3 <= row_rgb.len()`.
            unsafe {
                let v00 = *r0_ptr.add(o + ch) as f32;
                let v10 = *r0_ptr.add(o + 3 + ch) as f32;
                let v01 = *r1_ptr.add(o + ch) as f32;
                let v11 = *r1_ptr.add(o + 3 + ch) as f32;
                let top = v00 + (v10 - v00) * fx;
                let bot = v01 + (v11 - v01) * fx;
                let val = top + (bot - top) * fy;
                *row_rgb.get_unchecked_mut(off + ch) = if identity {
                    (val + 0.5) as u8
                } else {
                    rescale_u8(val, inv[ch], mins[ch])
                };
            }
        }
        px += 1;
    }
}

/// Fused clean RGB bilinear scanline sampler:
/// - Fast path 0: shared interleaved `RgbU8` slab (NEON/AVX2 on `Separable1D`, 1 cache-line stream).
/// - Fast path 1: single-block `U8`, `U16`, or `F32` (NEON 4-wide / AVX2 8-wide, zero block lookups).
/// - Fast path 2: multi-block aligned grid (computes floor/frac + `(bidx, p00, p01)` once per pixel instead of 3×).
/// - Fallback: per-band sampling for heterogeneous grids.
pub(super) fn sample_rgb_bilinear_row_clean<const N: usize>(
    plan: &CoordPlan<N>,
    py: usize,
    srcs: &[BandSource<'_>; 3],
    inv: [f32; 3],
    mins: [f32; 3],
    row_rgb: &mut [u8],
) {
    // Fast path 0: all 3 bands share ONE interleaved RGB8 slab (Sentinel-2 TCI, NAIP, JPEG/WebP RGB)
    // -> 6 contiguous bytes per row pair (1 cache line instead of 3 separate streams!)
    if let (Some((rgb0, 0, bc, br, st)), Some((rgb1, 1, _, _, _)), Some((rgb2, 2, _, _, _))) = (
        srcs[0].single_block_rgb_u8(),
        srcs[1].single_block_rgb_u8(),
        srcs[2].single_block_rgb_u8(),
    )
        && std::ptr::eq(rgb0.as_ptr(), rgb1.as_ptr())
            && std::ptr::eq(rgb0.as_ptr(), rgb2.as_ptr())
            && srcs[0].same_grid_as(&srcs[1])
            && srcs[0].same_grid_as(&srcs[2])
        {
            let bc_f = bc as f32;
            let br_f = br as f32;
            let ptr = rgb0.as_ptr();
            let stride3 = st * 3;
            let identity = mins == [0.0, 0.0, 0.0]
                && (inv[0] - 1.0).abs() < 1e-6
                && (inv[1] - 1.0).abs() < 1e-6
                && (inv[2] - 1.0).abs() < 1e-6;
            let max_x = (st as i32 - 2).max(0);
            let max_y = ((rgb0.len() / stride3) as i32 - 2).max(0);
            if let CoordPlan::Separable1D { u, v, .. } = plan
                && row_rgb.len() >= N * 3
                && rgb0.len() >= (max_y as usize + 2) * stride3
            {
                let (y0_raw, fy) = fast_floor_i32(v[py] - br_f);
                let y0 = y0_raw.clamp(0, max_y) as usize;
                let r0_ptr = unsafe { ptr.add(y0 * stride3) };
                let r1_ptr = unsafe { ptr.add((y0 + 1) * stride3) };
                #[cfg(target_arch = "aarch64")]
                {
                    unsafe {
                        bilinear_row_rgb_u8_separable_neon::<N>(
                            u, bc_f, r0_ptr, r1_ptr, max_x, fy, inv, mins, identity, row_rgb,
                        );
                    }
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx2")
                        && std::is_x86_feature_detected!("fma")
                    {
                        unsafe {
                            bilinear_row_rgb_u8_separable_avx2::<N>(
                                u, bc_f, r0_ptr, r1_ptr, max_x, fy, inv, mins, identity, row_rgb,
                            );
                        }
                        return;
                    }
                }
                #[allow(unreachable_code)]
                {
                    for px in 0..N {
                        let (x0_raw, fx) = fast_floor_i32(u[px] - bc_f);
                        let o = (x0_raw.clamp(0, max_x) as usize) * 3;
                        let off = px * 3;
                        unsafe {
                            for ch in 0..3 {
                                let v00 = *r0_ptr.add(o + ch) as f32;
                                let v10 = *r0_ptr.add(o + 3 + ch) as f32;
                                let v01 = *r1_ptr.add(o + ch) as f32;
                                let v11 = *r1_ptr.add(o + 3 + ch) as f32;
                                let top = v00 + (v10 - v00) * fx;
                                let bot = v01 + (v11 - v01) * fx;
                                let val = top + (bot - top) * fy;
                                *row_rgb.get_unchecked_mut(off + ch) = if identity {
                                    (val + 0.5) as u8
                                } else {
                                    rescale_u8(val, inv[ch], mins[ch])
                                };
                            }
                        }
                    }
                    return;
                }
            }
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            for px in 0..N {
                let (x0_raw, fx) = fast_floor_i32(u_row[px] - bc_f);
                let (y0_raw, fy) = fast_floor_i32(v_row[px] - br_f);
                let x0 = x0_raw.clamp(0, max_x) as usize;
                let y0 = y0_raw.clamp(0, max_y) as usize;
                let p00 = y0 * stride3 + x0 * 3;
                let p01 = p00 + stride3;
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;
                unsafe {
                    let r = (*ptr.add(p00) as f32) * w00
                        + (*ptr.add(p00 + 3) as f32) * w10
                        + (*ptr.add(p01) as f32) * w01
                        + (*ptr.add(p01 + 3) as f32) * w11;
                    let g = (*ptr.add(p00 + 1) as f32) * w00
                        + (*ptr.add(p00 + 4) as f32) * w10
                        + (*ptr.add(p01 + 1) as f32) * w01
                        + (*ptr.add(p01 + 4) as f32) * w11;
                    let b = (*ptr.add(p00 + 2) as f32) * w00
                        + (*ptr.add(p00 + 5) as f32) * w10
                        + (*ptr.add(p01 + 2) as f32) * w01
                        + (*ptr.add(p01 + 5) as f32) * w11;
                    let off = px * 3;
                    if identity {
                        *row_rgb.get_unchecked_mut(off) = (r + 0.5) as u8;
                        *row_rgb.get_unchecked_mut(off + 1) = (g + 0.5) as u8;
                        *row_rgb.get_unchecked_mut(off + 2) = (b + 0.5) as u8;
                    } else {
                        *row_rgb.get_unchecked_mut(off) = rescale_u8(r, inv[0], mins[0]);
                        *row_rgb.get_unchecked_mut(off + 1) = rescale_u8(g, inv[1], mins[1]);
                        *row_rgb.get_unchecked_mut(off + 2) = rescale_u8(b, inv[2], mins[2]);
                    }
                }
            }
            return;
        }

    // Fast path 1: all 3 bands live in a single clean block with matching native storage
    if let (Some((r, bc, br, st)), Some((g, _, _, _)), Some((b, _, _, _))) = (
        srcs[0].single_block_u8(),
        srcs[1].single_block_u8(),
        srcs[2].single_block_u8(),
    )
        && srcs[0].same_grid_as(&srcs[1]) && srcs[0].same_grid_as(&srcs[2]) {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            bilinear_rgb_single_block_row(
                &u_row, &v_row, r, g, b, bc, br, st, inv, mins, row_rgb, |v| v as f32,
            );
            return;
        }
    if let (Some((r, bc, br, st)), Some((g, _, _, _)), Some((b, _, _, _))) = (
        srcs[0].single_block_u16(),
        srcs[1].single_block_u16(),
        srcs[2].single_block_u16(),
    )
        && srcs[0].same_grid_as(&srcs[1]) && srcs[0].same_grid_as(&srcs[2]) {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            bilinear_rgb_single_block_row(
                &u_row, &v_row, r, g, b, bc, br, st, inv, mins, row_rgb, |v| v as f32,
            );
            return;
        }
    if let (Some((r, bc, br, st)), Some((g, _, _, _)), Some((b, _, _, _))) = (
        srcs[0].single_block_flat(),
        srcs[1].single_block_flat(),
        srcs[2].single_block_flat(),
    )
        && srcs[0].same_grid_as(&srcs[1]) && srcs[0].same_grid_as(&srcs[2]) {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            bilinear_rgb_single_block_row(
                &u_row, &v_row, r, g, b, bc, br, st, inv, mins, row_rgb, |v| v,
            );
            return;
        }

    // Fast path 2: multi-block aligned grid — compute floor/frac + (bidx, p00, p01) ONCE per pixel
    if srcs[0].same_grid_as(&srcs[1]) && srcs[0].same_grid_as(&srcs[2]) {
        let s0 = &srcs[0];
        let s1 = &srcs[1];
        let s2 = &srcs[2];
        #[inline(always)]
        fn eval_band_at(
            blk: &cogzilla_codec::DecodedBand,
            p00: usize,
            p01: usize,
            fx: f32,
            fy: f32,
        ) -> f32 {
            let (v00, v10, v01, v11) = match &blk.data {
                cogzilla_codec::BandData::F32(s) => {
                    let sl = s.as_slice();
                    unsafe {
                        (
                            *sl.get_unchecked(p00),
                            *sl.get_unchecked(p00 + 1),
                            *sl.get_unchecked(p01),
                            *sl.get_unchecked(p01 + 1),
                        )
                    }
                }
                cogzilla_codec::BandData::F64(s) => unsafe {
                    (
                        *s.get_unchecked(p00) as f32,
                        *s.get_unchecked(p00 + 1) as f32,
                        *s.get_unchecked(p01) as f32,
                        *s.get_unchecked(p01 + 1) as f32,
                    )
                },
                cogzilla_codec::BandData::U16 { data, .. } => unsafe {
                    (
                        *data.get_unchecked(p00) as f32,
                        *data.get_unchecked(p00 + 1) as f32,
                        *data.get_unchecked(p01) as f32,
                        *data.get_unchecked(p01 + 1) as f32,
                    )
                },
                cogzilla_codec::BandData::U8 { data, .. } => unsafe {
                    (
                        *data.get_unchecked(p00) as f32,
                        *data.get_unchecked(p00 + 1) as f32,
                        *data.get_unchecked(p01) as f32,
                        *data.get_unchecked(p01 + 1) as f32,
                    )
                },
                cogzilla_codec::BandData::RgbU8Channel { rgb, channel, .. } => unsafe {
                    let ch = *channel as usize;
                    (
                        *rgb.get_unchecked(p00 * 3 + ch) as f32,
                        *rgb.get_unchecked((p00 + 1) * 3 + ch) as f32,
                        *rgb.get_unchecked(p01 * 3 + ch) as f32,
                        *rgb.get_unchecked((p01 + 1) * 3 + ch) as f32,
                    )
                },
            };
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            top + (bot - top) * fy
        }

        for_each_pixel_in_row(plan, py, |px, u, v| {
            let (x0, fx) = fast_floor_i32(u);
            let (y0, fy) = fast_floor_i32(v);
            let (r, g, b) = if let Some((bidx, p00, p01)) =
                s0.interior_quad_loc(x0 as u32, y0 as u32)
            {
                unsafe {
                    if let (Some(b0), Some(b1), Some(b2)) = (
                        *s0.blocks.get_unchecked(bidx),
                        *s1.blocks.get_unchecked(bidx),
                        *s2.blocks.get_unchecked(bidx),
                    ) && b1.width == b0.width
                        && b1.height == b0.height
                        && b2.width == b0.width
                        && b2.height == b0.height
                    {
                        (
                            eval_band_at(b0, p00, p01, fx, fy),
                            eval_band_at(b1, p00, p01, fx, fy),
                            eval_band_at(b2, p00, p01, fx, fy),
                        )
                    } else {
                        (
                            sample_bilinear::<true>(s0, u, v),
                            sample_bilinear::<true>(s1, u, v),
                            sample_bilinear::<true>(s2, u, v),
                        )
                    }
                }
            } else {
                (
                    sample_bilinear::<true>(s0, u, v),
                    sample_bilinear::<true>(s1, u, v),
                    sample_bilinear::<true>(s2, u, v),
                )
            };
            let off = px * 3;
            row_rgb[off] = rescale_u8(r, inv[0], mins[0]);
            row_rgb[off + 1] = rescale_u8(g, inv[1], mins[1]);
            row_rgb[off + 2] = rescale_u8(b, inv[2], mins[2]);
        });
        return;
    }

    // Fallback for heterogeneous band grids
    for_each_pixel_in_row(plan, py, |px, u, v| {
        let r = sample_bilinear::<true>(&srcs[0], u, v);
        let g = sample_bilinear::<true>(&srcs[1], u, v);
        let b = sample_bilinear::<true>(&srcs[2], u, v);
        let off = px * 3;
        row_rgb[off] = rescale_u8(r, inv[0], mins[0]);
        row_rgb[off + 1] = rescale_u8(g, inv[1], mins[1]);
        row_rgb[off + 2] = rescale_u8(b, inv[2], mins[2]);
    });
}

/// Pre-resolved single-block `Separable1D` Nearest plan hoisted outside `for py in 0..N`
/// in `render_rgb_specialized`, eliminating `N` redundant mode dispatches, `u_row`/`v_row` copies,
/// and `N²` `fast_floor_i32` X-index computations.
pub(super) enum CleanRgbNearestPlan<'a, const N: usize> {
    InterleavedRgbU8 {
        rgb: &'a [u8],
        v: &'a [f32; N],
        br_f: f32,
        stride3: usize,
        max_y: i32,
        x_off3: [usize; N],
        identity: bool,
    },
    PlanarU8 {
        r: &'a [u8],
        g: &'a [u8],
        b: &'a [u8],
        v: &'a [f32; N],
        br_f: f32,
        st: usize,
        max_y: i32,
        x_idx: [usize; N],
        identity: bool,
    },
    PlanarU16 {
        r: &'a [u16],
        g: &'a [u16],
        b: &'a [u16],
        v: &'a [f32; N],
        br_f: f32,
        st: usize,
        max_y: i32,
        x_idx: [usize; N],
    },
    PlanarF32 {
        r: &'a [f32],
        g: &'a [f32],
        b: &'a [f32],
        v: &'a [f32; N],
        br_f: f32,
        st: usize,
        max_y: i32,
        x_idx: [usize; N],
    },
    Generic,
}

impl<'a, const N: usize> CleanRgbNearestPlan<'a, N> {
    pub(super) fn build(
        plan: &'a CoordPlan<N>,
        srcs: &[BandSource<'a>; 3],
        inv: [f32; 3],
        mins: [f32; 3],
    ) -> Self {
        let CoordPlan::Separable1D { u, v, .. } = plan else {
            return Self::Generic;
        };
        let identity = mins == [0.0, 0.0, 0.0]
            && (inv[0] - 1.0).abs() < 1e-6
            && (inv[1] - 1.0).abs() < 1e-6
            && (inv[2] - 1.0).abs() < 1e-6;

        if let (Some((rgb0, 0, bc, br, st)), Some((rgb1, 1, _, _, _)), Some((rgb2, 2, _, _, _))) = (
            srcs[0].single_block_rgb_u8(),
            srcs[1].single_block_rgb_u8(),
            srcs[2].single_block_rgb_u8(),
        )
            && std::ptr::eq(rgb0.as_ptr(), rgb1.as_ptr())
                && std::ptr::eq(rgb0.as_ptr(), rgb2.as_ptr())
                && srcs[0].same_grid_as(&srcs[1])
                && srcs[0].same_grid_as(&srcs[2])
            {
                let bc_f = bc as f32;
                let max_x = (st as i32 - 1).max(0);
                let stride3 = st * 3;
                let max_y = ((rgb0.len() / stride3) as i32 - 1).max(0);
                let mut x_off3 = [0usize; N];
                for px in 0..N {
                    x_off3[px] =
                        (fast_floor_i32(u[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize) * 3;
                }
                return Self::InterleavedRgbU8 {
                    rgb: rgb0,
                    v,
                    br_f: br as f32,
                    stride3,
                    max_y,
                    x_off3,
                    identity,
                };
            }
        if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
            srcs[0].single_block_u8(),
            srcs[1].single_block_u8(),
            srcs[2].single_block_u8(),
        )
            && bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
                let bc_f = bc as f32;
                let max_x = (st as i32 - 1).max(0);
                let min_len = r.len().min(g.len()).min(b.len());
                let max_y = ((min_len / st) as i32 - 1).max(0);
                let mut x_idx = [0usize; N];
                for px in 0..N {
                    x_idx[px] = fast_floor_i32(u[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
                }
                return Self::PlanarU8 {
                    r,
                    g,
                    b,
                    v,
                    br_f: br as f32,
                    st,
                    max_y,
                    x_idx,
                    identity,
                };
            }
        if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
            srcs[0].single_block_u16(),
            srcs[1].single_block_u16(),
            srcs[2].single_block_u16(),
        )
            && bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
                let bc_f = bc as f32;
                let max_x = (st as i32 - 1).max(0);
                let min_len = r.len().min(g.len()).min(b.len());
                let max_y = ((min_len / st) as i32 - 1).max(0);
                let mut x_idx = [0usize; N];
                for px in 0..N {
                    x_idx[px] = fast_floor_i32(u[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
                }
                return Self::PlanarU16 {
                    r,
                    g,
                    b,
                    v,
                    br_f: br as f32,
                    st,
                    max_y,
                    x_idx,
                };
            }
        if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
            srcs[0].single_block_flat(),
            srcs[1].single_block_flat(),
            srcs[2].single_block_flat(),
        )
            && bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
                let bc_f = bc as f32;
                let max_x = (st as i32 - 1).max(0);
                let min_len = r.len().min(g.len()).min(b.len());
                let max_y = ((min_len / st) as i32 - 1).max(0);
                let mut x_idx = [0usize; N];
                for px in 0..N {
                    x_idx[px] = fast_floor_i32(u[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
                }
                return Self::PlanarF32 {
                    r,
                    g,
                    b,
                    v,
                    br_f: br as f32,
                    st,
                    max_y,
                    x_idx,
                };
            }
        Self::Generic
    }

    #[inline(always)]
    pub(super) fn sample_row(
        &self,
        plan: &CoordPlan<N>,
        py: usize,
        srcs: &[BandSource<'_>; 3],
        inv: [f32; 3],
        mins: [f32; 3],
        row_rgb: &mut [u8],
    ) {
        match self {
            Self::InterleavedRgbU8 {
                rgb,
                v,
                br_f,
                stride3,
                max_y,
                x_off3,
                identity,
            } => {
                let y = fast_floor_i32(v[py] + 0.5 - *br_f).0.clamp(0, *max_y) as usize;
                let y_off = y * *stride3;
                let row_src = &rgb[y_off..];
                if *identity {
                    for px in 0..N {
                        let o = x_off3[px];
                        let d = px * 3;
                        row_rgb[d] = row_src[o];
                        row_rgb[d + 1] = row_src[o + 1];
                        row_rgb[d + 2] = row_src[o + 2];
                    }
                } else {
                    for px in 0..N {
                        let o = x_off3[px];
                        let d = px * 3;
                        row_rgb[d] = rescale_u8(row_src[o] as f32, inv[0], mins[0]);
                        row_rgb[d + 1] = rescale_u8(row_src[o + 1] as f32, inv[1], mins[1]);
                        row_rgb[d + 2] = rescale_u8(row_src[o + 2] as f32, inv[2], mins[2]);
                    }
                }
            }
            Self::PlanarU8 {
                r,
                g,
                b,
                v,
                br_f,
                st,
                max_y,
                x_idx,
                identity,
            } => {
                let y = fast_floor_i32(v[py] + 0.5 - *br_f).0.clamp(0, *max_y) as usize;
                let y_off = y * *st;
                let rr = &r[y_off..];
                let rg = &g[y_off..];
                let rb = &b[y_off..];
                if *identity {
                    for px in 0..N {
                        let x = x_idx[px];
                        let d = px * 3;
                        row_rgb[d] = rr[x];
                        row_rgb[d + 1] = rg[x];
                        row_rgb[d + 2] = rb[x];
                    }
                } else {
                    for px in 0..N {
                        let x = x_idx[px];
                        let d = px * 3;
                        row_rgb[d] = rescale_u8(rr[x] as f32, inv[0], mins[0]);
                        row_rgb[d + 1] = rescale_u8(rg[x] as f32, inv[1], mins[1]);
                        row_rgb[d + 2] = rescale_u8(rb[x] as f32, inv[2], mins[2]);
                    }
                }
            }
            Self::PlanarU16 {
                r,
                g,
                b,
                v,
                br_f,
                st,
                max_y,
                x_idx,
            } => {
                let y = fast_floor_i32(v[py] + 0.5 - *br_f).0.clamp(0, *max_y) as usize;
                let y_off = y * *st;
                let rr = &r[y_off..];
                let rg = &g[y_off..];
                let rb = &b[y_off..];
                for px in 0..N {
                    let x = x_idx[px];
                    let d = px * 3;
                    row_rgb[d] = rescale_u8(rr[x] as f32, inv[0], mins[0]);
                    row_rgb[d + 1] = rescale_u8(rg[x] as f32, inv[1], mins[1]);
                    row_rgb[d + 2] = rescale_u8(rb[x] as f32, inv[2], mins[2]);
                }
            }
            Self::PlanarF32 {
                r,
                g,
                b,
                v,
                br_f,
                st,
                max_y,
                x_idx,
            } => {
                let y = fast_floor_i32(v[py] + 0.5 - *br_f).0.clamp(0, *max_y) as usize;
                let y_off = y * *st;
                let rr = &r[y_off..];
                let rg = &g[y_off..];
                let rb = &b[y_off..];
                for px in 0..N {
                    let x = x_idx[px];
                    let d = px * 3;
                    row_rgb[d] = rescale_u8(rr[x], inv[0], mins[0]);
                    row_rgb[d + 1] = rescale_u8(rg[x], inv[1], mins[1]);
                    row_rgb[d + 2] = rescale_u8(rb[x], inv[2], mins[2]);
                }
            }
            Self::Generic => sample_rgb_nearest_row_clean(plan, py, srcs, inv, mins, row_rgb),
        }
    }
}

fn render_rgb_specialized<const N: usize, const BILINEAR: bool, const CLEAN: bool>(
    plan: &CoordPlan<N>,
    srcs: &[BandSource<'_>; 3],
    style: &RgbStyle,
) -> Bytes {
    let inv = [
        255.0 / safe_rescale_denom(style.rescale_min[0], style.rescale_max[0]),
        255.0 / safe_rescale_denom(style.rescale_min[1], style.rescale_max[1]),
        255.0 / safe_rescale_denom(style.rescale_min[2], style.rescale_max[2]),
    ];
    let mins = style.rescale_min;

    match style.format {
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
            let mut out = BytesMut::with_capacity(N * N * 3 * 4);
            for b in 0..3 {
                for py in 0..N {
                    for_each_pixel_in_row(plan, py, |_px, u, v| {
                        let val = sample_val::<BILINEAR, CLEAN>(&srcs[b], u, v);
                        out.put_slice(&val.to_le_bytes());
                    });
                }
            }
            out.freeze()
        }
        OutputFormat::Png => SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            if CLEAN {
                let stride = 1 + N * 3;
                let total = N * stride;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                if sc.row_buf.len() < N * 3 {
                    sc.row_buf.resize(N * 3, 0);
                }
                let nearest_plan = if !BILINEAR {
                    Some(CleanRgbNearestPlan::build(plan, srcs, inv, mins))
                } else {
                    None
                };
                for py in 0..N {
                    let row_rgb = &mut sc.row_buf[..N * 3];
                    if BILINEAR {
                        sample_rgb_bilinear_row_clean(plan, py, srcs, inv, mins, row_rgb);
                    } else if let Some(ref np) = nearest_plan {
                        np.sample_row(plan, py, srcs, inv, mins, row_rgb);
                    }
                    filter_sub_rgb_row(row_rgb, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                }
                encode_filtered_png(N as u32, N as u32, 2, &sc.img_buf[..total])
            } else {
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
                        let r = sample_val::<BILINEAR, false>(&srcs[0], u, v);
                        let g = sample_val::<BILINEAR, false>(&srcs[1], u, v);
                        let b = sample_val::<BILINEAR, false>(&srcs[2], u, v);
                        let off = px * 4;
                        if r.is_nan() || g.is_nan() || b.is_nan() {
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
            }
        }),
        OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            if CLEAN {
                let total = N * N * 3;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                let buf = &mut sc.img_buf[..total];
                let nearest_plan = if !BILINEAR {
                    Some(CleanRgbNearestPlan::build(plan, srcs, inv, mins))
                } else {
                    None
                };
                for py in 0..N {
                    let row_off = py * N * 3;
                    if BILINEAR {
                        sample_rgb_bilinear_row_clean(
                            plan,
                            py,
                            srcs,
                            inv,
                            mins,
                            &mut buf[row_off..row_off + N * 3],
                        );
                    } else if let Some(ref np) = nearest_plan {
                        np.sample_row(
                            plan,
                            py,
                            srcs,
                            inv,
                            mins,
                            &mut buf[row_off..row_off + N * 3],
                        );
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
            } else {
                let total = N * N * 4;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                let buf = &mut sc.img_buf[..total];
                for py in 0..N {
                    let row_off = py * N * 4;
                    for_each_pixel_in_row(plan, py, |px, u, v| {
                        let r = sample_val::<BILINEAR, false>(&srcs[0], u, v);
                        let g = sample_val::<BILINEAR, false>(&srcs[1], u, v);
                        let b = sample_val::<BILINEAR, false>(&srcs[2], u, v);
                        let off = row_off + px * 4;
                        if r.is_nan() || g.is_nan() || b.is_nan() {
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
            }
        }),
    }
}

pub fn render_rgb_bands<const N: usize>(
    plan: &CoordPlan<N>,
    srcs: &[BandSource<'_>; 3],
    style: &RgbStyle,
) -> Bytes {
    if srcs[0].all_nodata || srcs[1].all_nodata || srcs[2].all_nodata {
        return empty_tile_bytes(N as u32, style.format, 3);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = plan.all_in_bounds()
        && srcs[0].is_clean_inbounds(c0, c1, r0, r1)
        && srcs[1].is_clean_inbounds(c0, c1, r0, r1)
        && srcs[2].is_clean_inbounds(c0, c1, r0, r1);

    match (style.resample, clean) {
        (ResampleMode::Nearest, true) => render_rgb_specialized::<N, false, true>(plan, srcs, style),
        (ResampleMode::Nearest, false) => render_rgb_specialized::<N, false, false>(plan, srcs, style),
        (mode, _) if plan_separable_conv_scale(plan, mode).is_some() => {
            let (sx, sy) = coord_plan_scale(plan);
            let same_dims = srcs[1].ifd_w == srcs[0].ifd_w
                && srcs[1].ifd_h == srcs[0].ifd_h
                && srcs[2].ifd_w == srcs[0].ifd_w
                && srcs[2].ifd_h == srcs[0].ifd_h;
            if let (true, CoordPlan::Separable1D { u, v, .. }) = (same_dims, plan) {
                let cols = AxisTapTable::build(mode, &u[..], sx, srcs[0].ifd_w);
                let rows = AxisTapTable::build(mode, &v[..], sy, srcs[0].ifd_h);
                // Clean RGB JPEG/WebP/PNG hot path: stream row-by-row from L1-resident ring
                // buffers straight into `sc.img_buf`, avoiding 3 MiB of intermediate planes.
                if clean
                    && matches!(
                        style.format,
                        OutputFormat::Jpeg { .. } | OutputFormat::Webp { .. } | OutputFormat::Png
                    )
                {
                    let inv = [
                        255.0 / safe_rescale_denom(style.rescale_min[0], style.rescale_max[0]),
                        255.0 / safe_rescale_denom(style.rescale_min[1], style.rescale_max[1]),
                        255.0 / safe_rescale_denom(style.rescale_min[2], style.rescale_max[2]),
                    ];
                    let mins = style.rescale_min;
                    return SCRATCH.with(|cell| {
                        let mut guard = cell.borrow_mut();
                        let sc = &mut *guard;
                        if matches!(style.format, OutputFormat::Png) {
                            let stride = 1 + N * 3;
                            let total = N * stride;
                            if sc.img_buf.len() < total {
                                sc.img_buf.resize(total, 0);
                            }
                            if sc.row_buf.len() < N * 3 {
                                sc.row_buf.resize(N * 3, 0);
                            }
                            widened_rgb_separable_stream_clean(
                                u, v, srcs, mode, sx, sy, &cols, &rows,
                                |py, rr, rg, rb| {
                                    let row = &mut sc.row_buf[..N * 3];
                                    for px in 0..N {
                                        let off = px * 3;
                                        row[off] = rescale_u8(rr[px], inv[0], mins[0]);
                                        row[off + 1] = rescale_u8(rg[px], inv[1], mins[1]);
                                        row[off + 2] = rescale_u8(rb[px], inv[2], mins[2]);
                                    }
                                    filter_sub_rgb_row(
                                        row,
                                        &mut sc.img_buf[py * stride..(py + 1) * stride],
                                    );
                                },
                            );
                            return encode_filtered_png(N as u32, N as u32, 2, &sc.img_buf[..total]);
                        }
                        let total = N * N * 3;
                        if sc.img_buf.len() < total {
                            sc.img_buf.resize(total, 0);
                        }
                        let buf = &mut sc.img_buf[..total];
                        widened_rgb_separable_stream_clean(
                            u, v, srcs, mode, sx, sy, &cols, &rows,
                            |py, rr, rg, rb| {
                                let row = &mut buf[py * N * 3..(py + 1) * N * 3];
                                for px in 0..N {
                                    let off = px * 3;
                                    row[off] = rescale_u8(rr[px], inv[0], mins[0]);
                                    row[off + 1] = rescale_u8(rg[px], inv[1], mins[1]);
                                    row[off + 2] = rescale_u8(rb[px], inv[2], mins[2]);
                                }
                            },
                        );
                        match style.format {
                            OutputFormat::Jpeg { quality: jq } => {
                                encode_jpeg_rgb(N as u32, N as u32, buf, jq)
                            }
                            OutputFormat::Webp { quality } => {
                                encode_webp_rgb(N as u32, N as u32, buf, quality)
                            }
                            _ => unreachable!(),
                        }
                    });
                }
                RGB_PLANE_SCRATCH.with(|cell| {
                    let mut planes = cell.borrow_mut();
                    let len = N * N;
                    for p in planes.iter_mut() {
                        if p.len() < len {
                            p.resize(len, 0.0);
                        }
                    }
                    let [p0, p1, p2] = &mut *planes;
                    widened_plane_separable_into_with_tables(
                        &u[..], &v[..], &srcs[0], mode, sx, sy, &cols, &rows, &mut p0[..len],
                    );
                    widened_plane_separable_into_with_tables(
                        &u[..], &v[..], &srcs[1], mode, sx, sy, &cols, &rows, &mut p1[..len],
                    );
                    widened_plane_separable_into_with_tables(
                        &u[..], &v[..], &srcs[2], mode, sx, sy, &cols, &rows, &mut p2[..len],
                    );
                    render_rgb_planes::<N>([&p0[..len], &p1[..len], &p2[..len]], style, clean)
                })
            } else {
                let planes = [
                    widened_plane(plan, &srcs[0], mode, sx, sy),
                    widened_plane(plan, &srcs[1], mode, sx, sy),
                    widened_plane(plan, &srcs[2], mode, sx, sy),
                ];
                render_rgb_planes::<N>([&planes[0], &planes[1], &planes[2]], style, clean)
            }
        }
        (ResampleMode::Bilinear, true) => render_rgb_specialized::<N, true, true>(plan, srcs, style),
        (ResampleMode::Bilinear, false) => render_rgb_specialized::<N, true, false>(plan, srcs, style),
        (mode, _) => {
            let (sx, sy) = coord_plan_scale(plan);
            render_rgb_cubic::<N>(plan, style, clean, |b, u, v| {
                if mode == ResampleMode::Cubic {
                    sample_cubic(&srcs[b], u, v)
                } else {
                    sample_general_unwidened(&srcs[b], u, v, sx, sy, mode)
                }
            })
        }
    }
}

/// Direct encoder for 3 precomputed `N x N` f32 planes (zero `index_plan` allocation, direct flat
/// indexing, and 3-byte RGB fast path for clean JPEG/WebP without RGBA alpha stripping).
fn render_rgb_planes<const N: usize>(planes: [&[f32]; 3], style: &RgbStyle, clean: bool) -> Bytes {
    let inv = [
        255.0 / safe_rescale_denom(style.rescale_min[0], style.rescale_max[0]),
        255.0 / safe_rescale_denom(style.rescale_min[1], style.rescale_max[1]),
        255.0 / safe_rescale_denom(style.rescale_min[2], style.rescale_max[2]),
    ];
    let mins = style.rescale_min;
    let [pr, pg, pb] = planes;
    match style.format {
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
            let mut out = BytesMut::with_capacity(N * N * 12);
            for plane in planes {
                for &v in plane {
                    out.put_slice(&v.to_le_bytes());
                }
            }
            out.freeze()
        }
        OutputFormat::Png => SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            if clean {
                let stride = 1 + N * 3;
                let total = N * stride;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                if sc.row_buf.len() < N * 3 {
                    sc.row_buf.resize(N * 3, 0);
                }
                for py in 0..N {
                    let row = &mut sc.row_buf[..N * 3];
                    let base = py * N;
                    let rr = &pr[base..base + N];
                    let rg = &pg[base..base + N];
                    let rb = &pb[base..base + N];
                    for px in 0..N {
                        let off = px * 3;
                        row[off] = rescale_u8(rr[px], inv[0], mins[0]);
                        row[off + 1] = rescale_u8(rg[px], inv[1], mins[1]);
                        row[off + 2] = rescale_u8(rb[px], inv[2], mins[2]);
                    }
                    filter_sub_rgb_row(row, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                }
                return encode_filtered_png(N as u32, N as u32, 2, &sc.img_buf[..total]);
            }
            let stride = 1 + N * 4;
            let total = N * stride;
            if sc.img_buf.len() < total {
                sc.img_buf.resize(total, 0);
            }
            if sc.row_buf.len() < N * 4 {
                sc.row_buf.resize(N * 4, 0);
            }
            for py in 0..N {
                let row = &mut sc.row_buf[..N * 4];
                let base = py * N;
                let rr = &pr[base..base + N];
                let rg = &pg[base..base + N];
                let rb = &pb[base..base + N];
                for px in 0..N {
                    let (r, g, b) = (rr[px], rg[px], rb[px]);
                    let off = px * 4;
                    if r.is_nan() || g.is_nan() || b.is_nan() {
                        row[off..off + 4].fill(0);
                    } else {
                        row[off] = rescale_u8(r, inv[0], mins[0]);
                        row[off + 1] = rescale_u8(g, inv[1], mins[1]);
                        row[off + 2] = rescale_u8(b, inv[2], mins[2]);
                        row[off + 3] = 255;
                    }
                }
                filter_sub_rgba_row(row, &mut sc.img_buf[py * stride..(py + 1) * stride]);
            }
            encode_filtered_png(N as u32, N as u32, 6, &sc.img_buf[..total])
        }),
        OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            if clean && !matches!(style.format, OutputFormat::RawRgba) {
                let total = N * N * 3;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                let buf = &mut sc.img_buf[..total];
                for i in 0..N * N {
                    let off = i * 3;
                    buf[off] = rescale_u8(pr[i], inv[0], mins[0]);
                    buf[off + 1] = rescale_u8(pg[i], inv[1], mins[1]);
                    buf[off + 2] = rescale_u8(pb[i], inv[2], mins[2]);
                }
                return match style.format {
                    OutputFormat::Jpeg { quality: jq } => encode_jpeg_rgb(N as u32, N as u32, buf, jq),
                    OutputFormat::Webp { quality } => encode_webp_rgb(N as u32, N as u32, buf, quality),
                    _ => unreachable!(),
                };
            }
            let total = N * N * 4;
            if sc.img_buf.len() < total {
                sc.img_buf.resize(total, 0);
            }
            let buf = &mut sc.img_buf[..total];
            for i in 0..N * N {
                let (r, g, b) = (pr[i], pg[i], pb[i]);
                let off = i * 4;
                if r.is_nan() || g.is_nan() || b.is_nan() {
                    buf[off..off + 4].fill(0);
                } else {
                    buf[off] = rescale_u8(r, inv[0], mins[0]);
                    buf[off + 1] = rescale_u8(g, inv[1], mins[1]);
                    buf[off + 2] = rescale_u8(b, inv[2], mins[2]);
                    buf[off + 3] = 255;
                }
            }
            match style.format {
                OutputFormat::Jpeg { quality: jq } => encode_jpeg_rgba(N as u32, N as u32, buf, jq),
                OutputFormat::Webp { quality } => encode_webp_rgba(N as u32, N as u32, buf, quality),
                OutputFormat::RawRgba => Bytes::copy_from_slice(buf),
                _ => unreachable!(),
            }
        }),
    }
}

fn render_rgb_cubic<const N: usize>(
    plan: &CoordPlan<N>,
    style: &RgbStyle,
    clean: bool,
    sample_band: impl Fn(usize, f32, f32) -> f32,
) -> Bytes {
    let inv = [
        255.0 / safe_rescale_denom(style.rescale_min[0], style.rescale_max[0]),
        255.0 / safe_rescale_denom(style.rescale_min[1], style.rescale_max[1]),
        255.0 / safe_rescale_denom(style.rescale_min[2], style.rescale_max[2]),
    ];
    let mins = style.rescale_min;
    match style.format {
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
            let mut out = BytesMut::with_capacity(N * N * 12);
            for b in 0..3 {
                for py in 0..N {
                    for_each_pixel_in_row(plan, py, |_px, u, v| {
                        out.put_slice(&sample_band(b, u, v).to_le_bytes());
                    });
                }
            }
            out.freeze()
        }
        OutputFormat::Png => SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            if clean {
                let stride = 1 + N * 3;
                let total = N * stride;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                if sc.row_buf.len() < N * 3 {
                    sc.row_buf.resize(N * 3, 0);
                }
                for py in 0..N {
                    let row = &mut sc.row_buf[..N * 3];
                    for_each_pixel_in_row(plan, py, |px, u, v| {
                        let r = sample_band(0, u, v);
                        let g = sample_band(1, u, v);
                        let b = sample_band(2, u, v);
                        let off = px * 3;
                        row[off] = rescale_u8(r, inv[0], mins[0]);
                        row[off + 1] = rescale_u8(g, inv[1], mins[1]);
                        row[off + 2] = rescale_u8(b, inv[2], mins[2]);
                    });
                    filter_sub_rgb_row(row, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                }
                return encode_filtered_png(N as u32, N as u32, 2, &sc.img_buf[..total]);
            }
            let stride = 1 + N * 4;
            let total = N * stride;
            if sc.img_buf.len() < total {
                sc.img_buf.resize(total, 0);
            }
            if sc.row_buf.len() < N * 4 {
                sc.row_buf.resize(N * 4, 0);
            }
            for py in 0..N {
                let row = &mut sc.row_buf[..N * 4];
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    let r = sample_band(0, u, v);
                    let g = sample_band(1, u, v);
                    let b = sample_band(2, u, v);
                    let off = px * 4;
                    if r.is_nan() || g.is_nan() || b.is_nan() {
                        row[off..off + 4].fill(0);
                    } else {
                        row[off] = rescale_u8(r, inv[0], mins[0]);
                        row[off + 1] = rescale_u8(g, inv[1], mins[1]);
                        row[off + 2] = rescale_u8(b, inv[2], mins[2]);
                        row[off + 3] = 255;
                    }
                });
                filter_sub_rgba_row(row, &mut sc.img_buf[py * stride..(py + 1) * stride]);
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
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    let r = sample_band(0, u, v);
                    let g = sample_band(1, u, v);
                    let b = sample_band(2, u, v);
                    let off = (py * N + px) * 4;
                    if r.is_nan() || g.is_nan() || b.is_nan() {
                        buf[off..off + 4].fill(0);
                    } else {
                        buf[off] = rescale_u8(r, inv[0], mins[0]);
                        buf[off + 1] = rescale_u8(g, inv[1], mins[1]);
                        buf[off + 2] = rescale_u8(b, inv[2], mins[2]);
                        buf[off + 3] = 255;
                    }
                });
            }
            match style.format {
                OutputFormat::Jpeg { quality: jq } => encode_jpeg_rgba(N as u32, N as u32, buf, jq),
                OutputFormat::Webp { quality } => encode_webp_rgba(N as u32, N as u32, buf, quality),
                OutputFormat::RawRgba => Bytes::copy_from_slice(buf),
                _ => unreachable!(),
            }
        }),
    }
}

