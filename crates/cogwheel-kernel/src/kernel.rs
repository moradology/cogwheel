use std::cell::RefCell;

use bytes::{BufMut, Bytes, BytesMut};
use cogwheel_proj::{CoordPlan, GRID_DIM};

use crate::bandmath::{SpectralExpr, fast_db_from_power};
use crate::colormap::{ColormapLut, ColormapName};
use crate::png_fast::{
    empty_transparent_png, encode_filtered_png, filter_sub_rgb_row, filter_sub_rgba_row,
};
use crate::source::BandSource;
use crate::webp_fast::{empty_transparent_webp, encode_webp_rgb, encode_webp_rgba};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMode {
    Nearest,
    Bilinear,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Png,
    /// `quality: None` = lossless WebP, `Some(q)` = lossy quality (e.g. `82.0`).
    Webp { quality: Option<f32> },
    RawF32,
}

#[derive(Debug, Clone, Copy)]
pub struct SingleBandStyle {
    pub rescale_min: f32,
    pub rescale_max: f32,
    pub colormap: ColormapName,
    pub resample: ResampleMode,
    pub format: OutputFormat,
}

#[derive(Debug, Clone, Copy)]
pub struct RgbStyle {
    pub rescale_min: [f32; 3],
    pub rescale_max: [f32; 3],
    pub resample: ResampleMode,
    pub format: OutputFormat,
}

struct KernelScratch {
    img_buf: Vec<u8>,
    row_buf: Vec<u8>,
    f32_row: Vec<f32>,
}

impl KernelScratch {
    fn new() -> Self {
        Self {
            img_buf: Vec::with_capacity(512 * (1 + 512 * 4)),
            row_buf: vec![0u8; 512 * 4],
            f32_row: vec![0.0f32; 512],
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<KernelScratch> = RefCell::new(KernelScratch::new());
}

// ============================================================================
// SIMD Fast-Path Row Bilinear Samplers (Separable1D + Clean + Single-Block)
// ============================================================================

/// Compute an entire output row of `N` bilinearly resampled `f32` values directly from a single
/// flat source block using ARM64 NEON 4-wide FMA (hoisting `row0_ptr`, `row1_ptr`, and `fy`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bilinear_row_flat_neon<const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0_ptr: *const f32,
    row1_ptr: *const f32,
    fy: f32,
    out_row: &mut [f32],
) {
    use std::arch::aarch64::*;
    let v_fy = unsafe { vdupq_n_f32(fy) };
    let v_base = unsafe { vdupq_n_f32(base_col as f32) };
    let mut px = 0usize;

    while px + 4 <= N {
        unsafe {
            let u = vsubq_f32(vld1q_f32(u_coords.as_ptr().add(px)), v_base);
            let x0_f = vrndmq_f32(u);
            let fx = vsubq_f32(u, x0_f);
            let x0_i = vcvtq_s32_f32(x0_f);

            let c0 = vgetq_lane_s32::<0>(x0_i) as usize;
            let c1 = vgetq_lane_s32::<1>(x0_i) as usize;
            let c2 = vgetq_lane_s32::<2>(x0_i) as usize;
            let c3 = vgetq_lane_s32::<3>(x0_i) as usize;

            // Load adjacent (v00, v10) pairs with single 64-bit loads + unzip
            let t0 = vld1_f32(row0_ptr.add(c0));
            let t1 = vld1_f32(row0_ptr.add(c1));
            let t2 = vld1_f32(row0_ptr.add(c2));
            let t3 = vld1_f32(row0_ptr.add(c3));
            let t01 = vcombine_f32(t0, t1);
            let t23 = vcombine_f32(t2, t3);
            let v00 = vuzp1q_f32(t01, t23);
            let v10 = vuzp2q_f32(t01, t23);

            let b0 = vld1_f32(row1_ptr.add(c0));
            let b1 = vld1_f32(row1_ptr.add(c1));
            let b2 = vld1_f32(row1_ptr.add(c2));
            let b3 = vld1_f32(row1_ptr.add(c3));
            let b01 = vcombine_f32(b0, b1);
            let b23 = vcombine_f32(b2, b3);
            let v01 = vuzp1q_f32(b01, b23);
            let v11 = vuzp2q_f32(b01, b23);

            let top = vfmaq_f32(v00, vsubq_f32(v10, v00), fx);
            let bot = vfmaq_f32(v01, vsubq_f32(v11, v01), fx);
            let res = vfmaq_f32(top, vsubq_f32(bot, top), v_fy);

            vst1q_f32(out_row.as_mut_ptr().add(px), res);
        }
        px += 4;
    }
}

/// x86_64 AVX2+FMA 8-wide bilinear row sampler for flat single-block `Separable1D + CleanFastPath`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilinear_row_flat_avx2<const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0_ptr: *const f32,
    row1_ptr: *const f32,
    fy: f32,
    out_row: &mut [f32],
) {
    use std::arch::x86_64::*;
    let v_fy = _mm256_set1_ps(fy);
    let v_base = _mm256_set1_ps(base_col as f32);
    let mut px = 0usize;

    while px + 8 <= N {
        let u = _mm256_sub_ps(_mm256_loadu_ps(u_coords.as_ptr().add(px)), v_base);
        let x0_f = _mm256_floor_ps(u);
        let fx = _mm256_sub_ps(u, x0_f);
        let idx = _mm256_cvtps_epi32(x0_f);

        let v00 = _mm256_i32gather_ps::<4>(row0_ptr, idx);
        let v10 = _mm256_i32gather_ps::<4>(row0_ptr.add(1), idx);
        let v01 = _mm256_i32gather_ps::<4>(row1_ptr, idx);
        let v11 = _mm256_i32gather_ps::<4>(row1_ptr.add(1), idx);

        let top = _mm256_fmadd_ps(_mm256_sub_ps(v10, v00), fx, v00);
        let bot = _mm256_fmadd_ps(_mm256_sub_ps(v11, v01), fx, v01);
        let res = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), v_fy, top);

        _mm256_storeu_ps(out_row.as_mut_ptr().add(px), res);
        px += 8;
    }
}

#[inline(always)]
fn bilinear_row_flat_dispatch<const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0: &[f32],
    row1: &[f32],
    fy: f32,
    out_row: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            bilinear_row_flat_neon::<N>(u_coords, base_col, row0.as_ptr(), row1.as_ptr(), fy, out_row);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                bilinear_row_flat_avx2::<N>(u_coords, base_col, row0.as_ptr(), row1.as_ptr(), fy, out_row);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    {
        let bc = base_col as f32;
        for px in 0..N {
            let u = u_coords[px] - bc;
            let x0_f = u.floor();
            let fx = u - x0_f;
            let x0 = x0_f as usize;
            let v00 = unsafe { *row0.get_unchecked(x0) };
            let v10 = unsafe { *row0.get_unchecked(x0 + 1) };
            let v01 = unsafe { *row1.get_unchecked(x0) };
            let v11 = unsafe { *row1.get_unchecked(x0 + 1) };
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            out_row[px] = top + (bot - top) * fy;
        }
    }
}

// ============================================================================
// General Per-Pixel Samplers
// ============================================================================

#[inline(always)]
fn sample_bilinear<const CLEAN: bool>(src: &BandSource<'_>, u: f32, v: f32) -> f32 {
    if !CLEAN && (!u.is_finite() || !v.is_finite()) {
        return f32::NAN;
    }
    let x0_f = u.floor();
    let y0_f = v.floor();
    let fx = u - x0_f;
    let fy = v - y0_f;
    let x0 = x0_f as i32;
    let y0 = y0_f as i32;

    if CLEAN {
        let ux0 = x0 as u32;
        let uy0 = y0 as u32;
        let ux1 = ux0 + 1;
        let uy1 = uy0 + 1;
        let v00 = src.sample_inbounds(ux0, uy0);
        let v10 = src.sample_inbounds(ux1, uy0);
        let v01 = src.sample_inbounds(ux0, uy1);
        let v11 = src.sample_inbounds(ux1, uy1);
        let top = v00 + (v10 - v00) * fx;
        let bot = v01 + (v11 - v01) * fx;
        top + (bot - top) * fy
    } else {
        let v00 = src.sample_checked(x0, y0);
        let v10 = src.sample_checked(x0 + 1, y0);
        let v01 = src.sample_checked(x0, y0 + 1);
        let v11 = src.sample_checked(x0 + 1, y0 + 1);

        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        let mut sum = 0.0f32;
        let mut wsum = 0.0f32;
        if !v00.is_nan() {
            sum += v00 * w00;
            wsum += w00;
        }
        if !v10.is_nan() {
            sum += v10 * w10;
            wsum += w10;
        }
        if !v01.is_nan() {
            sum += v01 * w01;
            wsum += w01;
        }
        if !v11.is_nan() {
            sum += v11 * w11;
            wsum += w11;
        }
        if wsum > 1e-6 {
            sum / wsum
        } else {
            f32::NAN
        }
    }
}

#[inline(always)]
fn sample_nearest<const CLEAN: bool>(src: &BandSource<'_>, u: f32, v: f32) -> f32 {
    if !CLEAN && (!u.is_finite() || !v.is_finite()) {
        return f32::NAN;
    }
    let x = (u + 0.5).floor() as i32;
    let y = (v + 0.5).floor() as i32;
    if CLEAN {
        src.sample_inbounds(x as u32, y as u32)
    } else {
        src.sample_checked(x, y)
    }
}

#[inline(always)]
fn sample_val<const BILINEAR: bool, const CLEAN: bool>(
    src: &BandSource<'_>,
    u: f32,
    v: f32,
) -> f32 {
    if BILINEAR {
        sample_bilinear::<CLEAN>(src, u, v)
    } else {
        sample_nearest::<CLEAN>(src, u, v)
    }
}

#[inline(always)]
fn rescale_u8(val: f32, inv_range: f32, min_val: f32) -> u8 {
    ((val - min_val) * inv_range).clamp(0.0, 255.0) as u8
}

#[inline(always)]
fn for_each_pixel_in_row<const N: usize, F: FnMut(usize, f32, f32)>(
    plan: &CoordPlan<N>,
    py: usize,
    mut f: F,
) {
    match plan {
        CoordPlan::Separable1D { u, v, .. } => {
            let vy = v[py];
            for px in 0..N {
                f(px, u[px], vy);
            }
        }
        CoordPlan::Grid2D { grid_u, grid_v, .. } => {
            let cell_px = N / (GRID_DIM - 1);
            let inv_cell = 1.0 / (cell_px as f32);
            let gy = (py / cell_px).min(GRID_DIM - 2);
            let ty = (py - gy * cell_px) as f32 * inv_cell;
            let row0 = gy * GRID_DIM;
            let row1 = (gy + 1) * GRID_DIM;

            for gx in 0..(GRID_DIM - 1) {
                let u_l = grid_u[row0 + gx] + (grid_u[row1 + gx] - grid_u[row0 + gx]) * ty;
                let u_r = grid_u[row0 + gx + 1] + (grid_u[row1 + gx + 1] - grid_u[row0 + gx + 1]) * ty;
                let v_l = grid_v[row0 + gx] + (grid_v[row1 + gx] - grid_v[row0 + gx]) * ty;
                let v_r = grid_v[row0 + gx + 1] + (grid_v[row1 + gx + 1] - grid_v[row0 + gx + 1]) * ty;

                let du = (u_r - u_l) * inv_cell;
                let dv = (v_r - v_l) * inv_cell;
                let mut u = u_l;
                let mut v = v_l;
                let base_x = gx * cell_px;
                for k in 0..cell_px {
                    f(base_x + k, u, v);
                    u += du;
                    v += dv;
                }
            }
        }
        CoordPlan::Exact2D { u, v, .. } => {
            let row_off = py * N;
            for px in 0..N {
                f(px, u[row_off + px], v[row_off + px]);
            }
        }
    }
}

pub fn empty_tile_bytes(tile_size: u32, format: OutputFormat, num_bands: usize) -> Bytes {
    match format {
        OutputFormat::Png => empty_transparent_png(tile_size),
        OutputFormat::Webp { .. } => empty_transparent_webp(tile_size),
        OutputFormat::RawF32 => {
            let count = (tile_size as usize) * (tile_size as usize) * num_bands;
            let mut buf = BytesMut::with_capacity(count * 4);
            let nan_bytes = f32::NAN.to_le_bytes();
            for _ in 0..count {
                buf.put_slice(&nan_bytes);
            }
            buf.freeze()
        }
    }
}

// ============================================================================
// Monomorphized Single-Band Fused Kernels (with SIMD Flat-Block Fast Path!)
// ============================================================================

fn render_single_specialized<const N: usize, const BILINEAR: bool, const CLEAN: bool>(
    plan: &CoordPlan<N>,
    src: &BandSource<'_>,
    style: &SingleBandStyle,
) -> Bytes {
    let range = (style.rescale_max - style.rescale_min).max(1e-12);
    let inv_range = 255.0 / range;
    let min_val = style.rescale_min;

    // Ultra-hot SIMD path: Separable1D + Bilinear + CleanFastPath + Single-Block!
    if BILINEAR && CLEAN {
        if let (CoordPlan::Separable1D { u, v, .. }, Some((flat, base_col, base_row, row_stride))) =
            (plan, src.single_block_flat())
        {
            match style.format {
                OutputFormat::RawF32 => {
                    let mut out = BytesMut::zeroed(N * N * 4);
                    let out_f32 = unsafe {
                        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, N * N)
                    };
                    for py in 0..N {
                        let vy = v[py] - (base_row as f32);
                        let y0_f = vy.floor();
                        let fy = vy - y0_f;
                        let y0 = y0_f as usize;
                        let r0 = &flat[y0 * row_stride..];
                        let r1 = &flat[(y0 + 1) * row_stride..];
                        bilinear_row_flat_dispatch::<N>(
                            u,
                            base_col,
                            r0,
                            r1,
                            fy,
                            &mut out_f32[py * N..(py + 1) * N],
                        );
                    }
                    return out.freeze();
                }
                OutputFormat::Png => {
                    let lut: &ColormapLut = style.colormap.lut();
                    return SCRATCH.with(|cell| {
                        let mut guard = cell.borrow_mut();
                        let sc = &mut *guard;
                        let stride = 1 + N * 3;
                        let total = N * stride;
                        if sc.img_buf.len() < total {
                            sc.img_buf.resize(total, 0);
                        }
                        if sc.row_buf.len() < N * 3 {
                            sc.row_buf.resize(N * 3, 0);
                        }
                        if sc.f32_row.len() < N {
                            sc.f32_row.resize(N, 0.0);
                        }
                        for py in 0..N {
                            let vy = v[py] - (base_row as f32);
                            let y0_f = vy.floor();
                            let fy = vy - y0_f;
                            let y0 = y0_f as usize;
                            let r0 = &flat[y0 * row_stride..];
                            let r1 = &flat[(y0 + 1) * row_stride..];
                            bilinear_row_flat_dispatch::<N>(
                                u,
                                base_col,
                                r0,
                                r1,
                                fy,
                                &mut sc.f32_row[..N],
                            );
                            let row_rgb = &mut sc.row_buf[..N * 3];
                            for px in 0..N {
                                let idx = rescale_u8(sc.f32_row[px], inv_range, min_val) as usize;
                                let rgb = unsafe { *lut.rgb.get_unchecked(idx) };
                                let off = px * 3;
                                row_rgb[off] = rgb[0];
                                row_rgb[off + 1] = rgb[1];
                                row_rgb[off + 2] = rgb[2];
                            }
                            filter_sub_rgb_row(
                                row_rgb,
                                &mut sc.img_buf[py * stride..(py + 1) * stride],
                            );
                        }
                        encode_filtered_png(N as u32, N as u32, 2, &sc.img_buf[..total])
                    });
                }
                OutputFormat::Webp { quality } => {
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
                            let vy = v[py] - (base_row as f32);
                            let y0_f = vy.floor();
                            let fy = vy - y0_f;
                            let y0 = y0_f as usize;
                            let r0 = &flat[y0 * row_stride..];
                            let r1 = &flat[(y0 + 1) * row_stride..];
                            bilinear_row_flat_dispatch::<N>(
                                u,
                                base_col,
                                r0,
                                r1,
                                fy,
                                &mut sc.f32_row[..N],
                            );
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
                        encode_webp_rgb(N as u32, N as u32, &sc.img_buf[..total], quality)
                    });
                }
            }
        }
    }

    match style.format {
        OutputFormat::RawF32 => {
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
                if CLEAN {
                    let stride = 1 + N * 3;
                    let total = N * stride;
                    if sc.img_buf.len() < total {
                        sc.img_buf.resize(total, 0);
                    }
                    if sc.row_buf.len() < N * 3 {
                        sc.row_buf.resize(N * 3, 0);
                    }
                    for py in 0..N {
                        let row_rgb = &mut sc.row_buf[..N * 3];
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = sample_val::<BILINEAR, true>(src, u, v);
                            let idx = rescale_u8(val, inv_range, min_val) as usize;
                            let rgb = unsafe { *lut.rgb.get_unchecked(idx) };
                            let off = px * 3;
                            row_rgb[off] = rgb[0];
                            row_rgb[off + 1] = rgb[1];
                            row_rgb[off + 2] = rgb[2];
                        });
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
                        let row_u32 = unsafe {
                            std::slice::from_raw_parts_mut(row_rgba.as_mut_ptr() as *mut u32, N)
                        };
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = sample_val::<BILINEAR, false>(src, u, v);
                            if val.is_nan() {
                                row_u32[px] = 0;
                            } else {
                                let idx = rescale_u8(val, inv_range, min_val) as usize;
                                row_u32[px] = unsafe { *lut.rgba_u32.get_unchecked(idx) };
                            }
                        });
                        filter_sub_rgba_row(
                            row_rgba,
                            &mut sc.img_buf[py * stride..(py + 1) * stride],
                        );
                    }
                    encode_filtered_png(N as u32, N as u32, 6, &sc.img_buf[..total])
                }
            })
        }
        OutputFormat::Webp { quality } => {
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
                    encode_webp_rgb(N as u32, N as u32, buf, quality)
                } else {
                    let total = N * N * 4;
                    if sc.img_buf.len() < total {
                        sc.img_buf.resize(total, 0);
                    }
                    let buf = &mut sc.img_buf[..total];
                    let buf_u32 = unsafe {
                        std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u32, N * N)
                    };
                    for py in 0..N {
                        let row_off = py * N;
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = sample_val::<BILINEAR, false>(src, u, v);
                            buf_u32[row_off + px] = if val.is_nan() {
                                0
                            } else {
                                let idx = rescale_u8(val, inv_range, min_val) as usize;
                                unsafe { *lut.rgba_u32.get_unchecked(idx) }
                            };
                        });
                    }
                    encode_webp_rgba(N as u32, N as u32, buf, quality)
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
    let clean = !matches!(plan, CoordPlan::Exact2D { .. }) && src.is_clean_inbounds(c0, c1, r0, r1);

    match (style.resample, clean) {
        (ResampleMode::Bilinear, true) => render_single_specialized::<N, true, true>(plan, src, style),
        (ResampleMode::Bilinear, false) => render_single_specialized::<N, true, false>(plan, src, style),
        (ResampleMode::Nearest, true) => render_single_specialized::<N, false, true>(plan, src, style),
        (ResampleMode::Nearest, false) => render_single_specialized::<N, false, false>(plan, src, style),
    }
}

// ============================================================================
// Monomorphized 3-Band Satellite RGB Fused Kernels
// ============================================================================

fn render_rgb_specialized<const N: usize, const BILINEAR: bool, const CLEAN: bool>(
    plan: &CoordPlan<N>,
    srcs: &[BandSource<'_>; 3],
    style: &RgbStyle,
) -> Bytes {
    let inv = [
        255.0 / (style.rescale_max[0] - style.rescale_min[0]).max(1e-12),
        255.0 / (style.rescale_max[1] - style.rescale_min[1]).max(1e-12),
        255.0 / (style.rescale_max[2] - style.rescale_min[2]).max(1e-12),
    ];
    let mins = style.rescale_min;

    match style.format {
        OutputFormat::RawF32 => {
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
                for py in 0..N {
                    let row_rgb = &mut sc.row_buf[..N * 3];
                    for_each_pixel_in_row(plan, py, |px, u, v| {
                        let r = sample_val::<BILINEAR, true>(&srcs[0], u, v);
                        let g = sample_val::<BILINEAR, true>(&srcs[1], u, v);
                        let b = sample_val::<BILINEAR, true>(&srcs[2], u, v);
                        let off = px * 3;
                        row_rgb[off] = rescale_u8(r, inv[0], mins[0]);
                        row_rgb[off + 1] = rescale_u8(g, inv[1], mins[1]);
                        row_rgb[off + 2] = rescale_u8(b, inv[2], mins[2]);
                    });
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
        OutputFormat::Webp { quality } => SCRATCH.with(|cell| {
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
                        let r = sample_val::<BILINEAR, true>(&srcs[0], u, v);
                        let g = sample_val::<BILINEAR, true>(&srcs[1], u, v);
                        let b = sample_val::<BILINEAR, true>(&srcs[2], u, v);
                        let off = row_off + px * 3;
                        buf[off] = rescale_u8(r, inv[0], mins[0]);
                        buf[off + 1] = rescale_u8(g, inv[1], mins[1]);
                        buf[off + 2] = rescale_u8(b, inv[2], mins[2]);
                    });
                }
                encode_webp_rgb(N as u32, N as u32, buf, quality)
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
                encode_webp_rgba(N as u32, N as u32, buf, quality)
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
    let clean = !matches!(plan, CoordPlan::Exact2D { .. })
        && srcs[0].is_clean_inbounds(c0, c1, r0, r1)
        && srcs[1].is_clean_inbounds(c0, c1, r0, r1)
        && srcs[2].is_clean_inbounds(c0, c1, r0, r1);

    match (style.resample, clean) {
        (ResampleMode::Bilinear, true) => render_rgb_specialized::<N, true, true>(plan, srcs, style),
        (ResampleMode::Bilinear, false) => render_rgb_specialized::<N, true, false>(plan, srcs, style),
        (ResampleMode::Nearest, true) => render_rgb_specialized::<N, false, true>(plan, srcs, style),
        (ResampleMode::Nearest, false) => render_rgb_specialized::<N, false, false>(plan, srcs, style),
    }
}

// ============================================================================
// Fused Satellite Spectral Band Math Kernels (NDVI, NDWI, NBR, SAVI, EVI)
// Zero intermediate float tile allocation — computed directly in scanline loop!
// ============================================================================

fn render_eval_fn<const N: usize, F: Fn(f32, f32) -> f32>(
    plan: &CoordPlan<N>,
    style: &SingleBandStyle,
    eval_uv: F,
) -> Bytes {
    let range = (style.rescale_max - style.rescale_min).max(1e-12);
    let inv_range = 255.0 / range;
    let min_val = style.rescale_min;

    match style.format {
        OutputFormat::RawF32 => {
            let mut out = BytesMut::with_capacity(N * N * 4);
            for py in 0..N {
                for_each_pixel_in_row(plan, py, |_px, u, v| {
                    out.put_slice(&eval_uv(u, v).to_le_bytes());
                });
            }
            out.freeze()
        }
        OutputFormat::Png => {
            let lut: &ColormapLut = style.colormap.lut();
            SCRATCH.with(|cell| {
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
                    let row_u32 = unsafe {
                        std::slice::from_raw_parts_mut(row_rgba.as_mut_ptr() as *mut u32, N)
                    };
                    for_each_pixel_in_row(plan, py, |px, u, v| {
                        let val = eval_uv(u, v);
                        row_u32[px] = if val.is_nan() {
                            0
                        } else {
                            let idx = rescale_u8(val, inv_range, min_val) as usize;
                            unsafe { *lut.rgba_u32.get_unchecked(idx) }
                        };
                    });
                    filter_sub_rgba_row(row_rgba, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                }
                encode_filtered_png(N as u32, N as u32, 6, &sc.img_buf[..total])
            })
        }
        OutputFormat::Webp { quality } => {
            let lut: &ColormapLut = style.colormap.lut();
            SCRATCH.with(|cell| {
                let mut guard = cell.borrow_mut();
                let sc = &mut *guard;
                let total = N * N * 4;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                let buf = &mut sc.img_buf[..total];
                let buf_u32 = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u32, N * N)
                };
                for py in 0..N {
                    let row_off = py * N;
                    for_each_pixel_in_row(plan, py, |px, u, v| {
                        let val = eval_uv(u, v);
                        buf_u32[row_off + px] = if val.is_nan() {
                            0
                        } else {
                            let idx = rescale_u8(val, inv_range, min_val) as usize;
                            unsafe { *lut.rgba_u32.get_unchecked(idx) }
                        };
                    });
                }
                encode_webp_rgba(N as u32, N as u32, buf, quality)
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
    if src.all_nodata {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = !matches!(plan, CoordPlan::Exact2D { .. }) && src.is_clean_inbounds(c0, c1, r0, r1);

    match (style.resample, clean) {
        (ResampleMode::Bilinear, true) => {
            render_eval_fn(plan, style, |u, v| expr.eval1(sample_bilinear::<true>(src, u, v)))
        }
        (ResampleMode::Bilinear, false) => {
            render_eval_fn(plan, style, |u, v| expr.eval1(sample_bilinear::<false>(src, u, v)))
        }
        (ResampleMode::Nearest, true) => {
            render_eval_fn(plan, style, |u, v| expr.eval1(sample_nearest::<true>(src, u, v)))
        }
        (ResampleMode::Nearest, false) => {
            render_eval_fn(plan, style, |u, v| expr.eval1(sample_nearest::<false>(src, u, v)))
        }
    }
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
    let clean = !matches!(plan, CoordPlan::Exact2D { .. })
        && srcs[0].is_clean_inbounds(c0, c1, r0, r1)
        && srcs[1].is_clean_inbounds(c0, c1, r0, r1);

    let inv = [
        255.0 / (style.rescale_max[0] - style.rescale_min[0]).max(1e-12),
        255.0 / (style.rescale_max[1] - style.rescale_min[1]).max(1e-12),
        255.0 / (style.rescale_max[2] - style.rescale_min[2]).max(1e-12),
    ];
    let mins = style.rescale_min;
    let bilinear = style.resample == ResampleMode::Bilinear;

    let sample_pair = |u: f32, v: f32| -> (f32, f32, f32) {
        let (vv, vh) = match (bilinear, clean) {
            (true, true) => (
                sample_bilinear::<true>(&srcs[0], u, v),
                sample_bilinear::<true>(&srcs[1], u, v),
            ),
            (true, false) => (
                sample_bilinear::<false>(&srcs[0], u, v),
                sample_bilinear::<false>(&srcs[1], u, v),
            ),
            (false, true) => (
                sample_nearest::<true>(&srcs[0], u, v),
                sample_nearest::<true>(&srcs[1], u, v),
            ),
            (false, false) => (
                sample_nearest::<false>(&srcs[0], u, v),
                sample_nearest::<false>(&srcs[1], u, v),
            ),
        };
        let vv_db = fast_db_from_power(vv);
        let vh_db = fast_db_from_power(vh);
        (vv_db, vh_db, vv_db - vh_db)
    };

    match style.format {
        OutputFormat::RawF32 => {
            let mut out = BytesMut::with_capacity(N * N * 3 * 4);
            let mut g_plane = Vec::with_capacity(N * N * 4);
            let mut b_plane = Vec::with_capacity(N * N * 4);
            for py in 0..N {
                for_each_pixel_in_row(plan, py, |_px, u, v| {
                    let (r, g, b) = sample_pair(u, v);
                    out.put_slice(&r.to_le_bytes());
                    g_plane.extend_from_slice(&g.to_le_bytes());
                    b_plane.extend_from_slice(&b.to_le_bytes());
                });
            }
            out.put_slice(&g_plane);
            out.put_slice(&b_plane);
            out.freeze()
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
        OutputFormat::Webp { quality } => SCRATCH.with(|cell| {
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
            encode_webp_rgba(N as u32, N as u32, buf, quality)
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
    if srcs[0].all_nodata || srcs[1].all_nodata {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = srcs[0].is_clean_inbounds(c0, c1, r0, r1) && srcs[1].is_clean_inbounds(c0, c1, r0, r1);

    match (style.resample, clean) {
        (ResampleMode::Bilinear, true) => render_eval_fn(plan, style, |u, v| {
            let a = sample_bilinear::<true>(&srcs[0], u, v);
            let b = sample_bilinear::<true>(&srcs[1], u, v);
            expr.eval2(a, b)
        }),
        (ResampleMode::Bilinear, false) => render_eval_fn(plan, style, |u, v| {
            let a = sample_bilinear::<false>(&srcs[0], u, v);
            let b = sample_bilinear::<false>(&srcs[1], u, v);
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                expr.eval2(a, b)
            }
        }),
        (ResampleMode::Nearest, true) => render_eval_fn(plan, style, |u, v| {
            let a = sample_nearest::<true>(&srcs[0], u, v);
            let b = sample_nearest::<true>(&srcs[1], u, v);
            expr.eval2(a, b)
        }),
        (ResampleMode::Nearest, false) => render_eval_fn(plan, style, |u, v| {
            let a = sample_nearest::<false>(&srcs[0], u, v);
            let b = sample_nearest::<false>(&srcs[1], u, v);
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                expr.eval2(a, b)
            }
        }),
    }
}

/// Fused 3-band spectral index kernel (EVI).
pub fn render_bandmath3<const N: usize>(
    plan: &CoordPlan<N>,
    srcs: &[BandSource<'_>; 3],
    expr: SpectralExpr,
    style: &SingleBandStyle,
) -> Bytes {
    if srcs[0].all_nodata || srcs[1].all_nodata || srcs[2].all_nodata {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = !matches!(plan, CoordPlan::Exact2D { .. })
        && srcs[0].is_clean_inbounds(c0, c1, r0, r1)
        && srcs[1].is_clean_inbounds(c0, c1, r0, r1)
        && srcs[2].is_clean_inbounds(c0, c1, r0, r1);

    match (style.resample, clean) {
        (ResampleMode::Bilinear, true) => render_eval_fn(plan, style, |u, v| {
            let a = sample_bilinear::<true>(&srcs[0], u, v);
            let b = sample_bilinear::<true>(&srcs[1], u, v);
            let c = sample_bilinear::<true>(&srcs[2], u, v);
            expr.eval3(a, b, c)
        }),
        (ResampleMode::Bilinear, false) => render_eval_fn(plan, style, |u, v| {
            let a = sample_bilinear::<false>(&srcs[0], u, v);
            let b = sample_bilinear::<false>(&srcs[1], u, v);
            let c = sample_bilinear::<false>(&srcs[2], u, v);
            if a.is_nan() || b.is_nan() || c.is_nan() {
                f32::NAN
            } else {
                expr.eval3(a, b, c)
            }
        }),
        (ResampleMode::Nearest, true) => render_eval_fn(plan, style, |u, v| {
            let a = sample_nearest::<true>(&srcs[0], u, v);
            let b = sample_nearest::<true>(&srcs[1], u, v);
            let c = sample_nearest::<true>(&srcs[2], u, v);
            expr.eval3(a, b, c)
        }),
        (ResampleMode::Nearest, false) => render_eval_fn(plan, style, |u, v| {
            let a = sample_nearest::<false>(&srcs[0], u, v);
            let b = sample_nearest::<false>(&srcs[1], u, v);
            let c = sample_nearest::<false>(&srcs[2], u, v);
            if a.is_nan() || b.is_nan() || c.is_nan() {
                f32::NAN
            } else {
                expr.eval3(a, b, c)
            }
        }),
    }
}
