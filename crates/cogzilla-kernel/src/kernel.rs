use std::cell::RefCell;

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_proj::{CoordPlan, GRID_DIM};
use cogzilla_tiff::SampleType;

use crate::bandmath::{SpectralExpr, fast_db_from_power};
use crate::colormap::{ColormapLut, ColormapName};
use crate::expr_vm::ExprProgram;
use crate::jpeg_fast::{empty_black_jpeg, encode_jpeg_rgb, encode_jpeg_rgba};
use crate::png_fast::{
    empty_transparent_png, encode_filtered_png, encode_indexed_png, filter_sub_ga_row,
    filter_sub_rgb_row, filter_sub_rgba_row, filter_sub_u8_row,
};

#[inline]
fn build_plte_opaque(lut: &ColormapLut) -> [u8; 768] {
    let mut plte = [0u8; 768];
    for i in 0..256 {
        plte[i * 3] = lut.rgb[i][0];
        plte[i * 3 + 1] = lut.rgb[i][1];
        plte[i * 3 + 2] = lut.rgb[i][2];
    }
    plte
}

#[inline]
fn build_plte_nodata(lut: &ColormapLut) -> [u8; 768] {
    let mut plte = [0u8; 768];
    // Index 0 = transparent nodata (paired with tRNS = [0]); indices 1..=255 span lut[0..=255].
    for i in 1..256 {
        let src_idx = ((i - 1) * 255 + 127) / 254;
        plte[i * 3] = lut.rgb[src_idx][0];
        plte[i * 3 + 1] = lut.rgb[src_idx][1];
        plte[i * 3 + 2] = lut.rgb[src_idx][2];
    }
    plte
}
use crate::source::BandSource;
use crate::webp_fast::{empty_transparent_webp, encode_webp_rgb, encode_webp_rgba};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMode {
    Nearest,
    Bilinear,
    Cubic,
    CubicSpline,
    Lanczos,
    Average,
    Mode,
    Gauss,
    Rms,
    Min,
    Max,
    Med,
    Q1,
    Q3,
    Sum,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Png,
    /// `quality: None` = lossless WebP, `Some(q)` = lossy quality (e.g. `82.0`).
    Webp { quality: Option<f32> },
    /// JPEG (lossy quality `1..=100`, default `85`).
    Jpeg { quality: u8 },
    RawF32,
    /// NumPy v1.0 `.npy` float32 array (`shape = (C, H, W)`, little-endian `<f4`).
    Npy,
    /// Uncompressed RGBA8 (`N * N * 4` bytes), used internally for multi-scene mosaic compositing.
    RawRgba,
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
    dem_elev: Vec<f32>,
}

impl KernelScratch {
    fn new() -> Self {
        Self {
            img_buf: Vec::with_capacity(512 * (1 + 512 * 4)),
            row_buf: vec![0u8; 512 * 4],
            f32_row: vec![0.0f32; 512],
            dem_elev: Vec::new(),
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
    max_x: i32,
    fy: f32,
    out_row: &mut [f32],
) {
    use std::arch::aarch64::*;
    let v_fy = unsafe { vdupq_n_f32(fy) };
    let v_base = unsafe { vdupq_n_f32(base_col as f32) };
    let v_zero = unsafe { vdupq_n_s32(0) };
    let v_max_x = unsafe { vdupq_n_s32(max_x) };
    let mut px = 0usize;

    while px + 4 <= N {
        unsafe {
            let u = vsubq_f32(vld1q_f32(u_coords.as_ptr().add(px)), v_base);
            let x0_f = vrndmq_f32(u);
            let fx = vsubq_f32(u, x0_f);
            let x0_i = vminq_s32(vmaxq_s32(vcvtq_s32_f32(x0_f), v_zero), v_max_x);

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
    let bc_f = base_col as f32;
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc_f);
        let c = x0_raw.clamp(0, max_x) as usize;
        unsafe {
            let v00 = *row0_ptr.add(c);
            let v10 = *row0_ptr.add(c + 1);
            let v01 = *row1_ptr.add(c);
            let v11 = *row1_ptr.add(c + 1);
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            *out_row.get_unchecked_mut(px) = top + (bot - top) * fy;
        }
        px += 1;
    }
}

/// x86_64 AVX2+FMA 8-wide bilinear row sampler for flat single-block `Separable1D + CleanFastPath`.
/// Uses 64-bit adjacent-pair loads + shuffle instead of microcoded `vgatherdps` (much faster on
/// AMD Zen 4 / EPYC as well as Intel).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilinear_row_flat_avx2<const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0_ptr: *const f32,
    row1_ptr: *const f32,
    max_x: i32,
    fy: f32,
    out_row: &mut [f32],
) {
    use std::arch::x86_64::*;
    let v_fy = _mm256_set1_ps(fy);
    let v_base = _mm256_set1_ps(base_col as f32);
    let v_zero = _mm256_setzero_si256();
    let v_max_x = _mm256_set1_epi32(max_x);
    let mut idxs = [0i32; 8];
    let mut px = 0usize;

    macro_rules! load_8_pairs {
        ($ptr:expr) => {{
            let c0 = idxs[0] as usize;
            let c1 = idxs[1] as usize;
            let c2 = idxs[2] as usize;
            let c3 = idxs[3] as usize;
            let c4 = idxs[4] as usize;
            let c5 = idxs[5] as usize;
            let c6 = idxs[6] as usize;
            let c7 = idxs[7] as usize;

            let p01 = _mm_castpd_ps(_mm_set_pd(
                ($ptr.add(c1) as *const f64).read_unaligned(),
                ($ptr.add(c0) as *const f64).read_unaligned(),
            ));
            let p23 = _mm_castpd_ps(_mm_set_pd(
                ($ptr.add(c3) as *const f64).read_unaligned(),
                ($ptr.add(c2) as *const f64).read_unaligned(),
            ));
            let p45 = _mm_castpd_ps(_mm_set_pd(
                ($ptr.add(c5) as *const f64).read_unaligned(),
                ($ptr.add(c4) as *const f64).read_unaligned(),
            ));
            let p67 = _mm_castpd_ps(_mm_set_pd(
                ($ptr.add(c7) as *const f64).read_unaligned(),
                ($ptr.add(c6) as *const f64).read_unaligned(),
            ));

            let lo0 = _mm_shuffle_ps::<0x88>(p01, p23);
            let lo1 = _mm_shuffle_ps::<0xDD>(p01, p23);
            let hi0 = _mm_shuffle_ps::<0x88>(p45, p67);
            let hi1 = _mm_shuffle_ps::<0xDD>(p45, p67);
            (
                _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo0), hi0),
                _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo1), hi1),
            )
        }};
    }

    while px + 8 <= N {
        let u = _mm256_sub_ps(_mm256_loadu_ps(u_coords.as_ptr().add(px)), v_base);
        let x0_f = _mm256_floor_ps(u);
        let fx = _mm256_sub_ps(u, x0_f);
        let idx = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(x0_f), v_zero), v_max_x);
        _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, idx);

        let (v00, v10) = load_8_pairs!(row0_ptr);
        let (v01, v11) = load_8_pairs!(row1_ptr);

        let top = _mm256_fmadd_ps(_mm256_sub_ps(v10, v00), fx, v00);
        let bot = _mm256_fmadd_ps(_mm256_sub_ps(v11, v01), fx, v01);
        let res = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), v_fy, top);

        _mm256_storeu_ps(out_row.as_mut_ptr().add(px), res);
        px += 8;
    }
    let bc_f = base_col as f32;
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc_f);
        let c = x0_raw.clamp(0, max_x) as usize;
        let v00 = *row0_ptr.add(c);
        let v10 = *row0_ptr.add(c + 1);
        let v01 = *row1_ptr.add(c);
        let v11 = *row1_ptr.add(c + 1);
        let top = v00 + (v10 - v00) * fx;
        let bot = v01 + (v11 - v01) * fx;
        *out_row.get_unchecked_mut(px) = top + (bot - top) * fy;
        px += 1;
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
    let row_len = row0.len().min(row1.len());
    if row_len < 2 || out_row.len() < N {
        return;
    }
    let max_x = (row_len as i32) - 2;
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            bilinear_row_flat_neon::<N>(
                u_coords,
                base_col,
                row0.as_ptr(),
                row1.as_ptr(),
                max_x,
                fy,
                out_row,
            );
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                bilinear_row_flat_avx2::<N>(
                    u_coords,
                    base_col,
                    row0.as_ptr(),
                    row1.as_ptr(),
                    max_x,
                    fy,
                    out_row,
                );
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    {
        let bc = base_col as f32;
        let max_x = row0.len().min(row1.len()).saturating_sub(2);
        for px in 0..N {
            let u = u_coords[px] - bc;
            let x0_f = u.floor();
            let fx = u - x0_f;
            let x0 = (x0_f.max(0.0) as usize).min(max_x);
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

/// ARM NEON 4-wide bilinear row sampler for single-block `U16`/`U8` (`Separable1D + CleanFastPath`).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn bilinear_row_typed_neon<T: Copy, F: Fn(T) -> f32, const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0_ptr: *const T,
    row1_ptr: *const T,
    max_x: i32,
    fy: f32,
    out_row: &mut [f32],
    to_f32: &F,
) {
    use std::arch::aarch64::*;
    let (v_fy, v_base, v_zero, v_max_x) = unsafe {
        (
            vdupq_n_f32(fy),
            vdupq_n_f32(base_col as f32),
            vdupq_n_s32(0),
            vdupq_n_s32(max_x),
        )
    };
    let mut idxs = [0i32; 4];
    let mut px = 0usize;
    while px + 4 <= N {
        unsafe {
            let u = vsubq_f32(vld1q_f32(u_coords.as_ptr().add(px)), v_base);
            let x0_f = vrndmq_f32(u);
            let fx = vsubq_f32(u, x0_f);
            let idx = vminq_s32(vmaxq_s32(vcvtq_s32_f32(x0_f), v_zero), v_max_x);
            vst1q_s32(idxs.as_mut_ptr(), idx);

            let c0 = idxs[0] as usize;
            let c1 = idxs[1] as usize;
            let c2 = idxs[2] as usize;
            let c3 = idxs[3] as usize;

            let a00 = [
                to_f32(*row0_ptr.add(c0)),
                to_f32(*row0_ptr.add(c1)),
                to_f32(*row0_ptr.add(c2)),
                to_f32(*row0_ptr.add(c3)),
            ];
            let a10 = [
                to_f32(*row0_ptr.add(c0 + 1)),
                to_f32(*row0_ptr.add(c1 + 1)),
                to_f32(*row0_ptr.add(c2 + 1)),
                to_f32(*row0_ptr.add(c3 + 1)),
            ];
            let a01 = [
                to_f32(*row1_ptr.add(c0)),
                to_f32(*row1_ptr.add(c1)),
                to_f32(*row1_ptr.add(c2)),
                to_f32(*row1_ptr.add(c3)),
            ];
            let a11 = [
                to_f32(*row1_ptr.add(c0 + 1)),
                to_f32(*row1_ptr.add(c1 + 1)),
                to_f32(*row1_ptr.add(c2 + 1)),
                to_f32(*row1_ptr.add(c3 + 1)),
            ];

            let v00 = vld1q_f32(a00.as_ptr());
            let v10 = vld1q_f32(a10.as_ptr());
            let v01 = vld1q_f32(a01.as_ptr());
            let v11 = vld1q_f32(a11.as_ptr());

            let top = vfmaq_f32(v00, vsubq_f32(v10, v00), fx);
            let bot = vfmaq_f32(v01, vsubq_f32(v11, v01), fx);
            let res = vfmaq_f32(top, vsubq_f32(bot, top), v_fy);
            vst1q_f32(out_row.as_mut_ptr().add(px), res);
        }
        px += 4;
    }
    let bc_f = base_col as f32;
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc_f);
        let c = x0_raw.clamp(0, max_x) as usize;
        unsafe {
            let v00 = to_f32(*row0_ptr.add(c));
            let v10 = to_f32(*row0_ptr.add(c + 1));
            let v01 = to_f32(*row1_ptr.add(c));
            let v11 = to_f32(*row1_ptr.add(c + 1));
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            *out_row.get_unchecked_mut(px) = top + (bot - top) * fy;
        }
        px += 1;
    }
}

/// x86_64 AVX2+FMA 8-wide bilinear row sampler for single-block `U16`/`U8` (`Separable1D + CleanFastPath`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilinear_row_typed_avx2<T: Copy, F: Fn(T) -> f32, const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0_ptr: *const T,
    row1_ptr: *const T,
    max_x: i32,
    fy: f32,
    out_row: &mut [f32],
    to_f32: &F,
) {
    use std::arch::x86_64::*;
    let v_fy = _mm256_set1_ps(fy);
    let v_base = _mm256_set1_ps(base_col as f32);
    let v_zero = _mm256_setzero_si256();
    let v_max_x = _mm256_set1_epi32(max_x);
    let mut idxs = [0i32; 8];
    let mut px = 0usize;
    while px + 8 <= N {
        let u = _mm256_sub_ps(_mm256_loadu_ps(u_coords.as_ptr().add(px)), v_base);
        let x0_f = _mm256_floor_ps(u);
        let fx = _mm256_sub_ps(u, x0_f);
        let idx = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(x0_f), v_zero), v_max_x);
        _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, idx);

        let mut a00 = [0.0f32; 8];
        let mut a10 = [0.0f32; 8];
        let mut a01 = [0.0f32; 8];
        let mut a11 = [0.0f32; 8];
        for k in 0..8 {
            let c = idxs[k] as usize;
            a00[k] = to_f32(*row0_ptr.add(c));
            a10[k] = to_f32(*row0_ptr.add(c + 1));
            a01[k] = to_f32(*row1_ptr.add(c));
            a11[k] = to_f32(*row1_ptr.add(c + 1));
        }

        let v00 = _mm256_loadu_ps(a00.as_ptr());
        let v10 = _mm256_loadu_ps(a10.as_ptr());
        let v01 = _mm256_loadu_ps(a01.as_ptr());
        let v11 = _mm256_loadu_ps(a11.as_ptr());

        let top = _mm256_fmadd_ps(_mm256_sub_ps(v10, v00), fx, v00);
        let bot = _mm256_fmadd_ps(_mm256_sub_ps(v11, v01), fx, v01);
        let res = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), v_fy, top);
        _mm256_storeu_ps(out_row.as_mut_ptr().add(px), res);
        px += 8;
    }
    let bc_f = base_col as f32;
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc_f);
        let c = x0_raw.clamp(0, max_x) as usize;
        let v00 = to_f32(*row0_ptr.add(c));
        let v10 = to_f32(*row0_ptr.add(c + 1));
        let v01 = to_f32(*row1_ptr.add(c));
        let v11 = to_f32(*row1_ptr.add(c + 1));
        let top = v00 + (v10 - v00) * fx;
        let bot = v01 + (v11 - v01) * fx;
        *out_row.get_unchecked_mut(px) = top + (bot - top) * fy;
        px += 1;
    }
}

#[inline(always)]
fn bilinear_row_typed_dispatch<T: Copy, F: Fn(T) -> f32, const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0: &[T],
    row1: &[T],
    fy: f32,
    out_row: &mut [f32],
    to_f32: F,
) {
    let row_len = row0.len().min(row1.len());
    if row_len < 2 || out_row.len() < N {
        return;
    }
    let max_x = (row_len as i32) - 2;
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            bilinear_row_typed_neon::<T, F, N>(
                u_coords,
                base_col,
                row0.as_ptr(),
                row1.as_ptr(),
                max_x,
                fy,
                out_row,
                &to_f32,
            );
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                bilinear_row_typed_avx2::<T, F, N>(
                    u_coords,
                    base_col,
                    row0.as_ptr(),
                    row1.as_ptr(),
                    max_x,
                    fy,
                    out_row,
                    &to_f32,
                );
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    {
        let bc = base_col as f32;
        let max_x_u = row_len.saturating_sub(2);
        for px in 0..N {
            let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc);
            let x0 = (x0_raw.max(0) as usize).min(max_x_u);
            let v00 = to_f32(unsafe { *row0.get_unchecked(x0) });
            let v10 = to_f32(unsafe { *row0.get_unchecked(x0 + 1) });
            let v01 = to_f32(unsafe { *row1.get_unchecked(x0) });
            let v11 = to_f32(unsafe { *row1.get_unchecked(x0 + 1) });
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            out_row[px] = top + (bot - top) * fy;
        }
    }
}

/// Unified single-block typed view for clean (`no_nodata`, exact `f32` base, `w/h >= 2`) bands.
#[derive(Clone, Copy)]
enum SingleBlockRef<'a> {
    F32(&'a [f32], u32, u32, usize),
    U16(&'a [u16], u32, u32, usize),
    U8(&'a [u8], u32, u32, usize),
}

impl<'a> SingleBlockRef<'a> {
    #[inline(always)]
    fn from_source(src: &BandSource<'a>) -> Option<Self> {
        if let Some((s, bc, br, st)) = src.single_block_flat() {
            Some(Self::F32(s, bc, br, st))
        } else if let Some((s, bc, br, st)) = src.single_block_u16() {
            Some(Self::U16(s, bc, br, st))
        } else if let Some((s, bc, br, st)) = src.single_block_u8() {
            Some(Self::U8(s, bc, br, st))
        } else {
            None
        }
    }

    #[inline(always)]
    fn base_col(&self) -> u32 {
        match *self {
            Self::F32(_, bc, _, _) | Self::U16(_, bc, _, _) | Self::U8(_, bc, _, _) => bc,
        }
    }

    #[inline(always)]
    fn base_row(&self) -> u32 {
        match *self {
            Self::F32(_, _, br, _) | Self::U16(_, _, br, _) | Self::U8(_, _, br, _) => br,
        }
    }

    #[inline(always)]
    fn row_stride(&self) -> usize {
        match *self {
            Self::F32(_, _, _, st) | Self::U16(_, _, _, st) | Self::U8(_, _, _, st) => st,
        }
    }

    #[inline(always)]
    fn height(&self) -> usize {
        match *self {
            Self::F32(s, _, _, st) => s.len() / st,
            Self::U16(s, _, _, st) => s.len() / st,
            Self::U8(s, _, _, st) => s.len() / st,
        }
    }

    #[inline(always)]
    fn precompute_nearest_x<const N: usize>(&self, u: &[f32; N]) -> [usize; N] {
        let max_x = (self.row_stride() as i32 - 1).max(0);
        let bc_f = self.base_col() as f32;
        let mut x_idx = [0usize; N];
        for px in 0..N {
            x_idx[px] = fast_floor_i32(u[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
        }
        x_idx
    }

    #[inline(always)]
    fn sample_nearest_row_f32<const N: usize>(
        &self,
        x_idx: &[usize; N],
        vy: f32,
        out: &mut [f32],
    ) {
        let max_y = (self.height() as i32 - 1).max(0);
        let y0 = fast_floor_i32(vy + 0.5 - self.base_row() as f32)
            .0
            .clamp(0, max_y) as usize;
        match *self {
            Self::F32(s, _, _, st) => {
                let row = &s[y0 * st..];
                for px in 0..N {
                    out[px] = row[x_idx[px]];
                }
            }
            Self::U16(s, _, _, st) => {
                let row = &s[y0 * st..];
                for px in 0..N {
                    out[px] = row[x_idx[px]] as f32;
                }
            }
            Self::U8(s, _, _, st) => {
                let row = &s[y0 * st..];
                for px in 0..N {
                    out[px] = row[x_idx[px]] as f32;
                }
            }
        }
    }

    #[inline(always)]
    fn sample_bilinear_row_f32<const N: usize>(&self, u: &[f32; N], vy: f32, out: &mut [f32]) {
        let max_y = (self.height() as i32 - 2).max(0);
        let (y0_raw, fy) = fast_floor_i32(vy - self.base_row() as f32);
        let y0 = y0_raw.clamp(0, max_y) as usize;
        match *self {
            Self::F32(s, bc, _, st) => {
                let r0 = &s[y0 * st..(y0 + 1) * st];
                let r1 = &s[(y0 + 1) * st..(y0 + 2) * st];
                bilinear_row_flat_dispatch::<N>(u, bc, r0, r1, fy, out);
            }
            Self::U16(s, bc, _, st) => {
                let r0 = &s[y0 * st..(y0 + 1) * st];
                let r1 = &s[(y0 + 1) * st..(y0 + 2) * st];
                bilinear_row_typed_dispatch::<u16, _, N>(u, bc, r0, r1, fy, out, |v| v as f32);
            }
            Self::U8(s, bc, _, st) => {
                let r0 = &s[y0 * st..(y0 + 1) * st];
                let r1 = &s[(y0 + 1) * st..(y0 + 2) * st];
                bilinear_row_typed_dispatch::<u8, _, N>(u, bc, r0, r1, fy, out, |v| v as f32);
            }
        }
    }
}

// ============================================================================
// General Per-Pixel Samplers
// ============================================================================

/// Branchless inline floor returning `(floor_i32, frac)` without calling libm `floorf` on
/// baseline x86-64 SSE2 builds (where `f32::floor` emits a `floorf@PLT` call per pixel).
#[inline(always)]
fn fast_floor_i32(x: f32) -> (i32, f32) {
    #[cfg(any(target_arch = "aarch64", target_feature = "sse4.1"))]
    {
        let f = x.floor();
        (f as i32, x - f)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "sse4.1")))]
    {
        let i = x as i32;
        let adj = i.saturating_sub((i as f32 > x) as i32);
        (adj, x - (adj as f32))
    }
}

#[inline(always)]
fn sample_bilinear<const CLEAN: bool>(src: &BandSource<'_>, u: f32, v: f32) -> f32 {
    if !CLEAN
        && (!u.is_finite()
            || !v.is_finite()
            || u < -1.0
            || v < -1.0
            || u > src.ifd_w as f32
            || v > src.ifd_h as f32)
    {
        return f32::NAN;
    }
    let (x0, fx) = fast_floor_i32(u);
    let (y0, fy) = fast_floor_i32(v);

    if CLEAN {
        let (v00, v10, v01, v11) = src.sample_quad_inbounds(x0 as u32, y0 as u32);
        let top = v00 + (v10 - v00) * fx;
        let bot = v01 + (v11 - v01) * fx;
        top + (bot - top) * fy
    } else {
        let (v00, v10, v01, v11) = if x0 >= 0
            && y0 >= 0
            && x0 + 1 < src.ifd_w
            && y0 + 1 < src.ifd_h
            && src.nodata_override.is_none()
            && src.scale_offset.is_none()
        {
            src.sample_quad_inbounds(x0 as u32, y0 as u32)
        } else {
            (
                src.sample_checked(x0, y0),
                src.sample_checked(x0 + 1, y0),
                src.sample_checked(x0, y0 + 1),
                src.sample_checked(x0 + 1, y0 + 1),
            )
        };

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
    let (x, _) = fast_floor_i32(u + 0.5);
    let (y, _) = fast_floor_i32(v + 0.5);
    if CLEAN {
        src.sample_inbounds(x as u32, y as u32)
    } else {
        src.sample_checked(x, y)
    }
}

/// Branchless Keys/Catmull-Rom bicubic weights (`a = -0.5`, matching GDAL `GRA_Cubic`)
/// for fractional offset `t in [0.0, 1.0]`.
#[inline(always)]
fn cubic_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        (-0.5 * t3) + t2 - (0.5 * t),
        (1.5 * t3) - (2.5 * t2) + 1.0,
        (-1.5 * t3) + (2.0 * t2) + (0.5 * t),
        (0.5 * t3) - (0.5 * t2),
    ]
}

#[inline(always)]
pub fn sample_cubic(src: &BandSource<'_>, u: f32, v: f32) -> f32 {
    if !u.is_finite()
        || !v.is_finite()
        || u < -2.0
        || v < -2.0
        || u > src.ifd_w as f32 + 1.0
        || v > src.ifd_h as f32 + 1.0
    {
        return f32::NAN;
    }
    let (x0, fx) = fast_floor_i32(u);
    let (y0, fy) = fast_floor_i32(v);
    let wx = cubic_weights(fx);
    let wy = cubic_weights(fy);

    // Fast SIMD path: all 16 samples lie strictly inside one clean f32 block -> 4 contiguous 16B loads!
    if let Some((base_ptr, stride)) = src.interior_4x4_f32(x0, y0) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let vwx = vld1q_f32(wx.as_ptr());
            let r0 = vld1q_f32(base_ptr);
            let r1 = vld1q_f32(base_ptr.add(stride));
            let r2 = vld1q_f32(base_ptr.add(stride * 2));
            let r3 = vld1q_f32(base_ptr.add(stride * 3));
            // Vertical blend of the 4 rows first: acc = r0*wy[0] + r1*wy[1] + r2*wy[2] + r3*wy[3]
            let mut col = vmulq_n_f32(r0, wy[0]);
            col = vfmaq_n_f32(col, r1, wy[1]);
            col = vfmaq_n_f32(col, r2, wy[2]);
            col = vfmaq_n_f32(col, r3, wy[3]);
            return vaddvq_f32(vmulq_f32(col, vwx));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let vwx = _mm_loadu_ps(wx.as_ptr());
            let r0 = _mm_loadu_ps(base_ptr);
            let r1 = _mm_loadu_ps(base_ptr.add(stride));
            let r2 = _mm_loadu_ps(base_ptr.add(stride * 2));
            let r3 = _mm_loadu_ps(base_ptr.add(stride * 3));
            let col = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(r0, _mm_set1_ps(wy[0])), _mm_mul_ps(r1, _mm_set1_ps(wy[1]))),
                _mm_add_ps(_mm_mul_ps(r2, _mm_set1_ps(wy[2])), _mm_mul_ps(r3, _mm_set1_ps(wy[3]))),
            );
            let prod = _mm_mul_ps(col, vwx);
            let shuf = _mm_movehl_ps(prod, prod);
            let sums = _mm_add_ps(prod, shuf);
            let shuf2 = _mm_shuffle_ps::<1>(sums, sums);
            return _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        }
        #[allow(unreachable_code)]
        unsafe {
            let mut acc = 0.0f32;
            for j in 0..4 {
                let rp = base_ptr.add(j * stride);
                let row = *rp * wx[0] + *rp.add(1) * wx[1] + *rp.add(2) * wx[2] + *rp.add(3) * wx[3];
                acc += row * wy[j];
            }
            return acc;
        }
    }

    let mut acc = 0.0f32;
    for j in 0..4 {
        let mut row_acc = 0.0f32;
        let ry = y0 + (j as i32) - 1;
        for i in 0..4 {
            let v_s = src.sample_checked(x0 + (i as i32) - 1, ry);
            if v_s.is_nan() {
                return sample_bilinear::<false>(src, u, v);
            }
            row_acc += v_s * wx[i];
        }
        acc += row_acc * wy[j];
    }
    acc
}

#[inline(always)]
fn cubic_spline_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    [
        (omt * omt * omt) * (1.0 / 6.0),
        (3.0 * t3 - 6.0 * t2 + 4.0) * (1.0 / 6.0),
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) * (1.0 / 6.0),
        t3 * (1.0 / 6.0),
    ]
}

#[inline(always)]
fn lanczos3_weight(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 1e-6 {
        return 1.0;
    }
    if ax >= 3.0 {
        return 0.0;
    }
    let px = std::f32::consts::PI * ax;
    let px3 = px * (1.0 / 3.0);
    (px.sin() / px) * (px3.sin() / px3)
}

/// General sampler supporting all 15 GDAL/TiTiler resampling methods at 1:1 source scale.
#[inline]
pub fn sample_general(src: &BandSource<'_>, u: f32, v: f32, mode: ResampleMode) -> f32 {
    sample_general_scaled(src, u, v, 1.0, 1.0, mode)
}

/// General sampler supporting all 15 GDAL/TiTiler resampling methods with an explicit
/// destination-to-source pixel footprint `(scale_x, scale_y)` in source pixel units.
///
/// For footprint-based modes (`Average`, `Mode`, `Rms`, `Min`, `Max`, `Med`, `Q1`, `Q3`, `Sum`),
/// integrates over the exact source-pixel box `[u + 0.5 - 0.5*scale_x, u + 0.5 + 0.5*scale_x] x
/// [v + 0.5 - 0.5*scale_y, v + 0.5 + 0.5*scale_y]` weighted by exact overlap area:
/// - At 1:1 scale (`scale_x == 1, scale_y == 1`), reads the exact single source pixel (0 blur).
/// - At exact integer factors (2x, 4x, 8x, ...), computes the exact block mean / mode / stat.
/// - At non-integer factors, weights each overlapped source pixel by its exact overlap area.
#[inline]
pub fn sample_general_scaled(
    src: &BandSource<'_>,
    u: f32,
    v: f32,
    scale_x: f32,
    scale_y: f32,
    mode: ResampleMode,
) -> f32 {
    if !u.is_finite()
        || !v.is_finite()
        || u < -0.5
        || v < -0.5
        || u > src.ifd_w as f32 - 0.5
        || v > src.ifd_h as f32 - 0.5
    {
        return f32::NAN;
    }
    match mode {
        ResampleMode::Nearest => sample_nearest::<false>(src, u, v),
        ResampleMode::Bilinear => sample_bilinear::<false>(src, u, v),
        ResampleMode::Cubic => sample_cubic(src, u, v),
        ResampleMode::CubicSpline => {
            let (x0, fx) = fast_floor_i32(u);
            let (y0, fy) = fast_floor_i32(v);
            let wx = cubic_spline_weights(fx);
            let wy = cubic_spline_weights(fy);
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for j in 0..4 {
                let ry = y0 + (j as i32) - 1;
                let wyj = wy[j];
                for i in 0..4 {
                    let s = src.sample_checked(x0 + (i as i32) - 1, ry);
                    if !s.is_nan() {
                        let w = wx[i] * wyj;
                        sum += s * w;
                        wsum += w;
                    }
                }
            }
            if wsum > 1e-6 { sum / wsum } else { f32::NAN }
        }
        ResampleMode::Lanczos => {
            let (x0, fx) = fast_floor_i32(u);
            let (y0, fy) = fast_floor_i32(v);
            let mut wx = [0.0f32; 6];
            let mut wy = [0.0f32; 6];
            for k in 0..6 {
                let d = (k as i32 - 2) as f32;
                wx[k] = lanczos3_weight(d - fx);
                wy[k] = lanczos3_weight(d - fy);
            }
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for j in 0..6 {
                let ry = y0 + (j as i32) - 2;
                let wyj = wy[j];
                for i in 0..6 {
                    let s = src.sample_checked(x0 + (i as i32) - 2, ry);
                    if !s.is_nan() {
                        let w = wx[i] * wyj;
                        sum += s * w;
                        wsum += w;
                    }
                }
            }
            if wsum >= 0.5 {
                sum / wsum
            } else {
                sample_bilinear::<false>(src, u, v)
            }
        }
        ResampleMode::Gauss => {
            let cx = u.round() as i32;
            let cy = v.round() as i32;
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for dy in -1..=1 {
                let ry = cy + dy;
                let ydiff = ry as f32 - v;
                for dx in -1..=1 {
                    let rx = cx + dx;
                    let s = src.sample_checked(rx, ry);
                    if !s.is_nan() {
                        let xdiff = rx as f32 - u;
                        let w = (-0.888_888_9 * (xdiff * xdiff + ydiff * ydiff)).exp();
                        sum += s * w;
                        wsum += w;
                    }
                }
            }
            if wsum > 1e-6 { sum / wsum } else { f32::NAN }
        }
        ResampleMode::Average
        | ResampleMode::Mode
        | ResampleMode::Rms
        | ResampleMode::Min
        | ResampleMode::Max
        | ResampleMode::Med
        | ResampleMode::Q1
        | ResampleMode::Q3
        | ResampleMode::Sum => {
            sample_footprint_exact(src, u, v, scale_x, scale_y, mode)
        }
    }
}

/// Area-exact footprint sampler for `Average`, `Mode`, `Rms`, `Min`, `Max`, `Med`, `Q1`, `Q3`, `Sum`.
///
/// Maps destination pixel center `(u, v)` and footprint width/height `(scale_x, scale_y)` to
/// source corner box `[x0, x1] x [y0, y1]` (where source pixel `(ix, iy)` spans `[ix, ix+1] x
/// [iy, iy+1]`) and weights every overlapped valid source pixel by its exact overlap area.
#[inline]
fn sample_footprint_exact(
    src: &BandSource<'_>,
    u: f32,
    v: f32,
    scale_x: f32,
    scale_y: f32,
    mode: ResampleMode,
) -> f32 {
    let sx = (scale_x as f64).abs().max(1e-6);
    let sy = (scale_y as f64).abs().max(1e-6);
    let cx = u as f64 + 0.5;
    let cy = v as f64 + 0.5;
    let x0 = (cx - 0.5 * sx).clamp(0.0, src.ifd_w as f64);
    let x1 = (cx + 0.5 * sx).clamp(0.0, src.ifd_w as f64);
    let y0 = (cy - 0.5 * sy).clamp(0.0, src.ifd_h as f64);
    let y1 = (cy + 0.5 * sy).clamp(0.0, src.ifd_h as f64);
    if x1 <= x0 + 1e-12 || y1 <= y0 + 1e-12 {
        return f32::NAN;
    }

    let ix0 = x0.floor() as i32;
    let ix1 = x1.next_down().floor() as i32;
    let iy0 = y0.floor() as i32;
    let iy1 = y1.next_down().floor() as i32;

    // Fast 1-pixel path: exact 1:1 overview read or upsampling strictly inside one source pixel.
    if ix0 == ix1 && iy0 == iy1 {
        let s = src.sample_checked(ix0, iy0);
        return if matches!(mode, ResampleMode::Sum) && !s.is_nan() {
            ((s as f64) * (x1 - x0) * (y1 - y0)) as f32
        } else {
            s
        };
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn visit_footprint<F: FnMut(f32, f64)>(
        src: &BandSource<'_>,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        ix0: i32,
        ix1: i32,
        iy0: i32,
        iy1: i32,
        mut cb: F,
    ) {
        const MAX_EXACT_SPAN: i32 = 64;
        let nx = ix1 - ix0 + 1;
        let ny = iy1 - iy0 + 1;
        if nx <= MAX_EXACT_SPAN && ny <= MAX_EXACT_SPAN {
            for iy in iy0..=iy1 {
                let wy = ((iy + 1) as f64).min(y1) - (iy as f64).max(y0);
                if wy <= 1e-9 {
                    continue;
                }
                for ix in ix0..=ix1 {
                    let wx = ((ix + 1) as f64).min(x1) - (ix as f64).max(x0);
                    if wx <= 1e-9 {
                        continue;
                    }
                    let s = src.sample_checked(ix, iy);
                    if !s.is_nan() {
                        cb(s, wx * wy);
                    }
                }
            }
        } else {
            let steps_x = nx.min(MAX_EXACT_SPAN) as usize;
            let steps_y = ny.min(MAX_EXACT_SPAN) as usize;
            let dx = (x1 - x0) / (steps_x as f64);
            let dy = (y1 - y0) / (steps_y as f64);
            let cell_w = dx * dy;
            for gy in 0..steps_y {
                let iy = (y0 + (gy as f64 + 0.5) * dy).floor() as i32;
                for gx in 0..steps_x {
                    let ix = (x0 + (gx as f64 + 0.5) * dx).floor() as i32;
                    let s = src.sample_checked(ix, iy);
                    if !s.is_nan() {
                        cb(s, cell_w);
                    }
                }
            }
        }
    }

    match mode {
        ResampleMode::Average => {
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, w| {
                sum += (s as f64) * w;
                wsum += w;
            });
            if wsum > 0.0 {
                (sum / wsum) as f32
            } else {
                f32::NAN
            }
        }
        ResampleMode::Sum => {
            let mut sum = 0.0f64;
            let mut any = false;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, w| {
                sum += (s as f64) * w;
                any = true;
            });
            if any { sum as f32 } else { f32::NAN }
        }
        ResampleMode::Rms => {
            let mut sq = 0.0f64;
            let mut wsum = 0.0f64;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, w| {
                let sf = s as f64;
                sq += sf * sf * w;
                wsum += w;
            });
            if wsum > 0.0 {
                (sq / wsum).sqrt() as f32
            } else {
                f32::NAN
            }
        }
        ResampleMode::Min => {
            let mut best = f32::INFINITY;
            let mut any = false;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, _| {
                if s < best {
                    best = s;
                }
                any = true;
            });
            if any { best } else { f32::NAN }
        }
        ResampleMode::Max => {
            let mut best = f32::NEG_INFINITY;
            let mut any = false;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, _| {
                if s > best {
                    best = s;
                }
                any = true;
            });
            if any { best } else { f32::NAN }
        }
        ResampleMode::Mode => {
            // Stack-allocated fast path for <= 64 distinct values (covers up to 8x8 exact blocks
            // with zero heap allocation); falls back to Vec only for larger footprints.
            let mut stack_bins = [(0u32, 0.0f32, 0.0f64); 64];
            let mut stack_len = 0usize;
            let mut heap_bins: Option<Vec<(u32, f32, f64)>> = None;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, w| {
                // Normalize -0.0 to +0.0 so they bin together.
                let norm = if s == 0.0 { 0.0f32 } else { s };
                let bits = norm.to_bits();
                if let Some(ref mut bins) = heap_bins {
                    for b in bins.iter_mut() {
                        if b.0 == bits {
                            b.2 += w;
                            return;
                        }
                    }
                    bins.push((bits, norm, w));
                } else {
                    for b in &mut stack_bins[..stack_len] {
                        if b.0 == bits {
                            b.2 += w;
                            return;
                        }
                    }
                    if stack_len < stack_bins.len() {
                        stack_bins[stack_len] = (bits, norm, w);
                        stack_len += 1;
                    } else {
                        let mut v = Vec::with_capacity(128);
                        v.extend_from_slice(&stack_bins);
                        v.push((bits, norm, w));
                        heap_bins = Some(v);
                    }
                }
            });
            let bins: &[(u32, f32, f64)] = match heap_bins {
                Some(ref v) => v.as_slice(),
                None => &stack_bins[..stack_len],
            };
            if bins.is_empty() {
                return f32::NAN;
            }
            let mut best_val = bins[0].1;
            let mut best_w = bins[0].2;
            for &(_, val, wt) in &bins[1..] {
                if wt > best_w {
                    best_w = wt;
                    best_val = val;
                }
            }
            best_val
        }
        ResampleMode::Med | ResampleMode::Q1 | ResampleMode::Q3 => {
            let mut stack_vals = [0.0f32; 64];
            let mut stack_len = 0usize;
            let mut heap_vals: Option<Vec<f32>> = None;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, _| {
                if let Some(ref mut v) = heap_vals {
                    v.push(s);
                } else if stack_len < stack_vals.len() {
                    stack_vals[stack_len] = s;
                    stack_len += 1;
                } else {
                    let mut v = Vec::with_capacity(128);
                    v.extend_from_slice(&stack_vals);
                    v.push(s);
                    heap_vals = Some(v);
                }
            });
            let slice: &mut [f32] = match heap_vals {
                Some(ref mut v) => v.as_mut_slice(),
                None => &mut stack_vals[..stack_len],
            };
            if slice.is_empty() {
                return f32::NAN;
            }
            slice.sort_unstable_by(f32::total_cmp);
            let q = match mode {
                ResampleMode::Q1 => 0.25f32,
                ResampleMode::Med => 0.50f32,
                ResampleMode::Q3 => 0.75f32,
                _ => 0.50f32,
            };
            let pos = q * ((slice.len() - 1) as f32);
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(slice.len() - 1);
            let frac = pos - lo as f32;
            slice[lo] + (slice[hi] - slice[lo]) * frac
        }
        _ => unreachable!(),
    }
}

/// Estimate source-pixel footprint `(scale_x, scale_y)` per output pixel from a `CoordPlan`.
#[inline]
pub fn coord_plan_scale<const N: usize>(plan: &CoordPlan<N>) -> (f32, f32) {
    match plan {
        CoordPlan::Separable1D { u, v, .. } => {
            let denom = (N.saturating_sub(1)).max(1) as f32;
            let sx = (u[N - 1] - u[0]).abs() / denom;
            let sy = (v[N - 1] - v[0]).abs() / denom;
            (
                if sx.is_finite() && sx > 1e-4 { sx } else { 1.0 },
                if sy.is_finite() && sy > 1e-4 { sy } else { 1.0 },
            )
        }
        CoordPlan::Grid2D { grid_u, grid_v, .. } => {
            let cell_px = (N / (GRID_DIM - 1)).max(1) as f32;
            let mid = (GRID_DIM - 1) / 2;
            let i00 = mid * GRID_DIM + mid;
            let i10 = i00 + 1;
            let i01 = (mid + 1) * GRID_DIM + mid;
            let sx = (grid_u[i10] - grid_u[i00]).hypot(grid_v[i10] - grid_v[i00]) / cell_px;
            let sy = (grid_u[i01] - grid_u[i00]).hypot(grid_v[i01] - grid_v[i00]) / cell_px;
            (
                if sx.is_finite() && sx > 1e-4 { sx } else { 1.0 },
                if sy.is_finite() && sy > 1e-4 { sy } else { 1.0 },
            )
        }
        CoordPlan::Exact2D { u, v, .. } => {
            if N <= 1 {
                return (1.0, 1.0);
            }
            let probe = |py: usize, px: usize| -> Option<(f32, f32)> {
                let i00 = py * N + px;
                let i10 = i00 + 1;
                let i01 = (py + 1) * N + px;
                let sx = (u[i10] - u[i00]).hypot(v[i10] - v[i00]);
                let sy = (u[i01] - u[i00]).hypot(v[i01] - v[i00]);
                if sx.is_finite() && sx > 1e-4 && sy.is_finite() && sy > 1e-4 {
                    Some((sx, sy))
                } else {
                    None
                }
            };
            let mid = (N / 2).min(N - 2);
            if let Some(s) = probe(mid, mid) {
                return s;
            }
            let step = (N / 16).max(1);
            for py in (0..N - 1).step_by(step) {
                for px in (0..N - 1).step_by(step) {
                    if let Some(s) = probe(py, px) {
                        return s;
                    }
                }
            }
            for py in 0..N - 1 {
                for px in 0..N - 1 {
                    if let Some(s) = probe(py, px) {
                        return s;
                    }
                }
            }
            (1.0, 1.0)
        }
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

                let du = u_r - u_l;
                let dv = v_r - v_l;
                let base_x = gx * cell_px;
                for k in 0..cell_px {
                    let tx = (k as f32) * inv_cell;
                    f(base_x + k, u_l + du * tx, v_l + dv * tx);
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

/// Encode a 64-byte-aligned NumPy v1.0 `.npy` header for a C-contiguous array of the
/// dtype `descr` (`<f4`, `|u1`, ...) and of `shape`: the magic, the version, the length of
/// what follows, then the dict padded with spaces and ended by a newline so that the data
/// starts on a multiple of 64 bytes. Every `.npy` this crate writes begins with it.
pub fn encode_npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    use std::fmt::Write;
    let mut dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': (");
    for (i, dim) in shape.iter().enumerate() {
        if i > 0 {
            dict.push_str(", ");
        }
        let _ = write!(dict, "{dim}");
    }
    // A tuple of one is written `(n,)`.
    dict.push_str(if shape.len() == 1 { ",), }" } else { "), }" });
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    for _ in 0..pad {
        dict.push(' ');
    }
    dict.push('\n');
    let hlen = dict.len() as u16;
    let mut hdr = Vec::with_capacity(10 + dict.len());
    hdr.extend_from_slice(b"\x93NUMPY\x01\x00");
    hdr.extend_from_slice(&hlen.to_le_bytes());
    hdr.extend_from_slice(dict.as_bytes());
    hdr
}

/// Encode a 64-byte-aligned NumPy v1.0 `.npy` header for a C-contiguous `<f4` array
/// of shape `(bands, height, width)`.
pub fn encode_npy_f32_header(bands: usize, height: u32, width: u32) -> Vec<u8> {
    encode_npy_header("<f4", &[bands, height as usize, width as usize])
}

/// The NumPy dtype string of a sample type: little-endian, and `|` for the single bytes.
pub fn npy_descr(kind: SampleType) -> &'static str {
    match kind {
        SampleType::Uint8 => "|u1",
        SampleType::Int8 => "|i1",
        SampleType::Uint16 => "<u2",
        SampleType::Int16 => "<i2",
        SampleType::Uint32 => "<u4",
        SampleType::Int32 => "<i4",
        SampleType::Float32 => "<f4",
        SampleType::Float64 => "<f8",
    }
}

/// A `.npy` array of `bands` planes of `height` by `width`, written in the dtype of `kind`
/// from CHW float32 `values` in which NaN marks a value that is not there.
///
/// A value that is not there is written as `absent`, or as zero without one. Integers take
/// the value truncated toward zero and held to the type's range (a NaN `absent` is zero);
/// the float types hold an infinity to their largest finite number. With `mask`, one more
/// plane follows the bands, as rio-tiler's `ImageData.mask` is: the type's largest number
/// (`255` for `u1`, `65535` for `u2`, `f32::MAX` for `f4`, ...) where any band has a value,
/// and its smallest elsewhere (`0` for the unsigned types, `-32768` for `i2`, `f32::MIN`
/// for `f4`, ...). Only the first
/// `bands * height * width` of `values` are written, and a shorter slice makes a shorter
/// body under a header that still names every plane.
///
/// This is not the float32 path of [`wrap_raw_f32_as_npy_with_mask`], which copies its
/// planes as they are, counts an infinity as absent and writes `255.0` in its mask.
pub fn encode_npy_typed(
    values: &[f32],
    bands: usize,
    height: usize,
    width: usize,
    kind: SampleType,
    absent: Option<f32>,
    mask: bool,
) -> Bytes {
    let pixels = width * height;
    let planes = bands + usize::from(mask);
    let hdr = encode_npy_header(npy_descr(kind), &[planes, height, width]);
    let mut out = Vec::with_capacity(hdr.len() + planes * pixels * kind.bytes_per_sample());
    out.extend_from_slice(&hdr);
    let absent = f64::from(absent.unwrap_or(0.0));
    let data = &values[..(bands * pixels).min(values.len())];
    // One loop for each type, so the type is not asked about per value.
    macro_rules! write_planes {
        ($t:ty, $cast:expr) => {{
            let cast = $cast;
            for v in data {
                let x: $t = cast(if v.is_nan() { absent } else { f64::from(*v) });
                out.extend_from_slice(&x.to_le_bytes());
            }
            if mask {
                for i in 0..pixels {
                    let there = (0..bands).any(|band| data.get(band * pixels + i).is_some_and(|v| !v.is_nan()));
                    let x: $t = if there { <$t>::MAX } else { <$t>::MIN };
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }};
    }
    match kind {
        SampleType::Uint8 => write_planes!(u8, |v: f64| v as u8),
        SampleType::Int8 => write_planes!(i8, |v: f64| v as i8),
        SampleType::Uint16 => write_planes!(u16, |v: f64| v as u16),
        SampleType::Int16 => write_planes!(i16, |v: f64| v as i16),
        SampleType::Uint32 => write_planes!(u32, |v: f64| v as u32),
        SampleType::Int32 => write_planes!(i32, |v: f64| v as i32),
        SampleType::Float32 => write_planes!(f32, |v: f64| (v as f32).clamp(f32::MIN, f32::MAX)),
        SampleType::Float64 => write_planes!(f64, |v: f64| v.clamp(f64::MIN, f64::MAX)),
    }
    Bytes::from(out)
}

/// Prepend a NumPy v1.0 `.npy` `<f4` header to raw CHW little-endian float32 bytes.
pub fn wrap_raw_f32_as_npy(raw: Bytes, bands: usize, height: u32, width: u32) -> Bytes {
    wrap_raw_f32_as_npy_with_mask(raw, bands, height, width, false)
}

/// Prepend a NumPy v1.0 `.npy` `<f4` header to raw CHW little-endian float32 bytes,
/// optionally stacking a `(1, H, W)` float32 validity mask band (`255.0` valid / `0.0` nodata)
/// when `return_mask` is `true` (matching TiTiler / `rio-tiler` `.npy` output).
pub fn wrap_raw_f32_as_npy_with_mask(
    raw: Bytes,
    bands: usize,
    height: u32,
    width: u32,
    return_mask: bool,
) -> Bytes {
    if !return_mask {
        let hdr = encode_npy_f32_header(bands, height, width);
        let mut out = BytesMut::with_capacity(hdr.len() + raw.len());
        out.put_slice(&hdr);
        out.put_slice(&raw);
        return out.freeze();
    }
    let n_px = (height as usize).saturating_mul(width as usize);
    let plane_bytes = n_px * 4;
    let hdr = encode_npy_f32_header(bands + 1, height, width);
    let mut out = BytesMut::with_capacity(hdr.len() + raw.len() + plane_bytes);
    out.put_slice(&hdr);
    out.put_slice(&raw);
    let valid_le = 255.0f32.to_le_bytes();
    let zero_le = 0.0f32.to_le_bytes();
    for px in 0..n_px {
        let px_off = px * 4;
        let mut any_valid = false;
        for b in 0..bands {
            let off = b * plane_bytes + px_off;
            if off + 4 <= raw.len() {
                let v = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                if v.is_finite() {
                    any_valid = true;
                    break;
                }
            }
        }
        out.put_slice(if any_valid { &valid_le } else { &zero_le });
    }
    out.freeze()
}

pub fn empty_tile_bytes(tile_size: u32, format: OutputFormat, num_bands: usize) -> Bytes {
    use std::sync::OnceLock;
    match format {
        OutputFormat::Png => empty_transparent_png(tile_size),
        OutputFormat::Webp { .. } => empty_transparent_webp(tile_size),
        OutputFormat::Jpeg { .. } => empty_black_jpeg(tile_size),
        OutputFormat::RawF32 => {
            let build = |sz: u32, nb: usize| -> Bytes {
                let count = (sz as usize) * (sz as usize) * nb;
                let mut buf = BytesMut::with_capacity(count * 4);
                let nan_bytes = f32::NAN.to_le_bytes();
                for _ in 0..count {
                    buf.put_slice(&nan_bytes);
                }
                buf.freeze()
            };
            static F32_256_1: OnceLock<Bytes> = OnceLock::new();
            static F32_256_3: OnceLock<Bytes> = OnceLock::new();
            static F32_512_1: OnceLock<Bytes> = OnceLock::new();
            static F32_512_3: OnceLock<Bytes> = OnceLock::new();
            match (tile_size, num_bands) {
                (256, 1) => F32_256_1.get_or_init(|| build(256, 1)).clone(),
                (256, 3) => F32_256_3.get_or_init(|| build(256, 3)).clone(),
                (512, 1) => F32_512_1.get_or_init(|| build(512, 1)).clone(),
                (512, 3) => F32_512_3.get_or_init(|| build(512, 3)).clone(),
                _ => build(tile_size, num_bands),
            }
        }
        OutputFormat::Npy => {
            let hdr = encode_npy_f32_header(num_bands, tile_size, tile_size);
            let count = (tile_size as usize) * (tile_size as usize) * num_bands;
            let mut buf = BytesMut::with_capacity(hdr.len() + count * 4);
            buf.put_slice(&hdr);
            let nan_bytes = f32::NAN.to_le_bytes();
            for _ in 0..count {
                buf.put_slice(&nan_bytes);
            }
            buf.freeze()
        }
        OutputFormat::RawRgba => {
            static RGBA_256: OnceLock<Bytes> = OnceLock::new();
            static RGBA_512: OnceLock<Bytes> = OnceLock::new();
            match tile_size {
                256 => RGBA_256.get_or_init(|| Bytes::from(vec![0u8; 256 * 256 * 4])).clone(),
                512 => RGBA_512.get_or_init(|| Bytes::from(vec![0u8; 512 * 512 * 4])).clone(),
                _ => Bytes::from(vec![0u8; (tile_size as usize) * (tile_size as usize) * 4]),
            }
        }
    }
}

#[inline]
fn rgb_to_rgba_bytes(rgb: &[u8]) -> Bytes {
    let pixels = rgb.len() / 3;
    let mut out = Vec::with_capacity(pixels * 4);
    for chunk in rgb.chunks_exact(3) {
        out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
    }
    Bytes::from(out)
}

/// Encode an RGBA8 tile buffer into PNG (using opaque RGB when `all_opaque`), WebP, JPEG, or raw RGBA.
pub fn encode_rgba_tile(rgba: &[u8], size: u32, all_opaque: bool, format: OutputFormat) -> Bytes {
    let n = size as usize;
    match format {
        OutputFormat::Png => {
            let encode_with = |filtered: &mut Vec<u8>, rgb_row: &mut Vec<u8>| -> Bytes {
                if all_opaque {
                    let row_len = n * 3 + 1;
                    let total = n * row_len;
                    if filtered.len() < total {
                        filtered.resize(total, 0);
                    }
                    if rgb_row.len() < n * 3 {
                        rgb_row.resize(n * 3, 0);
                    }
                    let row_buf = &mut rgb_row[..n * 3];
                    for y in 0..n {
                        let src = &rgba[y * n * 4..(y + 1) * n * 4];
                        for x in 0..n {
                            row_buf[x * 3] = src[x * 4];
                            row_buf[x * 3 + 1] = src[x * 4 + 1];
                            row_buf[x * 3 + 2] = src[x * 4 + 2];
                        }
                        filter_sub_rgb_row(
                            row_buf,
                            &mut filtered[y * row_len..(y + 1) * row_len],
                        );
                    }
                    encode_filtered_png(size, size, 2, &filtered[..total])
                } else {
                    let row_len = n * 4 + 1;
                    let total = n * row_len;
                    if filtered.len() < total {
                        filtered.resize(total, 0);
                    }
                    for y in 0..n {
                        let src = &rgba[y * n * 4..(y + 1) * n * 4];
                        filter_sub_rgba_row(src, &mut filtered[y * row_len..(y + 1) * row_len]);
                    }
                    encode_filtered_png(size, size, 6, &filtered[..total])
                }
            };
            SCRATCH.with(|cell| {
                if let Ok(mut guard) = cell.try_borrow_mut() {
                    let sc = &mut *guard;
                    encode_with(&mut sc.img_buf, &mut sc.row_buf)
                } else {
                    let mut filtered = Vec::new();
                    let mut rgb_row = Vec::new();
                    encode_with(&mut filtered, &mut rgb_row)
                }
            })
        }
        OutputFormat::Webp { quality } => encode_webp_rgba(size, size, rgba, quality),
        OutputFormat::Jpeg { quality } => encode_jpeg_rgba(size, size, rgba, quality),
        OutputFormat::RawRgba | OutputFormat::RawF32 | OutputFormat::Npy => {
            Bytes::copy_from_slice(rgba)
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
    let range = safe_rescale_denom(style.rescale_min, style.rescale_max);
    let inv_range = 255.0 / range;
    let min_val = style.rescale_min;

    // Ultra-hot Nearest path: Separable1D + Nearest + CleanFastPath + Single-Block (F32/U16/U8)!
    // Hoists `[usize; N]` X-indices once outside `for py in 0..N` for direct flat slice indexing.
    if !BILINEAR && CLEAN {
        if let (CoordPlan::Separable1D { u, v, .. }, Some(blk)) =
            (plan, SingleBlockRef::from_source(src))
        {
            let x_idx = blk.precompute_nearest_x(u);
            let max_y = (blk.height() as i32 - 1).max(0);
            let br_f = blk.base_row() as f32;
            match style.format {
                OutputFormat::RawF32 | OutputFormat::Npy => {
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
                            encode_indexed_png(N as u32, N as u32, &plte, None, &sc.img_buf[..total])
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
    }

    // Ultra-hot SIMD path: Separable1D + Bilinear + CleanFastPath + Single-Block (F32/U16/U8)!
    if BILINEAR && CLEAN {
        if let (CoordPlan::Separable1D { u, v, .. }, Some(blk)) =
            (plan, SingleBlockRef::from_source(src))
        {
            match style.format {
                OutputFormat::RawF32 | OutputFormat::Npy => {
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
                            encode_indexed_png(N as u32, N as u32, &plte, None, &sc.img_buf[..total])
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
    }

    match style.format {
        OutputFormat::RawF32 | OutputFormat::Npy => {
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
                        encode_indexed_png(N as u32, N as u32, &plte, None, &sc.img_buf[..total])
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
                    let plte = build_plte_nodata(lut);
                    let inv_range_254 = 254.0 / range;
                    for py in 0..N {
                        let row_idx = &mut sc.row_buf[..N];
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = sample_val::<BILINEAR, false>(src, u, v);
                            if val.is_nan() {
                                row_idx[px] = 0;
                            } else {
                                let t = ((val - min_val) * inv_range_254).clamp(0.0, 254.0);
                                row_idx[px] = 1 + (t.round() as u8);
                            }
                        });
                        filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                    }
                    encode_indexed_png(
                        N as u32,
                        N as u32,
                        &plte,
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
        (ResampleMode::Bilinear, true) => render_single_specialized::<N, true, true>(plan, src, style),
        (ResampleMode::Bilinear, false) => render_single_specialized::<N, true, false>(plan, src, style),
        (ResampleMode::Nearest, true) => render_single_specialized::<N, false, true>(plan, src, style),
        (ResampleMode::Nearest, false) => render_single_specialized::<N, false, false>(plan, src, style),
        (ResampleMode::Cubic, _) => render_eval_fn(plan, style, |u, v| sample_cubic(src, u, v)),
        (other, _) => {
            let (sx, sy) = coord_plan_scale(plan);
            render_eval_fn(plan, style, |u, v| sample_general_scaled(src, u, v, sx, sy, other))
        }
    }
}

// ============================================================================
// Monomorphized 3-Band Satellite RGB Fused Kernels
// ============================================================================

#[inline(always)]
fn fill_uv_row<const N: usize>(
    plan: &CoordPlan<N>,
    py: usize,
    u_row: &mut [f32; N],
    v_row: &mut [f32; N],
) {
    match plan {
        CoordPlan::Separable1D { u, v, .. } => {
            u_row.copy_from_slice(&u[..]);
            v_row.fill(v[py]);
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
                let du = u_r - u_l;
                let dv = v_r - v_l;
                let base_x = gx * cell_px;
                for k in 0..cell_px {
                    let tx = (k as f32) * inv_cell;
                    u_row[base_x + k] = u_l + du * tx;
                    v_row[base_x + k] = v_l + dv * tx;
                }
            }
        }
        CoordPlan::Exact2D { u, v, .. } => {
            let row_off = py * N;
            u_row.copy_from_slice(&u[row_off..row_off + N]);
            v_row.copy_from_slice(&v[row_off..row_off + N]);
        }
    }
}

/// Fused 3-band bilinear + rescale for a single-block raw typed slice (`u8`, `u16`, or `f32`).
/// Computes floor/frac and `(p00, p01)` once per pixel across all 3 bands, with NEON 4-wide /
/// AVX2 8-wide FMA bilinear + rescale.
#[inline(always)]
fn bilinear_rgb_single_block_row<const N: usize, T: Copy, F>(
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
        let v_half = vdupq_n_f32(0.5);
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
                    let rounded = vaddq_f32(clamped, v_half);
                    vst1q_u32($out.as_mut_ptr(), vcvtq_u32_f32(rounded));
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
    let v_half = _mm256_set1_ps(0.5);
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
        let u = _mm256_sub_ps(_mm256_loadu_ps(u_row.as_ptr().add(px)), v_bc);
        let v = _mm256_sub_ps(_mm256_loadu_ps(v_row.as_ptr().add(px)), v_br);
        let x0_f = _mm256_floor_ps(u);
        let y0_f = _mm256_floor_ps(v);
        let fx = _mm256_sub_ps(u, x0_f);
        let fy = _mm256_sub_ps(v, y0_f);
        let vx = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(x0_f), v_zero_i), v_max_x);
        let vy = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(y0_f), v_zero_i), v_max_y);
        _mm256_storeu_si256(xs.as_mut_ptr() as *mut __m256i, vx);
        _mm256_storeu_si256(ys.as_mut_ptr() as *mut __m256i, vy);

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
                    a00[k] = to_f32(*$ptr.add(pk));
                    a10[k] = to_f32(*$ptr.add(pk + 1));
                    a01[k] = to_f32(*$ptr.add(pk + stride));
                    a11[k] = to_f32(*$ptr.add(pk + stride + 1));
                }
                let v00 = _mm256_loadu_ps(a00.as_ptr());
                let v10 = _mm256_loadu_ps(a10.as_ptr());
                let v01 = _mm256_loadu_ps(a01.as_ptr());
                let v11 = _mm256_loadu_ps(a11.as_ptr());
                let top = _mm256_fmadd_ps(_mm256_sub_ps(v10, v00), fx, v00);
                let bot = _mm256_fmadd_ps(_mm256_sub_ps(v11, v01), fx, v01);
                let val = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), fy, top);
                let scaled = _mm256_mul_ps(_mm256_sub_ps(val, $v_min), $v_inv);
                let clamped = _mm256_min_ps(_mm256_max_ps(scaled, v_zero), v_255);
                let rounded = _mm256_add_ps(clamped, v_half);
                _mm256_storeu_si256(
                    $out.as_mut_ptr() as *mut __m256i,
                    _mm256_cvttps_epi32(rounded),
                );
            }};
        }

        eval_band_avx2!(r_ptr, v_inv0, v_min0, out_r);
        eval_band_avx2!(g_ptr, v_inv1, v_min1, out_g);
        eval_band_avx2!(b_ptr, v_inv2, v_min2, out_b);

        let base = px * 3;
        for k in 0..8 {
            *row_rgb.get_unchecked_mut(base + k * 3) = out_r[k] as u8;
            *row_rgb.get_unchecked_mut(base + k * 3 + 1) = out_g[k] as u8;
            *row_rgb.get_unchecked_mut(base + k * 3 + 2) = out_b[k] as u8;
        }
        px += 8;
    }
}

/// Fused clean RGB nearest-neighbor scanline sampler (TiTiler default resampling):
/// - Fast path 0: shared interleaved `RgbU8` slab (single 3-byte load per pixel, direct copy on identity).
/// - Fast path 1: single-block `U8`, `U16`, or `F32` (zero block-table lookups per pixel).
/// - Fast path 2: multi-block aligned grid (computes `(x, y)` and block index once per pixel instead of 3×).
/// - Fallback: per-band `sample_nearest::<true>` for heterogeneous grids.
fn sample_rgb_nearest_row_clean<const N: usize>(
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
    ) {
        if std::ptr::eq(rgb0.as_ptr(), rgb1.as_ptr())
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
    }

    // Fast path 1a: single-block planar U8 across all 3 bands.
    if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
        srcs[0].single_block_u8(),
        srcs[1].single_block_u8(),
        srcs[2].single_block_u8(),
    ) {
        if bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
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
    }

    // Fast path 1b: single-block planar U16 across all 3 bands (Sentinel-2 / Landsat).
    if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
        srcs[0].single_block_u16(),
        srcs[1].single_block_u16(),
        srcs[2].single_block_u16(),
    ) {
        if bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
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
    }

    // Fast path 1c: single-block planar F32 across all 3 bands.
    if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
        srcs[0].single_block_flat(),
        srcs[1].single_block_flat(),
        srcs[2].single_block_flat(),
    ) {
        if bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
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
                        let clamped = vminq_f32(vmaxq_f32(scaled, v_zero), v_255);
                        vaddq_f32(clamped, v_half)
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
        let u = _mm256_sub_ps(_mm256_loadu_ps(u_coords.as_ptr().add(px)), v_bc);
        let x0_f = _mm256_floor_ps(u);
        let fx = _mm256_sub_ps(u, x0_f);
        let vx = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(x0_f), v_zero_i), v_max_x);
        _mm256_storeu_si256(xs.as_mut_ptr() as *mut __m256i, vx);

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
                    a00[k] = *r0_ptr.add(o + $ch) as f32;
                    a10[k] = *r0_ptr.add(o + 3 + $ch) as f32;
                    a01[k] = *r1_ptr.add(o + $ch) as f32;
                    a11[k] = *r1_ptr.add(o + 3 + $ch) as f32;
                }
                let v00 = _mm256_loadu_ps(a00.as_ptr());
                let v10 = _mm256_loadu_ps(a10.as_ptr());
                let v01 = _mm256_loadu_ps(a01.as_ptr());
                let v11 = _mm256_loadu_ps(a11.as_ptr());
                let top = _mm256_fmadd_ps(_mm256_sub_ps(v10, v00), fx, v00);
                let bot = _mm256_fmadd_ps(_mm256_sub_ps(v11, v01), fx, v01);
                let val = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), v_fy, top);
                let rounded = if identity {
                    _mm256_add_ps(val, v_half)
                } else {
                    let scaled = _mm256_mul_ps(_mm256_sub_ps(val, $v_min), $v_inv);
                    let clamped = _mm256_min_ps(_mm256_max_ps(scaled, v_zero), v_255);
                    _mm256_add_ps(clamped, v_half)
                };
                _mm256_storeu_si256($out.as_mut_ptr() as *mut __m256i, _mm256_cvttps_epi32(rounded));
            }};
        }

        eval_ch!(0, v_inv0, v_min0, out_r);
        eval_ch!(1, v_inv1, v_min1, out_g);
        eval_ch!(2, v_inv2, v_min2, out_b);

        let base = px * 3;
        for k in 0..8 {
            *row_rgb.get_unchecked_mut(base + k * 3) = out_r[k] as u8;
            *row_rgb.get_unchecked_mut(base + k * 3 + 1) = out_g[k] as u8;
            *row_rgb.get_unchecked_mut(base + k * 3 + 2) = out_b[k] as u8;
        }
        px += 8;
    }
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc);
        let o = (x0_raw.clamp(0, max_x) as usize) * 3;
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
        px += 1;
    }
}

/// Fused clean RGB bilinear scanline sampler:
/// - Fast path 0: shared interleaved `RgbU8` slab (NEON/AVX2 on `Separable1D`, 1 cache-line stream).
/// - Fast path 1: single-block `U8`, `U16`, or `F32` (NEON 4-wide / AVX2 8-wide, zero block lookups).
/// - Fast path 2: multi-block aligned grid (computes floor/frac + `(bidx, p00, p01)` once per pixel instead of 3×).
/// - Fallback: per-band sampling for heterogeneous grids.
fn sample_rgb_bilinear_row_clean<const N: usize>(
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
    ) {
        if std::ptr::eq(rgb0.as_ptr(), rgb1.as_ptr())
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
    }

    // Fast path 1: all 3 bands live in a single clean block with matching native storage
    if let (Some((r, bc, br, st)), Some((g, _, _, _)), Some((b, _, _, _))) = (
        srcs[0].single_block_u8(),
        srcs[1].single_block_u8(),
        srcs[2].single_block_u8(),
    ) {
        if srcs[0].same_grid_as(&srcs[1]) && srcs[0].same_grid_as(&srcs[2]) {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            bilinear_rgb_single_block_row(
                &u_row, &v_row, r, g, b, bc, br, st, inv, mins, row_rgb, |v| v as f32,
            );
            return;
        }
    }
    if let (Some((r, bc, br, st)), Some((g, _, _, _)), Some((b, _, _, _))) = (
        srcs[0].single_block_u16(),
        srcs[1].single_block_u16(),
        srcs[2].single_block_u16(),
    ) {
        if srcs[0].same_grid_as(&srcs[1]) && srcs[0].same_grid_as(&srcs[2]) {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            bilinear_rgb_single_block_row(
                &u_row, &v_row, r, g, b, bc, br, st, inv, mins, row_rgb, |v| v as f32,
            );
            return;
        }
    }
    if let (Some((r, bc, br, st)), Some((g, _, _, _)), Some((b, _, _, _))) = (
        srcs[0].single_block_flat(),
        srcs[1].single_block_flat(),
        srcs[2].single_block_flat(),
    ) {
        if srcs[0].same_grid_as(&srcs[1]) && srcs[0].same_grid_as(&srcs[2]) {
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            bilinear_rgb_single_block_row(
                &u_row, &v_row, r, g, b, bc, br, st, inv, mins, row_rgb, |v| v,
            );
            return;
        }
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
enum CleanRgbNearestPlan<'a, const N: usize> {
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
    fn build(
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
        ) {
            if std::ptr::eq(rgb0.as_ptr(), rgb1.as_ptr())
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
        }
        if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
            srcs[0].single_block_u8(),
            srcs[1].single_block_u8(),
            srcs[2].single_block_u8(),
        ) {
            if bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
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
        }
        if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
            srcs[0].single_block_u16(),
            srcs[1].single_block_u16(),
            srcs[2].single_block_u16(),
        ) {
            if bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
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
        }
        if let (Some((r, bc, br, st)), Some((g, bc1, br1, st1)), Some((b, bc2, br2, st2))) = (
            srcs[0].single_block_flat(),
            srcs[1].single_block_flat(),
            srcs[2].single_block_flat(),
        ) {
            if bc == bc1 && br == br1 && st == st1 && bc == bc2 && br == br2 && st == st2 {
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
        }
        Self::Generic
    }

    #[inline(always)]
    fn sample_row(
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
        OutputFormat::RawF32 | OutputFormat::Npy => {
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
        (ResampleMode::Bilinear, true) => render_rgb_specialized::<N, true, true>(plan, srcs, style),
        (ResampleMode::Bilinear, false) => render_rgb_specialized::<N, true, false>(plan, srcs, style),
        (ResampleMode::Nearest, true) => render_rgb_specialized::<N, false, true>(plan, srcs, style),
        (ResampleMode::Nearest, false) => render_rgb_specialized::<N, false, false>(plan, srcs, style),
        (ResampleMode::Cubic, _) => render_rgb_cubic::<N>(plan, srcs, style),
        (_, _) => render_rgb_cubic::<N>(plan, srcs, style),
    }
}

fn render_rgb_cubic<const N: usize>(
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
    let mode = style.resample;
    let (sx, sy) = coord_plan_scale(plan);
    let sample_px = |src: &BandSource<'_>, u: f32, v: f32| -> f32 {
        if mode == ResampleMode::Cubic {
            sample_cubic(src, u, v)
        } else {
            sample_general_scaled(src, u, v, sx, sy, mode)
        }
    };
    match style.format {
        OutputFormat::RawF32 | OutputFormat::Npy => {
            let mut out = BytesMut::with_capacity(N * N * 12);
            for b in 0..3 {
                for py in 0..N {
                    for_each_pixel_in_row(plan, py, |_px, u, v| {
                        out.put_slice(&sample_px(&srcs[b], u, v).to_le_bytes());
                    });
                }
            }
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
                let row = &mut sc.row_buf[..N * 4];
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    let r = sample_px(&srcs[0], u, v);
                    let g = sample_px(&srcs[1], u, v);
                    let b = sample_px(&srcs[2], u, v);
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
                    let r = sample_px(&srcs[0], u, v);
                    let g = sample_px(&srcs[1], u, v);
                    let b = sample_px(&srcs[2], u, v);
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

// ============================================================================
// Fused Satellite Spectral Band Math Kernels (NDVI, NDWI, NBR, SAVI, EVI)
// Zero intermediate float tile allocation — computed directly in scanline loop!
// ============================================================================

fn render_eval_fn<const N: usize, F: Fn(f32, f32) -> f32>(
    plan: &CoordPlan<N>,
    style: &SingleBandStyle,
    eval_uv: F,
) -> Bytes {
    let range = safe_rescale_denom(style.rescale_min, style.rescale_max);
    let inv_range = 255.0 / range;
    let min_val = style.rescale_min;

    match style.format {
        OutputFormat::RawF32 | OutputFormat::Npy => {
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
                let plte = build_plte_nodata(lut);
                let inv_range_254 = 254.0 / range;
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
                                let t = ((val - min_val) * inv_range_254).clamp(0.0, 254.0);
                                1 + (t.round() as u8)
                            };
                        });
                        filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                    }
                    encode_indexed_png(
                        N as u32,
                        N as u32,
                        &plte,
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
    let clean = plan.all_in_bounds() && src.is_clean_inbounds(c0, c1, r0, r1);

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
        (ResampleMode::Cubic, _) => {
            render_eval_fn(plan, style, |u, v| expr.eval1(sample_cubic(src, u, v)))
        }
        (other, _) => {
            let (scale_x, scale_y) = coord_plan_scale(plan);
            render_eval_fn(plan, style, |u, v| {
                expr.eval1(sample_general_scaled(src, u, v, scale_x, scale_y, other))
            })
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

    let sample_pair = |u: f32, v: f32| -> (f32, f32, f32) {
        let (vv, vh) = match (style.resample, clean) {
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
                sample_general_scaled(&srcs[0], u, v, scale_x, scale_y, other),
                sample_general_scaled(&srcs[1], u, v, scale_x, scale_y, other),
            ),
        };
        let vv_db = fast_db_from_power(vv);
        let vh_db = fast_db_from_power(vh);
        (vv_db, vh_db, vv_db - vh_db)
    };

    match style.format {
        OutputFormat::RawF32 | OutputFormat::Npy => {
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
    if srcs[0].all_nodata || srcs[1].all_nodata {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = plan.all_in_bounds()
        && srcs[0].is_clean_inbounds(c0, c1, r0, r1)
        && srcs[1].is_clean_inbounds(c0, c1, r0, r1);

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
        (ResampleMode::Cubic, _) => render_eval_fn(plan, style, |u, v| {
            let a = sample_cubic(&srcs[0], u, v);
            let b = sample_cubic(&srcs[1], u, v);
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                expr.eval2(a, b)
            }
        }),
        (other, _) => {
            let (scale_x, scale_y) = coord_plan_scale(plan);
            render_eval_fn(plan, style, |u, v| {
                let a = sample_general_scaled(&srcs[0], u, v, scale_x, scale_y, other);
                let b = sample_general_scaled(&srcs[1], u, v, scale_x, scale_y, other);
                if a.is_nan() || b.is_nan() {
                    f32::NAN
                } else {
                    expr.eval2(a, b)
                }
            })
        }
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
    let clean = plan.all_in_bounds()
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
        (ResampleMode::Cubic, _) => render_eval_fn(plan, style, |u, v| {
            let a = sample_cubic(&srcs[0], u, v);
            let b = sample_cubic(&srcs[1], u, v);
            let c = sample_cubic(&srcs[2], u, v);
            if a.is_nan() || b.is_nan() || c.is_nan() {
                f32::NAN
            } else {
                expr.eval3(a, b, c)
            }
        }),
        (other, _) => {
            let (scale_x, scale_y) = coord_plan_scale(plan);
            render_eval_fn(plan, style, |u, v| {
                let a = sample_general_scaled(&srcs[0], u, v, scale_x, scale_y, other);
                let b = sample_general_scaled(&srcs[1], u, v, scale_x, scale_y, other);
                let c = sample_general_scaled(&srcs[2], u, v, scale_x, scale_y, other);
                if a.is_nan() || b.is_nan() || c.is_nan() {
                    f32::NAN
                } else {
                    expr.eval3(a, b, c)
                }
            })
        }
    }
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
    let max_u = ifd_dims.0 as f32 - 0.5;
    let max_v = ifd_dims.1 as f32 - 0.5;
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let all_clean =
        plan.all_in_bounds() && srcs[..nb].iter().all(|s| s.is_clean_inbounds(c0, c1, r0, r1));
    let range = safe_rescale_denom(style.rescale_min, style.rescale_max);
    let inv_range = 255.0 / range;
    let inv_range_254 = 254.0 / range;
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

    let sample_row = |py: usize, band_rows: &mut [[f32; N]]| {
        match (style.resample, all_clean) {
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
                            sample_general_scaled(&srcs[b], u, v, scale_x, scale_y, other);
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
                if !u.is_finite()
                    || !v.is_finite()
                    || u < -0.5
                    || u > max_u
                    || v < -0.5
                    || v > max_v
                {
                    out_row[px] = f32::NAN;
                }
            });
        }
    };

    match style.format {
        OutputFormat::RawF32 | OutputFormat::Npy => {
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
                let plte = build_plte_nodata(lut);
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
                                let t = ((val - min_val) * inv_range_254).clamp(0.0, 254.0);
                                1 + (t.round() as u8)
                            };
                        }
                        filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                    }
                    encode_indexed_png(
                        N as u32,
                        N as u32,
                        &plte,
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

/// Standard RGB terrain elevation encodings for 3D web map engines (MapLibre, Mapbox, Deck.gl, Cesium).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainEncoding {
    /// Mapzen / AWS Terrarium: `elev = (R * 256 + G + B / 256) - 32768` (~3.9 mm precision).
    Terrarium,
    /// Mapbox Terrain-RGB: `elev = -10000 + ((R * 65536 + G * 256 + B) * 0.1)` (0.1 m precision).
    TerrainRgb,
    /// TiTiler DEM Contours: `linear_rescale(elev, (minz, maxz), (1, 255))` -> `terrain` colormap,
    /// with black (`[0, 0, 0]`) contour lines where `elev.rem_euclid(increment) < thickness`.
    Contours {
        increment: i32,
        thickness: u8,
        minz: i32,
        maxz: i32,
    },
}

impl TerrainEncoding {
    #[inline(always)]
    pub fn encode_rgb(self, elev_m: f32) -> [u8; 3] {
        match self {
            TerrainEncoding::Terrarium => encode_terrarium_rgb(elev_m),
            TerrainEncoding::TerrainRgb => encode_terrain_rgb(elev_m),
            TerrainEncoding::Contours {
                increment,
                thickness,
                minz,
                maxz,
            } => encode_contours_rgb(elev_m, increment, thickness, minz, maxz),
        }
    }
}

#[inline(always)]
pub fn encode_contours_rgb(
    elev_m: f32,
    increment: i32,
    thickness: u8,
    minz: i32,
    maxz: i32,
) -> [u8; 3] {
    let data = elev_m as f64;
    if increment > 0 && thickness > 0 && data.rem_euclid(increment as f64) < thickness as f64 {
        return [0, 0, 0];
    }
    let lo = minz as f64;
    let hi = maxz as f64;
    let denom = if (hi - lo).abs() < 1e-12 { 1.0 } else { hi - lo };
    let idx = (((data - lo) / denom) * 254.0 + 1.0).clamp(1.0, 255.0) as u8;
    let lut = crate::colormap::ColormapName::Terrain.lut();
    lut.rgb[idx as usize]
}

#[inline(always)]
pub fn encode_terrarium_rgb(elev_m: f32) -> [u8; 3] {
    let v = (elev_m as f64 + 32768.0).clamp(0.0, 65535.99609375);
    let r = (v / 256.0).floor() as u8;
    let g = (v.floor() as u32 & 0xFF) as u8;
    let b = ((v - v.floor()) * 256.0).floor().clamp(0.0, 255.0) as u8;
    [r, g, b]
}

#[inline(always)]
pub fn encode_terrain_rgb(elev_m: f32) -> [u8; 3] {
    let enc = ((elev_m as f64 + 10000.0) * 10.0)
        .round()
        .clamp(0.0, 16_777_215.0) as u32;
    [
        ((enc >> 16) & 0xFF) as u8,
        ((enc >> 8) & 0xFF) as u8,
        (enc & 0xFF) as u8,
    ]
}

/// Render a single-band DEM directly into a Terrarium or Mapbox Terrain-RGB tile in one fused pass.
pub fn render_terrain_tile<const N: usize>(
    plan: &CoordPlan<N>,
    src: &BandSource<'_>,
    encoding: TerrainEncoding,
    resample: ResampleMode,
    format: OutputFormat,
) -> Bytes {
    if src.all_nodata {
        return empty_tile_bytes(N as u32, format, 3);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = plan.all_in_bounds() && src.is_clean_inbounds(c0, c1, r0, r1);

    let encode_px = |v: f32| -> [u8; 3] { encoding.encode_rgb(v) };
    let (scale_x, scale_y) = coord_plan_scale(plan);

    let sample_px = |u: f32, v: f32| -> f32 {
        match (resample, clean) {
            (ResampleMode::Bilinear, true) => sample_bilinear::<true>(src, u, v),
            (ResampleMode::Bilinear, false) => sample_bilinear::<false>(src, u, v),
            (ResampleMode::Nearest, true) => sample_nearest::<true>(src, u, v),
            (ResampleMode::Nearest, false) => sample_nearest::<false>(src, u, v),
            (ResampleMode::Cubic, _) => sample_cubic(src, u, v),
            (other, _) => sample_general_scaled(src, u, v, scale_x, scale_y, other),
        }
    };

    match format {
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
                let flat_dem = if resample == ResampleMode::Bilinear {
                    src.single_block_flat()
                } else {
                    None
                };
                if sc.f32_row.len() < N {
                    sc.f32_row.resize(N, 0.0);
                }
                for py in 0..N {
                    let row_rgb = &mut sc.row_buf[..N * 3];
                    if let (CoordPlan::Separable1D { u, v, .. }, Some((slice, bc, br, stride))) =
                        (plan, flat_dem)
                    {
                        let (y0_raw, fy) = fast_floor_i32(v[py] - br as f32);
                        let max_y = ((slice.len() / stride) as i32 - 2).max(0);
                        let y0 = y0_raw.clamp(0, max_y) as usize;
                        let r0_off = y0 * stride;
                        let r1_off = r0_off + stride;
                        let frow = &mut sc.f32_row[..N];
                        bilinear_row_flat_dispatch::<N>(
                            u,
                            bc,
                            &slice[r0_off..r0_off + stride],
                            &slice[r1_off..r1_off + stride],
                            fy,
                            frow,
                        );
                        for px in 0..N {
                            let [r, g, b] = encode_px(frow[px]);
                            let off = px * 3;
                            row_rgb[off] = r;
                            row_rgb[off + 1] = g;
                            row_rgb[off + 2] = b;
                        }
                    } else {
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let [r, g, b] = encode_px(sample_px(u, v));
                            let off = px * 3;
                            row_rgb[off] = r;
                            row_rgb[off + 1] = g;
                            row_rgb[off + 2] = b;
                        });
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
                        let elev = sample_px(u, v);
                        let off = px * 4;
                        if elev.is_nan() {
                            row_rgba[off] = 0;
                            row_rgba[off + 1] = 0;
                            row_rgba[off + 2] = 0;
                            row_rgba[off + 3] = 0;
                        } else {
                            let [r, g, b] = encode_px(elev);
                            row_rgba[off] = r;
                            row_rgba[off + 1] = g;
                            row_rgba[off + 2] = b;
                            row_rgba[off + 3] = 255;
                        }
                    });
                    filter_sub_rgba_row(row_rgba, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                }
                encode_filtered_png(N as u32, N as u32, 6, &sc.img_buf[..total])
            }
        }),
        OutputFormat::RawF32 | OutputFormat::Npy => {
            let plane_bytes = N * N * 4;
            let mut out = vec![0u8; plane_bytes * 3];
            let (r_plane, rest) = out.split_at_mut(plane_bytes);
            let (g_plane, b_plane) = rest.split_at_mut(plane_bytes);
            for py in 0..N {
                let row_off = py * N;
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    let elev = sample_px(u, v);
                    let off = (row_off + px) * 4;
                    if elev.is_nan() {
                        let nan_b = f32::NAN.to_le_bytes();
                        r_plane[off..off + 4].copy_from_slice(&nan_b);
                        g_plane[off..off + 4].copy_from_slice(&nan_b);
                        b_plane[off..off + 4].copy_from_slice(&nan_b);
                    } else {
                        let [r, g, b] = encode_px(elev);
                        r_plane[off..off + 4].copy_from_slice(&(r as f32).to_le_bytes());
                        g_plane[off..off + 4].copy_from_slice(&(g as f32).to_le_bytes());
                        b_plane[off..off + 4].copy_from_slice(&(b as f32).to_le_bytes());
                    }
                });
            }
            Bytes::from(out)
        }
        OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } if clean => {
            SCRATCH.with(|cell| {
                let mut guard = cell.borrow_mut();
                let sc = &mut *guard;
                let total = N * N * 3;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                let rgb = &mut sc.img_buf[..total];
                for py in 0..N {
                    let row_off = py * N * 3;
                    for_each_pixel_in_row(plan, py, |px, u, v| {
                        let [r, g, b] = encode_px(sample_px(u, v));
                        let off = row_off + px * 3;
                        rgb[off] = r;
                        rgb[off + 1] = g;
                        rgb[off + 2] = b;
                    });
                }
                match format {
                    OutputFormat::Webp { quality } => {
                        encode_webp_rgb(N as u32, N as u32, rgb, quality)
                    }
                    OutputFormat::Jpeg { quality } => {
                        encode_jpeg_rgb(N as u32, N as u32, rgb, quality)
                    }
                    _ => unreachable!(),
                }
            })
        }
        OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
            SCRATCH.with(|cell| {
                let mut guard = cell.borrow_mut();
                let sc = &mut *guard;
                let total = N * N * 4;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                let rgba = &mut sc.img_buf[..total];
                if clean {
                    for py in 0..N {
                        let row_off = py * N * 4;
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let [r, g, b] = encode_px(sample_px(u, v));
                            let off = row_off + px * 4;
                            rgba[off] = r;
                            rgba[off + 1] = g;
                            rgba[off + 2] = b;
                            rgba[off + 3] = 255;
                        });
                    }
                } else {
                    for py in 0..N {
                        let row_off = py * N * 4;
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let elev = sample_px(u, v);
                            let off = row_off + px * 4;
                            if elev.is_nan() {
                                rgba[off] = 0;
                                rgba[off + 1] = 0;
                                rgba[off + 2] = 0;
                                rgba[off + 3] = 0;
                            } else {
                                let [r, g, b] = encode_px(elev);
                                rgba[off] = r;
                                rgba[off + 1] = g;
                                rgba[off + 2] = b;
                                rgba[off + 3] = 255;
                            }
                        });
                    }
                }
                encode_rgba_tile(rgba, N as u32, clean, format)
            })
        }
    }
}

/// On-the-fly 3×3 Horn DEM derivative algorithms (matching `gdaldem`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DemAlgorithm {
    /// Horn's 3×3 shaded relief in `[0.0, 255.0]`.
    Hillshade {
        azimuth_deg: f32,
        altitude_deg: f32,
        z_factor: f32,
    },
    /// Terrain slope in degrees `[0.0, 90.0]`.
    Slope { z_factor: f32 },
    /// Compass aspect in degrees `[0.0, 360.0)` clockwise from North (`0=N, 90=E, 180=S, 270=W`).
    Aspect,
    /// Elevation contour lines at multiples of `increment` overlaid on `[minz, maxz]`.
    Contours {
        increment: f32,
        thickness: u32,
        minz: f32,
        maxz: f32,
    },
}

/// Rescale denominator preserving sign for inverted stretches (`?rescale=255,0`) and negative
/// GDAL `scale` (`?unscale=true`), while avoiding division by zero on degenerate `min == max`.
#[inline(always)]
pub fn safe_rescale_denom(min: f32, max: f32) -> f32 {
    let d = max - min;
    if d.abs() < 1e-12 { 1e-12 } else { d }
}

/// Evaluate a 3×3 Horn stencil centered at `(z_nw..z_se)` with isotropic ground spacing `1/(8*dx)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn eval_horn_3x3(
    z_nw: f32,
    z_n: f32,
    z_ne: f32,
    z_w: f32,
    z_c: f32,
    z_e: f32,
    z_sw: f32,
    z_s: f32,
    z_se: f32,
    inv_8dx: f32,
    alg: DemAlgorithm,
) -> f32 {
    eval_horn_3x3_aniso(
        z_nw, z_n, z_ne, z_w, z_c, z_e, z_sw, z_s, z_se, inv_8dx, inv_8dx, alg,
    )
}

/// Evaluate a 3×3 Horn stencil with separate East-West (`inv_8dx`) and North-South (`inv_8dy`)
/// inverse ground spacings (required for WGS84 / non-square bboxes where `dx_m != dy_m`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn eval_horn_3x3_aniso(
    z_nw: f32,
    z_n: f32,
    z_ne: f32,
    z_w: f32,
    z_c: f32,
    z_e: f32,
    z_sw: f32,
    z_s: f32,
    z_se: f32,
    inv_8dx: f32,
    inv_8dy: f32,
    alg: DemAlgorithm,
) -> f32 {
    if z_c.is_nan() {
        return f32::NAN;
    }
    // Fill any missing border neighbor with center elevation so coastal edges stay clean.
    let nw = if z_nw.is_nan() { z_c } else { z_nw };
    let n = if z_n.is_nan() { z_c } else { z_n };
    let ne = if z_ne.is_nan() { z_c } else { z_ne };
    let w = if z_w.is_nan() { z_c } else { z_w };
    let e = if z_e.is_nan() { z_c } else { z_e };
    let sw = if z_sw.is_nan() { z_c } else { z_sw };
    let s = if z_s.is_nan() { z_c } else { z_s };
    let se = if z_se.is_nan() { z_c } else { z_se };

    // dz/dx (positive going East), dz/dy (positive going South)
    let dz_dx = ((ne + 2.0 * e + se) - (nw + 2.0 * w + sw)) * inv_8dx;
    let dz_dy = ((sw + 2.0 * s + se) - (nw + 2.0 * n + ne)) * inv_8dy;

    match alg {
        DemAlgorithm::Hillshade {
            azimuth_deg,
            altitude_deg,
            z_factor,
        } => {
            let zenith = (90.0 - altitude_deg).to_radians();
            // Sun vector in East/North/Up coordinates:
            // azimuth_deg is clockwise from North (0=N, 90=E, 315=NW).
            let az = azimuth_deg.to_radians();
            let sun_e = az.sin() * zenith.sin();
            let sun_n = az.cos() * zenith.sin();
            let sun_u = zenith.cos();
            // Surface normal in (East, North, Up) = (-z*dz_dx, +z*dz_dy, 1) normalized
            let nx = -z_factor * dz_dx;
            let ny = z_factor * dz_dy;
            let inv_len = 1.0 / (nx * nx + ny * ny + 1.0).sqrt();
            let dot = (nx * sun_e + ny * sun_n + sun_u) * inv_len;
            (dot.max(0.0) * 255.0).clamp(0.0, 255.0)
        }
        DemAlgorithm::Slope { z_factor } => {
            let grad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt() * z_factor;
            grad.atan().to_degrees()
        }
        DemAlgorithm::Aspect => {
            if dz_dx.abs() < 1e-12 && dz_dy.abs() < 1e-12 {
                return f32::NAN;
            }
            // Downhill vector in (East, North) is (-dz_dx, +dz_dy)
            let bearing = (-dz_dx).atan2(dz_dy).to_degrees();
            bearing.rem_euclid(360.0)
        }
        DemAlgorithm::Contours {
            increment,
            thickness,
            minz,
            maxz,
        } => {
            let inc = increment.max(1e-4);
            let in_range = z_c >= minz && z_c <= maxz;
            let level = (z_c / inc).floor();
            let is_line = if in_range {
                if thickness >= 2 {
                    (n / inc).floor() != level
                        || (s / inc).floor() != level
                        || (e / inc).floor() != level
                        || (w / inc).floor() != level
                } else {
                    (n / inc).floor() != level || (e / inc).floor() != level
                }
            } else {
                false
            };
            if is_line {
                0.0
            } else {
                (((z_c - minz) / safe_rescale_denom(minz, maxz)) * 255.0).clamp(1.0, 255.0)
            }
        }
    }
}

/// SIMD Horn 3×3 hillshade row evaluator for clean (`no_nodata`) DEM tiles:
/// processes 4 pixels/iter on ARM64 NEON or 8 pixels/iter on x86-64 AVX2 using pure contiguous
/// vector loads from `row0`, `row1`, `row2` (`N + 2` floats each) and writes rescaled `u8` indices.
#[inline(always)]
fn eval_horn_hillshade_row_clean<const N: usize>(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    inv_8dx: f32,
    azimuth_deg: f32,
    altitude_deg: f32,
    z_factor: f32,
    inv_range: f32,
    min_val: f32,
    out_idx: &mut [u8],
) {
    let zenith = (90.0 - altitude_deg).to_radians();
    let az = azimuth_deg.to_radians();
    let sun_e = az.sin() * zenith.sin();
    let sun_n = az.cos() * zenith.sin();
    let sun_u = zenith.cos();
    let neg_z_scale = -z_factor * inv_8dx;
    let pos_z_scale = z_factor * inv_8dx;
    #[allow(unused_assignments, unused_mut)]
    let mut start_px = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let v_two = vdupq_n_f32(2.0);
        let v_one = vdupq_n_f32(1.0);
        let v_zero = vdupq_n_f32(0.0);
        let v_255 = vdupq_n_f32(255.0);
        let v_half = vdupq_n_f32(0.5);
        let v_neg_zs = vdupq_n_f32(neg_z_scale);
        let v_pos_zs = vdupq_n_f32(pos_z_scale);
        let v_se = vdupq_n_f32(sun_e);
        let v_sn = vdupq_n_f32(sun_n);
        let v_su = vdupq_n_f32(sun_u);
        let v_inv_r = vdupq_n_f32(inv_range);
        let v_min = vdupq_n_f32(min_val);

        let r0 = row0.as_ptr();
        let r1 = row1.as_ptr();
        let r2 = row2.as_ptr();
        let mut tmp = [0u32; 4];
        let mut px = 0usize;
        while px + 4 <= N {
            let nw = vld1q_f32(r0.add(px));
            let n = vld1q_f32(r0.add(px + 1));
            let ne = vld1q_f32(r0.add(px + 2));
            let w = vld1q_f32(r1.add(px));
            let e = vld1q_f32(r1.add(px + 2));
            let sw = vld1q_f32(r2.add(px));
            let s = vld1q_f32(r2.add(px + 1));
            let se = vld1q_f32(r2.add(px + 2));

            let east = vaddq_f32(vfmaq_f32(ne, v_two, e), se);
            let west = vaddq_f32(vfmaq_f32(nw, v_two, w), sw);
            let nx = vmulq_f32(vsubq_f32(east, west), v_neg_zs);

            let south = vaddq_f32(vfmaq_f32(sw, v_two, s), se);
            let north = vaddq_f32(vfmaq_f32(nw, v_two, n), ne);
            let ny = vmulq_f32(vsubq_f32(south, north), v_pos_zs);

            let len_sq = vfmaq_f32(vfmaq_f32(v_one, nx, nx), ny, ny);
            let num = vfmaq_f32(vfmaq_f32(v_su, nx, v_se), ny, v_sn);
            let dot = vdivq_f32(num, vsqrtq_f32(len_sq));
            let hs = vminq_f32(vmulq_f32(vmaxq_f32(dot, v_zero), v_255), v_255);
            let scaled = vminq_f32(vmaxq_f32(vmulq_f32(vsubq_f32(hs, v_min), v_inv_r), v_zero), v_255);
            vst1q_u32(tmp.as_mut_ptr(), vcvtq_u32_f32(vaddq_f32(scaled, v_half)));
            *out_idx.get_unchecked_mut(px) = tmp[0] as u8;
            *out_idx.get_unchecked_mut(px + 1) = tmp[1] as u8;
            *out_idx.get_unchecked_mut(px + 2) = tmp[2] as u8;
            *out_idx.get_unchecked_mut(px + 3) = tmp[3] as u8;
            px += 4;
        }
        start_px = px;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                eval_horn_hillshade_row_clean_avx2::<N>(
                    row0.as_ptr(),
                    row1.as_ptr(),
                    row2.as_ptr(),
                    neg_z_scale,
                    pos_z_scale,
                    sun_e,
                    sun_n,
                    sun_u,
                    inv_range,
                    min_val,
                    out_idx,
                );
            }
            start_px = (N / 8) * 8;
        }
    }

    for px in start_px..N {
        let c = px + 1;
        let val = eval_horn_3x3(
            row0[c - 1], row0[c], row0[c + 1],
            row1[c - 1], row1[c], row1[c + 1],
            row2[c - 1], row2[c], row2[c + 1],
            inv_8dx,
            DemAlgorithm::Hillshade { azimuth_deg, altitude_deg, z_factor },
        );
        out_idx[px] = rescale_u8(val, inv_range, min_val);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn eval_horn_hillshade_row_clean_avx2<const N: usize>(
    r0: *const f32,
    r1: *const f32,
    r2: *const f32,
    neg_z_scale: f32,
    pos_z_scale: f32,
    sun_e: f32,
    sun_n: f32,
    sun_u: f32,
    inv_range: f32,
    min_val: f32,
    out_idx: &mut [u8],
) {
    use std::arch::x86_64::*;
    let v_two = _mm256_set1_ps(2.0);
    let v_one = _mm256_set1_ps(1.0);
    let v_zero = _mm256_setzero_ps();
    let v_255 = _mm256_set1_ps(255.0);
    let v_half = _mm256_set1_ps(0.5);
    let v_neg_zs = _mm256_set1_ps(neg_z_scale);
    let v_pos_zs = _mm256_set1_ps(pos_z_scale);
    let v_se = _mm256_set1_ps(sun_e);
    let v_sn = _mm256_set1_ps(sun_n);
    let v_su = _mm256_set1_ps(sun_u);
    let v_inv_r = _mm256_set1_ps(inv_range);
    let v_min = _mm256_set1_ps(min_val);

    let mut tmp = [0i32; 8];
    let mut px = 0usize;
    while px + 8 <= N {
        let nw = _mm256_loadu_ps(r0.add(px));
        let n = _mm256_loadu_ps(r0.add(px + 1));
        let ne = _mm256_loadu_ps(r0.add(px + 2));
        let w = _mm256_loadu_ps(r1.add(px));
        let e = _mm256_loadu_ps(r1.add(px + 2));
        let sw = _mm256_loadu_ps(r2.add(px));
        let s = _mm256_loadu_ps(r2.add(px + 1));
        let se = _mm256_loadu_ps(r2.add(px + 2));

        let east = _mm256_add_ps(_mm256_fmadd_ps(v_two, e, ne), se);
        let west = _mm256_add_ps(_mm256_fmadd_ps(v_two, w, nw), sw);
        let nx = _mm256_mul_ps(_mm256_sub_ps(east, west), v_neg_zs);

        let south = _mm256_add_ps(_mm256_fmadd_ps(v_two, s, sw), se);
        let north = _mm256_add_ps(_mm256_fmadd_ps(v_two, n, nw), ne);
        let ny = _mm256_mul_ps(_mm256_sub_ps(south, north), v_pos_zs);

        let len_sq = _mm256_fmadd_ps(ny, ny, _mm256_fmadd_ps(nx, nx, v_one));
        let num = _mm256_fmadd_ps(ny, v_sn, _mm256_fmadd_ps(nx, v_se, v_su));
        let dot = _mm256_div_ps(num, _mm256_sqrt_ps(len_sq));
        let hs = _mm256_min_ps(_mm256_mul_ps(_mm256_max_ps(dot, v_zero), v_255), v_255);
        let scaled = _mm256_min_ps(
            _mm256_max_ps(_mm256_mul_ps(_mm256_sub_ps(hs, v_min), v_inv_r), v_zero),
            v_255,
        );
        _mm256_storeu_si256(
            tmp.as_mut_ptr() as *mut __m256i,
            _mm256_cvttps_epi32(_mm256_add_ps(scaled, v_half)),
        );
        for k in 0..8 {
            *out_idx.get_unchecked_mut(px + k) = tmp[k] as u8;
        }
        px += 8;
    }
}

/// Render an on-the-fly DEM derivative tile (Hillshade, Slope, or Aspect) using Horn's 3×3
/// stencil over a 1-pixel-padded resampled elevation buffer for seamless tile boundaries.
pub fn render_dem_algorithm<const N: usize>(
    plan: &CoordPlan<N>,
    src: &BandSource<'_>,
    alg: DemAlgorithm,
    pixel_size_m: f32,
    style: &SingleBandStyle,
) -> Bytes {
    if src.all_nodata {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = plan.all_in_bounds() && src.is_clean_inbounds(c0, c1, r0, r1);
    let (scale_x, scale_y) = coord_plan_scale(plan);

    let sample_px = |u: f32, v: f32| -> f32 {
        match (style.resample, clean) {
            (ResampleMode::Bilinear, true) => sample_bilinear::<true>(src, u, v),
            (ResampleMode::Bilinear, false) => sample_bilinear::<false>(src, u, v),
            (ResampleMode::Nearest, true) => sample_nearest::<true>(src, u, v),
            (ResampleMode::Nearest, false) => sample_nearest::<false>(src, u, v),
            (ResampleMode::Cubic, _) => sample_cubic(src, u, v),
            (other, _) => sample_general_scaled(src, u, v, scale_x, scale_y, other),
        }
    };
    // Sub-pixel bounds-checked sampler for the 1-pixel extrapolated border so tile edges
    // match neighboring tiles' interior pixels with zero staircase seam.
    let sample_border = |u: f32, v: f32| -> f32 {
        match style.resample {
            ResampleMode::Bilinear => sample_bilinear::<false>(src, u, v),
            ResampleMode::Nearest => sample_nearest::<false>(src, u, v),
            ResampleMode::Cubic => sample_cubic(src, u, v),
            other => sample_general_scaled(src, u, v, scale_x, scale_y, other),
        }
    };

    // Reuse thread-local scratch for the (N+2)×(N+2) elevation buffer + PNG/RGBA output buffers
    // (eliminates up to 1.05 MiB heap alloc+zero per @2x DEM tile).
    let pad = N + 2;
    SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let sc = &mut *guard;
        if sc.dem_elev.len() < pad * pad {
            sc.dem_elev.resize(pad * pad, 0.0);
        }
        let elev = &mut sc.dem_elev[..pad * pad];
        let mut u_first = [0.0f32; N];
        let mut v_first = [0.0f32; N];
        let mut u_second = [0.0f32; N];
        let mut v_second = [0.0f32; N];
        let mut u_prev = [0.0f32; N];
        let mut v_prev = [0.0f32; N];
        let mut u_row = [0.0f32; N];
        let mut v_row = [0.0f32; N];

        let flat_dem = if style.resample == ResampleMode::Bilinear && clean {
            src.single_block_flat()
        } else {
            None
        };

        for py in 0..N {
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            if py == 0 {
                u_first = u_row;
                v_first = v_row;
            } else if py == 1 {
                u_second = u_row;
                v_second = v_row;
            }
            let row_off = (py + 1) * pad;
            if let (CoordPlan::Separable1D { u, v, .. }, Some((slice, bc, br, stride))) =
                (plan, flat_dem)
            {
                let (y0_raw, fy) = fast_floor_i32(v[py] - br as f32);
                let max_y = ((slice.len() / stride) as i32 - 2).max(0);
                let y0 = y0_raw.clamp(0, max_y) as usize;
                let r0_off = y0 * stride;
                let r1_off = r0_off + stride;
                bilinear_row_flat_dispatch::<N>(
                    u,
                    bc,
                    &slice[r0_off..r0_off + stride],
                    &slice[r1_off..r1_off + stride],
                    fy,
                    &mut elev[row_off + 1..row_off + 1 + N],
                );
            } else {
                for px in 0..N {
                    elev[row_off + px + 1] = sample_px(u_row[px], v_row[px]);
                }
            }
            // Left/right 1-px extrapolated border using the exact same sub-pixel interpolation
            let du_l = u_row[1] - u_row[0];
            let dv_l = v_row[1] - v_row[0];
            elev[row_off] = sample_border(u_row[0] - du_l, v_row[0] - dv_l);
            let du_r = u_row[N - 1] - u_row[N - 2];
            let dv_r = v_row[N - 1] - v_row[N - 2];
            elev[row_off + N + 1] = sample_border(u_row[N - 1] + du_r, v_row[N - 1] + dv_r);
            if py + 1 < N {
                u_prev = u_row;
                v_prev = v_row;
            }
        }
        // Top/bottom 1-px extrapolated border using the exact same sub-pixel interpolation
        for px in 0..N {
            let u_top = 2.0 * u_first[px] - u_second[px];
            let v_top = 2.0 * v_first[px] - v_second[px];
            elev[px + 1] = sample_border(u_top, v_top);
            let u_bot = 2.0 * u_row[px] - u_prev[px];
            let v_bot = 2.0 * v_row[px] - v_prev[px];
            elev[(N + 1) * pad + px + 1] = sample_border(u_bot, v_bot);
        }
        // 4 extrapolated corners for seamless corner stencils
        elev[0] = sample_border(
            (2.0 * u_first[0] - u_second[0]) - (u_first[1] - u_first[0]),
            (2.0 * v_first[0] - v_second[0]) - (v_first[1] - v_first[0]),
        );
        elev[pad - 1] = sample_border(
            (2.0 * u_first[N - 1] - u_second[N - 1]) + (u_first[N - 1] - u_first[N - 2]),
            (2.0 * v_first[N - 1] - v_second[N - 1]) + (v_first[N - 1] - v_first[N - 2]),
        );
        elev[(N + 1) * pad] = sample_border(
            (2.0 * u_row[0] - u_prev[0]) - (u_row[1] - u_row[0]),
            (2.0 * v_row[0] - v_prev[0]) - (v_row[1] - v_row[0]),
        );
        elev[(N + 1) * pad + pad - 1] = sample_border(
            (2.0 * u_row[N - 1] - u_prev[N - 1]) + (u_row[N - 1] - u_row[N - 2]),
            (2.0 * v_row[N - 1] - v_prev[N - 1]) + (v_row[N - 1] - v_row[N - 2]),
        );

        // When interior pixels are all valid (`clean == true`) but the tile abuts a DEM raster
        // edge, `sample_border` writes `f32::NAN` into the 1-px extrapolated stencil ring.
        // Clamp any NaN border cell to its adjacent interior pixel (matching `eval_horn_3x3`'s
        // edge-fill) so the branchless SIMD hillshade path never turns raster-edge borders black.
        if clean {
            for py in 0..N {
                let row_off = (py + 1) * pad;
                if elev[row_off].is_nan() {
                    elev[row_off] = elev[row_off + 1];
                }
                if elev[row_off + N + 1].is_nan() {
                    elev[row_off + N + 1] = elev[row_off + N];
                }
            }
            for px in 0..N {
                let c = px + 1;
                if elev[c].is_nan() {
                    elev[c] = elev[pad + c];
                }
                if elev[(N + 1) * pad + c].is_nan() {
                    elev[(N + 1) * pad + c] = elev[N * pad + c];
                }
            }
            if elev[0].is_nan() {
                elev[0] = elev[pad + 1];
            }
            if elev[pad - 1].is_nan() {
                elev[pad - 1] = elev[pad + N];
            }
            if elev[(N + 1) * pad].is_nan() {
                elev[(N + 1) * pad] = elev[N * pad + 1];
            }
            if elev[(N + 1) * pad + pad - 1].is_nan() {
                elev[(N + 1) * pad + pad - 1] = elev[N * pad + N];
            }
        }

        let inv_8dx = 1.0 / (8.0 * pixel_size_m.max(1e-3));
        let inv_range = 255.0 / safe_rescale_denom(style.rescale_min, style.rescale_max);
        let min_val = style.rescale_min;
        let lut = style.colormap.lut();

        match style.format {
            OutputFormat::RawF32 | OutputFormat::Npy => {
                let mut out = BytesMut::with_capacity(N * N * 4);
                for py in 0..N {
                    let r0 = py * pad;
                    let r1 = (py + 1) * pad;
                    let r2 = (py + 2) * pad;
                    for px in 0..N {
                        let c = px + 1;
                        let val = eval_horn_3x3(
                            elev[r0 + c - 1],
                            elev[r0 + c],
                            elev[r0 + c + 1],
                            elev[r1 + c - 1],
                            elev[r1 + c],
                            elev[r1 + c + 1],
                            elev[r2 + c - 1],
                            elev[r2 + c],
                            elev[r2 + c + 1],
                            inv_8dx,
                            alg,
                        );
                        out.put_slice(&val.to_le_bytes());
                    }
                }
                out.freeze()
            }
            OutputFormat::Png => {
                let stride = 1 + N;
                let total = N * stride;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                if sc.row_buf.len() < N {
                    sc.row_buf.resize(N, 0);
                }
                let mut has_nodata = false;
                if clean {
                    if let DemAlgorithm::Hillshade {
                        azimuth_deg,
                        altitude_deg,
                        z_factor,
                    } = alg
                    {
                        for py in 0..N {
                            let r0 = py * pad;
                            let r1 = (py + 1) * pad;
                            let r2 = (py + 2) * pad;
                            let row_idx = &mut sc.row_buf[..N];
                            eval_horn_hillshade_row_clean::<N>(
                                &elev[r0..r0 + pad],
                                &elev[r1..r1 + pad],
                                &elev[r2..r2 + pad],
                                inv_8dx,
                                azimuth_deg,
                                altitude_deg,
                                z_factor,
                                inv_range,
                                min_val,
                                row_idx,
                            );
                            filter_sub_u8_row(
                                row_idx,
                                &mut sc.img_buf[py * stride..(py + 1) * stride],
                            );
                        }
                        if matches!(style.colormap, crate::colormap::ColormapName::Grayscale) {
                            return encode_filtered_png(N as u32, N as u32, 0, &sc.img_buf[..total]);
                        }
                        let plte = build_plte_opaque(lut);
                        return encode_indexed_png(
                            N as u32,
                            N as u32,
                            &plte,
                            None,
                            &sc.img_buf[..total],
                        );
                    }
                }
                let is_opaque_dem = clean && !matches!(alg, DemAlgorithm::Aspect);
                let inv_range_254 = inv_range * (254.0 / 255.0);
                for py in 0..N {
                    let r0 = py * pad;
                    let r1 = (py + 1) * pad;
                    let r2 = (py + 2) * pad;
                    let row_idx = &mut sc.row_buf[..N];
                    for px in 0..N {
                        let c = px + 1;
                        let val = eval_horn_3x3(
                            elev[r0 + c - 1],
                            elev[r0 + c],
                            elev[r0 + c + 1],
                            elev[r1 + c - 1],
                            elev[r1 + c],
                            elev[r1 + c + 1],
                            elev[r2 + c - 1],
                            elev[r2 + c],
                            elev[r2 + c + 1],
                            inv_8dx,
                            alg,
                        );
                        if val.is_nan() {
                            has_nodata = true;
                            row_idx[px] = 0;
                        } else if is_opaque_dem {
                            row_idx[px] = rescale_u8(val, inv_range, min_val);
                        } else {
                            let t = ((val - min_val) * inv_range_254).clamp(0.0, 254.0);
                            row_idx[px] = 1 + (t.round() as u8);
                        }
                    }
                    filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                }
                if !has_nodata && is_opaque_dem {
                    if matches!(style.colormap, crate::colormap::ColormapName::Grayscale) {
                        encode_filtered_png(N as u32, N as u32, 0, &sc.img_buf[..total])
                    } else {
                        let plte = build_plte_opaque(lut);
                        encode_indexed_png(N as u32, N as u32, &plte, None, &sc.img_buf[..total])
                    }
                } else {
                    let plte = build_plte_nodata(lut);
                    let trns = has_nodata.then_some(&[0u8][..]);
                    encode_indexed_png(N as u32, N as u32, &plte, trns, &sc.img_buf[..total])
                }
            }
            OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
                let total = N * N * 4;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                let rgba = &mut sc.img_buf[..total];
                rgba.fill(0);
                let mut all_opaque = true;
                for py in 0..N {
                    let r0 = py * pad;
                    let r1 = (py + 1) * pad;
                    let r2 = (py + 2) * pad;
                    let row_off = py * N * 4;
                    for px in 0..N {
                        let c = px + 1;
                        let val = eval_horn_3x3(
                            elev[r0 + c - 1],
                            elev[r0 + c],
                            elev[r0 + c + 1],
                            elev[r1 + c - 1],
                            elev[r1 + c],
                            elev[r1 + c + 1],
                            elev[r2 + c - 1],
                            elev[r2 + c],
                            elev[r2 + c + 1],
                            inv_8dx,
                            alg,
                        );
                        let off = row_off + px * 4;
                        if val.is_nan() {
                            all_opaque = false;
                        } else {
                            let idx = rescale_u8(val, inv_range, min_val) as usize;
                            let rgb = lut.rgb[idx];
                            rgba[off] = rgb[0];
                            rgba[off + 1] = rgb[1];
                            rgba[off + 2] = rgb[2];
                            rgba[off + 3] = 255;
                        }
                    }
                }
                encode_rgba_tile(rgba, N as u32, all_opaque, style.format)
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};

    fn make_plan_32() -> CoordPlan<32> {
        let mut u = [0.0f32; 32];
        let mut v = [0.0f32; 32];
        for i in 0..32 {
            u[i] = 1.25 + (i as f32) * 1.73;
            v[i] = 2.35 + (i as f32) * 1.61;
        }
        CoordPlan::Separable1D {
            u: Box::new(u),
            v: Box::new(v),
            src_min_col: 1,
            src_max_col: 60,
            src_min_row: 2,
            src_max_row: 60,
            all_in_bounds: true,
        }
    }

    #[test]
    fn fused_rgb_bilinear_matches_scalar_for_u8_u16_and_f32() {
        let plan = make_plan_32();
        let tw = 64u32;
        let th = 64u32;
        let len = (tw * th) as usize;
        let stats = BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 0.0,
            max: 255.0,
        };

        // 1. U8 single-block
        let b0_u8 = DecodedBand {
            width: tw,
            height: th,
            data: BandData::U8 {
                data: (0..len).map(|i| ((i * 7 + 3) % 256) as u8).collect(),
                nodata: None,
            },
            stats,
        };
        let b1_u8 = DecodedBand {
            width: tw,
            height: th,
            data: BandData::U8 {
                data: (0..len).map(|i| ((i * 11 + 17) % 256) as u8).collect(),
                nodata: None,
            },
            stats,
        };
        let b2_u8 = DecodedBand {
            width: tw,
            height: th,
            data: BandData::U8 {
                data: (0..len).map(|i| ((i * 13 + 29) % 256) as u8).collect(),
                nodata: None,
            },
            stats,
        };
        let blks0 = [Some(&b0_u8)];
        let blks1 = [Some(&b1_u8)];
        let blks2 = [Some(&b2_u8)];
        let srcs_u8 = [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks0),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks1),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks2),
        ];

        // 2. U16 single-block
        let b0_u16 = DecodedBand {
            width: tw,
            height: th,
            data: BandData::U16 {
                data: (0..len).map(|i| ((i * 37 + 100) % 20000) as u16).collect(),
                nodata: None,
            },
            stats,
        };
        let b1_u16 = DecodedBand {
            width: tw,
            height: th,
            data: BandData::U16 {
                data: (0..len).map(|i| ((i * 53 + 200) % 20000) as u16).collect(),
                nodata: None,
            },
            stats,
        };
        let b2_u16 = DecodedBand {
            width: tw,
            height: th,
            data: BandData::U16 {
                data: (0..len).map(|i| ((i * 71 + 300) % 20000) as u16).collect(),
                nodata: None,
            },
            stats,
        };
        let u0 = [Some(&b0_u16)];
        let u1 = [Some(&b1_u16)];
        let u2 = [Some(&b2_u16)];
        let srcs_u16 = [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &u0),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &u1),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &u2),
        ];

        // 3. F32 single-block
        let mut s0 = AlignedF32Slab::filled(len, 0.0);
        let mut s1 = AlignedF32Slab::filled(len, 0.0);
        let mut s2 = AlignedF32Slab::filled(len, 0.0);
        for i in 0..len {
            s0.as_mut_slice()[i] = ((i * 19) % 1000) as f32 * 0.001;
            s1.as_mut_slice()[i] = ((i * 23) % 1000) as f32 * 0.001;
            s2.as_mut_slice()[i] = ((i * 29) % 1000) as f32 * 0.001;
        }
        let b0_f32 = DecodedBand { width: tw, height: th, data: BandData::F32(s0), stats };
        let b1_f32 = DecodedBand { width: tw, height: th, data: BandData::F32(s1), stats };
        let b2_f32 = DecodedBand { width: tw, height: th, data: BandData::F32(s2), stats };
        let f0 = [Some(&b0_f32)];
        let f1 = [Some(&b1_f32)];
        let f2 = [Some(&b2_f32)];
        let srcs_f32 = [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &f0),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &f1),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &f2),
        ];

        for (name, srcs, mins, maxs) in [
            ("u8", &srcs_u8, [0.0f32; 3], [255.0f32; 3]),
            ("u16", &srcs_u16, [500.0; 3], [19500.0; 3]),
            ("f32", &srcs_f32, [0.0; 3], [1.0; 3]),
        ] {
            let inv = [
                255.0 / (maxs[0] - mins[0]),
                255.0 / (maxs[1] - mins[1]),
                255.0 / (maxs[2] - mins[2]),
            ];
            for py in 0..32 {
                let mut fused = [0u8; 32 * 3];
                let mut scalar = [0u8; 32 * 3];
                sample_rgb_bilinear_row_clean(&plan, py, srcs, inv, mins, &mut fused);
                for_each_pixel_in_row(&plan, py, |px, u, v| {
                    let off = px * 3;
                    scalar[off] = rescale_u8(sample_bilinear::<true>(&srcs[0], u, v), inv[0], mins[0]);
                    scalar[off + 1] = rescale_u8(sample_bilinear::<true>(&srcs[1], u, v), inv[1], mins[1]);
                    scalar[off + 2] = rescale_u8(sample_bilinear::<true>(&srcs[2], u, v), inv[2], mins[2]);
                });
                for i in 0..fused.len() {
                    let diff = (fused[i] as i32 - scalar[i] as i32).abs();
                    assert!(
                        diff <= 1,
                        "{name} row {py} idx {i}: fused={} scalar={}",
                        fused[i],
                        scalar[i]
                    );
                }
            }
        }
    }

    #[test]
    fn terrain_encodings_round_trip_accurately() {
        for &elev in &[-430.5f32, -10.25, 0.0, 123.45, 2450.7, 8848.86] {
            let [r, g, b] = encode_terrarium_rgb(elev);
            let dec_terr = (r as f64 * 256.0 + g as f64 + b as f64 / 256.0) - 32768.0;
            assert!(
                (dec_terr - elev as f64).abs() < 0.005,
                "terrarium mismatch for {elev}: got {dec_terr}"
            );

            let [mr, mg, mb] = encode_terrain_rgb(elev);
            let dec_mb = -10000.0 + ((mr as u32 * 65536 + mg as u32 * 256 + mb as u32) as f64 * 0.1);
            assert!(
                (dec_mb - elev as f64).abs() <= 0.055,
                "terrainrgb mismatch for {elev}: got {dec_mb}"
            );
        }
    }

    #[test]
    fn dem_horn_3x3_matches_analytical_flat_and_45deg_east_ramp() {
        let dx = 10.0f32;
        let inv_8dx = 1.0 / (8.0 * dx);

        // Flat plane z = 500m -> slope = 0 deg, hillshade(alt=45) = 255 * sin(45 deg) ~= 180.3
        let flat_slope = eval_horn_3x3(
            500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, inv_8dx,
            DemAlgorithm::Slope { z_factor: 1.0 },
        );
        assert!(flat_slope.abs() < 1e-4, "expected 0 deg slope, got {flat_slope}");

        let flat_hs = eval_horn_3x3(
            500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, inv_8dx,
            DemAlgorithm::Hillshade {
                azimuth_deg: 315.0,
                altitude_deg: 45.0,
                z_factor: 1.0,
            },
        );
        assert!(
            (flat_hs - 180.312).abs() < 0.1,
            "expected ~180.3 flat hillshade, got {flat_hs}"
        );

        // 45-degree east-facing ramp: z drops by dx (=10m) each step East -> dz/dx = -1, dz/dy = 0
        let ramp_slope = eval_horn_3x3(
            110.0, 100.0, 90.0, 110.0, 100.0, 90.0, 110.0, 100.0, 90.0, inv_8dx,
            DemAlgorithm::Slope { z_factor: 1.0 },
        );
        assert!(
            (ramp_slope - 45.0).abs() < 1e-3,
            "expected 45 deg slope, got {ramp_slope}"
        );

        let ramp_aspect = eval_horn_3x3(
            110.0, 100.0, 90.0, 110.0, 100.0, 90.0, 110.0, 100.0, 90.0, inv_8dx,
            DemAlgorithm::Aspect,
        );
        assert!(
            (ramp_aspect - 90.0).abs() < 1e-3,
            "expected 90 deg (East) aspect, got {ramp_aspect}"
        );
    }

    #[test]
    fn dem_adjacent_tiles_have_zero_boundary_seam() {
        // Smooth tilted plane z(c, r) = 1000 - 10*c - 5*r sampled at 4x zoom (step = 0.25 px)
        // across two adjacent 32x32 tiles. Every border pixel must produce the exact same slope
        // as the interior (no 1-pixel staircase spike along tile boundaries!).
        let tw = 64u32;
        let th = 64u32;
        let len = (tw * th) as usize;
        let mut slab = AlignedF32Slab::filled(len, 0.0);
        for r in 0..th {
            for c in 0..tw {
                slab.as_mut_slice()[(r * tw + c) as usize] =
                    1000.0 - 10.0 * (c as f32) - 5.0 * (r as f32);
            }
        }
        let band = DecodedBand {
            width: tw,
            height: th,
            data: BandData::F32(slab),
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 0.0,
                max: 1000.0,
            },
        };
        let blks = [Some(&band)];
        let src = BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks);

        let make_tile_plan = |u0: f32, v0: f32| -> CoordPlan<32> {
            let mut u = [0.0f32; 32];
            let mut v = [0.0f32; 32];
            for i in 0..32 {
                u[i] = u0 + (i as f32) * 0.25;
                v[i] = v0 + (i as f32) * 0.25;
            }
            CoordPlan::Separable1D {
                u: Box::new(u),
                v: Box::new(v),
                src_min_col: 2,
                src_max_col: 50,
                src_min_row: 2,
                src_max_row: 50,
                all_in_bounds: true,
            }
        };

        let left_plan = make_tile_plan(10.0, 10.0);
        let right_plan = make_tile_plan(18.0, 10.0);
        let style = SingleBandStyle {
            rescale_min: 0.0,
            rescale_max: 90.0,
            colormap: ColormapName::Viridis,
            resample: ResampleMode::Bilinear,
            format: OutputFormat::RawF32,
        };
        let left_bytes = render_dem_algorithm(
            &left_plan,
            &src,
            DemAlgorithm::Slope { z_factor: 1.0 },
            2.5,
            &style,
        );
        let right_bytes = render_dem_algorithm(
            &right_plan,
            &src,
            DemAlgorithm::Slope { z_factor: 1.0 },
            2.5,
            &style,
        );
        let read_f32 = |b: &[u8], py: usize, px: usize| -> f32 {
            let off = (py * 32 + px) * 4;
            f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
        };
        let expected = read_f32(&left_bytes, 16, 16);
        for py in 0..32 {
            let l_edge = read_f32(&left_bytes, py, 31);
            let r_edge = read_f32(&right_bytes, py, 0);
            let top_edge = read_f32(&left_bytes, 0, py);
            let bot_edge = read_f32(&left_bytes, 31, py);
            assert!((l_edge - expected).abs() < 1e-3, "left edge seam at py={py}: {l_edge} vs {expected}");
            assert!((r_edge - expected).abs() < 1e-3, "right edge seam at py={py}: {r_edge} vs {expected}");
            assert!((top_edge - expected).abs() < 1e-3, "top edge seam at px={py}: {top_edge} vs {expected}");
            assert!((bot_edge - expected).abs() < 1e-3, "bot edge seam at px={py}: {bot_edge} vs {expected}");
        }
    }

    #[test]
    fn simd_cubic_and_horn_hillshade_match_scalar() {
        let tw = 512u32;
        let th = 512u32;
        let len = (tw * th) as usize;
        let mut slab = AlignedF32Slab::filled(len, 0.0);
        for r in 0..th {
            for c in 0..tw {
                let x = c as f32 * 0.07;
                let y = r as f32 * 0.09;
                slab.as_mut_slice()[(r * tw + c) as usize] =
                    1500.0 + 400.0 * x.sin() * y.cos() + 120.0 * (x * 1.7 + y * 0.5).cos();
            }
        }
        let band = DecodedBand {
            width: tw,
            height: th,
            data: BandData::F32(slab),
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 900.0,
                max: 2100.0,
            },
        };
        let blks = [Some(&band)];
        let src = BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks);

        // 1. Verify SIMD sample_cubic against scalar reference
        for i in 0..64 {
            let u = 10.13 + (i as f32) * 3.71;
            let v = 12.29 + (i as f32) * 3.53;
            let fast = sample_cubic(&src, u, v);
            let (x0, fx) = fast_floor_i32(u);
            let (y0, fy) = fast_floor_i32(v);
            let wx = cubic_weights(fx);
            let wy = cubic_weights(fy);
            let mut ref_val = 0.0f32;
            for j in 0..4 {
                let mut r_acc = 0.0f32;
                for k in 0..4 {
                    r_acc += src.sample_checked(x0 + k - 1, y0 + j - 1) * wx[k as usize];
                }
                ref_val += r_acc * wy[j as usize];
            }
            assert!(
                (fast - ref_val).abs() < 1e-3,
                "cubic mismatch at ({u},{v}): fast={fast} ref={ref_val}"
            );
        }

        // 2. Verify SIMD Horn 3x3 hillshade row against scalar eval_horn_3x3
        let mut r0 = [0.0f32; 258];
        let mut r1 = [0.0f32; 258];
        let mut r2 = [0.0f32; 258];
        for i in 0..258 {
            r0[i] = src.sample_inbounds(i as u32 + 10, 10);
            r1[i] = src.sample_inbounds(i as u32 + 10, 11);
            r2[i] = src.sample_inbounds(i as u32 + 10, 12);
        }
        let mut simd_out = [0u8; 256];
        let inv_8dx = 1.0 / (8.0 * 30.0);
        eval_horn_hillshade_row_clean::<256>(
            &r0, &r1, &r2, inv_8dx, 315.0, 45.0, 2.0, 1.0, 0.0, &mut simd_out,
        );
        for px in 0..256 {
            let c = px + 1;
            let s = eval_horn_3x3(
                r0[c - 1], r0[c], r0[c + 1],
                r1[c - 1], r1[c], r1[c + 1],
                r2[c - 1], r2[c], r2[c + 1],
                inv_8dx,
                DemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 2.0,
                },
            );
            let scalar_u8 = rescale_u8(s, 1.0, 0.0);
            let diff = (simd_out[px] as i32 - scalar_u8 as i32).abs();
            assert!(diff <= 1, "horn mismatch at px={px}: simd={} scalar={scalar_u8}", simd_out[px]);
        }

        // 3. Time 100 iterations of 256x256 cubic hillshade PNG
        let mut u = [0.0f32; 256];
        let mut v = [0.0f32; 256];
        for i in 0..256 {
            u[i] = 10.0 + (i as f32) * 1.5;
            v[i] = 10.0 + (i as f32) * 1.5;
        }
        let plan = CoordPlan::Separable1D {
            u: Box::new(u),
            v: Box::new(v),
            src_min_col: 10,
            src_max_col: 400,
            src_min_row: 10,
            src_max_row: 400,
            all_in_bounds: true,
        };
        let style = SingleBandStyle {
            rescale_min: 0.0,
            rescale_max: 255.0,
            colormap: ColormapName::from_str_opt("gray").unwrap(),
            resample: ResampleMode::Cubic,
            format: OutputFormat::Png,
        };
        let alg = DemAlgorithm::Hillshade {
            azimuth_deg: 315.0,
            altitude_deg: 45.0,
            z_factor: 2.0,
        };
        // warmup
        let _ = render_dem_algorithm(&plan, &src, alg, 30.0, &style);
        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            let png = render_dem_algorithm(&plan, &src, alg, 30.0, &style);
            std::hint::black_box(png);
        }
        let avg_us = t0.elapsed().as_secs_f64() * 1e6 / 100.0;
        eprintln!("256x256 cubic SIMD hillshade PNG avg: {avg_us:.1} µs");
    }

    #[test]
    fn dem_horn_anisotropic_spacing_matches_exact_slope() {
        // At 60°N in EPSG:4326, dx_m = 5.0 m and dy_m = 10.0 m.
        // A 45° East-facing ramp rises 5 m per East pixel (dz/dx = 1.0, dz/dy = 0.0).
        let slope = eval_horn_3x3_aniso(
            0.0, 5.0, 10.0,
            0.0, 5.0, 10.0,
            0.0, 5.0, 10.0,
            1.0 / (8.0 * 5.0),
            1.0 / (8.0 * 10.0),
            DemAlgorithm::Slope { z_factor: 1.0 },
        );
        assert!((slope - 45.0).abs() < 1e-4, "expected 45.0° slope, got {slope}");
    }

    #[test]
    fn npy_v1_header_is_64_byte_aligned_and_valid() {
        let raw = Bytes::from_iter([1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()));
        let npy = wrap_raw_f32_as_npy(raw.clone(), 1, 2, 2);
        assert_eq!(&npy[..8], b"\x93NUMPY\x01\x00");
        let hlen = u16::from_le_bytes([npy[8], npy[9]]) as usize;
        let total_hdr = 10 + hlen;
        assert_eq!(total_hdr % 64, 0, "NPY v1.0 header must be 64-byte aligned");
        let dict = std::str::from_utf8(&npy[10..total_hdr]).unwrap();
        assert!(dict.contains("'descr': '<f4'"));
        assert!(dict.contains("'fortran_order': False"));
        assert!(dict.contains("'shape': (1, 2, 2)"));
        assert!(dict.ends_with('\n'));
        assert_eq!(&npy[total_hdr..], raw.as_ref());
    }

    #[test]
    fn sample_rgb_nearest_row_clean_matches_scalar() {
        use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
        use std::sync::Arc;

        let tw = 64u32;
        let th = 64u32;
        let n_px = (tw * th) as usize;
        let mut u_arr = [0.0f32; 256];
        let mut v_arr = [0.0f32; 256];
        for i in 0..256 {
            u_arr[i] = 1.2 + (i as f32) * 0.23;
            v_arr[i] = 2.4 + (i as f32) * 0.22;
        }
        let plan = CoordPlan::Separable1D {
            u: Box::new(u_arr),
            v: Box::new(v_arr),
            src_min_col: 1,
            src_max_col: 62,
            src_min_row: 2,
            src_max_row: 62,
            all_in_bounds: true,
        };
        let stats = BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 0.0,
            max: 255.0,
        };

        let check = |srcs: [BandSource<'_>; 3], inv: [f32; 3], mins: [f32; 3], label: &str| {
            let mut fast = vec![0u8; 256 * 3];
            let mut ref_row = vec![0u8; 256 * 3];
            for py in [0usize, 17, 128, 255] {
                sample_rgb_nearest_row_clean(&plan, py, &srcs, inv, mins, &mut fast);
                for_each_pixel_in_row(&plan, py, |px, u, v| {
                    ref_row[px * 3] = rescale_u8(sample_nearest::<true>(&srcs[0], u, v), inv[0], mins[0]);
                    ref_row[px * 3 + 1] = rescale_u8(sample_nearest::<true>(&srcs[1], u, v), inv[1], mins[1]);
                    ref_row[px * 3 + 2] = rescale_u8(sample_nearest::<true>(&srcs[2], u, v), inv[2], mins[2]);
                });
                assert_eq!(fast, ref_row, "{label} mismatch at py={py}");
            }
        };

        // 1. Interleaved RgbU8Channel (identity + rescaled)
        let rgb_vec: Vec<u8> = (0..n_px * 3).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let rgb_arc: Arc<[u8]> = Arc::from(rgb_vec);
        let mk_rgb = |ch: u8| DecodedBand {
            data: BandData::RgbU8Channel { rgb: rgb_arc.clone(), channel: ch, nodata: None },
            width: tw,
            height: th,
            stats,
        };
        let (b0, b1, b2) = (mk_rgb(0), mk_rgb(1), mk_rgb(2));
        let (s0, s1, s2) = ([Some(&b0)], [Some(&b1)], [Some(&b2)]);
        check(
            [
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s0),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s1),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s2),
            ],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            "RgbU8 identity",
        );
        check(
            [
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s0),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s1),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s2),
            ],
            [1.5, 0.8, 2.0],
            [10.0, 20.0, 5.0],
            "RgbU8 rescaled",
        );

        // 2. Planar U8
        let mk_u8 = |seed: usize| DecodedBand {
            data: BandData::U8 {
                data: (0..n_px).map(|i| ((i * seed + 7) % 256) as u8).collect(),
                nodata: None,
            },
            width: tw,
            height: th,
            stats,
        };
        let (u0, u1, u2) = (mk_u8(13), mk_u8(29), mk_u8(53));
        let (su0, su1, su2) = ([Some(&u0)], [Some(&u1)], [Some(&u2)]);
        check(
            [
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &su0),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &su1),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &su2),
            ],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            "U8 identity",
        );

        // 3. Planar U16
        let mk_u16 = |seed: usize| DecodedBand {
            data: BandData::U16 {
                data: (0..n_px).map(|i| ((i * seed + 100) % 4096) as u16).collect(),
                nodata: None,
            },
            width: tw,
            height: th,
            stats,
        };
        let (w0, w1, w2) = (mk_u16(17), mk_u16(31), mk_u16(47));
        let (sw0, sw1, sw2) = ([Some(&w0)], [Some(&w1)], [Some(&w2)]);
        check(
            [
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sw0),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sw1),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sw2),
            ],
            [255.0 / 3000.0; 3],
            [100.0; 3],
            "U16 rescaled",
        );

        // 4. Planar F32
        let mk_f32 = |seed: f32| {
            let mut slab = AlignedF32Slab::filled(n_px, 0.0);
            for (i, v) in slab.as_mut_slice().iter_mut().enumerate() {
                *v = (i as f32) * 0.01 * seed;
            }
            DecodedBand {
                data: BandData::F32(slab),
                width: tw,
                height: th,
                stats,
            }
        };
        let (f0, f1, f2) = (mk_f32(1.0), mk_f32(1.7), mk_f32(2.3));
        let (sf0, sf1, sf2) = ([Some(&f0)], [Some(&f1)], [Some(&f2)]);
        check(
            [
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sf0),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sf1),
                BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sf2),
            ],
            [255.0 / 100.0; 3],
            [0.0; 3],
            "F32 rescaled",
        );
    }

    #[test]
    fn single_block_u16_u8_and_rgb_separable_fast_paths_match_scalar() {
        use crate::colormap::ColormapName;
        use cogzilla_codec::{BandData, BlockStats, DecodedBand};
        use std::sync::Arc;

        let tw = 64u32;
        let th = 64u32;
        let n_px = (tw * th) as usize;
        let stats = BlockStats {
            min: 0.0,
            max: 4000.0,
            all_nodata: false,
            no_nodata: true,
        };
        let mut u = [0.0f32; 256];
        let mut v = [0.0f32; 256];
        for i in 0..256 {
            u[i] = 1.25 + (i as f32) * (60.0 / 255.0);
            v[i] = 1.75 + (i as f32) * (59.5 / 255.0);
        }
        let plan = CoordPlan::Separable1D {
            u: Box::new(u),
            v: Box::new(v),
            src_min_col: 1,
            src_max_col: 62,
            src_min_row: 1,
            src_max_row: 62,
            all_in_bounds: true,
        };

        // 1. Single-band U16 & U8 (Nearest + Bilinear) in render_single_specialized + render_expr_vm
        let u16_a = DecodedBand {
            data: BandData::U16 {
                data: (0..n_px).map(|i| ((i * 29 + 17) % 3500) as u16).collect(),
                nodata: None,
            },
            width: tw,
            height: th,
            stats,
        };
        let u16_b = DecodedBand {
            data: BandData::U16 {
                data: (0..n_px).map(|i| ((i * 13 + 211) % 3500) as u16).collect(),
                nodata: None,
            },
            width: tw,
            height: th,
            stats,
        };
        let u8_a = DecodedBand {
            data: BandData::U8 {
                data: (0..n_px).map(|i| ((i * 7 + 5) & 0xFF) as u8).collect(),
                nodata: None,
            },
            width: tw,
            height: th,
            stats,
        };
        let s_u16_a = [Some(&u16_a)];
        let s_u16_b = [Some(&u16_b)];
        let s_u8_a = [Some(&u8_a)];
        let bs_u16_a = BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s_u16_a);
        let bs_u8_a = BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s_u8_a);

        for (label, bs) in [("U16", &bs_u16_a), ("U8", &bs_u8_a)] {
            for (is_bilin, resample) in [
                (false, ResampleMode::Nearest),
                (true, ResampleMode::Bilinear),
            ] {
                let style = SingleBandStyle {
                    rescale_min: 0.0,
                    rescale_max: 255.0,
                    colormap: ColormapName::Viridis,
                    format: OutputFormat::RawF32,
                    resample,
                };
                let got = if is_bilin {
                    render_single_specialized::<256, true, true>(&plan, bs, &style)
                } else {
                    render_single_specialized::<256, false, true>(&plan, bs, &style)
                };
                for py in 0..256 {
                    for px in 0..256 {
                        let idx = (py * 256 + px) * 4;
                        let g = f32::from_le_bytes(got[idx..idx + 4].try_into().unwrap());
                        let e = if is_bilin {
                            sample_bilinear::<true>(bs, u[px], v[py])
                        } else {
                            sample_nearest::<true>(bs, u[px], v[py])
                        };
                        assert!(
                            (g - e).abs() <= 1e-3,
                            "{label} bilin={is_bilin} mismatch at ({px},{py}): got {g}, exp {e}"
                        );
                    }
                }
            }
        }

        // 2. render_expr_vm on U16 pair (Nearest + Bilinear)
        let prog = crate::expr_vm::ExprProgram::compile("(b1 - b2) / (b1 + b2 + 1.0)").unwrap();
        let pair = [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s_u16_a),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s_u16_b),
        ];
        for resample in [ResampleMode::Nearest, ResampleMode::Bilinear] {
            let style = SingleBandStyle {
                rescale_min: -1.0,
                rescale_max: 1.0,
                colormap: ColormapName::Viridis,
                format: OutputFormat::RawF32,
                resample,
            };
            let got = render_expr_vm(&plan, &pair, &prog, &style, (tw, th));
            for py in 0..256 {
                for px in 0..256 {
                    let idx = (py * 256 + px) * 4;
                    let g = f32::from_le_bytes(got[idx..idx + 4].try_into().unwrap());
                    let v1 = match resample {
                        ResampleMode::Bilinear => sample_bilinear::<true>(&pair[0], u[px], v[py]),
                        _ => sample_nearest::<true>(&pair[0], u[px], v[py]),
                    };
                    let v2 = match resample {
                        ResampleMode::Bilinear => sample_bilinear::<true>(&pair[1], u[px], v[py]),
                        _ => sample_nearest::<true>(&pair[1], u[px], v[py]),
                    };
                    let e = (v1 - v2) / (v1 + v2 + 1.0);
                    assert!(
                        (g - e).abs() <= 1e-4,
                        "expr_vm {resample:?} mismatch at ({px},{py}): got {g}, exp {e}"
                    );
                }
            }
        }

        // 3. Interleaved RgbU8Channel on Separable1D (hoisted Nearest + SIMD Bilinear)
        let rgb_raw: Arc<[u8]> = (0..n_px * 3).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
        let ch = |c: u8| DecodedBand {
            data: BandData::RgbU8Channel {
                rgb: Arc::clone(&rgb_raw),
                channel: c,
                nodata: None,
            },
            width: tw,
            height: th,
            stats,
        };
        let (c0, c1, c2) = (ch(0), ch(1), ch(2));
        let (sc0, sc1, sc2) = ([Some(&c0)], [Some(&c1)], [Some(&c2)]);
        let rgb_srcs = [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sc0),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sc1),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sc2),
        ];
        for (mins, maxs) in [([0.0; 3], [255.0; 3]), ([10.0, 20.0, 30.0], [220.0, 230.0, 240.0])] {
            let inv = [
                255.0 / (maxs[0] - mins[0]),
                255.0 / (maxs[1] - mins[1]),
                255.0 / (maxs[2] - mins[2]),
            ];
            let np = CleanRgbNearestPlan::build(&plan, &rgb_srcs, inv, mins);
            let mut row_fast = [0u8; 256 * 3];
            let mut row_ref = [0u8; 256 * 3];
            for py in [0usize, 13, 128, 255] {
                np.sample_row(&plan, py, &rgb_srcs, inv, mins, &mut row_fast);
                sample_rgb_nearest_row_clean(&plan, py, &rgb_srcs, inv, mins, &mut row_ref);
                assert_eq!(row_fast, row_ref, "CleanRgbNearestPlan mismatch at py={py}");

                sample_rgb_bilinear_row_clean(&plan, py, &rgb_srcs, inv, mins, &mut row_fast);
                for px in 0..256 {
                    for ch_i in 0..3 {
                        let val = sample_bilinear::<true>(&rgb_srcs[ch_i], u[px], v[py]);
                        let exp = rescale_u8(val, inv[ch_i], mins[ch_i]);
                        let diff = (row_fast[px * 3 + ch_i] as i32 - exp as i32).abs();
                        assert!(
                            diff <= 1,
                            "RgbU8 bilinear mismatch at ({px},{py}) ch={ch_i}: got {}, exp {exp}",
                            row_fast[px * 3 + ch_i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn render_expr_vm_masks_oob_pixels_for_constant_and_where_expressions() {
        let mut u = [0.0f32; 16];
        let mut v = [0.0f32; 16];
        // Left half (-8..-1) is out-of-bounds, right half (0..7) is inside a 16x16 IFD.
        for i in 0..16 {
            u[i] = i as f32 - 8.0;
            v[i] = i as f32;
        }
        let plan = CoordPlan::Separable1D {
            u: Box::new(u),
            v: Box::new(v),
            src_min_col: -8,
            src_max_col: 7,
            src_min_row: 0,
            src_max_row: 15,
            all_in_bounds: false,
        };
        let style = SingleBandStyle {
            rescale_min: 0.0,
            rescale_max: 10.0,
            colormap: ColormapName::Grayscale,
            format: OutputFormat::RawF32,
            resample: ResampleMode::Nearest,
        };

        // 1. Constant expression (`1`) with empty sources
        let const_prog = crate::expr_vm::ExprProgram::compile("1").unwrap();
        let out_const = render_expr_vm(&plan, &[], &const_prog, &style, (16, 16));
        let oob_val = f32::from_le_bytes(out_const[0..4].try_into().unwrap());
        let in_val = f32::from_le_bytes(out_const[8 * 4..9 * 4].try_into().unwrap());
        assert!(oob_val.is_nan(), "OOB pixel for constant expr must be NaN, got {oob_val}");
        assert_eq!(in_val, 1.0, "in-bounds pixel for constant expr must be 1.0");

        // 2. where(1, 5, b1) with a real band source
        let band = cogzilla_codec::DecodedBand {
            width: 16,
            height: 16,
            data: cogzilla_codec::BandData::U8 {
                data: vec![2u8; 16 * 16].into_boxed_slice(),
                nodata: None,
            },
            stats: cogzilla_codec::BlockStats {
                min: 2.0,
                max: 2.0,
                all_nodata: false,
                no_nodata: true,
            },
        };
        let refs = [Some(&band)];
        let src = [BandSource::new(16, 16, 16, 16, 0, 0, 1, 1, &refs)];
        let where_prog = crate::expr_vm::ExprProgram::compile("where(1, 5, b1)").unwrap();
        let out_where = render_expr_vm(&plan, &src, &where_prog, &style, (16, 16));
        let oob_w = f32::from_le_bytes(out_where[0..4].try_into().unwrap());
        let in_w = f32::from_le_bytes(out_where[8 * 4..9 * 4].try_into().unwrap());
        assert!(oob_w.is_nan(), "OOB pixel for where() expr must be NaN, got {oob_w}");
        assert_eq!(in_w, 5.0, "in-bounds pixel for where(1, 5, b1) must be 5.0");
    }
}

/// Golden bytes of the `.npy` float32 writers: the header in full for a spread of shapes,
/// and the length and SHA-256 of whole arrays. These pin the response bytes, so a change
/// here is a change to what clients receive.
#[cfg(test)]
mod npy_golden_tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The header as it must be written: magic, version 1.0, dict length, the dict, `pad`
    /// spaces and a newline.
    fn header(dict: &str, pad: usize) -> Vec<u8> {
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&((dict.len() + pad + 1) as u16).to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.extend(std::iter::repeat_n(b' ', pad));
        out.push(b'\n');
        out
    }

    /// Reproducible float32 planes with NaN, both infinities, negatives and fractions.
    fn planes(bands: usize, height: u32, width: u32, seed: u32) -> Vec<u8> {
        let count = bands * height as usize * width as usize;
        let mut state = seed;
        let mut out = Vec::with_capacity(count * 4);
        for _ in 0..count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = match (state >> 16) % 11 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                _ => ((state >> 8) as i32 - (1 << 23)) as f32 / 1000.0,
            };
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn f32_header_bytes_are_pinned() {
        // (bands, height, width), the dict, and the spaces that pad it.
        let cases: [(usize, u32, u32, &str, usize); 7] = [
            (1, 2, 2, "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2, 2), }", 55),
            (3, 256, 256, "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 256, 256), }", 51),
            (4, 7, 13, "{'descr': '<f4', 'fortran_order': False, 'shape': (4, 7, 13), }", 54),
            (10, 1024, 512, "{'descr': '<f4', 'fortran_order': False, 'shape': (10, 1024, 512), }", 49),
            (2, 12345, 1, "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 12345, 1), }", 51),
            (0, 0, 0, "{'descr': '<f4', 'fortran_order': False, 'shape': (0, 0, 0), }", 55),
            (
                usize::MAX,
                u32::MAX,
                u32::MAX,
                "{'descr': '<f4', 'fortran_order': False, 'shape': (18446744073709551615, 4294967295, 4294967295), }",
                18,
            ),
        ];
        for (bands, height, width, dict, pad) in cases {
            let got = encode_npy_f32_header(bands, height, width);
            assert_eq!(got.len(), 128, "header length for ({bands}, {height}, {width})");
            assert_eq!(got, header(dict, pad), "header bytes for ({bands}, {height}, {width})");
        }
    }

    #[test]
    fn f32_small_arrays_are_pinned_in_full() {
        let le = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        let nan = f32::NAN;

        // One band, 2x2: the planes follow the header untouched, then the mask.
        let values = [1.5, nan, f32::NEG_INFINITY, -2.0];
        let mut exp = header("{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2, 2), }", 55);
        exp.extend_from_slice(&le(&values));
        let got = wrap_raw_f32_as_npy(Bytes::from(le(&values)), 1, 2, 2);
        assert_eq!(got.as_ref(), exp.as_slice());
        let got = wrap_raw_f32_as_npy_with_mask(Bytes::from(le(&values)), 1, 2, 2, false);
        assert_eq!(got.as_ref(), exp.as_slice());

        let mut exp = header("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 2), }", 55);
        exp.extend_from_slice(&le(&values));
        exp.extend_from_slice(&le(&[255.0, 0.0, 0.0, 255.0]));
        let got = wrap_raw_f32_as_npy_with_mask(Bytes::from(le(&values)), 1, 2, 2, true);
        assert_eq!(got.as_ref(), exp.as_slice());

        // Three bands, 1 row of 2: a pixel is masked out only when no band is finite there.
        let values = [nan, 1.0, f32::INFINITY, nan, nan, -0.0];
        let mut exp = header("{'descr': '<f4', 'fortran_order': False, 'shape': (4, 1, 2), }", 55);
        exp.extend_from_slice(&le(&values));
        exp.extend_from_slice(&le(&[0.0, 255.0]));
        let got = wrap_raw_f32_as_npy_with_mask(Bytes::from(le(&values)), 3, 1, 2, true);
        assert_eq!(got.as_ref(), exp.as_slice());

        // An empty tile is the header and NaN throughout.
        let mut exp = header("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 3), }", 55);
        exp.extend_from_slice(&le(&[nan; 18]));
        assert_eq!(empty_tile_bytes(3, OutputFormat::Npy, 2).as_ref(), exp.as_slice());
    }

    #[test]
    fn f32_arrays_are_pinned_by_length_and_digest() {
        // bands, height, width, mask, bytes cut from the end of the planes, length, SHA-256.
        let cases: [(usize, u32, u32, bool, usize, usize, &str); 14] = [
            (1, 2, 2, false, 0, 144, "a2f108947a23fd0bf4a45863a61642c8737f4a737de4ae562e04a90bc71c05f4"),
            (1, 2, 2, true, 0, 160, "d0037009d0ba0f13e146d568787c345a423ec2907d6cb00b31000ec65258a30b"),
            (3, 1, 5, true, 0, 208, "bf56f2c2b8000609fb42496e22e5f943de6b0a877470fe254d27cb89cab7e05f"),
            (3, 7, 13, false, 0, 1220, "17cdaaea973e5db420409f60adeab2ee13586b2544a1a5fc1ba1ef7f07e74e6d"),
            (3, 7, 13, true, 0, 1584, "9d14efff4748fd24de1c2a71c6ad4847fa1ea5b8f5f522fcfe69edefd28c9a39"),
            (1, 256, 256, false, 0, 262272, "7d0665ee74caf962f474512cae016c2846cd45d6196a7f1651a73156b3dcc705"),
            (1, 256, 256, true, 0, 524416, "020d46b447418a8fe60c40d93f4fc624d2149590ad38d6d38ee90130a725a835"),
            (3, 256, 256, true, 0, 1048704, "2eb4f63e99fb4a12b94f3a397ed4bd1e6e33e7ce715ad8de64aa5eca21adc716"),
            (3, 512, 512, false, 0, 3145856, "87028dd38eb31c700fef51f6918423e0674e54af6bf0106d9cef4eedbd1e7337"),
            (1, 1000, 3, true, 0, 24128, "0ec5029e87634968585a837ed56567549724a34e4a8c77bb196626a3269fe510"),
            (2, 1, 1024, true, 0, 12416, "58b72da7c4d034a04afe596f22dda8eea4c670780a1258d5629959694db7ac6f"),
            (4, 12345, 1, true, 0, 247028, "8e8bf86438d0a349a9780d8e920d24537ea1292fbf9e629eb88273ef25ba4c34"),
            (3, 9, 17, true, 150, 2426, "530137fad2f5a61fa10aa1043ed85832f2155ed0a9624fd94ecb8a9e1b9c4e33"),
            (0, 5, 6, true, 0, 248, "7225ef713825a642977bb6c7f903da5f140d1920d95aae073503dd2be4fdcc62"),
        ];
        for (i, (bands, height, width, mask, cut, len, digest)) in cases.into_iter().enumerate() {
            let mut raw = planes(bands, height, width, 0x9E37_79B9 ^ i as u32);
            raw.truncate(raw.len() - cut);
            let got = wrap_raw_f32_as_npy_with_mask(Bytes::from(raw.clone()), bands, height, width, mask);
            let what = format!("case {i}: ({bands}, {height}, {width}), mask {mask}, cut {cut}");
            assert_eq!(got.len(), len, "{what}");
            assert_eq!(sha256_hex(&got), digest, "{what}");
            if !mask {
                assert_eq!(got, wrap_raw_f32_as_npy(Bytes::from(raw), bands, height, width));
            }
        }
    }

    #[test]
    fn empty_npy_tiles_are_pinned_by_length_and_digest() {
        // tile size, bands, length, SHA-256.
        let cases: [(u32, usize, usize, &str); 5] = [
            (256, 1, 262272, "820ec7e5f3dfd5b867b3fd2e8d56e43f8c6afe3bde4a76720a77eafefdfd002f"),
            (256, 3, 786560, "803f28ae94e98efe8807f4fe51b0772f36d8318ba550ec45e5135cb70496d5de"),
            (512, 1, 1048704, "845324c7127f73c5a480b5b24d1a9b228c802e837171d77a33994bae555550cd"),
            (64, 2, 32896, "85a5dff8aa70b4e8df6b7455d9c44fcce85905cec33692d1b4c2776edfee5cee"),
            (1, 5, 148, "a5497f607d665ebcbf609c18495d17f61a4f30eb9107430f00f8c94f791dc6a5"),
        ];
        for (size, bands, len, digest) in cases {
            let got = empty_tile_bytes(size, OutputFormat::Npy, bands);
            assert_eq!(got.len(), len, "empty {size} tile of {bands} bands");
            assert_eq!(sha256_hex(&got), digest, "empty {size} tile of {bands} bands");
        }
    }
}

/// The typed `.npy` writer and the header it shares with the float32 one.
#[cfg(test)]
mod npy_typed_tests {
    use super::*;

    const KINDS: [(SampleType, &str, usize); 8] = [
        (SampleType::Uint8, "|u1", 1),
        (SampleType::Int8, "|i1", 1),
        (SampleType::Uint16, "<u2", 2),
        (SampleType::Int16, "<i2", 2),
        (SampleType::Uint32, "<u4", 4),
        (SampleType::Int32, "<i4", 4),
        (SampleType::Float32, "<f4", 4),
        (SampleType::Float64, "<f8", 8),
    ];

    /// Read a header back: the dtype, the order, the shape, and where the data starts.
    fn parse(npy: &[u8]) -> (String, String, Vec<usize>, usize) {
        assert_eq!(&npy[..8], b"\x93NUMPY\x01\x00");
        let start = 10 + usize::from(u16::from_le_bytes([npy[8], npy[9]]));
        assert_eq!(start % 64, 0, "the data starts on a multiple of 64 bytes");
        let dict = std::str::from_utf8(&npy[10..start]).unwrap();
        assert!(dict.ends_with('\n'));
        assert!(dict[..dict.len() - 1].trim_end_matches(' ').ends_with(", }"));
        let field = |name: &str, end: &str| {
            let from = dict.find(name).unwrap_or_else(|| panic!("{name} in {dict}")) + name.len();
            dict[from..from + dict[from..].find(end).unwrap()].to_string()
        };
        let shape = field("'shape': (", ")")
            .split(',')
            .map(str::trim)
            .filter(|dim| !dim.is_empty())
            .map(|dim| dim.parse().unwrap())
            .collect();
        (field("'descr': '", "'"), field("'fortran_order': ", ","), shape, start)
    }

    #[test]
    fn header_is_general_over_dtype_and_shape() {
        for (_, descr, _) in KINDS {
            for shape in [vec![], vec![7], vec![1, 2, 2], vec![4, 1024, 512], vec![2, 3, 4, 5]] {
                let hdr = encode_npy_header(descr, &shape);
                let (got_descr, order, got_shape, start) = parse(&hdr);
                assert_eq!((got_descr.as_str(), order.as_str()), (descr, "False"));
                assert_eq!(got_shape, shape);
                assert_eq!(start, hdr.len());
            }
        }
        // NumPy writes a tuple of one with its comma, and of none as `()`.
        let dict = |hdr: Vec<u8>| String::from_utf8(hdr[10..].to_vec()).unwrap();
        assert!(dict(encode_npy_header("<i2", &[7])).starts_with("{'descr': '<i2', 'fortran_order': False, 'shape': (7,), }"));
        assert!(dict(encode_npy_header("<f8", &[])).starts_with("{'descr': '<f8', 'fortran_order': False, 'shape': (), }"));
    }

    #[test]
    fn header_pads_to_the_next_64_bytes() {
        // Shapes of 2 to 40 dimensions, led by a number of one or two digits, so the dict
        // runs through every length from short to past 128 bytes.
        let (mut unpadded, mut short, mut long) = (false, false, false);
        for first in [1usize, 10] {
            for dims in 2..=40usize {
                let mut shape = vec![1usize; dims];
                shape[0] = first;
                let inner = shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
                let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({inner}), }}");
                let total = (10 + dict.len() + 1).div_ceil(64) * 64;
                let pad = total - 11 - dict.len();
                let hdr = encode_npy_header("<f4", &shape);
                assert_eq!(hdr.len(), total, "{dict}");
                assert_eq!(usize::from(u16::from_le_bytes([hdr[8], hdr[9]])), total - 10);
                assert_eq!(&hdr[10..10 + dict.len()], dict.as_bytes());
                assert!(hdr[10 + dict.len()..total - 1].iter().all(|b| *b == b' '));
                assert_eq!(hdr[total - 1], b'\n');
                unpadded |= pad == 0;
                short |= total == 128;
                long |= total == 192;
            }
        }
        assert!(unpadded && short && long, "a dict that fills its block, and both block counts");
    }

    #[test]
    fn f32_header_is_the_general_header() {
        for (bands, height, width) in [(1usize, 2u32, 2u32), (3, 256, 256), (10, 1024, 512), (usize::MAX, u32::MAX, u32::MAX)] {
            assert_eq!(
                encode_npy_f32_header(bands, height, width),
                encode_npy_header("<f4", &[bands, height as usize, width as usize])
            );
        }
    }

    #[test]
    fn every_dtype_parses_back_with_a_body_of_the_right_length() {
        let values: Vec<f32> = (0..3 * 5 * 7).map(|i| if i % 4 == 0 { f32::NAN } else { i as f32 }).collect();
        for (kind, descr, size) in KINDS {
            assert_eq!(npy_descr(kind), descr);
            assert_eq!(kind.bytes_per_sample(), size);
            for bands in [1usize, 3] {
                for mask in [false, true] {
                    let npy = encode_npy_typed(&values, bands, 5, 7, kind, None, mask);
                    let (got_descr, order, shape, start) = parse(&npy);
                    assert_eq!((got_descr.as_str(), order.as_str()), (descr, "False"));
                    assert_eq!(shape, [bands + usize::from(mask), 5, 7]);
                    assert_eq!(npy.len() - start, (bands + usize::from(mask)) * 5 * 7 * size);
                }
            }
        }
    }

    /// The body of a 1x4 single-band array of `values`, with its mask plane.
    fn body(values: &[f32], kind: SampleType, absent: Option<f32>, mask: bool) -> Vec<u8> {
        let npy = encode_npy_typed(values, 1, 1, values.len(), kind, absent, mask);
        npy[parse(&npy).3..].to_vec()
    }

    #[test]
    fn values_fill_and_mask_in_each_dtype() {
        let nan = f32::NAN;
        let values = [1.9, nan, -2.9, 70000.0];
        macro_rules! le {
            ($t:ty, $($v:expr),*) => {{
                let mut out = Vec::new();
                $(out.extend_from_slice(&<$t>::to_le_bytes($v));)*
                out
            }};
        }
        // Truncated toward zero, held to the range, absent as zero; then the mask, the
        // type's largest number where there is a value and its smallest where there is not.
        assert_eq!(body(&values, SampleType::Uint8, None, true), le!(u8, 1, 0, 0, 255, 255, 0, 255, 255));
        assert_eq!(body(&values, SampleType::Int8, None, true), le!(i8, 1, 0, -2, 127, 127, -128, 127, 127));
        assert_eq!(body(&values, SampleType::Uint16, None, true), le!(u16, 1, 0, 0, 65535, 65535, 0, 65535, 65535));
        assert_eq!(body(&values, SampleType::Int16, None, true), le!(i16, 1, 0, -2, 32767, 32767, -32768, 32767, 32767));
        assert_eq!(body(&values, SampleType::Uint32, None, true), le!(u32, 1, 0, 0, 70000, u32::MAX, 0, u32::MAX, u32::MAX));
        assert_eq!(body(&values, SampleType::Int32, None, true), le!(i32, 1, 0, -2, 70000, i32::MAX, i32::MIN, i32::MAX, i32::MAX));
        assert_eq!(
            body(&values, SampleType::Float32, None, true),
            le!(f32, 1.9, 0.0, -2.9, 70000.0, f32::MAX, f32::MIN, f32::MAX, f32::MAX)
        );
        assert_eq!(
            body(&values, SampleType::Float64, None, true),
            le!(f64, f64::from(1.9f32), 0.0, f64::from(-2.9f32), 70000.0, f64::MAX, f64::MIN, f64::MAX, f64::MAX)
        );
        // Without the mask the body is the bands alone.
        assert_eq!(body(&values, SampleType::Uint16, None, false), le!(u16, 1, 0, 0, 65535));
        assert_eq!(body(&values, SampleType::Float32, None, false), le!(f32, 1.9, 0.0, -2.9, 70000.0));
    }

    #[test]
    fn absent_values_take_the_named_fill() {
        let nan = f32::NAN;
        let values = [nan, 5.0, nan];
        assert_eq!(body(&values, SampleType::Uint8, Some(9.0), false), [9, 5, 9]);
        assert_eq!(body(&values, SampleType::Uint8, None, false), [0, 5, 0]);
        // The fill is cast as a value is: out of range it is held to the range.
        assert_eq!(body(&values, SampleType::Uint8, Some(-9999.0), false), [0, 5, 0]);
        assert_eq!(body(&values, SampleType::Int16, Some(-9999.0), false), [(-9999i16).to_le_bytes(), 5i16.to_le_bytes(), (-9999i16).to_le_bytes()].concat());
        assert_eq!(body(&values, SampleType::Float32, Some(-9999.0), false), [(-9999f32).to_le_bytes(), 5f32.to_le_bytes(), (-9999f32).to_le_bytes()].concat());
        // A fill of NaN is zero in an integer and stays NaN in a float.
        assert_eq!(body(&values, SampleType::Int32, Some(nan), false), [0i32.to_le_bytes(), 5i32.to_le_bytes(), 0i32.to_le_bytes()].concat());
        assert!(f32::from_le_bytes(body(&values, SampleType::Float32, Some(nan), false)[..4].try_into().unwrap()).is_nan());
        // The fill does not reach the mask, which reads the values given.
        assert_eq!(body(&values, SampleType::Uint8, Some(9.0), true), [9, 5, 9, 0, 255, 0]);
    }

    #[test]
    fn floats_hold_infinities_to_the_largest_finite_number() {
        let values = [f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(body(&values, SampleType::Float32, None, false), [f32::MAX.to_le_bytes(), f32::MIN.to_le_bytes()].concat());
        assert_eq!(
            body(&values, SampleType::Float64, None, false),
            [f64::MAX.to_le_bytes(), f64::MIN.to_le_bytes()].concat()
        );
        assert_eq!(body(&values, SampleType::Int16, None, true), [i16::MAX.to_le_bytes(), i16::MIN.to_le_bytes(), i16::MAX.to_le_bytes(), i16::MAX.to_le_bytes()].concat());
    }

    #[test]
    fn mask_is_set_where_any_band_has_a_value() {
        let nan = f32::NAN;
        // Three bands of 1x3: nothing at the first pixel, one band at the second, all at the third.
        let values = [nan, nan, 1.0, nan, 2.0, 3.0, nan, nan, 0.0];
        let npy = encode_npy_typed(&values, 3, 1, 3, SampleType::Uint16, None, true);
        let (_, _, shape, start) = parse(&npy);
        assert_eq!(shape, [4, 1, 3]);
        assert_eq!(&npy[start + 18..], [0u16.to_le_bytes(), u16::MAX.to_le_bytes(), u16::MAX.to_le_bytes()].concat().as_slice());
    }

    #[test]
    fn only_the_named_planes_are_read() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // More values than the planes hold: the rest is left out.
        let npy = encode_npy_typed(&values, 1, 2, 2, SampleType::Uint8, None, false);
        assert_eq!(&npy[parse(&npy).3..], [1, 2, 3, 4]);
        // Fewer: the body is short, and the mask counts what is missing as absent.
        let npy = encode_npy_typed(&values[..3], 1, 2, 2, SampleType::Uint8, None, true);
        let (_, _, shape, start) = parse(&npy);
        assert_eq!(shape, [2, 2, 2]);
        assert_eq!(&npy[start..], [1, 2, 3, 255, 255, 255, 0]);
        // No bands: a mask of nothing there, the type's smallest number.
        let npy = encode_npy_typed(&[], 0, 2, 2, SampleType::Int16, None, true);
        assert_eq!(&npy[parse(&npy).3..], i16::MIN.to_le_bytes().repeat(4).as_slice());
    }

    #[test]
    fn typed_float32_is_not_the_raw_float32_wrapper() {
        // The two float32 writers agree on the header and on finite values, and differ
        // in what they write for an absent value, an infinity and the mask.
        let values = [1.5f32, -2.0, 3.25, 4.0];
        let raw = Bytes::from(values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        assert_eq!(
            encode_npy_typed(&values, 1, 2, 2, SampleType::Float32, None, false),
            wrap_raw_f32_as_npy(raw.clone(), 1, 2, 2)
        );
        let typed = encode_npy_typed(&values, 1, 2, 2, SampleType::Float32, None, true);
        let wrapped = wrap_raw_f32_as_npy_with_mask(raw, 1, 2, 2, true);
        assert_eq!(typed[..144], wrapped[..144]);
        assert_eq!(&typed[144..148], f32::MAX.to_le_bytes());
        assert_eq!(&wrapped[144..148], 255.0f32.to_le_bytes());
    }

    #[test]
    fn simd_tail_loops_write_all_pixels_for_non_multiple_widths() {
        // N = 5 is neither a multiple of 4 (NEON) nor 8 (AVX2).
        let u = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let v = [1.0f32, 1.0, 1.0, 1.0, 1.0];
        let band = vec![100u8; 64];
        let mut rgb = [0xAAu8; 5 * 3];
        bilinear_rgb_single_block_row::<5, u8, _>(
            &u,
            &v,
            &band,
            &band,
            &band,
            0,
            0,
            8,
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            &mut rgb,
            |x| x as f32,
        );
        assert!(
            rgb.iter().all(|&b| b == 100),
            "tail pixel in bilinear_rgb_single_block_row::<5> was not written: {rgb:?}"
        );

        let r0 = [10.0f32; 7];
        let r1 = [10.0f32; 7];
        let r2 = [10.0f32; 7];
        let mut hs = [0xEEu8; 5];
        eval_horn_hillshade_row_clean::<5>(&r0, &r1, &r2, 1.0 / 80.0, 315.0, 45.0, 1.0, 1.0, 0.0, &mut hs);
        assert!(
            hs.iter().all(|&b| b == hs[0] && b != 0xEE),
            "tail pixel in eval_horn_hillshade_row_clean::<5> was not written: {hs:?}"
        );
    }

    #[test]
    fn exact2d_coord_plan_scale_handles_nan_center_and_n1() {
        // Audit #68 #2: N == 1 must not panic OOB
        let plan1 = CoordPlan::Exact2D {
            u: vec![5.0].into_boxed_slice(),
            v: vec![5.0].into_boxed_slice(),
            all_in_bounds: true,
            src_min_col: 0,
            src_max_col: 10,
            src_min_row: 0,
            src_max_row: 10,
        };
        assert_eq!(coord_plan_scale::<1>(&plan1), (1.0, 1.0));

        // Exact2D with NaN center (N/2, N/2) but valid 16x downscale in top-left quadrant
        let n = 8usize;
        let mut u = vec![f32::NAN; n * n];
        let mut v = vec![f32::NAN; n * n];
        for py in 0..3 {
            for px in 0..3 {
                u[py * n + px] = (px as f32) * 16.0;
                v[py * n + px] = (py as f32) * 16.0;
            }
        }
        let plan8 = CoordPlan::Exact2D {
            u: u.into_boxed_slice(),
            v: v.into_boxed_slice(),
            all_in_bounds: false,
            src_min_col: 0,
            src_max_col: 48,
            src_min_row: 0,
            src_max_row: 48,
        };
        let (sx, sy) = coord_plan_scale::<8>(&plan8);
        assert!((sx - 16.0).abs() < 1e-4 && (sy - 16.0).abs() < 1e-4, "got ({sx}, {sy})");
    }
}
