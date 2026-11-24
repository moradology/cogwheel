use std::cell::RefCell;

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_proj::{CoordPlan, GRID_DIM};

use crate::bandmath::{SpectralExpr, fast_db_from_power};
use crate::colormap::{ColormapLut, ColormapName};
use crate::expr_vm::ExprProgram;
use crate::jpeg_fast::{empty_black_jpeg, encode_jpeg_rgb, encode_jpeg_rgba};
use crate::png_fast::{
    empty_transparent_png, encode_filtered_png, encode_indexed_png, filter_sub_rgb_row,
    filter_sub_rgba_row, filter_sub_u8_row,
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Png,
    /// `quality: None` = lossless WebP, `Some(q)` = lossy quality (e.g. `82.0`).
    Webp { quality: Option<f32> },
    /// JPEG (lossy quality `1..=100`, default `85`).
    Jpeg { quality: u8 },
    RawF32,
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
/// Uses 64-bit adjacent-pair loads + shuffle instead of microcoded `vgatherdps` (much faster on
/// AMD Zen 4 / EPYC as well as Intel).
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
        let idx = _mm256_cvtps_epi32(x0_f);
        _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, idx);

        let (v00, v10) = load_8_pairs!(row0_ptr);
        let (v01, v11) = load_8_pairs!(row1_ptr);

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
        let adj = i - ((i as f32 > x) as i32);
        (adj, x - (adj as f32))
    }
}

#[inline(always)]
fn sample_bilinear<const CLEAN: bool>(src: &BandSource<'_>, u: f32, v: f32) -> f32 {
    if !CLEAN && (!u.is_finite() || !v.is_finite()) {
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
    if !u.is_finite() || !v.is_finite() {
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
        OutputFormat::Jpeg { .. } => empty_black_jpeg(tile_size),
        OutputFormat::RawF32 => {
            let count = (tile_size as usize) * (tile_size as usize) * num_bands;
            let mut buf = BytesMut::with_capacity(count * 4);
            let nan_bytes = f32::NAN.to_le_bytes();
            for _ in 0..count {
                buf.put_slice(&nan_bytes);
            }
            buf.freeze()
        }
        OutputFormat::RawRgba => Bytes::from(vec![0u8; (tile_size as usize) * (tile_size as usize) * 4]),
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
            if all_opaque {
                let row_len = n * 3 + 1;
                let mut filtered = vec![0u8; n * row_len];
                let mut rgb_row = vec![0u8; n * 3];
                for y in 0..n {
                    let src = &rgba[y * n * 4..(y + 1) * n * 4];
                    for x in 0..n {
                        rgb_row[x * 3] = src[x * 4];
                        rgb_row[x * 3 + 1] = src[x * 4 + 1];
                        rgb_row[x * 3 + 2] = src[x * 4 + 2];
                    }
                    filter_sub_rgb_row(&rgb_row, &mut filtered[y * row_len..(y + 1) * row_len]);
                }
                encode_filtered_png(size, size, 2, &filtered)
            } else {
                let row_len = n * 4 + 1;
                let mut filtered = vec![0u8; n * row_len];
                for y in 0..n {
                    let src = &rgba[y * n * 4..(y + 1) * n * 4];
                    filter_sub_rgba_row(src, &mut filtered[y * row_len..(y + 1) * row_len]);
                }
                encode_filtered_png(size, size, 6, &filtered)
            }
        }
        OutputFormat::Webp { quality } => encode_webp_rgba(size, size, rgba, quality),
        OutputFormat::Jpeg { quality } => encode_jpeg_rgba(size, size, rgba, quality),
        OutputFormat::RawRgba | OutputFormat::RawF32 => Bytes::copy_from_slice(rgba),
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
                            let row_idx = &mut sc.row_buf[..N];
                            for px in 0..N {
                                row_idx[px] = rescale_u8(sc.f32_row[px], inv_range, min_val);
                            }
                            filter_sub_u8_row(
                                row_idx,
                                &mut sc.img_buf[py * stride..(py + 1) * stride],
                            );
                        }
                        encode_indexed_png(N as u32, N as u32, &plte, None, &sc.img_buf[..total])
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
                let stride = 1 + N;
                let total = N * stride;
                if sc.img_buf.len() < total {
                    sc.img_buf.resize(total, 0);
                }
                if sc.row_buf.len() < N {
                    sc.row_buf.resize(N, 0);
                }
                if CLEAN {
                    let plte = build_plte_opaque(lut);
                    for py in 0..N {
                        let row_idx = &mut sc.row_buf[..N];
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let val = sample_val::<BILINEAR, true>(src, u, v);
                            row_idx[px] = rescale_u8(val, inv_range, min_val);
                        });
                        filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                    }
                    encode_indexed_png(N as u32, N as u32, &plte, None, &sc.img_buf[..total])
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
                                row_idx[px] = 1 + (t as u8);
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
                let du = (u_r - u_l) * inv_cell;
                let dv = (v_r - v_l) * inv_cell;
                let mut u = u_l;
                let mut v = v_l;
                let base_x = gx * cell_px;
                for k in 0..cell_px {
                    u_row[base_x + k] = u;
                    v_row[base_x + k] = v;
                    u += du;
                    v += dv;
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
        let mut out_b = [0u32; 4];

        let mut px = 0usize;
        while px + 4 <= N {
            let u = vsubq_f32(vld1q_f32(u_row.as_ptr().add(px)), v_bc);
            let v = vsubq_f32(vld1q_f32(v_row.as_ptr().add(px)), v_br);
            let x0_f = vrndmq_f32(u);
            let y0_f = vrndmq_f32(v);
            let fx = vsubq_f32(u, x0_f);
            let fy = vsubq_f32(v, y0_f);
            vst1q_s32(xs.as_mut_ptr(), vcvtq_s32_f32(x0_f));
            vst1q_s32(ys.as_mut_ptr(), vcvtq_s32_f32(y0_f));

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
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                bilinear_rgb_single_block_row_avx2::<N, T, F>(
                    u_row, v_row, r_ptr, g_ptr, b_ptr, bc, br, stride, inv, mins, row_rgb, &to_f32,
                );
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    for px in 0..N {
        let (x0, fx) = fast_floor_i32(u_row[px] - bc);
        let (y0, fy) = fast_floor_i32(v_row[px] - br);
        let p00 = (y0 as usize) * stride + (x0 as usize);
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
    let mut out_b = [0i32; 8];

    let mut px = 0usize;
    while px + 8 <= N {
        let u = _mm256_sub_ps(_mm256_loadu_ps(u_row.as_ptr().add(px)), v_bc);
        let v = _mm256_sub_ps(_mm256_loadu_ps(v_row.as_ptr().add(px)), v_br);
        let x0_f = _mm256_floor_ps(u);
        let y0_f = _mm256_floor_ps(v);
        let fx = _mm256_sub_ps(u, x0_f);
        let fy = _mm256_sub_ps(v, y0_f);
        _mm256_storeu_si256(xs.as_mut_ptr() as *mut __m256i, _mm256_cvtps_epi32(x0_f));
        _mm256_storeu_si256(ys.as_mut_ptr() as *mut __m256i, _mm256_cvtps_epi32(y0_f));

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

/// Fused clean RGB bilinear scanline sampler:
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
            let mut u_row = [0.0f32; N];
            let mut v_row = [0.0f32; N];
            fill_uv_row(plan, py, &mut u_row, &mut v_row);
            let bc_f = bc as f32;
            let br_f = br as f32;
            let ptr = rgb0.as_ptr();
            let stride3 = st * 3;
            let identity = mins == [0.0, 0.0, 0.0]
                && (inv[0] - 1.0).abs() < 1e-6
                && (inv[1] - 1.0).abs() < 1e-6
                && (inv[2] - 1.0).abs() < 1e-6;
            for px in 0..N {
                let (x0, fx) = fast_floor_i32(u_row[px] - bc_f);
                let (y0, fy) = fast_floor_i32(v_row[px] - br_f);
                let p00 = (y0 as usize) * stride3 + (x0 as usize) * 3;
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
                    ) {
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
                    if BILINEAR {
                        sample_rgb_bilinear_row_clean(plan, py, srcs, inv, mins, row_rgb);
                    } else {
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let r = sample_val::<false, true>(&srcs[0], u, v);
                            let g = sample_val::<false, true>(&srcs[1], u, v);
                            let b = sample_val::<false, true>(&srcs[2], u, v);
                            let off = px * 3;
                            row_rgb[off] = rescale_u8(r, inv[0], mins[0]);
                            row_rgb[off + 1] = rescale_u8(g, inv[1], mins[1]);
                            row_rgb[off + 2] = rescale_u8(b, inv[2], mins[2]);
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
                    } else {
                        for_each_pixel_in_row(plan, py, |px, u, v| {
                            let r = sample_val::<false, true>(&srcs[0], u, v);
                            let g = sample_val::<false, true>(&srcs[1], u, v);
                            let b = sample_val::<false, true>(&srcs[2], u, v);
                            let off = row_off + px * 3;
                            buf[off] = rescale_u8(r, inv[0], mins[0]);
                            buf[off + 1] = rescale_u8(g, inv[1], mins[1]);
                            buf[off + 2] = rescale_u8(b, inv[2], mins[2]);
                        });
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
    }
}

fn render_rgb_cubic<const N: usize>(
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
            let mut out = BytesMut::with_capacity(N * N * 12);
            for b in 0..3 {
                for py in 0..N {
                    for_each_pixel_in_row(plan, py, |_px, u, v| {
                        out.put_slice(&sample_cubic(&srcs[b], u, v).to_le_bytes());
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
                    let r = sample_cubic(&srcs[0], u, v);
                    let g = sample_cubic(&srcs[1], u, v);
                    let b = sample_cubic(&srcs[2], u, v);
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
                    let r = sample_cubic(&srcs[0], u, v);
                    let g = sample_cubic(&srcs[1], u, v);
                    let b = sample_cubic(&srcs[2], u, v);
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
                            1 + (t as u8)
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
    }
}

/// Evaluate an arbitrary compiled TiTiler `?expression=` bytecode program across any number of
/// bands using the scanline-vectorized [`ExprProgram::eval_row`] engine.
pub fn render_expr_vm<const N: usize>(
    plan: &CoordPlan<N>,
    srcs: &[BandSource<'_>],
    prog: &ExprProgram,
    style: &SingleBandStyle,
) -> Bytes {
    if srcs.is_empty() || srcs.iter().any(|s| s.all_nodata) {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let nb = srcs.len().min(16);
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let all_clean = plan.all_in_bounds() && srcs[..nb].iter().all(|s| s.is_clean_inbounds(c0, c1, r0, r1));
    let range = (style.rescale_max - style.rescale_min).max(1e-12);
    let inv_range = 255.0 / range;
    let inv_range_254 = 254.0 / range;
    let min_val = style.rescale_min;

    let sample_row = |py: usize, band_rows: &mut [[f32; N]]| {
        match (style.resample, all_clean) {
            (ResampleMode::Bilinear, true) => {
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
        }
    };

    let mut band_rows = vec![[0.0f32; N]; nb];
    let mut out_row = [0.0f32; N];

    match style.format {
        OutputFormat::RawF32 => {
            let mut out = BytesMut::with_capacity(N * N * 4);
            for py in 0..N {
                sample_row(py, &mut band_rows);
                prog.eval_row::<N>(&band_rows, &mut out_row);
                for &v in &out_row {
                    out.put_slice(&v.to_le_bytes());
                }
            }
            out.freeze()
        }
        OutputFormat::Png => {
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
                    prog.eval_row::<N>(&band_rows, &mut out_row);
                    let row_idx = &mut sc.row_buf[..N];
                    for px in 0..N {
                        let val = out_row[px];
                        row_idx[px] = if val.is_nan() {
                            0
                        } else {
                            let t = ((val - min_val) * inv_range_254).clamp(0.0, 254.0);
                            1 + (t as u8)
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
                let buf_u32 = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u32, N * N)
                };
                for py in 0..N {
                    sample_row(py, &mut band_rows);
                    prog.eval_row::<N>(&band_rows, &mut out_row);
                    let row_off = py * N;
                    for px in 0..N {
                        let val = out_row[px];
                        buf_u32[row_off + px] = if val.is_nan() {
                            0
                        } else {
                            let idx = rescale_u8(val, inv_range, min_val) as usize;
                            unsafe { *lut.rgba_u32.get_unchecked(idx) }
                        };
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

    let encode_px = |v: f32| -> [u8; 3] {
        match encoding {
            TerrainEncoding::Terrarium => encode_terrarium_rgb(v),
            TerrainEncoding::TerrainRgb => encode_terrain_rgb(v),
        }
    };

    let sample_px = |u: f32, v: f32| -> f32 {
        match (resample, clean) {
            (ResampleMode::Bilinear, true) => sample_bilinear::<true>(src, u, v),
            (ResampleMode::Bilinear, false) => sample_bilinear::<false>(src, u, v),
            (ResampleMode::Nearest, true) => sample_nearest::<true>(src, u, v),
            (ResampleMode::Nearest, false) => sample_nearest::<false>(src, u, v),
            (ResampleMode::Cubic, _) => sample_cubic(src, u, v),
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
                for py in 0..N {
                    let row_rgb = &mut sc.row_buf[..N * 3];
                    for_each_pixel_in_row(plan, py, |px, u, v| {
                        let [r, g, b] = encode_px(sample_px(u, v));
                        let off = px * 3;
                        row_rgb[off] = r;
                        row_rgb[off + 1] = g;
                        row_rgb[off + 2] = b;
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
        OutputFormat::RawF32 => {
            let mut out = BytesMut::with_capacity(N * N * 4);
            for py in 0..N {
                for_each_pixel_in_row(plan, py, |_px, u, v| {
                    out.put_slice(&sample_px(u, v).to_le_bytes());
                });
            }
            out.freeze()
        }
        OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
            let mut rgba = vec![0u8; N * N * 4];
            for py in 0..N {
                let row_off = py * N * 4;
                for_each_pixel_in_row(plan, py, |px, u, v| {
                    let elev = sample_px(u, v);
                    let off = row_off + px * 4;
                    if !elev.is_nan() {
                        let [r, g, b] = encode_px(elev);
                        rgba[off] = r;
                        rgba[off + 1] = g;
                        rgba[off + 2] = b;
                        rgba[off + 3] = 255;
                    }
                });
            }
            encode_rgba_tile(&rgba, N as u32, clean, format)
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
}

/// Evaluate a 3×3 Horn stencil centered at `(z_nw..z_se)` with ground spacing `pixel_size_m`.
#[inline(always)]
fn eval_horn_3x3(
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
    let dz_dy = ((sw + 2.0 * s + se) - (nw + 2.0 * n + ne)) * inv_8dx;

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
        return;
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
            return;
        }
    }

    #[allow(unreachable_code)]
    for px in 0..N {
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

    let sample_px = |u: f32, v: f32| -> f32 {
        match (style.resample, clean) {
            (ResampleMode::Bilinear, true) => sample_bilinear::<true>(src, u, v),
            (ResampleMode::Bilinear, false) => sample_bilinear::<false>(src, u, v),
            (ResampleMode::Nearest, true) => sample_nearest::<true>(src, u, v),
            (ResampleMode::Nearest, false) => sample_nearest::<false>(src, u, v),
            (ResampleMode::Cubic, _) => sample_cubic(src, u, v),
        }
    };
    // Sub-pixel bounds-checked sampler for the 1-pixel extrapolated border so tile edges
    // match neighboring tiles' interior pixels with zero staircase seam.
    let sample_border = |u: f32, v: f32| -> f32 {
        match style.resample {
            ResampleMode::Bilinear => sample_bilinear::<false>(src, u, v),
            ResampleMode::Nearest => sample_nearest::<false>(src, u, v),
            ResampleMode::Cubic => sample_cubic(src, u, v),
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
            for px in 0..N {
                elev[row_off + px + 1] = sample_px(u_row[px], v_row[px]);
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

        let inv_8dx = 1.0 / (8.0 * pixel_size_m.max(1e-3));
        let inv_range = 255.0 / (style.rescale_max - style.rescale_min).max(1e-12);
        let min_val = style.rescale_min;
        let lut = style.colormap.lut();

        match style.format {
            OutputFormat::RawF32 => {
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
                        } else {
                            let idx = rescale_u8(val, inv_range, min_val);
                            row_idx[px] = if clean && !matches!(alg, DemAlgorithm::Aspect) {
                                idx
                            } else {
                                idx.max(1)
                            };
                        }
                    }
                    filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                }
                if !has_nodata && clean && !matches!(alg, DemAlgorithm::Aspect) {
                    let plte = build_plte_opaque(lut);
                    encode_indexed_png(N as u32, N as u32, &plte, None, &sc.img_buf[..total])
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
}
