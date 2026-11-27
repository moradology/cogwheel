use cogzilla_tiff::ByteOrder;

use crate::slab::{AlignedF32Slab, BandData, BlockStats, DecodedBand};

#[inline(always)]
fn finalize_stats(any_valid: bool, any_nodata: bool, min: f32, max: f32) -> BlockStats {
    if !any_valid {
        BlockStats::ALL_NODATA
    } else {
        BlockStats {
            all_nodata: false,
            no_nodata: !any_nodata,
            min,
            max,
        }
    }
}

#[inline(always)]
fn check_val(v: f32, finite_nodata: Option<f32>) -> (f32, bool) {
    if v.is_nan() {
        (f32::NAN, true)
    } else if let Some(nd) = finite_nodata {
        if v == nd {
            (f32::NAN, true)
        } else {
            (v, false)
        }
    } else {
        (v, false)
    }
}

// ============================================================================
// ARM64 NEON Kernels (Apple Silicon / AWS Graviton)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefix_sum_stride1_neon(row: &mut [u8]) {
    use std::arch::aarch64::*;
    let len = row.len();
    let ptr = row.as_mut_ptr();
    let zero = unsafe { vdupq_n_u8(0) };
    let mut carry = zero;
    let mut i = 0usize;

    while i + 16 <= len {
        unsafe {
            let mut v = vld1q_u8(ptr.add(i));
            v = vaddq_u8(v, vextq_u8::<15>(zero, v));
            v = vaddq_u8(v, vextq_u8::<14>(zero, v));
            v = vaddq_u8(v, vextq_u8::<12>(zero, v));
            v = vaddq_u8(v, vextq_u8::<8>(zero, v));
            v = vaddq_u8(v, carry);
            vst1q_u8(ptr.add(i), v);
            carry = vdupq_laneq_u8::<15>(v);
        }
        i += 16;
    }

    if i < len {
        let mut acc = if i > 0 { row[i - 1] } else { 0 };
        for b in &mut row[i..] {
            acc = acc.wrapping_add(*b);
            *b = acc;
        }
    }
}

/// 16-lane NEON Kogge-Stone prefix sum for stride-3 interleaved byte streams (e.g. 3-band chunky
/// float32 Predictor 3 rows) using `vld3q_u8`/`vst3q_u8` to deinterleave all 3 channels in hardware.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefix_sum_stride3_neon(row: &mut [u8]) {
    use std::arch::aarch64::*;
    let len = row.len();
    let ptr = row.as_mut_ptr();
    let zero = unsafe { vdupq_n_u8(0) };
    let mut c0 = zero;
    let mut c1 = zero;
    let mut c2 = zero;
    let mut i = 0usize;

    while i + 48 <= len {
        unsafe {
            let mut v = vld3q_u8(ptr.add(i));
            macro_rules! kstone {
                ($ch:expr, $carry:expr) => {
                    $ch = vaddq_u8($ch, vextq_u8::<15>(zero, $ch));
                    $ch = vaddq_u8($ch, vextq_u8::<14>(zero, $ch));
                    $ch = vaddq_u8($ch, vextq_u8::<12>(zero, $ch));
                    $ch = vaddq_u8($ch, vextq_u8::<8>(zero, $ch));
                    $ch = vaddq_u8($ch, $carry);
                    $carry = vdupq_laneq_u8::<15>($ch);
                };
            }
            kstone!(v.0, c0);
            kstone!(v.1, c1);
            kstone!(v.2, c2);
            vst3q_u8(ptr.add(i), v);
        }
        i += 48;
    }

    if i < len {
        let mut idx = i.max(3);
        while idx < len {
            row[idx] = row[idx].wrapping_add(row[idx - 3]);
            idx += 1;
        }
    }
}

/// Single-pass NEON 3-band chunky `uint8` horizontal differencing (optional) + min/max/nodata stats.
/// Uses `vld3q_u8`/`vst3q_u8` to eliminate both the scalar prefix-sum and the second pass over RAM.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reconstruct_u8_rgb3_neon(
    slice: &mut [u8],
    w: usize,
    h: usize,
    horizontal_diff: bool,
    nd_u8: Option<u8>,
    any_valid: &mut [bool; 3],
    any_nodata: &mut [bool; 3],
    min_u8: &mut [u8; 3],
    max_u8: &mut [u8; 3],
) {
    use std::arch::aarch64::*;
    let zero = unsafe { vdupq_n_u8(0) };
    let v255 = unsafe { vdupq_n_u8(255) };
    let mut vmin = [v255; 3];
    let mut vmax = [zero; 3];
    let mut v_any_valid = [zero; 3];
    let mut v_any_nodata = [zero; 3];
    let v_nd = unsafe { vdupq_n_u8(nd_u8.unwrap_or(0)) };
    let has_nd = nd_u8.is_some();

    for y in 0..h {
        let row = &mut slice[y * w * 3..(y + 1) * w * 3];
        let ptr = row.as_mut_ptr();
        let mut c0 = zero;
        let mut c1 = zero;
        let mut c2 = zero;
        let mut x = 0usize;

        while x + 16 <= w {
            unsafe {
                let mut v = vld3q_u8(ptr.add(x * 3));
                if horizontal_diff {
                    macro_rules! kstone {
                        ($ch:expr, $carry:expr) => {
                            $ch = vaddq_u8($ch, vextq_u8::<15>(zero, $ch));
                            $ch = vaddq_u8($ch, vextq_u8::<14>(zero, $ch));
                            $ch = vaddq_u8($ch, vextq_u8::<12>(zero, $ch));
                            $ch = vaddq_u8($ch, vextq_u8::<8>(zero, $ch));
                            $ch = vaddq_u8($ch, $carry);
                            $carry = vdupq_laneq_u8::<15>($ch);
                        };
                    }
                    kstone!(v.0, c0);
                    kstone!(v.1, c1);
                    kstone!(v.2, c2);
                    vst3q_u8(ptr.add(x * 3), v);
                }

                let chs = [v.0, v.1, v.2];
                if has_nd {
                    for b in 0..3 {
                        let is_nd = vceqq_u8(chs[b], v_nd);
                        let is_val = vmvnq_u8(is_nd);
                        v_any_nodata[b] = vorrq_u8(v_any_nodata[b], is_nd);
                        v_any_valid[b] = vorrq_u8(v_any_valid[b], is_val);
                        vmin[b] = vminq_u8(vmin[b], vbslq_u8(is_val, chs[b], v255));
                        vmax[b] = vmaxq_u8(vmax[b], vbslq_u8(is_val, chs[b], zero));
                    }
                } else {
                    for b in 0..3 {
                        vmin[b] = vminq_u8(vmin[b], chs[b]);
                        vmax[b] = vmaxq_u8(vmax[b], chs[b]);
                    }
                }
            }
            x += 16;
        }

        if x < w {
            let mut acc = if horizontal_diff && x > 0 {
                [row[(x - 1) * 3], row[(x - 1) * 3 + 1], row[(x - 1) * 3 + 2]]
            } else {
                [0u8; 3]
            };
            for px in row[x * 3..].chunks_exact_mut(3) {
                for b in 0..3 {
                    let v = if horizontal_diff {
                        acc[b] = acc[b].wrapping_add(px[b]);
                        px[b] = acc[b];
                        acc[b]
                    } else {
                        px[b]
                    };
                    if nd_u8 == Some(v) {
                        any_nodata[b] = true;
                    } else {
                        any_valid[b] = true;
                        if v < min_u8[b] { min_u8[b] = v; }
                        if v > max_u8[b] { max_u8[b] = v; }
                    }
                }
            }
        }
    }

    if w >= 16 && h > 0 {
        unsafe {
            for b in 0..3 {
                if has_nd {
                    if vmaxvq_u8(v_any_nodata[b]) != 0 {
                        any_nodata[b] = true;
                    }
                    if vmaxvq_u8(v_any_valid[b]) != 0 {
                        any_valid[b] = true;
                        min_u8[b] = min_u8[b].min(vminvq_u8(vmin[b]));
                        max_u8[b] = max_u8[b].max(vmaxvq_u8(vmax[b]));
                    }
                } else {
                    any_valid[b] = true;
                    min_u8[b] = min_u8[b].min(vminvq_u8(vmin[b]));
                    max_u8[b] = max_u8[b].max(vmaxvq_u8(vmax[b]));
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_and_store_16_f32_neon(
    b0: std::arch::aarch64::uint8x16_t, // LSB (p3)
    b1: std::arch::aarch64::uint8x16_t, // byte 1 (p2)
    b2: std::arch::aarch64::uint8x16_t, // byte 2 (p1)
    b3: std::arch::aarch64::uint8x16_t, // MSB (p0)
    dst_f32: *mut f32,
    has_finite_nd: bool,
    v_nd: std::arch::aarch64::float32x4_t,
    v_nan: std::arch::aarch64::float32x4_t,
    v_min: &mut std::arch::aarch64::float32x4_t,
    v_max: &mut std::arch::aarch64::float32x4_t,
    v_any_valid: &mut std::arch::aarch64::uint32x4_t,
    v_any_nodata: &mut std::arch::aarch64::uint32x4_t,
) {
    use std::arch::aarch64::*;
    unsafe {
        let lo01 = vzip1q_u8(b0, b1);
        let hi01 = vzip2q_u8(b0, b1);
        let lo23 = vzip1q_u8(b2, b3);
        let hi23 = vzip2q_u8(b2, b3);

        let mut f0 = vreinterpretq_f32_u16(vzip1q_u16(
            vreinterpretq_u16_u8(lo01),
            vreinterpretq_u16_u8(lo23),
        ));
        let mut f1 = vreinterpretq_f32_u16(vzip2q_u16(
            vreinterpretq_u16_u8(lo01),
            vreinterpretq_u16_u8(lo23),
        ));
        let mut f2 = vreinterpretq_f32_u16(vzip1q_u16(
            vreinterpretq_u16_u8(hi01),
            vreinterpretq_u16_u8(hi23),
        ));
        let mut f3 = vreinterpretq_f32_u16(vzip2q_u16(
            vreinterpretq_u16_u8(hi01),
            vreinterpretq_u16_u8(hi23),
        ));

        macro_rules! proc_v {
            ($fv:expr) => {{
                let not_nan = vceqq_f32($fv, $fv);
                let valid = if has_finite_nd {
                    vbicq_u32(not_nan, vceqq_f32($fv, v_nd))
                } else {
                    not_nan
                };
                *v_any_valid = vorrq_u32(*v_any_valid, valid);
                *v_any_nodata = vorrq_u32(*v_any_nodata, vmvnq_u32(valid));
                if has_finite_nd {
                    $fv = vbslq_f32(valid, $fv, v_nan);
                }
                *v_min = vminnmq_f32(*v_min, $fv);
                *v_max = vmaxnmq_f32(*v_max, $fv);
            }};
        }

        proc_v!(f0);
        proc_v!(f1);
        proc_v!(f2);
        proc_v!(f3);

        vst1q_f32(dst_f32, f0);
        vst1q_f32(dst_f32.add(4), f1);
        vst1q_f32(dst_f32.add(8), f2);
        vst1q_f32(dst_f32.add(12), f3);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unshuffle_row_stride1_neon(
    p0: &[u8],
    p1: &[u8],
    p2: &[u8],
    p3: &[u8],
    dst: &mut [f32],
    finite_nodata: Option<f32>,
    any_valid: &mut bool,
    any_nodata: &mut bool,
    min_val: &mut f32,
    max_val: &mut f32,
) {
    use std::arch::aarch64::*;
    let n = dst.len();
    let mut i = 0usize;

    let mut v_min = unsafe { vdupq_n_f32(f32::INFINITY) };
    let mut v_max = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    let mut v_any_valid = unsafe { vdupq_n_u32(0) };
    let mut v_any_nodata = unsafe { vdupq_n_u32(0) };
    let v_nan = unsafe { vdupq_n_f32(f32::NAN) };
    let has_finite_nd = finite_nodata.is_some();
    let v_nd = unsafe { vdupq_n_f32(finite_nodata.unwrap_or(0.0)) };

    while i + 16 <= n {
        unsafe {
            let b0 = vld1q_u8(p3.as_ptr().add(i));
            let b1 = vld1q_u8(p2.as_ptr().add(i));
            let b2 = vld1q_u8(p1.as_ptr().add(i));
            let b3 = vld1q_u8(p0.as_ptr().add(i));
            transpose_and_store_16_f32_neon(
                b0,
                b1,
                b2,
                b3,
                dst.as_mut_ptr().add(i),
                has_finite_nd,
                v_nd,
                v_nan,
                &mut v_min,
                &mut v_max,
                &mut v_any_valid,
                &mut v_any_nodata,
            );
        }
        i += 16;
    }

    unsafe {
        if vmaxvq_u32(v_any_valid) != 0 {
            *any_valid = true;
            let mn = vminnmvq_f32(v_min);
            let mx = vmaxnmvq_f32(v_max);
            if mn < *min_val {
                *min_val = mn;
            }
            if mx > *max_val {
                *max_val = mx;
            }
        }
        if vmaxvq_u32(v_any_nodata) != 0 {
            *any_nodata = true;
        }
    }

    while i < n {
        let raw = f32::from_le_bytes([p3[i], p2[i], p1[i], p0[i]]);
        let (v, is_nd) = check_val(raw, finite_nodata);
        dst[i] = v;
        if is_nd {
            *any_nodata = true;
        } else {
            *any_valid = true;
            if v < *min_val {
                *min_val = v;
            }
            if v > *max_val {
                *max_val = v;
            }
        }
        i += 1;
    }
}

/// Hardware-accelerated 3-band chunky Predictor-3 unshuffle + planar deinterleave on ARM64 NEON
/// using `vld3q_u8` (deinterleaves 16 RGB pixels / 48 floats / 192 bytes per iteration in registers!).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unshuffle_row_stride3_neon(
    p0: &[u8],
    p1: &[u8],
    p2: &[u8],
    p3: &[u8],
    dst0: &mut [f32],
    dst1: &mut [f32],
    dst2: &mut [f32],
    finite_nodata: Option<f32>,
    any_valid: &mut [bool; 3],
    any_nodata: &mut [bool; 3],
    min_val: &mut [f32; 3],
    max_val: &mut [f32; 3],
) {
    use std::arch::aarch64::*;
    let w = dst0.len();
    let mut x = 0usize;

    let v_nan = unsafe { vdupq_n_f32(f32::NAN) };
    let has_finite_nd = finite_nodata.is_some();
    let v_nd = unsafe { vdupq_n_f32(finite_nodata.unwrap_or(0.0)) };

    let mut v_min = unsafe { [vdupq_n_f32(f32::INFINITY); 3] };
    let mut v_max = unsafe { [vdupq_n_f32(f32::NEG_INFINITY); 3] };
    let mut v_av = unsafe { [vdupq_n_u32(0); 3] };
    let mut v_an = unsafe { [vdupq_n_u32(0); 3] };

    while x + 16 <= w {
        let byte_off = x * 3;
        unsafe {
            // 4 hardware 3-way deinterleaving loads (48 bytes each)
            let d3 = vld3q_u8(p3.as_ptr().add(byte_off));
            let d2 = vld3q_u8(p2.as_ptr().add(byte_off));
            let d1 = vld3q_u8(p1.as_ptr().add(byte_off));
            let d0 = vld3q_u8(p0.as_ptr().add(byte_off));

            transpose_and_store_16_f32_neon(
                d3.0, d2.0, d1.0, d0.0,
                dst0.as_mut_ptr().add(x),
                has_finite_nd, v_nd, v_nan,
                &mut v_min[0], &mut v_max[0], &mut v_av[0], &mut v_an[0],
            );
            transpose_and_store_16_f32_neon(
                d3.1, d2.1, d1.1, d0.1,
                dst1.as_mut_ptr().add(x),
                has_finite_nd, v_nd, v_nan,
                &mut v_min[1], &mut v_max[1], &mut v_av[1], &mut v_an[1],
            );
            transpose_and_store_16_f32_neon(
                d3.2, d2.2, d1.2, d0.2,
                dst2.as_mut_ptr().add(x),
                has_finite_nd, v_nd, v_nan,
                &mut v_min[2], &mut v_max[2], &mut v_av[2], &mut v_an[2],
            );
        }
        x += 16;
    }

    unsafe {
        for b in 0..3 {
            if vmaxvq_u32(v_av[b]) != 0 {
                any_valid[b] = true;
                let mn = vminnmvq_f32(v_min[b]);
                let mx = vmaxnmvq_f32(v_max[b]);
                if mn < min_val[b] {
                    min_val[b] = mn;
                }
                if mx > max_val[b] {
                    max_val[b] = mx;
                }
            }
            if vmaxvq_u32(v_an[b]) != 0 {
                any_nodata[b] = true;
            }
        }
    }

    while x < w {
        let base = x * 3;
        for (b, dst) in [(0, &mut *dst0), (1, &mut *dst1), (2, &mut *dst2)] {
            let idx = base + b;
            let raw = f32::from_le_bytes([p3[idx], p2[idx], p1[idx], p0[idx]]);
            let (v, is_nd) = check_val(raw, finite_nodata);
            dst[x] = v;
            if is_nd {
                any_nodata[b] = true;
            } else {
                any_valid[b] = true;
                if v < min_val[b] {
                    min_val[b] = v;
                }
                if v > max_val[b] {
                    max_val[b] = v;
                }
            }
        }
        x += 1;
    }
}

/// SIMD 4-lane `u32` horizontal prefix-sum -> `f32` + nodata/stats on ARM64 NEON (Predictor 2).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reconstruct_predictor2_le_neon(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    dst: &mut [f32],
    finite_nodata: Option<f32>,
) -> BlockStats {
    use std::arch::aarch64::*;
    let w = tile_w as usize;
    let h = tile_h as usize;
    let row_bytes = w * 4;
    let zero = unsafe { vdupq_n_u32(0) };
    let v_nan = unsafe { vdupq_n_f32(f32::NAN) };
    let has_finite_nd = finite_nodata.is_some();
    let v_nd = unsafe { vdupq_n_f32(finite_nodata.unwrap_or(0.0)) };

    let mut v_min = unsafe { vdupq_n_f32(f32::INFINITY) };
    let mut v_max = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    let mut v_av = unsafe { vdupq_n_u32(0) };
    let mut v_an = unsafe { vdupq_n_u32(0) };
    let mut any_valid = false;
    let mut any_nodata = false;
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for y in 0..h {
        let row_src = &raw[y * row_bytes..(y + 1) * row_bytes];
        let row_dst = &mut dst[y * w..(y + 1) * w];
        let mut carry = zero;
        let mut i = 0usize;

        while i + 4 <= w {
            unsafe {
                let mut v = vld1q_u32(row_src.as_ptr().add(i * 4) as *const u32);
                v = vaddq_u32(v, vextq_u32::<3>(zero, v));
                v = vaddq_u32(v, vextq_u32::<2>(zero, v));
                v = vaddq_u32(v, carry);
                carry = vdupq_laneq_u32::<3>(v);

                let mut fv = vreinterpretq_f32_u32(v);
                let not_nan = vceqq_f32(fv, fv);
                let valid = if has_finite_nd {
                    vbicq_u32(not_nan, vceqq_f32(fv, v_nd))
                } else {
                    not_nan
                };
                v_av = vorrq_u32(v_av, valid);
                v_an = vorrq_u32(v_an, vmvnq_u32(valid));
                if has_finite_nd {
                    fv = vbslq_f32(valid, fv, v_nan);
                }
                v_min = vminnmq_f32(v_min, fv);
                v_max = vmaxnmq_f32(v_max, fv);
                vst1q_f32(row_dst.as_mut_ptr().add(i), fv);
            }
            i += 4;
        }

        if i < w {
            let mut acc: u32 = if i > 0 {
                unsafe { vgetq_lane_u32::<3>(carry) }
            } else {
                0
            };
            while i < w {
                let off = i * 4;
                let delta = u32::from_le_bytes([
                    row_src[off],
                    row_src[off + 1],
                    row_src[off + 2],
                    row_src[off + 3],
                ]);
                acc = acc.wrapping_add(delta);
                let (v, is_nd) = check_val(f32::from_bits(acc), finite_nodata);
                row_dst[i] = v;
                if is_nd {
                    any_nodata = true;
                } else {
                    any_valid = true;
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                }
                i += 1;
            }
        }
    }

    unsafe {
        if vmaxvq_u32(v_av) != 0 {
            any_valid = true;
            let mn = vminnmvq_f32(v_min);
            let mx = vmaxnmvq_f32(v_max);
            if mn < min_val { min_val = mn; }
            if mx > max_val { max_val = mx; }
        }
        if vmaxvq_u32(v_an) != 0 {
            any_nodata = true;
        }
    }

    finalize_stats(any_valid, any_nodata, min_val, max_val)
}

// ============================================================================
// x86_64 AVX2 Kernels (AMD EPYC / Intel Xeon Cloud Servers)
// ============================================================================

// SAFETY: Caller checks `is_x86_feature_detected!("avx2")`; SIMD loads/stores stay within `row.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn prefix_sum_stride1_avx2(row: &mut [u8]) {
    use std::arch::x86_64::*;
    let len = row.len();
    let ptr = row.as_mut_ptr();
    let mut carry = _mm_setzero_si128();
    let bcast_last = _mm_set1_epi8(15);
    let mut i = 0usize;

    while i + 16 <= len {
        let mut v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
        v = _mm_add_epi8(v, _mm_slli_si128::<1>(v));
        v = _mm_add_epi8(v, _mm_slli_si128::<2>(v));
        v = _mm_add_epi8(v, _mm_slli_si128::<4>(v));
        v = _mm_add_epi8(v, _mm_slli_si128::<8>(v));
        v = _mm_add_epi8(v, carry);
        _mm_storeu_si128(ptr.add(i) as *mut __m128i, v);
        carry = _mm_shuffle_epi8(v, bcast_last);
        i += 16;
    }
    if i < len {
        let mut acc = if i > 0 { row[i - 1] } else { 0 };
        for b in &mut row[i..] {
            acc = acc.wrapping_add(*b);
            *b = acc;
        }
    }
}

/// 12-byte (4-triplet) AVX2/SSSE3 Kogge-Stone prefix sum for stride-3 interleaved byte streams
/// (3-band chunky float32 Predictor 3 rows on x86_64 cloud servers).
// SAFETY: Caller checks `is_x86_feature_detected!("avx2")`; SIMD loads/stores stay within `row.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn prefix_sum_stride3_avx2(row: &mut [u8]) {
    use std::arch::x86_64::*;
    let len = row.len();
    let ptr = row.as_mut_ptr();
    let mut carry = _mm_setzero_si128();
    // Broadcast lanes [9, 10, 11] across all four 3-byte triplets in lanes 0..12.
    let bcast_last3 = _mm_setr_epi8(9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11, -1, -1, -1, -1);
    let mut i = 0usize;

    while i + 16 <= len {
        let mut v = unsafe { _mm_loadu_si128(ptr.add(i) as *const __m128i) };
        v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
        v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
        v = _mm_add_epi8(v, carry);
        // Write exact 12 bytes (8 + 4) so bytes i+12..i+16 are never clobbered before next load.
        unsafe {
            _mm_storel_epi64(ptr.add(i) as *mut __m128i, v);
            (ptr.add(i + 8) as *mut i32).write_unaligned(_mm_extract_epi32::<2>(v));
        }
        carry = _mm_shuffle_epi8(v, bcast_last3);
        i += 12;
    }

    if i < len {
        let mut idx = i.max(3);
        while idx < len {
            row[idx] = row[idx].wrapping_add(row[idx - 3]);
            idx += 1;
        }
    }
}

/// Single-pass AVX2 3-band chunky `uint8` horizontal differencing (optional) + min/max/nodata stats.
// SAFETY: Caller checks `is_x86_feature_detected!("avx2")`; SIMD loops stay within input/output slice bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reconstruct_u8_rgb3_avx2(
    slice: &mut [u8],
    w: usize,
    h: usize,
    horizontal_diff: bool,
    nd_u8: Option<u8>,
    any_valid: &mut [bool; 3],
    any_nodata: &mut [bool; 3],
    min_u8: &mut [u8; 3],
    max_u8: &mut [u8; 3],
) {
    use std::arch::x86_64::*;
    let zero = _mm_setzero_si128();
    let v255 = _mm_set1_epi8(-1i8);
    let bcast_last3 = _mm_setr_epi8(9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11, -1, -1, -1, -1);
    let mut vmin = v255;
    let mut vmax = zero;
    let mut v_any_valid = zero;
    let mut v_any_nodata = zero;
    let has_nd = nd_u8.is_some();
    let v_nd = _mm_set1_epi8(nd_u8.unwrap_or(0) as i8);
    let mut simd_ran = false;

    for y in 0..h {
        let row = &mut slice[y * w * 3..(y + 1) * w * 3];
        let row_bytes = w * 3;
        let ptr = row.as_mut_ptr();
        let mut carry = zero;
        let mut byte_i = 0usize;

        while byte_i + 16 <= row_bytes {
            simd_ran = true;
            let mut v = unsafe { _mm_loadu_si128(ptr.add(byte_i) as *const __m128i) };
            if horizontal_diff {
                v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                v = _mm_add_epi8(v, carry);
                unsafe {
                    _mm_storel_epi64(ptr.add(byte_i) as *mut __m128i, v);
                    (ptr.add(byte_i + 8) as *mut i32).write_unaligned(_mm_extract_epi32::<2>(v));
                }
                carry = _mm_shuffle_epi8(v, bcast_last3);
            }
            if has_nd {
                let is_nd = _mm_cmpeq_epi8(v, v_nd);
                let is_val = _mm_andnot_si128(is_nd, v255);
                v_any_nodata = _mm_or_si128(v_any_nodata, is_nd);
                v_any_valid = _mm_or_si128(v_any_valid, is_val);
                vmin = _mm_min_epu8(vmin, _mm_blendv_epi8(v255, v, is_val));
                vmax = _mm_max_epu8(vmax, _mm_blendv_epi8(zero, v, is_val));
            } else {
                vmin = _mm_min_epu8(vmin, v);
                vmax = _mm_max_epu8(vmax, v);
            }
            byte_i += 12;
        }

        if byte_i < row_bytes {
            let mut acc = if horizontal_diff && byte_i >= 3 {
                [row[byte_i - 3], row[byte_i - 2], row[byte_i - 1]]
            } else {
                [0u8; 3]
            };
            for px in row[byte_i..].chunks_exact_mut(3) {
                for b in 0..3 {
                    let v = if horizontal_diff {
                        acc[b] = acc[b].wrapping_add(px[b]);
                        px[b] = acc[b];
                        acc[b]
                    } else {
                        px[b]
                    };
                    if nd_u8 == Some(v) {
                        any_nodata[b] = true;
                    } else {
                        any_valid[b] = true;
                        if v < min_u8[b] { min_u8[b] = v; }
                        if v > max_u8[b] { max_u8[b] = v; }
                    }
                }
            }
        }
    }

    if simd_ran {
        let mut mn_arr = [0u8; 16];
        let mut mx_arr = [0u8; 16];
        let mut av_arr = [0u8; 16];
        let mut an_arr = [0u8; 16];
        unsafe {
            _mm_storeu_si128(mn_arr.as_mut_ptr() as *mut __m128i, vmin);
            _mm_storeu_si128(mx_arr.as_mut_ptr() as *mut __m128i, vmax);
            _mm_storeu_si128(av_arr.as_mut_ptr() as *mut __m128i, v_any_valid);
            _mm_storeu_si128(an_arr.as_mut_ptr() as *mut __m128i, v_any_nodata);
        }
        for k in 0..12 {
            let b = k % 3;
            if has_nd {
                if an_arr[k] != 0 {
                    any_nodata[b] = true;
                }
                if av_arr[k] != 0 {
                    any_valid[b] = true;
                    min_u8[b] = min_u8[b].min(mn_arr[k]);
                    max_u8[b] = max_u8[b].max(mx_arr[k]);
                }
            } else {
                any_valid[b] = true;
                min_u8[b] = min_u8[b].min(mn_arr[k]);
                max_u8[b] = max_u8[b].max(mx_arr[k]);
            }
        }
    }
}

// SAFETY: Caller checks `is_x86_feature_detected!("avx2")`; SIMD loads/stores stay within row/out slice bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn unshuffle_row_stride1_avx2(
    p0: &[u8],
    p1: &[u8],
    p2: &[u8],
    p3: &[u8],
    dst: &mut [f32],
    finite_nodata: Option<f32>,
    any_valid: &mut bool,
    any_nodata: &mut bool,
    min_val: &mut f32,
    max_val: &mut f32,
) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let mut i = 0usize;

    let mut v_min = _mm256_set1_ps(f32::INFINITY);
    let mut v_max = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut v_any_valid = _mm256_setzero_ps();
    let mut v_any_nodata = _mm256_setzero_ps();
    let v_nan = _mm256_set1_ps(f32::NAN);
    let has_finite_nd = finite_nodata.is_some();
    let v_nd = _mm256_set1_ps(finite_nodata.unwrap_or(0.0));

    while i + 32 <= n {
        // Load 32 bytes from each of the 4 planes -> 32 f32s (128 bytes)
        let b0 = _mm256_loadu_si256(p3.as_ptr().add(i) as *const __m256i);
        let b1 = _mm256_loadu_si256(p2.as_ptr().add(i) as *const __m256i);
        let b2 = _mm256_loadu_si256(p1.as_ptr().add(i) as *const __m256i);
        let b3 = _mm256_loadu_si256(p0.as_ptr().add(i) as *const __m256i);

        // Permute lanes to keep linear order after in-lane AVX2 unpacks
        let lo01 = _mm256_unpacklo_epi8(b0, b1);
        let hi01 = _mm256_unpackhi_epi8(b0, b1);
        let lo23 = _mm256_unpacklo_epi8(b2, b3);
        let hi23 = _mm256_unpackhi_epi8(b2, b3);

        let t0 = _mm256_unpacklo_epi16(lo01, lo23); // lanes: [0..3 | 16..19]
        let t1 = _mm256_unpackhi_epi16(lo01, lo23); // lanes: [4..7 | 20..23]
        let t2 = _mm256_unpacklo_epi16(hi01, hi23); // lanes: [8..11 | 24..27]
        let t3 = _mm256_unpackhi_epi16(hi01, hi23); // lanes: [12..15 | 28..31]

        let mut f0 = _mm256_castsi256_ps(_mm256_permute2x128_si256::<0x20>(t0, t1)); // [0..7]
        let mut f1 = _mm256_castsi256_ps(_mm256_permute2x128_si256::<0x20>(t2, t3)); // [8..15]
        let mut f2 = _mm256_castsi256_ps(_mm256_permute2x128_si256::<0x31>(t0, t1)); // [16..23]
        let mut f3 = _mm256_castsi256_ps(_mm256_permute2x128_si256::<0x31>(t2, t3)); // [24..31]

        macro_rules! proc_avx {
            ($fv:expr) => {{
                let not_nan = _mm256_cmp_ps::<_CMP_ORD_Q>($fv, $fv);
                let valid = if has_finite_nd {
                    _mm256_andnot_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>($fv, v_nd), not_nan)
                } else {
                    not_nan
                };
                v_any_valid = _mm256_or_ps(v_any_valid, valid);
                v_any_nodata = _mm256_or_ps(v_any_nodata, _mm256_andnot_ps(valid, _mm256_castsi256_ps(_mm256_set1_epi32(-1))));
                if has_finite_nd {
                    $fv = _mm256_blendv_ps(v_nan, $fv, valid);
                }
                // Masked min/max preserving valid finite extrema
                let mn_cand = _mm256_blendv_ps(v_min, $fv, valid);
                let mx_cand = _mm256_blendv_ps(v_max, $fv, valid);
                v_min = _mm256_min_ps(v_min, mn_cand);
                v_max = _mm256_max_ps(v_max, mx_cand);
            }};
        }

        proc_avx!(f0);
        proc_avx!(f1);
        proc_avx!(f2);
        proc_avx!(f3);

        _mm256_storeu_ps(dst.as_mut_ptr().add(i), f0);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), f1);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), f2);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), f3);
        i += 32;
    }

    if _mm256_movemask_ps(v_any_valid) != 0 {
        *any_valid = true;
        let mut mn_arr = [0.0f32; 8];
        let mut mx_arr = [0.0f32; 8];
        _mm256_storeu_ps(mn_arr.as_mut_ptr(), v_min);
        _mm256_storeu_ps(mx_arr.as_mut_ptr(), v_max);
        for k in 0..8 {
            if mn_arr[k] < *min_val {
                *min_val = mn_arr[k];
            }
            if mx_arr[k] > *max_val {
                *max_val = mx_arr[k];
            }
        }
    }
    if _mm256_movemask_ps(v_any_nodata) != 0 {
        *any_nodata = true;
    }

    while i < n {
        let raw = f32::from_le_bytes([p3[i], p2[i], p1[i], p0[i]]);
        let (v, is_nd) = check_val(raw, finite_nodata);
        dst[i] = v;
        if is_nd {
            *any_nodata = true;
        } else {
            *any_valid = true;
            if v < *min_val {
                *min_val = v;
            }
            if v > *max_val {
                *max_val = v;
            }
        }
        i += 1;
    }
}

// SAFETY: Caller checks `is_x86_feature_detected!("avx2")`; SIMD loads/stores stay within input/output slice bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reconstruct_predictor1_le_avx2(
    raw: &[u8],
    dst: &mut [f32],
    finite_nodata: Option<f32>,
) -> BlockStats {
    use std::arch::x86_64::*;
    let pixels = dst.len();
    let mut i = 0usize;
    let mut any_valid = false;
    let mut any_nodata = false;
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    let mut v_min = _mm256_set1_ps(f32::INFINITY);
    let mut v_max = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut v_any_valid = _mm256_setzero_ps();
    let mut v_any_nodata = _mm256_setzero_ps();
    let v_nan = _mm256_set1_ps(f32::NAN);
    let has_finite_nd = finite_nodata.is_some();
    let v_nd = _mm256_set1_ps(finite_nodata.unwrap_or(0.0));
    let all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));

    while i + 32 <= pixels {
        let src_ptr = raw.as_ptr().add(i * 4) as *const f32;
        let mut f0 = _mm256_loadu_ps(src_ptr);
        let mut f1 = _mm256_loadu_ps(src_ptr.add(8));
        let mut f2 = _mm256_loadu_ps(src_ptr.add(16));
        let mut f3 = _mm256_loadu_ps(src_ptr.add(24));

        macro_rules! proc_avx1 {
            ($fv:expr) => {{
                let not_nan = _mm256_cmp_ps::<_CMP_ORD_Q>($fv, $fv);
                let valid = if has_finite_nd {
                    _mm256_andnot_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>($fv, v_nd), not_nan)
                } else {
                    not_nan
                };
                v_any_valid = _mm256_or_ps(v_any_valid, valid);
                v_any_nodata = _mm256_or_ps(v_any_nodata, _mm256_andnot_ps(valid, all_ones));
                if has_finite_nd {
                    $fv = _mm256_blendv_ps(v_nan, $fv, valid);
                }
                v_min = _mm256_min_ps(v_min, _mm256_blendv_ps(v_min, $fv, valid));
                v_max = _mm256_max_ps(v_max, _mm256_blendv_ps(v_max, $fv, valid));
            }};
        }

        proc_avx1!(f0);
        proc_avx1!(f1);
        proc_avx1!(f2);
        proc_avx1!(f3);

        let dst_ptr = dst.as_mut_ptr().add(i);
        _mm256_storeu_ps(dst_ptr, f0);
        _mm256_storeu_ps(dst_ptr.add(8), f1);
        _mm256_storeu_ps(dst_ptr.add(16), f2);
        _mm256_storeu_ps(dst_ptr.add(24), f3);
        i += 32;
    }

    if _mm256_movemask_ps(v_any_valid) != 0 {
        any_valid = true;
        let mut mn_arr = [0.0f32; 8];
        let mut mx_arr = [0.0f32; 8];
        _mm256_storeu_ps(mn_arr.as_mut_ptr(), v_min);
        _mm256_storeu_ps(mx_arr.as_mut_ptr(), v_max);
        for k in 0..8 {
            if mn_arr[k] < min_val {
                min_val = mn_arr[k];
            }
            if mx_arr[k] > max_val {
                max_val = mx_arr[k];
            }
        }
    }
    if _mm256_movemask_ps(v_any_nodata) != 0 {
        any_nodata = true;
    }

    while i < pixels {
        let off = i * 4;
        let v_raw = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
        let (v, is_nd) = check_val(v_raw, finite_nodata);
        dst[i] = v;
        if is_nd {
            any_nodata = true;
        } else {
            any_valid = true;
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        i += 1;
    }
    finalize_stats(any_valid, any_nodata, min_val, max_val)
}

/// SIMD 8-lane `u32` horizontal prefix-sum -> `f32` + nodata/stats on x86_64 AVX2 (Predictor 2).
// SAFETY: Caller checks `is_x86_feature_detected!("avx2")`; SIMD loads/stores stay within input/output slice bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reconstruct_predictor2_le_avx2(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    dst: &mut [f32],
    finite_nodata: Option<f32>,
) -> BlockStats {
    use std::arch::x86_64::*;
    let w = tile_w as usize;
    let h = tile_h as usize;
    let row_bytes = w * 4;

    let mut v_min = _mm256_set1_ps(f32::INFINITY);
    let mut v_max = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut v_any_valid = _mm256_setzero_ps();
    let mut v_any_nodata = _mm256_setzero_ps();
    let v_nan = _mm256_set1_ps(f32::NAN);
    let has_finite_nd = finite_nodata.is_some();
    let v_nd = _mm256_set1_ps(finite_nodata.unwrap_or(0.0));
    let all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));

    let mut any_valid = false;
    let mut any_nodata = false;
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for y in 0..h {
        let row_src = &raw[y * row_bytes..(y + 1) * row_bytes];
        let row_dst = &mut dst[y * w..(y + 1) * w];
        let mut carry = _mm256_setzero_si256();
        let mut i = 0usize;

        while i + 8 <= w {
            let mut v = _mm256_loadu_si256(row_src.as_ptr().add(i * 4) as *const __m256i);
            // In-lane 4-wide u32 prefix sum
            v = _mm256_add_epi32(v, _mm256_slli_si256::<4>(v));
            v = _mm256_add_epi32(v, _mm256_slli_si256::<8>(v));
            // Propagate lane-3 sum into upper 128-bit lane
            let hi_carry = _mm256_shuffle_epi32::<0xFF>(_mm256_permute2x128_si256::<0x08>(v, v));
            v = _mm256_add_epi32(v, hi_carry);
            v = _mm256_add_epi32(v, carry);
            // Broadcast lane 7 as next carry
            carry = _mm256_shuffle_epi32::<0xFF>(_mm256_permute2x128_si256::<0x11>(v, v));

            let mut fv = _mm256_castsi256_ps(v);
            let not_nan = _mm256_cmp_ps::<_CMP_ORD_Q>(fv, fv);
            let valid = if has_finite_nd {
                _mm256_andnot_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(fv, v_nd), not_nan)
            } else {
                not_nan
            };
            v_any_valid = _mm256_or_ps(v_any_valid, valid);
            v_any_nodata = _mm256_or_ps(v_any_nodata, _mm256_andnot_ps(valid, all_ones));
            if has_finite_nd {
                fv = _mm256_blendv_ps(v_nan, fv, valid);
            }
            v_min = _mm256_min_ps(v_min, _mm256_blendv_ps(v_min, fv, valid));
            v_max = _mm256_max_ps(v_max, _mm256_blendv_ps(v_max, fv, valid));
            _mm256_storeu_ps(row_dst.as_mut_ptr().add(i), fv);
            i += 8;
        }

        if i < w {
            let mut acc = _mm_cvtsi128_si32(_mm256_castsi256_si128(carry)) as u32;
            while i < w {
                let off = i * 4;
                let delta = u32::from_le_bytes([
                    row_src[off],
                    row_src[off + 1],
                    row_src[off + 2],
                    row_src[off + 3],
                ]);
                acc = acc.wrapping_add(delta);
                let (v, is_nd) = check_val(f32::from_bits(acc), finite_nodata);
                row_dst[i] = v;
                if is_nd {
                    any_nodata = true;
                } else {
                    any_valid = true;
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                }
                i += 1;
            }
        }
    }

    if _mm256_movemask_ps(v_any_valid) != 0 {
        any_valid = true;
        let mut mn_arr = [0.0f32; 8];
        let mut mx_arr = [0.0f32; 8];
        _mm256_storeu_ps(mn_arr.as_mut_ptr(), v_min);
        _mm256_storeu_ps(mx_arr.as_mut_ptr(), v_max);
        for k in 0..8 {
            if mn_arr[k] < min_val { min_val = mn_arr[k]; }
            if mx_arr[k] > max_val { max_val = mx_arr[k]; }
        }
    }
    if _mm256_movemask_ps(v_any_nodata) != 0 {
        any_nodata = true;
    }

    finalize_stats(any_valid, any_nodata, min_val, max_val)
}

/// SIMD 8-lane `u16` horizontal prefix-sum directly into compact `u16` + nodata/stats on x86_64 AVX2.
// SAFETY: Caller checks `is_x86_feature_detected!("avx2")`; SIMD loads/stores stay within input/output slice bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn u16_pred2_row_avx2(
    src_u16: &[u8],
    dst_u16: &mut [u16],
    nd_u16: Option<u16>,
    any_valid: &mut bool,
    any_nodata: &mut bool,
    min_val: &mut f32,
    max_val: &mut f32,
) {
    use std::arch::x86_64::*;
    let w = dst_u16.len();
    let mut carry = _mm_setzero_si128();
    let bcast_last = _mm_setr_epi8(14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15);

    let has_nd = nd_u16.is_some();
    let v_nd = _mm_set1_epi16(nd_u16.unwrap_or(0) as i16);
    let v_max_u16 = _mm_set1_epi16(-1i16);
    let v_zero = _mm_setzero_si128();

    let mut v_min = v_max_u16;
    let mut v_max = v_zero;
    let mut v_av = v_zero;
    let mut v_an = v_zero;

    let mut i = 0usize;
    while i + 8 <= w {
        let mut v = _mm_loadu_si128(src_u16.as_ptr().add(i * 2) as *const __m128i);
        v = _mm_add_epi16(v, _mm_slli_si128::<2>(v));
        v = _mm_add_epi16(v, _mm_slli_si128::<4>(v));
        v = _mm_add_epi16(v, _mm_slli_si128::<8>(v));
        v = _mm_add_epi16(v, carry);
        carry = _mm_shuffle_epi8(v, bcast_last);
        _mm_storeu_si128(dst_u16.as_mut_ptr().add(i) as *mut __m128i, v);

        if has_nd {
            let is_nd = _mm_cmpeq_epi16(v, v_nd);
            let valid = _mm_andnot_si128(is_nd, v_max_u16);
            v_av = _mm_or_si128(v_av, valid);
            v_an = _mm_or_si128(v_an, is_nd);
            // Replace nodata lanes with 65535 for min and 0 for max
            let mn_c = _mm_or_si128(_mm_and_si128(valid, v), _mm_and_si128(is_nd, v_max_u16));
            let mx_c = _mm_and_si128(valid, v);
            v_min = _mm_min_epu16(v_min, mn_c);
            v_max = _mm_max_epu16(v_max, mx_c);
        } else {
            v_av = v_max_u16;
            v_min = _mm_min_epu16(v_min, v);
            v_max = _mm_max_epu16(v_max, v);
        }
        i += 8;
    }

    if _mm_movemask_epi8(v_av) != 0 {
        *any_valid = true;
        let mut mn_arr = [0u16; 8];
        let mut mx_arr = [0u16; 8];
        _mm_storeu_si128(mn_arr.as_mut_ptr() as *mut __m128i, v_min);
        _mm_storeu_si128(mx_arr.as_mut_ptr() as *mut __m128i, v_max);
        for k in 0..8 {
            let mn = mn_arr[k] as f32;
            let mx = mx_arr[k] as f32;
            if mn < *min_val { *min_val = mn; }
            if mx > *max_val { *max_val = mx; }
        }
    }
    if _mm_movemask_epi8(v_an) != 0 {
        *any_nodata = true;
    }

    if i < w {
        let mut acc: u16 = if i > 0 {
            _mm_extract_epi16::<7>(carry) as u16
        } else {
            0
        };
        while i < w {
            let off = i * 2;
            acc = acc.wrapping_add(u16::from_le_bytes([src_u16[off], src_u16[off + 1]]));
            dst_u16[i] = acc;
            if nd_u16 == Some(acc) {
                *any_nodata = true;
            } else {
                *any_valid = true;
                let vf = acc as f32;
                if vf < *min_val { *min_val = vf; }
                if vf > *max_val { *max_val = vf; }
            }
            i += 1;
        }
    }
}

/// Hardware-accelerated 3-band chunky Predictor-3 unshuffle + planar deinterleave on x86_64 AVX2
/// using `pshufb` 3-way deinterleave (16 RGB pixels / 48 floats / 192 bytes per iteration).
// SAFETY: Caller checks `is_x86_feature_detected!("avx2")`; SIMD loads/stores stay within row/out slice bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn unshuffle_row_stride3_avx2(
    p0: &[u8],
    p1: &[u8],
    p2: &[u8],
    p3: &[u8],
    dst0: &mut [f32],
    dst1: &mut [f32],
    dst2: &mut [f32],
    finite_nodata: Option<f32>,
    any_valid: &mut [bool; 3],
    any_nodata: &mut [bool; 3],
    min_val: &mut [f32; 3],
    max_val: &mut [f32; 3],
) {
    use std::arch::x86_64::*;
    let w = dst0.len();
    let mut x = 0usize;

    let v_nan = _mm256_set1_ps(f32::NAN);
    let has_finite_nd = finite_nodata.is_some();
    let v_nd = _mm256_set1_ps(finite_nodata.unwrap_or(0.0));
    let all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));

    let mut v_min = [_mm256_set1_ps(f32::INFINITY); 3];
    let mut v_max = [_mm256_set1_ps(f32::NEG_INFINITY); 3];
    let mut v_av = [_mm256_setzero_ps(); 3];
    let mut v_an = [_mm256_setzero_ps(); 3];

    // 3-way deinterleave masks for 48 bytes (a=0..15, b=16..31, c=32..47) -> 16 bytes per band
    let r_a = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let r_b = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let r_c = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);

    let g_a = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let g_b = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let g_c = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);

    let b_a = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let b_b = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let b_c = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

    macro_rules! deinterleave_plane {
        ($ptr:expr) => {{
            let a = _mm_loadu_si128($ptr as *const __m128i);
            let b = _mm_loadu_si128($ptr.add(16) as *const __m128i);
            let c = _mm_loadu_si128($ptr.add(32) as *const __m128i);
            let c0 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, r_a), _mm_shuffle_epi8(b, r_b)), _mm_shuffle_epi8(c, r_c));
            let c1 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, g_a), _mm_shuffle_epi8(b, g_b)), _mm_shuffle_epi8(c, g_c));
            let c2 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, b_a), _mm_shuffle_epi8(b, b_b)), _mm_shuffle_epi8(c, b_c));
            (c0, c1, c2)
        }};
    }

    macro_rules! transpose_store_16 {
        ($b0:expr, $b1:expr, $b2:expr, $b3:expr, $dst:expr, $bi:expr) => {{
            let lo01 = _mm_unpacklo_epi8($b0, $b1);
            let hi01 = _mm_unpackhi_epi8($b0, $b1);
            let lo23 = _mm_unpacklo_epi8($b2, $b3);
            let hi23 = _mm_unpackhi_epi8($b2, $b3);

            let f0 = _mm_castsi128_ps(_mm_unpacklo_epi16(lo01, lo23));
            let f1 = _mm_castsi128_ps(_mm_unpackhi_epi16(lo01, lo23));
            let f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(hi01, hi23));
            let f3 = _mm_castsi128_ps(_mm_unpackhi_epi16(hi01, hi23));

            let mut v0 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(f0), f1);
            let mut v1 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(f2), f3);

            for fv in [&mut v0, &mut v1] {
                let not_nan = _mm256_cmp_ps::<_CMP_ORD_Q>(*fv, *fv);
                let valid = if has_finite_nd {
                    _mm256_andnot_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(*fv, v_nd), not_nan)
                } else {
                    not_nan
                };
                v_av[$bi] = _mm256_or_ps(v_av[$bi], valid);
                v_an[$bi] = _mm256_or_ps(v_an[$bi], _mm256_andnot_ps(valid, all_ones));
                if has_finite_nd {
                    *fv = _mm256_blendv_ps(v_nan, *fv, valid);
                }
                v_min[$bi] = _mm256_min_ps(v_min[$bi], _mm256_blendv_ps(v_min[$bi], *fv, valid));
                v_max[$bi] = _mm256_max_ps(v_max[$bi], _mm256_blendv_ps(v_max[$bi], *fv, valid));
            }
            _mm256_storeu_ps($dst, v0);
            _mm256_storeu_ps($dst.add(8), v1);
        }};
    }

    while x + 16 <= w {
        let byte_off = x * 3;
        let d3 = deinterleave_plane!(p3.as_ptr().add(byte_off));
        let d2 = deinterleave_plane!(p2.as_ptr().add(byte_off));
        let d1 = deinterleave_plane!(p1.as_ptr().add(byte_off));
        let d0 = deinterleave_plane!(p0.as_ptr().add(byte_off));

        transpose_store_16!(d3.0, d2.0, d1.0, d0.0, dst0.as_mut_ptr().add(x), 0);
        transpose_store_16!(d3.1, d2.1, d1.1, d0.1, dst1.as_mut_ptr().add(x), 1);
        transpose_store_16!(d3.2, d2.2, d1.2, d0.2, dst2.as_mut_ptr().add(x), 2);
        x += 16;
    }

    for b in 0..3 {
        if _mm256_movemask_ps(v_av[b]) != 0 {
            any_valid[b] = true;
            let mut mn_arr = [0.0f32; 8];
            let mut mx_arr = [0.0f32; 8];
            _mm256_storeu_ps(mn_arr.as_mut_ptr(), v_min[b]);
            _mm256_storeu_ps(mx_arr.as_mut_ptr(), v_max[b]);
            for k in 0..8 {
                if mn_arr[k] < min_val[b] { min_val[b] = mn_arr[k]; }
                if mx_arr[k] > max_val[b] { max_val[b] = mx_arr[k]; }
            }
        }
        if _mm256_movemask_ps(v_an[b]) != 0 {
            any_nodata[b] = true;
        }
    }

    while x < w {
        let base = x * 3;
        for (b, dst) in [(0, &mut *dst0), (1, &mut *dst1), (2, &mut *dst2)] {
            let idx = base + b;
            let raw = f32::from_le_bytes([p3[idx], p2[idx], p1[idx], p0[idx]]);
            let (v, is_nd) = check_val(raw, finite_nodata);
            dst[x] = v;
            if is_nd {
                any_nodata[b] = true;
            } else {
                any_valid[b] = true;
                if v < min_val[b] { min_val[b] = v; }
                if v > max_val[b] { max_val[b] = v; }
            }
        }
        x += 1;
    }
}

// ============================================================================
// Scalar Fallbacks & Public Dispatch
// ============================================================================

#[inline(always)]
fn prefix_sum_scalar(row: &mut [u8], stride: usize) {
    if stride == 1 {
        let mut acc = 0u8;
        for b in row.iter_mut() {
            acc = acc.wrapping_add(*b);
            *b = acc;
        }
    } else if stride == 3 {
        let mut a0 = 0u8;
        let mut a1 = 0u8;
        let mut a2 = 0u8;
        let mut i = 0usize;
        let len = row.len();
        while i + 3 <= len {
            a0 = a0.wrapping_add(row[i]);
            row[i] = a0;
            a1 = a1.wrapping_add(row[i + 1]);
            row[i + 1] = a1;
            a2 = a2.wrapping_add(row[i + 2]);
            row[i + 2] = a2;
            i += 3;
        }
        while i < len {
            row[i] = row[i].wrapping_add(row[i - 3]);
            i += 1;
        }
    } else {
        for i in stride..row.len() {
            row[i] = row[i].wrapping_add(row[i - stride]);
        }
    }
}

/// Reconstruct Predictor 3 (TIFF Technical Note 3 floating-point horizontal differencing + byte shuffle)
/// into `bands_in_tile` contiguous planar `DecodedBand`s.
pub fn reconstruct_predictor3(
    raw: &mut [u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    nodata: Option<f32>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let w = tile_w as usize;
    let h = tile_h as usize;
    let samples_per_row = w * bands_in_tile;
    let row_bytes = samples_per_row * 4;

    if bands_in_tile == 1 {
        let mut slab = AlignedF32Slab::new_uninit(w * h);
        let mut any_valid = false;
        let mut any_nodata = false;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        #[cfg(target_arch = "x86_64")]
        let has_avx2 = std::is_x86_feature_detected!("avx2");

        for y in 0..h {
            let row = &mut raw[y * row_bytes..(y + 1) * row_bytes];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                prefix_sum_stride1_neon(row);
            }
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2 {
                    unsafe { prefix_sum_stride1_avx2(row) };
                } else {
                    prefix_sum_scalar(row, 1);
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            prefix_sum_scalar(row, 1);

            let p0 = &row[0..w];
            let p1 = &row[w..2 * w];
            let p2 = &row[2 * w..3 * w];
            let p3 = &row[3 * w..4 * w];
            let dst_row = &mut slab[y * w..(y + 1) * w];

            #[cfg(target_arch = "aarch64")]
            unsafe {
                unshuffle_row_stride1_neon(
                    p0, p1, p2, p3, dst_row, finite_nodata,
                    &mut any_valid, &mut any_nodata, &mut min_val, &mut max_val,
                );
            }
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2 {
                    unsafe {
                        unshuffle_row_stride1_avx2(
                            p0, p1, p2, p3, dst_row, finite_nodata,
                            &mut any_valid, &mut any_nodata, &mut min_val, &mut max_val,
                        );
                    }
                } else {
                    for x in 0..w {
                        let v_raw = f32::from_le_bytes([p3[x], p2[x], p1[x], p0[x]]);
                        let (v, is_nd) = check_val(v_raw, finite_nodata);
                        dst_row[x] = v;
                        if is_nd {
                            any_nodata = true;
                        } else {
                            any_valid = true;
                            if v < min_val { min_val = v; }
                            if v > max_val { max_val = v; }
                        }
                    }
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                for x in 0..w {
                    let v_raw = f32::from_le_bytes([p3[x], p2[x], p1[x], p0[x]]);
                    let (v, is_nd) = check_val(v_raw, finite_nodata);
                    dst_row[x] = v;
                    if is_nd {
                        any_nodata = true;
                    } else {
                        any_valid = true;
                        if v < min_val { min_val = v; }
                        if v > max_val { max_val = v; }
                    }
                }
            }
        }

        vec![DecodedBand {
            data: BandData::F32(slab),
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid, any_nodata, min_val, max_val),
        }]
    } else if bands_in_tile == 3 {
        // Dedicated 3-band chunky satellite RGB fast path!
        let mut s0 = AlignedF32Slab::new_uninit(w * h);
        let mut s1 = AlignedF32Slab::new_uninit(w * h);
        let mut s2 = AlignedF32Slab::new_uninit(w * h);
        let mut any_valid = [false; 3];
        let mut any_nodata = [false; 3];
        let mut min_val = [f32::INFINITY; 3];
        let mut max_val = [f32::NEG_INFINITY; 3];

        #[cfg(target_arch = "x86_64")]
        let has_avx2 = std::is_x86_feature_detected!("avx2");

        for y in 0..h {
            let row = &mut raw[y * row_bytes..(y + 1) * row_bytes];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                prefix_sum_stride3_neon(row);
            }
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2 {
                    unsafe { prefix_sum_stride3_avx2(row) };
                } else {
                    prefix_sum_scalar(row, 3);
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            prefix_sum_scalar(row, 3);

            let p0 = &row[0..samples_per_row];
            let p1 = &row[samples_per_row..2 * samples_per_row];
            let p2 = &row[2 * samples_per_row..3 * samples_per_row];
            let p3 = &row[3 * samples_per_row..4 * samples_per_row];
            let row_off = y * w;

            #[cfg(target_arch = "aarch64")]
            unsafe {
                unshuffle_row_stride3_neon(
                    p0, p1, p2, p3,
                    &mut s0[row_off..row_off + w],
                    &mut s1[row_off..row_off + w],
                    &mut s2[row_off..row_off + w],
                    finite_nodata,
                    &mut any_valid, &mut any_nodata, &mut min_val, &mut max_val,
                );
            }
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2 {
                    unsafe {
                        unshuffle_row_stride3_avx2(
                            p0, p1, p2, p3,
                            &mut s0[row_off..row_off + w],
                            &mut s1[row_off..row_off + w],
                            &mut s2[row_off..row_off + w],
                            finite_nodata,
                            &mut any_valid, &mut any_nodata, &mut min_val, &mut max_val,
                        );
                    }
                } else {
                    for x in 0..w {
                        let base = x * 3;
                        for (b, slab) in [(0, &mut s0), (1, &mut s1), (2, &mut s2)] {
                            let idx = base + b;
                            let v_raw = f32::from_le_bytes([p3[idx], p2[idx], p1[idx], p0[idx]]);
                            let (v, is_nd) = check_val(v_raw, finite_nodata);
                            slab[row_off + x] = v;
                            if is_nd {
                                any_nodata[b] = true;
                            } else {
                                any_valid[b] = true;
                                if v < min_val[b] { min_val[b] = v; }
                                if v > max_val[b] { max_val[b] = v; }
                            }
                        }
                    }
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                for x in 0..w {
                    let base = x * 3;
                    for (b, slab) in [(0, &mut s0), (1, &mut s1), (2, &mut s2)] {
                        let idx = base + b;
                        let v_raw = f32::from_le_bytes([p3[idx], p2[idx], p1[idx], p0[idx]]);
                        let (v, is_nd) = check_val(v_raw, finite_nodata);
                        slab[row_off + x] = v;
                        if is_nd {
                            any_nodata[b] = true;
                        } else {
                            any_valid[b] = true;
                            if v < min_val[b] { min_val[b] = v; }
                            if v > max_val[b] { max_val[b] = v; }
                        }
                    }
                }
            }
        }

        vec![
            DecodedBand {
                data: BandData::F32(s0),
                width: tile_w,
                height: tile_h,
                stats: finalize_stats(any_valid[0], any_nodata[0], min_val[0], max_val[0]),
            },
            DecodedBand {
                data: BandData::F32(s1),
                width: tile_w,
                height: tile_h,
                stats: finalize_stats(any_valid[1], any_nodata[1], min_val[1], max_val[1]),
            },
            DecodedBand {
                data: BandData::F32(s2),
                width: tile_w,
                height: tile_h,
                stats: finalize_stats(any_valid[2], any_nodata[2], min_val[2], max_val[2]),
            },
        ]
    } else {
        let mut slabs: Vec<AlignedF32Slab> = (0..bands_in_tile)
            .map(|_| AlignedF32Slab::new_uninit(w * h))
            .collect();
        let mut any_valid = vec![false; bands_in_tile];
        let mut any_nodata = vec![false; bands_in_tile];
        let mut min_val = vec![f32::INFINITY; bands_in_tile];
        let mut max_val = vec![f32::NEG_INFINITY; bands_in_tile];

        for y in 0..h {
            let row = &mut raw[y * row_bytes..(y + 1) * row_bytes];
            prefix_sum_scalar(row, bands_in_tile);

            let p0 = &row[0..samples_per_row];
            let p1 = &row[samples_per_row..2 * samples_per_row];
            let p2 = &row[2 * samples_per_row..3 * samples_per_row];
            let p3 = &row[3 * samples_per_row..4 * samples_per_row];
            let row_off = y * w;

            for x in 0..w {
                let base = x * bands_in_tile;
                for b in 0..bands_in_tile {
                    let idx = base + b;
                    let v_raw = f32::from_le_bytes([p3[idx], p2[idx], p1[idx], p0[idx]]);
                    let (v, is_nd) = check_val(v_raw, finite_nodata);
                    slabs[b][row_off + x] = v;
                    if is_nd {
                        any_nodata[b] = true;
                    } else {
                        any_valid[b] = true;
                        if v < min_val[b] { min_val[b] = v; }
                        if v > max_val[b] { max_val[b] = v; }
                    }
                }
            }
        }

        slabs
            .into_iter()
            .enumerate()
            .map(|(b, data)| DecodedBand {
                data: BandData::F32(data),
                width: tile_w,
                height: tile_h,
                stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
            })
            .collect()
    }
}

/// Reconstruct Predictor 1 (None) into planar `DecodedBand`s with nodata normalization + `BlockStats`.
pub fn reconstruct_predictor1(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    nodata: Option<f32>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w.checked_mul(h).expect("tile pixel count overflow");
    let expected_bytes = pixels
        .checked_mul(bands_in_tile)
        .and_then(|n| n.checked_mul(4))
        .expect("tile byte count overflow");
    assert!(
        raw.len() >= expected_bytes,
        "reconstruct_predictor1: raw buffer too short ({} < {expected_bytes})",
        raw.len()
    );

    if bands_in_tile == 1 {
        let mut slab = AlignedF32Slab::new_uninit(pixels);

        #[cfg(target_arch = "aarch64")]
        if byte_order == ByteOrder::Little {
            use std::arch::aarch64::*;
            let dst = slab.as_mut_slice();
            let mut i = 0usize;
            let mut any_valid = false;
            let mut any_nodata = false;
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            let mut v_min = unsafe { vdupq_n_f32(f32::INFINITY) };
            let mut v_max = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
            let mut v_any_valid = unsafe { vdupq_n_u32(0) };
            let mut v_any_nodata = unsafe { vdupq_n_u32(0) };
            let v_nan = unsafe { vdupq_n_f32(f32::NAN) };
            let has_finite_nd = finite_nodata.is_some();
            let v_nd = unsafe { vdupq_n_f32(finite_nodata.unwrap_or(0.0)) };

            while i + 16 <= pixels {
                unsafe {
                    let src_ptr = raw.as_ptr().add(i * 4) as *const f32;
                    let mut f0 = vld1q_f32(src_ptr);
                    let mut f1 = vld1q_f32(src_ptr.add(4));
                    let mut f2 = vld1q_f32(src_ptr.add(8));
                    let mut f3 = vld1q_f32(src_ptr.add(12));

                    macro_rules! proc1 {
                        ($fv:expr) => {{
                            let not_nan = vceqq_f32($fv, $fv);
                            let valid = if has_finite_nd {
                                vbicq_u32(not_nan, vceqq_f32($fv, v_nd))
                            } else {
                                not_nan
                            };
                            v_any_valid = vorrq_u32(v_any_valid, valid);
                            v_any_nodata = vorrq_u32(v_any_nodata, vmvnq_u32(valid));
                            if has_finite_nd {
                                $fv = vbslq_f32(valid, $fv, v_nan);
                            }
                            v_min = vminnmq_f32(v_min, $fv);
                            v_max = vmaxnmq_f32(v_max, $fv);
                        }};
                    }

                    proc1!(f0);
                    proc1!(f1);
                    proc1!(f2);
                    proc1!(f3);

                    let dst_ptr = dst.as_mut_ptr().add(i);
                    vst1q_f32(dst_ptr, f0);
                    vst1q_f32(dst_ptr.add(4), f1);
                    vst1q_f32(dst_ptr.add(8), f2);
                    vst1q_f32(dst_ptr.add(12), f3);
                }
                i += 16;
            }
            unsafe {
                if vmaxvq_u32(v_any_valid) != 0 {
                    any_valid = true;
                    min_val = vminnmvq_f32(v_min);
                    max_val = vmaxnmvq_f32(v_max);
                }
                if vmaxvq_u32(v_any_nodata) != 0 {
                    any_nodata = true;
                }
            }
            while i < pixels {
                let off = i * 4;
                let v_raw = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                let (v, is_nd) = check_val(v_raw, finite_nodata);
                dst[i] = v;
                if is_nd {
                    any_nodata = true;
                } else {
                    any_valid = true;
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                }
                i += 1;
            }
            return vec![DecodedBand {
                data: BandData::F32(slab),
                width: tile_w,
                height: tile_h,
                stats: finalize_stats(any_valid, any_nodata, min_val, max_val),
            }];
        }

        #[cfg(target_arch = "x86_64")]
        if byte_order == ByteOrder::Little && std::is_x86_feature_detected!("avx2") {
            let stats = unsafe { reconstruct_predictor1_le_avx2(raw, slab.as_mut_slice(), finite_nodata) };
            return vec![DecodedBand {
                data: BandData::F32(slab),
                width: tile_w,
                height: tile_h,
                stats,
            }];
        }

        let mut any_valid = false;
        let mut any_nodata = false;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..pixels {
            let off = i * 4;
            let bytes = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
            let v_raw = match byte_order {
                ByteOrder::Little => f32::from_le_bytes(bytes),
                ByteOrder::Big => f32::from_be_bytes(bytes),
            };
            let (v, is_nd) = check_val(v_raw, finite_nodata);
            slab[i] = v;
            if is_nd {
                any_nodata = true;
            } else {
                any_valid = true;
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
            }
        }

        vec![DecodedBand {
            data: BandData::F32(slab),
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid, any_nodata, min_val, max_val),
        }]
    } else {
        let mut slabs: Vec<AlignedF32Slab> = (0..bands_in_tile)
            .map(|_| AlignedF32Slab::new_uninit(pixels))
            .collect();
        let mut any_valid = vec![false; bands_in_tile];
        let mut any_nodata = vec![false; bands_in_tile];
        let mut min_val = vec![f32::INFINITY; bands_in_tile];
        let mut max_val = vec![f32::NEG_INFINITY; bands_in_tile];

        for i in 0..pixels {
            for b in 0..bands_in_tile {
                let off = (i * bands_in_tile + b) * 4;
                let bytes = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
                let v_raw = match byte_order {
                    ByteOrder::Little => f32::from_le_bytes(bytes),
                    ByteOrder::Big => f32::from_be_bytes(bytes),
                };
                let (v, is_nd) = check_val(v_raw, finite_nodata);
                slabs[b][i] = v;
                if is_nd {
                    any_nodata[b] = true;
                } else {
                    any_valid[b] = true;
                    if v < min_val[b] { min_val[b] = v; }
                    if v > max_val[b] { max_val[b] = v; }
                }
            }
        }

        slabs
            .into_iter()
            .enumerate()
            .map(|(b, data)| DecodedBand {
                data: BandData::F32(data),
                width: tile_w,
                height: tile_h,
                stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
            })
            .collect()
    }
}

/// Reconstruct Predictor 2 (Horizontal 32-bit integer differencing) into planar `DecodedBand`s.
pub fn reconstruct_predictor2(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    nodata: Option<f32>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;

    #[cfg(target_arch = "aarch64")]
    if bands_in_tile == 1 && byte_order == ByteOrder::Little {
        let mut slab = AlignedF32Slab::new_uninit(pixels);
        let stats = unsafe {
            reconstruct_predictor2_le_neon(raw, tile_w, tile_h, slab.as_mut_slice(), finite_nodata)
        };
        return vec![DecodedBand {
            data: BandData::F32(slab),
            width: tile_w,
            height: tile_h,
            stats,
        }];
    }

    #[cfg(target_arch = "x86_64")]
    if bands_in_tile == 1 && byte_order == ByteOrder::Little && std::is_x86_feature_detected!("avx2") {
        let mut slab = AlignedF32Slab::new_uninit(pixels);
        let stats = unsafe {
            reconstruct_predictor2_le_avx2(raw, tile_w, tile_h, slab.as_mut_slice(), finite_nodata)
        };
        return vec![DecodedBand {
            data: BandData::F32(slab),
            width: tile_w,
            height: tile_h,
            stats,
        }];
    }

    let mut slabs: Vec<AlignedF32Slab> = (0..bands_in_tile)
        .map(|_| AlignedF32Slab::new_uninit(pixels))
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_val = vec![f32::INFINITY; bands_in_tile];
    let mut max_val = vec![f32::NEG_INFINITY; bands_in_tile];

    let mut acc = vec![0u32; bands_in_tile];

    for y in 0..h {
        acc.fill(0);
        let row_off = y * w;
        for x in 0..w {
            for b in 0..bands_in_tile {
                let off = ((row_off + x) * bands_in_tile + b) * 4;
                let bytes = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
                let delta = match byte_order {
                    ByteOrder::Little => u32::from_le_bytes(bytes),
                    ByteOrder::Big => u32::from_be_bytes(bytes),
                };
                acc[b] = acc[b].wrapping_add(delta);
                let v_raw = f32::from_bits(acc[b]);
                let (v, is_nd) = check_val(v_raw, finite_nodata);
                slabs[b][row_off + x] = v;
                if is_nd {
                    any_nodata[b] = true;
                } else {
                    any_valid[b] = true;
                    if v < min_val[b] { min_val[b] = v; }
                    if v > max_val[b] { max_val[b] = v; }
                }
            }
        }
    }

    slabs
        .into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::F32(data),
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
        })
        .collect()
}

// ============================================================================
// uint16 Decoders (Landsat Collection 2 & Sentinel-2 L2A -> AlignedF32Slab)
// ============================================================================

/// SIMD 8-lane `u16` horizontal prefix-sum directly into compact `u16` + nodata/stats on ARM64 NEON.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn u16_pred2_row_neon(
    src_u16: &[u8],
    dst_u16: &mut [u16],
    nd_u16: Option<u16>,
    any_valid: &mut bool,
    any_nodata: &mut bool,
    min_val: &mut f32,
    max_val: &mut f32,
) {
    use std::arch::aarch64::*;
    let w = dst_u16.len();
    let zero = unsafe { vdupq_n_u16(0) };
    let mut carry = zero;
    let max_u16 = unsafe { vdupq_n_u16(u16::MAX) };
    let mut v_min = max_u16;
    let mut v_max = zero;
    let mut v_av = zero;
    let mut v_an = zero;
    let has_nd = nd_u16.is_some();
    let v_nd = unsafe { vdupq_n_u16(nd_u16.unwrap_or(0)) };

    let mut i = 0usize;
    while i + 8 <= w {
        unsafe {
            let mut v = vld1q_u16(src_u16.as_ptr().add(i * 2) as *const u16);
            // 8-lane Kogge-Stone u16 prefix sum
            v = vaddq_u16(v, vextq_u16::<7>(zero, v));
            v = vaddq_u16(v, vextq_u16::<6>(zero, v));
            v = vaddq_u16(v, vextq_u16::<4>(zero, v));
            v = vaddq_u16(v, carry);
            carry = vdupq_laneq_u16::<7>(v);
            vst1q_u16(dst_u16.as_mut_ptr().add(i), v);

            if has_nd {
                let is_nd = vceqq_u16(v, v_nd);
                let valid = vmvnq_u16(is_nd);
                v_av = vorrq_u16(v_av, valid);
                v_an = vorrq_u16(v_an, is_nd);
                v_min = vminq_u16(v_min, vbslq_u16(valid, v, max_u16));
                v_max = vmaxq_u16(v_max, vbslq_u16(valid, v, zero));
            } else {
                v_av = max_u16;
                v_min = vminq_u16(v_min, v);
                v_max = vmaxq_u16(v_max, v);
            }
        }
        i += 8;
    }

    unsafe {
        if vmaxvq_u16(v_av) != 0 {
            *any_valid = true;
            let mn = vminvq_u16(v_min) as f32;
            let mx = vmaxvq_u16(v_max) as f32;
            if mn < *min_val { *min_val = mn; }
            if mx > *max_val { *max_val = mx; }
        }
        if vmaxvq_u16(v_an) != 0 {
            *any_nodata = true;
        }
    }

    if i < w {
        let mut acc: u16 = if i > 0 {
            unsafe { vgetq_lane_u16::<7>(carry) }
        } else {
            0
        };
        while i < w {
            let off = i * 2;
            acc = acc.wrapping_add(u16::from_le_bytes([src_u16[off], src_u16[off + 1]]));
            dst_u16[i] = acc;
            if nd_u16 == Some(acc) {
                *any_nodata = true;
            } else {
                *any_valid = true;
                let vf = acc as f32;
                if vf < *min_val { *min_val = vf; }
                if vf > *max_val { *max_val = vf; }
            }
            i += 1;
        }
    }
}

/// Reconstruct `uint16` Predictor 1 (None) or Predictor 2 (16-bit horizontal diff, Landsat C2)
/// directly into compact `BandData::U16` (2 bytes/pixel, 2× less cache RAM) with `BlockStats`.
pub fn reconstruct_u16(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    horizontal_diff: bool,
    nodata: Option<f32>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let nd_u16 = finite_nodata.and_then(|nd| {
        if nd >= 0.0 && nd <= (u16::MAX as f32) && (nd as u16) as f32 == nd {
            Some(nd as u16)
        } else {
            None
        }
    });
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;

    // Fast path: single-band little-endian u16 + Predictor 2 (Landsat Collection 2 & Sentinel-2!)
    #[cfg(target_arch = "aarch64")]
    if bands_in_tile == 1 && byte_order == ByteOrder::Little && horizontal_diff {
        let mut data = vec![0u16; pixels].into_boxed_slice();
        let mut any_valid = false;
        let mut any_nodata = false;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let row_bytes = w * 2;
        for y in 0..h {
            unsafe {
                u16_pred2_row_neon(
                    &raw[y * row_bytes..(y + 1) * row_bytes],
                    &mut data[y * w..(y + 1) * w],
                    nd_u16,
                    &mut any_valid,
                    &mut any_nodata,
                    &mut min_val,
                    &mut max_val,
                );
            }
        }
        return vec![DecodedBand {
            data: BandData::U16 { data, nodata: nd_u16 },
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid, any_nodata, min_val, max_val),
        }];
    }

    #[cfg(target_arch = "x86_64")]
    if bands_in_tile == 1
        && byte_order == ByteOrder::Little
        && horizontal_diff
        && std::is_x86_feature_detected!("avx2")
    {
        let mut data = vec![0u16; pixels].into_boxed_slice();
        let mut any_valid = false;
        let mut any_nodata = false;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let row_bytes = w * 2;
        for y in 0..h {
            unsafe {
                u16_pred2_row_avx2(
                    &raw[y * row_bytes..(y + 1) * row_bytes],
                    &mut data[y * w..(y + 1) * w],
                    nd_u16,
                    &mut any_valid,
                    &mut any_nodata,
                    &mut min_val,
                    &mut max_val,
                );
            }
        }
        return vec![DecodedBand {
            data: BandData::U16 { data, nodata: nd_u16 },
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid, any_nodata, min_val, max_val),
        }];
    }

    // Fast path: 1-band Little-Endian Predictor 1 (`!horizontal_diff`) with integer min/max.
    if bands_in_tile == 1 && !horizontal_diff && matches!(byte_order, ByteOrder::Little) {
        let mut data = vec![0u16; pixels].into_boxed_slice();
        let mut any_valid = false;
        let mut any_nodata = false;
        let mut min_u = u16::MAX;
        let mut max_u = u16::MIN;
        if let Some(nd) = nd_u16 {
            for (dst, chunk) in data.iter_mut().zip(raw.chunks_exact(2)) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                *dst = v;
                if v == nd {
                    any_nodata = true;
                } else {
                    any_valid = true;
                    if v < min_u { min_u = v; }
                    if v > max_u { max_u = v; }
                }
            }
        } else {
            for (dst, chunk) in data.iter_mut().zip(raw.chunks_exact(2)) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                *dst = v;
                if v < min_u { min_u = v; }
                if v > max_u { max_u = v; }
            }
            any_valid = pixels > 0;
        }
        return vec![DecodedBand {
            data: BandData::U16 { data, nodata: nd_u16 },
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(
                any_valid,
                any_nodata,
                if any_valid { min_u as f32 } else { f32::INFINITY },
                if any_valid { max_u as f32 } else { f32::NEG_INFINITY },
            ),
        }];
    }

    // General path (Predictor 1, Big-Endian, or multi-band chunky u16) -> compact U16
    let mut slabs: Vec<Box<[u16]>> = (0..bands_in_tile)
        .map(|_| vec![0u16; pixels].into_boxed_slice())
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_u = vec![u16::MAX; bands_in_tile];
    let mut max_u = vec![u16::MIN; bands_in_tile];
    let mut acc = vec![0u16; bands_in_tile];

    for y in 0..h {
        if horizontal_diff {
            acc.fill(0);
        }
        let row_off = y * w;
        for x in 0..w {
            for b in 0..bands_in_tile {
                let off = ((row_off + x) * bands_in_tile + b) * 2;
                let sample = match byte_order {
                    ByteOrder::Little => u16::from_le_bytes([raw[off], raw[off + 1]]),
                    ByteOrder::Big => u16::from_be_bytes([raw[off], raw[off + 1]]),
                };
                let val_u16 = if horizontal_diff {
                    acc[b] = acc[b].wrapping_add(sample);
                    acc[b]
                } else {
                    sample
                };
                slabs[b][row_off + x] = val_u16;
                if nd_u16 == Some(val_u16) {
                    any_nodata[b] = true;
                } else {
                    any_valid[b] = true;
                    if val_u16 < min_u[b] { min_u[b] = val_u16; }
                    if val_u16 > max_u[b] { max_u[b] = val_u16; }
                }
            }
        }
    }

    slabs
        .into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::U16 { data, nodata: nd_u16 },
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(
                any_valid[b],
                any_nodata[b],
                if any_valid[b] { min_u[b] as f32 } else { f32::INFINITY },
                if any_valid[b] { max_u[b] as f32 } else { f32::NEG_INFINITY },
            ),
        })
        .collect()
}

/// Reconstruct `int16` Predictor 1 or Predictor 2 (SRTM / Copernicus DEMs, MODIS) -> `AlignedF32Slab`.
pub fn reconstruct_i16(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    horizontal_diff: bool,
    nodata: Option<f32>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;

    let mut slabs: Vec<AlignedF32Slab> = (0..bands_in_tile)
        .map(|_| AlignedF32Slab::new_uninit(pixels))
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_val = vec![f32::INFINITY; bands_in_tile];
    let mut max_val = vec![f32::NEG_INFINITY; bands_in_tile];
    let mut acc = vec![0i16; bands_in_tile];

    for y in 0..h {
        if horizontal_diff {
            acc.fill(0);
        }
        let row_off = y * w;
        for x in 0..w {
            for b in 0..bands_in_tile {
                let off = ((row_off + x) * bands_in_tile + b) * 2;
                let sample = match byte_order {
                    ByteOrder::Little => i16::from_le_bytes([raw[off], raw[off + 1]]),
                    ByteOrder::Big => i16::from_be_bytes([raw[off], raw[off + 1]]),
                };
                let val_i16 = if horizontal_diff {
                    acc[b] = acc[b].wrapping_add(sample);
                    acc[b]
                } else {
                    sample
                };
                let (v, is_nd) = check_val(val_i16 as f32, finite_nodata);
                slabs[b][row_off + x] = v;
                if is_nd {
                    any_nodata[b] = true;
                } else {
                    any_valid[b] = true;
                    if v < min_val[b] { min_val[b] = v; }
                    if v > max_val[b] { max_val[b] = v; }
                }
            }
        }
    }

    slabs
        .into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::F32(data),
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
        })
        .collect()
}

/// Reconstruct `int8` Predictor 1 or Predictor 2 -> `AlignedF32Slab`.
pub fn reconstruct_i8(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    horizontal_diff: bool,
    nodata: Option<f32>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;
    let mut slabs: Vec<AlignedF32Slab> = (0..bands_in_tile)
        .map(|_| AlignedF32Slab::new_uninit(pixels))
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_val = vec![f32::INFINITY; bands_in_tile];
    let mut max_val = vec![f32::NEG_INFINITY; bands_in_tile];
    let mut acc = vec![0i8; bands_in_tile];

    for y in 0..h {
        if horizontal_diff {
            acc.fill(0);
        }
        let row_off = y * w;
        for x in 0..w {
            for b in 0..bands_in_tile {
                let s = raw[(row_off + x) * bands_in_tile + b] as i8;
                let val = if horizontal_diff {
                    acc[b] = acc[b].wrapping_add(s);
                    acc[b]
                } else {
                    s
                };
                let (v, is_nd) = check_val(val as f32, finite_nodata);
                slabs[b][row_off + x] = v;
                if is_nd {
                    any_nodata[b] = true;
                } else {
                    any_valid[b] = true;
                    if v < min_val[b] { min_val[b] = v; }
                    if v > max_val[b] { max_val[b] = v; }
                }
            }
        }
    }
    slabs
        .into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::F32(data),
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
        })
        .collect()
}

/// Reconstruct `uint32` or `int32` Predictor 1 or Predictor 2 -> `AlignedF32Slab`.
#[inline(always)]
fn check_val_f64(v: f64, finite_nodata: Option<f64>) -> (f64, bool) {
    if v.is_nan() {
        return (f64::NAN, true);
    }
    if let Some(nd) = finite_nodata
        && v == nd
    {
        return (f64::NAN, true);
    }
    (v, false)
}

#[inline(always)]
fn clamp_f64_stat(v: f64) -> f32 {
    if !v.is_finite() {
        v as f32
    } else {
        (v as f32).clamp(f32::MIN, f32::MAX)
    }
}

pub fn reconstruct_int32(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    signed: bool,
    horizontal_diff: bool,
    nodata: Option<f64>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;
    let mut slabs: Vec<Vec<f64>> = (0..bands_in_tile)
        .map(|_| vec![0.0f64; pixels])
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_val = vec![f64::INFINITY; bands_in_tile];
    let mut max_val = vec![f64::NEG_INFINITY; bands_in_tile];
    let mut acc = vec![0u32; bands_in_tile];

    for y in 0..h {
        if horizontal_diff {
            acc.fill(0);
        }
        let row_off = y * w;
        for x in 0..w {
            for b in 0..bands_in_tile {
                let off = ((row_off + x) * bands_in_tile + b) * 4;
                let bytes = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
                let raw_u32 = match byte_order {
                    ByteOrder::Little => u32::from_le_bytes(bytes),
                    ByteOrder::Big => u32::from_be_bytes(bytes),
                };
                let bits = if horizontal_diff {
                    acc[b] = acc[b].wrapping_add(raw_u32);
                    acc[b]
                } else {
                    raw_u32
                };
                let vf = if signed { (bits as i32) as f64 } else { bits as f64 };
                let (v, is_nd) = check_val_f64(vf, finite_nodata);
                slabs[b][row_off + x] = v;
                if is_nd {
                    any_nodata[b] = true;
                } else {
                    any_valid[b] = true;
                    if v < min_val[b] { min_val[b] = v; }
                    if v > max_val[b] { max_val[b] = v; }
                }
            }
        }
    }
    slabs
        .into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::F64(data.into_boxed_slice()),
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(
                any_valid[b],
                any_nodata[b],
                clamp_f64_stat(min_val[b]),
                clamp_f64_stat(max_val[b]),
            ),
        })
        .collect()
}

/// Reconstruct 64-bit unsigned (`Uint64`) or signed (`Int64`) integer tiles into `BandData::F64`,
/// supporting both `Predictor::None` and `Predictor::HorizontalInt`.
pub fn reconstruct_int64(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    signed: bool,
    horizontal_diff: bool,
    nodata: Option<f64>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;
    let mut slabs: Vec<Vec<f64>> = (0..bands_in_tile)
        .map(|_| vec![0.0f64; pixels])
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_val = vec![f64::INFINITY; bands_in_tile];
    let mut max_val = vec![f64::NEG_INFINITY; bands_in_tile];
    let mut acc = vec![0u64; bands_in_tile];

    for y in 0..h {
        if horizontal_diff {
            acc.fill(0);
        }
        let row_off = y * w;
        for x in 0..w {
            for b in 0..bands_in_tile {
                let off = ((row_off + x) * bands_in_tile + b) * 8;
                let bytes: [u8; 8] = raw[off..off + 8].try_into().unwrap();
                let raw_u64 = match byte_order {
                    ByteOrder::Little => u64::from_le_bytes(bytes),
                    ByteOrder::Big => u64::from_be_bytes(bytes),
                };
                let bits = if horizontal_diff {
                    acc[b] = acc[b].wrapping_add(raw_u64);
                    acc[b]
                } else {
                    raw_u64
                };
                let vf = if signed { (bits as i64) as f64 } else { bits as f64 };
                let (v, is_nd) = check_val_f64(vf, finite_nodata);
                slabs[b][row_off + x] = v;
                if is_nd {
                    any_nodata[b] = true;
                } else {
                    any_valid[b] = true;
                    if v < min_val[b] {
                        min_val[b] = v;
                    }
                    if v > max_val[b] {
                        max_val[b] = v;
                    }
                }
            }
        }
    }
    slabs
        .into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::F64(data.into_boxed_slice()),
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(
                any_valid[b],
                any_nodata[b],
                clamp_f64_stat(min_val[b]),
                clamp_f64_stat(max_val[b]),
            ),
        })
        .collect()
}

/// Unpack MSB-first sub-byte (`bits == 1` or `bits == 4`) TIFF tiles into compact `BandData::U8`
/// bands (`0..=1` for 1-bit, `0..=15` for 4-bit), respecting TIFF's per-scanline byte padding.
pub fn reconstruct_subbyte(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    bits: u8,
    nodata: Option<f32>,
) -> Vec<DecodedBand> {
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;
    let nd_u8: Option<u8> = nodata
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= 255.0 && v.fract() == 0.0)
        .map(|v| v as u8);
    let samples_per_row = w * bands_in_tile;
    let row_bytes = if bits == 1 {
        samples_per_row.div_ceil(8)
    } else {
        samples_per_row.div_ceil(2)
    };
    let mut bufs: Vec<Vec<u8>> = (0..bands_in_tile).map(|_| vec![0u8; pixels]).collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_val = vec![f32::INFINITY; bands_in_tile];
    let mut max_val = vec![f32::NEG_INFINITY; bands_in_tile];

    for y in 0..h {
        let row = &raw[y * row_bytes..(y + 1) * row_bytes];
        let row_px = y * w;
        for x in 0..w {
            for b in 0..bands_in_tile {
                let s = x * bands_in_tile + b;
                let v = if bits == 1 {
                    (row[s >> 3] >> (7 - (s & 7))) & 1
                } else {
                    (row[s >> 1] >> (4 * (1 - (s & 1)))) & 0x0F
                };
                bufs[b][row_px + x] = v;
                if nd_u8 == Some(v) {
                    any_nodata[b] = true;
                } else {
                    any_valid[b] = true;
                    let vf = v as f32;
                    if vf < min_val[b] {
                        min_val[b] = vf;
                    }
                    if vf > max_val[b] {
                        max_val[b] = vf;
                    }
                }
            }
        }
    }
    bufs.into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::U8 {
                data: data.into_boxed_slice(),
                nodata: nd_u8,
            },
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
        })
        .collect()
}

/// Zero-copy `float64` (`Predictor::None`) decode directly into exact `BandData::F64`.
pub fn reconstruct_f64_uncompressed(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    nodata: Option<f64>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let pixels = (tile_w as usize) * (tile_h as usize);
    let mut slabs: Vec<Vec<f64>> = (0..bands_in_tile)
        .map(|_| vec![0.0f64; pixels])
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_val = vec![f64::INFINITY; bands_in_tile];
    let mut max_val = vec![f64::NEG_INFINITY; bands_in_tile];

    for i in 0..pixels {
        for b in 0..bands_in_tile {
            let off = (i * bands_in_tile + b) * 8;
            let bytes: [u8; 8] = raw[off..off + 8].try_into().unwrap();
            let vf = match byte_order {
                ByteOrder::Little => f64::from_le_bytes(bytes),
                ByteOrder::Big => f64::from_be_bytes(bytes),
            };
            let (v, is_nd) = check_val_f64(vf, finite_nodata);
            slabs[b][i] = v;
            if is_nd {
                any_nodata[b] = true;
            } else {
                any_valid[b] = true;
                if v < min_val[b] {
                    min_val[b] = v;
                }
                if v > max_val[b] {
                    max_val[b] = v;
                }
            }
        }
    }

    slabs
        .into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::F64(data.into_boxed_slice()),
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(
                any_valid[b],
                any_nodata[b],
                clamp_f64_stat(min_val[b]),
                clamp_f64_stat(max_val[b]),
            ),
        })
        .collect()
}

/// Reconstruct `float64` (Predictor 1, Predictor 2, or 8-plane Predictor 3) -> exact `BandData::F64`.
pub fn reconstruct_f64(
    raw: &mut [u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    predictor: cogzilla_tiff::Predictor,
    nodata: Option<f64>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;
    let mut slabs: Vec<Vec<f64>> = (0..bands_in_tile)
        .map(|_| vec![0.0f64; pixels])
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_val = vec![f64::INFINITY; bands_in_tile];
    let mut max_val = vec![f64::NEG_INFINITY; bands_in_tile];

    match predictor {
        cogzilla_tiff::Predictor::FloatingPoint => {
            let samples_per_row = w * bands_in_tile;
            let row_bytes = samples_per_row * 8;
            for y in 0..h {
                let row = &mut raw[y * row_bytes..(y + 1) * row_bytes];
                prefix_sum_scalar(row, bands_in_tile);
                let row_off = y * w;
                for x in 0..w {
                    let base = x * bands_in_tile;
                    for b in 0..bands_in_tile {
                        let idx = base + b;
                        let bytes = [
                            row[7 * samples_per_row + idx],
                            row[6 * samples_per_row + idx],
                            row[5 * samples_per_row + idx],
                            row[4 * samples_per_row + idx],
                            row[3 * samples_per_row + idx],
                            row[2 * samples_per_row + idx],
                            row[samples_per_row + idx],
                            row[idx],
                        ];
                        let vf = f64::from_le_bytes(bytes);
                        let (v, is_nd) = check_val_f64(vf, finite_nodata);
                        slabs[b][row_off + x] = v;
                        if is_nd {
                            any_nodata[b] = true;
                        } else {
                            any_valid[b] = true;
                            if v < min_val[b] { min_val[b] = v; }
                            if v > max_val[b] { max_val[b] = v; }
                        }
                    }
                }
            }
        }
        cogzilla_tiff::Predictor::HorizontalInt => {
            let mut acc = vec![0u64; bands_in_tile];
            for y in 0..h {
                acc.fill(0);
                let row_off = y * w;
                for x in 0..w {
                    for b in 0..bands_in_tile {
                        let off = ((row_off + x) * bands_in_tile + b) * 8;
                        let bytes: [u8; 8] = raw[off..off + 8].try_into().unwrap();
                        let delta = match byte_order {
                            ByteOrder::Little => u64::from_le_bytes(bytes),
                            ByteOrder::Big => u64::from_be_bytes(bytes),
                        };
                        acc[b] = acc[b].wrapping_add(delta);
                        let vf = f64::from_bits(acc[b]);
                        let (v, is_nd) = check_val_f64(vf, finite_nodata);
                        slabs[b][row_off + x] = v;
                        if is_nd {
                            any_nodata[b] = true;
                        } else {
                            any_valid[b] = true;
                            if v < min_val[b] { min_val[b] = v; }
                            if v > max_val[b] { max_val[b] = v; }
                        }
                    }
                }
            }
        }
        cogzilla_tiff::Predictor::None => {
            for i in 0..pixels {
                for b in 0..bands_in_tile {
                    let off = (i * bands_in_tile + b) * 8;
                    let bytes: [u8; 8] = raw[off..off + 8].try_into().unwrap();
                    let vf = match byte_order {
                        ByteOrder::Little => f64::from_le_bytes(bytes),
                        ByteOrder::Big => f64::from_be_bytes(bytes),
                    };
                    let (v, is_nd) = check_val_f64(vf, finite_nodata);
                    slabs[b][i] = v;
                    if is_nd {
                        any_nodata[b] = true;
                    } else {
                        any_valid[b] = true;
                        if v < min_val[b] { min_val[b] = v; }
                        if v > max_val[b] { max_val[b] = v; }
                    }
                }
            }
        }
    }

    slabs
        .into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::F64(data.into_boxed_slice()),
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(
                any_valid[b],
                any_nodata[b],
                clamp_f64_stat(min_val[b]),
                clamp_f64_stat(max_val[b]),
            ),
        })
        .collect()
}

/// Reconstruct `uint8` Predictor 1 or Predictor 2 (Sentinel-2 TCI, visual RGB, masks) -> compact `BandData::U8` (1 B/px).
pub fn reconstruct_u8(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    horizontal_diff: bool,
    nodata: Option<f32>,
) -> Vec<DecodedBand> {
    let finite_nodata = nodata.filter(|v| !v.is_nan());
    let nd_u8 = finite_nodata.and_then(|nd| {
        if nd >= 0.0 && nd <= (u8::MAX as f32) && (nd as u8) as f32 == nd {
            Some(nd as u8)
        } else {
            None
        }
    });
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;

    // Zero-deinterleave fast path for chunky 3-band uint8 COGs (Sentinel-2 TCI, NAIP, JPEG/WebP RGB):
    // allocate a single shared Arc<[u8]> directly (undoing horizontal differencing in-place via
    // Arc::get_mut when enabled), eliminating a redundant 768 KiB Vec<u8> -> Arc<[u8]> copy!
    if bands_in_tile == 3 {
        let mut rgb: std::sync::Arc<[u8]> = std::sync::Arc::from(&raw[..pixels * 3]);
        let mut any_valid = [false; 3];
        let mut any_nodata = [false; 3];
        let mut min_u8 = [255u8; 3];
        let mut max_u8 = [0u8; 3];

        #[cfg(target_arch = "aarch64")]
        let handled = {
            let slice = std::sync::Arc::get_mut(&mut rgb).expect("unique Arc");
            unsafe {
                reconstruct_u8_rgb3_neon(
                    slice,
                    w,
                    h,
                    horizontal_diff,
                    nd_u8,
                    &mut any_valid,
                    &mut any_nodata,
                    &mut min_u8,
                    &mut max_u8,
                );
            }
            true
        };

        #[cfg(target_arch = "x86_64")]
        let handled = if std::is_x86_feature_detected!("avx2") {
            let slice = std::sync::Arc::get_mut(&mut rgb).expect("unique Arc");
            unsafe {
                reconstruct_u8_rgb3_avx2(
                    slice,
                    w,
                    h,
                    horizontal_diff,
                    nd_u8,
                    &mut any_valid,
                    &mut any_nodata,
                    &mut min_u8,
                    &mut max_u8,
                );
            }
            true
        } else {
            false
        };

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let handled = false;

        if !handled {
            if horizontal_diff {
                let slice = std::sync::Arc::get_mut(&mut rgb).expect("unique Arc");
                for y in 0..h {
                    let row = &mut slice[y * w * 3..(y + 1) * w * 3];
                    let mut acc = [0u8; 3];
                    for px in row.chunks_exact_mut(3) {
                        acc[0] = acc[0].wrapping_add(px[0]);
                        acc[1] = acc[1].wrapping_add(px[1]);
                        acc[2] = acc[2].wrapping_add(px[2]);
                        px[0] = acc[0];
                        px[1] = acc[1];
                        px[2] = acc[2];
                    }
                }
            }

            if let Some(nd) = nd_u8 {
                for px in rgb.chunks_exact(3) {
                    for b in 0..3 {
                        let v = px[b];
                        if v == nd {
                            any_nodata[b] = true;
                        } else {
                            any_valid[b] = true;
                            if v < min_u8[b] { min_u8[b] = v; }
                            if v > max_u8[b] { max_u8[b] = v; }
                        }
                    }
                }
            } else {
                any_valid = [pixels > 0; 3];
                for px in rgb.chunks_exact(3) {
                    if px[0] < min_u8[0] { min_u8[0] = px[0]; }
                    if px[0] > max_u8[0] { max_u8[0] = px[0]; }
                    if px[1] < min_u8[1] { min_u8[1] = px[1]; }
                    if px[1] > max_u8[1] { max_u8[1] = px[1]; }
                    if px[2] < min_u8[2] { min_u8[2] = px[2]; }
                    if px[2] > max_u8[2] { max_u8[2] = px[2]; }
                }
            }
        }
        return (0..3u8)
            .map(|b| {
                let bi = b as usize;
                DecodedBand {
                    data: BandData::RgbU8Channel {
                        rgb: rgb.clone(),
                        channel: b,
                        nodata: nd_u8,
                    },
                    width: tile_w,
                    height: tile_h,
                    stats: finalize_stats(
                        any_valid[bi],
                        any_nodata[bi],
                        if any_valid[bi] { min_u8[bi] as f32 } else { f32::INFINITY },
                        if any_valid[bi] { max_u8[bi] as f32 } else { f32::NEG_INFINITY },
                    ),
                }
            })
            .collect();
    }

    // Fast path: 1-band u8 (Predictor 1 & 2) using memcpy + SIMD prefix-sum + integer min/max.
    if bands_in_tile == 1 {
        let mut data: Box<[u8]> = Box::from(&raw[..pixels]);
        if horizontal_diff {
            for y in 0..h {
                let row = &mut data[y * w..(y + 1) * w];
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    prefix_sum_stride1_neon(row);
                }
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            prefix_sum_stride1_avx2(row);
                        }
                    } else {
                        let mut acc = 0u8;
                        for v in row.iter_mut() {
                            acc = acc.wrapping_add(*v);
                            *v = acc;
                        }
                    }
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                {
                    let mut acc = 0u8;
                    for v in row.iter_mut() {
                        acc = acc.wrapping_add(*v);
                        *v = acc;
                    }
                }
            }
        }
        let mut any_valid = false;
        let mut any_nodata = false;
        let mut min_u = u8::MAX;
        let mut max_u = u8::MIN;
        if let Some(nd) = nd_u8 {
            for &v in data.iter() {
                if v == nd {
                    any_nodata = true;
                } else {
                    any_valid = true;
                    if v < min_u { min_u = v; }
                    if v > max_u { max_u = v; }
                }
            }
        } else {
            for &v in data.iter() {
                if v < min_u { min_u = v; }
                if v > max_u { max_u = v; }
            }
            any_valid = pixels > 0;
        }
        return vec![DecodedBand {
            data: BandData::U8 { data, nodata: nd_u8 },
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(
                any_valid,
                any_nodata,
                if any_valid { min_u as f32 } else { f32::INFINITY },
                if any_valid { max_u as f32 } else { f32::NEG_INFINITY },
            ),
        }];
    }

    let mut slabs: Vec<Box<[u8]>> = (0..bands_in_tile)
        .map(|_| vec![0u8; pixels].into_boxed_slice())
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_u = vec![u8::MAX; bands_in_tile];
    let mut max_u = vec![u8::MIN; bands_in_tile];
    let mut acc = vec![0u8; bands_in_tile];

    for y in 0..h {
        if horizontal_diff {
            acc.fill(0);
        }
        let row_off = y * w;
        for x in 0..w {
            for b in 0..bands_in_tile {
                let off = (row_off + x) * bands_in_tile + b;
                let sample = raw[off];
                let val_u8 = if horizontal_diff {
                    acc[b] = acc[b].wrapping_add(sample);
                    acc[b]
                } else {
                    sample
                };
                slabs[b][row_off + x] = val_u8;
                if nd_u8 == Some(val_u8) {
                    any_nodata[b] = true;
                } else {
                    any_valid[b] = true;
                    if val_u8 < min_u[b] { min_u[b] = val_u8; }
                    if val_u8 > max_u[b] { max_u[b] = val_u8; }
                }
            }
        }
    }

    slabs
        .into_iter()
        .enumerate()
        .map(|(b, data)| DecodedBand {
            data: BandData::U8 { data, nodata: nd_u8 },
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(
                any_valid[b],
                any_nodata[b],
                if any_valid[b] { min_u[b] as f32 } else { f32::INFINITY },
                if any_valid[b] { max_u[b] as f32 } else { f32::NEG_INFINITY },
            ),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_rgb3_pred1_and_pred2_match_scalar_across_widths_and_nodata() {
        for &(w, h) in &[(1u32, 1u32), (7, 3), (15, 2), (16, 4), (17, 5), (63, 7), (256, 4)] {
            let pixels = (w * h) as usize;
            let raw: Vec<u8> = (0..pixels * 3)
                .map(|i| ((i.wrapping_mul(37) ^ (i >> 3)) & 0xFF) as u8)
                .collect();

            for horizontal_diff in [false, true] {
                for nodata in [None, Some(0.0f32), Some(255.0f32)] {
                    let bands = reconstruct_u8(&raw, w, h, 3, horizontal_diff, nodata);
                    assert_eq!(bands.len(), 3);

                    // Scalar reference
                    let mut expected = raw.clone();
                    if horizontal_diff {
                        for y in 0..h as usize {
                            let row = &mut expected[y * (w as usize) * 3..(y + 1) * (w as usize) * 3];
                            let mut acc = [0u8; 3];
                            for px in row.chunks_exact_mut(3) {
                                for b in 0..3 {
                                    acc[b] = acc[b].wrapping_add(px[b]);
                                    px[b] = acc[b];
                                }
                            }
                        }
                    }
                    let nd_u8 = nodata.map(|v| v as u8);
                    let mut av = [false; 3];
                    let mut an = [false; 3];
                    let mut mn = [255u8; 3];
                    let mut mx = [0u8; 3];
                    for px in expected.chunks_exact(3) {
                        for b in 0..3 {
                            if nd_u8 == Some(px[b]) {
                                an[b] = true;
                            } else {
                                av[b] = true;
                                mn[b] = mn[b].min(px[b]);
                                mx[b] = mx[b].max(px[b]);
                            }
                        }
                    }

                    for b in 0..3 {
                        let BandData::RgbU8Channel { ref rgb, .. } = bands[b].data else {
                            panic!("expected RgbU8Channel");
                        };
                        assert_eq!(rgb.as_ref(), expected.as_slice());
                        let exp_stats = finalize_stats(
                            av[b],
                            an[b],
                            if av[b] { mn[b] as f32 } else { f32::INFINITY },
                            if av[b] { mx[b] as f32 } else { f32::NEG_INFINITY },
                        );
                        assert_eq!(bands[b].stats.all_nodata, exp_stats.all_nodata);
                        assert_eq!(bands[b].stats.no_nodata, exp_stats.no_nodata);
                        assert_eq!(bands[b].stats.min.to_bits(), exp_stats.min.to_bits());
                        assert_eq!(bands[b].stats.max.to_bits(), exp_stats.max.to_bits());
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn prefix_sum_stride3_neon_matches_scalar() {
        for len in [0usize, 3, 6, 45, 48, 51, 96, 1536, 6144] {
            let input: Vec<u8> = (0..len)
                .map(|i| ((i.wrapping_mul(29) ^ (i >> 2)) & 0xFF) as u8)
                .collect();
            let mut a = input.clone();
            let mut b = input;
            prefix_sum_scalar(&mut a, 3);
            unsafe { prefix_sum_stride3_neon(&mut b) };
            assert_eq!(a, b, "mismatch for len={len}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn prefix_sum_stride3_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in [0usize, 3, 6, 12, 15, 18, 24, 48, 51, 96, 1536, 6144] {
            let input: Vec<u8> = (0..len)
                .map(|i| ((i.wrapping_mul(29) ^ (i >> 2)) & 0xFF) as u8)
                .collect();
            let mut a = input.clone();
            let mut b = input;
            prefix_sum_scalar(&mut a, 3);
            unsafe { prefix_sum_stride3_avx2(&mut b) };
            assert_eq!(a, b, "mismatch for len={len}");
        }
    }
}
