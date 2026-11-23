use cogwheel_tiff::ByteOrder;

use crate::slab::{AlignedF32Slab, BlockStats, DecodedBand};

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

// ============================================================================
// x86_64 AVX2 Kernels (AMD EPYC / Intel Xeon Cloud Servers)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
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
            data: slab,
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

        for y in 0..h {
            let row = &mut raw[y * row_bytes..(y + 1) * row_bytes];
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
            #[cfg(not(target_arch = "aarch64"))]
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
                data: s0,
                width: tile_w,
                height: tile_h,
                stats: finalize_stats(any_valid[0], any_nodata[0], min_val[0], max_val[0]),
            },
            DecodedBand {
                data: s1,
                width: tile_w,
                height: tile_h,
                stats: finalize_stats(any_valid[1], any_nodata[1], min_val[1], max_val[1]),
            },
            DecodedBand {
                data: s2,
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
                data,
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
    let pixels = w * h;

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
                data: slab,
                width: tile_w,
                height: tile_h,
                stats: finalize_stats(any_valid, any_nodata, min_val, max_val),
            }];
        }

        #[cfg(target_arch = "x86_64")]
        if byte_order == ByteOrder::Little && std::is_x86_feature_detected!("avx2") {
            let stats = unsafe { reconstruct_predictor1_le_avx2(raw, slab.as_mut_slice(), finite_nodata) };
            return vec![DecodedBand {
                data: slab,
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
            data: slab,
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
                data,
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
            data,
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
        })
        .collect()
}

// ============================================================================
// uint16 Decoders (Landsat Collection 2 & Sentinel-2 L2A -> AlignedF32Slab)
// ============================================================================

/// SIMD 8-lane `u16` horizontal prefix-sum + widen to `f32` + nodata/stats on ARM64 NEON.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn u16_pred2_row_neon(
    src_u16: &[u8],
    dst_f32: &mut [f32],
    finite_nodata: Option<f32>,
    any_valid: &mut bool,
    any_nodata: &mut bool,
    min_val: &mut f32,
    max_val: &mut f32,
) {
    use std::arch::aarch64::*;
    let w = dst_f32.len();
    let zero = unsafe { vdupq_n_u16(0) };
    let mut carry = zero;
    let mut v_min = unsafe { vdupq_n_f32(f32::INFINITY) };
    let mut v_max = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    let mut v_av = unsafe { vdupq_n_u32(0) };
    let mut v_an = unsafe { vdupq_n_u32(0) };
    let v_nan = unsafe { vdupq_n_f32(f32::NAN) };
    let has_nd = finite_nodata.is_some();
    let v_nd = unsafe { vdupq_n_f32(finite_nodata.unwrap_or(0.0)) };

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

            let mut f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v)));
            let mut f1 = vcvtq_f32_u32(vmovl_high_u16(v));

            macro_rules! proc_u16f {
                ($fv:expr) => {{
                    if has_nd {
                        let valid = vmvnq_u32(vceqq_f32($fv, v_nd));
                        v_av = vorrq_u32(v_av, valid);
                        v_an = vorrq_u32(v_an, vmvnq_u32(valid));
                        $fv = vbslq_f32(valid, $fv, v_nan);
                    } else {
                        v_av = vdupq_n_u32(1);
                    }
                    v_min = vminnmq_f32(v_min, $fv);
                    v_max = vmaxnmq_f32(v_max, $fv);
                }};
            }

            proc_u16f!(f0);
            proc_u16f!(f1);

            vst1q_f32(dst_f32.as_mut_ptr().add(i), f0);
            vst1q_f32(dst_f32.as_mut_ptr().add(i + 4), f1);
        }
        i += 8;
    }

    unsafe {
        if vmaxvq_u32(v_av) != 0 {
            *any_valid = true;
            let mn = vminnmvq_f32(v_min);
            let mx = vmaxnmvq_f32(v_max);
            if mn < *min_val { *min_val = mn; }
            if mx > *max_val { *max_val = mx; }
        }
        if vmaxvq_u32(v_an) != 0 {
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
            let (v, is_nd) = check_val(acc as f32, finite_nodata);
            dst_f32[i] = v;
            if is_nd {
                *any_nodata = true;
            } else {
                *any_valid = true;
                if v < *min_val { *min_val = v; }
                if v > *max_val { *max_val = v; }
            }
            i += 1;
        }
    }
}

/// Reconstruct `uint16` Predictor 1 (None) or Predictor 2 (16-bit horizontal diff, Landsat C2)
/// directly into planar 64-byte-aligned `f32` slabs with `BlockStats`.
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
    let w = tile_w as usize;
    let h = tile_h as usize;
    let pixels = w * h;

    // Fast path: single-band little-endian u16 + Predictor 2 (Landsat Collection 2 & Sentinel-2!)
    #[cfg(target_arch = "aarch64")]
    if bands_in_tile == 1 && byte_order == ByteOrder::Little && horizontal_diff {
        let mut slab = AlignedF32Slab::new_uninit(pixels);
        let mut any_valid = false;
        let mut any_nodata = false;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let row_bytes = w * 2;
        for y in 0..h {
            unsafe {
                u16_pred2_row_neon(
                    &raw[y * row_bytes..(y + 1) * row_bytes],
                    &mut slab[y * w..(y + 1) * w],
                    finite_nodata,
                    &mut any_valid,
                    &mut any_nodata,
                    &mut min_val,
                    &mut max_val,
                );
            }
        }
        return vec![DecodedBand {
            data: slab,
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid, any_nodata, min_val, max_val),
        }];
    }

    // General path (Predictor 1, Big-Endian, x86_64, or multi-band chunky u16)
    let mut slabs: Vec<AlignedF32Slab> = (0..bands_in_tile)
        .map(|_| AlignedF32Slab::new_uninit(pixels))
        .collect();
    let mut any_valid = vec![false; bands_in_tile];
    let mut any_nodata = vec![false; bands_in_tile];
    let mut min_val = vec![f32::INFINITY; bands_in_tile];
    let mut max_val = vec![f32::NEG_INFINITY; bands_in_tile];
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
                let (v, is_nd) = check_val(val_u16 as f32, finite_nodata);
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
            data,
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
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
            data,
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
            data,
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
        })
        .collect()
}

/// Reconstruct `uint32` or `int32` Predictor 1 or Predictor 2 -> `AlignedF32Slab`.
pub fn reconstruct_int32(
    raw: &[u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    signed: bool,
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
                let vf = if signed { (bits as i32) as f32 } else { bits as f32 };
                let (v, is_nd) = check_val(vf, finite_nodata);
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
            data,
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
        })
        .collect()
}

/// Reconstruct `float64` (Predictor 1, Predictor 2, or 8-plane Predictor 3) -> `AlignedF32Slab`.
pub fn reconstruct_f64(
    raw: &mut [u8],
    tile_w: u32,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    predictor: cogwheel_tiff::Predictor,
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

    match predictor {
        cogwheel_tiff::Predictor::FloatingPoint => {
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
                        let vf = f64::from_le_bytes(bytes) as f32;
                        let (v, is_nd) = check_val(vf, finite_nodata);
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
        cogwheel_tiff::Predictor::HorizontalInt => {
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
                        let vf = f64::from_bits(acc[b]) as f32;
                        let (v, is_nd) = check_val(vf, finite_nodata);
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
        cogwheel_tiff::Predictor::None => {
            for i in 0..pixels {
                for b in 0..bands_in_tile {
                    let off = (i * bands_in_tile + b) * 8;
                    let bytes: [u8; 8] = raw[off..off + 8].try_into().unwrap();
                    let vf = match byte_order {
                        ByteOrder::Little => f64::from_le_bytes(bytes) as f32,
                        ByteOrder::Big => f64::from_be_bytes(bytes) as f32,
                    };
                    let (v, is_nd) = check_val(vf, finite_nodata);
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
            data,
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
        })
        .collect()
}

/// Reconstruct `uint8` Predictor 1 or Predictor 2 (Sentinel-2 TCI, visual RGB, masks) -> `AlignedF32Slab`.
pub fn reconstruct_u8(
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
                let (v, is_nd) = check_val(val_u8 as f32, finite_nodata);
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
            data,
            width: tile_w,
            height: tile_h,
            stats: finalize_stats(any_valid[b], any_nodata[b], min_val[b], max_val[b]),
        })
        .collect()
}
