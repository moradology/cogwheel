//! Terrain elevation RGB encodings (Terrarium, Terrain-RGB, Contours) and DEM derivative
//! rendering (Hillshade, Slope, Aspect) with 1-pixel seamless boundary padding.

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_proj::CoordPlan;

use crate::colormap::nodata_index;
use crate::jpeg_fast::encode_jpeg_rgb;
use crate::png_fast::{
    encode_filtered_png, encode_indexed_png, filter_sub_ga_row, filter_sub_rgb_row,
    filter_sub_rgba_row, filter_sub_u8_row,
};
use crate::source::BandSource;
use crate::webp_fast::encode_webp_rgb;

use super::{
    OutputFormat, ResampleMode, SCRATCH, SingleBandStyle, WidenedSampler,
    bilinear_row_flat_dispatch, build_plte_nodata, build_plte_opaque, coord_plan_scale,
    empty_tile_bytes, encode_rgba_tile, fast_floor_i32, fill_uv_row, for_each_pixel_in_row,
    plan_widened_scale, rescale_u8, sample_bilinear, sample_cubic, sample_general_unwidened,
    sample_kernel_widened, sample_nearest,
};

pub use cogzilla_raster::DemAlgorithm;

#[inline(always)]
pub fn encode_terrain_pixel(algorithm: DemAlgorithm, elev_m: f32) -> [u8; 3] {
    match algorithm {
        DemAlgorithm::Terrarium => encode_terrarium_rgb(elev_m),
        DemAlgorithm::TerrainRgb => encode_terrain_rgb(elev_m),
        DemAlgorithm::Contours {
            increment,
            thickness,
            minz,
            maxz,
        } => encode_contours_rgb(elev_m, increment, thickness, minz, maxz),
        _ => [0, 0, 0],
    }
}

#[inline(always)]
pub fn encode_contours_rgb(
    elev_m: f32,
    increment: f32,
    thickness: u32,
    minz: f32,
    maxz: f32,
) -> [u8; 3] {
    let data = elev_m as f64;
    if increment > 0.0 && thickness > 0 && data.rem_euclid(increment as f64) < thickness as f64 {
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
    algorithm: DemAlgorithm,
    resample: ResampleMode,
    format: OutputFormat,
) -> Bytes {
    if src.all_nodata {
        return empty_tile_bytes(N as u32, format, 3);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = plan.all_in_bounds() && src.is_clean_inbounds(c0, c1, r0, r1);

    let encode_px = |v: f32| -> [u8; 3] { encode_terrain_pixel(algorithm, v) };
    let (scale_x, scale_y) = coord_plan_scale(plan);
    let widen = resample != ResampleMode::Nearest && plan_widened_scale(plan, resample).is_some();
    let wide = widen.then(|| WidenedSampler::new(resample, scale_x, scale_y));

    let sample_px = |u: f32, v: f32| -> f32 {
        match (resample, clean) {
            (
                ResampleMode::Bilinear | ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Lanczos,
                _,
            ) if widen => wide.as_ref().map_or(f32::NAN, |w| w.sample(src, u, v)),
            (ResampleMode::Bilinear, true) => sample_bilinear::<true>(src, u, v),
            (ResampleMode::Bilinear, false) => sample_bilinear::<false>(src, u, v),
            (ResampleMode::Nearest, true) => sample_nearest::<true>(src, u, v),
            (ResampleMode::Nearest, false) => sample_nearest::<false>(src, u, v),
            (ResampleMode::Cubic, _) => sample_cubic(src, u, v),
            (other, _) => sample_general_unwidened(src, u, v, scale_x, scale_y, other),
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
                let flat_dem = if resample == ResampleMode::Bilinear && !widen {
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
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
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

/// Rescale denominator preserving sign for inverted stretches (`?rescale=255,0`) and negative
/// GDAL `scale` (`?unscale=true`), while avoiding division by zero on degenerate `min == max`.
#[inline(always)]
pub fn safe_rescale_denom(min: f32, max: f32) -> f32 {
    let d = max - min;
    if d.abs() < 1e-12 { 1e-12 } else { d }
}

/// numpy's gradient along one axis at a pixel `c` between `lo` and `hi`, in elevation per
/// pixel: the centred difference over two pixels, or the one-sided difference over one where
/// a neighbour is missing, as numpy takes it on an array's edge.
#[inline(always)]
pub fn dem_gradient(lo: f32, c: f32, hi: f32) -> f32 {
    match (!lo.is_finite(), !hi.is_finite()) {
        (false, false) => 0.5 * (hi - lo),
        (true, false) => hi - c,
        (false, true) => c - lo,
        (true, true) => 0.0,
    }
}

/// Evaluate a DEM algorithm at the pixel `z_c` from its four neighbours (North, West, East,
/// South), with the inverse pixel sizes East-West (`inv_dx`) and North-South (`inv_dy`) in the
/// grid's own units (TiTiler divides by the transform's pixel size). Hillshade, as TiTiler's,
/// takes the gradient per pixel and does not use them.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn eval_dem_stencil(
    n: f32,
    w: f32,
    z_c: f32,
    e: f32,
    s: f32,
    inv_dx: f32,
    inv_dy: f32,
    alg: DemAlgorithm,
) -> f32 {
    if !z_c.is_finite() {
        return f32::NAN;
    }
    // Per pixel, East-going and South-going.
    let gx = dem_gradient(w, z_c, e);
    let gy = dem_gradient(n, z_c, s);

    match alg {
        DemAlgorithm::Hillshade {
            azimuth_deg,
            altitude_deg,
            z_factor,
        } => {
            // TiTiler lights `sin(alt)·sin(slope) + cos(alt)·cos(slope)·cos(az - aspect)` and
            // writes `255·(shaded + 1) / 2` as bytes (truncated, before any rescale); the dot
            // product below is the same in closed form.
            let zenith = (90.0 - altitude_deg).to_radians();
            // Sun vector in East/North/Up coordinates:
            // azimuth_deg is clockwise from North (0=N, 90=E, 315=NW).
            let az = azimuth_deg.to_radians();
            let sun_e = az.sin() * zenith.sin();
            let sun_n = az.cos() * zenith.sin();
            let sun_u = zenith.cos();
            // Surface normal in (East, North, Up) = (-z*gx, +z*gy, 1) normalized
            let nx = -z_factor * gx;
            let ny = z_factor * gy;
            let dot = (nx * sun_e + ny * sun_n + sun_u) / (nx * nx + ny * ny + 1.0).sqrt();
            (dot * 127.5 + 127.5).clamp(0.0, 255.0).trunc()
        }
        DemAlgorithm::Slope { z_factor } => {
            let dz_dx = gx * inv_dx;
            let dz_dy = gy * inv_dy;
            let grad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt() * z_factor;
            grad.atan().to_degrees()
        }
        DemAlgorithm::Aspect => {
            let dz_dx = gx * inv_dx;
            let dz_dy = gy * inv_dy;
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
            // A missing neighbour reads as the pixel itself, so no line is drawn toward it.
            let fill = |v: f32| if v.is_nan() { z_c } else { v };
            let (n, s, e, w) = (fill(n), fill(s), fill(e), fill(w));
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
        DemAlgorithm::Terrarium | DemAlgorithm::TerrainRgb => z_c,
    }
}

/// SIMD hillshade row evaluator for clean (`no_nodata`) DEM tiles: TiTiler's shading from
/// numpy's centred gradient per pixel, 4 pixels/iter on ARM64 NEON or 8 pixels/iter on x86-64
/// AVX2, from contiguous loads of `row0`, `row1`, `row2` (`N + 2` floats each, the rows above,
/// at and below), writing rescaled `u8` indices. Clean tiles hold no NaN (block statistics
/// count NaN as nodata), so no neighbour is missing.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn eval_hillshade_row_clean<const N: usize>(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
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
    // The centred difference halves the step across two pixels.
    let neg_z_scale = -0.5 * z_factor;
    let pos_z_scale = 0.5 * z_factor;
    #[allow(unused_assignments, unused_mut)]
    let mut start_px = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let v_one = vdupq_n_f32(1.0);
        let v_zero = vdupq_n_f32(0.0);
        let v_255 = vdupq_n_f32(255.0);
        let v_half_255 = vdupq_n_f32(127.5);
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
        let out_ptr = out_idx.as_mut_ptr();
        let mut px = 0usize;
        while px + 4 <= N {
            let n = vld1q_f32(r0.add(px + 1));
            let w = vld1q_f32(r1.add(px));
            let e = vld1q_f32(r1.add(px + 2));
            let s = vld1q_f32(r2.add(px + 1));

            let nx = vmulq_f32(vsubq_f32(e, w), v_neg_zs);
            let ny = vmulq_f32(vsubq_f32(s, n), v_pos_zs);

            let len_sq = vfmaq_f32(vfmaq_f32(v_one, nx, nx), ny, ny);
            let num = vfmaq_f32(vfmaq_f32(v_su, nx, v_se), ny, v_sn);
            let dot = vdivq_f32(num, vsqrtq_f32(len_sq));
            let hs = vrndq_f32(vminq_f32(vmaxq_f32(vfmaq_f32(v_half_255, dot, v_half_255), v_zero), v_255));
            let scaled = vminq_f32(vmaxq_f32(vmulq_f32(vsubq_f32(hs, v_min), v_inv_r), v_zero), v_255);
            // Narrow 4x u32 -> 4x u8 in registers and emit a single unaligned u32 store.
            let u32_v = vcvtq_u32_f32(scaled);
            let u16_v = vmovn_u32(u32_v);
            let u8_v = vmovn_u16(vcombine_u16(u16_v, u16_v));
            let packed = vget_lane_u32::<0>(vreinterpret_u32_u8(u8_v));
            (out_ptr.add(px) as *mut u32).write_unaligned(packed);
            px += 4;
        }
        start_px = px;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                eval_hillshade_row_clean_avx2::<N>(
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
        let val = eval_dem_stencil(
            row0[c],
            row1[c - 1],
            row1[c],
            row1[c + 1],
            row2[c],
            1.0,
            1.0,
            DemAlgorithm::Hillshade { azimuth_deg, altitude_deg, z_factor },
        );
        out_idx[px] = rescale_u8(val, inv_range, min_val);
    }
}

// SAFETY: Caller checks AVX2+FMA and guarantees 3x3 stencil rows (`r0`, `r1`, `r2`) have `>= N + 2` elements and `out_idx.len() >= N`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn eval_hillshade_row_clean_avx2<const N: usize>(
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
    let v_one = _mm256_set1_ps(1.0);
    let v_zero = _mm256_setzero_ps();
    let v_255 = _mm256_set1_ps(255.0);
    let v_half_255 = _mm256_set1_ps(127.5);
    let v_neg_zs = _mm256_set1_ps(neg_z_scale);
    let v_pos_zs = _mm256_set1_ps(pos_z_scale);
    let v_se = _mm256_set1_ps(sun_e);
    let v_sn = _mm256_set1_ps(sun_n);
    let v_su = _mm256_set1_ps(sun_u);
    let v_inv_r = _mm256_set1_ps(inv_range);
    let v_min = _mm256_set1_ps(min_val);

    let out_ptr = out_idx.as_mut_ptr();
    let mut px = 0usize;
    while px + 8 <= N {
        // SAFETY: `px + 8 <= N` guarantees `px + 2 + 8 <= N + 2` readable `f32`s in `r0..r2`.
        let (n, w, e, s) = unsafe {
            (
                _mm256_loadu_ps(r0.add(px + 1)),
                _mm256_loadu_ps(r1.add(px)),
                _mm256_loadu_ps(r1.add(px + 2)),
                _mm256_loadu_ps(r2.add(px + 1)),
            )
        };

        let nx = _mm256_mul_ps(_mm256_sub_ps(e, w), v_neg_zs);
        let ny = _mm256_mul_ps(_mm256_sub_ps(s, n), v_pos_zs);

        let len_sq = _mm256_fmadd_ps(ny, ny, _mm256_fmadd_ps(nx, nx, v_one));
        let num = _mm256_fmadd_ps(ny, v_sn, _mm256_fmadd_ps(nx, v_se, v_su));
        // An exact square root and division, as the scalar path: levels are truncated, so
        // an approximate reciprocal would move pixels across a level.
        let dot = _mm256_div_ps(num, _mm256_sqrt_ps(len_sq));
        let hs = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(_mm256_min_ps(
            _mm256_max_ps(_mm256_fmadd_ps(dot, v_half_255, v_half_255), v_zero),
            v_255,
        ));
        let scaled = _mm256_min_ps(
            _mm256_max_ps(_mm256_mul_ps(_mm256_sub_ps(hs, v_min), v_inv_r), v_zero),
            v_255,
        );
        // Narrow 8x i32 -> 8x u8 in registers and emit a single unaligned u64 store.
        let i32_v = _mm256_cvttps_epi32(scaled);
        let lo128 = _mm256_castsi256_si128(i32_v);
        let hi128 = _mm256_extracti128_si256::<1>(i32_v);
        let u16_v = _mm_packus_epi32(lo128, hi128);
        let u8_v = _mm_packus_epi16(u16_v, u16_v);
        let packed = _mm_cvtsi128_si64(u8_v) as u64;
        // SAFETY: `px + 8 <= N <= out_idx.len()` guarantees 8 writable bytes at `out_ptr.add(px)`.
        unsafe { (out_ptr.add(px) as *mut u64).write_unaligned(packed) };
        px += 8;
    }
}

/// Render an on-the-fly DEM derivative tile (Hillshade, Slope, or Aspect) from numpy's
/// gradient over a 1-pixel-padded resampled elevation buffer for seamless tile boundaries.
/// `pixel_size` is the tile's pixel size in its own system's units (Web Mercator metres).
pub fn render_dem_algorithm<const N: usize>(
    plan: &CoordPlan<N>,
    src: &BandSource<'_>,
    alg: DemAlgorithm,
    pixel_size: f32,
    style: &SingleBandStyle,
) -> Bytes {
    if src.all_nodata {
        return empty_tile_bytes(N as u32, style.format, 1);
    }
    let (c0, c1, r0, r1) = plan.src_pixel_bounds();
    let clean = plan.all_in_bounds() && src.is_clean_inbounds(c0, c1, r0, r1);
    let (scale_x, scale_y) = coord_plan_scale(plan);
    let widen = style.resample != ResampleMode::Nearest && plan_widened_scale(plan, style.resample).is_some();
    let wide = widen.then(|| WidenedSampler::new(style.resample, scale_x, scale_y));

    let sample_px = |u: f32, v: f32| -> f32 {
        match (style.resample, clean) {
            (
                ResampleMode::Bilinear | ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Lanczos,
                _,
            ) if widen => wide.as_ref().map_or(f32::NAN, |w| w.sample(src, u, v)),
            (ResampleMode::Bilinear, true) => sample_bilinear::<true>(src, u, v),
            (ResampleMode::Bilinear, false) => sample_bilinear::<false>(src, u, v),
            (ResampleMode::Nearest, true) => sample_nearest::<true>(src, u, v),
            (ResampleMode::Nearest, false) => sample_nearest::<false>(src, u, v),
            (ResampleMode::Cubic, _) => sample_cubic(src, u, v),
            (other, _) => sample_general_unwidened(src, u, v, scale_x, scale_y, other),
        }
    };
    // Sub-pixel bounds-checked sampler for the 1-pixel extrapolated border so tile edges
    // match neighboring tiles' interior pixels with zero staircase seam.
    let sample_border = |u: f32, v: f32| -> f32 {
        match style.resample {
            ResampleMode::Bilinear | ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Lanczos
                if widen =>
            {
                sample_kernel_widened(src, u, v, scale_x, scale_y, style.resample)
            }
            ResampleMode::Bilinear => sample_bilinear::<false>(src, u, v),
            ResampleMode::Nearest => sample_nearest::<false>(src, u, v),
            ResampleMode::Cubic => sample_cubic(src, u, v),
            other => sample_general_unwidened(src, u, v, scale_x, scale_y, other),
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

        let flat_dem = if style.resample == ResampleMode::Bilinear && clean && !widen {
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

        // The branchless SIMD hillshade path assumes every 4-neighbour (including the 1-px
        // extrapolated halo) is finite so centred differences match `dem_gradient`. If a clean
        // tile abuts a DEM raster edge, leave those border cells as NaN (do NOT clamp them,
        // which would halve one-sided edge gradients) and fall back to scalar `eval_dem_stencil`.
        let mut halo_clean = clean;
        if halo_clean {
            for py in 0..N {
                let row_off = (py + 1) * pad;
                if !elev[row_off].is_finite() || !elev[row_off + N + 1].is_finite() {
                    halo_clean = false;
                    break;
                }
            }
        }
        if halo_clean {
            for px in 0..N {
                let c = px + 1;
                if !elev[c].is_finite() || !elev[(N + 1) * pad + c].is_finite() {
                    halo_clean = false;
                    break;
                }
            }
        }

        let inv_dx = 1.0 / pixel_size.max(1e-9);
        let inv_range = 255.0 / safe_rescale_denom(style.rescale_min, style.rescale_max);
        let min_val = style.rescale_min;
        let lut = style.colormap.lut();

        match style.format {
            OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
                let mut out = BytesMut::with_capacity(N * N * 4);
                for py in 0..N {
                    let r0 = py * pad;
                    let r1 = (py + 1) * pad;
                    let r2 = (py + 2) * pad;
                    for px in 0..N {
                        let c = px + 1;
                        let val = eval_dem_stencil(elev[r0 + c], elev[r1 + c - 1], elev[r1 + c], elev[r1 + c + 1], elev[r2 + c], inv_dx, inv_dx, alg);
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
                if halo_clean
                    && let DemAlgorithm::Hillshade {
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
                            eval_hillshade_row_clean::<N>(
                                &elev[r0..r0 + pad],
                                &elev[r1..r1 + pad],
                                &elev[r2..r2 + pad],
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
                            plte,
                            None,
                            &sc.img_buf[..total],
                        );
                    }
                let is_opaque_dem = clean && !matches!(alg, DemAlgorithm::Aspect);
                let is_gray = matches!(style.colormap, crate::colormap::ColormapName::Grayscale);
                if is_gray && !is_opaque_dem {
                    let ga_stride = 1 + N * 2;
                    let ga_total = N * ga_stride;
                    if sc.img_buf.len() < ga_total {
                        sc.img_buf.resize(ga_total, 0);
                    }
                    let mut row_ga = vec![0u8; N * 2];
                    for py in 0..N {
                        let r0 = py * pad;
                        let r1 = (py + 1) * pad;
                        let r2 = (py + 2) * pad;
                        for px in 0..N {
                            let c = px + 1;
                            let val = eval_dem_stencil(elev[r0 + c], elev[r1 + c - 1], elev[r1 + c], elev[r1 + c + 1], elev[r2 + c], inv_dx, inv_dx, alg);
                            if val.is_nan() {
                                row_ga[px * 2] = 0;
                                row_ga[px * 2 + 1] = 0;
                            } else {
                                row_ga[px * 2] = rescale_u8(val, inv_range, min_val);
                                row_ga[px * 2 + 1] = 255;
                            }
                        }
                        filter_sub_ga_row(&row_ga, &mut sc.img_buf[py * ga_stride..(py + 1) * ga_stride]);
                    }
                    return encode_filtered_png(N as u32, N as u32, 4, &sc.img_buf[..ga_total]);
                }
                let (nodata_plte, merged) = build_plte_nodata(lut);
                for py in 0..N {
                    let r0 = py * pad;
                    let r1 = (py + 1) * pad;
                    let r2 = (py + 2) * pad;
                    let row_idx = &mut sc.row_buf[..N];
                    for px in 0..N {
                        let c = px + 1;
                        let val = eval_dem_stencil(elev[r0 + c], elev[r1 + c - 1], elev[r1 + c], elev[r1 + c + 1], elev[r2 + c], inv_dx, inv_dx, alg);
                        if val.is_nan() {
                            has_nodata = true;
                            row_idx[px] = 0;
                        } else if is_opaque_dem {
                            row_idx[px] = rescale_u8(val, inv_range, min_val);
                        } else {
                            row_idx[px] = nodata_index(rescale_u8(val, inv_range, min_val), merged);
                        }
                    }
                    filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                }
                if !has_nodata && is_opaque_dem {
                    if is_gray {
                        encode_filtered_png(N as u32, N as u32, 0, &sc.img_buf[..total])
                    } else {
                        let plte = build_plte_opaque(lut);
                        encode_indexed_png(N as u32, N as u32, plte, None, &sc.img_buf[..total])
                    }
                } else if is_opaque_dem && has_nodata {
                    // Rare case: `clean == true` float32 DEM contained `±Inf`, producing `NaN`
                    // gradients (`inf - inf`). Re-encode with nodata-safe indices so valid index 0
                    // pixels are never made transparent.
                    if is_gray {
                        let ga_stride = 1 + N * 2;
                        let ga_total = N * ga_stride;
                        if sc.img_buf.len() < ga_total {
                            sc.img_buf.resize(ga_total, 0);
                        }
                        let mut row_ga = vec![0u8; N * 2];
                        for py in 0..N {
                            let r0 = py * pad;
                            let r1 = (py + 1) * pad;
                            let r2 = (py + 2) * pad;
                            for px in 0..N {
                                let c = px + 1;
                                let val = eval_dem_stencil(elev[r0 + c], elev[r1 + c - 1], elev[r1 + c], elev[r1 + c + 1], elev[r2 + c], inv_dx, inv_dx, alg);
                                if val.is_nan() {
                                    row_ga[px * 2] = 0;
                                    row_ga[px * 2 + 1] = 0;
                                } else {
                                    row_ga[px * 2] = rescale_u8(val, inv_range, min_val);
                                    row_ga[px * 2 + 1] = 255;
                                }
                            }
                            filter_sub_ga_row(&row_ga, &mut sc.img_buf[py * ga_stride..(py + 1) * ga_stride]);
                        }
                        encode_filtered_png(N as u32, N as u32, 4, &sc.img_buf[..ga_total])
                    } else {
                        for py in 0..N {
                            let r0 = py * pad;
                            let r1 = (py + 1) * pad;
                            let r2 = (py + 2) * pad;
                            let row_idx = &mut sc.row_buf[..N];
                            for px in 0..N {
                                let c = px + 1;
                                let val = eval_dem_stencil(elev[r0 + c], elev[r1 + c - 1], elev[r1 + c], elev[r1 + c + 1], elev[r2 + c], inv_dx, inv_dx, alg);
                                row_idx[px] = if val.is_nan() {
                                    0
                                } else {
                                    nodata_index(rescale_u8(val, inv_range, min_val), merged)
                                };
                            }
                            filter_sub_u8_row(row_idx, &mut sc.img_buf[py * stride..(py + 1) * stride]);
                        }
                        encode_indexed_png(N as u32, N as u32, nodata_plte, Some(&[0u8][..]), &sc.img_buf[..total])
                    }
                } else {
                    let trns = has_nodata.then_some(&[0u8][..]);
                    encode_indexed_png(N as u32, N as u32, nodata_plte, trns, &sc.img_buf[..total])
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
                        let val = eval_dem_stencil(elev[r0 + c], elev[r1 + c - 1], elev[r1 + c], elev[r1 + c + 1], elev[r2 + c], inv_dx, inv_dx, alg);
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
