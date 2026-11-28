//! Widened convolution resampling (GDAL-compatible downsampling kernels).

use std::cell::RefCell;

use cogzilla_proj::CoordPlan;

use super::{ResampleMode, coord_plan_scale, for_each_pixel_in_row};
use crate::source::BandSource;

// GDAL resamples with bilinear, cubic, cubicspline and lanczos as convolution kernels. When the
// output is coarser than the source (both in the warper, `GWKResample`, and in RasterIO's
// `GDALResampleChunk_Convolution`), it stretches the kernel by the shrink factor: with `s` source
// pixels per output pixel, the support grows from `r` to `r * s` source pixels and each tap at
// distance `d` from the sample point weighs `K(d / s)`, normalised by the sum of the weights. The
// fixed 2x2 / 4x4 / 6x6 samplers above only see the pixels next to the sample point, which aliases
// when zoomed out. Upsampling (`s <= 1`) keeps the fixed kernels.

/// Most taps per axis the widened sampler evaluates. Beyond this (only reachable when a COG
/// without overviews is shrunk more than ~20x) the kernel stops widening.
const WIDE_MAX_TAPS: usize = 128;

#[inline(always)]
fn conv_kernel_radius(mode: ResampleMode) -> f32 {
    match mode {
        ResampleMode::Bilinear => 1.0,
        ResampleMode::Cubic | ResampleMode::CubicSpline => 2.0,
        ResampleMode::Lanczos => 3.0,
        _ => 0.0,
    }
}

/// GDAL's kernel scale for one axis, in source pixels per output pixel (`>= 1`). Like the GDAL
/// warper, a factor within 0.05 of a whole number snaps to it, so a near-1:1 request keeps the
/// fixed kernel (and its fast paths).
#[inline]
pub fn gdal_kernel_scale(s: f32) -> f32 {
    if !s.is_finite() || s <= 1.0 {
        return 1.0;
    }
    let r = s.round();
    if (s - r).abs() < 0.05 { r } else { s }
}

/// Kernel scale actually used by the widened sampler (GDAL's scale, capped at `WIDE_MAX_TAPS`).
#[inline(always)]
fn widened_scale(mode: ResampleMode, s: f32) -> f32 {
    let r = conv_kernel_radius(mode).max(1.0);
    gdal_kernel_scale(s).min((WIDE_MAX_TAPS as f32 - 2.0) / (2.0 * r))
}

/// True when `mode` is one of GDAL's convolution kernels and the source-pixel footprint
/// `(scale_x, scale_y)` shrinks the image on either axis, so the kernel must widen.
#[inline]
pub fn kernel_widens(mode: ResampleMode, scale_x: f32, scale_y: f32) -> bool {
    matches!(
        mode,
        ResampleMode::Bilinear | ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Lanczos
    ) && (gdal_kernel_scale(scale_x) > 1.0 || gdal_kernel_scale(scale_y) > 1.0)
}

/// Support radius, in source pixels, of `mode` at the source-pixel footprint `scale` (for fetch
/// margins). `0` for modes that are not convolution kernels.
#[inline]
pub fn widened_kernel_radius(mode: ResampleMode, scale: f32) -> f32 {
    conv_kernel_radius(mode) * widened_scale(mode, scale)
}

/// GDAL's kernel functions: `GWKBilinear` (tent), `GWKCubic` (Keys, `a = -0.5`), `GWKBSpline`
/// (cubic B-spline) and `GWKLanczosSinc` (radius 3).
#[inline(always)]
fn conv_kernel_weight(mode: ResampleMode, x: f32) -> f32 {
    let ax = x.abs();
    match mode {
        ResampleMode::Bilinear => (1.0 - ax).max(0.0),
        ResampleMode::Cubic => {
            if ax < 1.0 {
                (1.5 * ax - 2.5) * ax * ax + 1.0
            } else if ax < 2.0 {
                ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0
            } else {
                0.0
            }
        }
        ResampleMode::CubicSpline => {
            if ax < 1.0 {
                (0.5 * ax - 1.0) * ax * ax + (2.0 / 3.0)
            } else if ax < 2.0 {
                let t = 2.0 - ax;
                t * t * t * (1.0 / 6.0)
            } else {
                0.0
            }
        }
        ResampleMode::Lanczos => {
            if ax < 1e-6 {
                1.0
            } else if ax >= 3.0 {
                0.0
            } else {
                // sinc(x) * sinc(x / 3) with one sine: sin(3t) = 3 sin(t) - 4 sin(t)^3, t = pi x / 3.
                let t = std::f32::consts::FRAC_PI_3 * ax;
                let s = t.sin();
                (3.0 - 4.0 * s * s) * s * s / (3.0 * t * t)
            }
        }
        _ => 0.0,
    }
}

/// Taps of the widened kernel along one axis for sample position `c` (source pixel-centre
/// coordinates) at kernel scale `s`: every source pixel `j` with `|j - c| < radius * s`, weighed
/// `K((j - c) / s)`. Writes the weights into `w` and returns `(first_pixel, tap_count)`.
#[inline(always)]
fn widened_axis_taps(mode: ResampleMode, c: f32, s: f32, w: &mut [f32]) -> (i32, usize) {
    let r = conv_kernel_radius(mode) * s;
    let inv = 1.0 / s;
    let start = (c - r).floor() as i32 + 1;
    let stop = (c + r).ceil() as i32 - 1;
    let n = ((stop - start + 1).max(0) as usize).min(w.len());
    let mut d = (start as f32 - c) * inv;
    for wk in w.iter_mut().take(n) {
        *wk = conv_kernel_weight(mode, d);
        d += inv;
    }
    (start, n)
}

/// `sum_j wy[j] * sum_i wx[i] * sl[base + j * stride + i]` over `K` columns (`wx` zero-padded).
#[inline(always)]
fn widened_accumulate_fixed<const K: usize>(sl: &[f32], base: usize, stride: usize, wx: &[f32], wy: &[f32]) -> f32 {
    let mut wk = [0.0f32; K];
    wk[..wx.len()].copy_from_slice(wx);
    let mut col = [0.0f32; K];
    for (j, &wyj) in wy.iter().enumerate() {
        let off = base + j * stride;
        let row: &[f32; K] = sl[off..off + K].try_into().unwrap();
        for k in 0..K {
            col[k] += row[k] * wyj;
        }
    }
    let mut acc = 0.0f32;
    for k in 0..K {
        acc += col[k] * wk[k];
    }
    acc
}

/// [`widened_accumulate_fixed`] skipping `NaN` (nodata) values: `(sum, valid_weight)`.
#[inline(always)]
fn widened_accumulate_fixed_nan<const K: usize>(
    sl: &[f32],
    base: usize,
    stride: usize,
    wx: &[f32],
    wy: &[f32],
) -> (f32, f32) {
    let mut col = [0.0f32; K];
    let mut colw = [0.0f32; K];
    for (j, &wyj) in wy.iter().enumerate() {
        let off = base + j * stride;
        let row: &[f32; K] = sl[off..off + K].try_into().unwrap();
        for k in 0..K {
            let ok = !row[k].is_nan();
            col[k] += if ok { row[k] * wyj } else { 0.0 };
            colw[k] += if ok { wyj } else { 0.0 };
        }
    }
    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    for k in 0..K {
        sum += col[k] * wx[k];
        wsum += colw[k] * wx[k];
    }
    (sum, wsum)
}

/// Sum of `w[i] * value` over the valid taps of the `wx.len() x wy.len()` footprint at `(x0, y0)`,
/// with the valid and total weights: `(sum, valid_weight, total_weight)`.
#[inline(always)]
fn widened_accumulate(src: &BandSource<'_>, x0: i32, wx: &[f32], y0: i32, wy: &[f32]) -> (f32, f32, f32) {
    let nx = wx.len();
    let ny = wy.len();
    let wxs: f32 = wx.iter().sum();
    let wys: f32 = wy.iter().sum();
    let wtot = wxs * wys;
    // Clean f32 block, kernel up to 4 or 8 columns wide: pad the columns with zero weights to a
    // fixed width so each source row is one vector multiply-add.
    if src.no_nodata && nx <= 8 {
        let k = if nx <= 4 { 4 } else { 8 };
        if let Some((band, base, stride)) = src.rect_in_block(x0, y0, k as u32, ny as u32)
            && let Some(sl) = band.data.as_f32_slice()
        {
            let acc = if k == 4 {
                widened_accumulate_fixed::<4>(sl, base, stride, wx, wy)
            } else {
                widened_accumulate_fixed::<8>(sl, base, stride, wx, wy)
            };
            return (acc, wtot, wtot);
        }
    }
    let block = src.rect_in_block(x0, y0, nx as u32, ny as u32);
    if let Some((band, base, stride)) = block
        && !src.no_nodata
        && src.nodata_override.is_none()
        && src.scale_offset.is_none()
    {
        // One block that may hold nodata (`NaN` once read as f32): skip those taps.
        let mut sum = 0.0f32;
        let mut wsum = 0.0f32;
        for (j, &wyj) in wy.iter().enumerate() {
            let off = base + j * stride;
            let mut rs = 0.0f32;
            let mut rw = 0.0f32;
            for (i, &wxi) in wx.iter().enumerate() {
                let s = unsafe { band.data.get_f32_unchecked(off + i) };
                if !s.is_nan() {
                    rs += s * wxi;
                    rw += wxi;
                }
            }
            sum += rs * wyj;
            wsum += rw * wyj;
        }
        return (sum, wsum, wtot);
    }
    if let Some((band, base, stride)) = block
        && src.no_nodata
    {
        let mut acc = 0.0f32;
        if let Some(sl) = band.data.as_f32_slice() {
            for (j, &wyj) in wy.iter().enumerate() {
                let row = &sl[base + j * stride..base + j * stride + nx];
                let mut r = 0.0f32;
                for (a, b) in row.iter().zip(wx) {
                    r += a * b;
                }
                acc += r * wyj;
            }
        } else {
            for (j, &wyj) in wy.iter().enumerate() {
                let off = base + j * stride;
                let mut r = 0.0f32;
                for (i, &wxi) in wx.iter().enumerate() {
                    r += unsafe { band.data.get_f32_unchecked(off + i) } * wxi;
                }
                acc += r * wyj;
            }
        }
        return (acc, wtot, wtot);
    }
    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    for (j, &wyj) in wy.iter().enumerate() {
        let ry = y0 + j as i32;
        let mut rs = 0.0f32;
        let mut rw = 0.0f32;
        for (i, &wxi) in wx.iter().enumerate() {
            let s = src.sample_checked(x0 + i as i32, ry);
            if !s.is_nan() {
                rs += s * wxi;
                rw += wxi;
            }
        }
        sum += rs * wyj;
        wsum += rw * wyj;
    }
    (sum, wsum, wtot)
}

/// Normalise a widened-kernel accumulation. Kernels with negative lobes (cubic, lanczos) fall back
/// to the widened tent when less than half the kernel weight lands on valid pixels (raster edges,
/// nodata), mirroring the fixed samplers' bilinear fallback; GDAL renormalises over the valid taps.
#[inline(always)]
fn widened_finish(mode: ResampleMode, sum: f32, wsum: f32, wtot: f32) -> Option<f32> {
    let positive = matches!(mode, ResampleMode::Bilinear | ResampleMode::CubicSpline);
    if positive {
        if wsum > 1e-6 * wtot.max(1e-6) { Some(sum / wsum) } else { Some(f32::NAN) }
    } else if wsum >= 0.5 * wtot && wsum > 1e-6 {
        Some(sum / wsum)
    } else {
        None
    }
}

/// GDAL-style widened convolution sample at `(u, v)` for `Bilinear` / `Cubic` / `CubicSpline` /
/// `Lanczos` with source-pixel footprint `(scale_x, scale_y)` (see the section comment). An axis
/// whose scale is `<= 1` keeps the fixed kernel on that axis.
pub fn sample_kernel_widened(
    src: &BandSource<'_>,
    u: f32,
    v: f32,
    scale_x: f32,
    scale_y: f32,
    mode: ResampleMode,
) -> f32 {
    if !src.contains_sample(u, v) {
        return f32::NAN;
    }
    let sx = widened_scale(mode, scale_x);
    let sy = widened_scale(mode, scale_y);
    // Size the tap buffers to the kernel so the common small kernels don't clear 1 KiB per sample.
    let need = (2.0 * conv_kernel_radius(mode) * sx.max(sy)).ceil() as usize + 1;
    let val = if need <= 8 {
        sample_kernel_widened_taps::<8>(src, u, v, sx, sy, mode)
    } else if need <= 24 {
        sample_kernel_widened_taps::<24>(src, u, v, sx, sy, mode)
    } else {
        sample_kernel_widened_taps::<WIDE_MAX_TAPS>(src, u, v, sx, sy, mode)
    };
    match val {
        Some(val) => val,
        None => sample_kernel_widened(src, u, v, scale_x, scale_y, ResampleMode::Bilinear),
    }
}

#[inline(always)]
fn sample_kernel_widened_taps<const M: usize>(
    src: &BandSource<'_>,
    u: f32,
    v: f32,
    sx: f32,
    sy: f32,
    mode: ResampleMode,
) -> Option<f32> {
    let mut wx = [0.0f32; M];
    let mut wy = [0.0f32; M];
    let (x0, nx) = widened_axis_taps(mode, u, sx, &mut wx);
    let (y0, ny) = widened_axis_taps(mode, v, sy, &mut wy);
    let (sum, wsum, wtot) = widened_accumulate(src, x0, &wx[..nx], y0, &wy[..ny]);
    widened_finish(mode, sum, wsum, wtot)
}

/// Most taps per axis [`WidenedSampler`] caches; wider kernels go straight to
/// [`sample_kernel_widened`].
const WIDE_CACHE_TAPS: usize = 16;
/// Entries in [`WidenedSampler`]'s direct-mapped column-weight cache.
const WIDE_CACHE_COLS: usize = 4096;

/// Cached taps of one axis position: key (`f32` bits of the coordinate), first pixel, tap count
/// and the weights normalised to sum to 1, zero-padded to `WIDE_CACHE_TAPS`.
#[derive(Clone, Copy)]
struct CachedTaps {
    key: u32,
    first: i32,
    count: u32,
    w: [f32; WIDE_CACHE_TAPS],
}

impl CachedTaps {
    const EMPTY: Self = Self {
        // A NaN key never matches a finite coordinate.
        key: 0x7fc0_0000,
        first: 0,
        count: 0,
        w: [0.0; WIDE_CACHE_TAPS],
    };

    #[inline(always)]
    fn fill(&mut self, mode: ResampleMode, c: f32, s: f32) {
        let (first, n) = widened_axis_taps(mode, c, s, &mut self.w);
        let sum: f32 = self.w[..n].iter().sum();
        let inv = if sum.abs() > 1e-12 { 1.0 / sum } else { 0.0 };
        for w in &mut self.w[..n] {
            *w *= inv;
        }
        self.w[n..].fill(0.0);
        self.key = c.to_bits();
        self.first = first;
        self.count = n as u32;
    }
}

/// [`sample_kernel_widened`] for one request (fixed mode and footprint) sampled pixel by pixel from
/// coordinate maps. It remembers the row weights of the last `v` and the column weights of recent
/// `u` values, so maps whose rows share `v` and whose columns repeat `u` (every same-CRS bbox and
/// preview) compute each kernel weight once instead of once per pixel, and it reads clean blocks
/// with fixed-width (4, 8 or 16 column) multiply-adds. Not `Sync`: one per thread.
pub struct WidenedSampler {
    mode: ResampleMode,
    scale_x: f32,
    scale_y: f32,
    sx: f32,
    sy: f32,
    /// Padded column width (4, 8 or 16), or 0 when the kernel is too wide to cache.
    width: usize,
    row: std::cell::UnsafeCell<CachedTaps>,
    cols: std::cell::UnsafeCell<Vec<CachedTaps>>,
    grid: Option<std::cell::UnsafeCell<GridPlanes>>,
}

/// Most output pixels for which [`WidenedSampler::with_grid`] keeps whole planes (16 MiB each).
const WIDE_GRID_MAX_PIXELS: usize = 4 << 20;
/// Most source bands whose planes one [`WidenedSampler`] keeps.
const WIDE_GRID_MAX_PLANES: usize = 16;

/// Identity of a [`BandSource`]'s data: its first decoded block, the block grid and how values
/// are read (nodata override, scale/offset). The blocks outlive the sampler's render.
type PlaneKey = (usize, usize, u32, u32, u64, u64);

/// Separable source grid of a [`WidenedSampler`] with the planes computed on it so far.
struct GridPlanes {
    us: Vec<f32>,
    vs: Vec<f32>,
    planes: Vec<GridPlane>,
}

struct GridPlane {
    key: PlaneKey,
    data: std::rc::Rc<[f32]>,
}

#[inline(always)]
fn plane_key(src: &BandSource<'_>) -> Option<PlaneKey> {
    let first: &cogzilla_codec::DecodedBand = match src.blocks.first() {
        Some(Some(b)) => b,
        _ => src.blocks.iter().flatten().next()?,
    };
    let nodata = src.nodata_override.map_or(u64::MAX, |nd| nd.to_bits() as u64);
    let so = src
        .scale_offset
        .map_or(u64::MAX, |(sc, off)| ((sc.to_bits() as u64) << 32) | off.to_bits() as u64);
    Some((
        first as *const cogzilla_codec::DecodedBand as usize,
        src.blocks.len(),
        src.tx0,
        src.ty0,
        nodata,
        so,
    ))
}

/// The column coordinates of row 0 and the row coordinates of column 0 when the `w x h`
/// row-major maps are separable (every row of `u_map` equal, `v_map` constant along each row)
/// and finite; `None` otherwise.
pub fn separable_axes(u_map: &[f32], v_map: &[f32], w: usize, h: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    if w == 0 || h == 0 || u_map.len() < w * h || v_map.len() < w * h {
        return None;
    }
    let us = &u_map[..w];
    if !us.iter().all(|x| x.is_finite()) {
        return None;
    }
    let mut vs = Vec::with_capacity(h);
    for y in 0..h {
        let row_u = &u_map[y * w..(y + 1) * w];
        let row_v = &v_map[y * w..(y + 1) * w];
        let v0 = row_v[0];
        if !v0.is_finite()
            || row_u.iter().zip(us).any(|(a, b)| a.to_bits() != b.to_bits())
            || row_v.iter().any(|a| a.to_bits() != v0.to_bits())
        {
            return None;
        }
        vs.push(v0);
    }
    Some((us.to_vec(), vs))
}

impl WidenedSampler {
    /// A sampler that also knows the maps are the separable source grid `us` x `vs` (see
    /// [`separable_axes`]): [`Self::grid_plane`] then gives a band's whole plane, computed once
    /// with [`widened_plane_separable`]. `None` when the grid exceeds `WIDE_GRID_MAX_PIXELS`.
    pub fn with_grid(mode: ResampleMode, scale_x: f32, scale_y: f32, us: Vec<f32>, vs: Vec<f32>) -> Option<Self> {
        if us.is_empty() || vs.is_empty() || us.len().saturating_mul(vs.len()) > WIDE_GRID_MAX_PIXELS {
            return None;
        }
        let mut s = Self::new(mode, scale_x, scale_y);
        s.grid = Some(std::cell::UnsafeCell::new(GridPlanes { us, vs, planes: Vec::new() }));
        Some(s)
    }

    /// A sampler without the column-tap cache, for maps whose columns do not repeat `u`
    /// (reprojections), where every lookup would miss.
    pub fn new_uncached(mode: ResampleMode, scale_x: f32, scale_y: f32) -> Self {
        let mut s = Self::new(mode, scale_x, scale_y);
        s.cols = std::cell::UnsafeCell::new(Vec::new());
        s
    }

    /// `src`'s widened plane on the grid (row-major, `us.len()` wide): at each output pixel the
    /// value [`Self::sample`] gives at that pixel's grid coordinates. Computed on first use per
    /// band; `None` without a grid or when the band cannot be keyed or cached.
    pub fn grid_plane(&self, src: &BandSource<'_>) -> Option<std::rc::Rc<[f32]>> {
        // SAFETY: as in `sample`, the sampler is `!Sync` and never re-entered.
        let g = unsafe { &mut *self.grid.as_ref()?.get() };
        let key = plane_key(src)?;
        if let Some(p) = g.planes.iter().find(|p| p.key == key) {
            return Some(p.data.clone());
        }
        if g.planes.len() >= WIDE_GRID_MAX_PLANES {
            return None;
        }
        let data: std::rc::Rc<[f32]> =
            widened_plane_separable(&g.us, &g.vs, src, self.mode, self.scale_x, self.scale_y).into();
        g.planes.push(GridPlane { key, data: data.clone() });
        Some(data)
    }

    pub fn new(mode: ResampleMode, scale_x: f32, scale_y: f32) -> Self {
        let sx = widened_scale(mode, scale_x);
        let sy = widened_scale(mode, scale_y);
        let need = (2.0 * conv_kernel_radius(mode) * sx.max(sy)).ceil() as usize + 1;
        let width = match need {
            0..=4 => 4,
            5..=8 => 8,
            9..=WIDE_CACHE_TAPS => WIDE_CACHE_TAPS,
            _ => 0,
        };
        let slots = if width > 0 { WIDE_CACHE_COLS } else { 0 };
        Self {
            mode,
            scale_x,
            scale_y,
            sx,
            sy,
            width,
            row: std::cell::UnsafeCell::new(CachedTaps::EMPTY),
            cols: std::cell::UnsafeCell::new(vec![CachedTaps::EMPTY; slots]),
            grid: None,
        }
    }

    /// Same value as `sample_kernel_widened(src, u, v, scale_x, scale_y, mode)`.
    #[inline]
    pub fn sample(&self, src: &BandSource<'_>, u: f32, v: f32) -> f32 {
        if self.width == 0 {
            return sample_kernel_widened(src, u, v, self.scale_x, self.scale_y, self.mode);
        }
        if !src.contains_sample(u, v) {
            return f32::NAN;
        }
        // SAFETY: `WidenedSampler` is `!Sync` (it holds `UnsafeCell`s) and `sample` never re-enters
        // itself, so these are the only live references to the caches.
        let row = unsafe { &mut *self.row.get() };
        let cols = unsafe { &mut *self.cols.get() };
        if row.key != v.to_bits() {
            row.fill(self.mode, v, self.sy);
        }
        let mut fresh = CachedTaps::EMPTY;
        let col: &CachedTaps = if cols.is_empty() {
            fresh.fill(self.mode, u, self.sx);
            &fresh
        } else {
            let ub = u.to_bits();
            let col = &mut cols[((ub ^ (ub >> 13)) as usize) & (WIDE_CACHE_COLS - 1)];
            if col.key != ub {
                col.fill(self.mode, u, self.sx);
            }
            col
        };
        let (x0, y0, ny) = (col.first, row.first, row.count as usize);
        let wy = &row.w[..ny];
        let block = if src.nodata_override.is_none() && src.scale_offset.is_none() {
            src.rect_in_block(x0, y0, self.width as u32, ny as u32)
                .and_then(|(band, base, stride)| Some((band.data.as_f32_slice()?, base, stride)))
        } else {
            None
        };
        let (sum, wsum, wtot) = match block {
            // Weights are normalised and zero-padded: on clean data the sum is the value.
            Some((sl, base, stride)) if src.no_nodata => {
                return match self.width {
                    4 => widened_accumulate_fixed::<4>(sl, base, stride, &col.w[..4], wy),
                    8 => widened_accumulate_fixed::<8>(sl, base, stride, &col.w[..8], wy),
                    _ => widened_accumulate_fixed::<WIDE_CACHE_TAPS>(sl, base, stride, &col.w, wy),
                };
            }
            Some((sl, base, stride)) => {
                let (sum, wsum) = match self.width {
                    4 => widened_accumulate_fixed_nan::<4>(sl, base, stride, &col.w[..4], wy),
                    8 => widened_accumulate_fixed_nan::<8>(sl, base, stride, &col.w[..8], wy),
                    _ => widened_accumulate_fixed_nan::<WIDE_CACHE_TAPS>(sl, base, stride, &col.w, wy),
                };
                (sum, wsum, 1.0)
            }
            None => widened_accumulate(src, x0, &col.w[..col.count as usize], y0, wy),
        };
        match widened_finish(self.mode, sum, wsum, wtot) {
            Some(val) => val,
            None => sample_kernel_widened(src, u, v, self.scale_x, self.scale_y, ResampleMode::Bilinear),
        }
    }
}

/// `f64` twin of [`sample_kernel_widened`] for wide dtypes (`Uint32`, `Int32`, `Float64`, ...).
pub fn sample_f64_kernel_widened(
    src: &BandSource<'_>,
    u: f64,
    v: f64,
    scale_x: f32,
    scale_y: f32,
    mode: ResampleMode,
) -> f64 {
    if !src.contains_sample_f64(u, v) {
        return f64::NAN;
    }
    let axis = |c: f64, s: f32| -> (i32, Vec<f64>) {
        let s = widened_scale(mode, s) as f64;
        let r = conv_kernel_radius(mode) as f64 * s;
        let start = (c - r).floor() as i32 + 1;
        let stop = (c + r).ceil() as i32 - 1;
        let w = (start..=stop)
            .map(|j| conv_kernel_weight(mode, ((j as f64 - c) / s) as f32) as f64)
            .collect();
        (start, w)
    };
    let (x0, wx) = axis(u, scale_x);
    let (y0, wy) = axis(v, scale_y);
    let wtot = wx.iter().sum::<f64>() * wy.iter().sum::<f64>();
    let mut sum = 0.0f64;
    let mut wsum = 0.0f64;
    for (j, &wyj) in wy.iter().enumerate() {
        for (i, &wxi) in wx.iter().enumerate() {
            let s = src.sample_f64_checked(x0 + i as i32, y0 + j as i32);
            if !s.is_nan() {
                sum += s * wxi * wyj;
                wsum += wxi * wyj;
            }
        }
    }
    match widened_finish(mode, sum as f32, wsum as f32, wtot as f32) {
        Some(val) if val.is_nan() => f64::NAN,
        Some(_) => sum / wsum,
        None => sample_f64_kernel_widened(src, u, v, scale_x, scale_y, ResampleMode::Bilinear),
    }
}

/// Scratch for [`widened_plane_separable`]: source row segment, ring of horizontal-pass rows
/// (sums and, on data with nodata, valid weights), the source row held by each ring slot, and a
/// row of vertical-pass valid weights.
struct WideScratch {
    seg: Vec<f32>,
    hs: Vec<f32>,
    hw: Vec<f32>,
    tags: Vec<i32>,
    wrow: Vec<f32>,
}

thread_local! {
    static WIDE_SCRATCH: RefCell<WideScratch> = const {
        RefCell::new(WideScratch { seg: Vec::new(), hs: Vec::new(), hw: Vec::new(), tags: Vec::new(), wrow: Vec::new() })
    };
}

/// Precomputed widened-kernel taps for every output column (or row) of a separable plan.
pub(super) struct AxisTapTable {
    start: Vec<i32>,
    count: Vec<u32>,
    stride: usize,
    /// `len * stride` weights, normalised to sum to 1 per entry.
    weights: Vec<f32>,
}

impl AxisTapTable {
    pub(super) fn build(mode: ResampleMode, coords: &[f32], scale: f32, lim: i32) -> Self {
        let s = widened_scale(mode, scale);
        let bound = ((2.0 * conv_kernel_radius(mode) * s).ceil() as usize + 1).min(WIDE_MAX_TAPS);
        let mut start = Vec::with_capacity(coords.len());
        let mut count = Vec::with_capacity(coords.len());
        let mut packed = vec![0.0f32; coords.len() * bound];
        let mut tmp = [0.0f32; WIDE_MAX_TAPS];
        let mut most = 0usize;
        for (k, &c) in coords.iter().enumerate() {
            let (st, n) = if BandSource::axis_contains_sample(c, lim) {
                widened_axis_taps(mode, c, s, &mut tmp[..bound])
            } else {
                (0, 0)
            };
            let sum: f32 = tmp[..n].iter().sum();
            let inv = if sum.abs() > 1e-12 { 1.0 / sum } else { 0.0 };
            for i in 0..n {
                packed[k * bound + i] = tmp[i] * inv;
            }
            start.push(st);
            count.push(n as u32);
            most = most.max(n);
        }
        // The widest entry rounded up to a multiple of 4, so short kernels run as fixed-width,
        // zero-padded dots.
        let stride = (most.max(1).div_ceil(4) * 4).min(bound.div_ceil(4) * 4);
        let mut weights = vec![0.0f32; coords.len() * stride];
        for (k, &n) in count.iter().enumerate() {
            let n = n as usize;
            weights[k * stride..k * stride + n].copy_from_slice(&packed[k * bound..k * bound + n]);
        }
        Self { start, count, stride, weights }
    }

    /// All `stride` weights of entry `k`, zero past its taps.
    #[inline(always)]
    fn taps_padded(&self, k: usize) -> &[f32] {
        &self.weights[k * self.stride..(k + 1) * self.stride]
    }

    #[inline(always)]
    fn taps(&self, k: usize) -> (i32, &[f32]) {
        let n = self.count[k] as usize;
        (self.start[k], &self.weights[k * self.stride..k * self.stride + n])
    }

    /// Inclusive range of source pixels touched by any entry, or `None` when no entry has taps.
    fn span(&self) -> Option<(i32, i32)> {
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        for (k, &st) in self.start.iter().enumerate() {
            let n = self.count[k] as i32;
            if n > 0 {
                lo = lo.min(st);
                hi = hi.max(st + n - 1);
            }
        }
        (lo <= hi).then_some((lo, hi))
    }
}

/// Render the `N x N` widened-kernel plane of `src` for a plan, as row-major `f32` (`NaN` = no
/// data): [`widened_plane_separable`] for separable plans, [`sample_kernel_widened`] per pixel
/// otherwise.
pub fn widened_plane<const N: usize>(
    plan: &CoordPlan<N>,
    src: &BandSource<'_>,
    mode: ResampleMode,
    scale_x: f32,
    scale_y: f32,
) -> Vec<f32> {
    if let CoordPlan::Separable1D { u, v, .. } = plan {
        return widened_plane_separable(&u[..], &v[..], src, mode, scale_x, scale_y);
    }
    let mut out = vec![f32::NAN; N * N];
    let sampler = WidenedSampler::new_uncached(mode, scale_x, scale_y);
    for py in 0..N {
        for_each_pixel_in_row(plan, py, |px, uu, vv| {
            out[py * N + px] = sampler.sample(src, uu, vv);
        });
    }
    out
}

/// Clean horizontal pass with `K`-wide zero-padded taps: `hs_row[px] = sum_i w_i * seg[off + i]`.
#[inline(always)]
fn h_pass_fixed<const K: usize>(seg: &[f32], gc0: i32, cols: &AxisTapTable, col_ok: &[bool], hs_row: &mut [f32]) {
    for (px, out) in hs_row.iter_mut().enumerate() {
        if col_ok[px] {
            let off = (cols.start[px] - gc0) as usize;
            let a: &[f32; K] = seg[off..off + K].try_into().unwrap();
            let b: &[f32; K] = cols.taps_padded(px).try_into().unwrap();
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += a[k] * b[k];
            }
            *out = acc;
        }
    }
}

/// [`h_pass_fixed`] skipping `NaN` (nodata) values, with the valid weight in `hw_row`.
#[inline(always)]
fn h_pass_fixed_nan<const K: usize>(
    seg: &[f32],
    gc0: i32,
    cols: &AxisTapTable,
    col_ok: &[bool],
    hs_row: &mut [f32],
    hw_row: &mut [f32],
) {
    for (px, (out, wout)) in hs_row.iter_mut().zip(hw_row.iter_mut()).enumerate() {
        if col_ok[px] {
            let off = (cols.start[px] - gc0) as usize;
            let a: &[f32; K] = seg[off..off + K].try_into().unwrap();
            let b: &[f32; K] = cols.taps_padded(px).try_into().unwrap();
            let (mut acc, mut wacc) = (0.0f32, 0.0f32);
            for k in 0..K {
                let ok = !a[k].is_nan();
                acc += if ok { a[k] * b[k] } else { 0.0 };
                wacc += if ok { b[k] } else { 0.0 };
            }
            *out = acc;
            *wout = wacc;
        }
    }
}

/// Widened-kernel plane of `src` on the separable grid `us` (column coordinates) x `vs` (row
/// coordinates), row-major `vs.len() x us.len()` (`NaN` = no data); every value equals
/// [`sample_kernel_widened`] at `(us[px], vs[py])`. Runs as two passes over precomputed
/// per-column / per-row taps: each source row is filtered horizontally once (into a small ring,
/// as consecutive output rows share source rows), then each output row is a vertical blend of
/// those. On clean data the passes skip validity bookkeeping and the few columns / rows whose
/// footprint leaves the raster are sampled per pixel; with nodata (or `NaN`) the passes carry the
/// valid weight and renormalise like the per-pixel sampler.
pub fn widened_plane_separable(
    us: &[f32],
    vs: &[f32],
    src: &BandSource<'_>,
    mode: ResampleMode,
    scale_x: f32,
    scale_y: f32,
) -> Vec<f32> {
    let cols = AxisTapTable::build(mode, us, scale_x, src.ifd_w);
    let rows = AxisTapTable::build(mode, vs, scale_y, src.ifd_h);
    widened_plane_separable_with_tables(us, vs, src, mode, scale_x, scale_y, &cols, &rows)
}

pub(super) fn widened_plane_separable_with_tables(
    us: &[f32],
    vs: &[f32],
    src: &BandSource<'_>,
    mode: ResampleMode,
    scale_x: f32,
    scale_y: f32,
    cols: &AxisTapTable,
    rows: &AxisTapTable,
) -> Vec<f32> {
    let (w, h) = (us.len(), vs.len());
    let mut out = vec![f32::NAN; w * h];
    let (Some((c_lo, c_hi)), Some(_)) = (cols.span(), rows.span()) else {
        return out;
    };
    let clean = src.no_nodata;
    // Columns / rows the passes handle: sample point in range and, on clean data, the whole
    // footprint inside the raster (taps outside it are skipped like nodata on the other path).
    let col_ok: Vec<bool> = (0..w)
        .map(|k| {
            let (st, t) = cols.taps(k);
            BandSource::axis_contains_sample(us[k], src.ifd_w)
                && (!clean || (st >= 0 && st + t.len() as i32 <= src.ifd_w))
        })
        .collect();
    let row_ok: Vec<bool> = (0..h)
        .map(|k| {
            let (st, t) = rows.taps(k);
            BandSource::axis_contains_sample(vs[k], src.ifd_h)
                && (!clean || (st >= 0 && st + t.len() as i32 <= src.ifd_h))
        })
        .collect();
    let (gc0, gc1) = if clean { (c_lo.max(0), c_hi.min(src.ifd_w - 1)) } else { (c_lo, c_hi) };
    let positive = matches!(mode, ResampleMode::Bilinear | ResampleMode::CubicSpline);
    let ring = rows.stride + 1;
    let mut fallback: Vec<usize> = Vec::new();
    WIDE_SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let sc = &mut *guard;
        let seg_w = if gc0 <= gc1 { (gc1 - gc0 + 1) as usize } else { 0 };
        // Zero tail so fixed-width dots may read past the last column (with zero weights).
        let pad = cols.stride;
        if sc.seg.len() < seg_w + pad {
            sc.seg.resize(seg_w + pad, 0.0);
        }
        sc.seg[seg_w..seg_w + pad].fill(0.0);
        for buf in [&mut sc.hs, &mut sc.hw] {
            if buf.len() < ring * w {
                buf.resize(ring * w, 0.0);
            }
        }
        if sc.wrow.len() < w {
            sc.wrow.resize(w, 0.0);
        }
        sc.tags.clear();
        sc.tags.resize(ring, i32::MIN);
        let WideScratch { seg, hs, hw, tags, wrow } = sc;
        let seg = &mut seg[..seg_w + pad];
        let wrow = &mut wrow[..w];
        for py in 0..h {
            let row_out = &mut out[py * w..(py + 1) * w];
            if row_ok[py] && seg_w > 0 {
                let (st, wy) = rows.taps(py);
                // Horizontal pass for the source rows this output row needs and the ring lacks.
                for j in 0..wy.len() {
                    let r = st + j as i32;
                    let slot = r.rem_euclid(ring as i32) as usize;
                    if tags[slot] == r {
                        continue;
                    }
                    tags[slot] = r;
                    src.read_row_segment(r, gc0, &mut seg[..seg_w]);
                    let hs_row = &mut hs[slot * w..(slot + 1) * w];
                    if clean {
                        match cols.stride {
                            4 => h_pass_fixed::<4>(seg, gc0, cols, &col_ok, hs_row),
                            8 => h_pass_fixed::<8>(seg, gc0, cols, &col_ok, hs_row),
                            _ => {
                                for px in 0..w {
                                    if col_ok[px] {
                                        let (cst, wx) = cols.taps(px);
                                        let off = (cst - gc0) as usize;
                                        let mut acc = 0.0f32;
                                        for (a, b) in seg[off..off + wx.len()].iter().zip(wx) {
                                            acc += a * b;
                                        }
                                        hs_row[px] = acc;
                                    }
                                }
                            }
                        }
                    } else {
                        let hw_row = &mut hw[slot * w..(slot + 1) * w];
                        match cols.stride {
                            4 => h_pass_fixed_nan::<4>(seg, gc0, cols, &col_ok, hs_row, hw_row),
                            8 => h_pass_fixed_nan::<8>(seg, gc0, cols, &col_ok, hs_row, hw_row),
                            _ => {
                                for px in 0..w {
                                    if col_ok[px] {
                                        let (cst, wx) = cols.taps(px);
                                        let off = (cst - gc0) as usize;
                                        let (mut acc, mut wacc) = (0.0f32, 0.0f32);
                                        for (&a, &b) in seg[off..off + wx.len()].iter().zip(wx) {
                                            let ok = !a.is_nan();
                                            acc += if ok { a * b } else { 0.0 };
                                            wacc += if ok { b } else { 0.0 };
                                        }
                                        hs_row[px] = acc;
                                        hw_row[px] = wacc;
                                    }
                                }
                            }
                        }
                    }
                }
                // Vertical pass.
                row_out.fill(0.0);
                if !clean {
                    wrow.fill(0.0);
                }
                for (j, &wj) in wy.iter().enumerate() {
                    let slot = (st + j as i32).rem_euclid(ring as i32) as usize;
                    for (o, &a) in row_out.iter_mut().zip(&hs[slot * w..(slot + 1) * w]) {
                        *o += a * wj;
                    }
                    if !clean {
                        for (o, &a) in wrow.iter_mut().zip(&hw[slot * w..(slot + 1) * w]) {
                            *o += a * wj;
                        }
                    }
                }
            }
            for px in 0..w {
                if !row_ok[py] || !col_ok[px] || seg_w == 0 {
                    row_out[px] = sample_kernel_widened(src, us[px], vs[py], scale_x, scale_y, mode);
                } else if clean {
                    if row_out[px].is_nan() {
                        row_out[px] = sample_kernel_widened(src, us[px], vs[py], scale_x, scale_y, mode);
                    }
                } else {
                    // Normalised weights: the valid weight is the share of the kernel on valid pixels.
                    let ws = wrow[px];
                    row_out[px] = if positive {
                        if ws > 1e-6 { row_out[px] / ws } else { f32::NAN }
                    } else if ws >= 0.5 {
                        row_out[px] / ws
                    } else {
                        fallback.push(py * w + px);
                        f32::NAN
                    };
                }
            }
        }
    });
    // Cubic / lanczos pixels with under half the kernel on valid data take the widened tent, as in
    // `sample_kernel_widened`: one tent pass over just the rows that need it (nodata-heavy images
    // have many such pixels, and sampling each one alone dominated the render).
    if !fallback.is_empty() {
        let mut sub_of_row = vec![u32::MAX; h];
        let mut sub_vs = Vec::new();
        for &i in &fallback {
            let py = i / w;
            if sub_of_row[py] == u32::MAX {
                sub_of_row[py] = sub_vs.len() as u32;
                sub_vs.push(vs[py]);
            }
        }
        let tent = widened_plane_separable(us, &sub_vs, src, ResampleMode::Bilinear, scale_x, scale_y);
        for &i in &fallback {
            out[i] = tent[sub_of_row[i / w] as usize * w + i % w];
        }
    }
    out
}

/// `(scale_x, scale_y)` of `plan` when `mode` is a convolution kernel that widens on it.
#[inline]
pub fn plan_widened_scale<const N: usize>(plan: &CoordPlan<N>, mode: ResampleMode) -> Option<(f32, f32)> {
    if !matches!(
        mode,
        ResampleMode::Bilinear | ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Lanczos
    ) {
        return None;
    }
    let (sx, sy) = coord_plan_scale(plan);
    kernel_widens(mode, sx, sy).then_some((sx, sy))
}

#[cfg(test)]
mod widened_kernel_tests {
    use super::super::{sample_bilinear, sample_cubic, sample_general_scaled};
    use super::*;
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};

    fn f32_band(w: u32, h: u32, f: impl Fn(u32, u32) -> f32) -> DecodedBand {
        let mut slab = AlignedF32Slab::new_uninit((w * h) as usize);
        let mut no_nodata = true;
        for (i, d) in slab.as_mut_slice().iter_mut().enumerate() {
            *d = f(i as u32 % w, i as u32 / w);
            no_nodata &= !d.is_nan();
        }
        DecodedBand {
            width: w,
            height: h,
            data: BandData::F32(slab),
            stats: BlockStats { all_nodata: false, no_nodata, min: 0.0, max: 1.0 },
        }
    }

    fn pattern(x: u32, y: u32) -> f32 {
        ((x * 7919 + y * 104_729) % 1000) as f32
    }

    /// Straight `f64` GDAL convolution: every pixel `j` with `|j - c| < r * s`, weight `K((j - c) / s)`.
    fn reference(src: &BandSource<'_>, u: f32, v: f32, s: f32, mode: ResampleMode) -> f64 {
        let r = conv_kernel_radius(mode) as f64 * s as f64;
        let (mut sum, mut wsum) = (0.0f64, 0.0f64);
        for y in 0..src.ifd_h {
            let dy = y as f64 - v as f64;
            if dy.abs() >= r {
                continue;
            }
            for x in 0..src.ifd_w {
                let dx = x as f64 - u as f64;
                if dx.abs() >= r {
                    continue;
                }
                let w = conv_kernel_weight(mode, (dx / s as f64) as f32) as f64
                    * conv_kernel_weight(mode, (dy / s as f64) as f32) as f64;
                sum += w * src.sample_checked(x, y) as f64;
                wsum += w;
            }
        }
        sum / wsum
    }

    #[test]
    fn kernel_scale_snaps_like_the_gdal_warper() {
        assert_eq!(gdal_kernel_scale(0.5), 1.0);
        assert_eq!(gdal_kernel_scale(1.0), 1.0);
        assert_eq!(gdal_kernel_scale(1.04), 1.0);
        assert_eq!(gdal_kernel_scale(1.97), 2.0);
        assert!((gdal_kernel_scale(1.56) - 1.56).abs() < 1e-6);
        assert!(!kernel_widens(ResampleMode::Bilinear, 1.04, 0.5));
        assert!(kernel_widens(ResampleMode::Bilinear, 1.0, 1.5));
        assert!(kernel_widens(ResampleMode::Lanczos, 2.0, 2.0));
        assert!(!kernel_widens(ResampleMode::Nearest, 4.0, 4.0));
        assert!(!kernel_widens(ResampleMode::Average, 4.0, 4.0));
        assert!(!kernel_widens(ResampleMode::Gauss, 4.0, 4.0));
        assert_eq!(widened_kernel_radius(ResampleMode::Cubic, 1.5), 3.0);
        assert_eq!(widened_kernel_radius(ResampleMode::Nearest, 3.0), 0.0);
    }

    #[test]
    fn halving_bilinear_is_the_gdal_1_3_3_1_filter() {
        let b = f32_band(16, 16, pattern);
        let blks = [Some(&b)];
        let src = BandSource::new(16, 16, 16, 16, 0, 0, 1, 1, &blks);
        // Output pixel centred between source pixels 5 and 6 (and rows 7 and 8) at 2x shrink.
        let got = sample_kernel_widened(&src, 5.5, 7.5, 2.0, 2.0, ResampleMode::Bilinear);
        let w = [0.125f64, 0.375, 0.375, 0.125];
        let mut want = 0.0f64;
        for (j, wy) in w.iter().enumerate() {
            for (i, wx) in w.iter().enumerate() {
                want += wx * wy * pattern(4 + i as u32, 6 + j as u32) as f64;
            }
        }
        assert!((got as f64 - want).abs() < 1e-3, "got {got}, want {want}");
    }

    #[test]
    fn widened_sampler_matches_a_direct_gdal_convolution() {
        let b = f32_band(64, 64, pattern);
        let blks = [Some(&b)];
        let src = BandSource::new(64, 64, 64, 64, 0, 0, 1, 1, &blks);
        for mode in [ResampleMode::Bilinear, ResampleMode::Cubic, ResampleMode::CubicSpline, ResampleMode::Lanczos] {
            for &s in &[1.3f32, 1.56, 2.0, 3.7] {
                let sampler = WidenedSampler::new(mode, s, s);
                for &(u, v) in &[(20.0f32, 30.0f32), (31.25, 17.8), (40.6, 40.1)] {
                    let want = reference(&src, u, v, gdal_kernel_scale(s), mode);
                    let got = sample_kernel_widened(&src, u, v, s, s, mode) as f64;
                    let cached = sampler.sample(&src, u, v) as f64;
                    assert!((got - want).abs() < 2e-2, "{mode:?} s={s} ({u},{v}): {got} vs {want}");
                    assert!((cached - want).abs() < 2e-2, "{mode:?} s={s} ({u},{v}) cached: {cached} vs {want}");
                }
            }
        }
    }

    #[test]
    fn widened_kernel_skips_nodata_and_out_of_raster_taps() {
        // Left half NaN: a sample next to the boundary averages only the valid pixels.
        let b = f32_band(16, 16, |x, _| if x < 8 { f32::NAN } else { 10.0 });
        let blks = [Some(&b)];
        let src = BandSource::new(16, 16, 16, 16, 0, 0, 1, 1, &blks);
        let v = sample_kernel_widened(&src, 7.5, 8.0, 2.0, 2.0, ResampleMode::Bilinear);
        assert!((v - 10.0).abs() < 1e-5, "{v}");
        let sampler = WidenedSampler::new(ResampleMode::Lanczos, 2.0, 2.0);
        let l = sampler.sample(&src, 7.5, 8.0);
        assert!((l - 10.0).abs() < 1e-4, "{l}");
        // Fully inside the NaN half: no data.
        assert!(sample_kernel_widened(&src, 2.0, 8.0, 2.0, 2.0, ResampleMode::Bilinear).is_nan());
        // Raster corner: renormalised over the in-raster taps.
        let c = f32_band(16, 16, |_, _| 3.0);
        let cb = [Some(&c)];
        let csrc = BandSource::new(16, 16, 16, 16, 0, 0, 1, 1, &cb);
        let e = sample_kernel_widened(&csrc, 0.0, 0.0, 3.0, 3.0, ResampleMode::Cubic);
        assert!((e - 3.0).abs() < 1e-5, "{e}");
        // Sample point outside the raster stays no data.
        assert!(sample_kernel_widened(&csrc, -0.7, 4.0, 3.0, 3.0, ResampleMode::Cubic).is_nan());
    }

    #[test]
    fn separable_widened_plane_matches_per_pixel_sampling() {
        // 2x2 grid of 32x32 blocks, one with a nodata patch, so the plane crosses blocks and meets
        // nodata (wide enough that cubic / lanczos pixels on its rim fall back to the tent).
        let blocks: Vec<DecodedBand> = (0..4u32)
            .map(|k| {
                f32_band(32, 32, move |x, y| {
                    if k == 3 && x < 12 && y < 12 {
                        f32::NAN
                    } else {
                        pattern(x + 32 * (k % 2), y + 32 * (k / 2))
                    }
                })
            })
            .collect();
        let refs: Vec<Option<&DecodedBand>> = blocks.iter().map(Some).collect();
        let src = BandSource::new(64, 64, 32, 32, 0, 0, 2, 2, &refs);
        const N: usize = 32;
        let s = 1.9f32;
        let mut u = Box::new([0.0f32; N]);
        let mut v = Box::new([0.0f32; N]);
        for i in 0..N {
            u[i] = (i as f32 + 0.5) * s - 0.5 - 1.0;
            v[i] = (i as f32 + 0.5) * s - 0.5 + 0.3;
        }
        let plan = CoordPlan::Separable1D {
            u: u.clone(),
            v: v.clone(),
            src_min_col: -1,
            src_max_col: 62,
            src_min_row: 0,
            src_max_row: 61,
            all_in_bounds: false,
        };
        let (sx, sy) = coord_plan_scale(&plan);
        for mode in [ResampleMode::Bilinear, ResampleMode::Cubic, ResampleMode::CubicSpline, ResampleMode::Lanczos] {
            assert!(plan_widened_scale(&plan, mode).is_some());
            let plane = widened_plane(&plan, &src, mode, sx, sy);
            for py in 0..N {
                for px in 0..N {
                    let want = sample_kernel_widened(&src, u[px], v[py], sx, sy, mode);
                    let got = plane[py * N + px];
                    assert!(
                        (want.is_nan() && got.is_nan()) || (got - want).abs() < 1e-2,
                        "{mode:?} ({px},{py}): plane {got} vs pixel {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn grid_sampler_and_uncached_sampler_match_per_pixel_sampling() {
        // Clean data across a 2x2 block grid, non-square output grid that runs past the raster.
        let blocks: Vec<DecodedBand> = (0..4u32)
            .map(|k| f32_band(32, 32, move |x, y| pattern(x + 32 * (k % 2), y + 32 * (k / 2))))
            .collect();
        let refs: Vec<Option<&DecodedBand>> = blocks.iter().map(Some).collect();
        let src = BandSource::new(64, 64, 32, 32, 0, 0, 2, 2, &refs);
        let (w, h) = (40usize, 23usize);
        let us: Vec<f32> = (0..w).map(|i| (i as f32 + 0.5) * 1.62 - 1.5).collect();
        let vs: Vec<f32> = (0..h).map(|i| (i as f32 + 0.5) * 2.7 + 0.2).collect();
        let mut u_map = Vec::new();
        let mut v_map = Vec::new();
        for &v in &vs {
            for &u in &us {
                u_map.push(u);
                v_map.push(v);
            }
        }
        let (gu, gv) = separable_axes(&u_map, &v_map, w, h).expect("separable");
        assert_eq!((gu.len(), gv.len()), (w, h));
        let mut skewed = v_map.clone();
        skewed[w + 3] += 0.25;
        assert!(separable_axes(&u_map, &skewed, w, h).is_none());
        for mode in [ResampleMode::Bilinear, ResampleMode::Cubic, ResampleMode::CubicSpline, ResampleMode::Lanczos] {
            let grid = WidenedSampler::with_grid(mode, 1.62, 2.7, gu.clone(), gv.clone()).unwrap();
            let uncached = WidenedSampler::new_uncached(mode, 1.62, 2.7);
            for i in 0..w * h {
                let want = sample_kernel_widened(&src, u_map[i], v_map[i], 1.62, 2.7, mode);
                let g = grid.grid_plane(&src).unwrap()[i];
                let c = uncached.sample(&src, u_map[i], v_map[i]);
                for got in [g, c] {
                    assert!(
                        (want.is_nan() && got.is_nan()) || (got - want).abs() < 1e-2,
                        "{mode:?} pixel {i}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn upsampling_keeps_the_fixed_kernels() {
        let b = f32_band(16, 16, pattern);
        let blks = [Some(&b)];
        let src = BandSource::new(16, 16, 16, 16, 0, 0, 1, 1, &blks);
        for &(u, v) in &[(5.3f32, 6.7f32), (8.0, 8.5)] {
            assert_eq!(
                sample_general_scaled(&src, u, v, 0.5, 0.5, ResampleMode::Bilinear),
                sample_bilinear::<false>(&src, u, v)
            );
            assert_eq!(sample_general_scaled(&src, u, v, 1.02, 0.7, ResampleMode::Cubic), sample_cubic(&src, u, v));
        }
    }

    #[test]
    fn widened_plane_separable_ignores_far_out_of_range_coords_with_nodata() {
        // Out-of-range coordinates (e.g. low-zoom tile over a small COG) must not inflate the
        // separable tap span or overflow i32 when no_nodata is false.
        let b = f32_band(16, 16, |x, _| if x == 0 { f32::NAN } else { 7.0 });
        let blks = [Some(&b)];
        let mut src = BandSource::new(16, 16, 16, 16, 0, 0, 1, 1, &blks);
        src.no_nodata = false;
        let us = [-1e9f32, -500_000.0, 4.5, 1e9];
        let vs = [-1e9f32, 4.5, 1e9];
        for mode in [ResampleMode::Bilinear, ResampleMode::Cubic, ResampleMode::CubicSpline, ResampleMode::Lanczos] {
            let plane = widened_plane_separable(&us, &vs, &src, mode, 2.0, 2.0);
            assert_eq!(plane.len(), us.len() * vs.len());
            for (py, &v) in vs.iter().enumerate() {
                for (px, &u) in us.iter().enumerate() {
                    let got = plane[py * us.len() + px];
                    let want = sample_kernel_widened(&src, u, v, 2.0, 2.0, mode);
                    if want.is_nan() {
                        assert!(got.is_nan(), "{mode:?} ({px},{py}): expected NaN, got {got}");
                    } else {
                        assert!((got - want).abs() < 1e-4, "{mode:?} ({px},{py}): {got} vs {want}");
                    }
                }
            }
        }
    }
}

