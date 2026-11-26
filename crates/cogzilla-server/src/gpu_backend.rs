use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use cogzilla_gpu::{
    CoalescedJob, CoalescedOutput, DispatchPolicy, DispatchRequest, GpuBandStats, GpuCoalescer,
    GpuColorFormula, GpuColormapMode, GpuColormapRenderer, GpuContext, GpuDemAlgorithm,
    GpuExprRenderer, GpuInstr, GpuJpegEncoder, GpuMosaicCompositor, GpuMultiBandCache, GpuOp,
    GpuPixelOutput, GpuPixelSelection, GpuResampler, GpuSlab, GpuSlabCache, GpuStatsReducer,
    GpuTerrainRenderer, GpuTileBatcher, GpuWeightedBandStats, GpuWeightedStatsReducer,
    MultiBandSlabKey, OwnedCoordPlan, PolicyOverride, SlabKey, TileCoordSlice, opcodes,
};
use cogzilla_kernel::expr_vm::Op;
use cogzilla_kernel::png_fast::{encode_indexed_png, filter_sub_u8_row};
use cogzilla_kernel::{
    BandSource, ColorFormula, ColormapName, CustomColormap, DemAlgorithm, ExprProgram,
    OutputFormat, ResampleMode, RgbStyle, SingleBandStyle, SpectralExpr, TerrainEncoding,
    encode_rgba_tile,
};
use cogzilla_proj::CoordPlan;

/// Coordinate mapping passed from `titiler::render_bbox_impl` or `TileEngine` into `GpuBackend`.
#[derive(Clone, Copy)]
pub enum BboxCoordSource<'a> {
    /// Separable 1D axes (used by `/cog/preview` native previews & WebMercator/EPSG:4326 tiles).
    Separable { u: &'a [f32], v: &'a [f32] },
    /// 17×17 coarse bilinear warp grid (used by projected UTM/Albers/LCC/Polar tiles & bboxes).
    Grid2D {
        dst_w: usize,
        dst_h: usize,
        grid_u: &'a [f32; 289],
        grid_v: &'a [f32; 289],
    },
    /// Per-pixel 2D coordinate maps (used by extreme-curvature CRSs & GCP rasters).
    Exact2D {
        dst_w: usize,
        dst_h: usize,
        u_map: &'a [f32],
        v_map: &'a [f32],
    },
    /// Separable 1D axes + 1-bit-per-pixel polygon mask (64× smaller than `Exact2D`!).
    SeparableMasked {
        u: &'a [f32],
        v: &'a [f32],
        mask_bits: &'a [u32],
    },
    /// 17×17 coarse warp grid + 1-bit-per-pixel polygon mask (64× smaller than `Exact2D`!).
    Grid2DMasked {
        dst_w: usize,
        dst_h: usize,
        grid_u: &'a [f32; 289],
        grid_v: &'a [f32; 289],
        mask_bits: &'a [u32],
    },
}

impl BboxCoordSource<'_> {
    fn dims_and_upload_bytes(&self) -> Option<(usize, usize, usize)> {
        match *self {
            Self::Separable { u, v } => Some((u.len(), v.len(), 0)),
            Self::Grid2D { dst_w, dst_h, .. } => Some((dst_w, dst_h, 0)),
            Self::Exact2D {
                dst_w,
                dst_h,
                u_map,
                v_map,
            } => {
                if u_map.len() < dst_w * dst_h || v_map.len() < dst_w * dst_h {
                    return None;
                }
                Some((dst_w, dst_h, dst_w * dst_h * 8))
            }
            Self::SeparableMasked { u, v, mask_bits } => {
                let dst_w = u.len();
                let dst_h = v.len();
                let need = (dst_w * dst_h).div_ceil(32);
                if mask_bits.len() < need {
                    return None;
                }
                Some((dst_w, dst_h, need * 4))
            }
            Self::Grid2DMasked {
                dst_w,
                dst_h,
                mask_bits,
                ..
            } => {
                let need = (dst_w * dst_h).div_ceil(32);
                if mask_bits.len() < need {
                    return None;
                }
                Some((dst_w, dst_h, need * 4))
            }
        }
    }
}

/// Optional GPU acceleration backend owned by `TileEngine` when built with `--features gpu`.
pub struct GpuBackend {
    pub ctx: GpuContext,
    pub policy: DispatchPolicy,
    pub slab_cache: GpuSlabCache,
    pub multi_slab_cache: GpuMultiBandCache,
    pub coalescer: GpuCoalescer,
    resampler: GpuResampler,
    #[allow(dead_code)]
    colormap: Mutex<GpuColormapRenderer>,
    #[allow(dead_code)]
    terrain: Mutex<GpuTerrainRenderer>,
    #[allow(dead_code)]
    expr: Mutex<GpuExprRenderer>,
    mosaic: Mutex<GpuMosaicCompositor>,
    batcher: Mutex<GpuTileBatcher>,
    jpeg: Mutex<GpuJpegEncoder>,
    stats: Mutex<GpuStatsReducer>,
    weighted_stats: Mutex<GpuWeightedStatsReducer>,
    in_flight: AtomicUsize,
}

struct InFlightGuard<'a>(&'a AtomicUsize);
impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

fn map_resample(mode: ResampleMode) -> Option<GpuColormapMode> {
    match mode {
        ResampleMode::Nearest => Some(GpuColormapMode::Nearest),
        ResampleMode::Bilinear => Some(GpuColormapMode::Bilinear),
        ResampleMode::Cubic => Some(GpuColormapMode::Cubic),
        ResampleMode::Lanczos => Some(GpuColormapMode::Lanczos3),
        _ => None,
    }
}

fn build_plte_nodata(rgb: &[[u8; 3]; 256]) -> [u8; 768] {
    let mut plte = [0u8; 768];
    for i in 1..256 {
        let src_idx = ((i - 1) * 255 + 127) / 254;
        plte[i * 3] = rgb[src_idx][0];
        plte[i * 3 + 1] = rgb[src_idx][1];
        plte[i * 3 + 2] = rgb[src_idx][2];
    }
    plte
}

fn encode_indexed_u8_png(idx_bytes: &[u8], width: u32, height: u32, rgb: &[[u8; 3]; 256]) -> Bytes {
    let w = width as usize;
    let h = height as usize;
    let mut filtered = vec![0u8; h * (1 + w)];
    for py in 0..h {
        let src_row = &idx_bytes[py * w..(py + 1) * w];
        let dst_row = &mut filtered[py * (1 + w)..(py + 1) * (1 + w)];
        filter_sub_u8_row(src_row, dst_row);
    }
    let plte = build_plte_nodata(rgb);
    let trns = [0u8];
    encode_indexed_png(width, height, &plte, Some(&trns), &filtered)
}

/// Compute the stitched slab geometry `(base_col, base_row, width, height)` for a `BandSource`.
pub fn stitch_band_source_geom(src: &BandSource<'_>) -> Option<(u32, u32, u32, u32)> {
    let base_col = src.tx0.checked_mul(src.tile_w)?;
    let base_row = src.ty0.checked_mul(src.tile_h)?;
    let max_w = (src.ifd_w.max(0) as u32).saturating_sub(base_col);
    let max_h = (src.ifd_h.max(0) as u32).saturating_sub(base_row);
    let w = src.grid_w.saturating_mul(src.tile_w).min(max_w);
    let h = src.grid_h.saturating_mul(src.tile_h).min(max_h);
    if w == 0 || h == 0 || (w as u64) * (h as u64) > 64 * 1024 * 1024 {
        return None;
    }
    Some((base_col, base_row, w, h))
}

/// Stitch any single- or multi-block `BandSource` (`F32`, `U16`, `U8`, `RgbU8Channel`,
/// sparse blocks, nodata override, scale/offset) directly into a caller-provided `out` slice
/// (e.g. a `mapped_at_creation` GPU buffer) using block-wise hoisted SIMD-friendly loops.
pub fn stitch_band_source_into(src: &BandSource<'_>, w: u32, h: u32, out: &mut [f32]) {
    use cogzilla_codec::BandData;

    let w_us = w as usize;
    let h_us = h as usize;
    let tw = src.tile_w as usize;
    let th = src.tile_h as usize;
    let nd_ovr = src.nodata_override;
    let sc_off = src.scale_offset;

    // Only pre-fill NaN if any block is missing/all_nodata or shorter than the tile grid.
    let all_full = (src.grid_w as usize * tw == w_us)
        && (src.grid_h as usize * th == h_us)
        && src.blocks.iter().all(|b| {
            b.is_some_and(|blk| {
                !blk.stats.all_nodata && blk.width as usize == tw && blk.height as usize == th
            })
        });
    if !all_full {
        out[..w_us * h_us].fill(f32::NAN);
    }

    for gy in 0..src.grid_h as usize {
        let dst_y0 = gy * th;
        if dst_y0 >= h_us {
            break;
        }
        let max_rows = (h_us - dst_y0).min(th);
        for gx in 0..src.grid_w as usize {
            let dst_x0 = gx * tw;
            if dst_x0 >= w_us {
                break;
            }
            let max_cols = (w_us - dst_x0).min(tw);
            let Some(blk) = src.blocks[gy * (src.grid_w as usize) + gx] else {
                continue;
            };
            if blk.stats.all_nodata {
                continue;
            }
            let copy_h = max_rows.min(blk.height as usize);
            let copy_w = max_cols.min(blk.width as usize);
            if copy_h == 0 || copy_w == 0 {
                continue;
            }

            match &blk.data {
                BandData::F32(slab) => {
                    let sdata = slab.as_slice();
                    if nd_ovr.is_none() && sc_off.is_none() {
                        for ly in 0..copy_h {
                            let s0 = ly * tw;
                            let d0 = (dst_y0 + ly) * w_us + dst_x0;
                            out[d0..d0 + copy_w].copy_from_slice(&sdata[s0..s0 + copy_w]);
                        }
                    } else {
                        for ly in 0..copy_h {
                            let srow = &sdata[ly * tw..ly * tw + copy_w];
                            let d0 = (dst_y0 + ly) * w_us + dst_x0;
                            let drow = &mut out[d0..d0 + copy_w];
                            for (d, &v) in drow.iter_mut().zip(srow) {
                                let mut val = v;
                                if let Some(nd) = nd_ovr
                                    && val == nd
                                {
                                    val = f32::NAN;
                                }
                                if let Some((sc, off)) = sc_off
                                    && !val.is_nan()
                                {
                                    val = val * sc + off;
                                }
                                *d = val;
                            }
                        }
                    }
                }
                BandData::U16 { data, nodata } => {
                    let nd_u16 = *nodata;
                    if nd_u16.is_none() && nd_ovr.is_none() && sc_off.is_none() {
                        for ly in 0..copy_h {
                            let srow = &data[ly * tw..ly * tw + copy_w];
                            let d0 = (dst_y0 + ly) * w_us + dst_x0;
                            let drow = &mut out[d0..d0 + copy_w];
                            for (d, &s) in drow.iter_mut().zip(srow) {
                                *d = s as f32;
                            }
                        }
                    } else {
                        for ly in 0..copy_h {
                            let srow = &data[ly * tw..ly * tw + copy_w];
                            let d0 = (dst_y0 + ly) * w_us + dst_x0;
                            let drow = &mut out[d0..d0 + copy_w];
                            for (d, &s) in drow.iter_mut().zip(srow) {
                                let mut val = if nd_u16 == Some(s) {
                                    f32::NAN
                                } else {
                                    s as f32
                                };
                                if let Some(nd) = nd_ovr
                                    && val == nd
                                {
                                    val = f32::NAN;
                                }
                                if let Some((sc, off)) = sc_off
                                    && !val.is_nan()
                                {
                                    val = val * sc + off;
                                }
                                *d = val;
                            }
                        }
                    }
                }
                BandData::U8 { data, nodata } => {
                    let nd_u8 = *nodata;
                    if nd_u8.is_none() && nd_ovr.is_none() && sc_off.is_none() {
                        for ly in 0..copy_h {
                            let srow = &data[ly * tw..ly * tw + copy_w];
                            let d0 = (dst_y0 + ly) * w_us + dst_x0;
                            let drow = &mut out[d0..d0 + copy_w];
                            for (d, &s) in drow.iter_mut().zip(srow) {
                                *d = s as f32;
                            }
                        }
                    } else {
                        for ly in 0..copy_h {
                            let srow = &data[ly * tw..ly * tw + copy_w];
                            let d0 = (dst_y0 + ly) * w_us + dst_x0;
                            let drow = &mut out[d0..d0 + copy_w];
                            for (d, &s) in drow.iter_mut().zip(srow) {
                                let mut val = if nd_u8 == Some(s) {
                                    f32::NAN
                                } else {
                                    s as f32
                                };
                                if let Some(nd) = nd_ovr
                                    && val == nd
                                {
                                    val = f32::NAN;
                                }
                                if let Some((sc, off)) = sc_off
                                    && !val.is_nan()
                                {
                                    val = val * sc + off;
                                }
                                *d = val;
                            }
                        }
                    }
                }
                BandData::RgbU8Channel {
                    rgb,
                    channel,
                    nodata,
                } => {
                    let ch = *channel as usize;
                    let nd_u8 = *nodata;
                    if nd_u8.is_none() && nd_ovr.is_none() && sc_off.is_none() {
                        for ly in 0..copy_h {
                            let sbase = ly * tw * 3 + ch;
                            let d0 = (dst_y0 + ly) * w_us + dst_x0;
                            let drow = &mut out[d0..d0 + copy_w];
                            for (lx, d) in drow.iter_mut().enumerate() {
                                *d = rgb[sbase + lx * 3] as f32;
                            }
                        }
                    } else {
                        for ly in 0..copy_h {
                            let sbase = ly * tw * 3 + ch;
                            let d0 = (dst_y0 + ly) * w_us + dst_x0;
                            let drow = &mut out[d0..d0 + copy_w];
                            for (lx, d) in drow.iter_mut().enumerate() {
                                let s = rgb[sbase + lx * 3];
                                let mut val = if nd_u8 == Some(s) {
                                    f32::NAN
                                } else {
                                    s as f32
                                };
                                if let Some(nd) = nd_ovr
                                    && val == nd
                                {
                                    val = f32::NAN;
                                }
                                if let Some((sc, off)) = sc_off
                                    && !val.is_nan()
                                {
                                    val = val * sc + off;
                                }
                                *d = val;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Parallel block-row wrapper around `stitch_band_source_into`: when a single band spans
/// multiple block rows (`grid_h >= 2`) and `w * h >= 524_288`, stitches disjoint horizontal
/// block-row strips in parallel via `std::thread::scope` directly into `out`.
pub fn stitch_band_source_into_parallel(src: &BandSource<'_>, w: u32, h: u32, out: &mut [f32]) {
    let w_us = w as usize;
    let h_us = h as usize;
    let th = src.tile_h as usize;
    if src.grid_h >= 2 && w_us * h_us >= 524_288 && th > 0 {
        let strip_elems = th * w_us;
        std::thread::scope(|s| {
            for (gy, strip) in out[..w_us * h_us].chunks_mut(strip_elems).enumerate() {
                if gy >= src.grid_h as usize {
                    break;
                }
                s.spawn(move || {
                    stitch_band_row_strip(src, w_us, gy, strip);
                });
            }
        });
    } else {
        stitch_band_source_into(src, w, h, out);
    }
}

fn stitch_band_row_strip(src: &BandSource<'_>, w_us: usize, gy: usize, strip: &mut [f32]) {
    use cogzilla_codec::BandData;

    let tw = src.tile_w as usize;
    let th = src.tile_h as usize;
    let strip_rows = strip.len() / w_us;
    let max_rows = strip_rows.min(th);
    let nd_ovr = src.nodata_override;
    let sc_off = src.scale_offset;

    let row_blocks = &src.blocks[gy * (src.grid_w as usize)..(gy + 1) * (src.grid_w as usize)];
    let all_full = (src.grid_w as usize * tw == w_us)
        && (max_rows == th)
        && row_blocks.iter().all(|b| {
            b.is_some_and(|blk| {
                !blk.stats.all_nodata && blk.width as usize == tw && blk.height as usize == th
            })
        });
    if !all_full {
        strip.fill(f32::NAN);
    }

    for (gx, &blk_opt) in row_blocks.iter().enumerate() {
        let dst_x0 = gx * tw;
        if dst_x0 >= w_us {
            break;
        }
        let max_cols = (w_us - dst_x0).min(tw);
        let Some(blk) = blk_opt else {
            continue;
        };
        if blk.stats.all_nodata {
            continue;
        }
        let copy_h = max_rows.min(blk.height as usize);
        let copy_w = max_cols.min(blk.width as usize);
        if copy_h == 0 || copy_w == 0 {
            continue;
        }
        match &blk.data {
            BandData::F32(slab) => {
                let sdata = slab.as_slice();
                if nd_ovr.is_none() && sc_off.is_none() {
                    for ly in 0..copy_h {
                        let s0 = ly * tw;
                        let d0 = ly * w_us + dst_x0;
                        strip[d0..d0 + copy_w].copy_from_slice(&sdata[s0..s0 + copy_w]);
                    }
                } else {
                    for ly in 0..copy_h {
                        let srow = &sdata[ly * tw..ly * tw + copy_w];
                        let d0 = ly * w_us + dst_x0;
                        let drow = &mut strip[d0..d0 + copy_w];
                        for (d, &v) in drow.iter_mut().zip(srow) {
                            let mut val = v;
                            if let Some(nd) = nd_ovr
                                && val == nd
                            {
                                val = f32::NAN;
                            }
                            if let Some((sc, off)) = sc_off
                                && !val.is_nan()
                            {
                                val = val * sc + off;
                            }
                            *d = val;
                        }
                    }
                }
            }
            BandData::U16 { data, nodata } => {
                let nd_u16 = *nodata;
                if nd_u16.is_none() && nd_ovr.is_none() && sc_off.is_none() {
                    for ly in 0..copy_h {
                        let srow = &data[ly * tw..ly * tw + copy_w];
                        let d0 = ly * w_us + dst_x0;
                        let drow = &mut strip[d0..d0 + copy_w];
                        for (d, &s) in drow.iter_mut().zip(srow) {
                            *d = s as f32;
                        }
                    }
                } else {
                    for ly in 0..copy_h {
                        let srow = &data[ly * tw..ly * tw + copy_w];
                        let d0 = ly * w_us + dst_x0;
                        let drow = &mut strip[d0..d0 + copy_w];
                        for (d, &s) in drow.iter_mut().zip(srow) {
                            let mut val = if nd_u16 == Some(s) {
                                f32::NAN
                            } else {
                                s as f32
                            };
                            if let Some(nd) = nd_ovr
                                && val == nd
                            {
                                val = f32::NAN;
                            }
                            if let Some((sc, off)) = sc_off
                                && !val.is_nan()
                            {
                                val = val * sc + off;
                            }
                            *d = val;
                        }
                    }
                }
            }
            BandData::U8 { data, nodata } => {
                let nd_u8 = *nodata;
                if nd_u8.is_none() && nd_ovr.is_none() && sc_off.is_none() {
                    for ly in 0..copy_h {
                        let srow = &data[ly * tw..ly * tw + copy_w];
                        let d0 = ly * w_us + dst_x0;
                        let drow = &mut strip[d0..d0 + copy_w];
                        for (d, &s) in drow.iter_mut().zip(srow) {
                            *d = s as f32;
                        }
                    }
                } else {
                    for ly in 0..copy_h {
                        let srow = &data[ly * tw..ly * tw + copy_w];
                        let d0 = ly * w_us + dst_x0;
                        let drow = &mut strip[d0..d0 + copy_w];
                        for (d, &s) in drow.iter_mut().zip(srow) {
                            let mut val = if nd_u8 == Some(s) {
                                f32::NAN
                            } else {
                                s as f32
                            };
                            if let Some(nd) = nd_ovr
                                && val == nd
                            {
                                val = f32::NAN;
                            }
                            if let Some((sc, off)) = sc_off
                                && !val.is_nan()
                            {
                                val = val * sc + off;
                            }
                            *d = val;
                        }
                    }
                }
            }
            BandData::RgbU8Channel {
                rgb,
                channel,
                nodata,
            } => {
                let ch = *channel as usize;
                let nd_u8 = *nodata;
                if nd_u8.is_none() && nd_ovr.is_none() && sc_off.is_none() {
                    for ly in 0..copy_h {
                        let sbase = ly * tw * 3 + ch;
                        let d0 = ly * w_us + dst_x0;
                        let drow = &mut strip[d0..d0 + copy_w];
                        for (lx, d) in drow.iter_mut().enumerate() {
                            *d = rgb[sbase + lx * 3] as f32;
                        }
                    }
                } else {
                    for ly in 0..copy_h {
                        let sbase = ly * tw * 3 + ch;
                        let d0 = ly * w_us + dst_x0;
                        let drow = &mut strip[d0..d0 + copy_w];
                        for (lx, d) in drow.iter_mut().enumerate() {
                            let s = rgb[sbase + lx * 3];
                            let mut val = if nd_u8 == Some(s) {
                                f32::NAN
                            } else {
                                s as f32
                            };
                            if let Some(nd) = nd_ovr
                                && val == nd
                            {
                                val = f32::NAN;
                            }
                            if let Some((sc, off)) = sc_off
                                && !val.is_nan()
                            {
                                val = val * sc + off;
                            }
                            *d = val;
                        }
                    }
                }
            }
        }
    }
}

/// Stitch any single- or multi-block `BandSource` into a newly allocated contiguous `Vec<f32>`.
pub fn stitch_band_source(src: &BandSource<'_>) -> Option<(Vec<f32>, u32, u32, u32, u32)> {
    let (base_col, base_row, w, h) = stitch_band_source_geom(src)?;
    let mut out = vec![0.0f32; (w as usize) * (h as usize)];
    stitch_band_source_into(src, w, h, &mut out);
    Some((out, base_col, base_row, w, h))
}

fn to_owned_plan(coords: BboxCoordSource<'_>, bc: f32, br: f32) -> OwnedCoordPlan {
    match coords {
        BboxCoordSource::Separable { u, v } => OwnedCoordPlan::Separable1D {
            u: u.iter().map(|&x| x - bc).collect(),
            v: v.iter().map(|&y| y - br).collect(),
        },
        BboxCoordSource::Grid2D {
            dst_w,
            dst_h,
            grid_u,
            grid_v,
        } => {
            let mut gu = [0.0f32; 289];
            let mut gv = [0.0f32; 289];
            for i in 0..289 {
                gu[i] = grid_u[i] - bc;
                gv[i] = grid_v[i] - br;
            }
            OwnedCoordPlan::Grid2D {
                dst_w,
                dst_h,
                grid_u: gu,
                grid_v: gv,
            }
        }
        BboxCoordSource::Exact2D {
            dst_w,
            dst_h,
            u_map,
            v_map,
        } => {
            let n = dst_w * dst_h;
            let shift_slice = |src: &[f32], delta: f32| -> Vec<f32> {
                if delta == 0.0 {
                    return src.to_vec();
                }
                let mut out = vec![0.0f32; src.len()];
                if src.len() >= 256 * 256 {
                    let threads = std::thread::available_parallelism()
                        .map(|p| p.get().clamp(2, 8))
                        .unwrap_or(4);
                    let chunk = src.len().div_ceil(threads);
                    std::thread::scope(|s| {
                        for (dst_c, src_c) in out.chunks_mut(chunk).zip(src.chunks(chunk)) {
                            s.spawn(move || {
                                for (d, &v) in dst_c.iter_mut().zip(src_c.iter()) {
                                    *d = v - delta;
                                }
                            });
                        }
                    });
                } else {
                    for (d, &v) in out.iter_mut().zip(src.iter()) {
                        *d = v - delta;
                    }
                }
                out
            };
            let (u, v) = if n >= 256 * 256 {
                std::thread::scope(|s| {
                    let hu = s.spawn(|| shift_slice(&u_map[..n], bc));
                    let v = shift_slice(&v_map[..n], br);
                    (hu.join().unwrap(), v)
                })
            } else {
                (shift_slice(&u_map[..n], bc), shift_slice(&v_map[..n], br))
            };
            OwnedCoordPlan::Exact2D { dst_w, dst_h, u, v }
        }
        BboxCoordSource::SeparableMasked { u, v, mask_bits } => {
            let dst_w = u.len();
            let dst_h = v.len();
            let need = (dst_w * dst_h).div_ceil(32);
            let mut uv_packed = Vec::with_capacity(dst_w + dst_h);
            uv_packed.extend(u.iter().map(|&x| x - bc));
            uv_packed.extend(v.iter().map(|&y| y - br));
            OwnedCoordPlan::SeparableMasked {
                dst_w,
                dst_h,
                uv_packed,
                mask_bits: mask_bits[..need].to_vec(),
            }
        }
        BboxCoordSource::Grid2DMasked {
            dst_w,
            dst_h,
            grid_u,
            grid_v,
            mask_bits,
        } => {
            let need = (dst_w * dst_h).div_ceil(32);
            let mut uv_packed = Vec::with_capacity(578);
            uv_packed.extend(grid_u.iter().map(|&x| x - bc));
            uv_packed.extend(grid_v.iter().map(|&y| y - br));
            OwnedCoordPlan::Grid2DMasked {
                dst_w,
                dst_h,
                uv_packed,
                mask_bits: mask_bits[..need].to_vec(),
            }
        }
    }
}

fn source_slab_key(cog_id: u64, ifd_idx: usize, band0: u16, src: &BandSource<'_>) -> SlabKey {
    SlabKey {
        dataset_id: cog_id,
        ifd_idx: ifd_idx as u16,
        band: band0,
        tx: (src.tx0 << 8) | (src.grid_w & 0xff),
        ty: (src.ty0 << 8) | (src.grid_h & 0xff),
    }
}

impl GpuBackend {
    /// Initialize GPU context + calibrated `DispatchPolicy` if a hardware GPU is available and
    /// `COGZILLA_GPU_POLICY != never`.
    pub fn try_init(slab_cache_bytes: usize) -> Option<Arc<Self>> {
        if PolicyOverride::from_env() == PolicyOverride::Never {
            return None;
        }
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                tracing::info!("GPU backend disabled (no hardware adapter): {e}");
                return None;
            }
        };
        let policy = DispatchPolicy::calibrate(&ctx);
        tracing::info!(
            adapter = %ctx.info.name,
            backend = ?ctx.info.backend,
            index = ctx.info.index,
            unified_memory = ctx.info.unified_memory,
            floor_us = policy.profile.dispatch_floor_us,
            upload_gb_s = policy.profile.upload_gb_s,
            readback_gb_s = policy.profile.readback_gb_s,
            override_mode = ?policy.override_mode,
            "cogzilla GPU backend initialized"
        );
        let resampler = GpuResampler::new(&ctx);
        let coalesce_us = if ctx.info.unified_memory { 400 } else { 250 };
        let coalescer = GpuCoalescer::new(&ctx, 32, coalesce_us);
        let colormap = Mutex::new(GpuColormapRenderer::new(&ctx));
        let terrain = Mutex::new(GpuTerrainRenderer::new(&ctx));
        let expr = Mutex::new(GpuExprRenderer::new(&ctx));
        let mosaic = Mutex::new(GpuMosaicCompositor::new(&ctx));
        let batcher = Mutex::new(GpuTileBatcher::new(&ctx));
        let jpeg = Mutex::new(GpuJpegEncoder::new(&ctx));
        let stats = Mutex::new(GpuStatsReducer::new(&ctx));
        let weighted_stats = Mutex::new(GpuWeightedStatsReducer::new(&ctx));
        Some(Arc::new(Self {
            ctx,
            policy,
            slab_cache: GpuSlabCache::new(slab_cache_bytes),
            multi_slab_cache: GpuMultiBandCache::new(slab_cache_bytes),
            coalescer,
            resampler,
            colormap,
            terrain,
            expr,
            mosaic,
            batcher,
            jpeg,
            stats,
            weighted_stats,
            in_flight: AtomicUsize::new(0),
        }))
    }

    /// Resolve `(Arc<GpuSlab>, base_col, base_row)` for either a single-block flat `f32` source
    /// (zero-copy upload) or a stitched multi-block / `U8` / `U16` neighborhood.
    fn resolve_source_slab(
        &self,
        key: SlabKey,
        warm: Option<Arc<GpuSlab>>,
        src: &BandSource<'_>,
    ) -> Option<(Arc<GpuSlab>, u32, u32)> {
        if let Some((slice, base_col, base_row, stride)) = src.single_block_flat() {
            let slab = match warm {
                Some(s) => s,
                None => {
                    let height = (slice.len() / stride) as u32;
                    let width = stride as u32;
                    self.slab_cache
                        .get_or_upload(key, slice, width, height, stride as u32, &self.resampler)
                        .ok()?
                }
            };
            return Some((slab, base_col, base_row));
        }

        let base_col = src.tx0.checked_mul(src.tile_w)?;
        let base_row = src.ty0.checked_mul(src.tile_h)?;
        if let Some(s) = warm {
            return Some((s, base_col, base_row));
        }
        let (bc, br, w, h) = stitch_band_source_geom(src)?;
        let slab = Arc::new(
            GpuSlab::from_mapped_fill(&self.ctx.device, w, h, w, |dst| {
                stitch_band_source_into_parallel(src, w, h, dst);
                Ok(())
            })
            .ok()?,
        );
        self.slab_cache.insert(key, Arc::clone(&slab));
        Some((slab, bc, br))
    }

    fn estimate_upload_bytes(warm: &Option<Arc<GpuSlab>>, src: &BandSource<'_>) -> usize {
        if warm.is_some() {
            return 0;
        }
        if let Some((slice, _, _, _)) = src.single_block_flat() {
            return slice.len() * std::mem::size_of::<f32>();
        }
        let w = (src.grid_w * src.tile_w) as usize;
        let h = (src.grid_h * src.tile_h) as usize;
        w.saturating_mul(h).saturating_mul(std::mem::size_of::<f32>())
    }

    async fn submit_coalesced_async(&self, job: CoalescedJob) -> Option<Vec<u8>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.coalescer
            .submit_with_callback(job, move |res| {
                let _ = tx.send(res);
            })
            .ok()?;
        rx.await.ok()?.ok()
    }

    /// Attempt GPU-accelerated single-band rendering for arbitrary `(u_coords, v_coords)` crops
    /// (`/cog/preview`, `/cog/bbox`, `/cog/feature`, or multi-block tiles).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_crop_single_band(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        u_coords: &[f32],
        v_coords: &[f32],
        src: &BandSource<'_>,
        style: &SingleBandStyle,
    ) -> Option<Bytes> {
        self.try_render_bbox_single_band(
            cog_id,
            ifd_idx,
            band0,
            BboxCoordSource::Separable {
                u: u_coords,
                v: v_coords,
            },
            src,
            style,
            true,
        )
    }

    /// Resolve the effective 256-entry RGB LUT for any single-band, DEM, CustomExpr, or Spectral
    /// GPU render, supporting opaque rescaled `CustomColormap` and zero-cost `ColorFormula`
    /// folding (all `Gamma` / `Sigmoidal` / `Saturation` / `Brightness` / `Contrast` ops folded
    /// in ~1 µs on 256 LUT entries instead of millions of pixels).
    pub fn resolve_styled_rgb_lut(
        colormap: ColormapName,
        custom_lut: Option<&CustomColormap>,
        color_formula: Option<&ColorFormula>,
    ) -> Option<[[u8; 3]; 256]> {
        let mut rgb = match custom_lut {
            None => colormap.lut().rgb,
            Some(CustomColormap::Discrete {
                lut,
                extra,
                direct_lookup: false,
            }) if extra.is_empty() && lut.rgba_u32.iter().all(|&p| (p >> 24) == 255) => lut.rgb,
            Some(_) => return None,
        };
        if let Some(cf) = color_formula {
            for entry in &mut rgb {
                let (r, g, b) = cf.apply_rgb(entry[0], entry[1], entry[2]);
                *entry = [r, g, b];
            }
        }
        Some(rgb)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_bbox_single_band(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
        active: usize,
    ) -> Option<(CoalescedJob, GpuPixelOutput, usize, usize)> {
        let gpu_mode = map_resample(style.resample)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let out_kind = match style.format {
            OutputFormat::Png if return_mask => GpuPixelOutput::IndexedU8,
            OutputFormat::Png
            | OutputFormat::Webp { .. }
            | OutputFormat::Jpeg { .. }
            | OutputFormat::RawRgba => GpuPixelOutput::Rgba8,
            _ => return None,
        };

        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src) + coord_upload_bytes;
        let readback_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => dst_w * dst_h,
            GpuPixelOutput::Rgba8 => dst_w * dst_h * 4,
        };
        let decision = self.policy.decide(&DispatchRequest {
            op: gpu_mode.to_policy_op(),
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }

        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(rgb_lut);
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        Some((
            CoalescedJob::Colormap {
                slab,
                plan,
                mode: gpu_mode,
                rescale_min: style.rescale_min,
                rescale_max: style.rescale_max,
                lut_u32,
                output: out_kind.into(),
            },
            out_kind,
            dst_w,
            dst_h,
        ))
    }

    /// Encode an RGBA8 image buffer, automatically routing large `.jpg` outputs through
    /// `GpuJpegEncoder` (WGSL 4:2:0 YCbCr + 8×8 FDCT + Annex K Huffman) above the
    /// adapter-calibrated pixel crossover:
    /// - Discrete / Vulkan (`!unified_memory`): `>= 512×512` (`262_144` px, 2.12×–3.23× faster)
    /// - Unified / Metal (`unified_memory`): `>= 1024×1024` (`1_048_576` px, 1.01×–2.48× faster)
    pub fn encode_rgba_output(
        &self,
        dst_w: usize,
        dst_h: usize,
        rgba: &[u8],
        format: OutputFormat,
        return_mask: bool,
    ) -> Bytes {
        if let OutputFormat::Jpeg { quality } = format {
            let min_px = if self.ctx.info.unified_memory {
                1024 * 1024
            } else {
                512 * 512
            };
            if dst_w.saturating_mul(dst_h) >= min_px {
                if let Ok(mut enc) = self.jpeg.lock() {
                    if let Ok(jpg) = enc.encode_rgba8(rgba, dst_w as u32, dst_h as u32, quality) {
                        return Bytes::from(jpg);
                    }
                }
            }
        }
        crate::titiler::encode_rgba_output(dst_w, dst_h, rgba, format, return_mask)
    }

    /// Compute GPU-accelerated band statistics, histogram, and percentiles for large `f32`
    /// rasters (`>= 262_144` pixels on discrete/Vulkan, `>= 524_288` on unified/Metal).
    pub fn try_reduce_f32_stats(
        &self,
        pixels: &[f32],
        nodata: Option<f32>,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Option<GpuBandStats> {
        let min_px = if self.ctx.info.unified_memory {
            512 * 1024
        } else {
            512 * 512
        };
        if pixels.len() < min_px {
            return None;
        }
        let mut guard = self.stats.lock().ok()?;
        guard
            .reduce_f32(pixels, nodata, num_bins, hist_range, percentiles)
            .ok()
    }

    /// Single-submit GPU weighted zonal statistics across one or more aligned bands sharing
    /// the same analytical coverage mask `covs` and normalized per-row areas `row_areas`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_reduce_weighted_zonal_stats(
        &self,
        bands: &[&[f32]],
        covs: &[f32],
        win_w: usize,
        row_areas: &[f32],
        area_weighted: bool,
        cov_eps: f32,
        pixel_set_eps: f32,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Option<Vec<GpuWeightedBandStats>> {
        let min_px = if self.ctx.info.unified_memory {
            512 * 512
        } else {
            256 * 256
        };
        if covs.len() < min_px || bands.is_empty() {
            return None;
        }
        let mut guard = self.weighted_stats.lock().ok()?;
        guard
            .reduce_weighted_bands(
                bands,
                covs,
                win_w,
                row_areas,
                area_weighted,
                cov_eps,
                pixel_set_eps,
                num_bins,
                hist_range,
                percentiles,
            )
            .ok()
    }

    fn fused_jpeg_quality(&self, dst_w: usize, dst_h: usize, format: OutputFormat) -> Option<u8> {
        if let OutputFormat::Jpeg { quality } = format {
            let min_px = if self.ctx.info.unified_memory {
                1024 * 1024
            } else {
                512 * 512
            };
            if dst_w.saturating_mul(dst_h) >= min_px {
                return Some(quality);
            }
        }
        None
    }

    fn submit_rgba(
        &self,
        job: CoalescedJob,
        dst_w: usize,
        dst_h: usize,
        format: OutputFormat,
        return_mask: bool,
    ) -> Option<Bytes> {
        if let Some(quality) = self.fused_jpeg_quality(dst_w, dst_h, format) {
            let coeffs = self
                .coalescer
                .submit(CoalescedJob::WithJpeg {
                    inner: Box::new(job),
                    quality,
                })
                .ok()?;
            return Some(Bytes::from(cogzilla_gpu::huffman_encode_coeffs(
                &coeffs,
                dst_w as u32,
                dst_h as u32,
                quality,
            )));
        }
        let raw = self.coalescer.submit(job).ok()?;
        Some(if format == OutputFormat::RawRgba {
            Bytes::from(raw)
        } else {
            self.encode_rgba_output(dst_w, dst_h, &raw, format, return_mask)
        })
    }

    async fn submit_rgba_async(
        &self,
        job: CoalescedJob,
        dst_w: usize,
        dst_h: usize,
        format: OutputFormat,
        return_mask: bool,
    ) -> Option<Bytes> {
        if let Some(quality) = self.fused_jpeg_quality(dst_w, dst_h, format) {
            let coeffs = self
                .submit_coalesced_async(CoalescedJob::WithJpeg {
                    inner: Box::new(job),
                    quality,
                })
                .await?;
            return Some(Bytes::from(cogzilla_gpu::huffman_encode_coeffs(
                &coeffs,
                dst_w as u32,
                dst_h as u32,
                quality,
            )));
        }
        let raw = self.submit_coalesced_async(job).await?;
        Some(if format == OutputFormat::RawRgba {
            Bytes::from(raw)
        } else {
            self.encode_rgba_output(dst_w, dst_h, &raw, format, return_mask)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_styled(
        &self,
        job: CoalescedJob,
        out_kind: GpuPixelOutput,
        dst_w: usize,
        dst_h: usize,
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
    ) -> Option<Bytes> {
        if out_kind == GpuPixelOutput::Rgba8 {
            return self.submit_rgba(job, dst_w, dst_h, style.format, return_mask);
        }
        let raw = self.coalescer.submit(job).ok()?;
        Some(self.finish_styled_output(
            raw,
            out_kind,
            dst_w,
            dst_h,
            style,
            rgb_lut,
            return_mask,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_styled_async(
        &self,
        job: CoalescedJob,
        out_kind: GpuPixelOutput,
        dst_w: usize,
        dst_h: usize,
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
    ) -> Option<Bytes> {
        if out_kind == GpuPixelOutput::Rgba8 {
            return self
                .submit_rgba_async(job, dst_w, dst_h, style.format, return_mask)
                .await;
        }
        let raw = self.submit_coalesced_async(job).await?;
        Some(self.finish_styled_output(
            raw,
            out_kind,
            dst_w,
            dst_h,
            style,
            rgb_lut,
            return_mask,
        ))
    }

    fn finish_styled_output(
        &self,
        raw: Vec<u8>,
        out_kind: GpuPixelOutput,
        dst_w: usize,
        dst_h: usize,
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
    ) -> Bytes {
        match out_kind {
            GpuPixelOutput::IndexedU8 => {
                encode_indexed_u8_png(&raw, dst_w as u32, dst_h as u32, rgb_lut)
            }
            GpuPixelOutput::Rgba8 => {
                if style.format == OutputFormat::RawRgba {
                    Bytes::from(raw)
                } else {
                    self.encode_rgba_output(dst_w, dst_h, &raw, style.format, return_mask)
                }
            }
        }
    }

    /// Attempt GPU-accelerated single-band rendering for `/cog/preview`, `/cog/bbox`,
    /// `/cog/crop`, and `/cog/feature` (sync, for `spawn_blocking` callers).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_single_band(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        style: &SingleBandStyle,
        return_mask: bool,
    ) -> Option<Bytes> {
        self.try_render_bbox_single_band_with_lut(
            cog_id,
            ifd_idx,
            band0,
            coords,
            src,
            style,
            &style.colormap.lut().rgb,
            return_mask,
        )
    }

    /// Single-band GPU render with an explicit 256-entry RGB LUT (supporting `CustomColormap`
    /// and zero-cost `ColorFormula` folding).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_single_band_with_lut(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
    ) -> Option<Bytes> {
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_single_band(
            cog_id, ifd_idx, band0, coords, src, style, rgb_lut, return_mask, active,
        )?;
        self.submit_styled(job, out_kind, dst_w, dst_h, style, rgb_lut, return_mask)
    }

    /// Attempt GPU-accelerated DEM Horn 3×3 (`Hillshade` / `Slope` / `Aspect`) rendering for
    /// arbitrary `(u_coords, v_coords)` crops or multi-block tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_crop_dem(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        u_coords: &[f32],
        v_coords: &[f32],
        src: &BandSource<'_>,
        alg: DemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
        style: &SingleBandStyle,
    ) -> Option<Bytes> {
        self.try_render_bbox_dem(
            cog_id,
            ifd_idx,
            band0,
            BboxCoordSource::Separable {
                u: u_coords,
                v: v_coords,
            },
            src,
            alg,
            pixel_size_x_m,
            pixel_size_y_m,
            style,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_bbox_dem(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        alg: DemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
        active: usize,
    ) -> Option<(CoalescedJob, GpuPixelOutput, usize, usize)> {
        let gpu_mode = map_resample(style.resample)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let gpu_alg = match alg {
            DemAlgorithm::Hillshade {
                azimuth_deg,
                altitude_deg,
                z_factor,
            } => GpuDemAlgorithm::Hillshade {
                azimuth_deg,
                altitude_deg,
                z_factor,
            },
            DemAlgorithm::Slope { z_factor } => GpuDemAlgorithm::Slope { z_factor },
            DemAlgorithm::Aspect => GpuDemAlgorithm::Aspect,
            DemAlgorithm::Contours { .. } => return None,
        };
        let out_kind = match style.format {
            OutputFormat::Png if return_mask => GpuPixelOutput::IndexedU8,
            OutputFormat::Png
            | OutputFormat::Webp { .. }
            | OutputFormat::Jpeg { .. }
            | OutputFormat::RawRgba => GpuPixelOutput::Rgba8,
            _ => return None,
        };

        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src) + coord_upload_bytes;
        let readback_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => dst_w * dst_h,
            GpuPixelOutput::Rgba8 => dst_w * dst_h * 4,
        };
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::DemHornLanczos3,
            _ => GpuOp::DemHornCubic,
        };
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }

        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(rgb_lut);
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        Some((
            CoalescedJob::Terrain {
                slab,
                plan,
                mode: gpu_mode,
                alg: gpu_alg,
                pixel_size_x_m,
                pixel_size_y_m,
                rescale_min: style.rescale_min,
                rescale_max: style.rescale_max,
                lut_u32,
                output: out_kind.into(),
            },
            out_kind,
            dst_w,
            dst_h,
        ))
    }

    /// Attempt GPU-accelerated DEM Horn 3×3 rendering for `/cog/preview`, `/cog/bbox`,
    /// `/cog/crop`, and `/cog/feature` (sync, for `spawn_blocking` callers).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_dem(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        alg: DemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
        style: &SingleBandStyle,
        return_mask: bool,
    ) -> Option<Bytes> {
        self.try_render_bbox_dem_with_lut(
            cog_id,
            ifd_idx,
            band0,
            coords,
            src,
            alg,
            pixel_size_x_m,
            pixel_size_y_m,
            style,
            &style.colormap.lut().rgb,
            return_mask,
        )
    }

    /// DEM Horn 3×3 GPU render with an explicit 256-entry RGB LUT (supporting `CustomColormap`
    /// and zero-cost `ColorFormula` folding).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_dem_with_lut(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        alg: DemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
    ) -> Option<Bytes> {
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_dem(
            cog_id,
            ifd_idx,
            band0,
            coords,
            src,
            alg,
            pixel_size_x_m,
            pixel_size_y_m,
            style,
            rgb_lut,
            return_mask,
            active,
        )?;
        self.submit_styled(job, out_kind, dst_w, dst_h, style, rgb_lut, return_mask)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_bbox_terrain(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        encoding: TerrainEncoding,
        resample: ResampleMode,
        format: OutputFormat,
        active: usize,
    ) -> Option<(CoalescedJob, usize, usize)> {
        let gpu_mode = map_resample(resample)?;
        let gpu_alg = match encoding {
            TerrainEncoding::Terrarium => GpuDemAlgorithm::Terrarium,
            TerrainEncoding::TerrainRgb => GpuDemAlgorithm::TerrainRgb,
            TerrainEncoding::Contours { .. } => return None,
        };
        if !matches!(
            format,
            OutputFormat::Png
                | OutputFormat::Webp { .. }
                | OutputFormat::Jpeg { .. }
                | OutputFormat::RawRgba
        ) {
            return None;
        }
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src) + coord_upload_bytes;
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::ResampleLanczos3,
            _ => GpuOp::DemTerrainRgbCubic,
        };
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes: dst_w * dst_h * 4,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }

        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        Some((
            CoalescedJob::Terrain {
                slab,
                plan,
                mode: gpu_mode,
                alg: gpu_alg,
                pixel_size_x_m: 30.0,
                pixel_size_y_m: 30.0,
                rescale_min: 0.0,
                rescale_max: 255.0,
                lut_u32: [0u32; 256],
                output: CoalescedOutput::Rgba8,
            },
            dst_w,
            dst_h,
        ))
    }

    /// Attempt GPU-accelerated Terrarium / TerrainRGB rendering for `/cog/preview` & `/cog/bbox`
    /// (sync, for `spawn_blocking` callers).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_terrain(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        encoding: TerrainEncoding,
        resample: ResampleMode,
        format: OutputFormat,
        return_mask: bool,
    ) -> Option<Bytes> {
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let (job, dst_w, dst_h) = self.prepare_bbox_terrain(
            cog_id, ifd_idx, band0, coords, src, encoding, resample, format, active,
        )?;
        self.submit_rgba(job, dst_w, dst_h, format, return_mask)
    }

    /// Attempt GPU-accelerated single-band tile rendering (non-blocking async coalesced).
    #[allow(clippy::too_many_arguments)]
    pub async fn try_render_single_band<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        _tx0: u32,
        _ty0: u32,
        plan: &CoordPlan<N>,
        src: &BandSource<'_>,
        style: &SingleBandStyle,
    ) -> Option<Bytes> {
        let coords = match plan {
            CoordPlan::Separable1D { u, v, .. } => BboxCoordSource::Separable {
                u: u.as_slice(),
                v: v.as_slice(),
            },
            CoordPlan::Grid2D { grid_u, grid_v, .. } => BboxCoordSource::Grid2D {
                dst_w: N,
                dst_h: N,
                grid_u,
                grid_v,
            },
            CoordPlan::Exact2D { u, v, .. } => BboxCoordSource::Exact2D {
                dst_w: N,
                dst_h: N,
                u_map: &u[..],
                v_map: &v[..],
            },
        };
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let rgb_lut = &style.colormap.lut().rgb;
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_single_band(
            cog_id, ifd_idx, band0, coords, src, style, rgb_lut, true, active,
        )?;
        self.submit_styled_async(job, out_kind, dst_w, dst_h, style, rgb_lut, true)
            .await
    }

    /// Attempt GPU-accelerated DEM Horn 3×3 tile rendering (non-blocking async coalesced).
    #[allow(clippy::too_many_arguments)]
    pub async fn try_render_dem<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        _tx0: u32,
        _ty0: u32,
        plan: &CoordPlan<N>,
        src: &BandSource<'_>,
        alg: DemAlgorithm,
        pixel_size_m: f32,
        style: &SingleBandStyle,
    ) -> Option<Bytes> {
        let coords = match plan {
            CoordPlan::Separable1D { u, v, .. } => BboxCoordSource::Separable {
                u: u.as_slice(),
                v: v.as_slice(),
            },
            CoordPlan::Grid2D { grid_u, grid_v, .. } => BboxCoordSource::Grid2D {
                dst_w: N,
                dst_h: N,
                grid_u,
                grid_v,
            },
            CoordPlan::Exact2D { u, v, .. } => BboxCoordSource::Exact2D {
                dst_w: N,
                dst_h: N,
                u_map: &u[..],
                v_map: &v[..],
            },
        };
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let rgb_lut = &style.colormap.lut().rgb;
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_dem(
            cog_id,
            ifd_idx,
            band0,
            coords,
            src,
            alg,
            pixel_size_m,
            pixel_size_m,
            style,
            rgb_lut,
            true,
            active,
        )?;
        self.submit_styled_async(job, out_kind, dst_w, dst_h, style, rgb_lut, true)
            .await
    }

    /// Attempt GPU-accelerated `Terrarium` / `TerrainRgb` tile rendering (non-blocking async coalesced).
    #[allow(clippy::too_many_arguments)]
    pub async fn try_render_terrain<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        _tx0: u32,
        _ty0: u32,
        plan: &CoordPlan<N>,
        src: &BandSource<'_>,
        encoding: TerrainEncoding,
        resample: ResampleMode,
        format: OutputFormat,
    ) -> Option<Bytes> {
        let coords = match plan {
            CoordPlan::Separable1D { u, v, .. } => BboxCoordSource::Separable {
                u: u.as_slice(),
                v: v.as_slice(),
            },
            CoordPlan::Grid2D { grid_u, grid_v, .. } => BboxCoordSource::Grid2D {
                dst_w: N,
                dst_h: N,
                grid_u,
                grid_v,
            },
            CoordPlan::Exact2D { u, v, .. } => BboxCoordSource::Exact2D {
                dst_w: N,
                dst_h: N,
                u_map: &u[..],
                v_map: &v[..],
            },
        };
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let (job, dst_w, dst_h) = self.prepare_bbox_terrain(
            cog_id, ifd_idx, band0, coords, src, encoding, resample, format, active,
        )?;
        self.submit_rgba_async(job, dst_w, dst_h, format, true).await
    }

    /// Translate a compiled `cogzilla-kernel::ExprProgram` into GPU WGSL VM bytecode.
    /// Returns `None` if the expression uses an op not supported by the GPU VM (e.g. `Median(n)`),
    /// causing seamless CPU fallback.
    pub fn compile_expr_to_gpu(prog: &ExprProgram) -> Option<Vec<GpuInstr>> {
        if prog.ops.is_empty() || prog.ops.len() > 256 || prog.used_bands.len() > 8 {
            return None;
        }
        let mut out = Vec::with_capacity(prog.ops.len());
        for &op in &prog.ops {
            let ins = match op {
                Op::PushConst(v) => GpuInstr::push_const(v),
                Op::LoadBand(slot) => GpuInstr::load_band(slot as u32),
                Op::Add => GpuInstr::op(opcodes::ADD),
                Op::Sub => GpuInstr::op(opcodes::SUB),
                Op::Mul => GpuInstr::op(opcodes::MUL),
                Op::Div => GpuInstr::op(opcodes::DIV),
                Op::Pow => GpuInstr::op(opcodes::POW),
                Op::Neg => GpuInstr::op(opcodes::NEG),
                Op::Log10 => GpuInstr::op(opcodes::LOG10),
                Op::Ln => GpuInstr::op(opcodes::LN),
                Op::Log2 => GpuInstr::op(opcodes::LOG2),
                Op::Db => GpuInstr::op(opcodes::DB),
                Op::Exp => GpuInstr::op(opcodes::EXP),
                Op::Sqrt => GpuInstr::op(opcodes::SQRT),
                Op::Abs => GpuInstr::op(opcodes::ABS),
                Op::Sin => GpuInstr::op(opcodes::SIN),
                Op::Cos => GpuInstr::op(opcodes::COS),
                Op::Tan => GpuInstr::op(opcodes::TAN),
                Op::Min => GpuInstr::op(opcodes::MIN),
                Op::Max => GpuInstr::op(opcodes::MAX),
                Op::Clip => GpuInstr::op(opcodes::CLIP),
                Op::Lt => GpuInstr::op(opcodes::LT),
                Op::Le => GpuInstr::op(opcodes::LE),
                Op::Gt => GpuInstr::op(opcodes::GT),
                Op::Ge => GpuInstr::op(opcodes::GE),
                Op::Eq => GpuInstr::op(opcodes::EQ),
                Op::Ne => GpuInstr::op(opcodes::NE),
                Op::Where => GpuInstr::op(opcodes::WHERE),
                Op::Ceil => GpuInstr::op(opcodes::CEIL),
                Op::Floor => GpuInstr::op(opcodes::FLOOR),
                Op::Round => GpuInstr::op(opcodes::ROUND),
                _ => return None,
            };
            out.push(ins);
        }
        Some(out)
    }

    /// Build a cache key for a single-COG multi-band `GpuMultiBandSlab` when all band sources
    /// share identical tile-grid geometry and have no per-request nodata/scale overrides.
    pub fn multiband_key(
        cog_id: u64,
        ifd_idx: usize,
        bands_1based: &[u16],
        srcs: &[BandSource<'_>],
    ) -> Option<MultiBandSlabKey> {
        if srcs.is_empty() || srcs.len() > 8 || bands_1based.len() != srcs.len() {
            return None;
        }
        let s0 = &srcs[0];
        for s in srcs {
            if s.tx0 != s0.tx0
                || s.ty0 != s0.ty0
                || s.grid_w != s0.grid_w
                || s.grid_h != s0.grid_h
                || s.tile_w != s0.tile_w
                || s.tile_h != s0.tile_h
                || s.nodata_override.is_some()
                || s.scale_offset.is_some()
            {
                return None;
            }
        }
        let mut bands = [0u16; 8];
        for (i, &b) in bands_1based.iter().enumerate() {
            bands[i] = b;
        }
        Some(MultiBandSlabKey {
            dataset_id: cog_id,
            ifd_idx: ifd_idx as u16,
            num_bands: srcs.len() as u8,
            bands,
            tx: (s0.tx0 << 8) | (s0.grid_w & 0xff),
            ty: (s0.ty0 << 8) | (s0.grid_h & 0xff),
        })
    }

    fn translate_color_formula(cf: Option<&ColorFormula>) -> Option<GpuColorFormula> {
        match cf {
            None => Some(GpuColorFormula {
                rgb_lut: None,
                saturation: None,
            }),
            Some(formula) => {
                let (lut_opt, saturation) = formula.as_lut_then_saturation()?;
                let rgb_lut = lut_opt.map(|(r, g, b)| GpuColorFormula::pack_rgb_luts(r, g, b));
                Some(GpuColorFormula {
                    rgb_lut,
                    saturation,
                })
            }
        }
    }

    /// Compile a built-in `SpectralExpr` (`Db`, `NormalizedDiff`, `Savi`, `Ratio`, `Diff`, `Evi`)
    /// directly into GPU VM bytecode without string parsing overhead.
    pub fn compile_spectral_to_gpu(expr: SpectralExpr) -> Option<Vec<GpuInstr>> {
        let instrs = match expr {
            SpectralExpr::Db => vec![GpuInstr::load_band(0), GpuInstr::op(opcodes::DB)],
            SpectralExpr::NormalizedDiff => vec![
                GpuInstr::load_band(0),
                GpuInstr::load_band(1),
                GpuInstr::op(opcodes::SUB),
                GpuInstr::load_band(0),
                GpuInstr::load_band(1),
                GpuInstr::op(opcodes::ADD),
                GpuInstr::op(opcodes::DIV),
            ],
            SpectralExpr::Savi => vec![
                GpuInstr::push_const(1.5),
                GpuInstr::load_band(0),
                GpuInstr::load_band(1),
                GpuInstr::op(opcodes::SUB),
                GpuInstr::op(opcodes::MUL),
                GpuInstr::load_band(0),
                GpuInstr::load_band(1),
                GpuInstr::op(opcodes::ADD),
                GpuInstr::push_const(0.5),
                GpuInstr::op(opcodes::ADD),
                GpuInstr::op(opcodes::DIV),
            ],
            SpectralExpr::Ratio => vec![
                GpuInstr::load_band(0),
                GpuInstr::load_band(1),
                GpuInstr::op(opcodes::DIV),
            ],
            SpectralExpr::Diff => vec![
                GpuInstr::load_band(0),
                GpuInstr::load_band(1),
                GpuInstr::op(opcodes::SUB),
            ],
            SpectralExpr::Evi => vec![
                GpuInstr::push_const(2.5),
                GpuInstr::load_band(0),
                GpuInstr::load_band(1),
                GpuInstr::op(opcodes::SUB),
                GpuInstr::op(opcodes::MUL),
                GpuInstr::load_band(0),
                GpuInstr::push_const(6.0),
                GpuInstr::load_band(1),
                GpuInstr::op(opcodes::MUL),
                GpuInstr::op(opcodes::ADD),
                GpuInstr::push_const(7.5),
                GpuInstr::load_band(2),
                GpuInstr::op(opcodes::MUL),
                GpuInstr::op(opcodes::SUB),
                GpuInstr::push_const(1.0),
                GpuInstr::op(opcodes::ADD),
                GpuInstr::op(opcodes::DIV),
            ],
            SpectralExpr::SarRgb => return None,
        };
        Some(instrs)
    }

    /// Attempt GPU-accelerated built-in `SpectralExpr` tile rendering across all `CoordPlan`
    /// variants (`Separable1D`, `Grid2D` 17×17 UTM/Albers/LCC, and `Exact2D`).
    /// Resolve `(Arc<GpuMultiBandSlab>, base_col, base_row)` from `multi_slab_cache` (lock-free on
    /// warm hit) or stitch + upload via `self.expr` on a cold miss.
    fn resolve_multiband_slab(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        warm_slab: Option<Arc<cogzilla_gpu::GpuMultiBandSlab>>,
        srcs: &[BandSource<'_>],
    ) -> Option<(Arc<cogzilla_gpu::GpuMultiBandSlab>, u32, u32)> {
        if let Some(slab) = warm_slab {
            let bc = srcs[0].tx0.checked_mul(srcs[0].tile_w)?;
            let br = srcs[0].ty0.checked_mul(srcs[0].tile_h)?;
            return Some((slab, bc, br));
        }
        let (base_col, base_row, w, h) = stitch_band_source_geom(&srcs[0])?;
        for src in &srcs[1..] {
            if stitch_band_source_geom(src)? != (base_col, base_row, w, h) {
                return None;
            }
        }
        let plane = (w as usize) * (h as usize);
        let slab = Arc::new(
            cogzilla_gpu::GpuMultiBandSlab::from_mapped_fill(
                &self.ctx.device,
                srcs.len(),
                w,
                h,
                w,
                |dst| {
                    if srcs.len() >= 2 && plane >= 262_144 {
                        std::thread::scope(|s| {
                            for (src, plane_dst) in srcs.iter().zip(dst.chunks_exact_mut(plane)) {
                                s.spawn(move || {
                                    stitch_band_source_into(src, w, h, plane_dst);
                                });
                            }
                        });
                    } else {
                        for (src, plane_dst) in srcs.iter().zip(dst.chunks_exact_mut(plane)) {
                            stitch_band_source_into(src, w, h, plane_dst);
                        }
                    }
                    Ok(())
                },
            )
            .ok()?,
        );
        if let Some(k) = cache_key {
            self.multi_slab_cache.insert(k, Arc::clone(&slab));
        }
        Some((slab, base_col, base_row))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_bbox_instrs(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        instrs: &[GpuInstr],
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
        active: usize,
    ) -> Option<(CoalescedJob, GpuPixelOutput, usize, usize)> {
        let gpu_mode = map_resample(style.resample)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || srcs.is_empty() || srcs.len() > 8 {
            return None;
        }
        let out_kind = match style.format {
            OutputFormat::Png if return_mask => GpuPixelOutput::IndexedU8,
            OutputFormat::Png
            | OutputFormat::Webp { .. }
            | OutputFormat::Jpeg { .. }
            | OutputFormat::RawRgba => GpuPixelOutput::Rgba8,
            _ => return None,
        };
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let band_upload_bytes: usize = if warm_slab.is_some() {
            0
        } else {
            srcs.iter()
                .map(|s| Self::estimate_upload_bytes(&None, s))
                .sum()
        };
        let upload_bytes = band_upload_bytes + coord_upload_bytes;
        let readback_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => dst_w * dst_h,
            GpuPixelOutput::Rgba8 => dst_w * dst_h * 4,
        };
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::FusedColormapLanczos3,
            GpuColormapMode::Cubic => GpuOp::DemHornCubic,
            GpuColormapMode::Bilinear | GpuColormapMode::Nearest => GpuOp::FusedColormapBilinear,
        };
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }
        let (slab, base_col, base_row) = self.resolve_multiband_slab(cache_key, warm_slab, srcs)?;
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(rgb_lut);
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        Some((
            CoalescedJob::Expr {
                slab,
                plan,
                mode: gpu_mode,
                instrs: instrs.to_vec(),
                rescale_min: style.rescale_min,
                rescale_max: style.rescale_max,
                lut_u32,
                output: out_kind.into(),
            },
            out_kind,
            dst_w,
            dst_h,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn try_render_spectral<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        bands_1based: &[u16],
        plan: &CoordPlan<N>,
        srcs: &[BandSource<'_>],
        expr: SpectralExpr,
        style: &SingleBandStyle,
    ) -> Option<Bytes> {
        let instrs = Self::compile_spectral_to_gpu(expr)?;
        let coords = match plan {
            CoordPlan::Separable1D { u, v, .. } => BboxCoordSource::Separable {
                u: u.as_slice(),
                v: v.as_slice(),
            },
            CoordPlan::Grid2D { grid_u, grid_v, .. } => BboxCoordSource::Grid2D {
                dst_w: N,
                dst_h: N,
                grid_u,
                grid_v,
            },
            CoordPlan::Exact2D { u, v, .. } => BboxCoordSource::Exact2D {
                dst_w: N,
                dst_h: N,
                u_map: &u[..],
                v_map: &v[..],
            },
        };
        let cache_key = Self::multiband_key(cog_id, ifd_idx, bands_1based, srcs);
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let rgb_lut = &style.colormap.lut().rgb;
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_instrs(
            cache_key, coords, srcs, &instrs, style, rgb_lut, true, active,
        )?;
        self.submit_styled_async(job, out_kind, dst_w, dst_h, style, rgb_lut, true)
            .await
    }

    /// Attempt GPU-accelerated built-in `SpectralExpr` visual rendering for arbitrary
    /// `BboxCoordSource` (`/cog/preview`, `/cog/bbox`, `/cog/crop`, `/cog/feature`, and tiles).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_spectral(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        expr: SpectralExpr,
        style: &SingleBandStyle,
        return_mask: bool,
    ) -> Option<Bytes> {
        self.try_render_bbox_spectral_with_lut(
            cache_key,
            coords,
            srcs,
            expr,
            style,
            &style.colormap.lut().rgb,
            return_mask,
        )
    }

    /// Built-in `SpectralExpr` GPU render with an explicit 256-entry RGB LUT.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_spectral_with_lut(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        expr: SpectralExpr,
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
    ) -> Option<Bytes> {
        let instrs = Self::compile_spectral_to_gpu(expr)?;
        self.try_render_bbox_instrs(cache_key, coords, srcs, &instrs, style, rgb_lut, return_mask)
    }

    /// Attempt GPU-accelerated built-in `SpectralExpr` raw `f32` rendering for `.npy`, `.tif`,
    /// `.f32`, and multi-scene mosaic routes.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_spectral_f32(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        expr: SpectralExpr,
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let instrs = Self::compile_spectral_to_gpu(expr)?;
        self.try_render_bbox_instrs_f32(cache_key, coords, srcs, &instrs, resample)
    }

    /// Attempt GPU-accelerated multi-band `ExprProgram` tile rendering (non-blocking async coalesced).
    #[allow(clippy::too_many_arguments)]
    pub async fn try_render_expr<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        plan: &CoordPlan<N>,
        srcs: &[BandSource<'_>],
        prog: &ExprProgram,
        style: &SingleBandStyle,
    ) -> Option<Bytes> {
        let instrs = Self::compile_expr_to_gpu(prog)?;
        let coords = match plan {
            CoordPlan::Separable1D { u, v, .. } => BboxCoordSource::Separable {
                u: u.as_slice(),
                v: v.as_slice(),
            },
            CoordPlan::Grid2D { grid_u, grid_v, .. } => BboxCoordSource::Grid2D {
                dst_w: N,
                dst_h: N,
                grid_u,
                grid_v,
            },
            CoordPlan::Exact2D { u, v, .. } => BboxCoordSource::Exact2D {
                dst_w: N,
                dst_h: N,
                u_map: &u[..],
                v_map: &v[..],
            },
        };
        let cache_key = Self::multiband_key(cog_id, ifd_idx, &prog.used_bands, srcs);
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let rgb_lut = &style.colormap.lut().rgb;
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_instrs(
            cache_key, coords, srcs, &instrs, style, rgb_lut, true, active,
        )?;
        self.submit_styled_async(job, out_kind, dst_w, dst_h, style, rgb_lut, true)
            .await
    }

    /// Attempt GPU-accelerated 3-band RGB tile rendering (non-blocking async coalesced).
    #[allow(clippy::too_many_arguments)]
    pub async fn try_render_rgb<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        bands_1based: [u16; 3],
        plan: &CoordPlan<N>,
        srcs: &[BandSource<'_>; 3],
        style: &RgbStyle,
    ) -> Option<Bytes> {
        let coords = match plan {
            CoordPlan::Separable1D { u, v, .. } => BboxCoordSource::Separable {
                u: u.as_slice(),
                v: v.as_slice(),
            },
            CoordPlan::Grid2D { grid_u, grid_v, .. } => BboxCoordSource::Grid2D {
                dst_w: N,
                dst_h: N,
                grid_u,
                grid_v,
            },
            CoordPlan::Exact2D { u, v, .. } => BboxCoordSource::Exact2D {
                dst_w: N,
                dst_h: N,
                u_map: &u[..],
                v_map: &v[..],
            },
        };
        let cache_key = Self::multiband_key(cog_id, ifd_idx, &bands_1based, srcs);
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let (job, dst_w, dst_h) =
            self.prepare_bbox_rgb(cache_key, coords, srcs, style, None, active)?;
        self.submit_rgba_async(job, dst_w, dst_h, style.format, true)
            .await
    }

    /// Attempt GPU-accelerated multi-band `ExprProgram` rendering for arbitrary `BboxCoordSource`
    /// (`Separable`, `Grid2D`, or `Exact2D` for `/cog/preview`, `/cog/bbox`, `/cog/feature`, and tiles).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_expr(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        prog: &ExprProgram,
        style: &SingleBandStyle,
        return_mask: bool,
    ) -> Option<Bytes> {
        self.try_render_bbox_expr_with_lut(
            cache_key,
            coords,
            srcs,
            prog,
            style,
            &style.colormap.lut().rgb,
            return_mask,
        )
    }

    /// Multi-band `ExprProgram` GPU render with an explicit 256-entry RGB LUT.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_expr_with_lut(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        prog: &ExprProgram,
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
    ) -> Option<Bytes> {
        let instrs = Self::compile_expr_to_gpu(prog)?;
        self.try_render_bbox_instrs(cache_key, coords, srcs, &instrs, style, rgb_lut, return_mask)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_instrs(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        instrs: &[GpuInstr],
        style: &SingleBandStyle,
        rgb_lut: &[[u8; 3]; 256],
        return_mask: bool,
    ) -> Option<Bytes> {
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_instrs(
            cache_key,
            coords,
            srcs,
            instrs,
            style,
            rgb_lut,
            return_mask,
            active,
        )?;
        self.submit_styled(job, out_kind, dst_w, dst_h, style, rgb_lut, return_mask)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_bbox_rgb(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>; 3],
        style: &RgbStyle,
        color_formula: Option<&ColorFormula>,
        active: usize,
    ) -> Option<(CoalescedJob, usize, usize)> {
        let gpu_mode = map_resample(style.resample)?;
        let gpu_formula = Self::translate_color_formula(color_formula)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        if !matches!(
            style.format,
            OutputFormat::Png
                | OutputFormat::Webp { .. }
                | OutputFormat::Jpeg { .. }
                | OutputFormat::RawRgba
        ) {
            return None;
        }
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let band_upload_bytes: usize = if warm_slab.is_some() {
            0
        } else {
            srcs.iter()
                .map(|s| Self::estimate_upload_bytes(&None, s))
                .sum()
        };
        let upload_bytes = band_upload_bytes + coord_upload_bytes;
        let readback_bytes = dst_w * dst_h * 4;
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::FusedColormapLanczos3,
            GpuColormapMode::Cubic => GpuOp::DemHornCubic,
            GpuColormapMode::Bilinear | GpuColormapMode::Nearest => GpuOp::FusedColormapBilinear,
        };
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }
        let (slab, base_col, base_row) = self.resolve_multiband_slab(cache_key, warm_slab, srcs)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        Some((
            CoalescedJob::Rgb {
                slab,
                plan,
                mode: gpu_mode,
                rescale_min: style.rescale_min,
                rescale_max: style.rescale_max,
                formula: gpu_formula,
            },
            dst_w,
            dst_h,
        ))
    }

    /// Attempt GPU-accelerated 3-band RGB rendering (with optional fused `ColorFormula`:
    /// Gamma/Sigmoidal LUT + CIE LCH Saturation) for arbitrary `BboxCoordSource`
    /// (`Separable`, `Grid2D`, or `Exact2D` for `/cog/preview`, `/cog/bbox`, `/cog/feature`, and tiles).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_rgb(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>; 3],
        style: &RgbStyle,
        color_formula: Option<&ColorFormula>,
        return_mask: bool,
    ) -> Option<Bytes> {
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let (job, dst_w, dst_h) =
            self.prepare_bbox_rgb(cache_key, coords, srcs, style, color_formula, active)?;
        self.submit_rgba(job, dst_w, dst_h, style.format, return_mask)
    }

    fn mosaic_policy_op(method: GpuPixelSelection, num_scenes: usize) -> GpuOp {
        match method {
            GpuPixelSelection::Median if num_scenes >= 6 => GpuOp::DemHornLanczos3,
            GpuPixelSelection::Median | GpuPixelSelection::Stdev => GpuOp::DemHornCubic,
            _ => GpuOp::FusedColormapBilinear,
        }
    }

    /// Attempt GPU-accelerated multi-scene pixel selection (`First`, `Highest`, `Lowest`, `Mean`,
    /// `Median`, `Stdev`, `Count`) over `2..=32` aligned `f32` planes, returning reduced `f32` pixels.
    pub fn try_composite_mosaic_f32(
        &self,
        width: u32,
        height: u32,
        planes: &[&[f32]],
        method: GpuPixelSelection,
    ) -> Option<Vec<f32>> {
        if planes.len() < 2 || planes.len() > 32 || width == 0 || height == 0 {
            return None;
        }
        let pixels = (width as usize).checked_mul(height as usize)?;
        let upload_bytes = planes.len().saturating_mul(pixels).saturating_mul(4);
        let readback_bytes = pixels.saturating_mul(4);
        let decision = self.policy.decide(&DispatchRequest {
            op: Self::mosaic_policy_op(method, planes.len()),
            dst_pixels: pixels,
            upload_bytes,
            readback_bytes,
            batch_size: 1,
            in_flight: self.in_flight.load(Ordering::Relaxed),
        });
        if !decision.use_gpu {
            return None;
        }

        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let _guard = InFlightGuard(&self.in_flight);

        let mut mosaic_guard = self.mosaic.lock().ok()?;
        mosaic_guard
            .composite_f32_direct(planes, width, height, method)
            .ok()
    }

    /// Attempt GPU-accelerated multi-band, multi-scene pixel selection (`First`, `Highest`,
    /// `Lowest`, `Mean`, `Median`, `Stdev`, `Count`) over `B` bands × `2..=32` aligned `f32`
    /// planes in **1 `CommandEncoder` → 1 `queue.submit()` → 1 `map_async()`**.
    pub fn try_composite_mosaic_f32_bands(
        &self,
        width: u32,
        height: u32,
        band_planes: &[Vec<&[f32]>],
        method: GpuPixelSelection,
    ) -> Option<Vec<Vec<f32>>> {
        if band_planes.is_empty() || width == 0 || height == 0 {
            return None;
        }
        let num_scenes = band_planes[0].len();
        if !(2..=32).contains(&num_scenes)
            || band_planes.iter().any(|bp| bp.len() != num_scenes)
        {
            return None;
        }
        if band_planes.len() == 1 {
            return self
                .try_composite_mosaic_f32(width, height, &band_planes[0], method)
                .map(|v| vec![v]);
        }
        let pixels = (width as usize).checked_mul(height as usize)?;
        let nb = band_planes.len();
        let upload_bytes = nb
            .saturating_mul(num_scenes)
            .saturating_mul(pixels)
            .saturating_mul(4);
        let readback_bytes = nb.saturating_mul(pixels).saturating_mul(4);
        let decision = self.policy.decide(&DispatchRequest {
            op: Self::mosaic_policy_op(method, num_scenes),
            dst_pixels: pixels.saturating_mul(nb),
            upload_bytes,
            readback_bytes,
            batch_size: nb,
            in_flight: self.in_flight.load(Ordering::Relaxed),
        });
        if !decision.use_gpu {
            return None;
        }

        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let _guard = InFlightGuard(&self.in_flight);

        let mut mosaic_guard = self.mosaic.lock().ok()?;
        mosaic_guard
            .composite_f32_bands_direct(band_planes, width, height, method)
            .ok()
    }

    /// Attempt fused GPU multi-scene pixel selection + rescale + colormap directly to PNG / WebP /
    /// JPEG / RawRgba (using 4×-smaller `IndexedU8` readback for masked PNGs and 1-pass fused
    /// Mosaic → GPU JPEG FDCT with zero RGBA8 readback for `OutputFormat::Jpeg`).
    #[allow(clippy::too_many_arguments)]
    pub fn try_composite_mosaic_styled(
        &self,
        width: u32,
        height: u32,
        planes: &[&[f32]],
        method: GpuPixelSelection,
        rescale: (f32, f32),
        colormap: ColormapName,
        format: OutputFormat,
        return_mask: bool,
    ) -> Option<Bytes> {
        if planes.len() < 2 || planes.len() > 32 || width == 0 || height == 0 {
            return None;
        }
        let out_kind = match format {
            OutputFormat::Png if return_mask => GpuPixelOutput::IndexedU8,
            OutputFormat::Png
            | OutputFormat::Webp { .. }
            | OutputFormat::Jpeg { .. }
            | OutputFormat::RawRgba => GpuPixelOutput::Rgba8,
            _ => return None,
        };
        let pixels = (width as usize).checked_mul(height as usize)?;
        let upload_bytes = planes.len().saturating_mul(pixels).saturating_mul(4);
        let readback_bytes = match format {
            OutputFormat::Jpeg { .. } => {
                let mcu_w = (width as usize).div_ceil(16);
                let mcu_h = (height as usize).div_ceil(16);
                mcu_w.saturating_mul(mcu_h).saturating_mul(192 * 4)
            }
            _ => match out_kind {
                GpuPixelOutput::IndexedU8 => pixels,
                GpuPixelOutput::Rgba8 => pixels.saturating_mul(4),
            },
        };
        let decision = self.policy.decide(&DispatchRequest {
            op: Self::mosaic_policy_op(method, planes.len()),
            dst_pixels: pixels,
            upload_bytes,
            readback_bytes,
            batch_size: 1,
            in_flight: self.in_flight.load(Ordering::Relaxed),
        });
        if !decision.use_gpu {
            return None;
        }

        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let _guard = InFlightGuard(&self.in_flight);

        let lut = colormap.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
        let mut mosaic_guard = self.mosaic.lock().ok()?;

        // 1-pass fused Mosaic -> GPU JPEG FDCT: zero per-call VRAM alloc, zero RGBA8 readback,
        // zero CPU JPEG encode.
        if let OutputFormat::Jpeg { quality } = format {
            if let Ok(mut jpeg_guard) = self.jpeg.lock() {
                if let Ok(jpg) = mosaic_guard.composite_styled_jpeg_direct(
                    planes,
                    width,
                    height,
                    method,
                    rescale.0,
                    rescale.1,
                    &lut_u32,
                    &mut jpeg_guard,
                    quality,
                ) {
                    return Some(Bytes::from(jpg));
                }
            }
        }

        let raw = mosaic_guard
            .composite_styled_direct(
                planes,
                width,
                height,
                method,
                rescale.0,
                rescale.1,
                &lut_u32,
                out_kind,
            )
            .ok()?;
        drop(mosaic_guard);

        Some(match out_kind {
            GpuPixelOutput::IndexedU8 => encode_indexed_u8_png(&raw, width, height, &lut.rgb),
            GpuPixelOutput::Rgba8 => {
                if format == OutputFormat::RawRgba {
                    Bytes::from(raw)
                } else {
                    crate::titiler::encode_rgba_output(
                        width as usize,
                        height as usize,
                        &raw,
                        format,
                        return_mask,
                    )
                }
            }
        })
    }

    /// Attempt GPU-accelerated raw `f32` single-band resampling for `.npy`, `.tif`, `.f32`,
    /// and multi-scene mosaic scene fetches across `Separable`, `Grid2D`, and `Exact2D`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_single_band_f32(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample(resample)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src) + coord_upload_bytes;
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let decision = self.policy.decide(&DispatchRequest {
            op: gpu_mode.to_policy_op(),
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes: dst_w * dst_h * 4,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }

        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        let raw = self
            .coalescer
            .submit(CoalescedJob::Colormap {
                slab,
                plan,
                mode: gpu_mode,
                rescale_min: 0.0,
                rescale_max: 1.0,
                lut_u32: [0u32; 256],
                output: CoalescedOutput::RawF32,
            })
            .ok()?;
        Some(Bytes::from(raw))
    }

    /// 1-pass fused GPU single-band resampling + band statistics/histogram/percentiles
    /// (`CoalescedJob::WithStats` — zero `f32` readback, zero re-upload, 12 KB readback).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_single_band_stats(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        resample: ResampleMode,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Option<GpuBandStats> {
        let gpu_mode = map_resample(resample)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let min_px = if self.ctx.info.unified_memory {
            512 * 1024
        } else {
            512 * 512
        };
        if dst_w.saturating_mul(dst_h) < min_px {
            return None;
        }
        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let _ = coord_upload_bytes;
        let _active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);

        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        let raw = self
            .coalescer
            .submit(CoalescedJob::WithStats {
                inner: Box::new(CoalescedJob::Colormap {
                    slab,
                    plan,
                    mode: gpu_mode,
                    rescale_min: 0.0,
                    rescale_max: 1.0,
                    lut_u32: [0u32; 256],
                    output: CoalescedOutput::RawF32,
                }),
                nodata: None,
                num_bins,
                hist_range,
            })
            .ok()?;
        Some(cogzilla_gpu::decode_stats_readback(
            &raw,
            dst_w * dst_h,
            num_bins,
            hist_range,
            percentiles,
        ))
    }

    /// 1-pass fused GPU expression evaluation + band statistics/histogram/percentiles
    /// (`CoalescedJob::WithStats` — zero `f32` readback, zero re-upload, 12 KB readback).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_expr_stats(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        prog: &ExprProgram,
        resample: ResampleMode,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Option<GpuBandStats> {
        let instrs = Self::compile_expr_to_gpu(prog)?;
        self.try_render_bbox_instrs_stats(
            cache_key,
            coords,
            srcs,
            instrs,
            resample,
            num_bins,
            hist_range,
            percentiles,
        )
    }

    /// 1-pass fused GPU built-in spectral index (`SpectralExpr`) + band statistics/histogram/percentiles
    /// (`CoalescedJob::WithStats` — zero `f32` readback, zero re-upload, 12 KB readback).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_spectral_stats(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        expr: SpectralExpr,
        resample: ResampleMode,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Option<GpuBandStats> {
        let instrs = Self::compile_spectral_to_gpu(expr)?;
        self.try_render_bbox_instrs_stats(
            cache_key,
            coords,
            srcs,
            instrs,
            resample,
            num_bins,
            hist_range,
            percentiles,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_instrs_stats(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        instrs: Vec<GpuInstr>,
        resample: ResampleMode,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Option<GpuBandStats> {
        let gpu_mode = map_resample(resample)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || srcs.is_empty() || srcs.len() > 8 {
            return None;
        }
        let min_px = if self.ctx.info.unified_memory {
            512 * 1024
        } else {
            512 * 512
        };
        if dst_w.saturating_mul(dst_h) < min_px {
            return None;
        }
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let _ = coord_upload_bytes;
        let _active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);

        let (slab, base_col, base_row) = self.resolve_multiband_slab(cache_key, warm_slab, srcs)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        let raw = self
            .coalescer
            .submit(CoalescedJob::WithStats {
                inner: Box::new(CoalescedJob::Expr {
                    slab,
                    plan,
                    mode: gpu_mode,
                    instrs,
                    rescale_min: 0.0,
                    rescale_max: 1.0,
                    lut_u32: [0u32; 256],
                    output: CoalescedOutput::RawF32,
                }),
                nodata: None,
                num_bins,
                hist_range,
            })
            .ok()?;
        Some(cogzilla_gpu::decode_stats_readback(
            &raw,
            dst_w * dst_h,
            num_bins,
            hist_range,
            percentiles,
        ))
    }

    /// 1-pass fused GPU DEM Horn 3×3 (`Hillshade` / `Slope` / `Aspect`) + band statistics/histogram/percentiles
    /// (`CoalescedJob::WithStats` — zero `f32` readback, zero re-upload, 12 KB readback).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_dem_stats(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        alg: DemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
        resample: ResampleMode,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Option<GpuBandStats> {
        let gpu_mode = map_resample(resample)?;
        let gpu_alg = match alg {
            DemAlgorithm::Hillshade {
                azimuth_deg,
                altitude_deg,
                z_factor,
            } => GpuDemAlgorithm::Hillshade {
                azimuth_deg,
                altitude_deg,
                z_factor,
            },
            DemAlgorithm::Slope { z_factor } => GpuDemAlgorithm::Slope { z_factor },
            DemAlgorithm::Aspect => GpuDemAlgorithm::Aspect,
            DemAlgorithm::Contours { .. } => return None,
        };
        let (dst_w, dst_h, _) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let min_px = if self.ctx.info.unified_memory {
            512 * 1024
        } else {
            512 * 512
        };
        if dst_w.saturating_mul(dst_h) < min_px {
            return None;
        }
        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let _active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);

        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        let raw = self
            .coalescer
            .submit(CoalescedJob::WithStats {
                inner: Box::new(CoalescedJob::Terrain {
                    slab,
                    plan,
                    mode: gpu_mode,
                    alg: gpu_alg,
                    pixel_size_x_m,
                    pixel_size_y_m,
                    rescale_min: 0.0,
                    rescale_max: 255.0,
                    lut_u32: [0u32; 256],
                    output: CoalescedOutput::RawF32,
                }),
                nodata: None,
                num_bins,
                hist_range,
            })
            .ok()?;
        Some(cogzilla_gpu::decode_stats_readback(
            &raw,
            dst_w * dst_h,
            num_bins,
            hist_range,
            percentiles,
        ))
    }

    /// Attempt GPU-accelerated raw `f32` DEM Horn 3×3 (`Hillshade` / `Slope` / `Aspect`)
    /// evaluation for `.npy`, `.tif`, `.f32`, and mosaic routes.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_dem_f32(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        alg: DemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample(resample)?;
        let gpu_alg = match alg {
            DemAlgorithm::Hillshade {
                azimuth_deg,
                altitude_deg,
                z_factor,
            } => GpuDemAlgorithm::Hillshade {
                azimuth_deg,
                altitude_deg,
                z_factor,
            },
            DemAlgorithm::Slope { z_factor } => GpuDemAlgorithm::Slope { z_factor },
            DemAlgorithm::Aspect => GpuDemAlgorithm::Aspect,
            DemAlgorithm::Contours { .. } => return None,
        };
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src) + coord_upload_bytes;
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::DemHornLanczos3,
            _ => GpuOp::DemHornCubic,
        };
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes: dst_w * dst_h * 4,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }

        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        let raw = self
            .coalescer
            .submit(CoalescedJob::Terrain {
                slab,
                plan,
                mode: gpu_mode,
                alg: gpu_alg,
                pixel_size_x_m,
                pixel_size_y_m,
                rescale_min: 0.0,
                rescale_max: 255.0,
                lut_u32: [0u32; 256],
                output: CoalescedOutput::RawF32,
            })
            .ok()?;
        Some(Bytes::from(raw))
    }

    /// Attempt GPU-accelerated raw `f32` multi-band `ExprProgram` evaluation for `.npy`, `.tif`,
    /// `.f32`, and mosaic routes (reusing `multi_slab_cache`).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_expr_f32(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        prog: &ExprProgram,
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let instrs = Self::compile_expr_to_gpu(prog)?;
        self.try_render_bbox_instrs_f32(cache_key, coords, srcs, &instrs, resample)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_instrs_f32(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        instrs: &[GpuInstr],
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample(resample)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || srcs.is_empty() || srcs.len() > 8 {
            return None;
        }
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let band_upload_bytes: usize = if warm_slab.is_some() {
            0
        } else {
            srcs.iter()
                .map(|s| Self::estimate_upload_bytes(&None, s))
                .sum()
        };
        let upload_bytes = band_upload_bytes + coord_upload_bytes;
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::FusedColormapLanczos3,
            GpuColormapMode::Cubic => GpuOp::DemHornCubic,
            GpuColormapMode::Bilinear | GpuColormapMode::Nearest => GpuOp::FusedColormapBilinear,
        };
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes: dst_w * dst_h * 4,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }
        let (slab, base_col, base_row) = self.resolve_multiband_slab(cache_key, warm_slab, srcs)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        let raw = self
            .coalescer
            .submit(CoalescedJob::Expr {
                slab,
                plan,
                mode: gpu_mode,
                instrs: instrs.to_vec(),
                rescale_min: 0.0,
                rescale_max: 1.0,
                lut_u32: [0u32; 256],
                output: CoalescedOutput::RawF32,
            })
            .ok()?;
        Some(Bytes::from(raw))
    }

    /// Attempt 3D micro-batched GPU single-band rendering of `K >= 2` separable tiles sharing the
    /// same source slab in ONE GPU dispatch and ONE readback, amortizing the dispatch floor
    /// (~1.5 ms Metal / ~50–240 µs Vulkan) across the entire viewport batch.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_single_band_batch(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        tile_size: usize,
        tile_uvs: &[(&[f32], &[f32])],
        src: &BandSource<'_>,
        style: &SingleBandStyle,
    ) -> Option<Vec<Bytes>> {
        let batch_size = tile_uvs.len();
        if batch_size < 2 || tile_size == 0 || !tile_size.is_multiple_of(4) {
            return None;
        }
        let gpu_mode = map_resample(style.resample)?;
        let out_kind = match style.format {
            OutputFormat::Png => GpuPixelOutput::IndexedU8,
            OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
                GpuPixelOutput::Rgba8
            }
            _ => return None,
        };

        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src);
        let per_tile_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => tile_size * tile_size,
            GpuPixelOutput::Rgba8 => tile_size * tile_size * 4,
        };
        let decision = self.policy.decide(&DispatchRequest {
            op: gpu_mode.to_policy_op(),
            dst_pixels: tile_size * tile_size,
            upload_bytes,
            readback_bytes: per_tile_bytes.saturating_mul(batch_size),
            batch_size,
            in_flight: self.in_flight.load(Ordering::Relaxed),
        });
        if !decision.use_gpu {
            return None;
        }

        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let _guard = InFlightGuard(&self.in_flight);

        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let bc = base_col as f32;
        let br = base_row as f32;
        let mut local_u: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        let mut local_v: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        for &(u, v) in tile_uvs {
            if u.len() != tile_size || v.len() != tile_size {
                return None;
            }
            local_u.push(u.iter().map(|&x| x - bc).collect());
            local_v.push(v.iter().map(|&y| y - br).collect());
        }
        let slices: Vec<TileCoordSlice<'_>> = local_u
            .iter()
            .zip(local_v.iter())
            .map(|(u, v)| TileCoordSlice {
                u_coords: u.as_slice(),
                v_coords: v.as_slice(),
            })
            .collect();

        let lut = style.colormap.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
        let raw = self
            .batcher
            .lock()
            .ok()?
            .render_batch(
                &slab,
                &slices,
                gpu_mode,
                style.rescale_min,
                style.rescale_max,
                &lut_u32,
                out_kind,
            )
            .ok()?;

        if raw.len() != per_tile_bytes * batch_size {
            return None;
        }

        let mut out = Vec::with_capacity(batch_size);
        match out_kind {
            GpuPixelOutput::IndexedU8 => {
                for i in 0..batch_size {
                    let chunk = &raw[i * per_tile_bytes..(i + 1) * per_tile_bytes];
                    out.push(encode_indexed_u8_png(chunk, tile_size as u32, tile_size as u32, &lut.rgb));
                }
            }
            GpuPixelOutput::Rgba8 => {
                if style.format == OutputFormat::RawRgba {
                    let shared = Bytes::from(raw);
                    for i in 0..batch_size {
                        out.push(shared.slice(i * per_tile_bytes..(i + 1) * per_tile_bytes));
                    }
                } else {
                    for i in 0..batch_size {
                        let chunk = &raw[i * per_tile_bytes..(i + 1) * per_tile_bytes];
                        out.push(encode_rgba_tile(chunk, tile_size as u32, false, style.format));
                    }
                }
            }
        }
        Some(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_bbox_sar_rgb(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>; 2],
        style: &RgbStyle,
        color_formula: Option<&ColorFormula>,
        active: usize,
    ) -> Option<(CoalescedJob, usize, usize)> {
        let gpu_mode = map_resample(style.resample)?;
        let gpu_formula = Self::translate_color_formula(color_formula)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        if !matches!(
            style.format,
            OutputFormat::Png
                | OutputFormat::Webp { .. }
                | OutputFormat::Jpeg { .. }
                | OutputFormat::RawRgba
        ) {
            return None;
        }
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let band_upload_bytes: usize = if warm_slab.is_some() {
            0
        } else {
            srcs.iter()
                .map(|s| Self::estimate_upload_bytes(&None, s))
                .sum()
        };
        let upload_bytes = band_upload_bytes + coord_upload_bytes;
        let readback_bytes = dst_w * dst_h * 4;
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::FusedColormapLanczos3,
            GpuColormapMode::Cubic => GpuOp::DemHornCubic,
            GpuColormapMode::Bilinear | GpuColormapMode::Nearest => GpuOp::FusedColormapBilinear,
        };
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }
        let (slab, base_col, base_row) = self.resolve_multiband_slab(cache_key, warm_slab, srcs)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        Some((
            CoalescedJob::SarRgb {
                slab,
                plan,
                mode: gpu_mode,
                rescale_min: style.rescale_min,
                rescale_max: style.rescale_max,
                formula: gpu_formula,
            },
            dst_w,
            dst_h,
        ))
    }

    /// Attempt GPU-accelerated 2-band Dual-Pol SAR False-Color RGB (`[VV_dB, VH_dB, VV_dB - VH_dB]`)
    /// tile rendering (non-blocking async coalesced).
    #[allow(clippy::too_many_arguments)]
    pub async fn try_render_sar_rgb<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        bands_1based: [u16; 2],
        plan: &CoordPlan<N>,
        srcs: &[BandSource<'_>; 2],
        style: &RgbStyle,
    ) -> Option<Bytes> {
        let coords = match plan {
            CoordPlan::Separable1D { u, v, .. } => BboxCoordSource::Separable {
                u: u.as_slice(),
                v: v.as_slice(),
            },
            CoordPlan::Grid2D { grid_u, grid_v, .. } => BboxCoordSource::Grid2D {
                dst_w: N,
                dst_h: N,
                grid_u,
                grid_v,
            },
            CoordPlan::Exact2D { u, v, .. } => BboxCoordSource::Exact2D {
                dst_w: N,
                dst_h: N,
                u_map: &u[..],
                v_map: &v[..],
            },
        };
        let cache_key = Self::multiband_key(cog_id, ifd_idx, &bands_1based, srcs);
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let (job, dst_w, dst_h) =
            self.prepare_bbox_sar_rgb(cache_key, coords, srcs, style, None, active)?;
        self.submit_rgba_async(job, dst_w, dst_h, style.format, true)
            .await
    }

    /// Attempt GPU-accelerated Dual-Pol SAR False-Color RGB (with optional fused `ColorFormula`
    /// and `GpuMultiBandCache`) for `/cog/preview`, `/cog/bbox`, `/cog/feature`, and tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_sar_rgb(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>; 2],
        style: &RgbStyle,
        color_formula: Option<&ColorFormula>,
        return_mask: bool,
    ) -> Option<Bytes> {
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let (job, dst_w, dst_h) =
            self.prepare_bbox_sar_rgb(cache_key, coords, srcs, style, color_formula, active)?;
        self.submit_rgba(job, dst_w, dst_h, style.format, return_mask)
    }

    /// Compile 3 `ExprProgram`s (`[R, G, B]`) into a single packed GPU instruction buffer
    /// referencing a unified deduplicated `unique_bands` slice (`<= 8` bands, `<= 256` total ops).
    pub fn compile_expr_rgb_to_gpu(
        progs: &[ExprProgram; 3],
        unique_bands: &[u16],
    ) -> Option<(Vec<GpuInstr>, [u8; 3])> {
        if unique_bands.is_empty() || unique_bands.len() > 8 {
            return None;
        }
        let mut packed = Vec::new();
        let mut lens = [0u8; 3];
        for (c, prog) in progs.iter().enumerate() {
            let mut instrs = Self::compile_expr_to_gpu(prog)?;
            if instrs.len() > 255 {
                return None;
            }
            for ins in &mut instrs {
                if ins.opcode == 1 {
                    let b_1based = *prog.used_bands.get(ins.imm_u32 as usize)?;
                    let unified_slot = unique_bands.iter().position(|&b| b == b_1based)?;
                    ins.imm_u32 = unified_slot as u32;
                }
            }
            lens[c] = instrs.len() as u8;
            packed.extend_from_slice(&instrs);
        }
        if packed.is_empty() || packed.len() > 256 {
            return None;
        }
        Some((packed, lens))
    }

    /// Attempt GPU-accelerated 3-expression `CustomExprRgb` rendering (with optional fused
    /// `ColorFormula` and `GpuMultiBandCache`) for `/cog/preview`, `/cog/bbox`, `/cog/crop`,
    /// and `/cog/feature`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_expr_rgb(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        unique_bands: &[u16],
        srcs: &[BandSource<'_>],
        progs: &[ExprProgram; 3],
        style: &RgbStyle,
        color_formula: Option<&ColorFormula>,
        return_mask: bool,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample(style.resample)?;
        let gpu_formula = Self::translate_color_formula(color_formula)?;
        let (instrs, lens) = Self::compile_expr_rgb_to_gpu(progs, unique_bands)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || srcs.len() != unique_bands.len() {
            return None;
        }
        if !matches!(
            style.format,
            OutputFormat::Png
                | OutputFormat::Webp { .. }
                | OutputFormat::Jpeg { .. }
                | OutputFormat::RawRgba
        ) {
            return None;
        }
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let band_upload_bytes: usize = if warm_slab.is_some() {
            0
        } else {
            srcs.iter()
                .map(|s| Self::estimate_upload_bytes(&None, s))
                .sum()
        };
        let upload_bytes = band_upload_bytes + coord_upload_bytes;
        let readback_bytes = dst_w * dst_h * 4;
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::FusedColormapLanczos3,
            GpuColormapMode::Cubic => GpuOp::DemHornCubic,
            GpuColormapMode::Bilinear | GpuColormapMode::Nearest => GpuOp::FusedColormapBilinear,
        };
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels: dst_w * dst_h,
            upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }
        let (slab, base_col, base_row) = self.resolve_multiband_slab(cache_key, warm_slab, srcs)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        let job = CoalescedJob::ExprRgb {
            slab,
            plan,
            mode: gpu_mode,
            instrs,
            lens,
            rescale_min: style.rescale_min,
            rescale_max: style.rescale_max,
            formula: gpu_formula,
        };
        self.submit_rgba(job, dst_w, dst_h, style.format, return_mask)
    }

    /// Attempt GPU-accelerated multi-band planar `(num_bands, dst_h, dst_w)` Float32 resampling
    /// for `.npy`, Float32 `.tif`, and `.f32` scientific ML tensor cubes (`1..=8` bands).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_multiband_f32(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample(resample)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        let nb = srcs.len();
        if dst_w == 0 || dst_h == 0 || nb == 0 || nb > 8 {
            return None;
        }
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let band_upload_bytes: usize = if warm_slab.is_some() {
            0
        } else {
            srcs.iter()
                .map(|s| Self::estimate_upload_bytes(&None, s))
                .sum()
        };
        let upload_bytes = band_upload_bytes + coord_upload_bytes;
        let readback_bytes = dst_w * dst_h * nb * 4;
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let decision = self.policy.decide(&DispatchRequest {
            op: gpu_mode.to_policy_op(),
            dst_pixels: dst_w * dst_h * nb,
            upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }
        let (slab, base_col, base_row) = self.resolve_multiband_slab(cache_key, warm_slab, srcs)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        let raw = self
            .coalescer
            .submit(CoalescedJob::MultiBandF32 {
                slab,
                plan,
                mode: gpu_mode,
            })
            .ok()?;
        Some(Bytes::from(raw))
    }

    /// Attempt GPU-accelerated planar `(3, dst_h, dst_w)` Float32 evaluation for 3-expression
    /// `CustomExprRgb` `.npy` / Float32 `.tif` / `.f32` exports.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_bbox_expr_rgb_f32(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        unique_bands: &[u16],
        srcs: &[BandSource<'_>],
        progs: &[ExprProgram; 3],
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample(resample)?;
        let (instrs, lens) = Self::compile_expr_rgb_to_gpu(progs, unique_bands)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || srcs.len() != unique_bands.len() {
            return None;
        }
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let band_upload_bytes: usize = if warm_slab.is_some() {
            0
        } else {
            srcs.iter()
                .map(|s| Self::estimate_upload_bytes(&None, s))
                .sum()
        };
        let upload_bytes = band_upload_bytes + coord_upload_bytes;
        let readback_bytes = dst_w * dst_h * 3 * 4;
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::FusedColormapLanczos3,
            GpuColormapMode::Cubic => GpuOp::DemHornCubic,
            GpuColormapMode::Bilinear | GpuColormapMode::Nearest => GpuOp::FusedColormapBilinear,
        };
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _guard = InFlightGuard(&self.in_flight);
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels: dst_w * dst_h * 3,
            upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }
        let (slab, base_col, base_row) = self.resolve_multiband_slab(cache_key, warm_slab, srcs)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        let raw = self
            .coalescer
            .submit(CoalescedJob::ExprRgbF32 {
                slab,
                plan,
                mode: gpu_mode,
                instrs,
                lens,
            })
            .ok()?;
        Some(Bytes::from(raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_codec::{BandData, BlockStats, DecodedBand};
    use cogzilla_kernel::{ColormapName, sample_cubic};

    #[test]
    fn multiblock_u16_stitch_and_crop_matches_cpu() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU multi-block test (no adapter)");
            return;
        };

        // 2x2 grid of 256x256 U16 blocks = 512x512 virtual raster
        let tw = 256u32;
        let th = 256u32;
        let mut bands = Vec::new();
        for ty in 0..2u32 {
            for tx in 0..2u32 {
                let mut data = vec![0u16; (tw * th) as usize];
                for y in 0..th {
                    for x in 0..tw {
                        let gx = tx * tw + x;
                        let gy = ty * th + y;
                        data[(y * tw + x) as usize] = (500 + (gx * 2 + gy * 3) % 2000) as u16;
                    }
                }
                bands.push(DecodedBand {
                    data: BandData::U16 { data: data.into_boxed_slice(), nodata: None },
                    width: tw,
                    height: th,
                    stats: BlockStats {
                        all_nodata: false,
                        no_nodata: true,
                        min: 500.0,
                        max: 2500.0,
                    },
                });
            }
        }
        let refs: Vec<Option<&DecodedBand>> = bands.iter().map(Some).collect();
        let src = BandSource::new(512, 512, tw, th, 0, 0, 2, 2, &refs);

        // Force GPU via 1024x1024 Lanczos3 crop spanning all 4 blocks
        let n = 1024usize;
        let u_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (504.0 / n as f32)).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (504.0 / n as f32)).collect();
        let style = SingleBandStyle {
            rescale_min: 500.0,
            rescale_max: 2500.0,
            colormap: ColormapName::Viridis,
            resample: ResampleMode::Cubic,
            format: OutputFormat::RawRgba,
        };

        // Verify stitch matches BandSource::sample_inbounds at block boundaries
        let (stitched, bc, br, sw, sh) = stitch_band_source(&src).unwrap();
        assert_eq!((bc, br, sw, sh), (0, 0, 512, 512));
        for &(x, y) in &[(0u32, 0u32), (255, 255), (256, 256), (511, 511)] {
            assert_eq!(stitched[(y * sw + x) as usize], src.sample_inbounds(x, y));
        }

        // Verify GPU RawRgba crop output against CPU sample_cubic on a 1024x1024 crop
        if let Some(rgba) =
            gpu.try_render_crop_single_band(99, 0, 0, &u_coords, &v_coords, &src, &style)
        {
            assert_eq!(rgba.len(), n * n * 4);
            let lut = ColormapName::Viridis.lut();
            let inv255 = 255.0 / 2000.0;
            let mut max_diff = 0i32;
            for py in (0..n).step_by(17) {
                for px in (0..n).step_by(17) {
                    let v = sample_cubic(&src, u_coords[px], v_coords[py]);
                    let ci = ((v - 500.0) * inv255).clamp(0.0, 255.0) as usize;
                    let off = (py * n + px) * 4;
                    assert_eq!(rgba[off + 3], 255);
                    for ch in 0..3 {
                        max_diff = max_diff.max((lut.rgb[ci][ch] as i32 - rgba[off + ch] as i32).abs());
                    }
                }
            }
            assert!(max_diff <= 4, "multi-block U16 crop max channel diff {max_diff} > 4");
        }
    }

    #[test]
    fn gpu_expr_vm_translator_and_render_match_cpu() {
        let prog = ExprProgram::compile("db(b1)-db(b2)+sqrt(b3)").unwrap();
        let instrs = GpuBackend::compile_expr_to_gpu(&prog).expect("should translate to GPU VM");
        assert_eq!(instrs.len(), prog.ops.len());

        // Unsupported op (median) must cleanly return None for CPU fallback
        let med = ExprProgram::compile("median(b1, b2, b3)").unwrap();
        assert!(GpuBackend::compile_expr_to_gpu(&med).is_none());
    }

    #[test]
    fn bbox_exact2d_masked_single_and_dem_match_cpu() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU bbox Exact2D test (no adapter)");
            return;
        };

        let w = 256u32;
        let h = 256u32;
        let mut vals = vec![0u16; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                vals[(y * w + x) as usize] = (100.0
                    + 50.0 * (x as f32 * 0.04).sin()
                    + 40.0 * (y as f32 * 0.05).cos()) as u16;
            }
        }
        let b0 = DecodedBand {
            width: w,
            height: h,
            data: BandData::U16 {
                data: vals.into_boxed_slice(),
                nodata: None,
            },
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 0.0,
                max: 200.0,
            },
        };
        let blocks = [Some(&b0)];
        let src = BandSource::new(w, h, w, h, 0, 0, 1, 1, &blocks);

        // Non-square 640x480 curved Exact2D map with a circular polygon mask (outside = NaN)
        let dst_w = 640usize;
        let dst_h = 480usize;
        let mut u_map = vec![f32::NAN; dst_w * dst_h];
        let mut v_map = vec![f32::NAN; dst_w * dst_h];
        let cx = dst_w as f32 * 0.5;
        let cy = dst_h as f32 * 0.5;
        for py in 0..dst_h {
            for px in 0..dst_w {
                let dx = (px as f32 - cx) / cx;
                let dy = (py as f32 - cy) / cy;
                if dx * dx + dy * dy <= 0.85 {
                    let u = 8.0 + (px as f32 / dst_w as f32) * 238.0 + 3.0 * dy.sin();
                    let v = 8.0 + (py as f32 / dst_h as f32) * 238.0 + 3.0 * dx.cos();
                    u_map[py * dst_w + px] = u;
                    v_map[py * dst_w + px] = v;
                }
            }
        }

        let style = SingleBandStyle {
            rescale_min: 0.0,
            rescale_max: 200.0,
            colormap: ColormapName::Viridis,
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
        };
        if let Some(rgba) = gpu.try_render_bbox_single_band(
            777,
            0,
            0,
            BboxCoordSource::Exact2D {
                dst_w,
                dst_h,
                u_map: &u_map,
                v_map: &v_map,
            },
            &src,
            &style,
            true,
        ) {
            assert_eq!(rgba.len(), dst_w * dst_h * 4);
            let lut = ColormapName::Viridis.lut();
            let inv255 = 255.0 / 200.0;
            let mut checked_inside = 0usize;
            let mut checked_outside = 0usize;
            let mut max_diff = 0i32;
            for py in (0..dst_h).step_by(11) {
                for px in (0..dst_w).step_by(11) {
                    let i = py * dst_w + px;
                    let off = i * 4;
                    if u_map[i].is_nan() {
                        assert_eq!(rgba[off + 3], 0, "masked outside pixel must be transparent");
                        checked_outside += 1;
                    } else {
                        assert_eq!(rgba[off + 3], 255);
                        let v = cogzilla_kernel::sample_general_scaled(
                            &src,
                            u_map[i],
                            v_map[i],
                            1.0,
                            1.0,
                            ResampleMode::Lanczos,
                        );
                        let ci = (v * inv255).clamp(0.0, 255.0) as usize;
                        for ch in 0..3 {
                            max_diff = max_diff.max((lut.rgb[ci][ch] as i32 - rgba[off + ch] as i32).abs());
                        }
                        checked_inside += 1;
                    }
                }
            }
            assert!(checked_inside > 100 && checked_outside > 50);
            assert!(max_diff <= 4, "Exact2D masked bbox max channel diff {max_diff} > 4");
        }
    }

    #[test]
    fn rgb_color_formula_and_multiband_cache_match_cpu() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU RGB color_formula test (no adapter)");
            return;
        };
        let cf = ColorFormula::parse("Gamma RGB 1.5 Sigmoidal RGB 15 0.35 Saturation 1.2").unwrap();
        assert!(cf.as_lut_then_saturation().is_some());
        // Multi-stage post-saturation LUT must fall back cleanly to CPU
        let cf_complex = ColorFormula::parse("Saturation 1.2 Gamma RGB 1.5").unwrap();
        assert!(cf_complex.as_lut_then_saturation().is_none());

        let w = 256u32;
        let h = 256u32;
        let mk_band = |offset: f32| -> DecodedBand {
            let mut vals = vec![0u16; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    vals[(y * w + x) as usize] = (offset
                        + 400.0 * (x as f32 * 0.03).sin()
                        + 300.0 * (y as f32 * 0.04).cos())
                        .max(0.0) as u16;
                }
            }
            DecodedBand {
                width: w,
                height: h,
                data: BandData::U16 {
                    data: vals.into_boxed_slice(),
                    nodata: None,
                },
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 0.0,
                    max: 2000.0,
                },
            }
        };
        let b0 = mk_band(900.0);
        let b1 = mk_band(700.0);
        let b2 = mk_band(500.0);
        let bl0 = [Some(&b0)];
        let bl1 = [Some(&b1)];
        let bl2 = [Some(&b2)];
        let srcs = [
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl0),
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl1),
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl2),
        ];

        let n = 512usize;
        let u_coords: Vec<f32> = (0..n).map(|i| 4.0 + (i as f32 / n as f32) * 246.0).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| 4.0 + (i as f32 / n as f32) * 246.0).collect();
        let style = RgbStyle {
            rescale_min: [0.0; 3],
            rescale_max: [2000.0; 3],
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
        };
        let key = GpuBackend::multiband_key(999, 0, &[1, 2, 3], &srcs);
        assert!(key.is_some());

        // First call populates multi_slab_cache; second call hits VRAM cache with 0 upload
        let first = gpu.try_render_bbox_rgb(
            key,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &srcs,
            &style,
            Some(&cf),
            true,
        );
        if let Some(rgba1) = first {
            let st1 = gpu.multi_slab_cache.stats();
            assert_eq!(st1.entries, 1);
            let rgba2 = gpu
                .try_render_bbox_rgb(
                    key,
                    BboxCoordSource::Separable {
                        u: &u_coords,
                        v: &v_coords,
                    },
                    &srcs,
                    &style,
                    Some(&cf),
                    true,
                )
                .expect("warm cached RGB render should use GPU");
            assert_eq!(rgba1, rgba2, "cached multi-band slab output must be bit-identical");
            assert!(gpu.multi_slab_cache.stats().hits >= 1);

            let inv = 255.0 / 2000.0;
            let mut max_diff = 0i32;
            for py in (0..n).step_by(19) {
                for px in (0..n).step_by(19) {
                    let sr = cogzilla_kernel::sample_general_scaled(
                        &srcs[0], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let sg = cogzilla_kernel::sample_general_scaled(
                        &srcs[1], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let sb = cogzilla_kernel::sample_general_scaled(
                        &srcs[2], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let (cr, cg, cb) = cf.apply_rgb(
                        (sr * inv).clamp(0.0, 255.0) as u8,
                        (sg * inv).clamp(0.0, 255.0) as u8,
                        (sb * inv).clamp(0.0, 255.0) as u8,
                    );
                    let off = (py * n + px) * 4;
                    assert_eq!(rgba1[off + 3], 255);
                    max_diff = max_diff
                        .max((cr as i32 - rgba1[off] as i32).abs())
                        .max((cg as i32 - rgba1[off + 1] as i32).abs())
                        .max((cb as i32 - rgba1[off + 2] as i32).abs());
                }
            }
            assert!(max_diff <= 4, "RGB ColorFormula max channel diff {max_diff} > 4");
        }
    }

    #[test]
    fn mosaic_median_and_styled_match_cpu() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU mosaic test (no adapter)");
            return;
        };

        let w = 512u32;
        let h = 512u32;
        let n_px = (w * h) as usize;
        let n_scenes = 12usize;
        let mut scenes: Vec<Vec<f32>> = Vec::with_capacity(n_scenes);
        for s in 0..n_scenes {
            let mut plane = vec![f32::NAN; n_px];
            for i in 0..n_px {
                // Sprinkle 15% nodata holes across scenes
                if (i + s * 37) % 7 != 0 {
                    plane[i] = 100.0
                        + 50.0 * ((i as f32) * 0.001 + (s as f32) * 0.7).sin()
                        + (s as f32) * 8.0;
                }
            }
            scenes.push(plane);
        }
        let refs: Vec<&[f32]> = scenes.iter().map(Vec::as_slice).collect();

        if let Some(gpu_med) = gpu.try_composite_mosaic_f32(w, h, &refs, GpuPixelSelection::Median)
        {
            assert_eq!(gpu_med.len(), n_px);
            let mut scratch = Vec::with_capacity(n_scenes);
            let mut max_err = 0.0f32;
            for i in (0..n_px).step_by(31) {
                scratch.clear();
                for s in &scenes {
                    if s[i].is_finite() {
                        scratch.push(s[i]);
                    }
                }
                if scratch.is_empty() {
                    assert!(gpu_med[i].is_nan());
                } else {
                    scratch.sort_unstable_by(|a, b| a.total_cmp(b));
                    let k = scratch.len();
                    let cpu_v = if k % 2 == 1 {
                        scratch[k / 2]
                    } else {
                        0.5 * (scratch[k / 2 - 1] + scratch[k / 2])
                    };
                    max_err = max_err.max((cpu_v - gpu_med[i]).abs());
                }
            }
            assert!(max_err < 1e-4, "GPU mosaic median max err {max_err} >= 1e-4");
        }

        if let Some(png) = gpu.try_composite_mosaic_styled(
            w,
            h,
            &refs,
            GpuPixelSelection::Median,
            (50.0, 250.0),
            ColormapName::Viridis,
            OutputFormat::Png,
            true,
        ) {
            assert!(png.starts_with(b"\x89PNG"), "fused mosaic styled output must be valid PNG");
        }

        // 1-pass fused Mosaic -> GPU JPEG
        if let Some(jpg) = gpu.try_composite_mosaic_styled(
            w,
            h,
            &refs,
            GpuPixelSelection::Median,
            (50.0, 250.0),
            ColormapName::Viridis,
            OutputFormat::Jpeg { quality: 85 },
            false,
        ) {
            assert!(
                jpg.starts_with(&[0xff, 0xd8]) && jpg.ends_with(&[0xff, 0xd9]),
                "fused mosaic styled JPEG must be valid JFIF"
            );
        }

        // 1-submit batched 3-band RGB median must be bit-exact vs per-band GPU median
        let band_planes: Vec<Vec<&[f32]>> = vec![refs.clone(), refs.clone(), refs.clone()];
        if let (Some(single), Some(batched)) = (
            gpu.try_composite_mosaic_f32(w, h, &refs, GpuPixelSelection::Median),
            gpu.try_composite_mosaic_f32_bands(w, h, &band_planes, GpuPixelSelection::Median),
        ) {
            assert_eq!(batched.len(), 3);
            for (b, got) in batched.iter().enumerate() {
                assert_eq!(got.len(), single.len());
                for i in 0..n_px {
                    assert_eq!(
                        got[i].to_bits(),
                        single[i].to_bits(),
                        "batched band {b} px {i} mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn raw_f32_single_dem_and_expr_match_cpu() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU RawF32 test (no adapter)");
            return;
        };

        let w = 256u32;
        let h = 256u32;
        let mk_band = |base: f32, amp: f32| -> DecodedBand {
            let mut vals = vec![0u16; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    vals[(y * w + x) as usize] = (base
                        + amp * (x as f32 * 0.04).sin()
                        + (amp * 0.7) * (y as f32 * 0.05).cos())
                        .max(1.0) as u16;
                }
            }
            DecodedBand {
                width: w,
                height: h,
                data: BandData::U16 {
                    data: vals.into_boxed_slice(),
                    nodata: None,
                },
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 1.0,
                    max: 2500.0,
                },
            }
        };
        let b0 = mk_band(1200.0, 400.0);
        let b1 = mk_band(600.0, 200.0);
        let bl0 = [Some(&b0)];
        let bl1 = [Some(&b1)];
        let s0 = BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl0);
        let s1 = BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl1);

        let n = 512usize;
        let u_coords: Vec<f32> = (0..n).map(|i| 4.0 + (i as f32 / n as f32) * 246.0).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| 4.0 + (i as f32 / n as f32) * 246.0).collect();
        let coords = BboxCoordSource::Separable {
            u: &u_coords,
            v: &v_coords,
        };

        // 1. Single-band RawF32 Lanczos
        if let Some(raw) =
            gpu.try_render_bbox_single_band_f32(500, 0, 0, coords, &s0, ResampleMode::Lanczos)
        {
            assert_eq!(raw.len(), n * n * 4);
            let mut max_err = 0.0f32;
            for py in (4..n - 4).step_by(23) {
                for px in (4..n - 4).step_by(23) {
                    let off = (py * n + px) * 4;
                    let gv = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                    let cv = cogzilla_kernel::sample_general_scaled(
                        &s0, u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    max_err = max_err.max((gv - cv).abs());
                }
            }
            assert!(max_err < 1e-2, "Single RawF32 max err {max_err} >= 1e-2");
        }

        // 2. DEM Slope RawF32 Lanczos
        if let Some(raw) = gpu.try_render_bbox_dem_f32(
            500,
            0,
            0,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &s0,
            DemAlgorithm::Slope { z_factor: 1.0 },
            30.0,
            30.0,
            ResampleMode::Lanczos,
        ) {
            assert_eq!(raw.len(), n * n * 4);
            // Spot-check interior slope values are finite and in [0, 90] degrees
            let off = (256 * n + 256) * 4;
            let sv = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            assert!(sv.is_finite() && (0.0..=90.0).contains(&sv), "unexpected slope {sv}");
        }

        // 3. Expr NDVI RawF32 Lanczos
        let ndvi = ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap();
        let srcs = [s0, s1];
        let key = GpuBackend::multiband_key(500, 0, &ndvi.used_bands, &srcs);
        if let Some(raw) = gpu.try_render_bbox_expr_f32(
            key,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &srcs,
            &ndvi,
            ResampleMode::Lanczos,
        ) {
            assert_eq!(raw.len(), n * n * 4);
            let mut max_err = 0.0f32;
            for py in (4..n - 4).step_by(23) {
                for px in (4..n - 4).step_by(23) {
                    let off = (py * n + px) * 4;
                    let gv = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                    let c0 = cogzilla_kernel::sample_general_scaled(
                        &srcs[0], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let c1 = cogzilla_kernel::sample_general_scaled(
                        &srcs[1], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let cv = (c0 - c1) / (c0 + c1);
                    max_err = max_err.max((gv - cv).abs());
                }
            }
            assert!(max_err < 1e-3, "Expr NDVI RawF32 max err {max_err} >= 1e-3");
        }
    }

    #[test]
    fn spectral_indices_match_cpu() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU spectral test (no adapter)");
            return;
        };
        assert!(GpuBackend::compile_spectral_to_gpu(SpectralExpr::SarRgb).is_none());

        let w = 256u32;
        let h = 256u32;
        let mk_band = |base: f32, amp: f32| -> DecodedBand {
            let mut vals = vec![0u16; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    vals[(y * w + x) as usize] = (base
                        + amp * (x as f32 * 0.03).sin()
                        + (amp * 0.6) * (y as f32 * 0.04).cos())
                        .max(10.0) as u16;
                }
            }
            DecodedBand {
                width: w,
                height: h,
                data: BandData::U16 {
                    data: vals.into_boxed_slice(),
                    nodata: None,
                },
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 10.0,
                    max: 2500.0,
                },
            }
        };
        let b0 = mk_band(1500.0, 400.0);
        let b1 = mk_band(700.0, 200.0);
        let b2 = mk_band(400.0, 120.0);
        let bl0 = [Some(&b0)];
        let bl1 = [Some(&b1)];
        let bl2 = [Some(&b2)];
        let all_srcs = [
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl0),
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl1),
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl2),
        ];

        let n = 512usize;
        let u_coords: Vec<f32> = (0..n).map(|i| 4.0 + (i as f32 / n as f32) * 246.0).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| 4.0 + (i as f32 / n as f32) * 246.0).collect();

        // Check Db (1-band), NormalizedDiff / Savi / Ratio / Diff (2-band), and Evi (3-band)
        for expr in [
            SpectralExpr::Db,
            SpectralExpr::NormalizedDiff,
            SpectralExpr::Savi,
            SpectralExpr::Ratio,
            SpectralExpr::Diff,
            SpectralExpr::Evi,
        ] {
            let srcs = &all_srcs[..expr.required_bands()];
            let bands_1based: Vec<u16> = (1..=srcs.len() as u16).collect();
            let key = GpuBackend::multiband_key(888, 0, &bands_1based, srcs);
            if let Some(raw) = gpu.try_render_bbox_spectral_f32(
                key,
                BboxCoordSource::Separable {
                    u: &u_coords,
                    v: &v_coords,
                },
                srcs,
                expr,
                ResampleMode::Lanczos,
            ) {
                assert_eq!(raw.len(), n * n * 4);
                let mut max_err = 0.0f32;
                for py in (8..n - 8).step_by(29) {
                    for px in (8..n - 8).step_by(29) {
                        let off = (py * n + px) * 4;
                        let gv = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                        let c0 = cogzilla_kernel::sample_general_scaled(
                            &all_srcs[0], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                        );
                        let cv = match expr.required_bands() {
                            1 => expr.eval1(c0),
                            2 => {
                                let c1 = cogzilla_kernel::sample_general_scaled(
                                    &all_srcs[1], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                                );
                                expr.eval2(c0, c1)
                            }
                            _ => {
                                let c1 = cogzilla_kernel::sample_general_scaled(
                                    &all_srcs[1], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                                );
                                let c2 = cogzilla_kernel::sample_general_scaled(
                                    &all_srcs[2], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                                );
                                expr.eval3(c0, c1, c2)
                            }
                        };
                        if cv.is_finite() {
                            max_err = max_err.max((gv - cv).abs());
                        }
                    }
                }
                let tol = match expr {
                    // CPU `fast_db_from_power` uses a fast ~0.01 dB log2 polynomial vs GPU IEEE log10
                    SpectralExpr::Db | SpectralExpr::Diff => 2e-2,
                    _ => 5e-3,
                };
                assert!(
                    max_err < tol,
                    "SpectralExpr::{expr:?} max err {max_err} >= {tol}"
                );
            }
        }
    }

    #[test]
    fn batched_single_band_tiles_match_single_render() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU batch test (no adapter)");
            return;
        };

        let w = 512u32;
        let h = 512u32;
        let mut vals = vec![0u16; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                vals[(y * w + x) as usize] = (800.0
                    + 500.0 * (x as f32 * 0.02).sin()
                    + 350.0 * (y as f32 * 0.03).cos()) as u16;
            }
        }
        let b0 = DecodedBand {
            width: w,
            height: h,
            data: BandData::U16 {
                data: vals.into_boxed_slice(),
                nodata: None,
            },
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 0.0,
                max: 2000.0,
            },
        };
        let blocks = [Some(&b0)];
        let src = BandSource::new(w, h, w, h, 0, 0, 1, 1, &blocks);

        // 16 tiles (4x4 grid of 256x256 tiles) sampling sub-windows of the 512x512 slab
        let n = 256usize;
        let mut u_vecs = Vec::with_capacity(16);
        let mut v_vecs = Vec::with_capacity(16);
        for ty in 0..4 {
            for tx in 0..4 {
                let u0 = 8.0 + (tx as f32) * 120.0;
                let v0 = 8.0 + (ty as f32) * 120.0;
                u_vecs.push((0..n).map(|i| u0 + (i as f32) * (116.0 / n as f32)).collect::<Vec<_>>());
                v_vecs.push((0..n).map(|i| v0 + (i as f32) * (116.0 / n as f32)).collect::<Vec<_>>());
            }
        }
        let uvs: Vec<(&[f32], &[f32])> = u_vecs
            .iter()
            .zip(v_vecs.iter())
            .map(|(u, v)| (u.as_slice(), v.as_slice()))
            .collect();

        let style = SingleBandStyle {
            rescale_min: 0.0,
            rescale_max: 2000.0,
            colormap: ColormapName::Viridis,
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
        };
        if let Some(batch) = gpu.try_render_single_band_batch(4242, 0, 0, n, &uvs, &src, &style) {
            assert_eq!(batch.len(), 16);
            // Every tile in the 3D batch must match single-tile GPU render bit-identically
            for idx in [0usize, 5, 10, 15] {
                let single = gpu
                    .try_render_crop_single_band(4242, 0, 0, &u_vecs[idx], &v_vecs[idx], &src, &style)
                    .expect("warm single tile should render on GPU");
                assert_eq!(batch[idx], single, "batched tile {idx} differed from single render");
            }
        }
    }

    #[test]
    fn sar_rgb_and_color_formula_match_cpu() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU SAR RGB test (no adapter)");
            return;
        };

        let w = 256u32;
        let h = 256u32;
        let mk_band = |base: f32, amp: f32| -> DecodedBand {
            let mut vals = vec![0u16; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    vals[(y * w + x) as usize] = (base
                        + amp * (x as f32 * 0.03).sin()
                        + (amp * 0.5) * (y as f32 * 0.04).cos())
                        .max(5.0) as u16;
                }
            }
            DecodedBand {
                width: w,
                height: h,
                data: BandData::U16 {
                    data: vals.into_boxed_slice(),
                    nodata: None,
                },
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 5.0,
                    max: 2500.0,
                },
            }
        };
        let b0 = mk_band(1000.0, 400.0);
        let b1 = mk_band(250.0, 120.0);
        let bl0 = [Some(&b0)];
        let bl1 = [Some(&b1)];
        let srcs = [
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl0),
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl1),
        ];

        let n = 512usize;
        let u_coords: Vec<f32> = (0..n).map(|i| 4.0 + (i as f32 / n as f32) * 246.0).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| 4.0 + (i as f32 / n as f32) * 246.0).collect();
        let style = RgbStyle {
            rescale_min: [15.0, 10.0, 0.0],
            rescale_max: [35.0, 30.0, 15.0],
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
        };
        let cf = ColorFormula::parse("Gamma RGB 1.3 Saturation 1.15").unwrap();
        let key = GpuBackend::multiband_key(5555, 0, &[1, 2], &srcs);

        if let Some(rgba) = gpu.try_render_bbox_sar_rgb(
            key,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &srcs,
            &style,
            Some(&cf),
            true,
        ) {
            assert_eq!(rgba.len(), n * n * 4);
            let inv = [
                255.0 / (style.rescale_max[0] - style.rescale_min[0]),
                255.0 / (style.rescale_max[1] - style.rescale_min[1]),
                255.0 / (style.rescale_max[2] - style.rescale_min[2]),
            ];
            let mut max_diff = 0i32;
            for py in (8..n - 8).step_by(23) {
                for px in (8..n - 8).step_by(23) {
                    let p0 = cogzilla_kernel::sample_general_scaled(
                        &srcs[0], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let p1 = cogzilla_kernel::sample_general_scaled(
                        &srcs[1], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let vv = 10.0 * p0.log10();
                    let vh = 10.0 * p1.log10();
                    let (cr, cg, cb) = cf.apply_rgb(
                        ((vv - style.rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8,
                        ((vh - style.rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8,
                        (((vv - vh) - style.rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8,
                    );
                    let off = (py * n + px) * 4;
                    assert_eq!(rgba[off + 3], 255);
                    max_diff = max_diff
                        .max((cr as i32 - rgba[off] as i32).abs())
                        .max((cg as i32 - rgba[off + 1] as i32).abs())
                        .max((cb as i32 - rgba[off + 2] as i32).abs());
                }
            }
            assert!(max_diff <= 4, "SAR RGB + ColorFormula max channel diff {max_diff} > 4");
        }
    }

    #[test]
    fn color_formula_and_custom_lut_folding_match_cpu() {
        let cf = ColorFormula::parse("Gamma RGB 1.6 Sigmoidal RGB 6 0.5 Saturation 1.4")
            .expect("valid color_formula");
        // Verify direct_lookup or transparent custom LUT safely declines GPU
        let mut entries = [[100u8, 120, 140, 255]; 256];
        for (i, e) in entries.iter_mut().enumerate() {
            *e = [i as u8, (255 - i) as u8, ((i * 3) & 255) as u8, 255];
        }
        let direct_cmap = CustomColormap::from_rgba_palette(&entries);
        assert!(
            GpuBackend::resolve_styled_rgb_lut(ColormapName::Viridis, Some(&direct_cmap), None)
                .is_none(),
            "direct_lookup custom colormap must fall back to CPU"
        );
        let opaque_cmap = CustomColormap::Discrete {
            lut: Box::new(cogzilla_kernel::ColormapLut {
                rgb: std::array::from_fn(|i| [entries[i][0], entries[i][1], entries[i][2]]),
                rgba_u32: std::array::from_fn(|i| u32::from_le_bytes(entries[i])),
            }),
            extra: std::collections::HashMap::new(),
            direct_lookup: false,
        };
        let folded = GpuBackend::resolve_styled_rgb_lut(
            ColormapName::Viridis,
            Some(&opaque_cmap),
            Some(&cf),
        )
        .expect("opaque rescaled custom colormap + ColorFormula should fold");

        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU LUT formula test (no adapter)");
            return;
        };
        let w = 256u32;
        let h = 256u32;
        let data: Vec<u16> = (0..(w * h))
            .map(|i| 100 + ((i % w) * 3 + (i / w) * 2) as u16)
            .collect();
        let band = DecodedBand {
            width: w,
            height: h,
            data: BandData::U16 {
                data: data.into_boxed_slice(),
                nodata: None,
            },
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 100.0,
                max: 1400.0,
            },
        };
        let refs = [Some(&band)];
        let src = BandSource::new(w, h, w, h, 0, 0, 1, 1, &refs);
        let n = 512usize;
        let u_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (246.0 / n as f32)).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (246.0 / n as f32)).collect();
        let style = SingleBandStyle {
            rescale_min: 100.0,
            rescale_max: 1400.0,
            colormap: ColormapName::Viridis,
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
        };
        if let Some(rgba) = gpu.try_render_bbox_single_band_with_lut(
            9090,
            0,
            0,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &src,
            &style,
            &folded,
            true,
        ) {
            assert_eq!(rgba.len(), n * n * 4);
            let inv255 = 255.0 / 1300.0;
            let mut max_diff = 0i32;
            for py in (8..n - 8).step_by(29) {
                for px in (8..n - 8).step_by(29) {
                    let v = cogzilla_kernel::sample_general_scaled(
                        &src,
                        u_coords[px],
                        v_coords[py],
                        1.0,
                        1.0,
                        ResampleMode::Lanczos,
                    );
                    let ci = ((v - 100.0) * inv255).clamp(0.0, 255.0) as usize;
                    let (cr, cg, cb) = cf.apply_rgb(entries[ci][0], entries[ci][1], entries[ci][2]);
                    let off = (py * n + px) * 4;
                    assert_eq!(rgba[off + 3], 255);
                    max_diff = max_diff
                        .max((cr as i32 - rgba[off] as i32).abs())
                        .max((cg as i32 - rgba[off + 1] as i32).abs())
                        .max((cb as i32 - rgba[off + 2] as i32).abs());
                }
            }
            assert!(
                max_diff <= 4,
                "LUT-folded CustomColormap + ColorFormula max channel diff {max_diff} > 4"
            );
        }
    }

    #[test]
    fn custom_expr_rgb_and_color_formula_match_cpu() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU CustomExprRgb test (no adapter)");
            return;
        };
        let w = 256u32;
        let h = 256u32;
        let mk_band = |base: u16, sx: u16, sy: u16| {
            let vals: Vec<u16> = (0..(w * h))
                .map(|i| base + ((i % w) as u16 * sx) + ((i / w) as u16 * sy))
                .collect();
            DecodedBand {
                width: w,
                height: h,
                data: BandData::U16 {
                    data: vals.into_boxed_slice(),
                    nodata: None,
                },
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 10.0,
                    max: 2500.0,
                },
            }
        };
        let b0 = mk_band(400, 3, 2);
        let b1 = mk_band(200, 2, 3);
        let b2 = mk_band(100, 1, 2);
        let bl0 = [Some(&b0)];
        let bl1 = [Some(&b1)];
        let bl2 = [Some(&b2)];
        let srcs = [
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl0),
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl1),
            BandSource::new(w, h, w, h, 0, 0, 1, 1, &bl2),
        ];

        let progs = [
            ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap(),
            ExprProgram::compile("b2*0.001").unwrap(),
            ExprProgram::compile("sqrt(b3)*0.02").unwrap(),
        ];
        let unique_bands = [1u16, 2, 3];
        let cf = ColorFormula::parse("Gamma RGB 1.5 Sigmoidal RGB 6 0.5 Saturation 1.3").unwrap();
        let style = RgbStyle {
            rescale_min: [-1.0, 0.0, 0.0],
            rescale_max: [1.0, 1.5, 1.0],
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
        };
        let n = 512usize;
        let u_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (246.0 / n as f32)).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (246.0 / n as f32)).collect();

        if let Some(rgba) = gpu.try_render_bbox_expr_rgb(
            None,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &unique_bands,
            &srcs,
            &progs,
            &style,
            Some(&cf),
            true,
        ) {
            assert_eq!(rgba.len(), n * n * 4);
            let inv = [
                255.0 / (style.rescale_max[0] - style.rescale_min[0]),
                255.0 / (style.rescale_max[1] - style.rescale_min[1]),
                255.0 / (style.rescale_max[2] - style.rescale_min[2]),
            ];
            let mut max_diff = 0i32;
            for py in (8..n - 8).step_by(29) {
                for px in (8..n - 8).step_by(29) {
                    let s0 = cogzilla_kernel::sample_general_scaled(
                        &srcs[0], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let s1 = cogzilla_kernel::sample_general_scaled(
                        &srcs[1], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let s2 = cogzilla_kernel::sample_general_scaled(
                        &srcs[2], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let vr = (s0 - s1) / (s0 + s1);
                    let vg = s1 * 0.001;
                    let vb = s2.sqrt() * 0.02;
                    let (cr, cg, cb) = cf.apply_rgb(
                        ((vr - style.rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8,
                        ((vg - style.rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8,
                        ((vb - style.rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8,
                    );
                    let off = (py * n + px) * 4;
                    assert_eq!(rgba[off + 3], 255);
                    max_diff = max_diff
                        .max((cr as i32 - rgba[off] as i32).abs())
                        .max((cg as i32 - rgba[off + 1] as i32).abs())
                        .max((cb as i32 - rgba[off + 2] as i32).abs());
                }
            }
            assert!(
                max_diff <= 4,
                "CustomExprRgb + ColorFormula max channel diff {max_diff} > 4"
            );
        }
    }

    #[test]
    fn multiband_and_expr_rgb_raw_f32_match_cpu() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU MultiBand/ExprRgb RawF32 test (no adapter)");
            return;
        };
        let w = 256u32;
        let h = 256u32;
        let mk_band = |base: u16, sx: u16, sy: u16| {
            let vals: Vec<u16> = (0..(w * h))
                .map(|i| base + ((i % w) as u16 * sx) + ((i / w) as u16 * sy))
                .collect();
            DecodedBand {
                width: w,
                height: h,
                data: BandData::U16 {
                    data: vals.into_boxed_slice(),
                    nodata: None,
                },
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 10.0,
                    max: 2500.0,
                },
            }
        };
        let b0 = mk_band(400, 3, 2);
        let b1 = mk_band(200, 2, 3);
        let b2 = mk_band(100, 1, 2);
        let b3 = mk_band(500, 2, 1);
        let bl = [[Some(&b0)], [Some(&b1)], [Some(&b2)], [Some(&b3)]];
        let srcs: Vec<_> = bl
            .iter()
            .map(|r| BandSource::new(w, h, w, h, 0, 0, 1, 1, r))
            .collect();
        let n = 512usize;
        let u_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (246.0 / n as f32)).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * (246.0 / n as f32)).collect();

        // 1. 4-band planar RawF32 cube
        if let Some(raw) = gpu.try_render_bbox_multiband_f32(
            None,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &srcs,
            ResampleMode::Lanczos,
        ) {
            assert_eq!(raw.len(), 4 * n * n * 4);
            let mut max_err = 0.0f32;
            for band in 0..4 {
                for py in (8..n - 8).step_by(37) {
                    for px in (8..n - 8).step_by(37) {
                        let off = (band * n * n + py * n + px) * 4;
                        let gv = f32::from_le_bytes([
                            raw[off],
                            raw[off + 1],
                            raw[off + 2],
                            raw[off + 3],
                        ]);
                        let cv = cogzilla_kernel::sample_general_scaled(
                            &srcs[band],
                            u_coords[px],
                            v_coords[py],
                            1.0,
                            1.0,
                            ResampleMode::Lanczos,
                        );
                        max_err = max_err.max((gv - cv).abs());
                    }
                }
            }
            assert!(max_err < 1e-2, "4-band MultiBand RawF32 max err {max_err} >= 1e-2");
        }

        // 2. 3-expression CustomExprRgb planar RawF32 cube
        let progs = [
            ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap(),
            ExprProgram::compile("b2*0.001").unwrap(),
            ExprProgram::compile("sqrt(b3)").unwrap(),
        ];
        if let Some(raw) = gpu.try_render_bbox_expr_rgb_f32(
            None,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &[1, 2, 3],
            &srcs[..3],
            &progs,
            ResampleMode::Lanczos,
        ) {
            assert_eq!(raw.len(), 3 * n * n * 4);
            let mut max_err = 0.0f32;
            for py in (8..n - 8).step_by(37) {
                for px in (8..n - 8).step_by(37) {
                    let s0 = cogzilla_kernel::sample_general_scaled(
                        &srcs[0], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let s1 = cogzilla_kernel::sample_general_scaled(
                        &srcs[1], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let s2 = cogzilla_kernel::sample_general_scaled(
                        &srcs[2], u_coords[px], v_coords[py], 1.0, 1.0, ResampleMode::Lanczos,
                    );
                    let expected = [(s0 - s1) / (s0 + s1), s1 * 0.001, s2.sqrt()];
                    for (c, &exp) in expected.iter().enumerate() {
                        let off = (c * n * n + py * n + px) * 4;
                        let gv = f32::from_le_bytes([
                            raw[off],
                            raw[off + 1],
                            raw[off + 2],
                            raw[off + 3],
                        ]);
                        max_err = max_err.max((gv - exp).abs());
                    }
                }
            }
            assert!(max_err < 1e-2, "CustomExprRgb RawF32 max err {max_err} >= 1e-2");
        }
    }

    #[test]
    fn nearest_resample_and_fused_bbox_stats_match_cpu() {
        use cogzilla_codec::{BandData, BlockStats, DecodedBand};
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping nearest + fused stats test (no GPU adapter)");
            return;
        };
        let sw = 512usize;
        let sh = 512usize;
        let mk_band = |bias: u16, mx: usize, my: usize, modulo: usize| {
            let mut vals = Vec::with_capacity(sw * sh);
            for y in 0..sh {
                for x in 0..sw {
                    vals.push(bias + ((x * mx + y * my) % modulo) as u16);
                }
            }
            DecodedBand {
                width: sw as u32,
                height: sh as u32,
                data: BandData::U16 {
                    data: vals.into_boxed_slice(),
                    nodata: None,
                },
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: bias as f32,
                    max: (bias as usize + modulo) as f32,
                },
            }
        };
        let b0 = mk_band(500, 7, 13, 1000);
        let b1 = mk_band(1800, 11, 5, 1200);
        let g0 = [Some(&b0)];
        let g1 = [Some(&b1)];
        let s0 = BandSource::new(sw as u32, sh as u32, sw as u32, sh as u32, 0, 0, 1, 1, &g0);
        let s1 = BandSource::new(sw as u32, sh as u32, sw as u32, sh as u32, 0, 0, 1, 1, &g1);

        let n = 1024usize;
        let u_coords: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * (sw as f32 - 1.0)).collect();
        let v_coords: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * (sh as f32 - 1.0)).collect();

        // 1. Verify bit-exact ResampleMode::Nearest via try_render_bbox_single_band_f32
        if let Some(raw) = gpu.try_render_bbox_single_band_f32(
            777,
            0,
            0,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &s0,
            ResampleMode::Nearest,
        ) {
            assert_eq!(raw.len(), n * n * 4);
            for py in (0..n).step_by(31) {
                for px in (0..n).step_by(31) {
                    let expected = cogzilla_kernel::sample_general_scaled(
                        &s0,
                        u_coords[px],
                        v_coords[py],
                        1.0,
                        1.0,
                        ResampleMode::Nearest,
                    );
                    let off = (py * n + px) * 4;
                    let gv = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                    assert_eq!(gv.to_bits(), expected.to_bits(), "Nearest mismatch at ({px},{py})");
                }
            }
        }

        // 2. Verify 1-pass fused try_render_bbox_single_band_stats (Nearest)
        if let Some(st) = gpu.try_render_bbox_single_band_stats(
            777,
            0,
            0,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &s0,
            ResampleMode::Nearest,
            10,
            None,
            &[2.0, 50.0, 98.0],
        ) {
            assert_eq!(st.valid_pixels, n * n);
            assert!((st.min.unwrap() - 500.0).abs() < 1.0);
            assert!((st.max.unwrap() - 1499.0).abs() < 1.0);
        }

        // 3. Verify 1-pass fused try_render_bbox_expr_stats (NDVI, Nearest)
        let prog = ExprProgram::compile("(b1-b2)/(b1+b2)").expect("compile ndvi");
        let st = gpu
            .try_render_bbox_expr_stats(
                None,
                BboxCoordSource::Separable {
                    u: &u_coords,
                    v: &v_coords,
                },
                &[s1, s0],
                &prog,
                ResampleMode::Nearest,
                10,
                None,
                &[2.0, 50.0, 98.0],
            )
            .expect("fused expr stats");
        assert_eq!(st.valid_pixels, n * n);
        assert!(
            st.min.unwrap() > 0.0 && st.max.unwrap() < 1.0,
            "unexpected NDVI range: min={:?}, max={:?}",
            st.min,
            st.max
        );
        assert_eq!(st.hist_counts.iter().sum::<u64>(), (n * n) as u64);
    }
}
