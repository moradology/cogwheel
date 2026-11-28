use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
pub use cogzilla_gpu::{GpuBandStats, GpuInstr, GpuPixelSelection, GpuWeightedBandStats};

// Compile-time seam contract: pin that `GpuInstr` keeps the exact 4-field 16-byte layout
// mirrored by `gpu_stub::GpuInstr` (the other three re-exports are shared directly from
// `cogzilla-raster` so they cannot drift).
const _: () = {
    let _assert_size: [u8; 16] = [0; std::mem::size_of::<GpuInstr>()];
    let _check_fields = |ins: GpuInstr| -> (u32, u32, f32, u32) {
        (ins.opcode, ins.imm_u32, ins.imm_f32, ins._pad)
    };
    let _ = _check_fields;
};
use cogzilla_gpu::{
    CoalescedJob, CoalescedOutput, DispatchPolicy, DispatchRequest, GpuCoalescer,
    GpuColorFormula, GpuColormapMode, GpuContext, GpuJpegEncoder,
    GpuMosaicCompositor, GpuMultiBandCache, GpuOp, GpuPixelOutput, GpuSlab,
    GpuSlabCache, GpuStatsReducer, GpuTileBatcher, GpuWeightedStatsReducer,
    MosaicCompositeResult as GpuMosaicResult, MosaicOutput, MosaicUpload, MultiBandSlabKey,
    OwnedCoordPlan, PolicyOverride, SlabKey, TileCoordSlice, pack_rgb_lut,
};
use cogzilla_kernel::expr_vm::Op;
use cogzilla_kernel::png_fast::{
    encode_filtered_png, encode_indexed_png, filter_sub_ga_row, filter_sub_u8_row,
};
use cogzilla_kernel::{
    BandSource, ColorFormula, ColormapName, CustomColormap, DemAlgorithm, ExprProgram,
    OutputFormat, ResampleMode, RgbStyle, SingleBandStyle, SpectralExpr,
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

impl<'a> BboxCoordSource<'a> {
    fn from_coord_plan<const N: usize>(plan: &'a CoordPlan<N>) -> Self {
        match plan {
            CoordPlan::Separable1D { u, v, .. } => Self::Separable {
                u: u.as_slice(),
                v: v.as_slice(),
            },
            CoordPlan::Grid2D { grid_u, grid_v, .. } => Self::Grid2D {
                dst_w: N,
                dst_h: N,
                grid_u,
                grid_v,
            },
            CoordPlan::Exact2D { u, v, .. } => Self::Exact2D {
                dst_w: N,
                dst_h: N,
                u_map: &u[..],
                v_map: &v[..],
            },
        }
    }

    /// Source pixels per output pixel, as `cogzilla_kernel::coord_plan_scale` estimates it.
    fn scale(&self) -> (f32, f32) {
        match *self {
            Self::Separable { u, v } | Self::SeparableMasked { u, v, .. } => {
                cogzilla_kernel::separable_coord_scale(u, v)
            }
            Self::Grid2D { dst_w, dst_h, grid_u, grid_v }
            | Self::Grid2DMasked { dst_w, dst_h, grid_u, grid_v, .. } => {
                cogzilla_kernel::grid_coord_scale(grid_u, grid_v, dst_w, dst_h)
            }
            Self::Exact2D { dst_w, dst_h, u_map, v_map } => {
                cogzilla_kernel::exact_coord_scale(u_map, v_map, dst_w, dst_h)
            }
        }
    }

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
    mosaic: Mutex<GpuMosaicCompositor>,
    batcher: Mutex<GpuTileBatcher>,
    jpeg: Mutex<GpuJpegEncoder>,
    stats: Mutex<GpuStatsReducer>,
    weighted_stats: Mutex<GpuWeightedStatsReducer>,
    in_flight: AtomicUsize,
}

impl GpuBackend {
    /// Number of GPU tile/render operations currently executing.
    #[inline]
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }
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

/// [`map_resample`] for a render over `coords`, `None` (render on the CPU) when the request
/// shrinks the source: the CPU widens convolution kernels there like GDAL, the shaders keep the
/// fixed taps.
fn map_resample_on(mode: ResampleMode, coords: &BboxCoordSource<'_>) -> Option<GpuColormapMode> {
    let gpu_mode = map_resample(mode)?;
    if mode != ResampleMode::Nearest {
        let (sx, sy) = coords.scale();
        if cogzilla_kernel::kernel_widens(mode, sx, sy) {
            return None;
        }
    }
    Some(gpu_mode)
}

fn multiband_policy_op(mode: GpuColormapMode) -> GpuOp {
    match mode {
        GpuColormapMode::Lanczos3 => GpuOp::FusedColormapLanczos3,
        GpuColormapMode::Cubic => GpuOp::DemHornCubic,
        GpuColormapMode::Bilinear | GpuColormapMode::Nearest => GpuOp::FusedColormapBilinear,
    }
}

fn is_rgba_format(format: OutputFormat) -> bool {
    matches!(
        format,
        OutputFormat::Png
            | OutputFormat::Webp { .. }
            | OutputFormat::Jpeg { .. }
            | OutputFormat::RawRgba
    )
}

fn styled_pixel_output(format: OutputFormat, return_mask: bool) -> Option<GpuPixelOutput> {
    match format {
        OutputFormat::Png if return_mask => Some(GpuPixelOutput::IndexedU8),
        OutputFormat::Png
        | OutputFormat::Webp { .. }
        | OutputFormat::Jpeg { .. }
        | OutputFormat::RawRgba => Some(GpuPixelOutput::Rgba8),
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
    let plte = build_plte_nodata(rgb);
    if rgb.iter().all(|c| c[0] == c[1] && c[1] == c[2]) {
        // Grayscale LUT: emit PNG L (color_type=0) when opaque or LA (color_type=4) when any
        // nodata pixel is present, matching CPU/TiTiler's L/LA PNG mode instead of palette `P`.
        let all_opaque = !idx_bytes.contains(&0);
        if all_opaque {
            let mut row_l = vec![0u8; w];
            let mut filtered = vec![0u8; h * (1 + w)];
            for py in 0..h {
                let src_row = &idx_bytes[py * w..(py + 1) * w];
                for (px, &idx) in src_row.iter().enumerate() {
                    row_l[px] = plte[idx as usize * 3];
                }
                let dst_row = &mut filtered[py * (1 + w)..(py + 1) * (1 + w)];
                filter_sub_u8_row(&row_l, dst_row);
            }
            return encode_filtered_png(width, height, 0, &filtered);
        }
        let mut row_ga = vec![0u8; w * 2];
        let mut filtered = vec![0u8; h * (1 + w * 2)];
        for py in 0..h {
            let src_row = &idx_bytes[py * w..(py + 1) * w];
            for (px, &idx) in src_row.iter().enumerate() {
                if idx == 0 {
                    row_ga[px * 2] = 0;
                    row_ga[px * 2 + 1] = 0;
                } else {
                    row_ga[px * 2] = plte[idx as usize * 3];
                    row_ga[px * 2 + 1] = 255;
                }
            }
            let dst_row = &mut filtered[py * (1 + w * 2)..(py + 1) * (1 + w * 2)];
            filter_sub_ga_row(&row_ga, dst_row);
        }
        return encode_filtered_png(width, height, 4, &filtered);
    }
    let mut filtered = vec![0u8; h * (1 + w)];
    for py in 0..h {
        let src_row = &idx_bytes[py * w..(py + 1) * w];
        let dst_row = &mut filtered[py * (1 + w)..(py + 1) * (1 + w)];
        filter_sub_u8_row(src_row, dst_row);
    }
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
    let w_us = w as usize;
    let h_us = h as usize;
    let th = src.tile_h as usize;
    if w_us == 0 || h_us == 0 || th == 0 {
        return;
    }
    let strip_elems = th * w_us;
    for (gy, strip) in out[..w_us * h_us].chunks_mut(strip_elems).enumerate() {
        if gy >= src.grid_h as usize {
            strip.fill(f32::NAN);
            continue;
        }
        stitch_band_row_strip(src, w_us, gy, strip);
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
                    strip.fill(f32::NAN);
                    continue;
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
    let tw = src.tile_w as usize;
    let th = src.tile_h as usize;
    let strip_rows = strip.len() / w_us;
    let max_rows = strip_rows.min(th);

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
        cogzilla_kernel::copy_band_data_rect_to_f32(
            &blk.data,
            tw,
            0,
            0,
            copy_w,
            copy_h,
            strip,
            w_us,
            dst_x0,
            0,
            src.nodata_override,
            src.scale_offset,
            false,
        );
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

/// Input planes for [`GpuBackend::try_composite_mosaic`].
pub enum MosaicCompositeInput<'a> {
    /// Single band across `2..=32` aligned scenes.
    Single(&'a [&'a [f32]]),
    /// `B` bands × `2..=32` aligned scenes.
    Bands(&'a [Vec<&'a [f32]>]),
    /// 3 RGB bands × `2..=32` aligned scenes.
    Rgb(&'a [&'a [&'a [f32]]; 3]),
}

/// Desired output mode for [`GpuBackend::try_composite_mosaic`].
pub enum MosaicCompositeOutput {
    /// Raw reduced `f32` plane(s).
    F32,
    /// Fused single-band rescale + colormap to PNG / WebP / JPEG / RawRgba.
    Styled {
        rescale: (f32, f32),
        colormap: ColormapName,
        format: OutputFormat,
        return_mask: bool,
    },
    /// Fused 3-band RGB per-channel rescale to JPEG / PNG / WebP / RawRgba.
    RgbStyled {
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
        format: OutputFormat,
        return_mask: bool,
    },
}

/// Request descriptor for [`GpuBackend::try_composite_mosaic`].
pub struct MosaicCompositeRequest<'a> {
    pub width: u32,
    pub height: u32,
    pub input: MosaicCompositeInput<'a>,
    pub method: GpuPixelSelection,
    pub output: MosaicCompositeOutput,
}

/// Output produced by [`GpuBackend::try_composite_mosaic`].
pub enum MosaicCompositeResult {
    F32(Vec<f32>),
    F32Bands(Vec<Vec<f32>>),
    Bytes(Bytes),
}

impl MosaicCompositeResult {
    pub fn into_f32(self) -> Option<Vec<f32>> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn into_f32_bands(self) -> Option<Vec<Vec<f32>>> {
        match self {
            Self::F32Bands(v) => Some(v),
            _ => None,
        }
    }

    pub fn into_bytes(self) -> Option<Bytes> {
        match self {
            Self::Bytes(b) => Some(b),
            _ => None,
        }
    }
}

/// Band shape + resolved source planes for [`GpuBackend::try_render_bbox`].
pub enum GpuBboxBands<'a> {
    Single {
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        src: &'a BandSource<'a>,
    },
    Dem {
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        src: &'a BandSource<'a>,
        algorithm: DemAlgorithm,
        dx_m: f32,
        dy_m: f32,
    },
    Terrain {
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        src: &'a BandSource<'a>,
        algorithm: DemAlgorithm,
    },
    Spectral {
        cache_key: Option<MultiBandSlabKey>,
        srcs: &'a [BandSource<'a>],
        expr: SpectralExpr,
    },
    Expr {
        cache_key: Option<MultiBandSlabKey>,
        srcs: &'a [BandSource<'a>],
        prog: &'a ExprProgram,
    },
    Rgb {
        cache_key: Option<MultiBandSlabKey>,
        srcs: &'a [BandSource<'a>; 3],
    },
    SarRgb {
        cache_key: Option<MultiBandSlabKey>,
        srcs: &'a [BandSource<'a>; 2],
    },
    ExprRgb {
        cache_key: Option<MultiBandSlabKey>,
        unique_bands: &'a [u16],
        srcs: &'a [BandSource<'a>],
        progs: &'a [ExprProgram; 3],
    },
    MultiBand {
        cache_key: Option<MultiBandSlabKey>,
        srcs: &'a [BandSource<'a>],
    },
}

/// Desired output mode for [`GpuBackend::try_render_bbox`].
pub enum GpuBboxOutput<'a> {
    /// Styled single-band / DEM / Spectral / Expr output via a resolved 256-entry RGB LUT.
    StyledLut {
        style: &'a SingleBandStyle,
        rgb_lut: &'a [[u8; 3]; 256],
        return_mask: bool,
    },
    /// Styled RGB / SAR-RGB / Expr-RGB output with optional color formula.
    StyledRgb {
        style: &'a RgbStyle,
        color_formula: Option<&'a ColorFormula>,
        return_mask: bool,
    },
    /// Styled Terrain (Terrarium / TerrainRGB / Contours) output.
    StyledTerrain {
        resample: ResampleMode,
        format: OutputFormat,
        return_mask: bool,
    },
    /// Raw `f32` plane(s) for `.npy`, `.tif`, `.f32`, or multi-scene mosaic fetches.
    F32 { resample: ResampleMode },
    /// 1-pass fused GPU resampling + band statistics / histogram / percentiles.
    Stats {
        resample: ResampleMode,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &'a [f64],
    },
}

/// Output produced by [`GpuBackend::try_render_bbox`].
pub enum GpuBboxResult {
    Bytes(Bytes),
    Stats(GpuBandStats),
}

impl GpuBboxResult {
    pub fn into_bytes(self) -> Option<Bytes> {
        match self {
            Self::Bytes(b) => Some(b),
            _ => None,
        }
    }

    pub fn into_stats(self) -> Option<GpuBandStats> {
        match self {
            Self::Stats(s) => Some(s),
            _ => None,
        }
    }
}

fn source_slab_key(cog_id: u64, ifd_idx: usize, band0: u16, src: &BandSource<'_>) -> Option<SlabKey> {
    if src.nodata_override.is_some()
        || src.scale_offset.is_some()
        || src.grid_w >= 256
        || src.grid_h >= 256
    {
        return None;
    }
    Some(SlabKey {
        dataset_id: cog_id,
        ifd_idx: ifd_idx as u16,
        band: band0,
        tx: (src.tx0 << 8) | (src.grid_w & 0xff),
        ty: (src.ty0 << 8) | (src.grid_h & 0xff),
    })
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
        let mut policy = DispatchPolicy::calibrate(&ctx);
        // `GpuCoalescer` packs up to 32 concurrent jobs per batch asynchronously (without
        // blocking Tokio worker threads), so allow up to 128 coalesced batches (4,096 concurrent
        // requests) before shedding load to synchronous CPU fallback.
        policy.max_in_flight = 128;
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
        let coalesce_us = if ctx.info.unified_memory { 400 } else { 250 };
        let coalescer = GpuCoalescer::new(&ctx, 32, coalesce_us);
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
        key: Option<SlabKey>,
        warm: Option<Arc<GpuSlab>>,
        src: &BandSource<'_>,
    ) -> Option<(Arc<GpuSlab>, u32, u32)> {
        if let Some((slice, base_col, base_row, stride)) = src.single_block_flat() {
            let slab = match warm {
                Some(s) => s,
                None => {
                    let height = (slice.len() / stride) as u32;
                    let width = stride as u32;
                    if let Some(k) = key {
                        self.slab_cache
                            .get_or_upload(k, slice, width, height, stride as u32, &self.ctx.device)
                            .ok()?
                    } else {
                        Arc::new(
                            GpuSlab::upload(&self.ctx.device, slice, width, height, stride as u32)
                                .ok()?,
                        )
                    }
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
        if let Some(k) = key {
            self.slab_cache.insert(k, Arc::clone(&slab));
        }
        Some((slab, bc, br))
    }

    fn estimate_upload_bytes(warm: &Option<Arc<GpuSlab>>, src: &BandSource<'_>) -> usize {
        if warm.is_some() {
            return 0;
        }
        if let Some((slice, _, _, _)) = src.single_block_flat() {
            return std::mem::size_of_val(slice);
        }
        let w = (src.grid_w * src.tile_w) as usize;
        let h = (src.grid_h * src.tile_h) as usize;
        w.saturating_mul(h).saturating_mul(std::mem::size_of::<f32>())
    }

    #[inline]
    fn enter_in_flight(&self) -> (usize, InFlightGuard<'_>) {
        let active = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        (active, InFlightGuard(&self.in_flight))
    }

    #[allow(clippy::too_many_arguments)]
    fn gate_and_resolve_single_band(
        &self,
        cache_key: Option<SlabKey>,
        src: &BandSource<'_>,
        op: GpuOp,
        dst_pixels: usize,
        coord_upload_bytes: usize,
        readback_bytes: usize,
        active: usize,
    ) -> Option<(Arc<GpuSlab>, u32, u32)> {
        let warm_slab = cache_key.and_then(|k| self.slab_cache.get(&k));
        let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src) + coord_upload_bytes;
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels,
            upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }
        self.resolve_source_slab(cache_key, warm_slab, src)
    }

    #[allow(clippy::too_many_arguments)]
    fn gate_and_resolve_multiband(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        srcs: &[BandSource<'_>],
        op: GpuOp,
        dst_pixels: usize,
        coord_upload_bytes: usize,
        readback_bytes: usize,
        active: usize,
    ) -> Option<(Arc<cogzilla_gpu::GpuMultiBandSlab>, u32, u32)> {
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let band_upload_bytes: usize = if warm_slab.is_some() {
            0
        } else {
            srcs.iter()
                .map(|s| Self::estimate_upload_bytes(&None, s))
                .sum()
        };
        let decision = self.policy.decide(&DispatchRequest {
            op,
            dst_pixels,
            upload_bytes: band_upload_bytes + coord_upload_bytes,
            readback_bytes,
            batch_size: active.min(32),
            in_flight: (active - 1) / 32,
        });
        if !decision.use_gpu {
            return None;
        }
        self.resolve_multiband_slab(cache_key, warm_slab, srcs)
    }

    #[inline]
    fn fused_stats_min_pixels(&self) -> usize {
        if self.ctx.info.unified_memory {
            512 * 1024
        } else {
            512 * 512
        }
    }

    fn submit_fused_stats(
        &self,
        inner: CoalescedJob,
        total_pixels: usize,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Option<GpuBandStats> {
        let raw = self
            .coalescer
            .submit(CoalescedJob::WithStats {
                inner: Box::new(inner),
                nodata: None,
                num_bins,
                hist_range,
            })
            .ok()?;
        Some(cogzilla_gpu::decode_stats_readback(
            &raw,
            total_pixels,
            num_bins,
            hist_range,
            percentiles,
        ))
    }

    async fn submit_coalesced_async(&self, job: CoalescedJob) -> Option<Vec<u8>> {
        struct CancelOnDrop(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for CancelOnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let _cancel_guard = CancelOnDrop(Arc::clone(&cancelled));
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.coalescer
            .submit_cancellable(job, cancelled, move |res| {
                let _ = tx.send(res);
            })
            .ok()?;
        rx.await.ok()?.ok()
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
        match custom_lut {
            None => {
                // Match TiTiler / rio-tiler CPU rule: apply `color_formula` to the rescaled
                // grayscale byte `[i, i, i]` BEFORE looking up the built-in colormap.
                let base = colormap.lut().rgb;
                if let Some(cf) = color_formula {
                    let mut out = [[0u8; 3]; 256];
                    for (i, slot) in out.iter_mut().enumerate() {
                        let (r, _, _) = cf.apply_rgb(i as u8, i as u8, i as u8);
                        *slot = base[r as usize];
                    }
                    Some(out)
                } else {
                    Some(base)
                }
            }
            Some(CustomColormap::Discrete {
                lut,
                extra,
                direct_lookup: false,
            }) if extra.is_empty() && lut.rgba_u32.iter().all(|&p| (p >> 24) == 255) => {
                let mut rgb = lut.rgb;
                if let Some(cf) = color_formula {
                    for entry in &mut rgb {
                        let (r, g, b) = cf.apply_rgb(entry[0], entry[1], entry[2]);
                        *entry = [r, g, b];
                    }
                }
                Some(rgb)
            }
            Some(_) => None,
        }
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
        let gpu_mode = map_resample_on(style.resample, &coords)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let out_kind = styled_pixel_output(style.format, return_mask)?;
        let readback_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => dst_w * dst_h,
            GpuPixelOutput::Rgba8 => dst_w * dst_h * 4,
        };
        let (slab, base_col, base_row) = self.gate_and_resolve_single_band(
            source_slab_key(cog_id, ifd_idx, band0, src),
            src,
            gpu_mode.to_policy_op(),
            dst_w * dst_h,
            coord_upload_bytes,
            readback_bytes,
            active,
        )?;
        let lut_u32 = pack_rgb_lut(rgb_lut);
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
            if dst_w.saturating_mul(dst_h) >= min_px
                && let Ok(mut enc) = self.jpeg.lock()
                    && let Ok(jpg) = enc.encode_rgba8(rgba, dst_w as u32, dst_h as u32, quality) {
                        return Bytes::from(jpg);
                    }
        }
        crate::stats::encode_rgba_output(dst_w, dst_h, rgba, format, return_mask)
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

    /// 1-pass fused GPU expression VM + single-submit weighted zonal statistics across `E`
    /// expressions sharing `1..=8` raw input bands, analytical coverage `covs`, and normalized
    /// per-row areas `row_areas`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_reduce_weighted_zonal_expr_stats(
        &self,
        raw_bands: &[&[f32]],
        exprs: &[(&[GpuInstr], u32)],
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
        let min_px = if self.ctx.info.unified_memory && exprs.len() < 2 {
            512 * 512
        } else {
            256 * 256
        };
        if covs.len() < min_px || raw_bands.is_empty() || raw_bands.len() > 8 || exprs.is_empty() {
            return None;
        }
        let mut guard = self.weighted_stats.lock().ok()?;
        guard
            .reduce_weighted_exprs(
                raw_bands,
                exprs,
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

    /// Unified GPU bbox/tile/crop render entry point across all band shapes and output modes.
    ///
    /// Replaces the 18 per-(band-shape × output-mode) `try_render_bbox_*` entry points with one
    /// `(bands, out)` dispatch table. Every filled cell delegates to its specialized fused GPU
    /// kernel; unfilled cells return `None` to fall back cleanly to the CPU path.
    pub fn try_render_bbox(
        &self,
        coords: BboxCoordSource<'_>,
        bands: &GpuBboxBands<'_>,
        out: &GpuBboxOutput<'_>,
    ) -> Option<GpuBboxResult> {
        match (bands, out) {
            (
                GpuBboxBands::Single {
                    cog_id,
                    ifd_idx,
                    band0,
                    src,
                },
                GpuBboxOutput::StyledLut {
                    style,
                    rgb_lut,
                    return_mask,
                },
            ) => self
                .try_render_bbox_single_band_with_lut(
                    *cog_id,
                    *ifd_idx,
                    *band0,
                    coords,
                    src,
                    style,
                    rgb_lut,
                    *return_mask,
                )
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::Single {
                    cog_id,
                    ifd_idx,
                    band0,
                    src,
                },
                GpuBboxOutput::F32 { resample },
            ) => self
                .try_render_bbox_single_band_f32(*cog_id, *ifd_idx, *band0, coords, src, *resample)
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::Single {
                    cog_id,
                    ifd_idx,
                    band0,
                    src,
                },
                GpuBboxOutput::Stats {
                    resample,
                    num_bins,
                    hist_range,
                    percentiles,
                },
            ) => self
                .try_render_bbox_single_band_stats(
                    *cog_id,
                    *ifd_idx,
                    *band0,
                    coords,
                    src,
                    *resample,
                    *num_bins,
                    *hist_range,
                    percentiles,
                )
                .map(GpuBboxResult::Stats),
            (
                GpuBboxBands::Dem {
                    cog_id,
                    ifd_idx,
                    band0,
                    src,
                    algorithm,
                    dx_m,
                    dy_m,
                },
                GpuBboxOutput::StyledLut {
                    style,
                    rgb_lut,
                    return_mask,
                },
            ) => self
                .try_render_bbox_dem_with_lut(
                    *cog_id,
                    *ifd_idx,
                    *band0,
                    coords,
                    src,
                    *algorithm,
                    *dx_m,
                    *dy_m,
                    style,
                    rgb_lut,
                    *return_mask,
                )
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::Dem {
                    cog_id,
                    ifd_idx,
                    band0,
                    src,
                    algorithm,
                    dx_m,
                    dy_m,
                },
                GpuBboxOutput::F32 { resample },
            ) => self
                .try_render_bbox_dem_f32(
                    *cog_id, *ifd_idx, *band0, coords, src, *algorithm, *dx_m, *dy_m, *resample,
                )
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::Dem {
                    cog_id,
                    ifd_idx,
                    band0,
                    src,
                    algorithm,
                    dx_m,
                    dy_m,
                },
                GpuBboxOutput::Stats {
                    resample,
                    num_bins,
                    hist_range,
                    percentiles,
                },
            ) => self
                .try_render_bbox_dem_stats(
                    *cog_id,
                    *ifd_idx,
                    *band0,
                    coords,
                    src,
                    *algorithm,
                    *dx_m,
                    *dy_m,
                    *resample,
                    *num_bins,
                    *hist_range,
                    percentiles,
                )
                .map(GpuBboxResult::Stats),
            (
                GpuBboxBands::Terrain {
                    cog_id,
                    ifd_idx,
                    band0,
                    src,
                    algorithm,
                },
                GpuBboxOutput::StyledTerrain {
                    resample,
                    format,
                    return_mask,
                },
            ) => self
                .try_render_bbox_terrain(
                    *cog_id,
                    *ifd_idx,
                    *band0,
                    coords,
                    src,
                    *algorithm,
                    *resample,
                    *format,
                    *return_mask,
                )
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::Spectral {
                    cache_key,
                    srcs,
                    expr,
                },
                GpuBboxOutput::StyledLut {
                    style,
                    rgb_lut,
                    return_mask,
                },
            ) => self
                .try_render_bbox_spectral_with_lut(
                    *cache_key,
                    coords,
                    srcs,
                    *expr,
                    style,
                    rgb_lut,
                    *return_mask,
                )
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::Spectral {
                    cache_key,
                    srcs,
                    expr,
                },
                GpuBboxOutput::F32 { resample },
            ) => self
                .try_render_bbox_spectral_f32(*cache_key, coords, srcs, *expr, *resample)
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::Spectral {
                    cache_key,
                    srcs,
                    expr,
                },
                GpuBboxOutput::Stats {
                    resample,
                    num_bins,
                    hist_range,
                    percentiles,
                },
            ) => self
                .try_render_bbox_spectral_stats(
                    *cache_key,
                    coords,
                    srcs,
                    *expr,
                    *resample,
                    *num_bins,
                    *hist_range,
                    percentiles,
                )
                .map(GpuBboxResult::Stats),
            (
                GpuBboxBands::Expr {
                    cache_key,
                    srcs,
                    prog,
                },
                GpuBboxOutput::StyledLut {
                    style,
                    rgb_lut,
                    return_mask,
                },
            ) => self
                .try_render_bbox_expr_with_lut(
                    *cache_key,
                    coords,
                    srcs,
                    prog,
                    style,
                    rgb_lut,
                    *return_mask,
                )
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::Expr {
                    cache_key,
                    srcs,
                    prog,
                },
                GpuBboxOutput::F32 { resample },
            ) => self
                .try_render_bbox_expr_f32(*cache_key, coords, srcs, prog, *resample)
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::Expr {
                    cache_key,
                    srcs,
                    prog,
                },
                GpuBboxOutput::Stats {
                    resample,
                    num_bins,
                    hist_range,
                    percentiles,
                },
            ) => self
                .try_render_bbox_expr_stats(
                    *cache_key,
                    coords,
                    srcs,
                    prog,
                    *resample,
                    *num_bins,
                    *hist_range,
                    percentiles,
                )
                .map(GpuBboxResult::Stats),
            (
                GpuBboxBands::Rgb { cache_key, srcs },
                GpuBboxOutput::StyledRgb {
                    style,
                    color_formula,
                    return_mask,
                },
            ) => self
                .try_render_bbox_rgb(*cache_key, coords, srcs, style, *color_formula, *return_mask)
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::SarRgb { cache_key, srcs },
                GpuBboxOutput::StyledRgb {
                    style,
                    color_formula,
                    return_mask,
                },
            ) => self
                .try_render_bbox_sar_rgb(
                    *cache_key,
                    coords,
                    srcs,
                    style,
                    *color_formula,
                    *return_mask,
                )
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::ExprRgb {
                    cache_key,
                    unique_bands,
                    srcs,
                    progs,
                },
                GpuBboxOutput::StyledRgb {
                    style,
                    color_formula,
                    return_mask,
                },
            ) => self
                .try_render_bbox_expr_rgb(
                    *cache_key,
                    coords,
                    unique_bands,
                    srcs,
                    progs,
                    style,
                    *color_formula,
                    *return_mask,
                )
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::ExprRgb {
                    cache_key,
                    unique_bands,
                    srcs,
                    progs,
                },
                GpuBboxOutput::F32 { resample },
            ) => self
                .try_render_bbox_expr_rgb_f32(
                    *cache_key,
                    coords,
                    unique_bands,
                    srcs,
                    progs,
                    *resample,
                )
                .map(GpuBboxResult::Bytes),
            (
                GpuBboxBands::MultiBand { cache_key, srcs },
                GpuBboxOutput::F32 { resample },
            ) => self
                .try_render_bbox_multiband_f32(*cache_key, coords, srcs, *resample)
                .map(GpuBboxResult::Bytes),
            // Unfilled combinations fall back cleanly to the CPU path.
            _ => None,
        }
    }

    /// Single-band GPU render with an explicit 256-entry RGB LUT (supporting `CustomColormap`
    /// and zero-cost `ColorFormula` folding).
    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_single_band_with_lut(
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
        let (active, _guard) = self.enter_in_flight();
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_single_band(
            cog_id, ifd_idx, band0, coords, src, style, rgb_lut, return_mask, active,
        )?;
        self.submit_styled(job, out_kind, dst_w, dst_h, style, rgb_lut, return_mask)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_bbox_dem(
        &self,
        cache_key: Option<SlabKey>,
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
        let gpu_mode = map_resample_on(style.resample, &coords)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        if alg.is_terrain_rgb_encoding() {
            return None;
        }
        let gpu_alg = alg.gpu_mode().map(|_| alg)?;
        let out_kind = styled_pixel_output(style.format, return_mask)?;
        let readback_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => dst_w * dst_h,
            GpuPixelOutput::Rgba8 => dst_w * dst_h * 4,
        };
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::DemHornLanczos3,
            _ => GpuOp::DemHornCubic,
        };
        let (slab, base_col, base_row) = self.gate_and_resolve_single_band(
            cache_key,
            src,
            op,
            dst_w * dst_h,
            coord_upload_bytes,
            readback_bytes,
            active,
        )?;
        let lut_u32 = pack_rgb_lut(rgb_lut);
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

    /// DEM numpy-gradient GPU render with an explicit 256-entry RGB LUT (supporting `CustomColormap`
    /// and zero-cost `ColorFormula` folding).
    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_dem_with_lut(
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
        let (active, _guard) = self.enter_in_flight();
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_dem(
            source_slab_key(cog_id, ifd_idx, band0, src),
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
        cache_key: Option<SlabKey>,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        algorithm: DemAlgorithm,
        resample: ResampleMode,
        format: OutputFormat,
        active: usize,
    ) -> Option<(CoalescedJob, usize, usize)> {
        let gpu_mode = map_resample_on(resample, &coords)?;
        let gpu_alg = if algorithm.is_terrain_rgb_encoding() {
            algorithm
        } else {
            return None;
        };
        if !is_rgba_format(format) {
            return None;
        }
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::ResampleLanczos3,
            _ => GpuOp::DemTerrainRgbCubic,
        };
        let (slab, base_col, base_row) = self.gate_and_resolve_single_band(
            cache_key,
            src,
            op,
            dst_w * dst_h,
            coord_upload_bytes,
            dst_w * dst_h * 4,
            active,
        )?;
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
    fn try_render_bbox_terrain(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        algorithm: DemAlgorithm,
        resample: ResampleMode,
        format: OutputFormat,
        return_mask: bool,
    ) -> Option<Bytes> {
        let (active, _guard) = self.enter_in_flight();
        let (job, dst_w, dst_h) = self.prepare_bbox_terrain(
            source_slab_key(cog_id, ifd_idx, band0, src),
            coords,
            src,
            algorithm,
            resample,
            format,
            active,
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
        let coords = BboxCoordSource::from_coord_plan(plan);
        let (active, _guard) = self.enter_in_flight();
        let rgb_lut = &style.colormap.lut().rgb;
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_single_band(
            cog_id, ifd_idx, band0, coords, src, style, rgb_lut, true, active,
        )?;
        self.submit_styled_async(job, out_kind, dst_w, dst_h, style, rgb_lut, true)
            .await
    }

    /// Attempt GPU-accelerated DEM Horn 3×3 tile rendering (non-blocking async coalesced).
    /// Pass `source_id = None` for derived planes (e.g. band-math outputs) that have no
    /// `(cog_id, ifd_idx, band0)` slab-cache identity.
    #[allow(clippy::too_many_arguments)]
    pub async fn try_render_dem<const N: usize>(
        &self,
        source_id: Option<(u64, usize, u16)>,
        _tx0: u32,
        _ty0: u32,
        plan: &CoordPlan<N>,
        src: &BandSource<'_>,
        alg: DemAlgorithm,
        pixel_size_m: f32,
        style: &SingleBandStyle,
    ) -> Option<Bytes> {
        let coords = BboxCoordSource::from_coord_plan(plan);
        let (active, _guard) = self.enter_in_flight();
        let rgb_lut = &style.colormap.lut().rgb;
        let cache_key =
            source_id.and_then(|(cog_id, ifd_idx, band0)| source_slab_key(cog_id, ifd_idx, band0, src));
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_dem(
            cache_key,
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
    /// Pass `source_id = None` for derived planes (e.g. band-math outputs) that have no
    /// `(cog_id, ifd_idx, band0)` slab-cache identity.
    #[allow(clippy::too_many_arguments)]
    pub async fn try_render_terrain<const N: usize>(
        &self,
        source_id: Option<(u64, usize, u16)>,
        _tx0: u32,
        _ty0: u32,
        plan: &CoordPlan<N>,
        src: &BandSource<'_>,
        algorithm: DemAlgorithm,
        resample: ResampleMode,
        format: OutputFormat,
    ) -> Option<Bytes> {
        let coords = BboxCoordSource::from_coord_plan(plan);
        let (active, _guard) = self.enter_in_flight();
        let cache_key =
            source_id.and_then(|(cog_id, ifd_idx, band0)| source_slab_key(cog_id, ifd_idx, band0, src));
        let (job, dst_w, dst_h) = self.prepare_bbox_terrain(
            cache_key, coords, src, algorithm, resample, format, active,
        )?;
        self.submit_rgba_async(job, dst_w, dst_h, format, true).await
    }

    /// Translate a slice of `cogzilla-kernel::expr_vm::Op` into GPU WGSL VM bytecode.
    /// Returns `None` if any op is unsupported by the GPU VM (e.g. `Median(n)`), causing seamless
    /// CPU fallback.
    pub fn compile_ops_to_gpu(ops: &[Op]) -> Option<Vec<GpuInstr>> {
        if ops.is_empty() || ops.len() > 256 {
            return None;
        }
        let mut out = Vec::with_capacity(ops.len());
        for &op in ops {
            let opcode = op.wire_opcode()?;
            let ins = match op {
                Op::PushConst(v) => GpuInstr::push_const(v),
                Op::LoadBand(slot) => GpuInstr::load_band(slot as u32),
                Op::ModPow2(c) => GpuInstr::op_f32(opcode, c),
                Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Pow
                | Op::Neg
                | Op::Log10
                | Op::Ln
                | Op::Log2
                | Op::Db
                | Op::Exp
                | Op::Sqrt
                | Op::Abs
                | Op::Sin
                | Op::Cos
                | Op::Tan
                | Op::Min
                | Op::Max
                | Op::Clip
                | Op::Lt
                | Op::Le
                | Op::Gt
                | Op::Ge
                | Op::Eq
                | Op::Ne
                | Op::Where
                | Op::Ceil
                | Op::Floor
                | Op::Round
                | Op::Mod
                | Op::BitAnd
                | Op::BitOr
                | Op::BitNot
                | Op::Log1p
                | Op::Expm1
                | Op::Sign
                | Op::Arcsin
                | Op::Arccos
                | Op::Arctan
                | Op::Arctan2
                | Op::Sinh
                | Op::Cosh
                | Op::Tanh
                | Op::Trunc => GpuInstr::op(opcode),
                Op::Median(_) | Op::Std(_) | Op::Var(_) => return None,
            };
            out.push(ins);
        }
        Some(out)
    }

    /// Translate a compiled `cogzilla-kernel::ExprProgram` into GPU WGSL VM bytecode.
    pub fn compile_expr_to_gpu(prog: &ExprProgram) -> Option<Vec<GpuInstr>> {
        if prog.used_bands.len() > 8 {
            return None;
        }
        Self::compile_ops_to_gpu(&prog.ops)
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
                || s.grid_w >= 256
                || s.grid_h >= 256
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
    /// directly into GPU VM bytecode via `SpectralExpr::to_ops()` (single source of truth).
    pub fn compile_spectral_to_gpu(expr: SpectralExpr) -> Option<Vec<GpuInstr>> {
        Self::compile_ops_to_gpu(expr.to_ops()?)
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
        let gpu_mode = map_resample_on(style.resample, &coords)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || srcs.is_empty() || srcs.len() > 8 {
            return None;
        }
        let out_kind = styled_pixel_output(style.format, return_mask)?;
        let readback_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => dst_w * dst_h,
            GpuPixelOutput::Rgba8 => dst_w * dst_h * 4,
        };
        let (slab, base_col, base_row) = self.gate_and_resolve_multiband(
            cache_key,
            srcs,
            multiband_policy_op(gpu_mode),
            dst_w * dst_h,
            coord_upload_bytes,
            readback_bytes,
            active,
        )?;
        let lut_u32 = pack_rgb_lut(rgb_lut);
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
        let coords = BboxCoordSource::from_coord_plan(plan);
        let cache_key = Self::multiband_key(cog_id, ifd_idx, bands_1based, srcs);
        let (active, _guard) = self.enter_in_flight();
        let rgb_lut = &style.colormap.lut().rgb;
        let (job, out_kind, dst_w, dst_h) = self.prepare_bbox_instrs(
            cache_key, coords, srcs, &instrs, style, rgb_lut, true, active,
        )?;
        self.submit_styled_async(job, out_kind, dst_w, dst_h, style, rgb_lut, true)
            .await
    }

    /// Built-in `SpectralExpr` GPU render with an explicit 256-entry RGB LUT.
    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_spectral_with_lut(
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
    fn try_render_bbox_spectral_f32(
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
        let coords = BboxCoordSource::from_coord_plan(plan);
        let cache_key = Self::multiband_key(cog_id, ifd_idx, &prog.used_bands, srcs);
        let (active, _guard) = self.enter_in_flight();
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
        let coords = BboxCoordSource::from_coord_plan(plan);
        let cache_key = Self::multiband_key(cog_id, ifd_idx, &bands_1based, srcs);
        let (active, _guard) = self.enter_in_flight();
        let (job, dst_w, dst_h) =
            self.prepare_bbox_rgb(cache_key, coords, srcs, style, None, active)?;
        self.submit_rgba_async(job, dst_w, dst_h, style.format, true)
            .await
    }

    /// Multi-band `ExprProgram` GPU render with an explicit 256-entry RGB LUT.
    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_expr_with_lut(
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
        let (active, _guard) = self.enter_in_flight();
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
        let gpu_mode = map_resample_on(style.resample, &coords)?;
        let gpu_formula = Self::translate_color_formula(color_formula)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || !is_rgba_format(style.format) {
            return None;
        }
        let (slab, base_col, base_row) = self.gate_and_resolve_multiband(
            cache_key,
            srcs,
            multiband_policy_op(gpu_mode),
            dst_w * dst_h,
            coord_upload_bytes,
            dst_w * dst_h * 4,
            active,
        )?;
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
    fn try_render_bbox_rgb(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>; 3],
        style: &RgbStyle,
        color_formula: Option<&ColorFormula>,
        return_mask: bool,
    ) -> Option<Bytes> {
        let (active, _guard) = self.enter_in_flight();
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

    /// Unified GPU-accelerated multi-scene mosaic pixel-selection compositing (`First`, `Highest`,
    /// `Lowest`, `Mean`, `Median`, `Stdev`, `Count`) over `2..=32` aligned `f32` scenes.
    ///
    /// Collapses the single-band `f32`, multi-band `f32`, fused single-band styled (PNG `IndexedU8`
    /// / GPU JPEG FDCT / WebP / RawRgba), and fused 3-band RGB styled paths behind one dispatch
    /// gate and one `(input, output)` match table.
    pub fn try_composite_mosaic(
        &self,
        req: &MosaicCompositeRequest<'_>,
    ) -> Option<MosaicCompositeResult> {
        let (width, height, method) = (req.width, req.height, req.method);
        if width == 0 || height == 0 {
            return None;
        }
        let pixels = (width as usize).checked_mul(height as usize)?;
        let jpeg_readback = || {
            let mcu_w = (width as usize).div_ceil(16);
            let mcu_h = (height as usize).div_ceil(16);
            mcu_w.saturating_mul(mcu_h).saturating_mul(192 * 4)
        };

        // 1. Validate shape + compute policy dimensions in one table.
        let (num_scenes, dst_pixels, upload_bytes, readback_bytes, batch_size, styled_out_kind) =
            match (&req.input, &req.output) {
                (MosaicCompositeInput::Single(planes), MosaicCompositeOutput::F32) => {
                    if !(2..=32).contains(&planes.len()) {
                        return None;
                    }
                    (
                        planes.len(),
                        pixels,
                        planes.len().saturating_mul(pixels).saturating_mul(4),
                        pixels.saturating_mul(4),
                        1,
                        None,
                    )
                }
                (MosaicCompositeInput::Bands(band_planes), MosaicCompositeOutput::F32) => {
                    if band_planes.is_empty() {
                        return None;
                    }
                    let ns = band_planes[0].len();
                    if !(2..=32).contains(&ns) || band_planes.iter().any(|bp| bp.len() != ns) {
                        return None;
                    }
                    let nb = band_planes.len();
                    (
                        ns,
                        pixels.saturating_mul(nb),
                        nb.saturating_mul(ns).saturating_mul(pixels).saturating_mul(4),
                        nb.saturating_mul(pixels).saturating_mul(4),
                        nb,
                        None,
                    )
                }
                (
                    MosaicCompositeInput::Single(planes),
                    MosaicCompositeOutput::Styled {
                        format,
                        return_mask,
                        ..
                    },
                ) => {
                    if !(2..=32).contains(&planes.len()) {
                        return None;
                    }
                    let out_kind = match format {
                        OutputFormat::Png if *return_mask => GpuPixelOutput::IndexedU8,
                        OutputFormat::Png
                        | OutputFormat::Webp { .. }
                        | OutputFormat::Jpeg { .. }
                        | OutputFormat::RawRgba => GpuPixelOutput::Rgba8,
                        _ => return None,
                    };
                    let rb = match format {
                        OutputFormat::Jpeg { .. } => jpeg_readback(),
                        _ => match out_kind {
                            GpuPixelOutput::IndexedU8 => pixels,
                            GpuPixelOutput::Rgba8 => pixels.saturating_mul(4),
                        },
                    };
                    (
                        planes.len(),
                        pixels,
                        planes.len().saturating_mul(pixels).saturating_mul(4),
                        rb,
                        1,
                        Some(out_kind),
                    )
                }
                (
                    MosaicCompositeInput::Rgb(rgb_planes),
                    MosaicCompositeOutput::RgbStyled { format, .. },
                ) => {
                    let ns = rgb_planes[0].len();
                    if !(2..=32).contains(&ns)
                        || rgb_planes[1].len() != ns
                        || rgb_planes[2].len() != ns
                    {
                        return None;
                    }
                    if !matches!(
                        format,
                        OutputFormat::Png
                            | OutputFormat::Webp { .. }
                            | OutputFormat::Jpeg { .. }
                            | OutputFormat::RawRgba
                    ) {
                        return None;
                    }
                    let rb = match format {
                        OutputFormat::Jpeg { .. } => jpeg_readback(),
                        _ => pixels.saturating_mul(4),
                    };
                    (
                        ns,
                        pixels.saturating_mul(3),
                        3usize
                            .saturating_mul(ns)
                            .saturating_mul(pixels)
                            .saturating_mul(4),
                        rb,
                        3,
                        None,
                    )
                }
                // Unfilled combinations fall back to CPU cleanly.
                _ => return None,
            };

        let decision = self.policy.decide(&DispatchRequest {
            op: Self::mosaic_policy_op(method, num_scenes),
            dst_pixels,
            upload_bytes,
            readback_bytes,
            batch_size,
            in_flight: self.in_flight.load(Ordering::Relaxed) / 32,
        });
        if !decision.use_gpu {
            return None;
        }

        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let _guard = InFlightGuard(&self.in_flight);

        // 2. Dispatch through the single `GpuMosaicCompositor::composite` entry point.
        match (&req.input, &req.output) {
            (MosaicCompositeInput::Single(planes), MosaicCompositeOutput::F32) => {
                let mut mosaic_guard = self.mosaic.lock().ok()?;
                match mosaic_guard
                    .composite(
                        MosaicUpload::Direct {
                            planes,
                            width,
                            height,
                        },
                        method,
                        MosaicOutput::F32,
                    )
                    .ok()?
                {
                    GpuMosaicResult::F32(v) => Some(MosaicCompositeResult::F32(v)),
                    _ => None,
                }
            }
            (MosaicCompositeInput::Bands(band_planes), MosaicCompositeOutput::F32) => {
                let mut mosaic_guard = self.mosaic.lock().ok()?;
                match mosaic_guard
                    .composite(
                        MosaicUpload::DirectBands {
                            bands: band_planes,
                            width,
                            height,
                        },
                        method,
                        MosaicOutput::F32Bands,
                    )
                    .ok()?
                {
                    GpuMosaicResult::F32Bands(v) => Some(MosaicCompositeResult::F32Bands(v)),
                    _ => None,
                }
            }
            (
                MosaicCompositeInput::Single(planes),
                MosaicCompositeOutput::Styled {
                    rescale,
                    colormap,
                    format,
                    return_mask,
                },
            ) => {
                let out_kind = styled_out_kind?;
                let lut = colormap.lut();
                let lut_u32 = pack_rgb_lut(&lut.rgb);
                let mut mosaic_guard = self.mosaic.lock().ok()?;

                if let OutputFormat::Jpeg { quality } = *format
                    && let Ok(mut jpeg_guard) = self.jpeg.lock()
                    && let Ok(GpuMosaicResult::Bytes(jpg)) = mosaic_guard.composite(
                        MosaicUpload::Direct {
                            planes,
                            width,
                            height,
                        },
                        method,
                        MosaicOutput::StyledJpeg {
                            lut: &lut_u32,
                            jpeg: &mut jpeg_guard,
                            quality,
                            rescale: *rescale,
                        },
                    )
                {
                    return Some(MosaicCompositeResult::Bytes(Bytes::from(jpg)));
                }

                let GpuMosaicResult::Bytes(raw) = mosaic_guard
                    .composite(
                        MosaicUpload::Direct {
                            planes,
                            width,
                            height,
                        },
                        method,
                        MosaicOutput::Styled {
                            lut: &lut_u32,
                            out: out_kind,
                            rescale: *rescale,
                        },
                    )
                    .ok()?
                else {
                    return None;
                };
                drop(mosaic_guard);

                let bytes = match out_kind {
                    GpuPixelOutput::IndexedU8 => {
                        encode_indexed_u8_png(&raw, width, height, &lut.rgb)
                    }
                    GpuPixelOutput::Rgba8 => {
                        if *format == OutputFormat::RawRgba {
                            Bytes::from(raw)
                        } else {
                            crate::stats::encode_rgba_output(
                                width as usize,
                                height as usize,
                                &raw,
                                *format,
                                *return_mask,
                            )
                        }
                    }
                };
                Some(MosaicCompositeResult::Bytes(bytes))
            }
            (
                MosaicCompositeInput::Rgb(rgb_planes),
                MosaicCompositeOutput::RgbStyled {
                    rescale_min,
                    rescale_max,
                    format,
                    return_mask,
                },
            ) => {
                let mut mosaic_guard = self.mosaic.lock().ok()?;
                if let OutputFormat::Jpeg { quality } = *format
                    && let Ok(mut jpeg_guard) = self.jpeg.lock()
                    && let Ok(GpuMosaicResult::Bytes(jpg)) = mosaic_guard.composite(
                        MosaicUpload::DirectRgb {
                            planes: rgb_planes,
                            width,
                            height,
                        },
                        method,
                        MosaicOutput::RgbStyledJpeg {
                            jpeg: &mut jpeg_guard,
                            quality,
                            rescale_min: *rescale_min,
                            rescale_max: *rescale_max,
                        },
                    )
                {
                    return Some(MosaicCompositeResult::Bytes(Bytes::from(jpg)));
                }
                let GpuMosaicResult::Bytes(rgba) = mosaic_guard
                    .composite(
                        MosaicUpload::DirectRgb {
                            planes: rgb_planes,
                            width,
                            height,
                        },
                        method,
                        MosaicOutput::RgbStyled {
                            rescale_min: *rescale_min,
                            rescale_max: *rescale_max,
                        },
                    )
                    .ok()?
                else {
                    return None;
                };
                drop(mosaic_guard);

                let bytes = if *format == OutputFormat::RawRgba {
                    Bytes::from(rgba)
                } else {
                    crate::stats::encode_rgba_output(
                        width as usize,
                        height as usize,
                        &rgba,
                        *format,
                        *return_mask,
                    )
                };
                Some(MosaicCompositeResult::Bytes(bytes))
            }
            _ => None,
        }
    }

    /// Attempt GPU-accelerated raw `f32` single-band resampling for `.npy`, `.tif`, `.f32`,
    /// and multi-scene mosaic scene fetches across `Separable`, `Grid2D`, and `Exact2D`.
    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_single_band_f32(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        coords: BboxCoordSource<'_>,
        src: &BandSource<'_>,
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample_on(resample, &coords)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let (active, _guard) = self.enter_in_flight();
        let (slab, base_col, base_row) = self.gate_and_resolve_single_band(
            source_slab_key(cog_id, ifd_idx, band0, src),
            src,
            gpu_mode.to_policy_op(),
            dst_w * dst_h,
            coord_upload_bytes,
            dst_w * dst_h * 4,
            active,
        )?;
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
    fn try_render_bbox_single_band_stats(
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
        let gpu_mode = map_resample_on(resample, &coords)?;
        let (dst_w, dst_h, _) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || dst_w.saturating_mul(dst_h) < self.fused_stats_min_pixels() {
            return None;
        }
        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = key.and_then(|k| self.slab_cache.get(&k));
        let (_active, _guard) = self.enter_in_flight();
        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        self.submit_fused_stats(
            CoalescedJob::Colormap {
                slab,
                plan,
                mode: gpu_mode,
                rescale_min: 0.0,
                rescale_max: 1.0,
                lut_u32: [0u32; 256],
                output: CoalescedOutput::RawF32,
            },
            dst_w * dst_h,
            num_bins,
            hist_range,
            percentiles,
        )
    }

    /// 1-pass fused GPU expression evaluation + band statistics/histogram/percentiles
    /// (`CoalescedJob::WithStats` — zero `f32` readback, zero re-upload, 12 KB readback).
    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_expr_stats(
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
    fn try_render_bbox_spectral_stats(
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
        let gpu_mode = map_resample_on(resample, &coords)?;
        let (dst_w, dst_h, _) = coords.dims_and_upload_bytes()?;
        if dst_w == 0
            || dst_h == 0
            || srcs.is_empty()
            || srcs.len() > 8
            || dst_w.saturating_mul(dst_h) < self.fused_stats_min_pixels()
        {
            return None;
        }
        let warm_slab = cache_key.and_then(|k| self.multi_slab_cache.get(&k));
        let (_active, _guard) = self.enter_in_flight();
        let (slab, base_col, base_row) = self.resolve_multiband_slab(cache_key, warm_slab, srcs)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        self.submit_fused_stats(
            CoalescedJob::Expr {
                slab,
                plan,
                mode: gpu_mode,
                instrs,
                rescale_min: 0.0,
                rescale_max: 1.0,
                lut_u32: [0u32; 256],
                output: CoalescedOutput::RawF32,
            },
            dst_w * dst_h,
            num_bins,
            hist_range,
            percentiles,
        )
    }

    /// 1-pass fused GPU DEM Horn 3×3 (`Hillshade` / `Slope` / `Aspect`) + band statistics/histogram/percentiles
    /// (`CoalescedJob::WithStats` — zero `f32` readback, zero re-upload, 12 KB readback).
    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_dem_stats(
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
        let gpu_mode = map_resample_on(resample, &coords)?;
        if alg.is_terrain_rgb_encoding() {
            return None;
        }
        let gpu_alg = alg.gpu_mode().map(|_| alg)?;
        let (dst_w, dst_h, _) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || dst_w.saturating_mul(dst_h) < self.fused_stats_min_pixels() {
            return None;
        }
        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = key.and_then(|k| self.slab_cache.get(&k));
        let (_active, _guard) = self.enter_in_flight();
        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let plan = to_owned_plan(coords, base_col as f32, base_row as f32);
        self.submit_fused_stats(
            CoalescedJob::Terrain {
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
            },
            dst_w * dst_h,
            num_bins,
            hist_range,
            percentiles,
        )
    }

    /// Attempt GPU-accelerated raw `f32` DEM Horn 3×3 (`Hillshade` / `Slope` / `Aspect`)
    /// evaluation for `.npy`, `.tif`, `.f32`, and mosaic routes.
    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_dem_f32(
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
        let gpu_mode = map_resample_on(resample, &coords)?;
        if alg.is_terrain_rgb_encoding() {
            return None;
        }
        let gpu_alg = alg.gpu_mode().map(|_| alg)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::DemHornLanczos3,
            _ => GpuOp::DemHornCubic,
        };
        let (active, _guard) = self.enter_in_flight();
        let (slab, base_col, base_row) = self.gate_and_resolve_single_band(
            source_slab_key(cog_id, ifd_idx, band0, src),
            src,
            op,
            dst_w * dst_h,
            coord_upload_bytes,
            dst_w * dst_h * 4,
            active,
        )?;
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
    fn try_render_bbox_expr_f32(
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
        let gpu_mode = map_resample_on(resample, &coords)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || srcs.is_empty() || srcs.len() > 8 {
            return None;
        }
        let (active, _guard) = self.enter_in_flight();
        let (slab, base_col, base_row) = self.gate_and_resolve_multiband(
            cache_key,
            srcs,
            multiband_policy_op(gpu_mode),
            dst_w * dst_h,
            coord_upload_bytes,
            dst_w * dst_h * 4,
            active,
        )?;
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
        // The batch shares one zoom: the first tile's axes give its scale.
        let (u0, v0) = tile_uvs[0];
        let gpu_mode = map_resample_on(style.resample, &BboxCoordSource::Separable { u: u0, v: v0 })?;
        let out_kind = match style.format {
            OutputFormat::Png => GpuPixelOutput::IndexedU8,
            OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
                GpuPixelOutput::Rgba8
            }
            _ => return None,
        };

        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = key.and_then(|k| self.slab_cache.get(&k));
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
            in_flight: self.in_flight.load(Ordering::Relaxed) / 32,
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
        let lut_u32 = pack_rgb_lut(&lut.rgb);
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
        let gpu_mode = map_resample_on(style.resample, &coords)?;
        let gpu_formula = Self::translate_color_formula(color_formula)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || !is_rgba_format(style.format) {
            return None;
        }
        let (slab, base_col, base_row) = self.gate_and_resolve_multiband(
            cache_key,
            srcs,
            multiband_policy_op(gpu_mode),
            dst_w * dst_h,
            coord_upload_bytes,
            dst_w * dst_h * 4,
            active,
        )?;
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
        let coords = BboxCoordSource::from_coord_plan(plan);
        let cache_key = Self::multiband_key(cog_id, ifd_idx, &bands_1based, srcs);
        let (active, _guard) = self.enter_in_flight();
        let (job, dst_w, dst_h) =
            self.prepare_bbox_sar_rgb(cache_key, coords, srcs, style, None, active)?;
        self.submit_rgba_async(job, dst_w, dst_h, style.format, true)
            .await
    }

    /// Attempt GPU-accelerated Dual-Pol SAR False-Color RGB (with optional fused `ColorFormula`
    /// and `GpuMultiBandCache`) for `/cog/preview`, `/cog/bbox`, `/cog/feature`, and tiles.
    #[allow(clippy::too_many_arguments)]
    fn try_render_bbox_sar_rgb(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>; 2],
        style: &RgbStyle,
        color_formula: Option<&ColorFormula>,
        return_mask: bool,
    ) -> Option<Bytes> {
        let (active, _guard) = self.enter_in_flight();
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
    fn try_render_bbox_expr_rgb(
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
        let gpu_mode = map_resample_on(style.resample, &coords)?;
        let gpu_formula = Self::translate_color_formula(color_formula)?;
        let (instrs, lens) = Self::compile_expr_rgb_to_gpu(progs, unique_bands)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || srcs.len() != unique_bands.len() || !is_rgba_format(style.format) {
            return None;
        }
        let (active, _guard) = self.enter_in_flight();
        let (slab, base_col, base_row) = self.gate_and_resolve_multiband(
            cache_key,
            srcs,
            multiband_policy_op(gpu_mode),
            dst_w * dst_h,
            coord_upload_bytes,
            dst_w * dst_h * 4,
            active,
        )?;
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
    fn try_render_bbox_multiband_f32(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        srcs: &[BandSource<'_>],
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample_on(resample, &coords)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        let nb = srcs.len();
        if dst_w == 0 || dst_h == 0 || nb == 0 || nb > 8 {
            return None;
        }
        let (active, _guard) = self.enter_in_flight();
        let (slab, base_col, base_row) = self.gate_and_resolve_multiband(
            cache_key,
            srcs,
            gpu_mode.to_policy_op(),
            dst_w * dst_h * nb,
            coord_upload_bytes,
            dst_w * dst_h * nb * 4,
            active,
        )?;
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
    fn try_render_bbox_expr_rgb_f32(
        &self,
        cache_key: Option<MultiBandSlabKey>,
        coords: BboxCoordSource<'_>,
        unique_bands: &[u16],
        srcs: &[BandSource<'_>],
        progs: &[ExprProgram; 3],
        resample: ResampleMode,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample_on(resample, &coords)?;
        let (instrs, lens) = Self::compile_expr_rgb_to_gpu(progs, unique_bands)?;
        let (dst_w, dst_h, coord_upload_bytes) = coords.dims_and_upload_bytes()?;
        if dst_w == 0 || dst_h == 0 || srcs.len() != unique_bands.len() {
            return None;
        }
        let (active, _guard) = self.enter_in_flight();
        let (slab, base_col, base_row) = self.gate_and_resolve_multiband(
            cache_key,
            srcs,
            multiband_policy_op(gpu_mode),
            dst_w * dst_h * 3,
            coord_upload_bytes,
            dst_w * dst_h * 3 * 4,
            active,
        )?;
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
        if let Some(rgba) = gpu.try_render_bbox_single_band_with_lut(
            99,
            0,
            0,
            BboxCoordSource::Separable {
                u: &u_coords,
                v: &v_coords,
            },
            &src,
            &style,
            &style.colormap.lut().rgb,
            true,
        ) {
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

        // All 16 scalar opcodes unlocked in #435 must translate to GPU VM bytecode
        for expr in [
            "b1 % b2",
            "b1 % 4",
            "(b1 & b2) | (~b1)",
            "log1p(b1) + expm1(b2) + sign(b1 - b2)",
            "arcsin(b1) + arccos(b2) + arctan(b1) + arctan2(b1, b2)",
            "sinh(b1) + cosh(b2) + tanh(b1) + trunc(b2)",
        ] {
            let p = ExprProgram::compile(expr).unwrap_or_else(|e| panic!("compile {expr}: {e}"));
            assert!(
                GpuBackend::compile_expr_to_gpu(&p).is_some(),
                "expected {expr} to compile to GPU VM"
            );
        }

        // Unsupported multi-band reductions must cleanly return None for CPU fallback
        for expr in ["median(b1, b2, b3)", "std(b1, b2, b3)", "var(b1, b2, b3)"] {
            let p = ExprProgram::compile(expr).unwrap();
            assert!(GpuBackend::compile_expr_to_gpu(&p).is_none());
        }
    }

    #[test]
    fn gpu_expr_vm_scalar_opcodes_match_cpu_numerically() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            eprintln!("skipping GPU expr parity test (no adapter)");
            return;
        };
        let w = 256u32;
        let h = 256u32;
        let n = (w * h) as usize;
        let mut b1_vals = vec![0.0f32; n];
        let mut b2_vals = vec![0.0f32; n];
        for i in 0..n {
            // Mix positive, negative, fractional, and [-1, 1] values
            b1_vals[i] = -0.8 + (i as f32 / n as f32) * 1.6;
            b2_vals[i] = 0.25 + (i as f32 / n as f32) * 3.5;
        }
        // Inject edge-case pixels: NaN, 0.0, -1.0, out-of-domain ±1.5, small ±1e-5.
        let edge_b1 = [f32::NAN, 0.0, -1.0, 1.5, -1.5, 1e-5, -1e-5, 0.0];
        let edge_b2 = [1.0, 0.0, 2.0, f32::NAN, -2.5, 1e-5, -1e-5, -1.0];
        for (idx, (&x, &y)) in edge_b1.iter().zip(edge_b2.iter()).enumerate() {
            b1_vals[idx] = x;
            b2_vals[idx] = y;
        }
        let slab = Arc::new(
            cogzilla_gpu::GpuMultiBandSlab::upload_bands(
                &gpu.ctx.device,
                &[b1_vals.as_slice(), b2_vals.as_slice()],
                w,
                h,
                w,
            )
            .unwrap(),
        );

        let u: Vec<f32> = (0..w).map(|x| x as f32).collect();
        let v: Vec<f32> = (0..h).map(|y| y as f32).collect();

        for expr in [
            "b2 % 3.0",
            "b1 % 3.0",
            "b2 % 4",
            "(floor(b2 * 10) & 7) | 1",
            "~where(b1 > 0, 1, 0)",
            "log1p(b2)",
            "log1p(b1)",
            "expm1(b1)",
            "sign(b1)",
            "arcsin(b1)",
            "arccos(b1)",
            "arctan(b1)",
            "arctan2(b1, b2)",
            "arctan2(b1, b1)",
            "sinh(b1)",
            "cosh(b1)",
            "tanh(b1)",
            "trunc(b2 * 3.0 - 5.0)",
        ] {
            let prog = ExprProgram::compile(expr).unwrap();
            let mut instrs = GpuBackend::compile_expr_to_gpu(&prog).unwrap();
            // Map prog.used_bands slots directly into our 2-band slab (b1 -> 0, b2 -> 1)
            for ins in &mut instrs {
                if ins.opcode == Op::LoadBand(0).wire_opcode().unwrap() {
                    ins.imm_u32 = (prog.used_bands[ins.imm_u32 as usize] - 1) as u32;
                }
            }
            let gpu_bytes = gpu
                .coalescer
                .submit(CoalescedJob::Expr {
                    slab: Arc::clone(&slab),
                    plan: OwnedCoordPlan::Separable1D {
                        u: u.clone(),
                        v: v.clone(),
                    },
                    mode: map_resample(ResampleMode::Nearest).unwrap(),
                    instrs,
                    rescale_min: 0.0,
                    rescale_max: 1.0,
                    lut_u32: [0u32; 256],
                    output: CoalescedOutput::RawF32,
                })
                .unwrap_or_else(|e| panic!("GPU failed to render {expr}: {e}"));
            let gpu_f32: Vec<f32> = gpu_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let mut max_err = 0.0f32;
            for i in 0..n {
                let slot_vals: Vec<f32> = prog
                    .used_bands
                    .iter()
                    .map(|&bidx| if bidx == 1 { b1_vals[i] } else { b2_vals[i] })
                    .collect();
                let cpu_v = prog.eval(&slot_vals);
                let gpu_v = gpu_f32[i];
                if cpu_v.is_nan() {
                    assert!(gpu_v.is_nan(), "{expr} px {i}: cpu=NaN gpu={gpu_v}");
                } else {
                    max_err = max_err.max((cpu_v - gpu_v).abs());
                }
            }
            assert!(
                max_err < 1e-4,
                "{expr}: CPU vs GPU max_err={max_err} exceeded 1e-4"
            );
        }
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
        if let Some(rgba) = gpu.try_render_bbox_single_band_with_lut(
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
            &style.colormap.lut().rgb,
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

        if let Some(gpu_med) = gpu
            .try_composite_mosaic(&MosaicCompositeRequest {
                width: w,
                height: h,
                input: MosaicCompositeInput::Single(&refs),
                method: GpuPixelSelection::Median,
                output: MosaicCompositeOutput::F32,
            })
            .and_then(MosaicCompositeResult::into_f32)
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

        if let Some(png) = gpu
            .try_composite_mosaic(&MosaicCompositeRequest {
                width: w,
                height: h,
                input: MosaicCompositeInput::Single(&refs),
                method: GpuPixelSelection::Median,
                output: MosaicCompositeOutput::Styled {
                    rescale: (50.0, 250.0),
                    colormap: ColormapName::Viridis,
                    format: OutputFormat::Png,
                    return_mask: true,
                },
            })
            .and_then(MosaicCompositeResult::into_bytes)
        {
            assert!(png.starts_with(b"\x89PNG"), "fused mosaic styled output must be valid PNG");
        }

        // 1-pass fused Mosaic -> GPU JPEG
        if let Some(jpg) = gpu
            .try_composite_mosaic(&MosaicCompositeRequest {
                width: w,
                height: h,
                input: MosaicCompositeInput::Single(&refs),
                method: GpuPixelSelection::Median,
                output: MosaicCompositeOutput::Styled {
                    rescale: (50.0, 250.0),
                    colormap: ColormapName::Viridis,
                    format: OutputFormat::Jpeg { quality: 85 },
                    return_mask: false,
                },
            })
            .and_then(MosaicCompositeResult::into_bytes)
        {
            assert!(
                jpg.starts_with(&[0xff, 0xd8]) && jpg.ends_with(&[0xff, 0xd9]),
                "fused mosaic styled JPEG must be valid JFIF"
            );
        }

        // 1-submit batched 3-band RGB median must be bit-exact vs per-band GPU median
        let band_planes: Vec<Vec<&[f32]>> = vec![refs.clone(), refs.clone(), refs.clone()];
        if let (Some(single), Some(batched)) = (
            gpu.try_composite_mosaic(&MosaicCompositeRequest {
                width: w,
                height: h,
                input: MosaicCompositeInput::Single(&refs),
                method: GpuPixelSelection::Median,
                output: MosaicCompositeOutput::F32,
            })
            .and_then(MosaicCompositeResult::into_f32),
            gpu.try_composite_mosaic(&MosaicCompositeRequest {
                width: w,
                height: h,
                input: MosaicCompositeInput::Bands(&band_planes),
                method: GpuPixelSelection::Median,
                output: MosaicCompositeOutput::F32,
            })
            .and_then(MosaicCompositeResult::into_f32_bands),
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

        // 1-pass fused 3-band RGB Mosaic -> GPU JPEG & RawRgba
        let rgb_planes: [&[&[f32]]; 3] = [&refs, &refs, &refs];
        if let Some(rgb_jpg) = gpu
            .try_composite_mosaic(&MosaicCompositeRequest {
                width: w,
                height: h,
                input: MosaicCompositeInput::Rgb(&rgb_planes),
                method: GpuPixelSelection::Median,
                output: MosaicCompositeOutput::RgbStyled {
                    rescale_min: [50.0, 50.0, 50.0],
                    rescale_max: [250.0, 250.0, 250.0],
                    format: OutputFormat::Jpeg { quality: 85 },
                    return_mask: false,
                },
            })
            .and_then(MosaicCompositeResult::into_bytes)
        {
            assert!(
                rgb_jpg.starts_with(&[0xff, 0xd8]) && rgb_jpg.ends_with(&[0xff, 0xd9]),
                "fused RGB mosaic JPEG must be valid JFIF"
            );
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

        // 2. DEM Slope (anisotropic dx != dy), Hillshade (including negative altitude), and Aspect CPU/GPU parity
        let dem_cases: &[(DemAlgorithm, f32, f32, f32)] = &[
            (DemAlgorithm::Slope { z_factor: 1.5 }, 30.0, 45.0, 2e-2),
            (
                DemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 1.0,
                },
                1.0,
                1.0,
                1.0,
            ),
            (
                DemAlgorithm::Hillshade {
                    azimuth_deg: 90.0,
                    altitude_deg: -15.0,
                    z_factor: 1.0,
                },
                1.0,
                1.0,
                1.0,
            ),
            (DemAlgorithm::Aspect, 30.0, 45.0, 5e-2),
        ];
        for &(alg, dx, dy, tol) in dem_cases {
            if let Some(raw) = gpu.try_render_bbox_dem_f32(
                500,
                0,
                0,
                BboxCoordSource::Separable {
                    u: &u_coords,
                    v: &v_coords,
                },
                &s0,
                alg,
                dx,
                dy,
                ResampleMode::Bilinear,
            ) {
                assert_eq!(raw.len(), n * n * 4);
                let sample = |ix: usize, iy: usize| {
                    cogzilla_kernel::sample_general_scaled(
                        &s0,
                        u_coords[ix],
                        v_coords[iy],
                        1.0,
                        1.0,
                        ResampleMode::Bilinear,
                    )
                };
                let mut max_err = 0.0f32;
                for py in (4..n - 4).step_by(29) {
                    for px in (4..n - 4).step_by(29) {
                        let off = (py * n + px) * 4;
                        let gv = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                        let cv = cogzilla_kernel::eval_dem_stencil(
                            sample(px, py - 1),
                            sample(px - 1, py),
                            sample(px, py),
                            sample(px + 1, py),
                            sample(px, py + 1),
                            1.0 / dx,
                            1.0 / dy,
                            alg,
                        );
                        if cv.is_nan() {
                            assert!(gv.is_nan(), "{alg:?} ({px},{py}): expected NaN, got {gv}");
                        } else {
                            let mut err = (gv - cv).abs();
                            if matches!(alg, DemAlgorithm::Aspect) && err > 180.0 {
                                err = 360.0 - err;
                            }
                            max_err = max_err.max(err);
                        }
                    }
                }
                assert!(max_err <= tol, "DEM {alg:?} CPU/GPU max err {max_err} > {tol}");
            }
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
                    .try_render_bbox_single_band_with_lut(
                        4242,
                        0,
                        0,
                        BboxCoordSource::Separable {
                            u: &u_vecs[idx],
                            v: &v_vecs[idx],
                        },
                        &src,
                        &style,
                        &style.colormap.lut().rgb,
                        true,
                    )
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
            lut: Box::new(cogzilla_kernel::ColormapLut::new(
                std::array::from_fn(|i| u32::from_le_bytes(entries[i])),
                std::array::from_fn(|i| [entries[i][0], entries[i][1], entries[i][2]]),
            )),
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

    #[test]
    fn resolve_styled_rgb_lut_applies_formula_before_builtin_colormap() {
        let cf = ColorFormula::parse("Gamma RGB 2.0").expect("parse gamma");
        let gpu_lut =
            GpuBackend::resolve_styled_rgb_lut(ColormapName::Viridis, None, Some(&cf)).unwrap();
        let viridis = ColormapName::Viridis.lut().rgb;
        for i in 0..256 {
            let (mapped_idx, _, _) = cf.apply_rgb(i as u8, i as u8, i as u8);
            assert_eq!(
                gpu_lut[i], viridis[mapped_idx as usize],
                "mismatch at byte {i}: color_formula must apply before built-in colormap lookup"
            );
        }
    }

    #[test]
    fn source_slab_key_bypasses_cache_on_unscale_nodata_or_huge_grid() {
        let blocks: Vec<Option<&cogzilla_codec::DecodedBand>> = vec![None];
        let base = BandSource::new(256, 256, 256, 256, 0, 0, 1, 1, &blocks);
        assert!(source_slab_key(42, 0, 0, &base).is_some());

        let with_nd =
            BandSource::new(256, 256, 256, 256, 0, 0, 1, 1, &blocks).with_nodata_override(Some(0.0));
        assert!(source_slab_key(42, 0, 0, &with_nd).is_none());

        let with_unscale =
            BandSource::new(256, 256, 256, 256, 0, 0, 1, 1, &blocks).with_scale_offset(0.001, 10.0);
        assert!(source_slab_key(42, 0, 0, &with_unscale).is_none());

        let huge_grid = BandSource::new(256, 2048, 256, 8, 0, 0, 1, 256, &blocks);
        assert!(source_slab_key(42, 0, 0, &huge_grid).is_none());
    }

    #[test]
    fn encode_indexed_u8_png_emits_l_and_la_for_grayscale_lut() {
        let mut gray = [[0u8; 3]; 256];
        for (i, c) in gray.iter_mut().enumerate() {
            *c = [i as u8; 3];
        }
        // Opaque grayscale -> PNG color_type 0 (L)
        let opaque = encode_indexed_u8_png(&[1, 128, 200, 255], 2, 2, &gray);
        assert_eq!(&opaque[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(opaque[25], 0, "opaque grayscale PNG must use color_type 0 (L)");

        // Masked grayscale -> PNG color_type 4 (LA)
        let masked = encode_indexed_u8_png(&[0, 128, 200, 255], 2, 2, &gray);
        assert_eq!(masked[25], 4, "masked grayscale PNG must use color_type 4 (LA)");

        // Colored LUT -> PNG color_type 3 (Indexed P)
        let viridis = ColormapName::Viridis.lut().rgb;
        let colored = encode_indexed_u8_png(&[0, 128, 200, 255], 2, 2, &viridis);
        assert_eq!(colored[25], 3, "colored LUT PNG must use color_type 3 (Indexed P)");
    }

    #[tokio::test]
    async fn derived_terrain_and_dem_with_no_source_id_match_cpu_and_bypass_slab_cache() {
        let Some(gpu) = GpuBackend::try_init(64 * 1024 * 1024) else {
            return;
        };
        const N: usize = 256;
        let mut slab = cogzilla_codec::AlignedF32Slab::new_uninit(N * N);
        for y in 0..N {
            for x in 0..N {
                slab.as_mut_slice()[y * N + x] = 100.0 + (x as f32) * 3.5 + (y as f32) * 1.25;
            }
        }
        let band = cogzilla_codec::DecodedBand {
            width: N as u32,
            height: N as u32,
            data: cogzilla_codec::BandData::F32(slab),
            stats: cogzilla_codec::BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 100.0,
                max: 1400.0,
            },
        };
        let blocks = [Some(&band)];
        let src = BandSource::new(N as u32, N as u32, N as u32, N as u32, 0, 0, 1, 1, &blocks);
        let mut u = Box::new([0.0f32; N]);
        let mut v = Box::new([0.0f32; N]);
        for i in 0..N {
            u[i] = i as f32;
            v[i] = i as f32;
        }
        let plan = CoordPlan::<N>::Separable1D {
            u,
            v,
            src_min_col: 0,
            src_max_col: (N - 1) as i32,
            src_min_row: 0,
            src_max_row: (N - 1) as i32,
            all_in_bounds: true,
        };
        let before = gpu.slab_cache.stats();
        let gpu_terr = gpu
            .try_render_terrain::<N>(
                None,
                0,
                0,
                &plan,
                &src,
                DemAlgorithm::Terrarium,
                ResampleMode::Lanczos,
                OutputFormat::RawRgba,
            )
            .await
            .expect("GPU must execute derived terrain tile");
        let cpu_terr = cogzilla_kernel::render_terrain_tile::<N>(
            &plan,
            &src,
            DemAlgorithm::Terrarium,
            ResampleMode::Lanczos,
            OutputFormat::RawRgba,
        );
        assert_eq!(
            gpu_terr.as_ref(),
            cpu_terr.as_ref(),
            "derived terrain GPU output must match CPU"
        );

        let style = SingleBandStyle {
            rescale_min: 0.0,
            rescale_max: 255.0,
            colormap: ColormapName::Grayscale,
            resample: ResampleMode::Lanczos,
            format: OutputFormat::RawRgba,
        };
        let gpu_dem = gpu
            .try_render_dem::<N>(
                None,
                0,
                0,
                &plan,
                &src,
                DemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 1.0,
                },
                30.0,
                &style,
            )
            .await
            .expect("GPU must execute derived DEM tile");
        let cpu_dem = cogzilla_kernel::render_dem_algorithm::<N>(
            &plan,
            &src,
            DemAlgorithm::Hillshade {
                azimuth_deg: 315.0,
                altitude_deg: 45.0,
                z_factor: 1.0,
            },
            30.0,
            &style,
        );
        assert_eq!(gpu_dem.len(), cpu_dem.len());
        let max_diff = gpu_dem
            .iter()
            .zip(cpu_dem.iter())
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        assert!(max_diff <= 1, "derived DEM GPU output must match CPU within 1 ULP (got {max_diff})");

        let after = gpu.slab_cache.stats();
        assert_eq!(
            after.entries, before.entries,
            "source_id = None must not insert derived planes into slab_cache"
        );
    }
}
