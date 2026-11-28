//! Zero-cost CPU-only stub for `crate::gpu_backend` compiled when `--no-default-features`
//! disables the `gpu` Cargo feature (#458).
//!
//! Because `GpuBackend::try_init` always returns `None` and every method is `#[inline(always)]`
//! returning `None`, LLVM dead-code-eliminates every GPU attempt branch in CPU-only builds while
//! keeping `engine.rs`, `titiler.rs`, `zonal.rs`, `http.rs`, and `http/metrics.rs` completely
//! free of `#[cfg(feature = "gpu")]` attributes.

#![allow(dead_code, unused_variables)]

use std::sync::Arc;

use bytes::Bytes;
use cogzilla_kernel::{
    BandSource, ColorFormula, ColormapName, CustomColormap, DemAlgorithm, ExprProgram,
    OutputFormat, ResampleMode, RgbStyle, SingleBandStyle, SpectralExpr,
};
use cogzilla_proj::CoordPlan;

/// Stub GPU cache statistics returned by [`StubSlabCache::stats`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StubCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub used_bytes: usize,
    pub max_bytes: usize,
}

/// Stub GPU slab cache (never populated in CPU-only builds).
#[derive(Debug, Default)]
pub struct StubSlabCache;

impl StubSlabCache {
    #[inline(always)]
    pub fn clear(&self) {}

    #[inline(always)]
    pub fn stats(&self) -> StubCacheStats {
        StubCacheStats::default()
    }
}

/// Shared plain-data GPU types re-exported directly from `cogzilla-raster`
/// (identical across GPU and CPU-only builds; cannot drift).
pub use cogzilla_raster::{GpuBandStats, GpuPixelSelection, GpuWeightedBandStats};

/// Stub matching `cogzilla_gpu::GpuInstr` (16-byte POD instruction; field layout pinned by
/// a compile-time assertion in `gpu_backend.rs`).
#[derive(Clone, Copy, Debug)]
pub struct GpuInstr {
    pub opcode: u32,
    pub imm_u32: u32,
    pub imm_f32: f32,
    pub _pad: u32,
}

/// Stub re-export matching `cogzilla_gpu::MultiBandSlabKey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MultiBandSlabKey;

/// Stub matching `gpu_backend::BboxCoordSource`.
#[derive(Clone, Copy)]
pub enum BboxCoordSource<'a> {
    Separable {
        u: &'a [f32],
        v: &'a [f32],
    },
    Grid2D {
        dst_w: usize,
        dst_h: usize,
        grid_u: &'a [f32; 289],
        grid_v: &'a [f32; 289],
    },
    Exact2D {
        dst_w: usize,
        dst_h: usize,
        u_map: &'a [f32],
        v_map: &'a [f32],
    },
    SeparableMasked {
        u: &'a [f32],
        v: &'a [f32],
        mask_bits: &'a [u32],
    },
    Grid2DMasked {
        dst_w: usize,
        dst_h: usize,
        grid_u: &'a [f32; 289],
        grid_v: &'a [f32; 289],
        mask_bits: &'a [u32],
    },
    Exact2DMasked {
        dst_w: usize,
        dst_h: usize,
        u_map: &'a [f32],
        v_map: &'a [f32],
        mask_bits: &'a [u32],
    },
}

/// Stub matching `gpu_backend::GpuBboxBands`.
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

/// Stub matching `gpu_backend::GpuBboxOutput`.
pub enum GpuBboxOutput<'a> {
    StyledLut {
        style: &'a SingleBandStyle,
        rgb_lut: &'a [[u8; 3]; 256],
        return_mask: bool,
    },
    StyledRgb {
        style: &'a RgbStyle,
        color_formula: Option<&'a ColorFormula>,
        return_mask: bool,
    },
    StyledTerrain {
        resample: ResampleMode,
        format: OutputFormat,
        return_mask: bool,
    },
    F32 { resample: ResampleMode },
    Stats {
        resample: ResampleMode,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &'a [f64],
    },
}

/// Stub matching `gpu_backend::GpuBboxResult`.
pub enum GpuBboxResult {
    Bytes(Bytes),
    Stats(GpuBandStats),
}

impl GpuBboxResult {
    #[inline(always)]
    pub fn into_bytes(self) -> Option<Bytes> {
        match self {
            Self::Bytes(b) => Some(b),
            Self::Stats(_) => None,
        }
    }

    #[inline(always)]
    pub fn into_stats(self) -> Option<GpuBandStats> {
        match self {
            Self::Stats(s) => Some(s),
            Self::Bytes(_) => None,
        }
    }
}

/// Stub matching `gpu_backend::MosaicCompositeInput`.
pub enum MosaicCompositeInput<'a> {
    Single(&'a [&'a [f32]]),
    Bands(&'a [Vec<&'a [f32]>]),
    Rgb(&'a [&'a [&'a [f32]]; 3]),
}

/// Stub matching `gpu_backend::MosaicCompositeOutput`.
pub enum MosaicCompositeOutput {
    F32,
    Styled {
        rescale: (f32, f32),
        colormap: ColormapName,
        format: OutputFormat,
        return_mask: bool,
    },
    RgbStyled {
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
        format: OutputFormat,
        return_mask: bool,
    },
}

/// Stub matching `gpu_backend::MosaicCompositeRequest`.
pub struct MosaicCompositeRequest<'a> {
    pub width: u32,
    pub height: u32,
    pub input: MosaicCompositeInput<'a>,
    pub method: GpuPixelSelection,
    pub output: MosaicCompositeOutput,
}

/// Stub matching `gpu_backend::MosaicCompositeResult`.
pub enum MosaicCompositeResult {
    F32(Vec<f32>),
    F32Bands(Vec<Vec<f32>>),
    Bytes(Bytes),
}

impl MosaicCompositeResult {
    #[inline(always)]
    pub fn into_f32(self) -> Option<Vec<f32>> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn into_f32_bands(self) -> Option<Vec<Vec<f32>>> {
        match self {
            Self::F32Bands(v) => Some(v),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn into_bytes(self) -> Option<Bytes> {
        match self {
            Self::Bytes(b) => Some(b),
            _ => None,
        }
    }
}

/// CPU-only stub for `GpuBackend`: [`Self::try_init`] always returns `None`.
pub struct GpuBackend {
    pub slab_cache: StubSlabCache,
    pub multi_slab_cache: StubSlabCache,
}

impl GpuBackend {
    #[inline(always)]
    pub fn try_init(_slab_cache_bytes: usize) -> Option<Arc<Self>> {
        None
    }

    #[inline(always)]
    pub fn in_flight(&self) -> usize {
        0
    }

    #[inline(always)]
    pub fn resolve_styled_rgb_lut(
        _colormap: ColormapName,
        _custom_lut: Option<&CustomColormap>,
        _color_formula: Option<&ColorFormula>,
    ) -> Option<[[u8; 3]; 256]> {
        None
    }

    #[inline(always)]
    pub fn compile_expr_to_gpu(_prog: &ExprProgram) -> Option<Vec<GpuInstr>> {
        None
    }

    #[inline(always)]
    pub fn multiband_key(
        _cog_id: u64,
        _ifd_idx: usize,
        _bands_1based: &[u16],
        _srcs: &[BandSource<'_>],
    ) -> Option<MultiBandSlabKey> {
        None
    }

    #[inline(always)]
    pub fn encode_rgba_output(
        &self,
        width: usize,
        height: usize,
        rgba: &[u8],
        format: OutputFormat,
        return_mask: bool,
    ) -> Bytes {
        crate::stats::encode_rgba_output(width, height, rgba, format, return_mask)
    }

    #[inline(always)]
    pub fn try_render_bbox(
        &self,
        _coords: BboxCoordSource<'_>,
        _bands: &GpuBboxBands<'_>,
        _out: &GpuBboxOutput<'_>,
    ) -> Option<GpuBboxResult> {
        None
    }

    #[inline(always)]
    pub fn try_composite_mosaic(
        &self,
        _req: &MosaicCompositeRequest<'_>,
    ) -> Option<MosaicCompositeResult> {
        None
    }

    #[inline(always)]
    pub fn try_render_single_band_batch(
        &self,
        _cog_id: u64,
        _ifd_idx: usize,
        _band0: u16,
        _tile_size: usize,
        _tile_uvs: &[(&[f32], &[f32])],
        _src: &BandSource<'_>,
        _style: &SingleBandStyle,
    ) -> Option<Vec<Bytes>> {
        None
    }

    #[inline(always)]
    pub fn try_reduce_f32_stats(
        &self,
        _pixels: &[f32],
        _nodata: Option<f32>,
        _num_bins: usize,
        _hist_range: Option<(f64, f64)>,
        _percentiles: &[f64],
    ) -> Option<GpuBandStats> {
        None
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_reduce_weighted_zonal_stats(
        &self,
        _bands: &[&[f32]],
        _covs: &[f32],
        _win_w: usize,
        _row_areas: &[f32],
        _area_weighted: bool,
        _cov_eps: f32,
        _pixel_set_eps: f32,
        _num_bins: usize,
        _hist_range: Option<(f64, f64)>,
        _percentiles: &[f64],
    ) -> Option<Vec<GpuWeightedBandStats>> {
        None
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_reduce_weighted_zonal_expr_stats(
        &self,
        _raw_bands: &[&[f32]],
        _exprs: &[(&[GpuInstr], u32)],
        _covs: &[f32],
        _win_w: usize,
        _row_areas: &[f32],
        _area_weighted: bool,
        _cov_eps: f32,
        _pixel_set_eps: f32,
        _num_bins: usize,
        _hist_range: Option<(f64, f64)>,
        _percentiles: &[f64],
    ) -> Option<Vec<GpuWeightedBandStats>> {
        None
    }

    #[inline(always)]
    pub async fn try_render_single_band<const N: usize>(
        &self,
        _cog_id: u64,
        _ifd_idx: usize,
        _band0: u16,
        _tx0: u32,
        _ty0: u32,
        _plan: &CoordPlan<N>,
        _src: &BandSource<'_>,
        _style: &SingleBandStyle,
    ) -> Option<Bytes> {
        None
    }

    #[inline(always)]
    pub async fn try_render_dem<const N: usize>(
        &self,
        _source_id: Option<(u64, usize, u16)>,
        _tx0: u32,
        _ty0: u32,
        _plan: &CoordPlan<N>,
        _src: &BandSource<'_>,
        _alg: DemAlgorithm,
        _pixel_size_m: f32,
        _style: &SingleBandStyle,
    ) -> Option<Bytes> {
        None
    }

    #[inline(always)]
    pub async fn try_render_terrain<const N: usize>(
        &self,
        _source_id: Option<(u64, usize, u16)>,
        _tx0: u32,
        _ty0: u32,
        _plan: &CoordPlan<N>,
        _src: &BandSource<'_>,
        _algorithm: DemAlgorithm,
        _resample: ResampleMode,
        _format: OutputFormat,
    ) -> Option<Bytes> {
        None
    }

    #[inline(always)]
    pub async fn try_render_spectral<const N: usize>(
        &self,
        _cog_id: u64,
        _ifd_idx: usize,
        _bands_1based: &[u16],
        _plan: &CoordPlan<N>,
        _srcs: &[BandSource<'_>],
        _expr: SpectralExpr,
        _style: &SingleBandStyle,
    ) -> Option<Bytes> {
        None
    }

    #[inline(always)]
    pub async fn try_render_expr<const N: usize>(
        &self,
        _cog_id: u64,
        _ifd_idx: usize,
        _plan: &CoordPlan<N>,
        _srcs: &[BandSource<'_>],
        _prog: &ExprProgram,
        _style: &SingleBandStyle,
    ) -> Option<Bytes> {
        None
    }

    #[inline(always)]
    pub async fn try_render_rgb<const N: usize>(
        &self,
        _cog_id: u64,
        _ifd_idx: usize,
        _bands_1based: [u16; 3],
        _plan: &CoordPlan<N>,
        _srcs: &[BandSource<'_>; 3],
        _style: &RgbStyle,
    ) -> Option<Bytes> {
        None
    }

    #[inline(always)]
    pub async fn try_render_sar_rgb<const N: usize>(
        &self,
        _cog_id: u64,
        _ifd_idx: usize,
        _bands_1based: [u16; 2],
        _plan: &CoordPlan<N>,
        _srcs: &[BandSource<'_>; 2],
        _style: &RgbStyle,
    ) -> Option<Bytes> {
        None
    }
}
