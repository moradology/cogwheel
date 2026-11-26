use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use cogzilla_io::{CogStore, IoError, OpenedCog};
use cogzilla_kernel::{
    BandSource, ColormapName, DemAlgorithm, ExprProgram, OutputFormat, ResampleMode, RgbStyle,
    SingleBandStyle, SpectralExpr, TerrainEncoding, empty_tile_bytes, encode_rgba_tile,
    render_bandmath1, render_bandmath2, render_bandmath3, render_dem_algorithm, render_expr_vm,
    render_rgb_bands, render_sar_rgb, render_single_band, render_terrain_tile,
};
use cogzilla_proj::{
    CoordPlan, CrsModel, ProjError, TileCoord, mercator_to_lonlat_deg, tile_bounds_3857,
};
use cogzilla_tiff::{Compression, IfdMeta, PlanarConfig, SampleType};
use dashmap::DashMap;
use thiserror::Error;

use crate::catalog::{MosaicJsonCatalog, SceneCatalog};
use crate::tile_cache::TileCache;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("I/O error: {0}")]
    Io(#[from] IoError),
    #[error("projection error: {0}")]
    Proj(#[from] ProjError),
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("payload too large: {0}")]
    PayloadTooLarge(String),
    #[error("request timeout: {0}")]
    RequestTimeout(String),
    #[error("dataset not found: {0}")]
    NotFound(String),
    #[error("upstream error: {0}")]
    BadGateway(String),
    #[error("no content")]
    NoContent,
    #[error("method not allowed")]
    MethodNotAllowed(&'static str),
    #[error("{0}")]
    Shared(Arc<EngineError>),
}

#[derive(Debug, Clone)]
pub enum BandRequest {
    Single {
        band_1based: u16,
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    Rgb {
        bands_1based: [u16; 3],
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
    },
    Spectral1 {
        expr: SpectralExpr,
        band_1based: u16,
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    SarRgb {
        bands_1based: [u16; 2],
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
    },
    Spectral2 {
        expr: SpectralExpr,
        bands_1based: [u16; 2],
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    Spectral3 {
        expr: SpectralExpr,
        bands_1based: [u16; 3],
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    CustomExpr {
        prog: ExprProgram,
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    CustomExprRgb {
        progs: [ExprProgram; 3],
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
    },
    Terrain {
        band_1based: u16,
        encoding: TerrainEncoding,
    },
    Dem {
        band_1based: u16,
        algorithm: DemAlgorithm,
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    /// Arbitrary N-band selection (e.g. 2-band or 4..=15-band) for scientific `.tif`/`.npy`/`.f32` output.
    MultiBand {
        bands_1based: Vec<u16>,
    },
}

#[derive(Debug, Clone)]
pub struct TileRequest {
    pub tile: TileCoord,
    pub tile_size: u32,
    pub bands: BandRequest,
    pub resample: ResampleMode,
    pub format: OutputFormat,
    pub unscale: bool,
}

/// Returns `true` when `?unscale=true` must be applied per-sample in `BandSource` (rather than
/// being zero-cost folded into `Single`/`Rgb` rescale bounds): raw float output (`.npy`), DEM
/// algorithms, terrain encodings, spectral indices, SAR dB, and custom band-math expressions.
#[inline]
pub(crate) fn needs_source_unscale(bands: &BandRequest, format: OutputFormat) -> bool {
    matches!(format, OutputFormat::RawF32 | OutputFormat::Npy)
        || !matches!(bands, BandRequest::Single { .. } | BandRequest::Rgb { .. })
}

#[derive(Default)]
pub struct EngineStats {
    pub tiles_served: AtomicU64,
    pub empty_short_circuits: AtomicU64,
    pub passthrough_hits: AtomicU64,
    pub http_304_not_modified: AtomicU64,
    pub gpu_dispatches: AtomicU64,
}

/// Map a fetch/I/O error on an asset referenced by a STAC item or MosaicJSON document to
/// `502 Bad Gateway` naming the asset (while preserving `403 Forbidden` for allowlist rejections
/// and `400 Bad Request` for invalid URIs / corrupt TIFFs).
pub fn asset_fetch_error(asset_uri: &str, err: IoError) -> EngineError {
    match err.root() {
        IoError::Forbidden(_)
        | IoError::InvalidUri(_)
        | IoError::Tiff(_)
        | IoError::SourceChanged { .. } => EngineError::Io(err),
        _ => {
            let safe = cogzilla_io::redact_http_url(asset_uri);
            EngineError::BadGateway(format!(
                "referenced asset '{safe}' could not be fetched: {err}"
            ))
        }
    }
}

pub struct TileEngine {
    pub store: Arc<CogStore>,
    crs_cache: DashMap<u64, CrsModel, ahash::RandomState>,
    scenes: DashMap<String, Vec<Arc<OpenedCog>>, ahash::RandomState>,
    catalogs: DashMap<String, Arc<dyn SceneCatalog>, ahash::RandomState>,
    pub stac_cache: DashMap<String, Arc<crate::stac::StacItem>, ahash::RandomState>,
    pub mosaicjson_cache: DashMap<String, Arc<MosaicJsonCatalog>, ahash::RandomState>,
    pub tile_cache: Arc<TileCache>,
    tile_cache_epoch: AtomicU64,
    pub stats: Arc<EngineStats>,
    #[cfg(feature = "gpu")]
    pub gpu: Option<Arc<crate::gpu_backend::GpuBackend>>,
}

fn affines_match(a: &cogzilla_tiff::Affine2D, b: &cogzilla_tiff::Affine2D) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-6 * (1.0 + x.abs().max(y.abs()));
    close(a.a, b.a)
        && close(a.b, b.b)
        && close(a.c, b.c)
        && close(a.d, b.d)
        && close(a.e, b.e)
        && close(a.f, b.f)
}

/// Check whether two `GeoKeys` describe the same CRS (falling back to full equality for
/// user-defined GeoTIFF codes `32767`).
fn geo_keys_match(a: &cogzilla_tiff::GeoKeys, b: &cogzilla_tiff::GeoKeys) -> bool {
    if a.model_type != b.model_type
        || a.projected_cs_type != b.projected_cs_type
        || a.geographic_type != b.geographic_type
    {
        return false;
    }
    if a.projected_cs_type.is_none()
        || a.projected_cs_type == Some(32767)
        || a.geographic_type == Some(32767)
    {
        return a == b;
    }
    true
}

/// Resampling kernel support margin `(lo, hi)` in source pixels relative to
/// `[floor(min), floor(max)]`.
pub(crate) fn resample_fetch_margin(resample: ResampleMode, is_dem: bool) -> (i32, i32) {
    let (lo, hi) = match resample {
        ResampleMode::Nearest => (0, 0),
        ResampleMode::Bilinear => (0, 1),
        ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Gauss => (1, 2),
        ResampleMode::Lanczos => (2, 3),
        // Footprint modes (Average/Mode/Rms/Min/Max/Med/Q1/Q3/Sum) use exact footprint bounds in
        // `render_bbox_impl` and 1-px neighborhood padding on tile routes.
        _ => (1, 2),
    };
    if is_dem {
        (lo + 2, hi + 2)
    } else {
        (lo, hi)
    }
}

/// Fast check whether all COGs in a multi-COG / STAC composite share the exact same base
/// dimensions, tile grid, overview count, CRS, and geotransform across all IFDs (allowing the
/// single-grid fast path).
pub(crate) fn cogs_share_base_grid(cogs: &[Arc<OpenedCog>]) -> bool {
    if cogs.len() <= 1 {
        return true;
    }
    let p_gk = &cogs[0].index.geo_keys;
    let n_ifds = cogs[0].index.ifds.len();
    cogs.iter().skip(1).all(|c| {
        if c.index.ifds.len() != n_ifds || !geo_keys_match(&c.index.geo_keys, p_gk) {
            return false;
        }
        (0..n_ifds).all(|idx| {
            let p = &cogs[0].index.ifds[idx];
            let o = &c.index.ifds[idx];
            o.width == p.width
                && o.height == p.height
                && o.tile_width == p.tile_width
                && o.tile_height == p.tile_height
                && o.tiles_across == p.tiles_across
                && o.tiles_down == p.tiles_down
                && affines_match(&o.pixel_to_crs, &p.pixel_to_crs)
        })
    })
}

/// Verify that all secondary COGs in a multi-COG / STAC composite share the exact same overview
/// index, image dimensions, internal tile grid, CRS, and geotransform as `cogs[0]` so `BandSource`
/// never reads OOB or misregisters pixels.
pub(crate) fn validate_multi_cog_grid(
    cogs: &[Arc<OpenedCog>],
    ifd_idx: usize,
) -> Result<(), EngineError> {
    if cogs.len() <= 1 {
        return Ok(());
    }
    let primary_ifd = &cogs[0].index.ifds[ifd_idx];
    let primary_gk = &cogs[0].index.geo_keys;
    for (k, c) in cogs.iter().enumerate().skip(1) {
        if !geo_keys_match(&c.index.geo_keys, primary_gk) {
            return Err(EngineError::BadRequest(format!(
                "multi-COG asset #{k} CRS does not match primary asset"
            )));
        }
        let Some(other) = c.index.ifds.get(ifd_idx) else {
            return Err(EngineError::BadRequest(format!(
                "multi-COG asset #{k} has {} overviews, fewer than primary asset (need IFD {ifd_idx})",
                c.index.ifds.len()
            )));
        };
        if other.width != primary_ifd.width
            || other.height != primary_ifd.height
            || other.tile_width != primary_ifd.tile_width
            || other.tile_height != primary_ifd.tile_height
            || other.tiles_across != primary_ifd.tiles_across
            || other.tiles_down != primary_ifd.tiles_down
            || !affines_match(&other.pixel_to_crs, &primary_ifd.pixel_to_crs)
        {
            return Err(EngineError::BadRequest(format!(
                "multi-COG asset #{k} grid ({}x{}, tile {}x{}) does not match primary ({}x{}, tile {}x{})",
                other.width,
                other.height,
                other.tile_width,
                other.tile_height,
                primary_ifd.width,
                primary_ifd.height,
                primary_ifd.tile_width,
                primary_ifd.tile_height
            )));
        }
    }
    Ok(())
}

impl TileEngine {
    pub fn new(cache_max_bytes: usize) -> Self {
        Self::with_store(Arc::new(CogStore::new(cache_max_bytes)))
    }

    pub fn with_store(store: Arc<CogStore>) -> Self {
        Self::with_store_and_tile_cache(store, 0)
    }

    pub fn with_store_and_tile_cache(store: Arc<CogStore>, tile_cache_max_bytes: usize) -> Self {
        Self {
            store,
            crs_cache: DashMap::with_hasher(ahash::RandomState::new()),
            scenes: DashMap::with_hasher(ahash::RandomState::new()),
            catalogs: DashMap::with_hasher(ahash::RandomState::new()),
            stac_cache: DashMap::with_hasher(ahash::RandomState::new()),
            mosaicjson_cache: DashMap::with_hasher(ahash::RandomState::new()),
            tile_cache: Arc::new(TileCache::new(tile_cache_max_bytes)),
            tile_cache_epoch: AtomicU64::new(0),
            stats: Arc::new(EngineStats::default()),
            #[cfg(feature = "gpu")]
            gpu: crate::gpu_backend::GpuBackend::try_init(256 * 1024 * 1024),
        }
    }

    /// Flush the rendered tile cache if any underlying source was invalidated or replaced.
    /// Costs a single relaxed/acquire atomic load on the warm path.
    #[inline(always)]
    pub fn sync_tile_cache_epoch(&self) {
        let cur = self.store.invalidation_epoch.load(Ordering::Acquire);
        let seen = self.tile_cache_epoch.load(Ordering::Relaxed);
        if cur != seen && self.tile_cache_epoch.swap(cur, Ordering::AcqRel) != cur {
            self.tile_cache.clear();
            #[cfg(feature = "gpu")]
            if let Some(gpu) = &self.gpu {
                gpu.slab_cache.clear();
                gpu.multi_slab_cache.clear();
            }
        }
    }

    /// Register a pluggable spatial catalog (`SceneCatalog`) for `/mosaic/{name}/tiles`.
    pub fn register_catalog(&self, name: impl Into<String>, catalog: Arc<dyn SceneCatalog>) {
        self.catalogs.insert(name.into(), catalog);
    }

    pub fn get_catalog(&self, name: &str) -> Option<Arc<dyn SceneCatalog>> {
        self.catalogs.get(name).map(|v| v.clone())
    }

    pub fn catalog_count(&self) -> usize {
        self.catalogs.len()
    }

    /// Pre-register a multi-COG satellite scene (e.g., Landsat `[B4, B3, B2]` or SAR `[VV, VH]`).
    pub fn register_local_scene<P: AsRef<Path>>(
        &self,
        name: impl Into<String>,
        paths: &[P],
    ) -> Result<Vec<Arc<OpenedCog>>, EngineError> {
        let name_str = name.into();
        let mut cogs = Vec::with_capacity(paths.len());
        for (i, p) in paths.iter().enumerate() {
            let key = format!("{name_str}__band{i}");
            let cog = self.store.register_local(key, p)?;
            let _ = self.get_or_build_crs(&cog)?;
            cogs.push(cog);
        }
        self.scenes.insert(name_str, cogs.clone());
        Ok(cogs)
    }

    pub fn get_scene(&self, name: &str) -> Option<Vec<Arc<OpenedCog>>> {
        self.scenes.get(name).map(|v| v.clone())
    }

    pub fn get_or_build_crs(&self, cog: &OpenedCog) -> Result<CrsModel, EngineError> {
        if !cog.index.is_georeferenced {
            return Err(EngineError::BadRequest(
                "dataset has no georeferencing (missing GeoTransform/GCPs); geographic operations require a georeferenced raster".into(),
            ));
        }
        if let Some(c) = self.crs_cache.get(&cog.id) {
            return Ok(c.clone());
        }
        let crs = CrsModel::from_geo_keys(&cog.index.geo_keys)?;
        if self.crs_cache.len() >= 8192 {
            self.crs_cache.clear();
        }
        self.crs_cache.insert(cog.id, crs.clone());
        Ok(crs)
    }

    /// Render a single XYZ tile for an already-opened COG.
    pub async fn render_tile(
        &self,
        cog: &OpenedCog,
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        self.stats.tiles_served.fetch_add(1, Ordering::Relaxed);
        if req.format == OutputFormat::Npy {
            let mut raw_req = req.clone();
            raw_req.format = OutputFormat::RawF32;
            let raw = match raw_req.tile_size {
                256 => self.render_tile_sized::<256>(cog, &raw_req).await?,
                512 => self.render_tile_sized::<512>(cog, &raw_req).await?,
                other => {
                    return Err(EngineError::BadRequest(format!(
                        "unsupported tile_size={other}, expected 256 or 512"
                    )));
                }
            };
            let nb = match req.bands {
                BandRequest::Rgb { .. }
                | BandRequest::SarRgb { .. }
                | BandRequest::Terrain { .. }
                | BandRequest::CustomExprRgb { .. } => 3,
                BandRequest::MultiBand { ref bands_1based } => bands_1based.len().max(1),
                _ => 1,
            };
            return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                raw,
                nb,
                req.tile_size,
                req.tile_size,
                true,
            ));
        }
        match req.tile_size {
            256 => self.render_tile_sized::<256>(cog, req).await,
            512 => self.render_tile_sized::<512>(cog, req).await,
            other => Err(EngineError::BadRequest(format!(
                "unsupported tile_size={other}, expected 256 or 512"
            ))),
        }
    }

    /// Render a batch of XYZ tiles for a single COG (e.g. a map viewport burst or pyramid
    /// cache-warming pass).
    ///
    /// When built with `--features gpu` and a hardware GPU is active, eligible single-band
    /// separable tiles that share the same `(ifd_idx, band, tile-block neighborhood, style)` are
    /// fused into a single 3D GPU micro-batch (`GpuTileBatcher`), amortizing the Metal/Vulkan
    /// dispatch floor across all `K` tiles. Any remaining tiles fall back transparently to
    /// `render_tile`.
    pub async fn render_tiles_batch(
        &self,
        cog: &OpenedCog,
        reqs: &[TileRequest],
    ) -> Vec<Result<Bytes, EngineError>> {
        if reqs.is_empty() {
            return Vec::new();
        }
        #[allow(unused_mut)]
        let mut results: Vec<Option<Result<Bytes, EngineError>>> =
            (0..reqs.len()).map(|_| None).collect();

        #[cfg(feature = "gpu")]
        if reqs.len() >= 2
            && let Some(ref gpu) = self.gpu
            && let Ok(crs) = self.get_or_build_crs(cog)
        {
            #[derive(Hash, PartialEq, Eq)]
            struct BatchKey {
                ifd_idx: usize,
                band0: u16,
                tx0: u32,
                ty0: u32,
                grid_w: u32,
                grid_h: u32,
                tile_size: u32,
                resample: u8,
                rmin_bits: u32,
                rmax_bits: u32,
                colormap: String,
                format_tag: u8,
                format_q: u32,
            }
            struct BatchItem {
                req_idx: usize,
                u: Vec<f32>,
                v: Vec<f32>,
            }
            let mut groups: std::collections::HashMap<BatchKey, (SingleBandStyle, Vec<BatchItem>)> =
                std::collections::HashMap::new();

            for (idx, req) in reqs.iter().enumerate() {
                if req.unscale {
                    continue;
                }
                let BandRequest::Single {
                    band_1based,
                    rescale,
                    colormap,
                } = req.bands
                else {
                    continue;
                };
                if req.tile_size != 256 && req.tile_size != 512 {
                    continue;
                }
                let (format_tag, format_q) = match req.format {
                    OutputFormat::Png => (0u8, 0u32),
                    OutputFormat::RawRgba => (1u8, 0u32),
                    OutputFormat::Webp { quality } => {
                        (2u8, quality.map(f32::to_bits).unwrap_or(0))
                    }
                    OutputFormat::Jpeg { quality } => (3u8, quality as u32),
                    _ => continue,
                };
                let (target_rx, target_ry) = crs.target_resolution_for_tile(req.tile, req.tile_size);
                let ifd_idx = cog.index.select_ifd(target_rx, target_ry);
                let ifd = &cog.index.ifds[ifd_idx];
                let band0 = band_1based.saturating_sub(1);
                if band0 >= ifd.samples_per_pixel {
                    continue;
                }
                let pad = match req.resample {
                    ResampleMode::Nearest => 0i64,
                    ResampleMode::Bilinear => 1i64,
                    ResampleMode::Cubic => 2i64,
                    _ => 3i64,
                };
                let uv_bounds: Option<(Vec<f32>, Vec<f32>, f32, f32, f32, f32)> =
                    if req.tile_size == 256 {
                        CoordPlan::<256>::build_with_gcp(
                            &crs,
                            &ifd.crs_to_pixel,
                            ifd.gcp_poly.as_deref(),
                            ifd.width,
                            ifd.height,
                            req.tile,
                        )
                        .and_then(|p| match p {
                            CoordPlan::Separable1D { u, v, .. } => {
                                let (mut u0, mut u1) = (f32::INFINITY, f32::NEG_INFINITY);
                                let (mut v0, mut v1) = (f32::INFINITY, f32::NEG_INFINITY);
                                for &x in u.iter() {
                                    u0 = u0.min(x);
                                    u1 = u1.max(x);
                                }
                                for &y in v.iter() {
                                    v0 = v0.min(y);
                                    v1 = v1.max(y);
                                }
                                Some((u.to_vec(), v.to_vec(), u0, u1, v0, v1))
                            }
                            _ => None,
                        })
                    } else {
                        CoordPlan::<512>::build_with_gcp(
                            &crs,
                            &ifd.crs_to_pixel,
                            ifd.gcp_poly.as_deref(),
                            ifd.width,
                            ifd.height,
                            req.tile,
                        )
                        .and_then(|p| match p {
                            CoordPlan::Separable1D { u, v, .. } => {
                                let (mut u0, mut u1) = (f32::INFINITY, f32::NEG_INFINITY);
                                let (mut v0, mut v1) = (f32::INFINITY, f32::NEG_INFINITY);
                                for &x in u.iter() {
                                    u0 = u0.min(x);
                                    u1 = u1.max(x);
                                }
                                for &y in v.iter() {
                                    v0 = v0.min(y);
                                    v1 = v1.max(y);
                                }
                                Some((u.to_vec(), v.to_vec(), u0, u1, v0, v1))
                            }
                            _ => None,
                        })
                    };
                let Some((u_vec, v_vec, min_u, max_u, min_v, max_v)) = uv_bounds else {
                    continue;
                };
                if !min_u.is_finite() || !min_v.is_finite() {
                    continue;
                }
                let px0 = (min_u.floor() as i64 - pad).clamp(0, ifd.width.saturating_sub(1) as i64) as u32;
                let py0 = (min_v.floor() as i64 - pad).clamp(0, ifd.height.saturating_sub(1) as i64) as u32;
                let px1 = (max_u.ceil() as i64 + pad).clamp(0, ifd.width.saturating_sub(1) as i64) as u32;
                let py1 = (max_v.ceil() as i64 + pad).clamp(0, ifd.height.saturating_sub(1) as i64) as u32;
                let tx0 = px0 / ifd.tile_width;
                let ty0 = py0 / ifd.tile_height;
                let tx1 = (px1 / ifd.tile_width).min(ifd.tiles_across.saturating_sub(1));
                let ty1 = (py1 / ifd.tile_height).min(ifd.tiles_down.saturating_sub(1));
                let grid_w = tx1 - tx0 + 1;
                let grid_h = ty1 - ty0 + 1;

                let key = BatchKey {
                    ifd_idx,
                    band0,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    tile_size: req.tile_size,
                    resample: req.resample as u8,
                    rmin_bits: rescale.0.to_bits(),
                    rmax_bits: rescale.1.to_bits(),
                    colormap: format!("{colormap:?}"),
                    format_tag,
                    format_q,
                };
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap,
                    resample: req.resample,
                    format: req.format,
                };
                groups
                    .entry(key)
                    .or_insert_with(|| (style, Vec::new()))
                    .1
                    .push(BatchItem {
                        req_idx: idx,
                        u: u_vec,
                        v: v_vec,
                    });
            }

            for (key, (style, items)) in groups {
                if items.len() < 2 {
                    continue;
                }
                let ifd = &cog.index.ifds[key.ifd_idx];
                let mut indices = Vec::with_capacity((key.grid_w * key.grid_h) as usize);
                for ty in key.ty0..key.ty0 + key.grid_h {
                    for tx in key.tx0..key.tx0 + key.grid_w {
                        indices.push(Some(ifd.tile_index(tx, ty, key.band0) as u32));
                    }
                }
                let Ok(tiles) = self.store.get_blocks(cog, key.ifd_idx, &indices).await else {
                    continue;
                };
                let band_in_decoded = match ifd.planar_config {
                    PlanarConfig::Chunky => key.band0 as usize,
                    PlanarConfig::Planar => 0,
                };
                let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                    .iter()
                    .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                    .collect();
                let src = BandSource::new(
                    ifd.width,
                    ifd.height,
                    ifd.tile_width,
                    ifd.tile_height,
                    key.tx0,
                    key.ty0,
                    key.grid_w,
                    key.grid_h,
                    &block_refs,
                );
                let uvs: Vec<(&[f32], &[f32])> = items
                    .iter()
                    .map(|it| (it.u.as_slice(), it.v.as_slice()))
                    .collect();
                if let Some(batch_out) = gpu.try_render_single_band_batch(
                    cog.id,
                    key.ifd_idx,
                    key.band0,
                    key.tile_size as usize,
                    &uvs,
                    &src,
                    &style,
                ) {
                    self.stats
                        .tiles_served
                        .fetch_add(items.len() as u64, Ordering::Relaxed);
                    self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                    for (item, bytes) in items.into_iter().zip(batch_out.into_iter()) {
                        results[item.req_idx] = Some(Ok(bytes));
                    }
                }
            }
        }

        let mut final_out = Vec::with_capacity(reqs.len());
        for (slot, req) in results.into_iter().zip(reqs.iter()) {
            match slot {
                Some(res) => final_out.push(res),
                None => final_out.push(self.render_tile(cog, req).await),
            }
        }
        final_out
    }

    async fn render_tile_sized<const N: usize>(
        &self,
        cog: &OpenedCog,
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        let crs = self.get_or_build_crs(cog)?;
        let (target_rx, target_ry) = crs.target_resolution_for_tile(req.tile, N as u32);
        let ifd_idx = cog.index.select_ifd(target_rx, target_ry);
        let ifd = &cog.index.ifds[ifd_idx];

        // Zero-decode pass-through for WebMercator-aligned JPEG/WebP COGs:
        // when the requested XYZ tile matches an internal IFD tile 1:1 with identity RGB
        // styling and matching compression, stream raw compressed bytes straight through
        // without decoding, resampling, or re-encoding.
        if let Some(bytes) = self
            .try_zero_decode_passthrough::<N>(cog, &crs, ifd_idx, ifd, req)
            .await?
        {
            return Ok(bytes);
        }

        let Some(plan) = CoordPlan::<N>::build_with_gcp(
            &crs,
            &ifd.crs_to_pixel,
            ifd.gcp_poly.as_deref(),
            ifd.width,
            ifd.height,
            req.tile,
        ) else {
            self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
            let nb = match req.bands {
                BandRequest::Rgb { .. }
                | BandRequest::SarRgb { .. }
                | BandRequest::Terrain { .. }
                | BandRequest::CustomExprRgb { .. } => 3,
                BandRequest::MultiBand { ref bands_1based } => bands_1based.len().max(1),
                _ => 1,
            };
            return Ok(empty_tile_bytes(N as u32, req.format, nb));
        };

        let (pad_lo, pad_hi) =
            resample_fetch_margin(req.resample, matches!(req.bands, BandRequest::Dem { .. }));
        let (tx0, tx1, ty0, ty1) = if pad_lo > 0 || pad_hi > 1 {
            // CoordPlan::src_pixel_bounds() is the exact bilinear footprint [floor(min), floor(max)+1].
            let extra_lo = pad_lo;
            let extra_hi = (pad_hi - 1).max(0);
            let (c0, c1, r0, r1) = plan.src_pixel_bounds();
            let d_tx0 = (c0.saturating_sub(extra_lo).max(0) as u32) / ifd.tile_width;
            let d_tx1 = ((c1.saturating_add(extra_hi).max(0) as u32) / ifd.tile_width)
                .min(ifd.tiles_across.saturating_sub(1));
            let d_ty0 = (r0.saturating_sub(extra_lo).max(0) as u32) / ifd.tile_height;
            let d_ty1 = ((r1.saturating_add(extra_hi).max(0) as u32) / ifd.tile_height)
                .min(ifd.tiles_down.saturating_sub(1));
            (d_tx0, d_tx1, d_ty0, d_ty1)
        } else {
            plan.src_tile_range(ifd.tile_width, ifd.tile_height, ifd.tiles_across, ifd.tiles_down)
        };
        let grid_w = tx1 - tx0 + 1;
        let grid_h = ty1 - ty0 + 1;
        let grid_len = (grid_w * grid_h) as usize;

        match &req.bands {
            BandRequest::Single {
                band_1based,
                rescale,
                colormap,
            } => {
                let band0 = band_1based.saturating_sub(1);
                if band0 >= ifd.samples_per_pixel {
                    return Err(EngineError::BadRequest(format!(
                        "band {} out of range (dataset has {} bands)",
                        band_1based, ifd.samples_per_pixel
                    )));
                }
                let mut indices = Vec::with_capacity(grid_len);
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        let idx = ifd.tile_index(tx, ty, band0) as u32;
                        indices.push(Some(idx));
                    }
                }
                let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                let band_in_decoded = match ifd.planar_config {
                    PlanarConfig::Chunky => band0 as usize,
                    PlanarConfig::Planar => 0,
                };
                let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                    .iter()
                    .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                    .collect();

                let apply_unscale = req.unscale && needs_source_unscale(&req.bands, req.format);
                let (sc, off) = if apply_unscale {
                    (
                        cog.index.scales.get(band0 as usize).copied().unwrap_or(1.0),
                        cog.index.offsets.get(band0 as usize).copied().unwrap_or(0.0),
                    )
                } else {
                    (1.0, 0.0)
                };
                let src = BandSource::new(
                    ifd.width,
                    ifd.height,
                    ifd.tile_width,
                    ifd.tile_height,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    &block_refs,
                )
                .with_scale_offset(sc, off);
                if src.all_nodata {
                    self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                }
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                #[cfg(feature = "gpu")]
                if let Some(gpu) = &self.gpu {
                    if let Some(bytes) = gpu.try_render_single_band::<N>(
                        cog.id, ifd_idx, band0, tx0, ty0, &plan, &src, &style,
                    ) {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                }
                Ok(render_single_band::<N>(&plan, &src, &style))
            }
            BandRequest::Terrain {
                band_1based,
                encoding,
            } => {
                let band0 = band_1based.saturating_sub(1);
                if band0 >= ifd.samples_per_pixel {
                    return Err(EngineError::BadRequest(format!(
                        "band {} out of range (dataset has {} bands)",
                        band_1based, ifd.samples_per_pixel
                    )));
                }
                let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        let idx = ifd.tile_index(tx, ty, band0) as u32;
                        indices.push(Some(idx));
                    }
                }
                let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                let band_in_decoded = match ifd.planar_config {
                    PlanarConfig::Chunky => band0 as usize,
                    PlanarConfig::Planar => 0,
                };
                let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                    .iter()
                    .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                    .collect();
                let (sc, off) = if req.unscale {
                    (
                        cog.index.scales.get(band0 as usize).copied().unwrap_or(1.0),
                        cog.index.offsets.get(band0 as usize).copied().unwrap_or(0.0),
                    )
                } else {
                    (1.0, 0.0)
                };
                let src = BandSource::new(
                    ifd.width,
                    ifd.height,
                    ifd.tile_width,
                    ifd.tile_height,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    &block_refs,
                )
                .with_scale_offset(sc, off);
                if src.all_nodata {
                    self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                }
                #[cfg(feature = "gpu")]
                if let Some(gpu) = &self.gpu {
                    if let Some(bytes) = gpu.try_render_terrain::<N>(
                        cog.id,
                        ifd_idx,
                        band0,
                        tx0,
                        ty0,
                        &plan,
                        &src,
                        *encoding,
                        req.resample,
                        req.format,
                    ) {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                }
                Ok(render_terrain_tile::<N>(
                    &plan,
                    &src,
                    *encoding,
                    req.resample,
                    req.format,
                ))
            }
            BandRequest::Dem {
                band_1based,
                algorithm,
                rescale,
                colormap,
            } => {
                let band0 = band_1based.saturating_sub(1);
                if band0 >= ifd.samples_per_pixel {
                    return Err(EngineError::BadRequest(format!(
                        "band {} out of range (dataset has {} bands)",
                        band_1based, ifd.samples_per_pixel
                    )));
                }
                let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        let idx = ifd.tile_index(tx, ty, band0) as u32;
                        indices.push(Some(idx));
                    }
                }
                let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                let band_in_decoded = match ifd.planar_config {
                    PlanarConfig::Chunky => band0 as usize,
                    PlanarConfig::Planar => 0,
                };
                let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                    .iter()
                    .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                    .collect();
                let (sc, off) = if req.unscale {
                    (
                        cog.index.scales.get(band0 as usize).copied().unwrap_or(1.0),
                        cog.index.offsets.get(band0 as usize).copied().unwrap_or(0.0),
                    )
                } else {
                    (1.0, 0.0)
                };
                let src = BandSource::new(
                    ifd.width,
                    ifd.height,
                    ifd.tile_width,
                    ifd.tile_height,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    &block_refs,
                )
                .with_scale_offset(sc, off);
                if src.all_nodata {
                    self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                }
                // Ground pixel spacing in meters at the tile's center latitude (correcting WebMercator sec(lat)).
                let (bx0, by0, bx1, by1) = tile_bounds_3857(req.tile);
                let (_, lat_deg) = mercator_to_lonlat_deg((bx0 + bx1) * 0.5, (by0 + by1) * 0.5);
                let pixel_size_m =
                    (((bx1 - bx0) / N as f64) * lat_deg.to_radians().cos().max(0.05)) as f32;
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                #[cfg(feature = "gpu")]
                if let Some(gpu) = &self.gpu {
                    if let Some(bytes) = gpu.try_render_dem::<N>(
                        cog.id,
                        ifd_idx,
                        band0,
                        tx0,
                        ty0,
                        &plan,
                        &src,
                        *algorithm,
                        pixel_size_m,
                        &style,
                    ) {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                }
                Ok(render_dem_algorithm::<N>(
                    &plan,
                    &src,
                    *algorithm,
                    pixel_size_m,
                    &style,
                ))
            }
            BandRequest::Spectral1 {
                expr,
                band_1based,
                rescale,
                colormap,
            } => {
                let band0 = band_1based.saturating_sub(1);
                if band0 >= ifd.samples_per_pixel {
                    return Err(EngineError::BadRequest(format!(
                        "band {} out of range (dataset has {} bands)",
                        band_1based, ifd.samples_per_pixel
                    )));
                }
                let mut indices = Vec::with_capacity(grid_len);
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        indices.push(Some(ifd.tile_index(tx, ty, band0) as u32));
                    }
                }
                let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                let band_in_decoded = match ifd.planar_config {
                    PlanarConfig::Chunky => band0 as usize,
                    PlanarConfig::Planar => 0,
                };
                let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                    .iter()
                    .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                    .collect();
                let (sc, off) = if req.unscale {
                    (
                        cog.index.scales.get(band0 as usize).copied().unwrap_or(1.0),
                        cog.index.offsets.get(band0 as usize).copied().unwrap_or(0.0),
                    )
                } else {
                    (1.0, 0.0)
                };
                let src = BandSource::new(
                    ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                    grid_h, &block_refs,
                )
                .with_scale_offset(sc, off);
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                #[cfg(feature = "gpu")]
                if let Some(gpu) = &self.gpu {
                    if let Some(bytes) = gpu.try_render_spectral::<N>(
                        cog.id,
                        ifd_idx,
                        &[*band_1based],
                        &plan,
                        std::slice::from_ref(&src),
                        *expr,
                        &style,
                    ) {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                }
                Ok(render_bandmath1::<N>(&plan, &src, *expr, &style))
            }
            BandRequest::SarRgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let b0 = [
                    bands_1based[0].saturating_sub(1),
                    bands_1based[1].saturating_sub(1),
                ];
                for &b in &b0 {
                    if b >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            b + 1,
                            ifd.samples_per_pixel
                        )));
                    }
                }
                let style = RgbStyle {
                    rescale_min: *rescale_min,
                    rescale_max: *rescale_max,
                    resample: req.resample,
                    format: req.format,
                };
                let so = |b: u16| -> (f32, f32) {
                    if req.unscale {
                        (
                            cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                            cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                        )
                    } else {
                        (1.0, 0.0)
                    }
                };
                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let a_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[0] as usize)))
                            .collect();
                        let b_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[1] as usize)))
                            .collect();
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s1, o1),
                        ];
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_sar_rgb::<N>(
                                cog.id,
                                ifd_idx,
                                *bands_1based,
                                &plan,
                                &srcs,
                                &style,
                            ) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_sar_rgb::<N>(&plan, &srcs, &style))
                    }
                    PlanarConfig::Planar => {
                        let mut indices = Vec::with_capacity(grid_len * 2);
                        for &band in &b0 {
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                                }
                            }
                        }
                        let all_tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let a_refs: Vec<_> = all_tiles[0..grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let b_refs: Vec<_> = all_tiles[grid_len..grid_len * 2]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s1, o1),
                        ];
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_sar_rgb::<N>(
                                cog.id,
                                ifd_idx,
                                *bands_1based,
                                &plan,
                                &srcs,
                                &style,
                            ) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_sar_rgb::<N>(&plan, &srcs, &style))
                    }
                }
            }
            BandRequest::Rgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let b0 = [
                    bands_1based[0].saturating_sub(1),
                    bands_1based[1].saturating_sub(1),
                    bands_1based[2].saturating_sub(1),
                ];
                for &b in &b0 {
                    if b >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            b + 1,
                            ifd.samples_per_pixel
                        )));
                    }
                }

                let style = RgbStyle {
                    rescale_min: *rescale_min,
                    rescale_max: *rescale_max,
                    resample: req.resample,
                    format: req.format,
                };

                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        // All 3 bands live inside the exact same decoded tiles!
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let r_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[0] as usize)))
                            .collect();
                        let g_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[1] as usize)))
                            .collect();
                        let b_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[2] as usize)))
                            .collect();

                        let apply_unscale =
                            req.unscale && needs_source_unscale(&req.bands, req.format);
                        let so = |b: u16| -> (f32, f32) {
                            if apply_unscale {
                                (
                                    cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                                    cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                                )
                            } else {
                                (1.0, 0.0)
                            }
                        };
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let (s2, o2) = so(b0[2]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            )
                            .with_scale_offset(s1, o1),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s2, o2),
                        ];
                        if srcs[0].all_nodata || srcs[1].all_nodata || srcs[2].all_nodata {
                            self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                        }
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_rgb::<N>(cog.id, ifd_idx, *bands_1based, &plan, &srcs, &style) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_rgb_bands::<N>(&plan, &srcs, &style))
                    }
                    PlanarConfig::Planar => {
                        // Fetch all 3 planes in one coalesced call
                        let mut indices = Vec::with_capacity(grid_len * 3);
                        for &band in &b0 {
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                                }
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let r_refs: Vec<_> = tiles[0..grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let g_refs: Vec<_> = tiles[grid_len..2 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let b_refs: Vec<_> = tiles[2 * grid_len..3 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();

                        let apply_unscale =
                            req.unscale && needs_source_unscale(&req.bands, req.format);
                        let so = |b: u16| -> (f32, f32) {
                            if apply_unscale {
                                (
                                    cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                                    cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                                )
                            } else {
                                (1.0, 0.0)
                            }
                        };
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let (s2, o2) = so(b0[2]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            )
                            .with_scale_offset(s1, o1),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s2, o2),
                        ];
                        if srcs[0].all_nodata || srcs[1].all_nodata || srcs[2].all_nodata {
                            self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                        }
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_rgb::<N>(cog.id, ifd_idx, *bands_1based, &plan, &srcs, &style) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_rgb_bands::<N>(&plan, &srcs, &style))
                    }
                }
            }
            BandRequest::Spectral2 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let b0 = [
                    bands_1based[0].saturating_sub(1),
                    bands_1based[1].saturating_sub(1),
                ];
                for &b in &b0 {
                    if b >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            b + 1,
                            ifd.samples_per_pixel
                        )));
                    }
                }
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                let so = |b: u16| -> (f32, f32) {
                    if req.unscale {
                        (
                            cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                            cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                        )
                    } else {
                        (1.0, 0.0)
                    }
                };
                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let a_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[0] as usize)))
                            .collect();
                        let b_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[1] as usize)))
                            .collect();
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s1, o1),
                        ];
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_spectral::<N>(
                                cog.id,
                                ifd_idx,
                                bands_1based,
                                &plan,
                                &srcs,
                                *expr,
                                &style,
                            ) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_bandmath2::<N>(&plan, &srcs, *expr, &style))
                    }
                    PlanarConfig::Planar => {
                        let mut indices = Vec::with_capacity(grid_len * 2);
                        for &band in &b0 {
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                                }
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let a_refs: Vec<_> = tiles[0..grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let b_refs: Vec<_> = tiles[grid_len..2 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s1, o1),
                        ];
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_spectral::<N>(
                                cog.id,
                                ifd_idx,
                                bands_1based,
                                &plan,
                                &srcs,
                                *expr,
                                &style,
                            ) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_bandmath2::<N>(&plan, &srcs, *expr, &style))
                    }
                }
            }
            BandRequest::Spectral3 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let b0 = [
                    bands_1based[0].saturating_sub(1),
                    bands_1based[1].saturating_sub(1),
                    bands_1based[2].saturating_sub(1),
                ];
                for &b in &b0 {
                    if b >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            b + 1,
                            ifd.samples_per_pixel
                        )));
                    }
                }
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let r_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[0] as usize)))
                            .collect();
                        let g_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[1] as usize)))
                            .collect();
                        let b_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[2] as usize)))
                            .collect();
                        let so = |b: u16| -> (f32, f32) {
                            if req.unscale {
                                (
                                    cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                                    cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                                )
                            } else {
                                (1.0, 0.0)
                            }
                        };
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let (s2, o2) = so(b0[2]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            )
                            .with_scale_offset(s1, o1),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s2, o2),
                        ];
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_spectral::<N>(
                                cog.id,
                                ifd_idx,
                                bands_1based,
                                &plan,
                                &srcs,
                                *expr,
                                &style,
                            ) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_bandmath3::<N>(&plan, &srcs, *expr, &style))
                    }
                    PlanarConfig::Planar => {
                        let mut indices = Vec::with_capacity(grid_len * 3);
                        for &band in &b0 {
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                                }
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let r_refs: Vec<_> = tiles[0..grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let g_refs: Vec<_> = tiles[grid_len..2 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let b_refs: Vec<_> = tiles[2 * grid_len..3 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let so = |b: u16| -> (f32, f32) {
                            if req.unscale {
                                (
                                    cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                                    cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                                )
                            } else {
                                (1.0, 0.0)
                            }
                        };
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let (s2, o2) = so(b0[2]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            )
                            .with_scale_offset(s1, o1),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s2, o2),
                        ];
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_spectral::<N>(
                                cog.id,
                                ifd_idx,
                                bands_1based,
                                &plan,
                                &srcs,
                                *expr,
                                &style,
                            ) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_bandmath3::<N>(&plan, &srcs, *expr, &style))
                    }
                }
            }
            BandRequest::CustomExprRgb { .. } | BandRequest::MultiBand { .. } => Err(EngineError::BadRequest(
                "3-band custom expressions are rendered via bbox pipeline".into(),
            )),
            BandRequest::CustomExpr {
                prog,
                rescale,
                colormap,
            } => {
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                let nb = prog.used_bands.len();
                for &b in &prog.used_bands {
                    if b == 0 || b > ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "expression band b{b} out of range (dataset has {} bands)",
                            ifd.samples_per_pixel
                        )));
                    }
                }
                if nb == 0 {
                    return Ok(render_expr_vm::<N>(
                        &plan,
                        &[],
                        prog,
                        &style,
                        (ifd.width, ifd.height),
                    ));
                }
                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let mut band_vecs: Vec<Vec<Option<&cogzilla_codec::DecodedBand>>> =
                            Vec::with_capacity(nb);
                        for &b in &prog.used_bands {
                            let b0 = (b - 1) as usize;
                            band_vecs.push(
                                tiles
                                    .iter()
                                    .map(|o| o.as_ref().and_then(|t| t.bands.get(b0)))
                                    .collect(),
                            );
                        }
                        let srcs: Vec<BandSource<'_>> = band_vecs
                            .iter()
                            .enumerate()
                            .map(|(k, refs)| {
                                let b0 = prog.used_bands[k].saturating_sub(1) as usize;
                                let (sc, off) = if req.unscale {
                                    (
                                        cog.index.scales.get(b0).copied().unwrap_or(1.0),
                                        cog.index.offsets.get(b0).copied().unwrap_or(0.0),
                                    )
                                } else {
                                    (1.0, 0.0)
                                };
                                BandSource::new(
                                    ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0,
                                    ty0, grid_w, grid_h, refs,
                                )
                                .with_scale_offset(sc, off)
                            })
                            .collect();
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_expr::<N>(cog.id, ifd_idx, &plan, &srcs, prog, &style) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_expr_vm::<N>(&plan, &srcs, prog, &style, (ifd.width, ifd.height)))
                    }
                    PlanarConfig::Planar => {
                        let mut indices = Vec::with_capacity(grid_len * nb);
                        for &b in &prog.used_bands {
                            let b0 = b - 1;
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, b0) as u32));
                                }
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let mut band_vecs: Vec<Vec<Option<&cogzilla_codec::DecodedBand>>> =
                            Vec::with_capacity(nb);
                        for i in 0..nb {
                            band_vecs.push(
                                tiles[i * grid_len..(i + 1) * grid_len]
                                    .iter()
                                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                                    .collect(),
                            );
                        }
                        let srcs: Vec<BandSource<'_>> = band_vecs
                            .iter()
                            .enumerate()
                            .map(|(k, refs)| {
                                let b0 = prog.used_bands[k].saturating_sub(1) as usize;
                                let (sc, off) = if req.unscale {
                                    (
                                        cog.index.scales.get(b0).copied().unwrap_or(1.0),
                                        cog.index.offsets.get(b0).copied().unwrap_or(0.0),
                                    )
                                } else {
                                    (1.0, 0.0)
                                };
                                BandSource::new(
                                    ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0,
                                    ty0, grid_w, grid_h, refs,
                                )
                                .with_scale_offset(sc, off)
                            })
                            .collect();
                        #[cfg(feature = "gpu")]
                        if let Some(gpu) = &self.gpu {
                            if let Some(bytes) = gpu.try_render_expr::<N>(cog.id, ifd_idx, &plan, &srcs, prog, &style) {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        }
                        Ok(render_expr_vm::<N>(&plan, &srcs, prog, &style, (ifd.width, ifd.height)))
                    }
                }
            }
        }
    }

    /// Render a single XYZ tile by compositing 2 or 3 separate co-registered single-band COGs
    /// in parallel (e.g., Landsat C2 `[SR_B4, SR_B3, SR_B2]` RGB or OPERA RTC-S1 `[VV, VH]` SAR-RGB).
    pub async fn render_multi_cog_tile(
        &self,
        cogs: &[Arc<OpenedCog>],
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        if cogs.len() == 1 {
            return self.render_tile(&cogs[0], req).await;
        }
        self.stats.tiles_served.fetch_add(1, Ordering::Relaxed);
        if req.format == OutputFormat::Npy {
            let mut raw_req = req.clone();
            raw_req.format = OutputFormat::RawF32;
            let raw = match raw_req.tile_size {
                256 => self.render_multi_cog_sized::<256>(cogs, &raw_req).await?,
                512 => self.render_multi_cog_sized::<512>(cogs, &raw_req).await?,
                other => {
                    return Err(EngineError::BadRequest(format!(
                        "unsupported tile_size={other}, expected 256 or 512"
                    )));
                }
            };
            let nb = match req.bands {
                BandRequest::Rgb { .. }
                | BandRequest::SarRgb { .. }
                | BandRequest::Terrain { .. }
                | BandRequest::CustomExprRgb { .. } => 3,
                BandRequest::MultiBand { ref bands_1based } => bands_1based.len().max(1),
                _ => 1,
            };
            return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                raw,
                nb,
                req.tile_size,
                req.tile_size,
                true,
            ));
        }
        match req.tile_size {
            256 => self.render_multi_cog_sized::<256>(cogs, req).await,
            512 => self.render_multi_cog_sized::<512>(cogs, req).await,
            other => Err(EngineError::BadRequest(format!(
                "unsupported tile_size={other}, expected 256 or 512"
            ))),
        }
    }

    async fn render_multi_cog_sized<const N: usize>(
        &self,
        cogs: &[Arc<OpenedCog>],
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        if cogs.is_empty() || cogs.len() > 16 {
            return Err(EngineError::BadRequest(format!(
                "multi-COG compositing supports 1..=16 assets, got {}",
                cogs.len()
            )));
        }
        let primary = &cogs[0];
        let crs = self.get_or_build_crs(primary)?;
        let (target_rx, target_ry) = crs.target_resolution_for_tile(req.tile, N as u32);
        let ifd_idx = primary.index.select_ifd(target_rx, target_ry);
        validate_multi_cog_grid(cogs, ifd_idx)?;
        let ifd = &primary.index.ifds[ifd_idx];

        let Some(plan) = CoordPlan::<N>::build_with_gcp(
            &crs,
            &ifd.crs_to_pixel,
            ifd.gcp_poly.as_deref(),
            ifd.width,
            ifd.height,
            req.tile,
        ) else {
            self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
            let nb = match req.bands {
                BandRequest::Rgb { .. }
                | BandRequest::SarRgb { .. }
                | BandRequest::Terrain { .. }
                | BandRequest::CustomExprRgb { .. } => 3,
                BandRequest::MultiBand { ref bands_1based } => bands_1based.len().max(1),
                _ => 1,
            };
            return Ok(empty_tile_bytes(N as u32, req.format, nb));
        };

        let (pad_lo, pad_hi) =
            resample_fetch_margin(req.resample, matches!(req.bands, BandRequest::Dem { .. }));
        let (tx0, tx1, ty0, ty1) = if pad_lo > 0 || pad_hi > 1 {
            // CoordPlan::src_pixel_bounds() is the exact bilinear footprint [floor(min), floor(max)+1].
            let extra_lo = pad_lo;
            let extra_hi = (pad_hi - 1).max(0);
            let (c0, c1, r0, r1) = plan.src_pixel_bounds();
            let d_tx0 = (c0.saturating_sub(extra_lo).max(0) as u32) / ifd.tile_width;
            let d_tx1 = ((c1.saturating_add(extra_hi).max(0) as u32) / ifd.tile_width)
                .min(ifd.tiles_across.saturating_sub(1));
            let d_ty0 = (r0.saturating_sub(extra_lo).max(0) as u32) / ifd.tile_height;
            let d_ty1 = ((r1.saturating_add(extra_hi).max(0) as u32) / ifd.tile_height)
                .min(ifd.tiles_down.saturating_sub(1));
            (d_tx0, d_tx1, d_ty0, d_ty1)
        } else {
            plan.src_tile_range(ifd.tile_width, ifd.tile_height, ifd.tiles_across, ifd.tiles_down)
        };
        let grid_w = tx1 - tx0 + 1;
        let grid_h = ty1 - ty0 + 1;
        let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
            }
        }

        // Fetch all co-registered COGs concurrently via JoinSet (supports 2..=16 assets).
        let indices_arc = Arc::new(indices);
        let mut js = tokio::task::JoinSet::new();
        for (slot, c) in cogs.iter().enumerate() {
            let store = self.store.clone();
            let cog = c.clone();
            let idx_ref = indices_arc.clone();
            js.spawn(async move {
                let blocks = store.get_blocks(&cog, ifd_idx, &idx_ref).await?;
                Ok::<_, cogzilla_io::IoError>((slot, blocks))
            });
        }
        let mut fetched: Vec<Option<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>>> =
            vec![None; cogs.len()];
        while let Some(res) = js.join_next().await {
            let (slot, blocks) = res.map_err(|e| {
                EngineError::Io(cogzilla_io::IoError::Fs(std::io::Error::other(format!(
                    "multi-cog tile fetch task failed: {e}"
                ))))
            })??;
            fetched[slot] = Some(blocks);
        }
        let all_tiles: Vec<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>> =
            fetched.into_iter().map(|o| o.unwrap()).collect();
        let all_refs: Vec<Vec<Option<&cogzilla_codec::DecodedBand>>> = all_tiles
            .iter()
            .map(|ts| {
                ts.iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                    .collect()
            })
            .collect();

        let apply_unscale = req.unscale && needs_source_unscale(&req.bands, req.format);
        let make_src = |b_1based: u16| -> Result<BandSource<'_>, EngineError> {
            let idx = b_1based.saturating_sub(1) as usize;
            if b_1based == 0 || idx >= all_refs.len() {
                return Err(EngineError::BadRequest(format!(
                    "band b{b_1based} out of range (only {} assets were provided)",
                    all_refs.len()
                )));
            }
            let (sc, off) = if apply_unscale {
                (
                    cogs[idx].index.scales.first().copied().unwrap_or(1.0),
                    cogs[idx].index.offsets.first().copied().unwrap_or(0.0),
                )
            } else {
                (1.0, 0.0)
            };
            Ok(BandSource::new(
                ifd.width,
                ifd.height,
                ifd.tile_width,
                ifd.tile_height,
                tx0,
                ty0,
                grid_w,
                grid_h,
                &all_refs[idx],
            )
            .with_scale_offset(sc, off))
        };

        match &req.bands {
            BandRequest::Single {
                band_1based,
                rescale,
                colormap,
            } => {
                let src = make_src(*band_1based)?;
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                Ok(render_single_band::<N>(&plan, &src, &style))
            }
            BandRequest::Terrain {
                band_1based,
                encoding,
            } => {
                let src = make_src(*band_1based)?;
                Ok(render_terrain_tile::<N>(
                    &plan,
                    &src,
                    *encoding,
                    req.resample,
                    req.format,
                ))
            }
            BandRequest::Dem {
                band_1based,
                algorithm,
                rescale,
                colormap,
            } => {
                let src = make_src(*band_1based)?;
                let (bx0, by0, bx1, by1) = tile_bounds_3857(req.tile);
                let (_, lat_deg) = mercator_to_lonlat_deg((bx0 + bx1) * 0.5, (by0 + by1) * 0.5);
                let pixel_size_m =
                    (((bx1 - bx0) / N as f64) * lat_deg.to_radians().cos().max(0.05)) as f32;
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                Ok(render_dem_algorithm::<N>(
                    &plan,
                    &src,
                    *algorithm,
                    pixel_size_m,
                    &style,
                ))
            }
            BandRequest::Spectral1 {
                expr,
                band_1based,
                rescale,
                colormap,
            } => {
                let src = make_src(*band_1based)?;
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                Ok(render_bandmath1::<N>(&plan, &src, *expr, &style))
            }
            BandRequest::SarRgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let srcs = [make_src(bands_1based[0])?, make_src(bands_1based[1])?];
                let style = RgbStyle {
                    rescale_min: *rescale_min,
                    rescale_max: *rescale_max,
                    resample: req.resample,
                    format: req.format,
                };
                Ok(render_sar_rgb::<N>(&plan, &srcs, &style))
            }
            BandRequest::Spectral2 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let srcs = [make_src(bands_1based[0])?, make_src(bands_1based[1])?];
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                Ok(render_bandmath2::<N>(&plan, &srcs, *expr, &style))
            }
            BandRequest::Spectral3 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let srcs = [
                    make_src(bands_1based[0])?,
                    make_src(bands_1based[1])?,
                    make_src(bands_1based[2])?,
                ];
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                Ok(render_bandmath3::<N>(&plan, &srcs, *expr, &style))
            }
            BandRequest::Rgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let srcs = [
                    make_src(bands_1based[0])?,
                    make_src(bands_1based[1])?,
                    make_src(bands_1based[2])?,
                ];
                let style = RgbStyle {
                    rescale_min: *rescale_min,
                    rescale_max: *rescale_max,
                    resample: req.resample,
                    format: req.format,
                };
                Ok(render_rgb_bands::<N>(&plan, &srcs, &style))
            }
            BandRequest::CustomExprRgb { .. } | BandRequest::MultiBand { .. } => Err(EngineError::BadRequest(
                "3-band custom expressions are rendered via bbox pipeline".into(),
            )),
            BandRequest::CustomExpr {
                prog,
                rescale,
                colormap,
            } => {
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                let mut ordered = Vec::with_capacity(prog.used_bands.len());
                for &b in &prog.used_bands {
                    ordered.push(make_src(b)?);
                }
                Ok(render_expr_vm::<N>(&plan, &ordered, prog, &style, (ifd.width, ifd.height)))
            }
        }
    }

    /// Render a dynamic front-to-back mosaic across `urls` with **early-exit pixel coverage**:
    /// walks scenes in priority order, composites only into unfilled (`alpha == 0`) pixels, and
    /// stops immediately once all `tile_size * tile_size` pixels are opaque — never opening,
    /// fetching, or decoding remaining background scenes!
    pub async fn render_mosaic_tile(
        &self,
        urls: &[String],
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        Ok(self.render_mosaic_tile_with_report(urls, req).await?.0)
    }

    /// Render a dynamic front-to-back mosaic across `urls`, returning `(tile_bytes, failed_assets)`
    /// where `failed_assets` lists any redacted asset URLs that could not be fetched and were
    /// skipped before rendering completed.
    pub async fn render_mosaic_tile_with_report(
        &self,
        urls: &[String],
        req: &TileRequest,
    ) -> Result<(Bytes, Vec<String>), EngineError> {
        if urls.is_empty() {
            return Err(EngineError::BadRequest(
                "mosaic requires at least one ?url= parameter".into(),
            ));
        }
        // Bound per-tile scene fan-out so a huge MosaicJSON/catalog of transparent or OOB COGs
        // cannot trigger thousands of sequential open_uri + render_tile calls on a single GET.
        const MAX_MOSAIC_SCENES_PER_TILE: usize = 64;
        let urls = &urls[..urls.len().min(MAX_MOSAIC_SCENES_PER_TILE)];
        let n = req.tile_size as usize;
        let total_pixels = n * n;
        let f32_bands: usize = match req.bands {
            BandRequest::Rgb { .. }
            | BandRequest::SarRgb { .. }
            | BandRequest::Terrain { .. }
            | BandRequest::CustomExprRgb { .. } => 3,
            BandRequest::MultiBand { ref bands_1based } => bands_1based.len().max(1),
            _ => 1,
        };

        // Helper to fold a scene's GDAL scale/offset into Single/Rgb rescale bounds when `?unscale=true`.
        let fold_scene_unscale = |cog: &OpenedCog, base: &TileRequest| -> TileRequest {
            if base.unscale
                && matches!(
                    base.bands,
                    BandRequest::Single { .. } | BandRequest::Rgb { .. }
                )
            {
                let mut sr = base.clone();
                let adj = |b1: u16, mn: f32, mx: f32| -> (f32, f32) {
                    let i = b1.saturating_sub(1) as usize;
                    let sc = cog.index.scales.get(i).copied().unwrap_or(1.0);
                    let off = cog.index.offsets.get(i).copied().unwrap_or(0.0);
                    if ((sc - 1.0).abs() < 1e-9 && off.abs() < 1e-9) || sc.abs() < 1e-12 {
                        (mn, mx)
                    } else {
                        ((mn - off) / sc, (mx - off) / sc)
                    }
                };
                match &mut sr.bands {
                    BandRequest::Single {
                        band_1based,
                        rescale,
                        ..
                    } => {
                        *rescale = adj(*band_1based, rescale.0, rescale.1);
                    }
                    BandRequest::Rgb {
                        bands_1based,
                        rescale_min,
                        rescale_max,
                    } => {
                        for k in 0..3 {
                            let (mn, mx) = adj(bands_1based[k], rescale_min[k], rescale_max[k]);
                            rescale_min[k] = mn;
                            rescale_max[k] = mx;
                        }
                    }
                    _ => {}
                }
                sr
            } else {
                base.clone()
            }
        };

        // 1-scene fast path: skip canvas allocation, pixel compositing, and re-encoding so single-scene
        // mosaic tiles keep zero-decode JPEG/WebP passthrough and 8-bit indexed PLTE PNG encoding!
        if urls.len() == 1 {
            let cog = self
                .store
                .open_uri(&urls[0])
                .await
                .map_err(|e| asset_fetch_error(&urls[0], e))?;
            let scene_req = fold_scene_unscale(&cog, req);
            let bytes = self.render_tile(&cog, &scene_req).await?;
            return Ok((bytes, Vec::new()));
        }

        // RawF32 / Npy mosaic path: composite real f32 band-major planes (CHW) with NaN transparency
        // and early exit once every pixel has valid data.
        if matches!(req.format, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut canvas = vec![f32::NAN; f32_bands * total_pixels];
            let mut filled = 0usize;
            let mut scene_req = req.clone();
            scene_req.format = OutputFormat::RawF32;
            let empty_f32_sentinel = empty_tile_bytes(req.tile_size, OutputFormat::RawF32, f32_bands);

            let mut opened_any = false;
            let mut last_open_err: Option<EngineError> = None;
            let mut failed_assets: Vec<String> = Vec::new();
            for (scene_i, url) in urls.iter().enumerate() {
                if filled >= total_pixels {
                    break;
                }
                if scene_i > 0 {
                    tokio::task::consume_budget().await;
                }
                let cog = match self.store.open_uri(url).await {
                    Ok(c) => {
                        opened_any = true;
                        c
                    }
                    Err(e) => {
                        failed_assets.push(cogzilla_io::redact_http_url(url));
                        last_open_err = Some(asset_fetch_error(url, e));
                        continue;
                    }
                };
                let src = self.render_tile(&cog, &scene_req).await?;
                if src.len() != f32_bands * total_pixels * 4
                    || src.as_ptr() == empty_f32_sentinel.as_ptr()
                {
                    continue;
                }
                for i in 0..total_pixels {
                    if !canvas[i].is_nan() {
                        continue;
                    }
                    let mut all_valid = true;
                    let mut vals = [f32::NAN; 3];
                    for b in 0..f32_bands {
                        let byte_off = (b * total_pixels + i) * 4;
                        let v = f32::from_le_bytes([
                            src[byte_off],
                            src[byte_off + 1],
                            src[byte_off + 2],
                            src[byte_off + 3],
                        ]);
                        if v.is_nan() {
                            all_valid = false;
                            break;
                        }
                        vals[b] = v;
                    }
                    if all_valid {
                        for b in 0..f32_bands {
                            canvas[b * total_pixels + i] = vals[b];
                        }
                        filled += 1;
                    }
                }
            }

            if filled == 0 {
                if !opened_any
                    && let Some(err) = last_open_err
                {
                    return Err(err);
                }
                if req.format == OutputFormat::Npy {
                    let raw = empty_tile_bytes(req.tile_size, OutputFormat::RawF32, f32_bands);
                    return Ok((
                        cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                            raw,
                            f32_bands,
                            req.tile_size,
                            req.tile_size,
                            true,
                        ),
                        failed_assets,
                    ));
                }
                return Ok((
                    empty_tile_bytes(req.tile_size, req.format, f32_bands),
                    failed_assets,
                ));
            }
            let mut out = bytes::BytesMut::with_capacity(f32_bands * total_pixels * 4);
            for v in canvas {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let raw = out.freeze();
            if req.format == OutputFormat::Npy {
                return Ok((
                    cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                        raw,
                        f32_bands,
                        req.tile_size,
                        req.tile_size,
                        true,
                    ),
                    failed_assets,
                ));
            }
            return Ok((raw, failed_assets));
        }

        let mut canvas = vec![0u8; total_pixels * 4];
        let mut filled = 0usize;

        let mut base_scene_req = req.clone();
        base_scene_req.format = OutputFormat::RawRgba;
        let empty_rgba_sentinel = empty_tile_bytes(req.tile_size, OutputFormat::RawRgba, 1);

        let mut opened_any = false;
        let mut last_open_err: Option<EngineError> = None;
        let mut failed_assets: Vec<String> = Vec::new();
        for (scene_i, url) in urls.iter().enumerate() {
            if filled >= total_pixels {
                break;
            }
            if scene_i > 0 {
                tokio::task::consume_budget().await;
            }
            let cog = match self.store.open_uri(url).await {
                Ok(c) => {
                    opened_any = true;
                    c
                }
                Err(e) => {
                    failed_assets.push(cogzilla_io::redact_http_url(url));
                    last_open_err = Some(asset_fetch_error(url, e));
                    continue;
                }
            };
            let scene_req = fold_scene_unscale(&cog, &base_scene_req);
            let src = self.render_tile(&cog, &scene_req).await?;
            if src.len() != total_pixels * 4 || src.as_ptr() == empty_rgba_sentinel.as_ptr() {
                continue;
            }
            for i in 0..total_pixels {
                let off = i * 4;
                if canvas[off + 3] == 0 && src[off + 3] > 0 {
                    canvas[off] = src[off];
                    canvas[off + 1] = src[off + 1];
                    canvas[off + 2] = src[off + 2];
                    canvas[off + 3] = src[off + 3];
                    if src[off + 3] == 255 {
                        filled += 1;
                    }
                }
            }
        }

        if filled == 0 && canvas.chunks_exact(4).all(|p| p[3] == 0) {
            if !opened_any
                && let Some(err) = last_open_err
            {
                return Err(err);
            }
            return Ok((
                empty_tile_bytes(req.tile_size, req.format, f32_bands),
                failed_assets,
            ));
        }
        Ok((
            encode_rgba_tile(
                &canvas,
                req.tile_size,
                filled == total_pixels,
                req.format,
            ),
            failed_assets,
        ))
    }

    /// Query a pluggable `SceneCatalog` for the requested tile's EPSG:3857 bounding box
    /// and render an early-exit mosaic across the intersecting scenes.
    pub async fn render_catalog_mosaic_tile(
        &self,
        catalog: &dyn SceneCatalog,
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(req.tile);
        let urls = catalog.query_bbox_3857(min_x, min_y, max_x, max_y);
        if urls.is_empty() {
            self.stats
                .empty_short_circuits
                .fetch_add(1, Ordering::Relaxed);
            let f32_bands = match req.bands {
                BandRequest::Rgb { .. }
                | BandRequest::SarRgb { .. }
                | BandRequest::Terrain { .. }
                | BandRequest::CustomExprRgb { .. } => 3,
                BandRequest::MultiBand { ref bands_1based } => bands_1based.len().max(1),
                _ => 1,
            };
            return Ok(empty_tile_bytes(req.tile_size, req.format, f32_bands));
        }
        self.render_mosaic_tile(&urls, req).await
    }

    async fn try_zero_decode_passthrough<const N: usize>(
        &self,
        cog: &OpenedCog,
        crs: &CrsModel,
        ifd_idx: usize,
        ifd: &IfdMeta,
        req: &TileRequest,
    ) -> Result<Option<Bytes>, EngineError> {
        let Some((tx, ty)) =
            aligned_passthrough_tile_coord::<N>(crs, ifd, cog.index.nodata, req)
        else {
            return Ok(None);
        };

        let Some((offset, byte_len)) = ifd.tile_byte_range(tx, ty, 0) else {
            // Sparse empty internal tile -> short-circuit without decoding.
            self.stats
                .empty_short_circuits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(Some(empty_tile_bytes(N as u32, req.format, 3)));
        };

        let tile_idx = ifd.tile_index(tx, ty, 0) as u32;
        let raw = self
            .store
            .get_raw_block(cog, ifd_idx, tile_idx, offset, byte_len)
            .await?;

        let out = if ifd.compression == Compression::Jpeg {
            if let Some(tables) = ifd.jpeg_tables.as_deref() {
                splice_jpeg_tables(tables, &raw)
            } else {
                raw
            }
        } else {
            raw
        };

        self.stats.passthrough_hits.fetch_add(1, Ordering::Relaxed);
        Ok(Some(out))
    }
}

/// Pure eligibility check for zero-decode pass-through of WebMercator-aligned JPEG/WebP COGs.
/// Returns `Some((tx, ty))` when the requested XYZ tile maps 1:1 onto a full, in-bounds
/// internal COG tile with identical codec and identity RGB styling.
fn aligned_passthrough_tile_coord<const N: usize>(
    crs: &CrsModel,
    ifd: &IfdMeta,
    nodata: Option<f32>,
    req: &TileRequest,
) -> Option<(u32, u32)> {
    if !matches!(crs, CrsModel::Epsg3857) || req.unscale {
        return None;
    }
    if ifd.is_stripped
        || ifd.planar_config != PlanarConfig::Chunky
        || ifd.sample_type != SampleType::Uint8
        || ifd.samples_per_pixel != 3
        || ifd.mask_ifd.is_some()
        || !ifd.extra_samples.is_empty()
        || nodata.is_some()
        || !ifd.crs_to_pixel.is_axis_aligned()
        || ifd.tile_width != N as u32
        || ifd.tile_height != N as u32
    {
        return None;
    }

    // Only pass through browser-safe encodings: YCbCr JPEG (photometric=6) or WebP (self-describing).
    // TIFF RGB JPEG (photometric=2) may omit Adobe APP14 transform=0, causing browsers to decode
    // raw RGB as YCbCr (magenta/green corruption).
    let codec_matches = matches!(
        (ifd.compression, ifd.photometric, req.format),
        (Compression::Jpeg, 6, OutputFormat::Jpeg { .. })
            | (Compression::Webp, 2 | 6, OutputFormat::Webp { .. })
    );
    if !codec_matches {
        return None;
    }

    match &req.bands {
        BandRequest::Rgb {
            bands_1based: [1, 2, 3],
            rescale_min,
            rescale_max,
        } if *rescale_min == [0.0, 0.0, 0.0] && *rescale_max == [255.0, 255.0, 255.0] => {}
        _ => return None,
    }

    let (min_x, min_y, max_x, max_y) = tile_bounds_3857(req.tile);
    let (col0, row0) = ifd.crs_to_pixel.forward(min_x, max_y);
    let (col1, row1) = ifd.crs_to_pixel.forward(max_x, min_y);
    let nf = N as f64;
    const TOL: f64 = 1e-2;

    if (col1 - col0 - nf).abs() > TOL || (row1 - row0 - nf).abs() > TOL {
        return None;
    }
    if col0 < -TOL
        || row0 < -TOL
        || col1 > ifd.width as f64 + TOL
        || row1 > ifd.height as f64 + TOL
    {
        return None;
    }

    let tx_f = col0 / nf;
    let ty_f = row0 / nf;
    let tx_round = tx_f.round();
    let ty_round = ty_f.round();
    if (tx_f - tx_round).abs() * nf > TOL || (ty_f - ty_round).abs() * nf > TOL {
        return None;
    }
    let tx = tx_round as i64;
    let ty = ty_round as i64;
    if tx < 0 || ty < 0 || tx >= ifd.tiles_across as i64 || ty >= ifd.tiles_down as i64 {
        return None;
    }
    Some((tx as u32, ty as u32))
}

/// Merge TIFF tag 347 `JPEGTables` with an abbreviated JPEG tile bitstream into a
/// standalone browser-decodable JPEG stream.
fn splice_jpeg_tables(tables: &[u8], raw: &[u8]) -> Bytes {
    if tables.len() >= 4 && raw.len() >= 2 {
        let t_end = if tables.ends_with(&[0xFF, 0xD9]) {
            tables.len() - 2
        } else {
            tables.len()
        };
        let c_start = if raw.starts_with(&[0xFF, 0xD8]) { 2 } else { 0 };
        let mut out = Vec::with_capacity(t_end + (raw.len() - c_start));
        out.extend_from_slice(&tables[..t_end]);
        out.extend_from_slice(&raw[c_start..]);
        Bytes::from(out)
    } else {
        Bytes::copy_from_slice(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_tiff::Affine2D;

    fn make_aligned_ifd(
        tile: TileCoord,
        compression: Compression,
        tile_size: u32,
        tiles_across: u32,
        tiles_down: u32,
    ) -> IfdMeta {
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
        let res_x = (max_x - min_x) / tile_size as f64;
        let res_y = (max_y - min_y) / tile_size as f64;
        // Anchor COG origin 1 tile west and 1 tile north so `tile` sits at internal `(tx=1, ty=1)`.
        let origin_x = min_x - (max_x - min_x);
        let origin_y = max_y + (max_y - min_y);
        let pixel_to_crs = Affine2D {
            a: res_x,
            b: 0.0,
            c: origin_x,
            d: 0.0,
            e: -res_y,
            f: origin_y,
        };
        let crs_to_pixel = pixel_to_crs.invert().unwrap();
        let n_tiles = (tiles_across * tiles_down) as usize;
        IfdMeta {
            width: tile_size * tiles_across,
            height: tile_size * tiles_down,
            tile_width: tile_size,
            tile_height: tile_size,
            tiles_across,
            tiles_down,
            sample_type: SampleType::Uint8,
            samples_per_pixel: 3,
            planar_config: PlanarConfig::Chunky,
            compression,
            predictor: cogzilla_tiff::Predictor::None,
            photometric: 6,
            extra_samples: Vec::new(),
            jpeg_tables: None,
            mask_ifd: None,
            is_stripped: false,
            pixel_to_crs,
            crs_to_pixel,
            gcp_poly: None,
            tile_offsets: vec![0u64; n_tiles].into_boxed_slice(),
            tile_byte_counts: vec![128u32; n_tiles].into_boxed_slice(),
        }
    }

    #[test]
    fn zero_decode_passthrough_matches_aligned_jpeg_and_webp_and_rejects_transforms() {
        let tile = TileCoord { z: 12, x: 655, y: 1583 };
        let ifd_jpeg = make_aligned_ifd(tile, Compression::Jpeg, 256, 4, 4);
        let ifd_webp = make_aligned_ifd(tile, Compression::Webp, 256, 4, 4);

        let rgb_id = BandRequest::Rgb {
            bands_1based: [1, 2, 3],
            rescale_min: [0.0, 0.0, 0.0],
            rescale_max: [255.0, 255.0, 255.0],
        };

        let req_jpeg = TileRequest {
            tile,
            tile_size: 256,
            bands: rgb_id.clone(),
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Jpeg { quality: 85 },
            unscale: false,
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_jpeg, None, &req_jpeg),
            Some((1, 1))
        );

        let req_webp = TileRequest {
            tile,
            tile_size: 256,
            bands: rgb_id.clone(),
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Webp { quality: Some(85.0) },
            unscale: false,
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_webp, None, &req_webp),
            Some((1, 1))
        );

        // Reject mismatched format (PNG requested on JPEG COG)
        let req_png = TileRequest {
            format: OutputFormat::Png,
            ..req_jpeg.clone()
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_jpeg, None, &req_png),
            None
        );

        // Reject rescaled request
        let req_rescaled = TileRequest {
            bands: BandRequest::Rgb {
                bands_1based: [1, 2, 3],
                rescale_min: [10.0, 10.0, 10.0],
                rescale_max: [200.0, 200.0, 200.0],
            },
            ..req_jpeg.clone()
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_jpeg, None, &req_rescaled),
            None
        );

        // Reject band-swapped request
        let req_swap = TileRequest {
            bands: BandRequest::Rgb {
                bands_1based: [3, 2, 1],
                rescale_min: [0.0, 0.0, 0.0],
                rescale_max: [255.0, 255.0, 255.0],
            },
            ..req_jpeg.clone()
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_jpeg, None, &req_swap),
            None
        );

        // Reject non-aligned (half-tile shifted) IFD
        let mut ifd_shifted = ifd_jpeg.clone();
        ifd_shifted.pixel_to_crs.c += ifd_shifted.pixel_to_crs.a * 128.0;
        ifd_shifted.crs_to_pixel = ifd_shifted.pixel_to_crs.invert().unwrap();
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_shifted, None, &req_jpeg),
            None
        );

        // Verify JPEGTables splicing strips trailing EOI (FF D9) from tables and leading SOI (FF D8) from tile
        let spliced = splice_jpeg_tables(
            &[0xFF, 0xD8, 0xFF, 0xDB, 0x01, 0x02, 0xFF, 0xD9],
            &[0xFF, 0xD8, 0xFF, 0xC0, 0xAA, 0xBB, 0xFF, 0xD9],
        );
        assert_eq!(
            spliced.as_ref(),
            &[0xFF, 0xD8, 0xFF, 0xDB, 0x01, 0x02, 0xFF, 0xC0, 0xAA, 0xBB, 0xFF, 0xD9]
        );
    }

    #[tokio::test]
    async fn zero_decode_passthrough_streams_exact_bytes_without_decoding() {
        let tile = TileCoord { z: 12, x: 655, y: 1583 };
        let mut ifd = make_aligned_ifd(tile, Compression::Jpeg, 256, 2, 2);
        // Write a real payload to a temp file and point tile (1, 1) (index 3) to it.
        let payload = b"\xFF\xD8\xFF\xE0\x00\x10JFIF_ZERO_DECODE_TEST_PAYLOAD\xFF\xD9";
        let tmp_path = std::env::temp_dir().join(format!(
            "cogzilla_passthrough_test_{}.bin",
            std::process::id()
        ));
        std::fs::write(&tmp_path, payload).unwrap();
        ifd.tile_offsets = vec![0, 0, 0, 0].into_boxed_slice();
        ifd.tile_byte_counts = vec![0, 0, 0, payload.len() as u32].into_boxed_slice();

        let mut geo_keys = cogzilla_tiff::GeoKeys::default();
        geo_keys.model_type = Some(1);
        geo_keys.projected_cs_type = Some(3857);
        let index = cogzilla_tiff::CogIndex {
            byte_order: cogzilla_tiff::ByteOrder::Little,
            is_bigtiff: false,
            is_georeferenced: true,
            gcp_residual_px: None,
            geo_keys,
            nodata: None,
            scales: vec![1.0; 3],
            offsets: vec![0.0; 3],
            scales_f64: vec![1.0; 3],
            offsets_f64: vec![0.0; 3],
            band_metadata: vec![Vec::new(); 3],
            band_descriptions: vec![String::new(); 3],
            color_map: None,
            ifds: vec![ifd],
        };
        let source = cogzilla_io::CogSource::open_local(&tmp_path).unwrap();
        let cog = OpenedCog {
            id: 999_999,
            uri: tmp_path.to_string_lossy().into_owned(),
            index,
            source,
            freshness: cogzilla_io::store::SourceFreshness::Local {
                path: tmp_path.clone(),
                len: 0,
                mtime: None,
                inode: 0,
            },
        };

        let engine = TileEngine::new(16 * 1024 * 1024);
        let req = TileRequest {
            tile,
            tile_size: 256,
            bands: BandRequest::Rgb {
                bands_1based: [1, 2, 3],
                rescale_min: [0.0, 0.0, 0.0],
                rescale_max: [255.0, 255.0, 255.0],
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Jpeg { quality: 85 },
            unscale: false,
        };
        let out = engine.render_tile(&cog, &req).await.unwrap();
        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(out.as_ref(), payload);
        assert_eq!(engine.stats.passthrough_hits.load(Ordering::Relaxed), 1);
        assert_eq!(engine.store.cache.hits.load(Ordering::Relaxed), 0);
        assert_eq!(engine.store.cache.misses.load(Ordering::Relaxed), 0);
    }
}
