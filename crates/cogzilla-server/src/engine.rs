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
    #[error("dataset not found: {0}")]
    NotFound(String),
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
}

pub struct TileEngine {
    pub store: Arc<CogStore>,
    crs_cache: DashMap<u64, CrsModel, ahash::RandomState>,
    scenes: DashMap<String, Vec<Arc<OpenedCog>>, ahash::RandomState>,
    catalogs: DashMap<String, Arc<dyn SceneCatalog>, ahash::RandomState>,
    pub stac_cache: DashMap<String, Arc<crate::stac::StacItem>, ahash::RandomState>,
    pub mosaicjson_cache: DashMap<String, Arc<MosaicJsonCatalog>, ahash::RandomState>,
    pub tile_cache: Arc<TileCache>,
    pub stats: Arc<EngineStats>,
    pgstac: std::sync::OnceLock<Arc<cogzilla_pgstac::PgStac>>,
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
    if a.projected_cs_type == Some(32767) || a.geographic_type == Some(32767) {
        return a == b;
    }
    true
}

/// Resampling kernel support margin `(lo, hi)` in source pixels needed around `[floor(min), ceil(max)]`.
pub(crate) fn resample_fetch_margin(resample: ResampleMode, is_dem: bool) -> (i32, i32) {
    let (lo, hi) = match resample {
        ResampleMode::Cubic | ResampleMode::CubicSpline => (1, 2),
        ResampleMode::Lanczos => (2, 3),
        _ => (1, 1),
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
            stats: Arc::new(EngineStats::default()),
            pgstac: std::sync::OnceLock::new(),
        }
    }

    /// Attach the pgSTAC database behind the `/searches` routes. The first call wins.
    pub fn set_pgstac(&self, pgstac: Arc<cogzilla_pgstac::PgStac>) -> bool {
        self.pgstac.set(pgstac).is_ok()
    }

    /// The pgSTAC database, when one is configured.
    pub fn pgstac(&self) -> Option<&Arc<cogzilla_pgstac::PgStac>> {
        self.pgstac.get()
    }

    /// Attach the pgSTAC database named by `DATABASE_URL` or the `PG*` variables, if any.
    /// `Ok(false)` means none is configured. No connection is opened here.
    pub fn attach_pgstac_from_env(&self) -> Result<bool, String> {
        let Some(config) = cogzilla_pgstac::PgStacConfig::from_env().map_err(|e| e.to_string())?
        else {
            return Ok(false);
        };
        let pgstac = cogzilla_pgstac::PgStac::new(config).map_err(|e| e.to_string())?;
        Ok(self.set_pgstac(Arc::new(pgstac)))
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
                | BandRequest::Terrain { .. } => 3,
                _ => 1,
            };
            return Ok(cogzilla_kernel::wrap_raw_f32_as_npy(
                raw,
                nb,
                req.tile_size,
                req.tile_size,
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

        let Some(plan) = CoordPlan::<N>::build(
            &crs,
            &ifd.crs_to_pixel,
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
                _ => 1,
            };
            return Ok(empty_tile_bytes(N as u32, req.format, nb));
        };

        let (pad_lo, pad_hi) =
            resample_fetch_margin(req.resample, matches!(req.bands, BandRequest::Dem { .. }));
        let (tx0, tx1, ty0, ty1) = if pad_lo > 1 || pad_hi > 1 {
            // CoordPlan::src_pixel_bounds() already includes ±1 px padding.
            let extra_lo = pad_lo - 1;
            let extra_hi = pad_hi - 1;
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
                // Pad source block range by 2 source pixels so the 1-px DEM border stencil never
                // misses an adjacent COG internal block at tile boundaries.
                let (c0, c1, r0, r1) = plan.src_pixel_bounds();
                let d_tx0 = (c0.saturating_sub(2).max(0) as u32) / ifd.tile_width;
                let d_tx1 = ((c1.saturating_add(2).max(0) as u32) / ifd.tile_width)
                    .min(ifd.tiles_across.saturating_sub(1));
                let d_ty0 = (r0.saturating_sub(2).max(0) as u32) / ifd.tile_height;
                let d_ty1 = ((r1.saturating_add(2).max(0) as u32) / ifd.tile_height)
                    .min(ifd.tiles_down.saturating_sub(1));
                let d_gw = d_tx1 - d_tx0 + 1;
                let d_gh = d_ty1 - d_ty0 + 1;
                let mut indices = Vec::with_capacity((d_gw * d_gh) as usize);
                for ty in d_ty0..=d_ty1 {
                    for tx in d_tx0..=d_tx1 {
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
                    d_tx0,
                    d_ty0,
                    d_gw,
                    d_gh,
                    &block_refs,
                )
                .with_scale_offset(sc, off);
                if src.all_nodata {
                    self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
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
                // Pad source block range by 2 source pixels so the 1-px DEM border stencil never
                // misses an adjacent COG internal block at tile boundaries.
                let (c0, c1, r0, r1) = plan.src_pixel_bounds();
                let d_tx0 = (c0.saturating_sub(2).max(0) as u32) / ifd.tile_width;
                let d_tx1 = ((c1.saturating_add(2).max(0) as u32) / ifd.tile_width)
                    .min(ifd.tiles_across.saturating_sub(1));
                let d_ty0 = (r0.saturating_sub(2).max(0) as u32) / ifd.tile_height;
                let d_ty1 = ((r1.saturating_add(2).max(0) as u32) / ifd.tile_height)
                    .min(ifd.tiles_down.saturating_sub(1));
                let d_gw = d_tx1 - d_tx0 + 1;
                let d_gh = d_ty1 - d_ty0 + 1;
                let mut indices = Vec::with_capacity((d_gw * d_gh) as usize);
                for ty in d_ty0..=d_ty1 {
                    for tx in d_tx0..=d_tx1 {
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
                    d_tx0,
                    d_ty0,
                    d_gw,
                    d_gh,
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
                        Ok(render_bandmath3::<N>(&plan, &srcs, *expr, &style))
                    }
                }
            }
            BandRequest::CustomExprRgb { .. } => Err(EngineError::BadRequest(
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
                        Ok(render_expr_vm::<N>(&plan, &srcs, prog, &style))
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
                        Ok(render_expr_vm::<N>(&plan, &srcs, prog, &style))
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
                | BandRequest::Terrain { .. } => 3,
                _ => 1,
            };
            return Ok(cogzilla_kernel::wrap_raw_f32_as_npy(
                raw,
                nb,
                req.tile_size,
                req.tile_size,
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

        let Some(plan) = CoordPlan::<N>::build(
            &crs,
            &ifd.crs_to_pixel,
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
                _ => 1,
            };
            return Ok(empty_tile_bytes(N as u32, req.format, nb));
        };

        let (pad_lo, pad_hi) =
            resample_fetch_margin(req.resample, matches!(req.bands, BandRequest::Dem { .. }));
        let (tx0, tx1, ty0, ty1) = if pad_lo > 1 || pad_hi > 1 {
            // CoordPlan::src_pixel_bounds() already includes ±1 px padding.
            let extra_lo = pad_lo - 1;
            let extra_hi = pad_hi - 1;
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
            BandRequest::CustomExprRgb { .. } => Err(EngineError::BadRequest(
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
                Ok(render_expr_vm::<N>(&plan, &ordered, prog, &style))
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
        if urls.is_empty() {
            return Err(EngineError::BadRequest(
                "mosaic requires at least one ?url= parameter".into(),
            ));
        }
        // Bound per-tile scene fan-out so a huge MosaicJSON/catalog of transparent or OOB COGs
        // cannot trigger thousands of sequential open_uri + render_tile calls on a single GET.
        const MAX_MOSAIC_SCENES_PER_TILE: usize = 64;
        let scenes: Vec<Vec<String>> = urls
            .iter()
            .take(MAX_MOSAIC_SCENES_PER_TILE)
            .map(|u| vec![u.clone()])
            .collect();
        self.render_mosaic_scenes(&scenes, req).await
    }

    /// Open every asset of one mosaic scene, concurrently when there is more than one.
    async fn open_scene(&self, scene: &[String]) -> Result<Vec<Arc<OpenedCog>>, EngineError> {
        if let [only] = scene {
            return Ok(vec![self.store.open_uri(only).await?]);
        }
        let mut js = tokio::task::JoinSet::new();
        for (idx, uri) in scene.iter().enumerate() {
            let store = self.store.clone();
            let uri = uri.clone();
            js.spawn(async move { Ok::<_, IoError>((idx, store.open_uri(&uri).await?)) });
        }
        let mut slots: Vec<Option<Arc<OpenedCog>>> = vec![None; scene.len()];
        while let Some(res) = js.join_next().await {
            let (idx, cog) = res
                .map_err(|e| EngineError::BadRequest(format!("open task failed: {e}")))??;
            slots[idx] = Some(cog);
        }
        Ok(slots.into_iter().flatten().collect())
    }

    /// The mosaic behind [`Self::render_mosaic_tile`], over scenes that are each one asset or
    /// two to three co-registered single-band assets (rendered as one multi-asset composite).
    /// The caller bounds how many scenes it passes.
    pub async fn render_mosaic_scenes(
        &self,
        scenes: &[Vec<String>],
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        if scenes.is_empty() || scenes.iter().any(|s| s.is_empty()) {
            return Err(EngineError::BadRequest(
                "mosaic requires at least one asset per scene".into(),
            ));
        }
        let n = req.tile_size as usize;
        let total_pixels = n * n;
        let f32_bands: usize = match req.bands {
            BandRequest::Rgb { .. }
            | BandRequest::SarRgb { .. }
            | BandRequest::Terrain { .. }
            | BandRequest::CustomExprRgb { .. } => 3,
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
            let cog = self.store.open_uri(&urls[0]).await?;
            let scene_req = fold_scene_unscale(&cog, req);
            return self.render_tile(&cog, &scene_req).await;
        }

        // RawF32 / Npy mosaic path: composite real f32 band-major planes (CHW) with NaN transparency
        // and early exit once every pixel has valid data.
        if matches!(req.format, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut canvas = vec![f32::NAN; f32_bands * total_pixels];
            let mut filled = 0usize;
            let mut scene_req = req.clone();
            scene_req.format = OutputFormat::RawF32;
            let empty_f32_sentinel = empty_tile_bytes(req.tile_size, OutputFormat::RawF32, f32_bands);

            for (scene_i, url) in urls.iter().enumerate() {
                if filled >= total_pixels {
                    break;
                }
                if scene_i > 0 {
                    tokio::task::consume_budget().await;
                }
                let cog = self.store.open_uri(url).await?;
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
                return Ok(empty_tile_bytes(req.tile_size, req.format, f32_bands));
            }
            let hdr = if req.format == OutputFormat::Npy {
                Some(cogzilla_kernel::encode_npy_f32_header(
                    f32_bands,
                    req.tile_size,
                    req.tile_size,
                ))
            } else {
                None
            };
            let hdr_len = hdr.as_ref().map_or(0, |h| h.len());
            let mut out = bytes::BytesMut::with_capacity(hdr_len + f32_bands * total_pixels * 4);
            if let Some(h) = hdr {
                out.extend_from_slice(&h);
            }
            for v in canvas {
                out.extend_from_slice(&v.to_le_bytes());
            }
            return Ok(out.freeze());
        }

        let mut canvas = vec![0u8; total_pixels * 4];
        let mut filled = 0usize;

        let mut base_scene_req = req.clone();
        base_scene_req.format = OutputFormat::RawRgba;
        let empty_rgba_sentinel = empty_tile_bytes(req.tile_size, OutputFormat::RawRgba, 1);

        for (scene_i, url) in urls.iter().enumerate() {
            if filled >= total_pixels {
                break;
            }
            if scene_i > 0 {
                tokio::task::consume_budget().await;
            }
            let cog = self.store.open_uri(url).await?;
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
            return Ok(empty_tile_bytes(req.tile_size, req.format, f32_bands));
        }
        Ok(encode_rgba_tile(
            &canvas,
            req.tile_size,
            filled == total_pixels,
            req.format,
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
            geo_keys,
            nodata: None,
            scales: vec![1.0; 3],
            offsets: vec![0.0; 3],
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
