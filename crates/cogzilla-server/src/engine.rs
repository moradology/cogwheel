use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use cogzilla_io::{CogStore, IoError, OpenedCog};
use cogzilla_kernel::{
    BandSource, ColormapName, ExprProgram, OutputFormat, ResampleMode, RgbStyle, SingleBandStyle,
    SpectralExpr, empty_tile_bytes, render_bandmath1, render_bandmath2, render_bandmath3,
    render_expr_vm, render_rgb_bands, render_sar_rgb, render_single_band,
};
use cogzilla_proj::{CoordPlan, CrsModel, ProjError, TileCoord};
use cogzilla_tiff::PlanarConfig;
use dashmap::DashMap;
use thiserror::Error;

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
}

#[derive(Debug, Clone)]
pub struct TileRequest {
    pub tile: TileCoord,
    pub tile_size: u32,
    pub bands: BandRequest,
    pub resample: ResampleMode,
    pub format: OutputFormat,
}

#[derive(Default)]
pub struct EngineStats {
    pub tiles_served: AtomicU64,
    pub empty_short_circuits: AtomicU64,
}

pub struct TileEngine {
    pub store: Arc<CogStore>,
    crs_cache: DashMap<u64, CrsModel, ahash::RandomState>,
    scenes: DashMap<String, Vec<Arc<OpenedCog>>, ahash::RandomState>,
    pub stac_cache: DashMap<String, Arc<crate::stac::StacItem>, ahash::RandomState>,
    pub stats: Arc<EngineStats>,
}

impl TileEngine {
    pub fn new(cache_max_bytes: usize) -> Self {
        Self {
            store: Arc::new(CogStore::new(cache_max_bytes)),
            crs_cache: DashMap::with_hasher(ahash::RandomState::new()),
            scenes: DashMap::with_hasher(ahash::RandomState::new()),
            stac_cache: DashMap::with_hasher(ahash::RandomState::new()),
            stats: Arc::new(EngineStats::default()),
        }
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

        let Some(plan) = CoordPlan::<N>::build(
            &crs,
            &ifd.crs_to_pixel,
            ifd.width,
            ifd.height,
            req.tile,
        ) else {
            self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
            let nb = match req.bands {
                BandRequest::Rgb { .. } | BandRequest::SarRgb { .. } => 3,
                _ => 1,
            };
            return Ok(empty_tile_bytes(N as u32, req.format, nb));
        };

        let (tx0, tx1, ty0, ty1) =
            plan.src_tile_range(ifd.tile_width, ifd.tile_height, ifd.tiles_across, ifd.tiles_down);
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
                );
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
                let src = BandSource::new(
                    ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                    grid_h, &block_refs,
                );
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
                let style = RgbStyle {
                    rescale_min: *rescale_min,
                    rescale_max: *rescale_max,
                    resample: req.resample,
                    format: req.format,
                };
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
                let srcs = [
                    BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                        grid_h, &a_refs,
                    ),
                    BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                        grid_h, &b_refs,
                    ),
                ];
                Ok(render_sar_rgb::<N>(&plan, &srcs, &style))
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

                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            ),
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

                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            ),
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
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            ),
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
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            ),
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
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            ),
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
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            ),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            ),
                        ];
                        Ok(render_bandmath3::<N>(&plan, &srcs, *expr, &style))
                    }
                }
            }
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
                            .map(|refs| {
                                BandSource::new(
                                    ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0,
                                    ty0, grid_w, grid_h, refs,
                                )
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
                            .map(|refs| {
                                BandSource::new(
                                    ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0,
                                    ty0, grid_w, grid_h, refs,
                                )
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
        let primary = &cogs[0];
        let crs = self.get_or_build_crs(primary)?;
        let (target_rx, target_ry) = crs.target_resolution_for_tile(req.tile, N as u32);
        let ifd_idx = primary.index.select_ifd(target_rx, target_ry);
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
                BandRequest::Rgb { .. } | BandRequest::SarRgb { .. } => 3,
                _ => 1,
            };
            return Ok(empty_tile_bytes(N as u32, req.format, nb));
        };

        let (tx0, tx1, ty0, ty1) =
            plan.src_tile_range(ifd.tile_width, ifd.tile_height, ifd.tiles_across, ifd.tiles_down);
        let grid_w = tx1 - tx0 + 1;
        let grid_h = ty1 - ty0 + 1;
        let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
            }
        }

        match cogs.len() {
            2 => {
                // Fetch both COGs concurrently (e.g., SAR VV + VH or Landsat B5 + B4)
                let (t0, t1) = tokio::try_join!(
                    self.store.get_blocks(&cogs[0], ifd_idx, &indices),
                    self.store.get_blocks(&cogs[1], ifd_idx, &indices),
                )?;
                let r0: Vec<_> = t0
                    .iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                    .collect();
                let r1: Vec<_> = t1
                    .iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                    .collect();
                let srcs = [
                    BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                        grid_h, &r0,
                    ),
                    BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                        grid_h, &r1,
                    ),
                ];
                match &req.bands {
                    BandRequest::SarRgb {
                        rescale_min,
                        rescale_max,
                        ..
                    } => {
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
                        rescale,
                        colormap,
                        ..
                    } => {
                        let style = SingleBandStyle {
                            rescale_min: rescale.0,
                            rescale_max: rescale.1,
                            colormap: *colormap,
                            resample: req.resample,
                            format: req.format,
                        };
                        Ok(render_bandmath2::<N>(&plan, &srcs, *expr, &style))
                    }
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
                        Ok(render_expr_vm::<N>(&plan, &srcs, prog, &style))
                    }
                    _ => Err(EngineError::BadRequest(
                        "2-asset scene requires expr=sar_rgb, ndvi, ndwi, nbr, savi, ratio, diff, or custom expression"
                            .to_string(),
                    )),
                }
            }
            3 => {
                // Fetch all 3 COGs concurrently (e.g., Landsat SR_B4 + SR_B3 + SR_B2!)
                let (t0, t1, t2) = tokio::try_join!(
                    self.store.get_blocks(&cogs[0], ifd_idx, &indices),
                    self.store.get_blocks(&cogs[1], ifd_idx, &indices),
                    self.store.get_blocks(&cogs[2], ifd_idx, &indices),
                )?;
                let r0: Vec<_> = t0
                    .iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                    .collect();
                let r1: Vec<_> = t1
                    .iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                    .collect();
                let r2: Vec<_> = t2
                    .iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                    .collect();
                let srcs = [
                    BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                        grid_h, &r0,
                    ),
                    BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                        grid_h, &r1,
                    ),
                    BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                        grid_h, &r2,
                    ),
                ];
                match &req.bands {
                    BandRequest::Spectral3 {
                        expr,
                        rescale,
                        colormap,
                        ..
                    } => {
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
                        rescale_min,
                        rescale_max,
                        ..
                    } => {
                        let style = RgbStyle {
                            rescale_min: *rescale_min,
                            rescale_max: *rescale_max,
                            resample: req.resample,
                            format: req.format,
                        };
                        Ok(render_rgb_bands::<N>(&plan, &srcs, &style))
                    }
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
                        Ok(render_expr_vm::<N>(&plan, &srcs, prog, &style))
                    }
                    _ => Err(EngineError::BadRequest(
                        "3-asset scene requires RGB, expr=evi, or custom expression".to_string(),
                    )),
                }
            }
            other => Err(EngineError::BadRequest(format!(
                "multi-COG compositing supports 1, 2, or 3 assets, got {other}"
            ))),
        }
    }
}
