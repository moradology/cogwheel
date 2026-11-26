use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use cogzilla_gpu::{
    DispatchPolicy, DispatchRequest, GpuColormapMode, GpuColormapRenderer, GpuContext,
    GpuCoordPlan, GpuDemAlgorithm, GpuOp, GpuPixelOutput, GpuResampler, GpuSlab, GpuSlabCache,
    GpuTerrainRenderer, PolicyOverride, SlabKey,
};
use cogzilla_kernel::png_fast::{encode_indexed_png, filter_sub_u8_row};
use cogzilla_kernel::{
    BandSource, ColormapLut, DemAlgorithm, OutputFormat, ResampleMode, SingleBandStyle,
    TerrainEncoding, encode_rgba_tile,
};
use cogzilla_proj::CoordPlan;

/// Optional GPU acceleration backend owned by `TileEngine` when built with `--features gpu`.
pub struct GpuBackend {
    pub ctx: GpuContext,
    pub policy: DispatchPolicy,
    pub slab_cache: GpuSlabCache,
    resampler: GpuResampler,
    colormap: Mutex<GpuColormapRenderer>,
    terrain: Mutex<GpuTerrainRenderer>,
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
        ResampleMode::Bilinear => Some(GpuColormapMode::Bilinear),
        ResampleMode::Cubic => Some(GpuColormapMode::Cubic),
        ResampleMode::Lanczos => Some(GpuColormapMode::Lanczos3),
        _ => None,
    }
}

fn build_plte_nodata(lut: &ColormapLut) -> [u8; 768] {
    let mut plte = [0u8; 768];
    for i in 1..256 {
        let src_idx = ((i - 1) * 255 + 127) / 254;
        plte[i * 3] = lut.rgb[src_idx][0];
        plte[i * 3 + 1] = lut.rgb[src_idx][1];
        plte[i * 3 + 2] = lut.rgb[src_idx][2];
    }
    plte
}

fn encode_indexed_u8_png(idx_bytes: &[u8], width: u32, height: u32, lut: &ColormapLut) -> Bytes {
    let w = width as usize;
    let h = height as usize;
    let mut filtered = vec![0u8; h * (1 + w)];
    for py in 0..h {
        let src_row = &idx_bytes[py * w..(py + 1) * w];
        let dst_row = &mut filtered[py * (1 + w)..(py + 1) * (1 + w)];
        filter_sub_u8_row(src_row, dst_row);
    }
    let plte = build_plte_nodata(lut);
    let trns = [0u8];
    encode_indexed_png(width, height, &plte, Some(&trns), &filtered)
}

/// Stitch any single- or multi-block `BandSource` (F32, U16, U8, sparse blocks, nodata override,
/// scale/offset) into a contiguous `f32` slab `(data, base_col, base_row, width, height)`.
pub fn stitch_band_source(src: &BandSource<'_>) -> Option<(Vec<f32>, u32, u32, u32, u32)> {
    let base_col = src.tx0.checked_mul(src.tile_w)?;
    let base_row = src.ty0.checked_mul(src.tile_h)?;
    let max_w = (src.ifd_w.max(0) as u32).saturating_sub(base_col);
    let max_h = (src.ifd_h.max(0) as u32).saturating_sub(base_row);
    let w = src.grid_w.saturating_mul(src.tile_w).min(max_w);
    let h = src.grid_h.saturating_mul(src.tile_h).min(max_h);
    if w == 0 || h == 0 || (w as u64) * (h as u64) > 64 * 1024 * 1024 {
        return None;
    }

    let mut out = vec![f32::NAN; (w as usize) * (h as usize)];
    for row in 0..h {
        let dst_row = &mut out[(row as usize) * (w as usize)..((row + 1) as usize) * (w as usize)];
        let abs_y = base_row + row;
        for col in 0..w {
            dst_row[col as usize] = src.sample_inbounds(base_col + col, abs_y);
        }
    }
    Some((out, base_col, base_row, w, h))
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
        let colormap = Mutex::new(GpuColormapRenderer::new(&ctx));
        let terrain = Mutex::new(GpuTerrainRenderer::new(&ctx));
        Some(Arc::new(Self {
            ctx,
            policy,
            slab_cache: GpuSlabCache::new(slab_cache_bytes),
            resampler,
            colormap,
            terrain,
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
        let (stitched, bc, br, w, h) = stitch_band_source(src)?;
        let slab = self
            .slab_cache
            .get_or_upload(key, &stitched, w, h, w, &self.resampler)
            .ok()?;
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
        let gpu_mode = map_resample(style.resample)?;
        let dst_w = u_coords.len();
        let dst_h = v_coords.len();
        if dst_w == 0 || dst_h == 0 {
            return None;
        }
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
        let readback_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => dst_w * dst_h,
            GpuPixelOutput::Rgba8 => dst_w * dst_h * 4,
        };
        let decision = self.policy.decide(&DispatchRequest {
            op: gpu_mode.to_policy_op(),
            dst_pixels: dst_w * dst_h,
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

        let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
        let bc = base_col as f32;
        let br = base_row as f32;
        let u_local: Vec<f32> = u_coords.iter().map(|&u| u - bc).collect();
        let v_local: Vec<f32> = v_coords.iter().map(|&v| v - br).collect();

        let lut = style.colormap.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
        let raw = self
            .colormap
            .lock()
            .ok()?
            .render_slab(
                &slab,
                &u_local,
                &v_local,
                gpu_mode,
                style.rescale_min,
                style.rescale_max,
                &lut_u32,
                out_kind,
            )
            .ok()?;

        Some(match out_kind {
            GpuPixelOutput::IndexedU8 => {
                encode_indexed_u8_png(&raw, dst_w as u32, dst_h as u32, lut)
            }
            GpuPixelOutput::Rgba8 => {
                if style.format == OutputFormat::RawRgba {
                    Bytes::from(raw)
                } else {
                    encode_rgba_tile(&raw, dst_w as u32, false, style.format)
                }
            }
        })
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
        let gpu_mode = map_resample(style.resample)?;
        let dst_w = u_coords.len();
        let dst_h = v_coords.len();
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
            OutputFormat::Png => GpuPixelOutput::IndexedU8,
            OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
                GpuPixelOutput::Rgba8
            }
            _ => return None,
        };

        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src);
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
            batch_size: 1,
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
        let u_local: Vec<f32> = u_coords.iter().map(|&u| u - bc).collect();
        let v_local: Vec<f32> = v_coords.iter().map(|&v| v - br).collect();

        let lut = style.colormap.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
        let raw = self
            .terrain
            .lock()
            .ok()?
            .render_slab(
                &slab,
                &u_local,
                &v_local,
                gpu_mode,
                gpu_alg,
                pixel_size_x_m,
                pixel_size_y_m,
                style.rescale_min,
                style.rescale_max,
                &lut_u32,
                out_kind,
            )
            .ok()?;

        Some(match out_kind {
            GpuPixelOutput::IndexedU8 => {
                encode_indexed_u8_png(&raw, dst_w as u32, dst_h as u32, lut)
            }
            GpuPixelOutput::Rgba8 => {
                if style.format == OutputFormat::RawRgba {
                    Bytes::from(raw)
                } else {
                    encode_rgba_tile(&raw, dst_w as u32, false, style.format)
                }
            }
        })
    }

    /// Attempt GPU-accelerated single-band tile rendering (single- or multi-block, any dtype,
    /// and all `CoordPlan` variants: `Separable1D`, `Grid2D` 17×17 UTM/Albers/LCC, and `Exact2D`).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_single_band<const N: usize>(
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
        match plan {
            CoordPlan::Separable1D { u, v, .. } => self.try_render_crop_single_band(
                cog_id,
                ifd_idx,
                band0,
                u.as_slice(),
                v.as_slice(),
                src,
                style,
            ),
            CoordPlan::Grid2D { grid_u, grid_v, .. } => {
                let gpu_mode = map_resample(style.resample)?;
                let out_kind = match style.format {
                    OutputFormat::Png => GpuPixelOutput::IndexedU8,
                    OutputFormat::Webp { .. }
                    | OutputFormat::Jpeg { .. }
                    | OutputFormat::RawRgba => GpuPixelOutput::Rgba8,
                    _ => return None,
                };
                let key = source_slab_key(cog_id, ifd_idx, band0, src);
                let warm_slab = self.slab_cache.get(&key);
                let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src);
                let readback_bytes = match out_kind {
                    GpuPixelOutput::IndexedU8 => N * N,
                    GpuPixelOutput::Rgba8 => N * N * 4,
                };
                let decision = self.policy.decide(&DispatchRequest {
                    op: gpu_mode.to_policy_op(),
                    dst_pixels: N * N,
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
                let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
                let bc = base_col as f32;
                let br = base_row as f32;
                let mut gu = [0.0f32; 289];
                let mut gv = [0.0f32; 289];
                for i in 0..289 {
                    gu[i] = grid_u[i] - bc;
                    gv[i] = grid_v[i] - br;
                }
                let lut = style.colormap.lut();
                let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
                let raw = self
                    .colormap
                    .lock()
                    .ok()?
                    .render_slab_with_plan(
                        &slab,
                        GpuCoordPlan::Grid2D {
                            dst_w: N,
                            dst_h: N,
                            grid_u: &gu,
                            grid_v: &gv,
                        },
                        gpu_mode,
                        style.rescale_min,
                        style.rescale_max,
                        &lut_u32,
                        out_kind,
                    )
                    .ok()?;
                Some(match out_kind {
                    GpuPixelOutput::IndexedU8 => {
                        encode_indexed_u8_png(&raw, N as u32, N as u32, lut)
                    }
                    GpuPixelOutput::Rgba8 => {
                        if style.format == OutputFormat::RawRgba {
                            Bytes::from(raw)
                        } else {
                            encode_rgba_tile(&raw, N as u32, false, style.format)
                        }
                    }
                })
            }
            CoordPlan::Exact2D { u, v, .. } => {
                let gpu_mode = map_resample(style.resample)?;
                let out_kind = match style.format {
                    OutputFormat::Png => GpuPixelOutput::IndexedU8,
                    OutputFormat::Webp { .. }
                    | OutputFormat::Jpeg { .. }
                    | OutputFormat::RawRgba => GpuPixelOutput::Rgba8,
                    _ => return None,
                };
                let key = source_slab_key(cog_id, ifd_idx, band0, src);
                let warm_slab = self.slab_cache.get(&key);
                let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src) + N * N * 8;
                let readback_bytes = match out_kind {
                    GpuPixelOutput::IndexedU8 => N * N,
                    GpuPixelOutput::Rgba8 => N * N * 4,
                };
                let decision = self.policy.decide(&DispatchRequest {
                    op: gpu_mode.to_policy_op(),
                    dst_pixels: N * N,
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
                let (slab, base_col, base_row) = self.resolve_source_slab(key, warm_slab, src)?;
                let bc = base_col as f32;
                let br = base_row as f32;
                let u_local: Vec<f32> = u.iter().map(|&x| x - bc).collect();
                let v_local: Vec<f32> = v.iter().map(|&y| y - br).collect();
                let lut = style.colormap.lut();
                let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
                let raw = self
                    .colormap
                    .lock()
                    .ok()?
                    .render_slab_with_plan(
                        &slab,
                        GpuCoordPlan::Exact2D {
                            dst_w: N,
                            dst_h: N,
                            u: &u_local,
                            v: &v_local,
                        },
                        gpu_mode,
                        style.rescale_min,
                        style.rescale_max,
                        &lut_u32,
                        out_kind,
                    )
                    .ok()?;
                Some(match out_kind {
                    GpuPixelOutput::IndexedU8 => {
                        encode_indexed_u8_png(&raw, N as u32, N as u32, lut)
                    }
                    GpuPixelOutput::Rgba8 => {
                        if style.format == OutputFormat::RawRgba {
                            Bytes::from(raw)
                        } else {
                            encode_rgba_tile(&raw, N as u32, false, style.format)
                        }
                    }
                })
            }
        }
    }

    /// Attempt GPU-accelerated DEM Horn 3×3 tile rendering (single- or multi-block, any dtype).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_dem<const N: usize>(
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
        let CoordPlan::Separable1D { u, v, .. } = plan else {
            return None;
        };
        self.try_render_crop_dem(
            cog_id,
            ifd_idx,
            band0,
            u.as_slice(),
            v.as_slice(),
            src,
            alg,
            pixel_size_m,
            pixel_size_m,
            style,
        )
    }

    /// Attempt GPU-accelerated `Terrarium` / `TerrainRgb` tile rendering (single- or multi-block).
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_terrain<const N: usize>(
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
        let gpu_mode = map_resample(resample)?;
        let gpu_alg = match encoding {
            TerrainEncoding::Terrarium => GpuDemAlgorithm::Terrarium,
            TerrainEncoding::TerrainRgb => GpuDemAlgorithm::TerrainRgb,
            TerrainEncoding::Contours { .. } => return None,
        };
        if !matches!(
            format,
            OutputFormat::Png | OutputFormat::Webp { .. } | OutputFormat::RawRgba
        ) {
            return None;
        }
        let CoordPlan::Separable1D { u, v, .. } = plan else {
            return None;
        };
        let key = source_slab_key(cog_id, ifd_idx, band0, src);
        let warm_slab = self.slab_cache.get(&key);
        let upload_bytes = Self::estimate_upload_bytes(&warm_slab, src);
        let decision = self.policy.decide(&DispatchRequest {
            op: GpuOp::DemTerrainRgbCubic,
            dst_pixels: N * N,
            upload_bytes,
            readback_bytes: N * N * 4,
            batch_size: 1,
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
        let u_local: Vec<f32> = u.iter().map(|&x| x - bc).collect();
        let v_local: Vec<f32> = v.iter().map(|&y| y - br).collect();
        let dummy_lut = [0u32; 256];
        let raw = self
            .terrain
            .lock()
            .ok()?
            .render_slab(
                &slab,
                &u_local,
                &v_local,
                gpu_mode,
                gpu_alg,
                30.0,
                30.0,
                0.0,
                255.0,
                &dummy_lut,
                GpuPixelOutput::Rgba8,
            )
            .ok()?;

        Some(if format == OutputFormat::RawRgba {
            Bytes::from(raw)
        } else {
            encode_rgba_tile(&raw, N as u32, false, format)
        })
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
        if let Some(rgba) = gpu.try_render_crop_single_band(99, 0, 0, &u_coords, &v_coords, &src, &style) {
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
}
