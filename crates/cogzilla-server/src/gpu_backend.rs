use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use cogzilla_gpu::{
    DispatchPolicy, DispatchRequest, GpuColormapMode, GpuColormapRenderer, GpuContext,
    GpuDemAlgorithm, GpuOp, GpuPixelOutput, GpuResampler, GpuSlabCache, GpuTerrainRenderer,
    PolicyOverride, SlabKey,
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

    /// Attempt GPU-accelerated single-band rendering; returns `None` to fall back to CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_single_band<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        tx0: u32,
        ty0: u32,
        plan: &CoordPlan<N>,
        src: &BandSource<'_>,
        style: &SingleBandStyle,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample(style.resample)?;
        let CoordPlan::Separable1D { u, v, .. } = plan else {
            return None;
        };
        let (slice, base_col, base_row, stride) = src.single_block_flat()?;
        let out_kind = match style.format {
            OutputFormat::Png => GpuPixelOutput::IndexedU8,
            OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
                GpuPixelOutput::Rgba8
            }
            _ => return None,
        };

        let key = SlabKey {
            dataset_id: cog_id,
            ifd_idx: ifd_idx as u16,
            band: band0,
            tx: tx0,
            ty: ty0,
        };
        let warm_slab = self.slab_cache.get(&key);
        let height = (slice.len() / stride) as u32;
        let width = stride as u32;
        let upload_bytes = if warm_slab.is_some() {
            0
        } else {
            slice.len() * std::mem::size_of::<f32>()
        };
        let readback_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => N * N,
            GpuPixelOutput::Rgba8 => N * N * 4,
        };
        let cur_in_flight = self.in_flight.load(Ordering::Relaxed);
        let decision = self.policy.decide(&DispatchRequest {
            op: gpu_mode.to_policy_op(),
            dst_pixels: N * N,
            upload_bytes,
            readback_bytes,
            batch_size: 1,
            in_flight: cur_in_flight,
        });
        if !decision.use_gpu {
            return None;
        }

        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let _guard = InFlightGuard(&self.in_flight);

        let slab = match warm_slab {
            Some(s) => s,
            None => self
                .slab_cache
                .get_or_upload(key, slice, width, height, stride as u32, &self.resampler)
                .ok()?,
        };

        let bc = base_col as f32;
        let br = base_row as f32;
        let mut u_local = [0.0f32; N];
        let mut v_local = [0.0f32; N];
        for i in 0..N {
            u_local[i] = u[i] - bc;
            v_local[i] = v[i] - br;
        }

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
            GpuPixelOutput::IndexedU8 => encode_indexed_u8_png(&raw, N as u32, N as u32, lut),
            GpuPixelOutput::Rgba8 => {
                if style.format == OutputFormat::RawRgba {
                    Bytes::from(raw)
                } else {
                    encode_rgba_tile(&raw, N as u32, false, style.format)
                }
            }
        })
    }

    /// Attempt GPU-accelerated DEM Horn 3×3 (`Hillshade` / `Slope` / `Aspect`) rendering.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_dem<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        tx0: u32,
        ty0: u32,
        plan: &CoordPlan<N>,
        src: &BandSource<'_>,
        alg: DemAlgorithm,
        pixel_size_m: f32,
        style: &SingleBandStyle,
    ) -> Option<Bytes> {
        let gpu_mode = map_resample(style.resample)?;
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
        let CoordPlan::Separable1D { u, v, .. } = plan else {
            return None;
        };
        let (slice, base_col, base_row, stride) = src.single_block_flat()?;
        let out_kind = match style.format {
            OutputFormat::Png => GpuPixelOutput::IndexedU8,
            OutputFormat::Webp { .. } | OutputFormat::Jpeg { .. } | OutputFormat::RawRgba => {
                GpuPixelOutput::Rgba8
            }
            _ => return None,
        };

        let key = SlabKey {
            dataset_id: cog_id,
            ifd_idx: ifd_idx as u16,
            band: band0,
            tx: tx0,
            ty: ty0,
        };
        let warm_slab = self.slab_cache.get(&key);
        let height = (slice.len() / stride) as u32;
        let width = stride as u32;
        let upload_bytes = if warm_slab.is_some() {
            0
        } else {
            slice.len() * std::mem::size_of::<f32>()
        };
        let readback_bytes = match out_kind {
            GpuPixelOutput::IndexedU8 => N * N,
            GpuPixelOutput::Rgba8 => N * N * 4,
        };
        let op = match gpu_mode {
            GpuColormapMode::Lanczos3 => GpuOp::DemHornLanczos3,
            _ => GpuOp::DemHornCubic,
        };
        let decision = self.policy.decide(&DispatchRequest {
            op,
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

        let slab = match warm_slab {
            Some(s) => s,
            None => self
                .slab_cache
                .get_or_upload(key, slice, width, height, stride as u32, &self.resampler)
                .ok()?,
        };

        let bc = base_col as f32;
        let br = base_row as f32;
        let mut u_local = [0.0f32; N];
        let mut v_local = [0.0f32; N];
        for i in 0..N {
            u_local[i] = u[i] - bc;
            v_local[i] = v[i] - br;
        }

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
                pixel_size_m,
                pixel_size_m,
                style.rescale_min,
                style.rescale_max,
                &lut_u32,
                out_kind,
            )
            .ok()?;

        Some(match out_kind {
            GpuPixelOutput::IndexedU8 => encode_indexed_u8_png(&raw, N as u32, N as u32, lut),
            GpuPixelOutput::Rgba8 => {
                if style.format == OutputFormat::RawRgba {
                    Bytes::from(raw)
                } else {
                    encode_rgba_tile(&raw, N as u32, false, style.format)
                }
            }
        })
    }

    /// Attempt GPU-accelerated `Terrarium` / `TerrainRgb` tile rendering.
    #[allow(clippy::too_many_arguments)]
    pub fn try_render_terrain<const N: usize>(
        &self,
        cog_id: u64,
        ifd_idx: usize,
        band0: u16,
        tx0: u32,
        ty0: u32,
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
        let (slice, base_col, base_row, stride) = src.single_block_flat()?;
        let key = SlabKey {
            dataset_id: cog_id,
            ifd_idx: ifd_idx as u16,
            band: band0,
            tx: tx0,
            ty: ty0,
        };
        let warm_slab = self.slab_cache.get(&key);
        let height = (slice.len() / stride) as u32;
        let width = stride as u32;
        let upload_bytes = if warm_slab.is_some() {
            0
        } else {
            slice.len() * std::mem::size_of::<f32>()
        };
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

        let slab = match warm_slab {
            Some(s) => s,
            None => self
                .slab_cache
                .get_or_upload(key, slice, width, height, stride as u32, &self.resampler)
                .ok()?,
        };

        let bc = base_col as f32;
        let br = base_row as f32;
        let mut u_local = [0.0f32; N];
        let mut v_local = [0.0f32; N];
        for i in 0..N {
            u_local[i] = u[i] - bc;
            v_local[i] = v[i] - br;
        }
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
