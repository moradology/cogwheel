//! Single-submit GPU weighted zonal statistics, shared-memory histogram & percentile CDF.
//!
//! Backs `zonal::compute_native_exact_polygon_stats` and `compute_native_exact_multi_cog_expr_stats`
//! with exact analytical coverage-weighted & ellipsoidal area-weighted reductions in 1 GPU submit
//! and 1 packed 12 KB readback.

use std::sync::{Arc, mpsc};

use bytemuck::{Pod, Zeroable};

use crate::context::{GpuContext, GpuError};
use crate::expr::GpuInstr;
use crate::stats::{CDF_BINS, MAX_WG};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WeightedExprParams {
    len: u32,
    band_mask: u32,
    pc_start: u32,
    pc_end: u32,
    cov_eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WeightedStatsParams {
    len: u32,
    win_w: u32,
    num_wg: u32,
    num_bins: u32,
    has_hist_range: u32,
    area_weighted: u32,
    h_min: f32,
    h_max: f32,
    cov_eps: f32,
    pixel_set_eps: f32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WeightedPartial {
    min_val: f32,
    max_val: f32,
    w_total: f32,
    w_sum: f32,
    c_total: f32,
    c_sum: f32,
    area_valid: f32,
    max_w: f32,
    valid_px: u32,
    masked_px: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WeightedResolved {
    min_val: f32,
    max_val: f32,
    w_total: f32,
    w_sum: f32,
    mean: f32,
    c_total: f32,
    c_sum: f32,
    area_valid: f32,
    p_inv_span: f32,
    h_min: f32,
    h_max: f32,
    h_inv_span: f32,
    inv_max_w: f32,
    valid_px: u32,
    masked_px: u32,
    _pad0: u32,
}

/// Complete coverage-weighted / area-weighted zonal statistics computed on the GPU.
#[derive(Debug, Clone)]
pub struct GpuWeightedBandStats {
    pub valid_pixels: usize,
    pub masked_pixels: usize,
    pub count_cov: f64,
    pub sum_cov: f64,
    pub w_total: f64,
    pub area_valid_m2: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub majority: f64,
    pub minority: f64,
    pub unique_bins: usize,
    pub percentiles: Vec<(f64, f64)>,
    pub hist_counts: Vec<f64>,
    pub hist_edges: Vec<f64>,
}

const RESOLVED_BYTES: u64 = std::mem::size_of::<WeightedResolved>() as u64;
const HIST_BYTES: u64 = (2048 * std::mem::size_of::<u32>()) as u64;
const VAR_BYTES: u64 = (MAX_WG as u64) * (std::mem::size_of::<f32>() as u64);
const READBACK_BYTES: u64 = RESOLVED_BYTES + HIST_BYTES + VAR_BYTES;

struct WeightedBandSlot {
    src_buf: wgpu::Buffer,
    src_cap_elems: usize,
    expr_params_buf: wgpu::Buffer,
    partials_buf: wgpu::Buffer,
    resolved_buf: wgpu::Buffer,
    hist_buf: wgpu::Buffer,
    var_buf: wgpu::Buffer,
}

impl WeightedBandSlot {
    fn new(device: &wgpu::Device, init_cap: usize) -> Self {
        let src_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-src"),
            size: (init_cap * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let expr_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-expr-params"),
            size: std::mem::size_of::<WeightedExprParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-partials"),
            size: (MAX_WG as u64) * (std::mem::size_of::<WeightedPartial>() as u64),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let resolved_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-resolved"),
            size: RESOLVED_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-hist"),
            size: HIST_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-var"),
            size: VAR_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            src_buf,
            src_cap_elems: init_cap,
            expr_params_buf,
            partials_buf,
            resolved_buf,
            hist_buf,
            var_buf,
        }
    }
}

/// Single-submit GPU weighted zonal statistics reducer (batches all bands or expressions of a
/// polygon into 1 submit).
pub struct GpuWeightedStatsReducer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bgl: wgpu::BindGroupLayout,
    pipe_pass1: wgpu::ComputePipeline,
    pipe_finalize1: wgpu::ComputePipeline,
    pipe_pass2: wgpu::ComputePipeline,
    expr_bgl: wgpu::BindGroupLayout,
    pipe_expr_eval: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    cov_buf: wgpu::Buffer,
    cov_cap_elems: usize,
    row_area_buf: wgpu::Buffer,
    row_area_cap_elems: usize,
    raw_bands_buf: wgpu::Buffer,
    raw_bands_cap_elems: usize,
    instrs_buf: wgpu::Buffer,
    instrs_cap: usize,
    max_storage_bytes: u64,
    slots: Vec<WeightedBandSlot>,
    staging_buf: wgpu::Buffer,
    staging_cap_bands: usize,
}

impl GpuWeightedStatsReducer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-weighted-stats-wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::WEIGHTED_STATS_WGSL.into()),
        });

        use crate::wgpu_util::{bgl_storage_ro, bgl_storage_rw, bgl_uniform};
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("weighted-stats-bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_ro(3),
                bgl_storage_rw(4),
                bgl_storage_rw(5),
                bgl_storage_rw(6),
                bgl_storage_rw(7),
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("weighted-stats-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipe_pass1 = mk("wstats_reduce_pass1");
        let pipe_finalize1 = mk("wstats_finalize_pass1");
        let pipe_pass2 = mk("wstats_hist_pass2");

        let expr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-weighted-expr-wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::WEIGHTED_EXPR_WGSL.into()),
        });
        let expr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("weighted-expr-bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_ro(3),
                bgl_storage_rw(4),
            ],
        });
        let expr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("weighted-expr-pl"),
            bind_group_layouts: &[&expr_bgl],
            push_constant_ranges: &[],
        });
        let pipe_expr_eval = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("weighted_expr_eval"),
            layout: Some(&expr_pl),
            module: &expr_shader,
            entry_point: Some("weighted_expr_eval"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-params"),
            size: std::mem::size_of::<WeightedStatsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let init_cap = 1024usize;
        let cov_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-cov"),
            size: (init_cap * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let row_area_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-row-areas"),
            size: (init_cap * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raw_bands_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-raw-bands"),
            size: (init_cap * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let init_instrs = 64usize;
        let instrs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-expr-instrs"),
            size: (init_instrs * std::mem::size_of::<GpuInstr>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let limits = device.limits();
        let max_storage_bytes =
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        let slots = vec![WeightedBandSlot::new(&device, init_cap)];
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wstats-staging"),
            size: READBACK_BYTES * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            bgl,
            pipe_pass1,
            pipe_finalize1,
            pipe_pass2,
            expr_bgl,
            pipe_expr_eval,
            params_buf,
            cov_buf,
            cov_cap_elems: init_cap,
            row_area_buf,
            row_area_cap_elems: init_cap,
            raw_bands_buf,
            raw_bands_cap_elems: init_cap,
            instrs_buf,
            instrs_cap: init_instrs,
            max_storage_bytes,
            slots,
            staging_buf,
            staging_cap_bands: 4,
        }
    }

    fn ensure_buf(
        device: &wgpu::Device,
        buf: &mut wgpu::Buffer,
        cap: &mut usize,
        need: usize,
        max_elems: usize,
        label: &'static str,
    ) {
        if need > *cap {
            let new_cap = need.next_power_of_two().max(1024).min(max_elems.max(need));
            *buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (new_cap * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *cap = new_cap;
        }
    }

    fn wait_map_async(
        device: &wgpu::Device,
        slice: &wgpu::BufferSlice<'_>,
        sub_idx: wgpu::SubmissionIndex,
    ) -> Result<(), GpuError> {
        let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ready_cb = std::sync::Arc::clone(&ready);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            ready_cb.store(true, std::sync::atomic::Ordering::Release);
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub_idx));
        if !ready.load(std::sync::atomic::Ordering::Acquire) {
            device.poll(wgpu::Maintain::Wait);
        }
        rx.recv()
            .map_err(|_| GpuError::BufferMap("wstats channel dropped".into()))?
            .map_err(|e| GpuError::BufferMap(format!("{e:?}")))
    }

    /// Reduce one or more aligned band buffers sharing the same coverage mask `covs` and
    /// per-row areas `row_areas` (`covs` and `row_areas` are uploaded only ONCE).
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_weighted_bands(
        &mut self,
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
    ) -> Result<Vec<GpuWeightedBandStats>, GpuError> {
        let len = covs.len();
        const MAX_HIST_PIXELS: usize = (u32::MAX / 64) as usize;
        if len == 0
            || len > MAX_HIST_PIXELS
            || win_w == 0
            || row_areas.len() < len.div_ceil(win_w)
            || bands.is_empty()
        {
            return Err(GpuError::InvalidDimensions(
                "weighted stats requires 1..=67108863 pixels, non-empty bands, win_w, and valid row_areas".into(),
            ));
        }
        let needed_bytes = (len as u64).saturating_mul(4);
        if needed_bytes > self.max_storage_bytes {
            return Err(GpuError::InvalidDimensions(format!(
                "weighted stats buffer {needed_bytes} B exceeds adapter limit {} B",
                self.max_storage_bytes
            )));
        }
        let max_elems = (self.max_storage_bytes / 4) as usize;
        for b in bands {
            if b.len() != len {
                return Err(GpuError::InvalidDimensions(
                    "weighted stats band length mismatch".into(),
                ));
            }
        }
        let n_bins = num_bins.clamp(1, 1000);
        let num_wg = ((len as u32).div_ceil(256)).clamp(1, MAX_WG);
        let n_bands = bands.len();

        while self.slots.len() < n_bands {
            self.slots.push(WeightedBandSlot::new(
                &self.device,
                len.next_power_of_two().max(1024).min(max_elems),
            ));
        }
        if n_bands > self.staging_cap_bands {
            self.staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wstats-staging"),
                size: READBACK_BYTES * (n_bands as u64),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.staging_cap_bands = n_bands;
        }
        for slot in self.slots.iter_mut().take(n_bands) {
            Self::ensure_buf(
                &self.device,
                &mut slot.src_buf,
                &mut slot.src_cap_elems,
                len,
                max_elems,
                "wstats-src",
            );
        }
        Self::ensure_buf(
            &self.device,
            &mut self.cov_buf,
            &mut self.cov_cap_elems,
            len,
            max_elems,
            "wstats-cov",
        );
        Self::ensure_buf(
            &self.device,
            &mut self.row_area_buf,
            &mut self.row_area_cap_elems,
            row_areas.len(),
            max_elems,
            "wstats-row-areas",
        );

        // Upload shared coverage + row areas + params ONCE across all bands!
        self.queue
            .write_buffer(&self.cov_buf, 0, bytemuck::cast_slice(covs));
        self.queue
            .write_buffer(&self.row_area_buf, 0, bytemuck::cast_slice(row_areas));

        let (has_hist_range, h_min, h_max) = match hist_range {
            Some((lo, hi)) => (1u32, lo as f32, hi as f32),
            None => (0u32, 0.0, 0.0),
        };
        let params = WeightedStatsParams {
            len: len as u32,
            win_w: win_w as u32,
            num_wg,
            num_bins: n_bins as u32,
            has_hist_range,
            area_weighted: u32::from(area_weighted),
            h_min,
            h_max,
            cov_eps,
            pixel_set_eps,
            _pad0: 0,
            _pad1: 0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut bind_groups = Vec::with_capacity(n_bands);
        for (bi, &band) in bands.iter().enumerate() {
            let slot = &self.slots[bi];
            self.queue
                .write_buffer(&slot.src_buf, 0, bytemuck::cast_slice(band));
            bind_groups.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wstats-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: slot.src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.cov_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.row_area_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: slot.partials_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: slot.resolved_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: slot.hist_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: slot.var_buf.as_entire_binding(),
                    },
                ],
            }));
        }

        // Encode ALL B bands into ONE CommandEncoder -> ONE submit -> ONE map_async!
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wstats-batched-enc"),
            });
        for slot in self.slots.iter().take(n_bands) {
            enc.clear_buffer(&slot.hist_buf, 0, None);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wstats-3stage-batched"),
                timestamp_writes: None,
            });
            for bg in &bind_groups {
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(&self.pipe_pass1);
                pass.dispatch_workgroups(num_wg, 1, 1);

                pass.set_pipeline(&self.pipe_finalize1);
                pass.dispatch_workgroups(1, 1, 1);

                pass.set_pipeline(&self.pipe_pass2);
                pass.dispatch_workgroups(num_wg, 1, 1);
            }
        }
        let var_copy_bytes = (num_wg as u64) * 4;
        let per_band_bytes = RESOLVED_BYTES + HIST_BYTES + var_copy_bytes;
        for (bi, slot) in self.slots.iter().take(n_bands).enumerate() {
            let dst_off = (bi as u64) * per_band_bytes;
            enc.copy_buffer_to_buffer(&slot.resolved_buf, 0, &self.staging_buf, dst_off, RESOLVED_BYTES);
            enc.copy_buffer_to_buffer(
                &slot.hist_buf,
                0,
                &self.staging_buf,
                dst_off + RESOLVED_BYTES,
                HIST_BYTES,
            );
            enc.copy_buffer_to_buffer(
                &slot.var_buf,
                0,
                &self.staging_buf,
                dst_off + RESOLVED_BYTES + HIST_BYTES,
                var_copy_bytes,
            );
        }

        let sub_idx = self.queue.submit(Some(enc.finish()));
        let total_readback = per_band_bytes * (n_bands as u64);
        let slice = self.staging_buf.slice(0..total_readback);
        Self::wait_map_async(&self.device, &slice, sub_idx)?;

        let mapped = slice.get_mapped_range();
        let stride_bytes = per_band_bytes as usize;
        let mut results = Vec::with_capacity(n_bands);
        for bi in 0..n_bands {
            let off = bi * stride_bytes;
            results.push(decode_weighted_readback(
                &mapped[off..off + stride_bytes],
                num_wg as usize,
                n_bins,
                hist_range,
                percentiles,
            ));
        }
        drop(mapped);
        self.staging_buf.unmap();

        Ok(results)
    }

    /// 1-pass fused GPU expression VM + exact weighted zonal statistics across `E` expressions
    /// over `1..=8` shared raw input bands in **1 `CommandEncoder` → 1 `queue.submit()` →
    /// 1 `map_async()`** (uploads `raw_bands`, `covs`, and `row_areas` ONCE; zero CPU expression
    /// evaluation, zero `E × N` CPU expression buffers, and only `E × 12 KB` readback).
    ///
    /// Each entry in `exprs` is `(instructions, band_mask)` where `LoadBand(slot)` indexes
    /// `raw_bands[slot]` (`0..raw_bands.len()`) and `band_mask` has bit `slot` set for every
    /// band referenced by that expression (so pixels where any referenced band is non-finite are
    /// masked out to match CPU `all_valid` semantics).
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_weighted_exprs(
        &mut self,
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
    ) -> Result<Vec<GpuWeightedBandStats>, GpuError> {
        let len = covs.len();
        const MAX_HIST_PIXELS: usize = (u32::MAX / 64) as usize;
        if len == 0
            || len > MAX_HIST_PIXELS
            || win_w == 0
            || row_areas.len() < len.div_ceil(win_w)
            || raw_bands.is_empty()
            || raw_bands.len() > 8
            || exprs.is_empty()
        {
            return Err(GpuError::InvalidDimensions(
                "weighted expr stats requires 1..=8 raw bands, non-empty exprs, covs, win_w, and valid row_areas".into(),
            ));
        }
        let max_elems = (self.max_storage_bytes / 4) as usize;
        for b in raw_bands {
            if b.len() != len {
                return Err(GpuError::InvalidDimensions(
                    "weighted expr stats raw band length mismatch".into(),
                ));
            }
        }
        let total_raw_elems = len
            .checked_mul(raw_bands.len())
            .ok_or_else(|| GpuError::InvalidDimensions("raw bands elems overflow".into()))?;
        let total_raw_bytes = (total_raw_elems as u64) * 4;
        if total_raw_bytes > self.max_storage_bytes {
            return Err(GpuError::InvalidDimensions(format!(
                "raw bands buffer {total_raw_bytes} B exceeds adapter limit {} B",
                self.max_storage_bytes
            )));
        }
        let total_instrs: usize = exprs.iter().map(|(ins, _)| ins.len()).sum();
        if total_instrs == 0 {
            return Err(GpuError::InvalidDimensions(
                "weighted expr stats requires non-empty instructions".into(),
            ));
        }

        let n_bins = num_bins.clamp(1, 1000);
        let num_wg = ((len as u32).div_ceil(256)).clamp(1, MAX_WG);
        let expr_wg = ((len as u32).div_ceil(256)).clamp(1, 65535);
        let n_exprs = exprs.len();

        while self.slots.len() < n_exprs {
            self.slots.push(WeightedBandSlot::new(
                &self.device,
                len.next_power_of_two().max(1024).min(max_elems),
            ));
        }
        if n_exprs > self.staging_cap_bands {
            self.staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wstats-staging"),
                size: READBACK_BYTES * (n_exprs as u64),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.staging_cap_bands = n_exprs;
        }
        for slot in self.slots.iter_mut().take(n_exprs) {
            Self::ensure_buf(
                &self.device,
                &mut slot.src_buf,
                &mut slot.src_cap_elems,
                len,
                max_elems,
                "wstats-src",
            );
        }
        Self::ensure_buf(
            &self.device,
            &mut self.cov_buf,
            &mut self.cov_cap_elems,
            len,
            max_elems,
            "wstats-cov",
        );
        Self::ensure_buf(
            &self.device,
            &mut self.row_area_buf,
            &mut self.row_area_cap_elems,
            row_areas.len(),
            max_elems,
            "wstats-row-areas",
        );
        if total_raw_elems > self.raw_bands_cap_elems {
            self.raw_bands_cap_elems = total_raw_elems;
            self.raw_bands_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wstats-raw-bands"),
                size: total_raw_bytes.max(256),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if total_instrs > self.instrs_cap {
            self.instrs_cap = total_instrs.next_power_of_two().max(64);
            self.instrs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wstats-expr-instrs"),
                size: (self.instrs_cap * std::mem::size_of::<GpuInstr>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Upload shared raw bands + coverage + row areas + packed instructions ONCE!
        let band_bytes = (len * 4) as u64;
        for (bi, &band) in raw_bands.iter().enumerate() {
            self.queue.write_buffer(
                &self.raw_bands_buf,
                (bi as u64) * band_bytes,
                bytemuck::cast_slice(band),
            );
        }
        self.queue
            .write_buffer(&self.cov_buf, 0, bytemuck::cast_slice(covs));
        self.queue
            .write_buffer(&self.row_area_buf, 0, bytemuck::cast_slice(row_areas));

        let mut packed_instrs = Vec::with_capacity(total_instrs);
        let mut expr_ranges = Vec::with_capacity(n_exprs);
        for &(ins, band_mask) in exprs {
            let start = packed_instrs.len() as u32;
            packed_instrs.extend_from_slice(ins);
            let end = packed_instrs.len() as u32;
            expr_ranges.push((start, end, band_mask));
        }
        self.queue
            .write_buffer(&self.instrs_buf, 0, bytemuck::cast_slice(&packed_instrs));

        let (has_hist_range, h_min, h_max) = match hist_range {
            Some((lo, hi)) => (1u32, lo as f32, hi as f32),
            None => (0u32, 0.0, 0.0),
        };
        let params = WeightedStatsParams {
            len: len as u32,
            win_w: win_w as u32,
            num_wg,
            num_bins: n_bins as u32,
            has_hist_range,
            area_weighted: u32::from(area_weighted),
            h_min,
            h_max,
            cov_eps,
            pixel_set_eps,
            _pad0: 0,
            _pad1: 0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut expr_bgs = Vec::with_capacity(n_exprs);
        let mut stats_bgs = Vec::with_capacity(n_exprs);
        for (ei, &(pc_start, pc_end, band_mask)) in expr_ranges.iter().enumerate() {
            let slot = &self.slots[ei];
            let ep = WeightedExprParams {
                len: len as u32,
                band_mask,
                pc_start,
                pc_end,
                cov_eps,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            self.queue
                .write_buffer(&slot.expr_params_buf, 0, bytemuck::bytes_of(&ep));
            expr_bgs.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wstats-expr-bg"),
                layout: &self.expr_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: slot.expr_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.raw_bands_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.cov_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.instrs_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: slot.src_buf.as_entire_binding(),
                    },
                ],
            }));
            stats_bgs.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wstats-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: slot.src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.cov_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.row_area_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: slot.partials_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: slot.resolved_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: slot.hist_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: slot.var_buf.as_entire_binding(),
                    },
                ],
            }));
        }

        // Encode ALL E expression evals + 3-stage weighted stats into ONE CommandEncoder -> ONE submit!
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wstats-expr-batched-enc"),
            });
        for slot in self.slots.iter().take(n_exprs) {
            enc.clear_buffer(&slot.hist_buf, 0, None);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wstats-expr-eval-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_expr_eval);
            for bg in &expr_bgs {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(expr_wg, 1, 1);
            }
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wstats-3stage-batched"),
                timestamp_writes: None,
            });
            for bg in &stats_bgs {
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(&self.pipe_pass1);
                pass.dispatch_workgroups(num_wg, 1, 1);

                pass.set_pipeline(&self.pipe_finalize1);
                pass.dispatch_workgroups(1, 1, 1);

                pass.set_pipeline(&self.pipe_pass2);
                pass.dispatch_workgroups(num_wg, 1, 1);
            }
        }
        let var_copy_bytes = (num_wg as u64) * 4;
        let per_band_bytes = RESOLVED_BYTES + HIST_BYTES + var_copy_bytes;
        for (ei, slot) in self.slots.iter().take(n_exprs).enumerate() {
            let dst_off = (ei as u64) * per_band_bytes;
            enc.copy_buffer_to_buffer(
                &slot.resolved_buf,
                0,
                &self.staging_buf,
                dst_off,
                RESOLVED_BYTES,
            );
            enc.copy_buffer_to_buffer(
                &slot.hist_buf,
                0,
                &self.staging_buf,
                dst_off + RESOLVED_BYTES,
                HIST_BYTES,
            );
            enc.copy_buffer_to_buffer(
                &slot.var_buf,
                0,
                &self.staging_buf,
                dst_off + RESOLVED_BYTES + HIST_BYTES,
                var_copy_bytes,
            );
        }

        let sub_idx = self.queue.submit(Some(enc.finish()));
        let total_readback = per_band_bytes * (n_exprs as u64);
        let slice = self.staging_buf.slice(0..total_readback);
        Self::wait_map_async(&self.device, &slice, sub_idx)?;

        let mapped = slice.get_mapped_range();
        let stride_bytes = per_band_bytes as usize;
        let mut results = Vec::with_capacity(n_exprs);
        for ei in 0..n_exprs {
            let off = ei * stride_bytes;
            results.push(decode_weighted_readback(
                &mapped[off..off + stride_bytes],
                num_wg as usize,
                n_bins,
                hist_range,
                percentiles,
            ));
        }
        drop(mapped);
        self.staging_buf.unmap();

        Ok(results)
    }
}

fn decode_weighted_readback(
    raw: &[u8],
    num_wg: usize,
    n_bins: usize,
    hist_range: Option<(f64, f64)>,
    percentiles: &[f64],
) -> GpuWeightedBandStats {
    let rb = RESOLVED_BYTES as usize;
    let hb = HIST_BYTES as usize;
    let resolved: &WeightedResolved = bytemuck::from_bytes(&raw[..rb]);
    let hist_u32: &[u32] = bytemuck::cast_slice(&raw[rb..rb + hb]);
    let var_f32: &[f32] = bytemuck::cast_slice(&raw[rb + hb..rb + hb + num_wg * 4]);

    let valid_pixels = resolved.valid_px as usize;
    let masked_pixels = resolved.masked_px as usize;
    let w_total = resolved.w_total as f64;
    if valid_pixels == 0 || w_total <= 0.0 {
        return GpuWeightedBandStats {
            valid_pixels: 0,
            masked_pixels,
            count_cov: 0.0,
            sum_cov: 0.0,
            w_total: 0.0,
            area_valid_m2: 0.0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            median: 0.0,
            majority: 0.0,
            minority: 0.0,
            unique_bins: 0,
            percentiles: percentiles.iter().map(|&p| (p, 0.0)).collect(),
            hist_counts: vec![0.0; n_bins],
            hist_edges: vec![0.0; n_bins + 1],
        };
    }

    let min_v = resolved.min_val as f64;
    let max_v = resolved.max_val as f64;
    let mean = resolved.mean as f64;
    let var_sum: f64 = var_f32.iter().map(|&v| v as f64).sum();
    let std = (var_sum / w_total).max(0.0).sqrt();

    let cdf_u32 = &hist_u32[1024..2048];
    let total_cdf_w: u64 = cdf_u32.iter().map(|&c| c as u64).sum();
    let span = max_v - min_v;
    let bin_w = span / (CDF_BINS as f64);

    let quantile = |pct: f64| -> f64 {
        if span <= 1e-20 || total_cdf_w == 0 {
            return min_v;
        }
        let target = (pct.clamp(0.0, 100.0) / 100.0) * (total_cdf_w as f64);
        let mut cum = 0.0_f64;
        for (i, &cw) in cdf_u32.iter().enumerate() {
            let cwf = cw as f64;
            if cwf == 0.0 {
                continue;
            }
            if cum + cwf >= target {
                let frac = ((target - cum) / cwf).clamp(0.0, 1.0);
                return (min_v + (i as f64 + frac) * bin_w).clamp(min_v, max_v);
            }
            cum += cwf;
        }
        max_v
    };

    let median = quantile(50.0);
    let pct_out: Vec<(f64, f64)> = percentiles.iter().map(|&p| (p, quantile(p))).collect();

    let mut unique_bins = 0usize;
    let mut maj_bin = 0usize;
    let mut maj_w = 0u32;
    let mut min_bin = 0usize;
    let mut min_w = u32::MAX;
    for (i, &cw) in cdf_u32.iter().enumerate() {
        if cw > 0 {
            unique_bins += 1;
            if cw > maj_w {
                maj_w = cw;
                maj_bin = i;
            }
            if cw < min_w {
                min_w = cw;
                min_bin = i;
            }
        }
    }
    let majority = if span <= 1e-20 {
        min_v
    } else {
        min_v + (maj_bin as f64 + 0.5) * bin_w
    };
    let minority = if span <= 1e-20 {
        min_v
    } else {
        min_v + (min_bin as f64 + 0.5) * bin_w
    };

    let (h_lo, h_hi) = match hist_range {
        Some((lo, hi)) => (lo, hi),
        None => (min_v, max_v),
    };
    let h_span = h_hi - h_lo;
    let hist_edges: Vec<f64> = (0..=n_bins)
        .map(|i| h_lo + h_span * (i as f64) / (n_bins as f64))
        .collect();
    // Convert quantized coverage histogram counts (scaled by 64.0 in shader) back to coverage units
    // matching zonal.rs's coverage-weighted histogram counts.
    let raw_hist_sum: f64 = hist_u32[..n_bins].iter().map(|&c| c as f64).sum();
    let hist_scale = if raw_hist_sum > 0.0 {
        if hist_range.is_none() {
            (resolved.c_total as f64) / raw_hist_sum
        } else {
            1.0 / 64.0
        }
    } else {
        0.0
    };
    let hist_counts: Vec<f64> = hist_u32[..n_bins]
        .iter()
        .map(|&c| (c as f64) * hist_scale)
        .collect();

    GpuWeightedBandStats {
        valid_pixels,
        masked_pixels,
        count_cov: resolved.c_total as f64,
        sum_cov: resolved.c_sum as f64,
        w_total,
        area_valid_m2: resolved.area_valid as f64,
        min: min_v,
        max: max_v,
        mean,
        std,
        median,
        majority,
        minority,
        unique_bins,
        percentiles: pct_out,
        hist_counts,
        hist_edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::opcodes;

    #[test]
    fn gpu_weighted_expr_stats_matches_two_pass() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU weighted expr stats test (no adapter): {e}");
                return;
            }
        };
        let mut red = GpuWeightedStatsReducer::new(&ctx);

        let w = 256usize;
        let h = 256usize;
        let n = w * h;
        let mut b0 = vec![0.0f32; n];
        let mut b1 = vec![0.0f32; n];
        let mut b2 = vec![0.0f32; n];
        let mut b3 = vec![0.0f32; n];
        let mut cov = vec![0.0f32; n];
        let row_areas: Vec<f32> = (0..h)
            .map(|r| 0.85 + 0.15 * (r as f32 / h as f32))
            .collect();

        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let dx = (x as f32) - 128.0;
                let dy = (y as f32) - 128.0;
                let dist = (dx * dx + dy * dy).sqrt();
                cov[i] = if dist < 90.0 {
                    1.0
                } else if dist < 96.0 {
                    (96.0 - dist) / 6.0
                } else {
                    0.0
                };
                if i % 37 == 0 {
                    b2[i] = f32::NAN;
                } else {
                    b2[i] = 500.0 + ((i * 13) % 1500) as f32;
                }
                b0[i] = 200.0 + ((i * 7) % 900) as f32;
                b1[i] = 300.0 + ((i * 11) % 1200) as f32;
                b3[i] = 1000.0 + ((i * 17) % 3000) as f32;
            }
        }

        // Expr 0: NDVI = (b3 - b2) / (b3 + b2), uses bands {2, 3} -> mask 0b1100
        let ndvi_ins = vec![
            GpuInstr::load_band(3),
            GpuInstr::load_band(2),
            GpuInstr::op(opcodes::SUB),
            GpuInstr::load_band(3),
            GpuInstr::load_band(2),
            GpuInstr::op(opcodes::ADD),
            GpuInstr::op(opcodes::DIV),
        ];
        // Expr 1: NDWI = (b1 - b3) / (b1 + b3), uses bands {1, 3} -> mask 0b1010
        let ndwi_ins = vec![
            GpuInstr::load_band(1),
            GpuInstr::load_band(3),
            GpuInstr::op(opcodes::SUB),
            GpuInstr::load_band(1),
            GpuInstr::load_band(3),
            GpuInstr::op(opcodes::ADD),
            GpuInstr::op(opcodes::DIV),
        ];
        // Expr 2: EVI-like = 2.5 * (b3 - b2) / (b3 + 6.0 * b2 - 7.5 * b0 + 10000.0), uses {0, 2, 3} -> mask 0b1101
        let evi_ins = vec![
            GpuInstr::push_const(2.5),
            GpuInstr::load_band(3),
            GpuInstr::load_band(2),
            GpuInstr::op(opcodes::SUB),
            GpuInstr::op(opcodes::MUL),
            GpuInstr::load_band(3),
            GpuInstr::push_const(6.0),
            GpuInstr::load_band(2),
            GpuInstr::op(opcodes::MUL),
            GpuInstr::op(opcodes::ADD),
            GpuInstr::push_const(7.5),
            GpuInstr::load_band(0),
            GpuInstr::op(opcodes::MUL),
            GpuInstr::op(opcodes::SUB),
            GpuInstr::push_const(10000.0),
            GpuInstr::op(opcodes::ADD),
            GpuInstr::op(opcodes::DIV),
        ];

        // CPU reference evaluated bands
        let mut ndvi_cpu = vec![f32::NAN; n];
        let mut ndwi_cpu = vec![f32::NAN; n];
        let mut evi_cpu = vec![f32::NAN; n];
        for i in 0..n {
            if cov[i] <= 1e-12 {
                continue;
            }
            if b2[i].is_finite() && b3[i].is_finite() {
                ndvi_cpu[i] = (b3[i] - b2[i]) / (b3[i] + b2[i]);
            }
            if b1[i].is_finite() && b3[i].is_finite() {
                ndwi_cpu[i] = (b1[i] - b3[i]) / (b1[i] + b3[i]);
            }
            if b0[i].is_finite() && b2[i].is_finite() && b3[i].is_finite() {
                evi_cpu[i] =
                    2.5 * (b3[i] - b2[i]) / (b3[i] + 6.0 * b2[i] - 7.5 * b0[i] + 10000.0);
            }
        }

        let fused = red
            .reduce_weighted_exprs(
                &[&b0, &b1, &b2, &b3],
                &[
                    (&ndvi_ins, 0b1100),
                    (&ndwi_ins, 0b1010),
                    (&evi_ins, 0b1101),
                ],
                &cov,
                w,
                &row_areas,
                true,
                1e-12,
                1e-12,
                10,
                None,
                &[2.0, 98.0],
            )
            .unwrap();
        let ref_2pass = red
            .reduce_weighted_bands(
                &[&ndvi_cpu, &ndwi_cpu, &evi_cpu],
                &cov,
                w,
                &row_areas,
                true,
                1e-12,
                1e-12,
                10,
                None,
                &[2.0, 98.0],
            )
            .unwrap();

        assert_eq!(fused.len(), 3);
        for (ei, (g, r)) in fused.iter().zip(&ref_2pass).enumerate() {
            assert_eq!(g.valid_pixels, r.valid_pixels, "expr {ei} valid_pixels");
            assert_eq!(g.masked_pixels, r.masked_pixels, "expr {ei} masked_pixels");
            assert!((g.min - r.min).abs() < 1e-5, "expr {ei} min {} vs {}", g.min, r.min);
            assert!((g.max - r.max).abs() < 1e-5, "expr {ei} max {} vs {}", g.max, r.max);
            assert!((g.mean - r.mean).abs() < 1e-5, "expr {ei} mean {} vs {}", g.mean, r.mean);
            assert!((g.std - r.std).abs() < 1e-5, "expr {ei} std {} vs {}", g.std, r.std);
            assert!(
                (g.area_valid_m2 - r.area_valid_m2).abs() < 1e-2,
                "expr {ei} area {} vs {}",
                g.area_valid_m2,
                r.area_valid_m2
            );
        }
    }
}
