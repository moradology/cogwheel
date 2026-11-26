//! Single-submit GPU weighted zonal statistics, shared-memory histogram & percentile CDF.
//!
//! Backs `zonal::compute_native_exact_polygon_stats` and `compute_native_exact_multi_cog_expr_stats`
//! with exact analytical coverage-weighted & ellipsoidal area-weighted reductions in 1 GPU submit
//! and 1 packed 12 KB readback.

use std::sync::{Arc, mpsc};

use bytemuck::{Pod, Zeroable};

use crate::context::{GpuContext, GpuError};
use crate::stats::{CDF_BINS, MAX_WG};

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
            partials_buf,
            resolved_buf,
            hist_buf,
            var_buf,
        }
    }
}

/// Single-submit GPU weighted zonal statistics reducer (batches all bands of a polygon into 1 submit).
pub struct GpuWeightedStatsReducer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bgl: wgpu::BindGroupLayout,
    pipe_pass1: wgpu::ComputePipeline,
    pipe_finalize1: wgpu::ComputePipeline,
    pipe_pass2: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    cov_buf: wgpu::Buffer,
    cov_cap_elems: usize,
    row_area_buf: wgpu::Buffer,
    row_area_cap_elems: usize,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/weighted_stats.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("weighted-stats-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            params_buf,
            cov_buf,
            cov_cap_elems: init_cap,
            row_area_buf,
            row_area_cap_elems: init_cap,
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
        label: &'static str,
    ) {
        if need > *cap {
            let new_cap = need.next_power_of_two().max(1024);
            *buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (new_cap * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *cap = new_cap;
        }
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
        if len == 0 || win_w == 0 || row_areas.is_empty() || bands.is_empty() {
            return Err(GpuError::InvalidDimensions(
                "weighted stats requires non-empty bands, covs, win_w, and row_areas".into(),
            ));
        }
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
            self.slots.push(WeightedBandSlot::new(&self.device, len.next_power_of_two().max(1024)));
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
                "wstats-src",
            );
        }
        Self::ensure_buf(
            &self.device,
            &mut self.cov_buf,
            &mut self.cov_cap_elems,
            len,
            "wstats-cov",
        );
        Self::ensure_buf(
            &self.device,
            &mut self.row_area_buf,
            &mut self.row_area_cap_elems,
            row_areas.len(),
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
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sub_idx));
        rx.recv()
            .map_err(|_| GpuError::BufferMap("wstats channel dropped".into()))?
            .map_err(|e| GpuError::BufferMap(format!("{e:?}")))?;

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
    // Convert quantized weighted histogram counts back to exact coverage units summing to count_cov
    // (matching zonal.rs's weighted histogram counts).
    let raw_hist_sum: f64 = hist_u32[..n_bins].iter().map(|&c| c as f64).sum();
    let hist_scale = if raw_hist_sum > 0.0 {
        let target_sum = if hist_range.is_none() {
            resolved.c_total as f64
        } else if resolved.inv_max_w > 0.0 {
            raw_hist_sum / (resolved.inv_max_w as f64)
        } else {
            0.0
        };
        target_sum / raw_hist_sum
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
