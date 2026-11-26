//! GPU-accelerated Raster Band Statistics, Shared-Memory Atomic Histogram & Percentile CDF Reducer.
//!
//! Replaces single-threaded CPU `sort_unstable_by` over millions of `f32` pixels with a
//! two-pass WGSL compute reduction:
//! 1. **Pass 1 (`stats_reduce_pass1`)**: Grid-stride workgroup reduction (Kahan summation,
//!    exact `min`, `max`, `valid_count`).
//! 2. **Pass 2 (`stats_hist_pass2`)**: Workgroup-privatized 1024-bin shared-memory atomic
//!    histograms (user `num_bins` histogram + 1024-bin percentile CDF) + exact centered
//!    2-pass variance (`std`), reading back only ~28 KB regardless of input size.

use std::sync::{Arc, mpsc};

use bytemuck::{Pod, Zeroable};

use crate::context::{GpuContext, GpuError};
use crate::resample::GpuSlab;

const MAX_WG: u32 = 1024;
const CDF_BINS: usize = 1024;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct StatsParams {
    len: u32,
    has_nodata: u32,
    num_bins: u32,
    has_hist_range: u32,
    nodata_val: f32,
    mean: f32,
    p_min: f32,
    p_inv_span: f32,
    h_min: f32,
    h_max: f32,
    h_inv_span: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PartialStats {
    min_val: f32,
    max_val: f32,
    sum: f32,
    count: u32,
}

/// Complete TiTiler-compatible band statistics computed on the GPU.
#[derive(Debug, Clone)]
pub struct GpuBandStats {
    pub total_pixels: usize,
    pub valid_pixels: usize,
    pub masked_pixels: usize,
    pub valid_percent: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub sum: f64,
    pub std: Option<f64>,
    pub median: Option<f64>,
    pub majority: Option<f64>,
    pub minority: Option<f64>,
    pub unique_bins: usize,
    pub percentiles: Vec<(f64, Option<f64>)>,
    pub hist_counts: Vec<u64>,
    pub hist_edges: Vec<f64>,
}

/// GPU raster statistics & histogram reducer.
pub struct GpuStatsReducer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bgl_pass1: wgpu::BindGroupLayout,
    bgl_pass2: wgpu::BindGroupLayout,
    pipe_pass1: wgpu::ComputePipeline,
    pipe_pass2: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    src_buf: wgpu::Buffer,
    src_cap_elems: usize,
    partials_buf: wgpu::Buffer,
    partials_staging: wgpu::Buffer,
    hist_buf: wgpu::Buffer,
    hist_staging: wgpu::Buffer,
    var_buf: wgpu::Buffer,
    var_staging: wgpu::Buffer,
}

impl GpuStatsReducer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-stats-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stats.wgsl").into()),
        });

        let bgl_pass1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-stats-bgl1"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl1 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-stats-pl1"),
            bind_group_layouts: &[&bgl_pass1],
            push_constant_ranges: &[],
        });
        let pipe_pass1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-stats-pass1"),
            layout: Some(&pl1),
            module: &shader,
            entry_point: Some("stats_reduce_pass1"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bgl_pass2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-stats-bgl2"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let pl2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-stats-pl2"),
            bind_group_layouts: &[&bgl_pass2],
            push_constant_ranges: &[],
        });
        let pipe_pass2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-stats-pass2"),
            layout: Some(&pl2),
            module: &shader,
            entry_point: Some("stats_hist_pass2"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-params"),
            size: std::mem::size_of::<StatsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let src_cap_elems = 1024 * 1024;
        let src_buf = Self::alloc_src(&device, src_cap_elems);

        let partials_bytes = (MAX_WG as usize * std::mem::size_of::<PartialStats>()) as u64;
        let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-partials"),
            size: partials_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let partials_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-partials-staging"),
            size: partials_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hist_bytes = (2 * CDF_BINS * std::mem::size_of::<u32>()) as u64;
        let hist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-hist"),
            size: hist_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hist_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-hist-staging"),
            size: hist_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let var_bytes = (MAX_WG as usize * std::mem::size_of::<f32>()) as u64;
        let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-var"),
            size: var_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let var_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-var-staging"),
            size: var_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            bgl_pass1,
            bgl_pass2,
            pipe_pass1,
            pipe_pass2,
            params_buf,
            src_buf,
            src_cap_elems,
            partials_buf,
            partials_staging,
            hist_buf,
            hist_staging,
            var_buf,
            var_staging,
        }
    }

    fn alloc_src(device: &wgpu::Device, elems: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-src"),
            size: (elems.max(1) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Compute full band statistics, histogram, and percentiles from a host `f32` slice.
    pub fn reduce_f32(
        &mut self,
        pixels: &[f32],
        nodata: Option<f32>,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Result<GpuBandStats, GpuError> {
        if pixels.is_empty() {
            return Ok(empty_stats(0, percentiles));
        }
        if pixels.len() > self.src_cap_elems {
            self.src_cap_elems = pixels.len().next_power_of_two();
            self.src_buf = Self::alloc_src(&self.device, self.src_cap_elems);
        }
        self.queue
            .write_buffer(&self.src_buf, 0, bytemuck::cast_slice(pixels));
        let ptr = &self.src_buf as *const wgpu::Buffer;
        // SAFETY: `reduce_buffer_inner` only reads `src_buf` and never reallocates `self.src_buf`.
        unsafe {
            self.reduce_buffer_inner(
                &*ptr,
                pixels.len(),
                nodata,
                num_bins,
                hist_range,
                percentiles,
            )
        }
    }

    /// Compute full band statistics directly on a GPU-resident `GpuSlab` (zero CPU→GPU upload!).
    pub fn reduce_slab(
        &mut self,
        slab: &GpuSlab,
        nodata: Option<f32>,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Result<GpuBandStats, GpuError> {
        let len = (slab.stride as usize).saturating_mul(slab.height as usize);
        if len == 0 {
            return Ok(empty_stats(0, percentiles));
        }
        self.reduce_buffer_inner(&slab.buffer, len, nodata, num_bins, hist_range, percentiles)
    }

    fn reduce_buffer_inner(
        &mut self,
        src_buf: &wgpu::Buffer,
        len: usize,
        nodata: Option<f32>,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Result<GpuBandStats, GpuError> {
        let len_u32 = u32::try_from(len)
            .map_err(|_| GpuError::InvalidDimensions("stats len exceeds u32::MAX".into()))?;
        let num_wg = len_u32.div_ceil(256 * 16).clamp(1, MAX_WG);
        let n_bins = num_bins.clamp(1, CDF_BINS) as u32;

        // Pass 1: min / max / sum / valid_count
        let p1 = StatsParams {
            len: len_u32,
            has_nodata: u32::from(nodata.is_some()),
            num_bins: n_bins,
            has_hist_range: u32::from(hist_range.is_some()),
            nodata_val: nodata.unwrap_or(0.0),
            mean: 0.0,
            p_min: 0.0,
            p_inv_span: 1.0,
            h_min: 0.0,
            h_max: 0.0,
            h_inv_span: 1.0,
            _pad: 0.0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p1));

        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-stats-bg1"),
            layout: &self.bgl_pass1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.partials_buf.as_entire_binding(),
                },
            ],
        });

        let p1_bytes = (num_wg as usize * std::mem::size_of::<PartialStats>()) as u64;
        let mut enc1 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-stats-enc1"),
            });
        {
            let mut pass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-stats-pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_pass1);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(num_wg, 1, 1);
        }
        enc1.copy_buffer_to_buffer(&self.partials_buf, 0, &self.partials_staging, 0, p1_bytes);
        self.queue.submit(Some(enc1.finish()));

        let s1 = self.partials_staging.slice(..p1_bytes);
        let (tx1, rx1) = mpsc::channel();
        s1.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx1.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx1.recv()
            .map_err(|e| GpuError::BufferMap(e.to_string()))?
            .map_err(|e| GpuError::BufferMap(e.to_string()))?;

        let mut valid_px = 0usize;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        let mut sum = 0.0f64;
        {
            let mapped = s1.get_mapped_range();
            let partials: &[PartialStats] = bytemuck::cast_slice(&mapped);
            for p in partials {
                if p.count > 0 {
                    valid_px += p.count as usize;
                    min_v = min_v.min(p.min_val as f64);
                    max_v = max_v.max(p.max_val as f64);
                    sum += p.sum as f64;
                }
            }
        }
        self.partials_staging.unmap();

        if valid_px == 0 {
            return Ok(empty_stats(len, percentiles));
        }

        let mean = sum / valid_px as f64;
        let p_span = (max_v - min_v).max(1e-12);
        let (h_min, h_max) = hist_range.unwrap_or((min_v, max_v));
        let h_span = (h_max - h_min).max(1e-12);

        // Pass 2: privatized user histogram + 1024-bin percentile CDF + exact centered variance
        let p2 = StatsParams {
            len: len_u32,
            has_nodata: u32::from(nodata.is_some()),
            num_bins: n_bins,
            has_hist_range: u32::from(hist_range.is_some()),
            nodata_val: nodata.unwrap_or(0.0),
            mean: mean as f32,
            p_min: min_v as f32,
            p_inv_span: (1.0 / p_span) as f32,
            h_min: h_min as f32,
            h_max: h_max as f32,
            h_inv_span: (1.0 / h_span) as f32,
            _pad: 0.0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p2));

        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-stats-bg2"),
            layout: &self.bgl_pass2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.hist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.var_buf.as_entire_binding(),
                },
            ],
        });

        let hist_bytes = (2 * CDF_BINS * std::mem::size_of::<u32>()) as u64;
        let var_bytes = (num_wg as usize * std::mem::size_of::<f32>()) as u64;
        let mut enc2 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-stats-enc2"),
            });
        enc2.clear_buffer(&self.hist_buf, 0, None);
        {
            let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-stats-pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_pass2);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(num_wg, 1, 1);
        }
        enc2.copy_buffer_to_buffer(&self.hist_buf, 0, &self.hist_staging, 0, hist_bytes);
        enc2.copy_buffer_to_buffer(&self.var_buf, 0, &self.var_staging, 0, var_bytes);
        self.queue.submit(Some(enc2.finish()));

        let sh = self.hist_staging.slice(..hist_bytes);
        let sv = self.var_staging.slice(..var_bytes);
        let (txh, rxh) = mpsc::channel();
        let (txv, rxv) = mpsc::channel();
        sh.map_async(wgpu::MapMode::Read, move |r| {
            let _ = txh.send(r);
        });
        sv.map_async(wgpu::MapMode::Read, move |r| {
            let _ = txv.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rxh.recv()
            .map_err(|e| GpuError::BufferMap(e.to_string()))?
            .map_err(|e| GpuError::BufferMap(e.to_string()))?;
        rxv.recv()
            .map_err(|e| GpuError::BufferMap(e.to_string()))?
            .map_err(|e| GpuError::BufferMap(e.to_string()))?;

        let (hist_counts, cdf_counts, var_sum) = {
            let mh = sh.get_mapped_range();
            let mv = sv.get_mapped_range();
            let h_all: &[u32] = bytemuck::cast_slice(&mh);
            let v_all: &[f32] = bytemuck::cast_slice(&mv);
            let hc: Vec<u64> = h_all[..n_bins as usize].iter().map(|&c| c as u64).collect();
            let cc: Vec<u64> = h_all[CDF_BINS..2 * CDF_BINS]
                .iter()
                .map(|&c| c as u64)
                .collect();
            let vs: f64 = v_all.iter().map(|&x| x as f64).sum();
            (hc, cc, vs)
        };
        self.hist_staging.unmap();
        self.var_staging.unmap();

        let std_dev = (var_sum / valid_px as f64).sqrt();
        let median = Some(cdf_percentile(&cdf_counts, valid_px as u64, min_v, max_v, 50.0));
        let p_vals = percentiles
            .iter()
            .map(|&p| (p, Some(cdf_percentile(&cdf_counts, valid_px as u64, min_v, max_v, p))))
            .collect();
        let hist_edges: Vec<f64> = (0..=n_bins as usize)
            .map(|i| h_min + h_span * (i as f64 / n_bins as f64))
            .collect();
        let masked_pixels = len.saturating_sub(valid_px);
        let valid_percent = ((valid_px as f64 / len as f64) * 100.0 * 100.0).round() / 100.0;

        let mut maj_cnt = 0u64;
        let mut maj_bin = 0usize;
        let mut min_cnt = u64::MAX;
        let mut min_bin = 0usize;
        let mut unique_bins = 0usize;
        for (i, &c) in cdf_counts.iter().enumerate() {
            if c > 0 {
                unique_bins += 1;
                if c > maj_cnt {
                    maj_cnt = c;
                    maj_bin = i;
                }
                if c < min_cnt {
                    min_cnt = c;
                    min_bin = i;
                }
            }
        }
        let bin_w = (max_v - min_v) / (CDF_BINS as f64);
        let majority = Some(min_v + (maj_bin as f64 + 0.5) * bin_w);
        let minority = Some(min_v + (min_bin as f64 + 0.5) * bin_w);

        Ok(GpuBandStats {
            total_pixels: len,
            valid_pixels: valid_px,
            masked_pixels,
            valid_percent,
            min: Some(min_v),
            max: Some(max_v),
            mean: Some(mean),
            sum,
            std: Some(std_dev),
            median,
            majority,
            minority,
            unique_bins,
            percentiles: p_vals,
            hist_counts,
            hist_edges,
        })
    }
}

fn cdf_percentile(cdf: &[u64], total: u64, min_v: f64, max_v: f64, p: f64) -> f64 {
    if total == 0 || max_v <= min_v {
        return min_v;
    }
    let rank = (p.clamp(0.0, 100.0) / 100.0) * (total - 1) as f64;
    let mut acc = 0u64;
    let span = max_v - min_v;
    let bin_w = span / (cdf.len() as f64);
    for (i, &c) in cdf.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let next = acc + c;
        if (next as f64) > rank {
            let frac = ((rank - acc as f64) / (c as f64)).clamp(0.0, 1.0);
            return min_v + (i as f64 + frac) * bin_w;
        }
        acc = next;
    }
    max_v
}

fn empty_stats(total: usize, percentiles: &[f64]) -> GpuBandStats {
    GpuBandStats {
        total_pixels: total,
        valid_pixels: 0,
        masked_pixels: total,
        valid_percent: 0.0,
        min: None,
        max: None,
        mean: None,
        sum: 0.0,
        std: None,
        median: None,
        majority: None,
        minority: None,
        unique_bins: 0,
        percentiles: percentiles.iter().map(|&p| (p, None)).collect(),
        hist_counts: Vec::new(),
        hist_edges: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuOptions;

    #[test]
    fn gpu_stats_reducer_matches_cpu_exact_and_percentiles() {
        let Ok(ctx) = GpuContext::new(GpuOptions::default()) else {
            eprintln!("skipping gpu_stats test (no GPU adapter)");
            return;
        };
        let mut reducer = GpuStatsReducer::new(&ctx);
        let n = 512 * 512usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            if i % 17 == 0 {
                data.push(f32::NAN);
            } else if i % 29 == 0 {
                data.push(-9999.0);
            } else {
                let x = (i as f32) / (n as f32);
                data.push(100.0 + 900.0 * x + 50.0 * (x * 37.0).sin());
            }
        }
        let stats = reducer
            .reduce_f32(&data, Some(-9999.0), 10, None, &[2.0, 50.0, 98.0])
            .expect("gpu stats");

        let mut valid: Vec<f32> = data
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v != -9999.0)
            .collect();
        valid.sort_unstable_by(|a, b| a.total_cmp(b));
        let cpu_min = valid[0] as f64;
        let cpu_max = *valid.last().unwrap() as f64;
        let cpu_mean = valid.iter().map(|&v| v as f64).sum::<f64>() / valid.len() as f64;
        let cpu_std = (valid
            .iter()
            .map(|&v| {
                let d = v as f64 - cpu_mean;
                d * d
            })
            .sum::<f64>()
            / valid.len() as f64)
            .sqrt();
        let cpu_p2 = valid[((valid.len() - 1) as f64 * 0.02) as usize] as f64;
        let cpu_p50 = valid[((valid.len() - 1) as f64 * 0.50) as usize] as f64;
        let cpu_p98 = valid[((valid.len() - 1) as f64 * 0.98) as usize] as f64;

        assert_eq!(stats.valid_pixels, valid.len());
        assert_eq!(stats.masked_pixels, n - valid.len());
        assert!((stats.min.unwrap() - cpu_min).abs() < 1e-4);
        assert!((stats.max.unwrap() - cpu_max).abs() < 1e-4);
        assert!((stats.mean.unwrap() - cpu_mean).abs() < 1e-2);
        assert!((stats.std.unwrap() - cpu_std).abs() < 5e-2);
        assert_eq!(stats.hist_counts.iter().sum::<u64>(), valid.len() as u64);
        let span = cpu_max - cpu_min;
        assert!((stats.percentiles[0].1.unwrap() - cpu_p2).abs() < span * 0.002);
        assert!((stats.percentiles[1].1.unwrap() - cpu_p50).abs() < span * 0.002);
        assert!((stats.percentiles[2].1.unwrap() - cpu_p98).abs() < span * 0.002);
    }
}
