use std::time::{Duration, Instant};

use crate::context::GpuContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyOverride {
    Auto,
    Always,
    Never,
}

impl PolicyOverride {
    pub fn from_env() -> Self {
        match std::env::var("COGZILLA_GPU_POLICY")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("always") | Some("force") | Some("gpu") => Self::Always,
            Some("never") | Some("cpu") | Some("off") => Self::Never,
            _ => Self::Auto,
        }
    }
}

/// Supported GPU-offloadable operation families for cost estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOp {
    ResampleBilinear,
    ResampleCubic,
    ResampleLanczos3,
    FusedColormapBilinear,
    FusedColormapCubic,
    FusedColormapLanczos3,
    DemHornCubic,
    DemHornLanczos3,
    DemTerrainRgbCubic,
}

/// Startup-calibrated hardware profile for a `GpuContext`.
#[derive(Debug, Clone, Copy)]
pub struct GpuProfile {
    /// Synchronous submit + wait floor latency in microseconds.
    pub dispatch_floor_us: f64,
    /// Host -> GPU upload throughput in GB/s.
    pub upload_gb_s: f64,
    /// GPU -> Host readback throughput in GB/s.
    pub readback_gb_s: f64,
    /// True on Apple Silicon unified memory.
    pub unified_memory: bool,
}

impl GpuProfile {
    /// Run a fast (<15 ms) startup micro-probe to measure dispatch floor and transfer bandwidth.
    pub fn calibrate(ctx: &GpuContext) -> Self {
        let noop_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-cal-noop-wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                "@compute @workgroup_size(1, 1, 1) fn noop_dispatch() {}".into(),
            ),
        });
        let noop_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cogzilla-cal-noop-pl"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
        let noop_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cogzilla-cal-noop-pipe"),
                layout: Some(&noop_layout),
                module: &noop_shader,
                entry_point: Some("noop_dispatch"),
                compilation_options: Default::default(),
                cache: None,
            });
        let noop_dispatch_sync = || {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cogzilla-noop-enc"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cogzilla-noop-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&noop_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit(Some(encoder.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);
        };

        // 1. Measure dispatch + poll(Wait) floor (take p20 of 7 samples so a noisy neighbor
        //    spiking the GPU during startup does not permanently inflate the dispatch floor).
        for _ in 0..2 {
            noop_dispatch_sync();
        }
        let mut noop_samples = Vec::with_capacity(7);
        for _ in 0..7 {
            let t0 = Instant::now();
            noop_dispatch_sync();
            noop_samples.push(t0.elapsed());
        }
        noop_samples.sort();
        let dispatch_floor_us = (noop_samples[1].as_secs_f64() * 1e6).max(50.0);

        // 2. Measure 4 MiB upload + readback bandwidth
        let probe_bytes: usize = 4 * 1024 * 1024;
        let payload = vec![0x3fu8; probe_bytes];
        let gpu_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-cal-gpu"),
            size: probe_bytes as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-cal-staging"),
            size: probe_bytes as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Warmup upload
        ctx.queue.write_buffer(&gpu_buf, 0, &payload);
        ctx.queue.submit(None);
        ctx.device.poll(wgpu::Maintain::Wait);

        let t_up0 = Instant::now();
        ctx.queue.write_buffer(&gpu_buf, 0, &payload);
        ctx.queue.submit(None);
        ctx.device.poll(wgpu::Maintain::Wait);
        let up_s = t_up0.elapsed().as_secs_f64().max(1e-6);
        let upload_gb_s = ((probe_bytes as f64 / 1e9) / up_s).clamp(1.0, 200.0);

        // Measure copy + map readback
        let t_dn0 = Instant::now();
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-cal-enc"),
            });
        enc.copy_buffer_to_buffer(&gpu_buf, 0, &staging_buf, 0, probe_bytes as u64);
        ctx.queue.submit(Some(enc.finish()));
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        let readback_gb_s = if matches!(rx.recv(), Ok(Ok(()))) {
            {
                let mapped = slice.get_mapped_range();
                // Touch first and last byte to ensure pages are faulted
                std::hint::black_box(mapped[0]);
                std::hint::black_box(mapped[mapped.len() - 1]);
            }
            staging_buf.unmap();
            let dn_s = t_dn0.elapsed().as_secs_f64().max(1e-6);
            ((probe_bytes as f64 / 1e9) / dn_s).clamp(0.5, 200.0)
        } else {
            upload_gb_s
        };

        Self {
            dispatch_floor_us,
            upload_gb_s,
            readback_gb_s,
            unified_memory: ctx.info.unified_memory,
        }
    }

    /// Construct a deterministic profile for unit tests without needing GPU hardware.
    pub fn synthetic(
        dispatch_floor_us: f64,
        upload_gb_s: f64,
        readback_gb_s: f64,
        unified_memory: bool,
    ) -> Self {
        Self {
            dispatch_floor_us,
            upload_gb_s,
            readback_gb_s,
            unified_memory,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DispatchRequest {
    pub op: GpuOp,
    /// Total output pixels (`dst_w * dst_h`).
    pub dst_pixels: usize,
    /// Source upload size in bytes (`0` when slab is already GPU-resident / warm).
    pub upload_bytes: usize,
    /// Readback size in bytes (e.g. `dst_pixels * 1` for indexed u8, `* 4` for RGBA8/f32).
    pub readback_bytes: usize,
    /// Number of tiles batched into one command buffer (`1` for single-tile synchronous dispatch).
    pub batch_size: usize,
    /// Number of GPU submissions currently in flight on this device.
    pub in_flight: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchReason {
    PolicyForcedGpu,
    PolicyForcedCpu,
    QueueBackpressure,
    GpuFasterByMargin,
    CpuFasterOrBelowMargin,
}

#[derive(Debug, Clone, Copy)]
pub struct DispatchDecision {
    pub use_gpu: bool,
    pub est_cpu_us: f64,
    pub est_gpu_us: f64,
    pub reason: DispatchReason,
}

/// Adaptive CPU-vs-GPU dispatch policy driven by a startup-calibrated `GpuProfile`.
#[derive(Debug, Clone, Copy)]
pub struct DispatchPolicy {
    pub profile: GpuProfile,
    pub override_mode: PolicyOverride,
    /// Require `est_gpu_us * safety_margin < est_cpu_us` before offloading (default `1.25`).
    pub safety_margin: f64,
    /// Fall back to CPU if `in_flight >= max_in_flight` to avoid queueing behind busy GPUs.
    pub max_in_flight: usize,
}

impl DispatchPolicy {
    pub fn new(profile: GpuProfile) -> Self {
        Self {
            profile,
            override_mode: PolicyOverride::from_env(),
            safety_margin: 1.25,
            max_in_flight: 4,
        }
    }

    pub fn calibrate(ctx: &GpuContext) -> Self {
        Self::new(GpuProfile::calibrate(ctx))
    }

    /// Estimated CPU time in microseconds per pixel (conservative single-core baseline).
    /// Calibrated after separable SIMD + LUT Lanczos3/Cubic optimization (`~230 us` per 512² band,
    /// `~690 us` per 512² 3-band RGB on Apple M5; non-separable DEM Horn stays high).
    fn cpu_us_per_pixel(op: GpuOp) -> f64 {
        match op {
            GpuOp::ResampleBilinear => 0.0012,
            GpuOp::ResampleCubic => 0.0016,
            GpuOp::ResampleLanczos3 => 0.0018,
            GpuOp::FusedColormapBilinear => 0.0022,
            GpuOp::FusedColormapCubic => 0.0024,
            GpuOp::FusedColormapLanczos3 => 0.0028,
            GpuOp::DemHornCubic => 0.0085,
            GpuOp::DemHornLanczos3 => 0.0650,
            GpuOp::DemTerrainRgbCubic => 0.0055,
        }
    }

    /// Estimated GPU shader execution time in microseconds per pixel (excluding dispatch/transfer).
    fn gpu_compute_us_per_pixel(op: GpuOp) -> f64 {
        match op {
            GpuOp::ResampleBilinear | GpuOp::FusedColormapBilinear => 0.00008,
            GpuOp::ResampleCubic | GpuOp::FusedColormapCubic | GpuOp::DemTerrainRgbCubic => 0.00020,
            GpuOp::DemHornCubic => 0.00030,
            GpuOp::ResampleLanczos3 | GpuOp::FusedColormapLanczos3 | GpuOp::DemHornLanczos3 => {
                0.00050
            }
        }
    }

    pub fn decide(&self, req: &DispatchRequest) -> DispatchDecision {
        let pixels = req.dst_pixels as f64;
        let batch = req.batch_size.max(1) as f64;

        let est_cpu_us = pixels * Self::cpu_us_per_pixel(req.op);
        let dispatch_us = self.profile.dispatch_floor_us / batch;
        let upload_us = (req.upload_bytes as f64 / 1e3) / self.profile.upload_gb_s;
        let readback_us = (req.readback_bytes as f64 / 1e3) / self.profile.readback_gb_s;
        let compute_us = pixels * Self::gpu_compute_us_per_pixel(req.op);
        let est_gpu_us = dispatch_us + upload_us + compute_us + readback_us;

        match self.override_mode {
            PolicyOverride::Always => {
                return DispatchDecision {
                    use_gpu: true,
                    est_cpu_us,
                    est_gpu_us,
                    reason: DispatchReason::PolicyForcedGpu,
                };
            }
            PolicyOverride::Never => {
                return DispatchDecision {
                    use_gpu: false,
                    est_cpu_us,
                    est_gpu_us,
                    reason: DispatchReason::PolicyForcedCpu,
                };
            }
            PolicyOverride::Auto => {}
        }

        if req.in_flight >= self.max_in_flight {
            return DispatchDecision {
                use_gpu: false,
                est_cpu_us,
                est_gpu_us,
                reason: DispatchReason::QueueBackpressure,
            };
        }

        let use_gpu = est_gpu_us * self.safety_margin < est_cpu_us;
        DispatchDecision {
            use_gpu,
            est_cpu_us,
            est_gpu_us,
            reason: if use_gpu {
                DispatchReason::GpuFasterByMargin
            } else {
                DispatchReason::CpuFasterOrBelowMargin
            },
        }
    }
}

/// Helper for formatting durations in benchmark output.
pub fn duration_us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_routes_tiny_cubic_to_cpu_and_lanczos_or_large_crops_to_gpu() {
        // Synthetic M5-like profile: 1500 us floor, 12 GB/s upload, 8 GB/s readback
        let mut policy = DispatchPolicy::new(GpuProfile::synthetic(1500.0, 12.0, 8.0, true));
        policy.override_mode = PolicyOverride::Auto;

        // 256x256 warm Cubic -> should stay on CPU
        let d_small_cubic = policy.decide(&DispatchRequest {
            op: GpuOp::FusedColormapCubic,
            dst_pixels: 256 * 256,
            upload_bytes: 0,
            readback_bytes: 256 * 256,
            batch_size: 1,
            in_flight: 0,
        });
        assert!(!d_small_cubic.use_gpu);
        assert_eq!(d_small_cubic.reason, DispatchReason::CpuFasterOrBelowMargin);

        // 256x256 warm Lanczos3 -> now stays on CPU after separable SIMD + LUT optimization
        let d_small_lanczos = policy.decide(&DispatchRequest {
            op: GpuOp::FusedColormapLanczos3,
            dst_pixels: 256 * 256,
            upload_bytes: 0,
            readback_bytes: 256 * 256,
            batch_size: 1,
            in_flight: 0,
        });
        assert!(!d_small_lanczos.use_gpu);
        assert_eq!(d_small_lanczos.reason, DispatchReason::CpuFasterOrBelowMargin);

        // 256x256 warm DEM Horn Lanczos3 -> still routes to GPU (non-separable stencil)
        let d_small_dem = policy.decide(&DispatchRequest {
            op: GpuOp::DemHornLanczos3,
            dst_pixels: 256 * 256,
            upload_bytes: 0,
            readback_bytes: 256 * 256,
            batch_size: 1,
            in_flight: 0,
        });
        assert!(d_small_dem.use_gpu);
        assert_eq!(d_small_dem.reason, DispatchReason::GpuFasterByMargin);

        // 2048x2048 warm Cubic & Lanczos3 with indexed u8 readback -> should route to GPU
        let d_large_cubic = policy.decide(&DispatchRequest {
            op: GpuOp::FusedColormapCubic,
            dst_pixels: 2048 * 2048,
            upload_bytes: 0,
            readback_bytes: 2048 * 2048,
            batch_size: 1,
            in_flight: 0,
        });
        assert!(d_large_cubic.use_gpu);
        let d_large_lanczos = policy.decide(&DispatchRequest {
            op: GpuOp::FusedColormapLanczos3,
            dst_pixels: 2048 * 2048,
            upload_bytes: 0,
            readback_bytes: 2048 * 2048,
            batch_size: 1,
            in_flight: 0,
        });
        assert!(d_large_lanczos.use_gpu);

        // Backpressure fallback when queue is full
        let d_busy = policy.decide(&DispatchRequest {
            op: GpuOp::FusedColormapLanczos3,
            dst_pixels: 2048 * 2048,
            upload_bytes: 0,
            readback_bytes: 2048 * 2048,
            batch_size: 1,
            in_flight: 4,
        });
        assert!(!d_busy.use_gpu);
        assert_eq!(d_busy.reason, DispatchReason::QueueBackpressure);
    }
}
