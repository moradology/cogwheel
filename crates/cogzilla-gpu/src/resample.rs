use std::sync::Arc;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::context::{GpuContext, GpuError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuResampleMode {
    Cubic,
    Lanczos3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ResampleParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-resident `f32` raster slab (e.g. a decoded COG block or stitched tile neighborhood).
/// Upload once, then resample repeatedly across views/tiles/zooms with zero host->device copy.
pub struct GpuSlab {
    pub(crate) buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ResampleTimings {
    pub upload: Duration,
    pub compute_and_readback: Duration,
    pub total: Duration,
}

/// Compiled `wgpu` compute pipelines and reusable scratch buffers for separable 1D resampling.
pub struct GpuResampler {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    cubic_pipeline: wgpu::ComputePipeline,
    lanczos3_pipeline: wgpu::ComputePipeline,
    noop_pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    u_buf: wgpu::Buffer,
    u_cap: usize,
    v_buf: wgpu::Buffer,
    v_cap: usize,
    out_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    out_cap: usize,
}

impl GpuResampler {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-resample-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/resample.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-resample-bgl"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-resample-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let cubic_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-resample-cubic"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("resample_cubic"),
            compilation_options: Default::default(),
            cache: None,
        });

        let lanczos3_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-resample-lanczos3"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("resample_lanczos3"),
            compilation_options: Default::default(),
            cache: None,
        });

        let noop_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-noop-pl"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let noop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-noop"),
            layout: Some(&noop_layout),
            module: &shader,
            entry_point: Some("noop_dispatch"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-resample-params"),
            size: std::mem::size_of::<ResampleParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_coord_cap = 512usize;
        let u_buf = Self::alloc_coord_buf(&device, "cogzilla-u-coords", init_coord_cap);
        let v_buf = Self::alloc_coord_buf(&device, "cogzilla-v-coords", init_coord_cap);

        let init_out_cap = 512 * 512;
        let (out_buf, staging_buf) = Self::alloc_out_pair(&device, init_out_cap);

        Self {
            device,
            queue,
            bind_group_layout,
            cubic_pipeline,
            lanczos3_pipeline,
            noop_pipeline,
            params_buf,
            u_buf,
            u_cap: init_coord_cap,
            v_buf,
            v_cap: init_coord_cap,
            out_buf,
            staging_buf,
            out_cap: init_out_cap,
        }
    }

    fn alloc_coord_buf(device: &wgpu::Device, label: &str, elems: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (elems.max(1) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn alloc_out_pair(device: &wgpu::Device, elems: usize) -> (wgpu::Buffer, wgpu::Buffer) {
        let byte_size = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-resample-out"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-resample-staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (out_buf, staging_buf)
    }

    fn ensure_capacities(&mut self, dst_w: usize, dst_h: usize) {
        if dst_w > self.u_cap {
            self.u_cap = dst_w.next_power_of_two();
            self.u_buf = Self::alloc_coord_buf(&self.device, "cogzilla-u-coords", self.u_cap);
        }
        if dst_h > self.v_cap {
            self.v_cap = dst_h.next_power_of_two();
            self.v_buf = Self::alloc_coord_buf(&self.device, "cogzilla-v-coords", self.v_cap);
        }
        let out_elems = dst_w * dst_h;
        if out_elems > self.out_cap {
            self.out_cap = out_elems;
            let (out_buf, staging_buf) = Self::alloc_out_pair(&self.device, self.out_cap);
            self.out_buf = out_buf;
            self.staging_buf = staging_buf;
        }
    }

    /// Measure the minimal GPU dispatch + synchronization floor latency.
    pub fn noop_dispatch_sync(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-noop-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.noop_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Upload an `f32` raster slab to GPU memory for repeated warm resamples.
    pub fn upload_slab(
        &self,
        data: &[f32],
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<GpuSlab, GpuError> {
        let min_elems = (height as usize)
            .checked_mul(stride as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("height * stride overflow".into()))?;
        if width == 0 || height == 0 || stride < width || data.len() < min_elems {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid slab: width={width}, height={height}, stride={stride}, len={}",
                data.len()
            )));
        }
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cogzilla-gpu-slab"),
                contents: bytemuck::cast_slice(&data[..min_elems]),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(GpuSlab {
            buffer,
            width,
            height,
            stride,
        })
    }

    /// Resample from an already GPU-resident `GpuSlab` along separable 1D `(u_coords, v_coords)`.
    pub fn resample_slab(
        &mut self,
        slab: &GpuSlab,
        u_coords: &[f32],
        v_coords: &[f32],
        mode: GpuResampleMode,
    ) -> Result<Vec<f32>, GpuError> {
        let dst_w = u_coords.len();
        let dst_h = v_coords.len();
        if dst_w == 0 || dst_h == 0 {
            return Err(GpuError::InvalidDimensions(
                "destination width/height must be > 0".into(),
            ));
        }

        self.ensure_capacities(dst_w, dst_h);

        let params = ResampleParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(u_coords));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v_coords));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-resample-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slab.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline = match mode {
            GpuResampleMode::Cubic => &self.cubic_pipeline,
            GpuResampleMode::Lanczos3 => &self.lanczos3_pipeline,
        };

        let out_bytes = (dst_w * dst_h * std::mem::size_of::<f32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-resample-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-resample-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (dst_w as u32).div_ceil(16);
            let wg_y = (dst_h as u32).div_ceil(16);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging_buf, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buf.slice(..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::BufferMap(e.to_string()))?
            .map_err(|e| GpuError::BufferMap(e.to_string()))?;

        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.staging_buf.unmap();

        Ok(out)
    }

    /// Cold-path helper: upload `src` slab, execute compute shader, and read back `f32` output,
    /// returning both the result and stage timings.
    pub fn resample_cold_timed(
        &mut self,
        src: &[f32],
        src_w: u32,
        src_h: u32,
        src_stride: u32,
        u_coords: &[f32],
        v_coords: &[f32],
        mode: GpuResampleMode,
    ) -> Result<(Vec<f32>, ResampleTimings), GpuError> {
        let t0 = Instant::now();
        let slab = self.upload_slab(src, src_w, src_h, src_stride)?;
        let t1 = Instant::now();
        let out = self.resample_slab(&slab, u_coords, v_coords, mode)?;
        let t2 = Instant::now();
        Ok((
            out,
            ResampleTimings {
                upload: t1 - t0,
                compute_and_readback: t2 - t1,
                total: t2 - t0,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
    use cogzilla_kernel::{BandSource, ResampleMode, sample_cubic, sample_general};

    fn make_synthetic_dem(w: u32, h: u32, with_nodata: bool) -> Vec<f32> {
        let mut data = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if with_nodata && ((x >= 100 && x < 120 && y >= 80 && y < 105) || (x == 5 && y == 5)) {
                    data[idx] = f32::NAN;
                } else {
                    let fx = x as f32 * 0.031;
                    let fy = y as f32 * 0.027;
                    data[idx] = 1000.0 + 400.0 * fx.sin() * fy.cos() + 50.0 * (fx * 3.1 + fy).sin();
                }
            }
        }
        data
    }

    fn cpu_resample(
        src: &[f32],
        src_w: u32,
        src_h: u32,
        u_coords: &[f32],
        v_coords: &[f32],
        mode: ResampleMode,
    ) -> Vec<f32> {
        let mut slab = AlignedF32Slab::new_uninit((src_w * src_h) as usize);
        slab.as_mut_slice().copy_from_slice(src);
        let no_nodata = !src.iter().any(|v| v.is_nan());
        let band = DecodedBand {
            data: BandData::F32(slab),
            width: src_w,
            height: src_h,
            stats: BlockStats {
                all_nodata: false,
                no_nodata,
                min: 0.0,
                max: 2000.0,
            },
        };
        let blocks = [Some(&band)];
        let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
        let dst_w = u_coords.len();
        let dst_h = v_coords.len();
        let mut out = vec![0.0f32; dst_w * dst_h];
        for (py, &v) in v_coords.iter().enumerate() {
            let row = &mut out[py * dst_w..(py + 1) * dst_w];
            for (px, &u) in u_coords.iter().enumerate() {
                row[px] = match mode {
                    ResampleMode::Cubic => sample_cubic(&bs, u, v),
                    other => sample_general(&bs, u, v, other),
                };
            }
        }
        out
    }

    fn assert_parity(cpu: &[f32], gpu: &[f32], max_abs_tol: f32) {
        assert_eq!(cpu.len(), gpu.len());
        let mut max_err = 0.0f32;
        let mut nan_mismatches = 0usize;
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            if c.is_nan() != g.is_nan() {
                nan_mismatches += 1;
                if nan_mismatches <= 5 {
                    eprintln!("NaN mismatch at {i}: cpu={c}, gpu={g}");
                }
            } else if !c.is_nan() {
                let err = (c - g).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert_eq!(nan_mismatches, 0, "nodata mask must match 100%");
        assert!(
            max_err <= max_abs_tol,
            "max abs error {max_err} exceeded tolerance {max_abs_tol}"
        );
    }

    #[test]
    fn gpu_cubic_and_lanczos3_match_cpu_with_nodata() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU test (no adapter): {e}");
                return;
            }
        };
        let mut resampler = GpuResampler::new(&ctx);

        let src_w = 512u32;
        let src_h = 512u32;
        let src = make_synthetic_dem(src_w, src_h, true);

        // Cover interior, nodata hole, and out-of-bounds edges.
        let dst_n = 256usize;
        let u_coords: Vec<f32> = (0..dst_n)
            .map(|i| -1.25 + (i as f32) * ((src_w as f32 + 2.5) / dst_n as f32))
            .collect();
        let v_coords: Vec<f32> = (0..dst_n)
            .map(|i| -1.25 + (i as f32) * ((src_h as f32 + 2.5) / dst_n as f32))
            .collect();

        let slab = resampler.upload_slab(&src, src_w, src_h, src_w).unwrap();

        let cpu_cubic = cpu_resample(&src, src_w, src_h, &u_coords, &v_coords, ResampleMode::Cubic);
        let gpu_cubic = resampler
            .resample_slab(&slab, &u_coords, &v_coords, GpuResampleMode::Cubic)
            .unwrap();
        assert_parity(&cpu_cubic, &gpu_cubic, 1e-3);

        let cpu_lanczos =
            cpu_resample(&src, src_w, src_h, &u_coords, &v_coords, ResampleMode::Lanczos);
        let gpu_lanczos = resampler
            .resample_slab(&slab, &u_coords, &v_coords, GpuResampleMode::Lanczos3)
            .unwrap();
        assert_parity(&cpu_lanczos, &gpu_lanczos, 2e-3);
    }
}
