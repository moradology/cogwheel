use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::colormap::GpuPixelOutput;
use crate::context::{GpuContext, GpuError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPixelSelection {
    First,
    Highest,
    Lowest,
    Mean,
    Median,
    Stdev,
    Count,
}

impl GpuPixelSelection {
    fn as_u32(self) -> u32 {
        match self {
            Self::First => 0,
            Self::Highest => 1,
            Self::Lowest => 2,
            Self::Mean => 3,
            Self::Median => 4,
            Self::Stdev => 5,
            Self::Count => 6,
        }
    }
}

/// GPU-resident stack of `1..=32` aligned scene planes (`width * height` `f32` each).
pub struct GpuSceneStack {
    pub(crate) buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub plane_stride: u32,
    pub num_scenes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MosaicParams {
    width: u32,
    height: u32,
    plane_stride: u32,
    num_scenes: u32,
    method: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    _pad3: f32,
}

/// GPU multi-scene mosaic compositor and pixel-selection reducer
/// (`First`, `Highest`, `Lowest`, `Mean`, `Median`, `Stdev`, `Count`) with fused colormap output.
pub struct GpuMosaicCompositor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    f32_pipeline: wgpu::ComputePipeline,
    rgba8_pipeline: wgpu::ComputePipeline,
    indexed_pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    lut_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    out_words_cap: usize,
}

impl GpuMosaicCompositor {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-mosaic-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mosaic.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-mosaic-bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-mosaic-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-mosaic-f32"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("mosaic_select_f32"),
            compilation_options: Default::default(),
            cache: None,
        });
        let rgba8_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-mosaic-rgba8"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("mosaic_select_rgba8"),
            compilation_options: Default::default(),
            cache: None,
        });
        let indexed_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-mosaic-indexed"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("mosaic_select_indexed_u8"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-mosaic-params"),
            size: std::mem::size_of::<MosaicParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-mosaic-lut"),
            size: (256 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_words = 512 * 512;
        let (out_buf, staging_buf) = Self::alloc_out_pair(&device, init_words);

        Self {
            device,
            queue,
            bind_group_layout,
            f32_pipeline,
            rgba8_pipeline,
            indexed_pipeline,
            params_buf,
            lut_buf,
            out_buf,
            staging_buf,
            out_words_cap: init_words,
        }
    }

    fn alloc_out_pair(device: &wgpu::Device, words: usize) -> (wgpu::Buffer, wgpu::Buffer) {
        let byte_size = (words.max(1) * std::mem::size_of::<u32>()) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-mosaic-out"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-mosaic-staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (out_buf, staging_buf)
    }

    fn ensure_out_cap(&mut self, words: usize) {
        if words > self.out_words_cap {
            self.out_words_cap = words;
            let (out_buf, staging_buf) = Self::alloc_out_pair(&self.device, self.out_words_cap);
            self.out_buf = out_buf;
            self.staging_buf = staging_buf;
        }
    }

    /// Upload `1..=32` scene planes (`width * height` `f32` each) into a contiguous GPU stack.
    pub fn upload_scenes(
        &self,
        planes: &[&[f32]],
        width: u32,
        height: u32,
    ) -> Result<GpuSceneStack, GpuError> {
        if planes.is_empty() || planes.len() > 32 || width == 0 || height == 0 {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid scene stack: scenes={}, {width}x{height}",
                planes.len()
            )));
        }
        let plane_stride = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("plane_stride overflow".into()))?;
        let mut packed = Vec::with_capacity(plane_stride * planes.len());
        for (i, p) in planes.iter().enumerate() {
            if p.len() < plane_stride {
                return Err(GpuError::InvalidDimensions(format!(
                    "scene {i} len {} < {plane_stride}",
                    p.len()
                )));
            }
            packed.extend_from_slice(&p[..plane_stride]);
        }
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cogzilla-scene-stack"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(GpuSceneStack {
            buffer,
            width,
            height,
            plane_stride: plane_stride as u32,
            num_scenes: planes.len() as u32,
        })
    }

    /// Reduce `stack` via `method` into raw `f32` pixels (`width * height`).
    pub fn composite_f32(
        &mut self,
        stack: &GpuSceneStack,
        method: GpuPixelSelection,
    ) -> Result<Vec<f32>, GpuError> {
        let pixels = (stack.width as usize) * (stack.height as usize);
        self.ensure_out_cap(pixels);
        let params = MosaicParams {
            width: stack.width,
            height: stack.height,
            plane_stride: stack.plane_stride,
            num_scenes: stack.num_scenes,
            method: method.as_u32(),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            rescale_min: 0.0,
            inv_range_255: 1.0,
            inv_range_254: 1.0,
            _pad3: 0.0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let raw = self.run_pipeline(
            stack,
            &self.f32_pipeline,
            stack.width.div_ceil(16),
            stack.height.div_ceil(16),
            pixels * 4,
        )?;
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// Reduce `stack` via `method` fused with rescale + colormap into `Rgba8` or `IndexedU8`.
    #[allow(clippy::too_many_arguments)]
    pub fn composite_styled(
        &mut self,
        stack: &GpuSceneStack,
        method: GpuPixelSelection,
        rescale_min: f32,
        rescale_max: f32,
        lut_rgba_u32: &[u32; 256],
        output: GpuPixelOutput,
    ) -> Result<Vec<u8>, GpuError> {
        let w = stack.width as usize;
        let h = stack.height as usize;
        let words_per_row = match output {
            GpuPixelOutput::Rgba8 => w,
            GpuPixelOutput::IndexedU8 => w.div_ceil(4),
        };
        let total_words = words_per_row * h;
        self.ensure_out_cap(total_words);

        let range = rescale_max - rescale_min;
        let safe_range = if range.is_finite() && range.abs() > 1e-20 {
            range
        } else {
            1.0
        };
        let params = MosaicParams {
            width: stack.width,
            height: stack.height,
            plane_stride: stack.plane_stride,
            num_scenes: stack.num_scenes,
            method: method.as_u32(),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            rescale_min,
            inv_range_255: 255.0 / safe_range,
            inv_range_254: 254.0 / safe_range,
            _pad3: 0.0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        if output == GpuPixelOutput::Rgba8 {
            self.queue
                .write_buffer(&self.lut_buf, 0, bytemuck::cast_slice(lut_rgba_u32));
        }

        let (pipeline, wg_x, wg_y) = match output {
            GpuPixelOutput::Rgba8 => (
                &self.rgba8_pipeline,
                stack.width.div_ceil(16),
                stack.height.div_ceil(16),
            ),
            GpuPixelOutput::IndexedU8 => (
                &self.indexed_pipeline,
                (words_per_row as u32).div_ceil(16),
                stack.height.div_ceil(16),
            ),
        };
        let raw = self.run_pipeline(stack, pipeline, wg_x, wg_y, total_words * 4)?;
        if output == GpuPixelOutput::IndexedU8 && w % 4 != 0 {
            let row_bytes = words_per_row * 4;
            let mut out = Vec::with_capacity(w * h);
            for py in 0..h {
                let start = py * row_bytes;
                out.extend_from_slice(&raw[start..start + w]);
            }
            Ok(out)
        } else {
            Ok(raw)
        }
    }

    fn run_pipeline(
        &self,
        stack: &GpuSceneStack,
        pipeline: &wgpu::ComputePipeline,
        wg_x: u32,
        wg_y: u32,
        out_bytes: usize,
    ) -> Result<Vec<u8>, GpuError> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-mosaic-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stack.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-mosaic-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-mosaic-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging_buf, 0, out_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buf.slice(..out_bytes as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::BufferMap(e.to_string()))?
            .map_err(|e| GpuError::BufferMap(e.to_string()))?;

        let mapped = slice.get_mapped_range();
        let out = mapped.to_vec();
        drop(mapped);
        self.staging_buf.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_select(planes: &[Vec<f32>], idx: usize, method: GpuPixelSelection) -> f32 {
        let mut vals: Vec<f32> = planes
            .iter()
            .map(|p| p[idx])
            .filter(|v| v.is_finite())
            .collect();
        if vals.is_empty() {
            return f32::NAN;
        }
        match method {
            GpuPixelSelection::First => vals[0],
            GpuPixelSelection::Highest => vals.into_iter().fold(f32::NEG_INFINITY, f32::max),
            GpuPixelSelection::Lowest => vals.into_iter().fold(f32::INFINITY, f32::min),
            GpuPixelSelection::Mean => vals.iter().sum::<f32>() / vals.len() as f32,
            GpuPixelSelection::Median => {
                vals.sort_by(|a, b| a.total_cmp(b));
                let m = vals.len() / 2;
                if vals.len() % 2 == 1 {
                    vals[m]
                } else {
                    0.5 * (vals[m - 1] + vals[m])
                }
            }
            GpuPixelSelection::Stdev => {
                let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                    / vals.len() as f32;
                var.sqrt()
            }
            GpuPixelSelection::Count => vals.len() as f32,
        }
    }

    #[test]
    fn gpu_mosaic_all_seven_pixel_selection_methods_match_cpu() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU mosaic test (no adapter): {e}");
                return;
            }
        };
        let mut comp = GpuMosaicCompositor::new(&ctx);

        let w = 256u32;
        let h = 256u32;
        let n = (w * h) as usize;
        let num_scenes = 12usize;
        let mut planes = Vec::with_capacity(num_scenes);
        for s in 0..num_scenes {
            let mut p = vec![0.0f32; n];
            for i in 0..n {
                // Create deterministic cloud masks + an all-nodata region
                if i < 50 || ((i + s * 37) % 5 == 0) {
                    p[i] = f32::NAN;
                } else {
                    p[i] = 100.0 + ((i * 13 + s * 97) % 1000) as f32 * 0.5;
                }
            }
            planes.push(p);
        }
        let refs: Vec<&[f32]> = planes.iter().map(Vec::as_slice).collect();
        let stack = comp.upload_scenes(&refs, w, h).unwrap();

        for &method in &[
            GpuPixelSelection::First,
            GpuPixelSelection::Highest,
            GpuPixelSelection::Lowest,
            GpuPixelSelection::Mean,
            GpuPixelSelection::Median,
            GpuPixelSelection::Stdev,
            GpuPixelSelection::Count,
        ] {
            let gpu_out = comp.composite_f32(&stack, method).unwrap();
            let mut max_err = 0.0f32;
            let mut mask_err = 0usize;
            for i in 0..n {
                let c = cpu_select(&planes, i, method);
                let g = gpu_out[i];
                if c.is_nan() != g.is_nan() {
                    mask_err += 1;
                } else if !c.is_nan() {
                    max_err = max_err.max((c - g).abs());
                }
            }
            assert_eq!(mask_err, 0, "{method:?}: nodata mask mismatch");
            assert!(max_err <= 1e-3, "{method:?}: max err {max_err} > 1e-3");
        }
    }
}
