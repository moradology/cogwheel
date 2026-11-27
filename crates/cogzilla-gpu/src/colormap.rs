use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::context::{GpuContext, GpuError};
use crate::policy::GpuOp;
use crate::resample::GpuSlab;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuColormapMode {
    Bilinear,
    Cubic,
    Lanczos3,
    Nearest,
}

impl GpuColormapMode {
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            Self::Bilinear => 0,
            Self::Cubic => 1,
            Self::Lanczos3 => 2,
            Self::Nearest => 3,
        }
    }

    pub fn to_policy_op(self) -> GpuOp {
        match self {
            Self::Bilinear | Self::Nearest => GpuOp::FusedColormapBilinear,
            Self::Cubic => GpuOp::FusedColormapCubic,
            Self::Lanczos3 => GpuOp::FusedColormapLanczos3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPixelOutput {
    /// 4 bytes/pixel (`R, G, B, A`, transparent `[0,0,0,0]` for nodata).
    Rgba8,
    /// 1 byte/pixel (`0` = nodata, `1..=255` = palette index), 4× smaller PCIe readback.
    IndexedU8,
}

/// Coordinate mapping passed to `GpuColormapRenderer::render_slab_with_plan`.
pub enum GpuCoordPlan<'a> {
    /// Separable 1D `(u_coords[dst_w], v_coords[dst_h])` (EPSG:3857 / EPSG:4326 / axis-aligned crops).
    Separable1D {
        u_coords: &'a [f32],
        v_coords: &'a [f32],
    },
    /// 17×17 bilinear control grid (`grid_u[289], grid_v[289]`) for reprojected UTM / Albers / LCC / Polar tiles.
    Grid2D {
        dst_w: usize,
        dst_h: usize,
        grid_u: &'a [f32; 289],
        grid_v: &'a [f32; 289],
    },
    /// Exact per-pixel 2D coordinates (`u[dst_w * dst_h], v[dst_w * dst_h]`) for low-zoom reprojected tiles.
    Exact2D {
        dst_w: usize,
        dst_h: usize,
        u: &'a [f32],
        v: &'a [f32],
    },
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ColormapParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    mode: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    coord_mode: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Fused GPU resample -> rescale -> colormap renderer (RGBA8 or 4×-smaller Indexed `u8`).
pub struct GpuColormapRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    rgba8_pipeline: wgpu::ComputePipeline,
    indexed_pipeline: wgpu::ComputePipeline,
    f32_pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    lut_buf: wgpu::Buffer,
    u_buf: wgpu::Buffer,
    u_cap: usize,
    v_buf: wgpu::Buffer,
    v_cap: usize,
    out_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    out_words_cap: usize,
}

impl GpuColormapRenderer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-colormap-wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::COLORMAP_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-colormap-bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-colormap-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let rgba8_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-fused-rgba8"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fused_colormap_rgba8"),
            compilation_options: Default::default(),
            cache: None,
        });

        let indexed_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-fused-indexed-u8"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fused_colormap_indexed_u8"),
            compilation_options: Default::default(),
            cache: None,
        });

        let f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-resample-f32"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("resample_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-colormap-params"),
            size: std::mem::size_of::<ColormapParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-colormap-lut"),
            size: (256 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_coord_cap = 512usize;
        let u_buf = Self::alloc_coord_buf(&device, "cogzilla-cm-u", init_coord_cap);
        let v_buf = Self::alloc_coord_buf(&device, "cogzilla-cm-v", init_coord_cap);

        let init_words = 512 * 512;
        let (out_buf, staging_buf) = Self::alloc_out_pair(&device, init_words);

        Self {
            device,
            queue,
            bind_group_layout,
            rgba8_pipeline,
            indexed_pipeline,
            f32_pipeline,
            params_buf,
            lut_buf,
            u_buf,
            u_cap: init_coord_cap,
            v_buf,
            v_cap: init_coord_cap,
            out_buf,
            staging_buf,
            out_words_cap: init_words,
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

    fn alloc_out_pair(device: &wgpu::Device, words: usize) -> (wgpu::Buffer, wgpu::Buffer) {
        let byte_size = (words.max(1) * std::mem::size_of::<u32>()) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-cm-out"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-cm-staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (out_buf, staging_buf)
    }

    fn ensure_capacities(&mut self, dst_w: usize, dst_h: usize, out_words: usize) {
        if dst_w > self.u_cap {
            self.u_cap = dst_w.next_power_of_two();
            self.u_buf = Self::alloc_coord_buf(&self.device, "cogzilla-cm-u", self.u_cap);
        }
        if dst_h > self.v_cap {
            self.v_cap = dst_h.next_power_of_two();
            self.v_buf = Self::alloc_coord_buf(&self.device, "cogzilla-cm-v", self.v_cap);
        }
        if out_words > self.out_words_cap {
            self.out_words_cap = out_words;
            let (out_buf, staging_buf) = Self::alloc_out_pair(&self.device, self.out_words_cap);
            self.out_buf = out_buf;
            self.staging_buf = staging_buf;
        }
    }

    /// Pack a 256-entry RGB colormap LUT (`[[u8; 3]; 256]`) into little-endian RGBA8 `u32` words
    /// with alpha = 255.
    pub fn pack_rgb_lut(rgb: &[[u8; 3]; 256]) -> [u32; 256] {
        let mut packed = [0u32; 256];
        for (i, &[r, g, b]) in rgb.iter().enumerate() {
            packed[i] = (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | (0xffu32 << 24);
        }
        packed
    }

    /// Execute fused resample -> rescale -> colormap from a GPU-resident `GpuSlab` using
    /// separable 1D `(u_coords, v_coords)`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_slab(
        &mut self,
        slab: &GpuSlab,
        u_coords: &[f32],
        v_coords: &[f32],
        mode: GpuColormapMode,
        rescale_min: f32,
        rescale_max: f32,
        lut_rgba_u32: &[u32; 256],
        output: GpuPixelOutput,
    ) -> Result<Vec<u8>, GpuError> {
        self.render_slab_with_plan(
            slab,
            GpuCoordPlan::Separable1D { u_coords, v_coords },
            mode,
            rescale_min,
            rescale_max,
            lut_rgba_u32,
            output,
        )
    }

    /// Execute fused resample -> rescale -> colormap from a GPU-resident `GpuSlab` for any
    /// `GpuCoordPlan` (`Separable1D`, 17×17 `Grid2D`, or per-pixel `Exact2D`).
    #[allow(clippy::too_many_arguments)]
    pub fn render_slab_with_plan(
        &mut self,
        slab: &GpuSlab,
        plan: GpuCoordPlan<'_>,
        mode: GpuColormapMode,
        rescale_min: f32,
        rescale_max: f32,
        lut_rgba_u32: &[u32; 256],
        output: GpuPixelOutput,
    ) -> Result<Vec<u8>, GpuError> {
        let (dst_w, dst_h, coord_mode, u_slice, v_slice): (usize, usize, u32, &[f32], &[f32]) =
            match plan {
                GpuCoordPlan::Separable1D { u_coords, v_coords } => {
                    (u_coords.len(), v_coords.len(), 0, u_coords, v_coords)
                }
                GpuCoordPlan::Grid2D {
                    dst_w,
                    dst_h,
                    grid_u,
                    grid_v,
                } => (dst_w, dst_h, 1, grid_u.as_slice(), grid_v.as_slice()),
                GpuCoordPlan::Exact2D { dst_w, dst_h, u, v } => {
                    if u.len() < dst_w * dst_h || v.len() < dst_w * dst_h {
                        return Err(GpuError::InvalidDimensions(
                            "Exact2D u/v buffers shorter than dst_w * dst_h".into(),
                        ));
                    }
                    (dst_w, dst_h, 2, u, v)
                }
            };
        if dst_w == 0 || dst_h == 0 {
            return Err(GpuError::InvalidDimensions(
                "destination width/height must be > 0".into(),
            ));
        }

        let words_per_row = match output {
            GpuPixelOutput::Rgba8 => dst_w,
            GpuPixelOutput::IndexedU8 => dst_w.div_ceil(4),
        };
        let out_words = words_per_row * dst_h;
        self.ensure_capacities(u_slice.len(), v_slice.len(), out_words);

        let range = rescale_max - rescale_min;
        let safe_range = if range.is_finite() && range.abs() > 1e-20 {
            range
        } else {
            1.0
        };

        let params = ColormapParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            mode: mode.as_u32(),
            rescale_min,
            inv_range_255: 255.0 / safe_range,
            inv_range_254: 254.0 / safe_range,
            coord_mode,
            _pad1: 0,
            _pad2: 0,
        };

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(u_slice));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v_slice));
        if output == GpuPixelOutput::Rgba8 {
            self.queue
                .write_buffer(&self.lut_buf, 0, bytemuck::cast_slice(lut_rgba_u32));
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-cm-bg"),
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
                    resource: self.lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });

        let (pipeline, wg_x, wg_y) = match output {
            GpuPixelOutput::Rgba8 => (
                &self.rgba8_pipeline,
                (dst_w as u32).div_ceil(16),
                (dst_h as u32).div_ceil(16),
            ),
            GpuPixelOutput::IndexedU8 => (
                &self.indexed_pipeline,
                (words_per_row as u32).div_ceil(16),
                (dst_h as u32).div_ceil(16),
            ),
        };

        let out_bytes = (out_words * std::mem::size_of::<u32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-cm-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-cm-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
        let result = match output {
            GpuPixelOutput::Rgba8 => mapped.to_vec(),
            GpuPixelOutput::IndexedU8 => {
                if dst_w % 4 == 0 {
                    mapped.to_vec()
                } else {
                    let row_bytes = words_per_row * 4;
                    let mut out = Vec::with_capacity(dst_w * dst_h);
                    for py in 0..dst_h {
                        let start = py * row_bytes;
                        out.extend_from_slice(&mapped[start..start + dst_w]);
                    }
                    out
                }
            }
        };
        drop(mapped);
        self.staging_buf.unmap();

        Ok(result)
    }

    /// Resample a GPU-resident `GpuSlab` with an arbitrary `GpuCoordPlan` directly to raw
    /// little-endian `f32` bytes (`dst_w * dst_h * 4` bytes, for `.npy`, `.tif`, `.f32`, and mosaics).
    pub fn render_slab_f32_with_plan(
        &mut self,
        slab: &GpuSlab,
        plan: GpuCoordPlan<'_>,
        mode: GpuColormapMode,
    ) -> Result<Vec<u8>, GpuError> {
        let (dst_w, dst_h, coord_mode, u_slice, v_slice): (usize, usize, u32, &[f32], &[f32]) =
            match plan {
                GpuCoordPlan::Separable1D { u_coords, v_coords } => {
                    (u_coords.len(), v_coords.len(), 0, u_coords, v_coords)
                }
                GpuCoordPlan::Grid2D {
                    dst_w,
                    dst_h,
                    grid_u,
                    grid_v,
                } => (dst_w, dst_h, 1, grid_u.as_slice(), grid_v.as_slice()),
                GpuCoordPlan::Exact2D { dst_w, dst_h, u, v } => {
                    if u.len() != dst_w * dst_h || v.len() != dst_w * dst_h {
                        return Err(GpuError::InvalidDimensions(
                            "Exact2D u/v slice lengths must equal dst_w * dst_h".into(),
                        ));
                    }
                    (dst_w, dst_h, 2, u, v)
                }
            };
        if dst_w == 0 || dst_h == 0 {
            return Err(GpuError::InvalidDimensions(
                "dst_w and dst_h must be non-zero".into(),
            ));
        }

        let out_words = dst_w * dst_h;
        self.ensure_capacities(u_slice.len(), v_slice.len(), out_words);

        let params = ColormapParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            mode: mode.as_u32(),
            rescale_min: 0.0,
            inv_range_255: 1.0,
            inv_range_254: 1.0,
            coord_mode,
            _pad1: 0,
            _pad2: 0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(u_slice));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v_slice));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-cm-f32-bg"),
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
                    resource: self.lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });

        let out_bytes = (out_words * std::mem::size_of::<u32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-cm-f32-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-cm-f32-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.f32_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((dst_w as u32).div_ceil(16), (dst_h as u32).div_ceil(16), 1);
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

        let raw = slice.get_mapped_range().to_vec();
        self.staging_buf.unmap();
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::GpuResampler;
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
    use cogzilla_kernel::{BandSource, ColormapName, ResampleMode, sample_cubic, sample_general};

    #[test]
    fn fused_colormap_rgba8_and_indexed_match_cpu() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU test (no adapter): {e}");
                return;
            }
        };
        let resampler = GpuResampler::new(&ctx);
        let mut renderer = GpuColormapRenderer::new(&ctx);

        let src_w = 512u32;
        let src_h = 512u32;
        let mut src = vec![0.0f32; (src_w * src_h) as usize];
        for y in 0..src_h {
            for x in 0..src_w {
                let idx = (y * src_w + x) as usize;
                if x >= 90 && x < 120 && y >= 70 && y < 100 {
                    src[idx] = f32::NAN;
                } else {
                    src[idx] = 800.0 + 700.0 * (x as f32 * 0.02).sin() * (y as f32 * 0.025).cos();
                }
            }
        }

        let mut slab_cpu = AlignedF32Slab::new_uninit((src_w * src_h) as usize);
        slab_cpu.as_mut_slice().copy_from_slice(&src);
        let band = DecodedBand {
            data: BandData::F32(slab_cpu),
            width: src_w,
            height: src_h,
            stats: BlockStats {
                all_nodata: false,
                no_nodata: false,
                min: 100.0,
                max: 1500.0,
            },
        };
        let blocks = [Some(&band)];
        let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
        let gpu_slab = resampler.upload_slab(&src, src_w, src_h, src_w).unwrap();

        let lut = ColormapName::Viridis.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
        let rmin = 200.0f32;
        let rmax = 1400.0f32;
        let inv255 = 255.0 / (rmax - rmin);
        let inv254 = 254.0 / (rmax - rmin);

        let dst_n = 256usize;
        let u_coords: Vec<f32> = (0..dst_n)
            .map(|i| -0.75 + (i as f32) * ((src_w as f32 + 1.5) / dst_n as f32))
            .collect();
        let v_coords: Vec<f32> = (0..dst_n)
            .map(|i| -0.75 + (i as f32) * ((src_h as f32 + 1.5) / dst_n as f32))
            .collect();

        for &(cpu_mode, gpu_mode) in &[
            (ResampleMode::Bilinear, GpuColormapMode::Bilinear),
            (ResampleMode::Cubic, GpuColormapMode::Cubic),
            (ResampleMode::Lanczos, GpuColormapMode::Lanczos3),
        ] {
            let gpu_rgba = renderer
                .render_slab(
                    &gpu_slab,
                    &u_coords,
                    &v_coords,
                    gpu_mode,
                    rmin,
                    rmax,
                    &lut_u32,
                    GpuPixelOutput::Rgba8,
                )
                .unwrap();
            let gpu_idx = renderer
                .render_slab(
                    &gpu_slab,
                    &u_coords,
                    &v_coords,
                    gpu_mode,
                    rmin,
                    rmax,
                    &lut_u32,
                    GpuPixelOutput::IndexedU8,
                )
                .unwrap();

            let mut mask_mismatches = 0usize;
            let mut max_idx_diff = 0i32;
            let mut max_channel_diff = 0i32;

            for (py, &v) in v_coords.iter().enumerate() {
                for (px, &u) in u_coords.iter().enumerate() {
                    let val = match cpu_mode {
                        ResampleMode::Cubic => sample_cubic(&bs, u, v),
                        other => sample_general(&bs, u, v, other),
                    };
                    let p = py * dst_n + px;
                    let g_a = gpu_rgba[p * 4 + 3];
                    let g_i = gpu_idx[p];

                    if !val.is_finite() {
                        if g_a != 0 || g_i != 0 {
                            mask_mismatches += 1;
                        }
                    } else {
                        if g_a != 255 || g_i == 0 {
                            mask_mismatches += 1;
                            continue;
                        }
                        let ci = ((val - rmin) * inv255).clamp(0.0, 255.0) as usize;
                        let crgb = lut.rgb[ci];
                        for ch in 0..3 {
                            let d = (crgb[ch] as i32 - gpu_rgba[p * 4 + ch] as i32).abs();
                            max_channel_diff = max_channel_diff.max(d);
                        }
                        let c_idx = 1 + ((val - rmin) * inv254).clamp(0.0, 254.0) as i32;
                        max_idx_diff = max_idx_diff.max((c_idx - g_i as i32).abs());
                    }
                }
            }

            assert_eq!(mask_mismatches, 0, "{gpu_mode:?}: nodata mask must match 100%");
            assert!(
                max_idx_diff <= 1,
                "{gpu_mode:?}: indexed u8 diff {max_idx_diff} > 1 LSB"
            );
            assert!(
                max_channel_diff <= 4,
                "{gpu_mode:?}: RGBA8 channel diff {max_channel_diff} > 4 (1 LUT step)"
            );
        }
    }

    #[test]
    fn grid2d_curved_reprojection_matches_cpu() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU Grid2D test (no adapter): {e}");
                return;
            }
        };
        let resampler = GpuResampler::new(&ctx);
        let mut renderer = GpuColormapRenderer::new(&ctx);

        let src_w = 512u32;
        let src_h = 512u32;
        let mut src = vec![0.0f32; (src_w * src_h) as usize];
        for y in 0..src_h {
            for x in 0..src_w {
                let i = (y * src_w + x) as usize;
                if x >= 200 && x < 230 && y >= 180 && y < 210 {
                    src[i] = f32::NAN;
                } else {
                    src[i] = 500.0 + 900.0 * (x as f32 * 0.021).sin() * (y as f32 * 0.019).cos();
                }
            }
        }
        let mut slab_cpu = AlignedF32Slab::new_uninit(src.len());
        slab_cpu.as_mut_slice().copy_from_slice(&src);
        let band = DecodedBand {
            data: BandData::F32(slab_cpu),
            width: src_w,
            height: src_h,
            stats: BlockStats {
                all_nodata: false,
                no_nodata: false,
                min: -400.0,
                max: 1400.0,
            },
        };
        let blocks = [Some(&band)];
        let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
        let gpu_slab = resampler.upload_slab(&src, src_w, src_h, src_w).unwrap();

        // Construct a curved, rotated 17x17 UTM-like control grid
        let mut grid_u = [0.0f32; 289];
        let mut grid_v = [0.0f32; 289];
        let cos_a = 0.9659f32; // ~15 deg rotation
        let sin_a = 0.2588f32;
        for gy in 0..17 {
            for gx in 0..17 {
                let nx = (gx as f32 / 16.0) - 0.5;
                let ny = (gy as f32 / 16.0) - 0.5;
                let bend = 8.0 * nx * nx;
                grid_u[gy * 17 + gx] = 256.0 + 320.0 * (nx * cos_a - ny * sin_a);
                grid_v[gy * 17 + gx] = 256.0 + 320.0 * (nx * sin_a + ny * cos_a) + bend;
            }
        }

        let dst_n = 256usize;
        let lut = ColormapName::Viridis.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
        let rmin = -400.0f32;
        let rmax = 1400.0f32;
        let inv254 = 254.0 / (rmax - rmin);

        let gpu_idx = renderer
            .render_slab_with_plan(
                &gpu_slab,
                GpuCoordPlan::Grid2D {
                    dst_w: dst_n,
                    dst_h: dst_n,
                    grid_u: &grid_u,
                    grid_v: &grid_v,
                },
                GpuColormapMode::Cubic,
                rmin,
                rmax,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();

        let cell_px = dst_n / 16;
        let inv_cell = 1.0 / cell_px as f32;
        let mut mask_err = 0usize;
        let mut max_diff = 0i32;
        for py in 0..dst_n {
            let gy = (py / cell_px).min(15);
            let ty = (py - gy * cell_px) as f32 * inv_cell;
            let r0 = gy * 17;
            let r1 = (gy + 1) * 17;
            for px in 0..dst_n {
                let gx = (px / cell_px).min(15);
                let tx = (px - gx * cell_px) as f32 * inv_cell;
                let u_l = grid_u[r0 + gx] + (grid_u[r1 + gx] - grid_u[r0 + gx]) * ty;
                let u_r = grid_u[r0 + gx + 1] + (grid_u[r1 + gx + 1] - grid_u[r0 + gx + 1]) * ty;
                let v_l = grid_v[r0 + gx] + (grid_v[r1 + gx] - grid_v[r0 + gx]) * ty;
                let v_r = grid_v[r0 + gx + 1] + (grid_v[r1 + gx + 1] - grid_v[r0 + gx + 1]) * ty;
                let u = u_l + (u_r - u_l) * tx;
                let v = v_l + (v_r - v_l) * tx;
                let val = sample_cubic(&bs, u, v);
                let g = gpu_idx[py * dst_n + px];
                if !val.is_finite() {
                    if g != 0 {
                        mask_err += 1;
                    }
                } else if g == 0 {
                    mask_err += 1;
                } else {
                    let c = 1 + ((val - rmin) * inv254).clamp(0.0, 254.0) as i32;
                    max_diff = max_diff.max((c - g as i32).abs());
                }
            }
        }
        assert_eq!(mask_err, 0, "Grid2D nodata mask mismatch");
        assert!(max_diff <= 1, "Grid2D max diff {max_diff} > 1 LSB");
    }
}
