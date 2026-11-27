use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::colormap::{GpuColormapMode, GpuCoordPlan, GpuPixelOutput};
use crate::context::{GpuContext, GpuError};
use crate::resample::GpuSlab;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuDemAlgorithm {
    /// Horn 3×3 shaded relief in `[0.0, 255.0]`.
    Hillshade {
        azimuth_deg: f32,
        altitude_deg: f32,
        z_factor: f32,
    },
    /// Terrain slope in degrees `[0.0, 90.0]`.
    Slope { z_factor: f32 },
    /// Compass aspect in degrees `[0.0, 360.0)`.
    Aspect,
    /// Mapzen / AWS Terrarium 24-bit RGB elevation encoding.
    Terrarium,
    /// Mapbox Terrain-RGB 24-bit elevation encoding.
    TerrainRgb,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TerrainParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    resample_mode: u32,
    alg_mode: u32,
    coord_mode: u32,
    inv_8dx: f32,
    inv_8dy: f32,
    z_factor: f32,
    sun_e: f32,
    sun_n: f32,
    sun_u: f32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

/// Single-pass fused GPU DEM & terrain renderer using 18×18 workgroup shared-memory tiles
/// for 3×3 Horn stencils (`Hillshade`, `Slope`, `Aspect`) plus direct `Terrarium`/`TerrainRgb`.
pub struct GpuTerrainRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    horn_rgba8_pipeline: wgpu::ComputePipeline,
    horn_indexed_pipeline: wgpu::ComputePipeline,
    horn_f32_pipeline: wgpu::ComputePipeline,
    terrain_rgb_pipeline: wgpu::ComputePipeline,
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

impl GpuTerrainRenderer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-terrain-wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::TERRAIN_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-terrain-bgl"),
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
            label: Some("cogzilla-terrain-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let horn_rgba8_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cogzilla-dem-horn-rgba8"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("dem_horn_rgba8"),
                compilation_options: Default::default(),
                cache: None,
            });

        let horn_indexed_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cogzilla-dem-horn-indexed"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("dem_horn_indexed_u8"),
                compilation_options: Default::default(),
                cache: None,
            });

        let horn_f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-dem-horn-f32"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dem_horn_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let terrain_rgb_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cogzilla-dem-terrain-rgb"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("dem_terrain_rgb"),
                compilation_options: Default::default(),
                cache: None,
            });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-terrain-params"),
            size: std::mem::size_of::<TerrainParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-terrain-lut"),
            size: (256 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_coord_cap = 512usize;
        let u_buf = Self::alloc_coord_buf(&device, "cogzilla-tr-u", init_coord_cap);
        let v_buf = Self::alloc_coord_buf(&device, "cogzilla-tr-v", init_coord_cap);

        let init_words = 512 * 512;
        let (out_buf, staging_buf) = Self::alloc_out_pair(&device, init_words);

        Self {
            device,
            queue,
            bind_group_layout,
            horn_rgba8_pipeline,
            horn_indexed_pipeline,
            horn_f32_pipeline,
            terrain_rgb_pipeline,
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
            label: Some("cogzilla-tr-out"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-tr-staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (out_buf, staging_buf)
    }

    fn ensure_capacities(&mut self, dst_w: usize, dst_h: usize, out_words: usize) {
        if dst_w > self.u_cap {
            self.u_cap = dst_w.next_power_of_two();
            self.u_buf = Self::alloc_coord_buf(&self.device, "cogzilla-tr-u", self.u_cap);
        }
        if dst_h > self.v_cap {
            self.v_cap = dst_h.next_power_of_two();
            self.v_buf = Self::alloc_coord_buf(&self.device, "cogzilla-tr-v", self.v_cap);
        }
        if out_words > self.out_words_cap {
            self.out_words_cap = out_words;
            let (out_buf, staging_buf) = Self::alloc_out_pair(&self.device, self.out_words_cap);
            self.out_buf = out_buf;
            self.staging_buf = staging_buf;
        }
    }

    /// Render a DEM/terrain product from a GPU-resident `GpuSlab` using separable 1D coordinates.
    #[allow(clippy::too_many_arguments)]
    pub fn render_slab(
        &mut self,
        slab: &GpuSlab,
        u_coords: &[f32],
        v_coords: &[f32],
        resample: GpuColormapMode,
        alg: GpuDemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
        rescale_min: f32,
        rescale_max: f32,
        lut_rgba_u32: &[u32; 256],
        output: GpuPixelOutput,
    ) -> Result<Vec<u8>, GpuError> {
        self.render_slab_with_plan(
            slab,
            GpuCoordPlan::Separable1D { u_coords, v_coords },
            resample,
            alg,
            pixel_size_x_m,
            pixel_size_y_m,
            rescale_min,
            rescale_max,
            lut_rgba_u32,
            output,
        )
    }

    /// Render a DEM/terrain product from a GPU-resident `GpuSlab` for any `GpuCoordPlan`
    /// (`Separable1D`, 17×17 `Grid2D` UTM/Albers/Polar, or `Exact2D`).
    #[allow(clippy::too_many_arguments)]
    pub fn render_slab_with_plan(
        &mut self,
        slab: &GpuSlab,
        plan: GpuCoordPlan<'_>,
        resample: GpuColormapMode,
        alg: GpuDemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
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

        let is_terrain_rgb = matches!(
            alg,
            GpuDemAlgorithm::Terrarium | GpuDemAlgorithm::TerrainRgb
        );
        let eff_output = if is_terrain_rgb {
            GpuPixelOutput::Rgba8
        } else {
            output
        };

        let words_per_row = match eff_output {
            GpuPixelOutput::Rgba8 => dst_w,
            GpuPixelOutput::IndexedU8 => dst_w.div_ceil(4),
        };
        let out_words = words_per_row * dst_h;
        self.ensure_capacities(u_slice.len(), v_slice.len(), out_words);

        let (alg_mode, z_factor, sun_e, sun_n, sun_u) = match alg {
            GpuDemAlgorithm::Hillshade {
                azimuth_deg,
                altitude_deg,
                z_factor,
            } => {
                let zenith = (90.0 - altitude_deg).to_radians();
                let az = azimuth_deg.to_radians();
                (
                    0u32,
                    z_factor,
                    az.sin() * zenith.sin(),
                    az.cos() * zenith.sin(),
                    zenith.cos(),
                )
            }
            GpuDemAlgorithm::Slope { z_factor } => (1u32, z_factor, 0.0, 0.0, 1.0),
            GpuDemAlgorithm::Aspect => (2u32, 1.0, 0.0, 0.0, 1.0),
            GpuDemAlgorithm::Terrarium => (3u32, 1.0, 0.0, 0.0, 1.0),
            GpuDemAlgorithm::TerrainRgb => (4u32, 1.0, 0.0, 0.0, 1.0),
        };

        let resample_mode = match resample {
            GpuColormapMode::Bilinear => 0u32,
            GpuColormapMode::Cubic => 1u32,
            GpuColormapMode::Lanczos3 => 2u32,
            GpuColormapMode::Nearest => 3u32,
        };

        let range = rescale_max - rescale_min;
        let safe_range = if range.is_finite() && range.abs() > 1e-12 {
            range
        } else {
            1.0
        };

        let params = TerrainParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            resample_mode,
            alg_mode,
            coord_mode,
            inv_8dx: 1.0 / (8.0 * pixel_size_x_m.max(1e-6)),
            inv_8dy: 1.0 / (8.0 * pixel_size_y_m.max(1e-6)),
            z_factor,
            sun_e,
            sun_n,
            sun_u,
            rescale_min,
            inv_range_255: 255.0 / safe_range,
            inv_range_254: 254.0 / safe_range,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(u_slice));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v_slice));
        if !is_terrain_rgb && eff_output == GpuPixelOutput::Rgba8 {
            self.queue
                .write_buffer(&self.lut_buf, 0, bytemuck::cast_slice(lut_rgba_u32));
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-tr-bg"),
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

        let pipeline = if is_terrain_rgb {
            &self.terrain_rgb_pipeline
        } else {
            match eff_output {
                GpuPixelOutput::Rgba8 => &self.horn_rgba8_pipeline,
                GpuPixelOutput::IndexedU8 => &self.horn_indexed_pipeline,
            }
        };

        let out_bytes = (out_words * std::mem::size_of::<u32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-tr-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-tr-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                (dst_w as u32).div_ceil(16),
                (dst_h as u32).div_ceil(16),
                1,
            );
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
        let result = match eff_output {
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

    /// Evaluate Horn 3×3 (`Hillshade` / `Slope` / `Aspect`) over a GPU-resident `GpuSlab` with
    /// an arbitrary `GpuCoordPlan`, returning raw little-endian `f32` bytes (`dst_w * dst_h * 4`).
    #[allow(clippy::too_many_arguments)]
    pub fn render_slab_f32_with_plan(
        &mut self,
        slab: &GpuSlab,
        plan: GpuCoordPlan<'_>,
        resample: GpuColormapMode,
        alg: GpuDemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
    ) -> Result<Vec<u8>, GpuError> {
        let (alg_mode, sun_e, sun_n, sun_u, z_factor) = match alg {
            GpuDemAlgorithm::Hillshade {
                azimuth_deg,
                altitude_deg,
                z_factor,
            } => {
                let az = azimuth_deg.to_radians();
                let alt = altitude_deg.to_radians();
                (0u32, alt.cos() * az.sin(), alt.cos() * az.cos(), alt.sin(), z_factor)
            }
            GpuDemAlgorithm::Slope { z_factor } => (1u32, 0.0, 0.0, 1.0, z_factor),
            GpuDemAlgorithm::Aspect => (2u32, 0.0, 0.0, 1.0, 1.0),
            GpuDemAlgorithm::Terrarium | GpuDemAlgorithm::TerrainRgb => {
                return Err(GpuError::InvalidDimensions(
                    "render_slab_f32_with_plan supports Hillshade/Slope/Aspect only".into(),
                ));
            }
        };
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

        let params = TerrainParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            resample_mode: resample.as_u32(),
            alg_mode,
            coord_mode,
            inv_8dx: 1.0 / (8.0 * pixel_size_x_m.max(1e-3)),
            inv_8dy: 1.0 / (8.0 * pixel_size_y_m.max(1e-3)),
            z_factor,
            sun_e,
            sun_n,
            sun_u,
            rescale_min: 0.0,
            inv_range_255: 1.0,
            inv_range_254: 1.0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(u_slice));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v_slice));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-tr-f32-bg"),
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
                label: Some("cogzilla-tr-f32-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-tr-f32-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.horn_f32_pipeline);
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
    use crate::colormap::GpuColormapRenderer;
    use crate::resample::GpuResampler;
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
    use cogzilla_kernel::{
        BandSource, ColormapName, DemAlgorithm, encode_terrain_rgb, encode_terrarium_rgb,
        eval_horn_3x3_aniso, sample_cubic,
    };

    #[test]
    fn gpu_hillshade_slope_terrarium_terrainrgb_match_cpu() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU test (no adapter): {e}");
                return;
            }
        };
        let resampler = GpuResampler::new(&ctx);
        let mut terrain = GpuTerrainRenderer::new(&ctx);

        let src_w = 512u32;
        let src_h = 512u32;
        let mut src = vec![0.0f32; (src_w * src_h) as usize];
        for y in 0..src_h {
            for x in 0..src_w {
                let idx = (y * src_w + x) as usize;
                if x >= 110 && x < 130 && y >= 90 && y < 110 {
                    src[idx] = f32::NAN;
                } else {
                    let fx = x as f32 * 0.03;
                    let fy = y as f32 * 0.025;
                    src[idx] = 1500.0 + 800.0 * fx.sin() * fy.cos() + 120.0 * (fx * 2.3 + fy).sin();
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
                min: 500.0,
                max: 2500.0,
            },
        };
        let blocks = [Some(&band)];
        let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
        let gpu_slab = resampler.upload_slab(&src, src_w, src_h, src_w).unwrap();

        let n = 256usize;
        let u_coords: Vec<f32> = (0..n)
            .map(|i| 4.0 + (i as f32) * ((src_w as f32 - 8.0) / n as f32))
            .collect();
        let v_coords: Vec<f32> = (0..n)
            .map(|i| 4.0 + (i as f32) * ((src_h as f32 - 8.0) / n as f32))
            .collect();

        // Build CPU (n+2)x(n+2) padded elevation grid
        let pad = n + 2;
        let mut elev = vec![f32::NAN; pad * pad];
        let u_ext = |ix: i32| -> f32 {
            if ix < 0 {
                2.0 * u_coords[0] - u_coords[1]
            } else if ix >= n as i32 {
                2.0 * u_coords[n - 1] - u_coords[n - 2]
            } else {
                u_coords[ix as usize]
            }
        };
        let v_ext = |iy: i32| -> f32 {
            if iy < 0 {
                2.0 * v_coords[0] - v_coords[1]
            } else if iy >= n as i32 {
                2.0 * v_coords[n - 1] - v_coords[n - 2]
            } else {
                v_coords[iy as usize]
            }
        };
        for py in 0..pad {
            let v = v_ext(py as i32 - 1);
            for px in 0..pad {
                let u = u_ext(px as i32 - 1);
                elev[py * pad + px] = sample_cubic(&bs, u, v);
            }
        }

        let px_m = 30.0f32;
        let inv_8d = 1.0 / (8.0 * px_m);
        let lut = ColormapName::Viridis.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);

        // 1. Hillshade IndexedU8 parity
        let gpu_hs = terrain
            .render_slab(
                &gpu_slab,
                &u_coords,
                &v_coords,
                GpuColormapMode::Cubic,
                GpuDemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 1.5,
                },
                px_m,
                px_m,
                0.0,
                255.0,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();
        let cpu_alg = DemAlgorithm::Hillshade {
            azimuth_deg: 315.0,
            altitude_deg: 45.0,
            z_factor: 1.5,
        };
        let inv254 = 254.0 / 255.0;
        let mut max_hs_diff = 0i32;
        let mut mask_err = 0usize;
        for py in 0..n {
            for px in 0..n {
                let r0 = py * pad + px;
                let r1 = (py + 1) * pad + px;
                let r2 = (py + 2) * pad + px;
                let val = eval_horn_3x3_aniso(
                    elev[r0],
                    elev[r0 + 1],
                    elev[r0 + 2],
                    elev[r1],
                    elev[r1 + 1],
                    elev[r1 + 2],
                    elev[r2],
                    elev[r2 + 1],
                    elev[r2 + 2],
                    inv_8d,
                    inv_8d,
                    cpu_alg,
                );
                let g = gpu_hs[py * n + px];
                if !val.is_finite() {
                    if g != 0 {
                        mask_err += 1;
                    }
                } else {
                    if g == 0 {
                        mask_err += 1;
                    } else {
                        let c = 1 + (val * inv254).clamp(0.0, 254.0) as i32;
                        max_hs_diff = max_hs_diff.max((c - g as i32).abs());
                    }
                }
            }
        }
        assert_eq!(mask_err, 0, "hillshade nodata mask mismatch");
        assert!(max_hs_diff <= 1, "hillshade max diff {max_hs_diff} > 1");

        // 2. Terrarium & Terrain-RGB parity
        for &(alg, is_terrarium) in &[
            (GpuDemAlgorithm::Terrarium, true),
            (GpuDemAlgorithm::TerrainRgb, false),
        ] {
            let gpu_rgb = terrain
                .render_slab(
                    &gpu_slab,
                    &u_coords,
                    &v_coords,
                    GpuColormapMode::Cubic,
                    alg,
                    px_m,
                    px_m,
                    0.0,
                    255.0,
                    &lut_u32,
                    GpuPixelOutput::Rgba8,
                )
                .unwrap();
            let mut max_b_diff = 0i32;
            let mut rgb_mask_err = 0usize;
            for py in 0..n {
                for px in 0..n {
                    let z = elev[(py + 1) * pad + (px + 1)];
                    let p = (py * n + px) * 4;
                    if !z.is_finite() {
                        if gpu_rgb[p + 3] != 0 {
                            rgb_mask_err += 1;
                        }
                    } else {
                        if gpu_rgb[p + 3] != 255 {
                            rgb_mask_err += 1;
                            continue;
                        }
                        let c = if is_terrarium {
                            encode_terrarium_rgb(z)
                        } else {
                            encode_terrain_rgb(z)
                        };
                        // Compare reconstructed 24-bit integer (handles carry across B->G cleanly)
                        let c24 = ((c[0] as i32) << 16) | ((c[1] as i32) << 8) | (c[2] as i32);
                        let g24 = ((gpu_rgb[p] as i32) << 16)
                            | ((gpu_rgb[p + 1] as i32) << 8)
                            | (gpu_rgb[p + 2] as i32);
                        max_b_diff = max_b_diff.max((c24 - g24).abs());
                    }
                }
            }
            assert_eq!(rgb_mask_err, 0, "{alg:?} nodata mask mismatch");
            assert!(
                max_b_diff <= 2,
                "{alg:?} 24-bit encoding diff {max_b_diff} > 2 LSB"
            );
        }
    }

    #[test]
    fn gpu_curved_grid2d_hillshade_matches_cpu() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU curved terrain test (no adapter): {e}");
                return;
            }
        };
        let resampler = GpuResampler::new(&ctx);
        let mut terrain = GpuTerrainRenderer::new(&ctx);

        let src_w = 512u32;
        let src_h = 512u32;
        let mut src = vec![0.0f32; (src_w * src_h) as usize];
        for y in 0..src_h {
            for x in 0..src_w {
                let fx = x as f32 * 0.025;
                let fy = y as f32 * 0.021;
                src[(y * src_w + x) as usize] = 1500.0 + 700.0 * fx.sin() * fy.cos();
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
                no_nodata: true,
                min: 700.0,
                max: 2300.0,
            },
        };
        let blocks = [Some(&band)];
        let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
        let gpu_slab = resampler.upload_slab(&src, src_w, src_h, src_w).unwrap();

        let mut grid_u = [0.0f32; 289];
        let mut grid_v = [0.0f32; 289];
        let cos_a = 0.9659f32;
        let sin_a = 0.2588f32;
        for gy in 0..17 {
            for gx in 0..17 {
                let nx = (gx as f32 / 16.0) - 0.5;
                let ny = (gy as f32 / 16.0) - 0.5;
                grid_u[gy * 17 + gx] = 256.0 + 300.0 * (nx * cos_a - ny * sin_a);
                grid_v[gy * 17 + gx] = 256.0 + 300.0 * (nx * sin_a + ny * cos_a) + 6.0 * nx * nx;
            }
        }

        let n = 256usize;
        let lut = ColormapName::Viridis.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
        let px_m = 30.0f32;

        let gpu_hs = terrain
            .render_slab_with_plan(
                &gpu_slab,
                GpuCoordPlan::Grid2D {
                    dst_w: n,
                    dst_h: n,
                    grid_u: &grid_u,
                    grid_v: &grid_v,
                },
                GpuColormapMode::Cubic,
                GpuDemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 1.5,
                },
                px_m,
                px_m,
                0.0,
                255.0,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();

        // Build CPU interior elevation grid and check interior pixels (1..n-1)
        let cell_px = n / 16;
        let inv_cell = 1.0 / cell_px as f32;
        let mut elev = vec![0.0f32; n * n];
        for py in 0..n {
            let gy = (py / cell_px).min(15);
            let ty = (py - gy * cell_px) as f32 * inv_cell;
            let r0 = gy * 17;
            let r1 = (gy + 1) * 17;
            for px in 0..n {
                let gx = (px / cell_px).min(15);
                let tx = (px - gx * cell_px) as f32 * inv_cell;
                let u_l = grid_u[r0 + gx] + (grid_u[r1 + gx] - grid_u[r0 + gx]) * ty;
                let u_r = grid_u[r0 + gx + 1] + (grid_u[r1 + gx + 1] - grid_u[r0 + gx + 1]) * ty;
                let v_l = grid_v[r0 + gx] + (grid_v[r1 + gx] - grid_v[r0 + gx]) * ty;
                let v_r = grid_v[r0 + gx + 1] + (grid_v[r1 + gx + 1] - grid_v[r0 + gx + 1]) * ty;
                elev[py * n + px] = sample_cubic(&bs, u_l + (u_r - u_l) * tx, v_l + (v_r - v_l) * tx);
            }
        }
        let inv_8d = 1.0 / (8.0 * px_m);
        let inv254 = 254.0 / 255.0;
        let cpu_alg = DemAlgorithm::Hillshade {
            azimuth_deg: 315.0,
            altitude_deg: 45.0,
            z_factor: 1.5,
        };
        let mut max_diff = 0i32;
        for py in 1..(n - 1) {
            for px in 1..(n - 1) {
                let r0 = (py - 1) * n + (px - 1);
                let r1 = py * n + (px - 1);
                let r2 = (py + 1) * n + (px - 1);
                let val = eval_horn_3x3_aniso(
                    elev[r0], elev[r0 + 1], elev[r0 + 2],
                    elev[r1], elev[r1 + 1], elev[r1 + 2],
                    elev[r2], elev[r2 + 1], elev[r2 + 2],
                    inv_8d, inv_8d, cpu_alg,
                );
                let c = 1 + (val * inv254).clamp(0.0, 254.0) as i32;
                let g = gpu_hs[py * n + px] as i32;
                max_diff = max_diff.max((c - g).abs());
            }
        }
        assert!(max_diff <= 1, "curved Grid2D hillshade max diff {max_diff} > 1 LSB");
    }
}
