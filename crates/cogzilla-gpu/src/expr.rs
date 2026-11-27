use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::colormap::{GpuColormapMode, GpuCoordPlan, GpuPixelOutput};
use crate::context::{GpuContext, GpuError};

/// GPU-resident stacked multi-band `f32` slab (`num_bands` contiguous `height * stride` planes).
pub struct GpuMultiBandSlab {
    pub(crate) buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub band_stride: u32,
    pub num_bands: u32,
}

impl GpuMultiBandSlab {
    /// Allocate a `STORAGE` buffer with `mapped_at_creation: true` and invoke `fill` directly
    /// on the mapped `&mut [f32]` slice (`num_bands * band_stride` floats) — eliminating all
    /// intermediate CPU `Vec<f32>` allocations and extra copies, with no renderer mutex lock.
    pub fn from_mapped_fill<F>(
        device: &wgpu::Device,
        num_bands: usize,
        width: u32,
        height: u32,
        stride: u32,
        fill: F,
    ) -> Result<Self, GpuError>
    where
        F: FnOnce(&mut [f32]) -> Result<(), GpuError>,
    {
        if num_bands == 0 || num_bands > 8 || width == 0 || height == 0 || stride < width {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid multi-band slab: bands={num_bands}, {width}x{height} stride={stride}"
            )));
        }
        let band_stride = (height as usize)
            .checked_mul(stride as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("band_stride overflow".into()))?;
        let total_elems = band_stride
            .checked_mul(num_bands)
            .ok_or_else(|| GpuError::InvalidDimensions("total_elems overflow".into()))?;
        let byte_size = (total_elems * std::mem::size_of::<f32>()) as u64;
        let limits = device.limits();
        let max_bytes =
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        if byte_size > max_bytes {
            return Err(GpuError::InvalidDimensions(format!(
                "multi-band slab {byte_size} B exceeds adapter limit {max_bytes} B"
            )));
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-multiband-slab"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        let res = {
            let mut mapped = buffer.slice(..).get_mapped_range_mut();
            let dst: &mut [f32] = bytemuck::cast_slice_mut(&mut mapped);
            fill(dst)
        };
        buffer.unmap();
        res?;
        Ok(Self {
            buffer,
            width,
            height,
            stride,
            band_stride: band_stride as u32,
            num_bands: num_bands as u32,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuInstr {
    pub opcode: u32,
    pub imm_u32: u32,
    pub imm_f32: f32,
    pub _pad: u32,
}

impl GpuInstr {
    pub const fn push_const(v: f32) -> Self {
        Self { opcode: 0, imm_u32: 0, imm_f32: v, _pad: 0 }
    }
    pub const fn load_band(slot: u32) -> Self {
        Self { opcode: 1, imm_u32: slot, imm_f32: 0.0, _pad: 0 }
    }
    pub const fn op(opcode: u32) -> Self {
        Self { opcode, imm_u32: 0, imm_f32: 0.0, _pad: 0 }
    }
}

/// Opcode constants matching `shaders/expr.wgsl`.
pub mod opcodes {
    pub const ADD: u32 = 2;
    pub const SUB: u32 = 3;
    pub const MUL: u32 = 4;
    pub const DIV: u32 = 5;
    pub const POW: u32 = 6;
    pub const NEG: u32 = 7;
    pub const LOG10: u32 = 8;
    pub const LN: u32 = 9;
    pub const LOG2: u32 = 10;
    pub const DB: u32 = 11;
    pub const EXP: u32 = 12;
    pub const SQRT: u32 = 13;
    pub const ABS: u32 = 14;
    pub const SIN: u32 = 15;
    pub const COS: u32 = 16;
    pub const TAN: u32 = 17;
    pub const MIN: u32 = 18;
    pub const MAX: u32 = 19;
    pub const CLIP: u32 = 20;
    pub const LT: u32 = 21;
    pub const LE: u32 = 22;
    pub const GT: u32 = 23;
    pub const GE: u32 = 24;
    pub const EQ: u32 = 25;
    pub const NE: u32 = 26;
    pub const WHERE: u32 = 27;
    pub const CEIL: u32 = 28;
    pub const FLOOR: u32 = 29;
    pub const ROUND: u32 = 30;
}

/// Pre-fused RGB LUT + optional CIE LCH (D65) saturation for GPU `color_formula` execution.
#[derive(Debug, Clone)]
pub struct GpuColorFormula {
    pub rgb_lut: Option<[u32; 256]>,
    pub saturation: Option<f32>,
}

impl GpuColorFormula {
    /// Pack separate `[u8; 256]` R/G/B LUTs (e.g. from Gamma + Sigmoidal) into 256 `u32` words.
    pub fn pack_rgb_luts(r: &[u8; 256], g: &[u8; 256], b: &[u8; 256]) -> [u32; 256] {
        let mut out = [0u32; 256];
        for i in 0..256 {
            out[i] = (r[i] as u32) | ((g[i] as u32) << 8) | ((b[i] as u32) << 16);
        }
        out
    }

    /// Build Gamma + Sigmoidal LUT + optional LCH Saturation directly.
    pub fn from_gamma_sigmoidal_sat(
        gamma: Option<f32>,
        sigmoidal: Option<(f32, f32)>,
        saturation: Option<f32>,
    ) -> Self {
        let rgb_lut = if gamma.is_some() || sigmoidal.is_some() {
            let mut lut = [0u8; 256];
            for (i, slot) in lut.iter_mut().enumerate() {
                let mut v = (i as f32) / 255.0;
                if let Some(g) = gamma {
                    v = v.clamp(0.0, 1.0).powf(1.0 / g);
                }
                if let Some((contrast, bias)) = sigmoidal {
                    let s0 = 1.0 / (1.0 + (contrast * bias).exp());
                    let s1 = 1.0 / (1.0 + (contrast * (bias - 1.0)).exp());
                    let sv = 1.0 / (1.0 + (contrast * (bias - v)).exp());
                    v = ((sv - s0) / (s1 - s0)).clamp(0.0, 1.0);
                }
                *slot = (v * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            Some(Self::pack_rgb_luts(&lut, &lut, &lut))
        } else {
            None
        };
        Self {
            rgb_lut,
            saturation,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ExprParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    band_stride: u32,
    dst_w: u32,
    dst_h: u32,
    num_bands: u32,
    num_instrs: u32,
    resample_mode: u32,
    formula_flags: u32,
    coord_mode: u32,
    _pad1: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    saturation: f32,
    rgb_min: [f32; 4],
    rgb_inv_255: [f32; 4],
}

/// Fused GPU multi-band expression VM + 3-band RGB `color_formula` renderer.
pub struct GpuExprRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    expr_rgba8_pipeline: wgpu::ComputePipeline,
    expr_indexed_pipeline: wgpu::ComputePipeline,
    expr_f32_pipeline: wgpu::ComputePipeline,
    rgb_formula_pipeline: wgpu::ComputePipeline,
    sar_rgb_formula_pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    instr_buf: wgpu::Buffer,
    lut_buf: wgpu::Buffer,
    u_buf: wgpu::Buffer,
    u_cap: usize,
    v_buf: wgpu::Buffer,
    v_cap: usize,
    out_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    out_words_cap: usize,
}

impl GpuExprRenderer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-expr-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/expr.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-expr-bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-expr-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let expr_rgba8_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cogzilla-expr-rgba8"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("expr_eval_rgba8"),
                compilation_options: Default::default(),
                cache: None,
            });
        let expr_indexed_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cogzilla-expr-indexed"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("expr_eval_indexed_u8"),
                compilation_options: Default::default(),
                cache: None,
            });
        let expr_f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-expr-f32"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("expr_eval_f32"),
            compilation_options: Default::default(),
            cache: None,
        });
        let rgb_formula_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cogzilla-rgb-formula"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("rgb_color_formula_rgba8"),
                compilation_options: Default::default(),
                cache: None,
            });
        let sar_rgb_formula_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cogzilla-sar-rgb-formula"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("sar_rgb_color_formula_rgba8"),
                compilation_options: Default::default(),
                cache: None,
            });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-expr-params"),
            size: std::mem::size_of::<ExprParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instr_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-expr-instrs"),
            size: (256 * std::mem::size_of::<GpuInstr>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-expr-lut"),
            size: (256 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_coord_cap = 512usize;
        let u_buf = Self::alloc_coord_buf(&device, "cogzilla-expr-u", init_coord_cap);
        let v_buf = Self::alloc_coord_buf(&device, "cogzilla-expr-v", init_coord_cap);

        let init_words = 512 * 512;
        let (out_buf, staging_buf) = Self::alloc_out_pair(&device, init_words);

        Self {
            device,
            queue,
            bind_group_layout,
            expr_rgba8_pipeline,
            expr_indexed_pipeline,
            expr_f32_pipeline,
            rgb_formula_pipeline,
            sar_rgb_formula_pipeline,
            params_buf,
            instr_buf,
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
            label: Some("cogzilla-expr-out"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-expr-staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (out_buf, staging_buf)
    }

    fn ensure_capacities(&mut self, dst_w: usize, dst_h: usize, out_words: usize) {
        if dst_w > self.u_cap {
            self.u_cap = dst_w.next_power_of_two();
            self.u_buf = Self::alloc_coord_buf(&self.device, "cogzilla-expr-u", self.u_cap);
        }
        if dst_h > self.v_cap {
            self.v_cap = dst_h.next_power_of_two();
            self.v_buf = Self::alloc_coord_buf(&self.device, "cogzilla-expr-v", self.v_cap);
        }
        if out_words > self.out_words_cap {
            self.out_words_cap = out_words;
            let (out_buf, staging_buf) = Self::alloc_out_pair(&self.device, self.out_words_cap);
            self.out_buf = out_buf;
            self.staging_buf = staging_buf;
        }
    }

    /// Upload 1..=8 planar `f32` bands of equal `(width, height, stride)` into a single GPU buffer.
    pub fn upload_bands(
        &self,
        bands: &[&[f32]],
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<GpuMultiBandSlab, GpuError> {
        if bands.is_empty() || bands.len() > 8 || width == 0 || height == 0 || stride < width {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid multi-band slab: bands={}, {width}x{height} stride={stride}",
                bands.len()
            )));
        }
        let band_stride = (height as usize)
            .checked_mul(stride as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("band_stride overflow".into()))?;
        let mut packed = Vec::with_capacity(band_stride * bands.len());
        for (i, b) in bands.iter().enumerate() {
            if b.len() < band_stride {
                return Err(GpuError::InvalidDimensions(format!(
                    "band {i} len {} < {band_stride}",
                    b.len()
                )));
            }
            packed.extend_from_slice(&b[..band_stride]);
        }
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cogzilla-multiband-slab"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(GpuMultiBandSlab {
            buffer,
            width,
            height,
            stride,
            band_stride: band_stride as u32,
            num_bands: bands.len() as u32,
        })
    }

    /// Evaluate a bytecode expression over a GPU-resident `GpuMultiBandSlab` and emit `Rgba8` or `IndexedU8`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_expr(
        &mut self,
        slab: &GpuMultiBandSlab,
        u_coords: &[f32],
        v_coords: &[f32],
        resample: GpuColormapMode,
        instrs: &[GpuInstr],
        rescale_min: f32,
        rescale_max: f32,
        lut_rgba_u32: &[u32; 256],
        output: GpuPixelOutput,
    ) -> Result<Vec<u8>, GpuError> {
        self.render_expr_with_plan(
            slab,
            GpuCoordPlan::Separable1D {
                u_coords,
                v_coords,
            },
            resample,
            instrs,
            rescale_min,
            rescale_max,
            lut_rgba_u32,
            output,
        )
    }

    /// Evaluate a bytecode expression with an arbitrary `GpuCoordPlan` (`Separable1D`, `Grid2D` 17×17,
    /// or `Exact2D`) over a GPU-resident `GpuMultiBandSlab`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_expr_with_plan(
        &mut self,
        slab: &GpuMultiBandSlab,
        plan: GpuCoordPlan<'_>,
        resample: GpuColormapMode,
        instrs: &[GpuInstr],
        rescale_min: f32,
        rescale_max: f32,
        lut_rgba_u32: &[u32; 256],
        output: GpuPixelOutput,
    ) -> Result<Vec<u8>, GpuError> {
        let (dst_w, dst_h, coord_mode, u_slice, v_slice): (usize, usize, u32, &[f32], &[f32]) =
            match plan {
                GpuCoordPlan::Separable1D {
                    u_coords: u,
                    v_coords: v,
                } => (u.len(), v.len(), 0, u, v),
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
        if dst_w == 0 || dst_h == 0 || instrs.is_empty() || instrs.len() > 256 {
            return Err(GpuError::InvalidDimensions(
                "invalid expr dimensions or instruction count".into(),
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
        let resample_mode = match resample {
            GpuColormapMode::Bilinear => 0u32,
            GpuColormapMode::Cubic => 1u32,
            GpuColormapMode::Lanczos3 => 2u32,
            GpuColormapMode::Nearest => 3u32,
        };
        let params = ExprParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            band_stride: slab.band_stride,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            num_bands: slab.num_bands,
            num_instrs: instrs.len() as u32,
            resample_mode,
            formula_flags: 0,
            coord_mode,
            _pad1: 0,
            rescale_min,
            inv_range_255: 255.0 / safe_range,
            inv_range_254: 254.0 / safe_range,
            saturation: 1.0,
            rgb_min: [0.0; 4],
            rgb_inv_255: [1.0; 4],
        };

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(u_slice));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v_slice));
        self.queue
            .write_buffer(&self.instr_buf, 0, bytemuck::cast_slice(instrs));
        if output == GpuPixelOutput::Rgba8 {
            self.queue
                .write_buffer(&self.lut_buf, 0, bytemuck::cast_slice(lut_rgba_u32));
        }

        let (pipeline, wg_x, wg_y) = match output {
            GpuPixelOutput::Rgba8 => (
                &self.expr_rgba8_pipeline,
                (dst_w as u32).div_ceil(16),
                (dst_h as u32).div_ceil(16),
            ),
            GpuPixelOutput::IndexedU8 => (
                &self.expr_indexed_pipeline,
                (words_per_row as u32).div_ceil(16),
                (dst_h as u32).div_ceil(16),
            ),
        };
        self.dispatch_and_readback(slab, pipeline, wg_x, wg_y, dst_w, dst_h, words_per_row, output)
    }

    /// Evaluate a bytecode expression with an arbitrary `GpuCoordPlan` directly to raw
    /// little-endian `f32` bytes (`dst_w * dst_h * 4` bytes, for `.npy`, `.tif`, `.f32`, and mosaics).
    #[allow(clippy::too_many_arguments)]
    pub fn render_expr_f32_with_plan(
        &mut self,
        slab: &GpuMultiBandSlab,
        plan: GpuCoordPlan<'_>,
        resample: GpuColormapMode,
        instrs: &[GpuInstr],
    ) -> Result<Vec<u8>, GpuError> {
        let (dst_w, dst_h, coord_mode, u_slice, v_slice): (usize, usize, u32, &[f32], &[f32]) =
            match plan {
                GpuCoordPlan::Separable1D {
                    u_coords: u,
                    v_coords: v,
                } => (u.len(), v.len(), 0, u, v),
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
        if dst_w == 0 || dst_h == 0 || instrs.is_empty() || instrs.len() > 256 {
            return Err(GpuError::InvalidDimensions(
                "invalid expr dimensions or instruction count".into(),
            ));
        }

        let out_words = dst_w * dst_h;
        self.ensure_capacities(u_slice.len(), v_slice.len(), out_words);

        let resample_mode = match resample {
            GpuColormapMode::Bilinear => 0u32,
            GpuColormapMode::Cubic => 1u32,
            GpuColormapMode::Lanczos3 => 2u32,
            GpuColormapMode::Nearest => 3u32,
        };
        let params = ExprParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            band_stride: slab.band_stride,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            num_bands: slab.num_bands,
            num_instrs: instrs.len() as u32,
            resample_mode,
            formula_flags: 0,
            coord_mode,
            _pad1: 0,
            rescale_min: 0.0,
            inv_range_255: 1.0,
            inv_range_254: 1.0,
            saturation: 1.0,
            rgb_min: [0.0; 4],
            rgb_inv_255: [1.0; 4],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(u_slice));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v_slice));
        self.queue
            .write_buffer(&self.instr_buf, 0, bytemuck::cast_slice(instrs));

        self.dispatch_and_readback(
            slab,
            &self.expr_f32_pipeline,
            (dst_w as u32).div_ceil(16),
            (dst_h as u32).div_ceil(16),
            dst_w,
            dst_h,
            dst_w,
            GpuPixelOutput::Rgba8,
        )
    }

    /// Fused 3-band RGB resample -> per-channel rescale -> `GpuColorFormula` (Gamma/Sigmoidal LUT + LCH Saturation) -> `RGBA8`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_rgb_formula(
        &mut self,
        slab: &GpuMultiBandSlab,
        u_coords: &[f32],
        v_coords: &[f32],
        resample: GpuColormapMode,
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
        formula: &GpuColorFormula,
    ) -> Result<Vec<u8>, GpuError> {
        self.render_rgb_formula_with_plan(
            slab,
            GpuCoordPlan::Separable1D {
                u_coords,
                v_coords,
            },
            resample,
            rescale_min,
            rescale_max,
            formula,
        )
    }

    /// Fused 3-band RGB resample + `GpuColorFormula` with an arbitrary `GpuCoordPlan`
    /// (`Separable1D`, `Grid2D` 17×17, or `Exact2D`).
    #[allow(clippy::too_many_arguments)]
    pub fn render_rgb_formula_with_plan(
        &mut self,
        slab: &GpuMultiBandSlab,
        plan: GpuCoordPlan<'_>,
        resample: GpuColormapMode,
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
        formula: &GpuColorFormula,
    ) -> Result<Vec<u8>, GpuError> {
        let (dst_w, dst_h, coord_mode, u_slice, v_slice): (usize, usize, u32, &[f32], &[f32]) =
            match plan {
                GpuCoordPlan::Separable1D {
                    u_coords: u,
                    v_coords: v,
                } => (u.len(), v.len(), 0, u, v),
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
        if slab.num_bands < 3 || dst_w == 0 || dst_h == 0 {
            return Err(GpuError::InvalidDimensions(
                "render_rgb_formula requires >= 3 bands and non-zero output dims".into(),
            ));
        }
        let out_words = dst_w * dst_h;
        self.ensure_capacities(u_slice.len(), v_slice.len(), out_words);

        let inv = |min: f32, max: f32| {
            let d = max - min;
            255.0 / if d.is_finite() && d.abs() > 1e-20 { d } else { 1.0 }
        };
        let mut flags = 0u32;
        if let Some(lut) = &formula.rgb_lut {
            flags |= 1;
            self.queue
                .write_buffer(&self.lut_buf, 0, bytemuck::cast_slice(lut));
        }
        let sat = if let Some(s) = formula.saturation {
            flags |= 2;
            s
        } else {
            1.0
        };
        let resample_mode = match resample {
            GpuColormapMode::Bilinear => 0u32,
            GpuColormapMode::Cubic => 1u32,
            GpuColormapMode::Lanczos3 => 2u32,
            GpuColormapMode::Nearest => 3u32,
        };
        let params = ExprParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            band_stride: slab.band_stride,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            num_bands: 3,
            num_instrs: 0,
            resample_mode,
            formula_flags: flags,
            coord_mode,
            _pad1: 0,
            rescale_min: 0.0,
            inv_range_255: 1.0,
            inv_range_254: 1.0,
            saturation: sat,
            rgb_min: [rescale_min[0], rescale_min[1], rescale_min[2], 0.0],
            rgb_inv_255: [
                inv(rescale_min[0], rescale_max[0]),
                inv(rescale_min[1], rescale_max[1]),
                inv(rescale_min[2], rescale_max[2]),
                0.0,
            ],
        };

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(u_slice));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v_slice));

        self.dispatch_and_readback(
            slab,
            &self.rgb_formula_pipeline,
            (dst_w as u32).div_ceil(16),
            (dst_h as u32).div_ceil(16),
            dst_w,
            dst_h,
            dst_w,
            GpuPixelOutput::Rgba8,
        )
    }

    /// Fused 2-band Dual-Pol SAR False-Color RGB (`[VV_dB, VH_dB, VV_dB - VH_dB]`) +
    /// `GpuColorFormula` with an arbitrary `GpuCoordPlan` (`Separable1D`, `Grid2D`, `Exact2D`).
    #[allow(clippy::too_many_arguments)]
    pub fn render_sar_rgb_formula_with_plan(
        &mut self,
        slab: &GpuMultiBandSlab,
        plan: GpuCoordPlan<'_>,
        resample: GpuColormapMode,
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
        formula: &GpuColorFormula,
    ) -> Result<Vec<u8>, GpuError> {
        let (dst_w, dst_h, coord_mode, u_slice, v_slice): (usize, usize, u32, &[f32], &[f32]) =
            match plan {
                GpuCoordPlan::Separable1D {
                    u_coords: u,
                    v_coords: v,
                } => (u.len(), v.len(), 0, u, v),
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
        if slab.num_bands < 2 || dst_w == 0 || dst_h == 0 {
            return Err(GpuError::InvalidDimensions(
                "render_sar_rgb_formula requires >= 2 bands and non-zero output dims".into(),
            ));
        }
        let out_words = dst_w * dst_h;
        self.ensure_capacities(u_slice.len(), v_slice.len(), out_words);

        let inv = |min: f32, max: f32| {
            let d = max - min;
            255.0 / if d.is_finite() && d.abs() > 1e-20 { d } else { 1.0 }
        };
        let mut flags = 0u32;
        if let Some(lut) = &formula.rgb_lut {
            flags |= 1;
            self.queue
                .write_buffer(&self.lut_buf, 0, bytemuck::cast_slice(lut));
        }
        let sat = if let Some(s) = formula.saturation {
            flags |= 2;
            s
        } else {
            1.0
        };
        let resample_mode = match resample {
            GpuColormapMode::Bilinear => 0u32,
            GpuColormapMode::Cubic => 1u32,
            GpuColormapMode::Lanczos3 => 2u32,
            GpuColormapMode::Nearest => 3u32,
        };
        let params = ExprParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            band_stride: slab.band_stride,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            num_bands: 2,
            num_instrs: 0,
            resample_mode,
            formula_flags: flags,
            coord_mode,
            _pad1: 0,
            rescale_min: 0.0,
            inv_range_255: 1.0,
            inv_range_254: 1.0,
            saturation: sat,
            rgb_min: [rescale_min[0], rescale_min[1], rescale_min[2], 0.0],
            rgb_inv_255: [
                inv(rescale_min[0], rescale_max[0]),
                inv(rescale_min[1], rescale_max[1]),
                inv(rescale_min[2], rescale_max[2]),
                0.0,
            ],
        };

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(u_slice));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v_slice));

        self.dispatch_and_readback(
            slab,
            &self.sar_rgb_formula_pipeline,
            (dst_w as u32).div_ceil(16),
            (dst_h as u32).div_ceil(16),
            dst_w,
            dst_h,
            dst_w,
            GpuPixelOutput::Rgba8,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_and_readback(
        &self,
        slab: &GpuMultiBandSlab,
        pipeline: &wgpu::ComputePipeline,
        wg_x: u32,
        wg_y: u32,
        dst_w: usize,
        dst_h: usize,
        words_per_row: usize,
        output: GpuPixelOutput,
    ) -> Result<Vec<u8>, GpuError> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-expr-bg"),
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
                    resource: self.instr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });

        let out_bytes = (words_per_row * dst_h * std::mem::size_of::<u32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-expr-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-expr-pass"),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colormap::GpuColormapRenderer;
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
    use cogzilla_kernel::{BandSource, ColorFormula, ColormapName, ExprProgram, sample_cubic};

    #[test]
    fn gpu_ndvi_expr_and_rgb_color_formula_match_cpu() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU expr test (no adapter): {e}");
                return;
            }
        };
        let mut renderer = GpuExprRenderer::new(&ctx);

        let src_w = 512u32;
        let src_h = 512u32;
        let mut b0 = vec![0.0f32; (src_w * src_h) as usize];
        let mut b1 = vec![0.0f32; (src_w * src_h) as usize];
        let mut b2 = vec![0.0f32; (src_w * src_h) as usize];
        for y in 0..src_h {
            for x in 0..src_w {
                let i = (y * src_w + x) as usize;
                if x >= 80 && x < 100 && y >= 80 && y < 100 {
                    b0[i] = f32::NAN;
                    b1[i] = f32::NAN;
                    b2[i] = f32::NAN;
                } else {
                    let fx = x as f32 * 0.02;
                    let fy = y as f32 * 0.023;
                    b0[i] = 1500.0 + 1000.0 * fx.sin().abs();
                    b1[i] = 800.0 + 600.0 * fy.cos().abs();
                    b2[i] = 400.0 + 500.0 * (fx + fy).sin().abs();
                }
            }
        }

        let make_band = |v: &[f32]| {
            let mut s = AlignedF32Slab::new_uninit(v.len());
            s.as_mut_slice().copy_from_slice(v);
            DecodedBand {
                data: BandData::F32(s),
                width: src_w,
                height: src_h,
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: false,
                    min: 0.0,
                    max: 3000.0,
                },
            }
        };
        let db0 = make_band(&b0);
        let db1 = make_band(&b1);
        let db2 = make_band(&b2);
        let blk0 = [Some(&db0)];
        let blk1 = [Some(&db1)];
        let blk2 = [Some(&db2)];
        let bs0 = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blk0);
        let bs1 = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blk1);
        let bs2 = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blk2);

        let gpu_slab = renderer
            .upload_bands(&[&b0, &b1, &b2], src_w, src_h, src_w)
            .unwrap();

        let n = 256usize;
        let u_coords: Vec<f32> = (0..n)
            .map(|i| 4.0 + (i as f32) * ((src_w as f32 - 8.0) / n as f32))
            .collect();
        let v_coords: Vec<f32> = (0..n)
            .map(|i| 4.0 + (i as f32) * ((src_h as f32 - 8.0) / n as f32))
            .collect();

        // 1. NDVI: (b1 - b2) / (b1 + b2)
        let prog = ExprProgram::compile("(b1-b2)/(b1+b2)").unwrap();
        let instrs = [
            GpuInstr::load_band(0),
            GpuInstr::load_band(1),
            GpuInstr::op(opcodes::SUB),
            GpuInstr::load_band(0),
            GpuInstr::load_band(1),
            GpuInstr::op(opcodes::ADD),
            GpuInstr::op(opcodes::DIV),
        ];
        let lut = ColormapName::Viridis.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
        let rmin = -0.2f32;
        let rmax = 0.8f32;
        let inv254 = 254.0 / (rmax - rmin);

        let gpu_ndvi = renderer
            .render_expr(
                &gpu_slab,
                &u_coords,
                &v_coords,
                GpuColormapMode::Cubic,
                &instrs,
                rmin,
                rmax,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();

        let mut mask_err = 0usize;
        let mut max_diff = 0i32;
        for (py, &v) in v_coords.iter().enumerate() {
            for (px, &u) in u_coords.iter().enumerate() {
                let s0 = sample_cubic(&bs0, u, v);
                let s1 = sample_cubic(&bs1, u, v);
                let val = prog.eval(&[s0, s1]);
                let g = gpu_ndvi[py * n + px];
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
        assert_eq!(mask_err, 0, "NDVI nodata mask mismatch");
        assert!(max_diff <= 1, "NDVI max diff {max_diff} > 1 LSB");

        // 2. 3-band RGB + Gamma 1.5 + Sigmoidal 10 0.5 + LCH Saturation 1.35
        let cpu_cf = ColorFormula::parse("Gamma RGB 1.5 Sigmoidal RGB 10 0.5 Saturation 1.35").unwrap();
        let gpu_cf = GpuColorFormula::from_gamma_sigmoidal_sat(Some(1.5), Some((10.0, 0.5)), Some(1.35));
        let rgb_min = [0.0, 0.0, 0.0];
        let rgb_max = [2500.0, 1500.0, 1000.0];

        let gpu_rgb = renderer
            .render_rgb_formula(
                &gpu_slab,
                &u_coords,
                &v_coords,
                GpuColormapMode::Cubic,
                rgb_min,
                rgb_max,
                &gpu_cf,
            )
            .unwrap();

        let mut cf_mask_err = 0usize;
        let mut max_ch_diff = 0i32;
        for (py, &v) in v_coords.iter().enumerate() {
            for (px, &u) in u_coords.iter().enumerate() {
                let sr = sample_cubic(&bs0, u, v);
                let sg = sample_cubic(&bs1, u, v);
                let sb = sample_cubic(&bs2, u, v);
                let off = (py * n + px) * 4;
                if !sr.is_finite() || !sg.is_finite() || !sb.is_finite() {
                    if gpu_rgb[off + 3] != 0 {
                        cf_mask_err += 1;
                    }
                } else if gpu_rgb[off + 3] != 255 {
                    cf_mask_err += 1;
                } else {
                    let r = ((sr - rgb_min[0]) * (255.0 / (rgb_max[0] - rgb_min[0]))).clamp(0.0, 255.0) as u8;
                    let g = ((sg - rgb_min[1]) * (255.0 / (rgb_max[1] - rgb_min[1]))).clamp(0.0, 255.0) as u8;
                    let b = ((sb - rgb_min[2]) * (255.0 / (rgb_max[2] - rgb_min[2]))).clamp(0.0, 255.0) as u8;
                    let (cr, cg, cb) = cpu_cf.apply_rgb(r, g, b);
                    max_ch_diff = max_ch_diff
                        .max((cr as i32 - gpu_rgb[off] as i32).abs())
                        .max((cg as i32 - gpu_rgb[off + 1] as i32).abs())
                        .max((cb as i32 - gpu_rgb[off + 2] as i32).abs());
                }
            }
        }
        assert_eq!(cf_mask_err, 0, "RGB color_formula nodata mask mismatch");
        assert!(
            max_ch_diff <= 2,
            "RGB color_formula max channel diff {max_ch_diff} > 2 LSB"
        );
    }

    #[test]
    fn curved_grid2d_and_exact2d_expr_and_rgb_match_cpu() {
        let Ok(ctx) = GpuContext::new_default() else {
            eprintln!("skipping curved expr test (no hardware GPU)");
            return;
        };
        let mut renderer = GpuExprRenderer::new(&ctx);

        let w = 256usize;
        let h = 256usize;
        let mut b0 = vec![0.0f32; w * h];
        let mut b1 = vec![0.0f32; w * h];
        let mut b2 = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let fx = x as f32 / w as f32;
                let fy = y as f32 / h as f32;
                b0[y * w + x] = 1200.0 + 700.0 * (fx * 9.0).sin() + 300.0 * (fy * 7.0).cos();
                b1[y * w + x] = 600.0 + 350.0 * (fx * 6.0 + 0.5).cos() + 200.0 * (fy * 10.0).sin();
                b2[y * w + x] = 450.0 + 250.0 * ((fx + fy) * 8.0).sin();
            }
        }
        let slab = renderer
            .upload_bands(&[&b0, &b1, &b2], w as u32, h as u32, w as u32)
            .unwrap();

        let mk_band = |vals: &[f32]| -> DecodedBand {
            let mut s = AlignedF32Slab::filled(w * h, 0.0);
            s.as_mut_slice().copy_from_slice(vals);
            DecodedBand {
                data: BandData::F32(s),
                width: w as u32,
                height: h as u32,
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 0.0,
                    max: 2500.0,
                },
            }
        };
        let db0 = mk_band(&b0);
        let db1 = mk_band(&b1);
        let db2 = mk_band(&b2);
        let bl0 = [Some(&db0)];
        let bl1 = [Some(&db1)];
        let bl2 = [Some(&db2)];
        let bs0 = BandSource::new(w as u32, h as u32, w as u32, h as u32, 0, 0, 1, 1, &bl0);
        let bs1 = BandSource::new(w as u32, h as u32, w as u32, h as u32, 0, 0, 1, 1, &bl1);
        let bs2 = BandSource::new(w as u32, h as u32, w as u32, h as u32, 0, 0, 1, 1, &bl2);

        // 17x17 curved UTM-like warp grid
        let n = 128usize;
        let mut grid_u = [0.0f32; 289];
        let mut grid_v = [0.0f32; 289];
        for gy in 0..17 {
            for gx in 0..17 {
                let fx = gx as f32 / 16.0;
                let fy = gy as f32 / 16.0;
                grid_u[gy * 17 + gx] = 10.0 + fx * 230.0 + 4.0 * (fy * 3.14).sin();
                grid_v[gy * 17 + gx] = 10.0 + fy * 230.0 - 4.0 * (fx * 3.14).sin();
            }
        }
        let eval_grid = |px: usize, py: usize| -> (f32, f32) {
            let gxf = (px as f32 + 0.5) * 16.0 / n as f32;
            let gyf = (py as f32 + 0.5) * 16.0 / n as f32;
            let gx0 = (gxf as usize).min(15);
            let gy0 = (gyf as usize).min(15);
            let fx = gxf - gx0 as f32;
            let fy = gyf - gy0 as f32;
            let i00 = gy0 * 17 + gx0;
            let u0 = grid_u[i00] + (grid_u[i00 + 1] - grid_u[i00]) * fx;
            let u1 = grid_u[i00 + 17] + (grid_u[i00 + 18] - grid_u[i00 + 17]) * fx;
            let v0 = grid_v[i00] + (grid_v[i00 + 1] - grid_v[i00]) * fx;
            let v1 = grid_v[i00 + 17] + (grid_v[i00 + 18] - grid_v[i00 + 17]) * fx;
            (u0 + (u1 - u0) * fy, v0 + (v1 - v0) * fy)
        };

        // NDVI Grid2D IndexedU8 + Rgba8
        let ndvi = vec![
            GpuInstr::load_band(0),
            GpuInstr::load_band(1),
            GpuInstr::op(opcodes::SUB),
            GpuInstr::load_band(0),
            GpuInstr::load_band(1),
            GpuInstr::op(opcodes::ADD),
            GpuInstr::op(opcodes::DIV),
        ];
        let lut = ColormapName::Viridis.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);
        let gpu_rgba = renderer
            .render_expr_with_plan(
                &slab,
                GpuCoordPlan::Grid2D {
                    dst_w: n,
                    dst_h: n,
                    grid_u: &grid_u,
                    grid_v: &grid_v,
                },
                GpuColormapMode::Cubic,
                &ndvi,
                -0.2,
                0.8,
                &lut_u32,
                GpuPixelOutput::Rgba8,
            )
            .unwrap();
        let mut max_expr_diff = 0i32;
        for py in (0..n).step_by(3) {
            for px in (0..n).step_by(3) {
                let (u, v) = eval_grid(px, py);
                let s0 = sample_cubic(&bs0, u, v);
                let s1 = sample_cubic(&bs1, u, v);
                let nd = (s0 - s1) / (s0 + s1);
                let ci = ((nd + 0.2) * 255.0).clamp(0.0, 255.0) as usize;
                let off = (py * n + px) * 4;
                assert_eq!(gpu_rgba[off + 3], 255);
                for ch in 0..3 {
                    max_expr_diff =
                        max_expr_diff.max((lut.rgb[ci][ch] as i32 - gpu_rgba[off + ch] as i32).abs());
                }
            }
        }
        assert!(max_expr_diff <= 3, "Grid2D NDVI max channel diff {max_expr_diff} > 3");

        // Exact2D with NaN mask on 3-band RGB
        let mut u_exact = vec![f32::NAN; n * n];
        let mut v_exact = vec![f32::NAN; n * n];
        for py in 0..n {
            for px in 0..n {
                if px >= 16 && px < n - 16 && py >= 16 && py < n - 16 {
                    let (u, v) = eval_grid(px, py);
                    u_exact[py * n + px] = u;
                    v_exact[py * n + px] = v;
                }
            }
        }
        let identity = GpuColorFormula {
            rgb_lut: None,
            saturation: None,
        };
        let gpu_rgb = renderer
            .render_rgb_formula_with_plan(
                &slab,
                GpuCoordPlan::Exact2D {
                    dst_w: n,
                    dst_h: n,
                    u: &u_exact,
                    v: &v_exact,
                },
                GpuColormapMode::Cubic,
                [0.0; 3],
                [2500.0, 1500.0, 1000.0],
                &identity,
            )
            .unwrap();
        let mut max_rgb_diff = 0i32;
        for py in (0..n).step_by(3) {
            for px in (0..n).step_by(3) {
                let i = py * n + px;
                let off = i * 4;
                if u_exact[i].is_nan() {
                    assert_eq!(gpu_rgb[off + 3], 0);
                } else {
                    assert_eq!(gpu_rgb[off + 3], 255);
                    let sr = sample_cubic(&bs0, u_exact[i], v_exact[i]);
                    let sg = sample_cubic(&bs1, u_exact[i], v_exact[i]);
                    let sb = sample_cubic(&bs2, u_exact[i], v_exact[i]);
                    let r = (sr * (255.0 / 2500.0)).clamp(0.0, 255.0) as i32;
                    let g = (sg * (255.0 / 1500.0)).clamp(0.0, 255.0) as i32;
                    let b = (sb * (255.0 / 1000.0)).clamp(0.0, 255.0) as i32;
                    max_rgb_diff = max_rgb_diff
                        .max((r - gpu_rgb[off] as i32).abs())
                        .max((g - gpu_rgb[off + 1] as i32).abs())
                        .max((b - gpu_rgb[off + 2] as i32).abs());
                }
            }
        }
        assert!(max_rgb_diff <= 2, "Exact2D RGB max channel diff {max_rgb_diff} > 2");
    }
}
