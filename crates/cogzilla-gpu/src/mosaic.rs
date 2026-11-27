use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::colormap::GpuPixelOutput;
use crate::context::{GpuContext, GpuError};

pub use cogzilla_raster::GpuPixelSelection;

/// GPU-resident stack of `1..=32` aligned scene planes (`width * height` `f32` each).
/// For 3-band RGB stacks created by [`GpuMosaicCompositor::upload_rgb_scenes`], the buffer holds
/// `3 * num_scenes` planes laid out band-major (`R`, then `G`, then `B`).
pub struct GpuSceneStack {
    pub(crate) buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub plane_stride: u32,
    pub num_scenes: u32,
}

/// Scene upload mode for [`GpuMosaicCompositor::composite`].
pub enum MosaicUpload<'a> {
    /// Already-resident single-band scene stack.
    Staged(&'a GpuSceneStack),
    /// Single-band scene planes uploaded directly into the persistent `scene_buf`.
    Direct {
        planes: &'a [&'a [f32]],
        width: u32,
        height: u32,
    },
    /// Already-resident per-band scene stacks (for [`MosaicOutput::F32Bands`]).
    StagedBands(&'a [&'a GpuSceneStack]),
    /// Multi-band scene planes uploaded directly into the persistent `scene_buf`
    /// (for [`MosaicOutput::F32Bands`]).
    DirectBands {
        bands: &'a [Vec<&'a [f32]>],
        width: u32,
        height: u32,
    },
    /// Already-resident 3-band RGB scene stack from [`GpuMosaicCompositor::upload_rgb_scenes`].
    StagedRgb(&'a GpuSceneStack),
    /// 3-band RGB scene planes uploaded directly into the persistent `scene_buf`.
    DirectRgb {
        planes: &'a [&'a [&'a [f32]]; 3],
        width: u32,
        height: u32,
    },
}

/// Desired output shape for [`GpuMosaicCompositor::composite`].
pub enum MosaicOutput<'a> {
    /// Single-band raw `f32` composite (`width * height`).
    F32,
    /// Multi-band raw `f32` composites in a single batched submission.
    F32Bands,
    /// Single-band composite fused with rescale + colormap into `Rgba8` or `IndexedU8`.
    Styled {
        lut: &'a [u32; 256],
        out: GpuPixelOutput,
        rescale: (f32, f32),
    },
    /// Single-band composite fused with rescale + colormap -> 1-pass GPU 4:2:0 JPEG.
    StyledJpeg {
        lut: &'a [u32; 256],
        jpeg: &'a mut crate::GpuJpegEncoder,
        quality: u8,
        rescale: (f32, f32),
    },
    /// 3-band RGB composite fused with per-channel rescale -> RGBA8.
    RgbStyled {
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
    },
    /// 3-band RGB composite fused with per-channel rescale -> 1-pass GPU 4:2:0 JPEG.
    RgbStyledJpeg {
        jpeg: &'a mut crate::GpuJpegEncoder,
        quality: u8,
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
    },
}

/// Output produced by [`GpuMosaicCompositor::composite`].
#[derive(Debug)]
pub enum MosaicCompositeResult {
    F32(Vec<f32>),
    F32Bands(Vec<Vec<f32>>),
    Bytes(Vec<u8>),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MosaicParams {
    width: u32,
    height: u32,
    plane_stride: u32,
    num_scenes: u32,
    method: u32,
    band_stride: u32,
    _pad1: u32,
    _pad2: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    _pad3: f32,
    rescale_min_g: f32,
    inv_range_255_g: f32,
    rescale_min_b: f32,
    inv_range_255_b: f32,
}

impl MosaicParams {
    fn raw(
        width: u32,
        height: u32,
        plane_stride: u32,
        num_scenes: u32,
        method: GpuPixelSelection,
    ) -> Self {
        Self {
            width,
            height,
            plane_stride,
            num_scenes,
            method: method.as_u32(),
            band_stride: 0,
            _pad1: 0,
            _pad2: 0,
            rescale_min: 0.0,
            inv_range_255: 1.0,
            inv_range_254: 1.0,
            _pad3: 0.0,
            rescale_min_g: 0.0,
            inv_range_255_g: 1.0,
            rescale_min_b: 0.0,
            inv_range_255_b: 1.0,
        }
    }

    fn styled(
        width: u32,
        height: u32,
        plane_stride: u32,
        num_scenes: u32,
        method: GpuPixelSelection,
        rescale_min: f32,
        rescale_max: f32,
    ) -> Self {
        let (inv_range_255, inv_range_254) =
            crate::wgpu_util::safe_inv_ranges(rescale_min, rescale_max);
        Self {
            rescale_min,
            inv_range_255,
            inv_range_254,
            ..Self::raw(width, height, plane_stride, num_scenes, method)
        }
    }

    fn rgb(
        width: u32,
        height: u32,
        plane_stride: u32,
        num_scenes: u32,
        method: GpuPixelSelection,
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
    ) -> Self {
        let safe_inv = |lo: f32, hi: f32| 255.0 / crate::wgpu_util::safe_range(lo, hi);
        Self {
            band_stride: plane_stride * num_scenes,
            rescale_min: rescale_min[0],
            inv_range_255: safe_inv(rescale_min[0], rescale_max[0]),
            rescale_min_g: rescale_min[1],
            inv_range_255_g: safe_inv(rescale_min[1], rescale_max[1]),
            rescale_min_b: rescale_min[2],
            inv_range_255_b: safe_inv(rescale_min[2], rescale_max[2]),
            ..Self::raw(width, height, plane_stride, num_scenes, method)
        }
    }
}

/// GPU multi-scene mosaic compositor and pixel-selection reducer
/// (`First`, `Highest`, `Lowest`, `Mean`, `Median`, `Stdev`, `Count`) with fused colormap/RGB output.
pub struct GpuMosaicCompositor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    f32_pipeline: wgpu::ComputePipeline,
    rgba8_pipeline: wgpu::ComputePipeline,
    indexed_pipeline: wgpu::ComputePipeline,
    rgb_rgba8_pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    lut_buf: wgpu::Buffer,
    scene_buf: wgpu::Buffer,
    scene_bytes_cap: usize,
    max_scene_bytes: u64,
    min_storage_align: u64,
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
            source: wgpu::ShaderSource::Wgsl(crate::shaders::MOSAIC_WGSL.into()),
        });

        use crate::wgpu_util::{bgl_storage_ro, bgl_storage_rw, bgl_uniform};
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-mosaic-bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_rw(3),
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
        let rgb_rgba8_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-mosaic-rgb-rgba8"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("mosaic_select_rgb_rgba8"),
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
        let init_scene_bytes = 512 * 512 * 4 * 4;
        let scene_buf = Self::alloc_scene_buf(&device, init_scene_bytes);
        let limits = device.limits();
        let min_storage_align = limits.min_storage_buffer_offset_alignment.max(256) as u64;
        let max_scene_bytes = (limits.max_storage_buffer_binding_size as u64)
            .min(limits.max_buffer_size);

        Self {
            device,
            queue,
            bind_group_layout,
            f32_pipeline,
            rgba8_pipeline,
            indexed_pipeline,
            rgb_rgba8_pipeline,
            params_buf,
            lut_buf,
            scene_buf,
            scene_bytes_cap: init_scene_bytes,
            max_scene_bytes,
            min_storage_align,
            out_buf,
            staging_buf,
            out_words_cap: init_words,
        }
    }

    fn alloc_scene_buf(device: &wgpu::Device, bytes: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-mosaic-scenes"),
            size: bytes.max(256) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn ensure_scene_cap(&mut self, bytes: usize) {
        if bytes > self.scene_bytes_cap {
            self.scene_bytes_cap = bytes;
            self.scene_buf = Self::alloc_scene_buf(&self.device, self.scene_bytes_cap);
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

    fn validate_scene_planes(
        &self,
        planes: &[&[f32]],
        width: u32,
        height: u32,
    ) -> Result<(usize, usize, usize), GpuError> {
        if planes.is_empty() || planes.len() > 32 || width == 0 || height == 0 {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid scene stack: scenes={}, {width}x{height}",
                planes.len()
            )));
        }
        let plane_stride = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("plane_stride overflow".into()))?;
        for (i, p) in planes.iter().enumerate() {
            if p.len() < plane_stride {
                return Err(GpuError::InvalidDimensions(format!(
                    "scene {i} len {} < {plane_stride}",
                    p.len()
                )));
            }
        }
        let plane_bytes = plane_stride
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| GpuError::InvalidDimensions("plane_bytes overflow".into()))?;
        let total_bytes = plane_bytes
            .checked_mul(planes.len())
            .ok_or_else(|| GpuError::InvalidDimensions("total_bytes overflow".into()))?;
        if (total_bytes as u64) > self.max_scene_bytes {
            return Err(GpuError::InvalidDimensions(format!(
                "scene stack {total_bytes} B exceeds adapter limit {} B",
                self.max_scene_bytes
            )));
        }
        Ok((plane_stride, plane_bytes, total_bytes))
    }

    /// Upload `1..=32` scene planes (`width * height` `f32` each) into a contiguous GPU stack
    /// using zero-copy `mapped_at_creation` (parallelized across scenes for large stacks).
    pub fn upload_scenes(
        &self,
        planes: &[&[f32]],
        width: u32,
        height: u32,
    ) -> Result<GpuSceneStack, GpuError> {
        let (plane_stride, plane_bytes, total_bytes) =
            self.validate_scene_planes(planes, width, height)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-scene-stack"),
            size: total_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        {
            let mut mapped = buffer.slice(..).get_mapped_range_mut();
            if planes.len() >= 2 && total_bytes >= 2 * 1024 * 1024 {
                let n_threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
                    .min(planes.len());
                let scenes_per_chunk = planes.len().div_ceil(n_threads);
                let chunk_bytes = scenes_per_chunk * plane_bytes;
                std::thread::scope(|scope| {
                    for (chunk_idx, dst_chunk) in mapped.chunks_mut(chunk_bytes).enumerate() {
                        let start_scene = chunk_idx * scenes_per_chunk;
                        let end_scene = (start_scene + scenes_per_chunk).min(planes.len());
                        let src_planes = &planes[start_scene..end_scene];
                        scope.spawn(move || {
                            for (local_i, p) in src_planes.iter().enumerate() {
                                let dst =
                                    &mut dst_chunk[local_i * plane_bytes..(local_i + 1) * plane_bytes];
                                dst.copy_from_slice(bytemuck::cast_slice(&p[..plane_stride]));
                            }
                        });
                    }
                });
            } else {
                for (i, p) in planes.iter().enumerate() {
                    let dst = &mut mapped[i * plane_bytes..(i + 1) * plane_bytes];
                    dst.copy_from_slice(bytemuck::cast_slice(&p[..plane_stride]));
                }
            }
        }
        buffer.unmap();
        Ok(GpuSceneStack {
            buffer,
            width,
            height,
            plane_stride: plane_stride as u32,
            num_scenes: planes.len() as u32,
        })
    }

    /// Upload 3 bands × `1..=32` scene planes (`width * height` `f32` each) into a contiguous
    /// GPU stack laid out band-major (`R`, `G`, `B`) for [`MosaicUpload::StagedRgb`].
    pub fn upload_rgb_scenes(
        &self,
        rgb_planes: &[&[&[f32]]; 3],
        width: u32,
        height: u32,
    ) -> Result<GpuSceneStack, GpuError> {
        let (plane_stride, plane_bytes, band_bytes, total_bytes, num_scenes) =
            self.validate_rgb_scene_planes(rgb_planes, width, height)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-rgb-scene-stack"),
            size: total_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        {
            let mut mapped = buffer.slice(..).get_mapped_range_mut();
            for (c, planes) in rgb_planes.iter().enumerate() {
                let band_base = c * band_bytes;
                for (s, p) in planes.iter().enumerate() {
                    let off = band_base + s * plane_bytes;
                    mapped[off..off + plane_bytes]
                        .copy_from_slice(bytemuck::cast_slice(&p[..plane_stride]));
                }
            }
        }
        buffer.unmap();
        Ok(GpuSceneStack {
            buffer,
            width,
            height,
            plane_stride: plane_stride as u32,
            num_scenes: num_scenes as u32,
        })
    }

    /// Unified mosaic composite entry point over all upload modes (`Staged` / `Direct`) and
    /// output shapes (`F32`, `F32Bands`, `Styled`, `StyledJpeg`, `RgbStyled`, `RgbStyledJpeg`).
    pub fn composite(
        &mut self,
        upload: MosaicUpload<'_>,
        selection: GpuPixelSelection,
        output: MosaicOutput<'_>,
    ) -> Result<MosaicCompositeResult, GpuError> {
        match output {
            MosaicOutput::F32 => {
                let (width, height, plane_stride, num_scenes, direct_bytes) =
                    self.resolve_single_upload(&upload)?;
                let pixels = plane_stride as usize;
                self.ensure_out_cap(pixels);
                let params = MosaicParams::raw(width, height, plane_stride, num_scenes, selection);
                self.queue
                    .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
                let binding = self.single_binding(&upload, direct_bytes);
                let raw = self.run_pipeline_binding(
                    binding,
                    &self.f32_pipeline,
                    width.div_ceil(16),
                    height.div_ceil(16),
                    pixels * 4,
                )?;
                Ok(MosaicCompositeResult::F32(bytemuck::cast_slice(&raw).to_vec()))
            }
            MosaicOutput::F32Bands => self.composite_bands_impl(upload, selection),
            MosaicOutput::Styled { lut, out, rescale } => {
                let (width, height, plane_stride, num_scenes, direct_bytes) =
                    self.resolve_single_upload(&upload)?;
                self.prepare_styled(
                    width,
                    height,
                    plane_stride,
                    num_scenes,
                    selection,
                    rescale.0,
                    rescale.1,
                    lut,
                    out,
                );
                let binding = self.single_binding(&upload, direct_bytes);
                let bytes = self.run_styled_binding(binding, width, height, out)?;
                Ok(MosaicCompositeResult::Bytes(bytes))
            }
            MosaicOutput::StyledJpeg {
                lut,
                jpeg,
                quality,
                rescale,
            } => {
                let (width, height, plane_stride, num_scenes, direct_bytes) =
                    self.resolve_single_upload(&upload)?;
                self.ensure_out_cap(plane_stride as usize);
                let params = MosaicParams::styled(
                    width,
                    height,
                    plane_stride,
                    num_scenes,
                    selection,
                    rescale.0,
                    rescale.1,
                );
                self.queue
                    .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
                self.queue
                    .write_buffer(&self.lut_buf, 0, bytemuck::cast_slice(lut));
                let binding = self.single_binding(&upload, direct_bytes);
                let bytes = self.run_jpeg_binding(
                    binding,
                    &self.rgba8_pipeline,
                    width,
                    height,
                    jpeg,
                    quality,
                )?;
                Ok(MosaicCompositeResult::Bytes(bytes))
            }
            MosaicOutput::RgbStyled {
                rescale_min,
                rescale_max,
            } => {
                let (width, height, plane_stride, num_scenes, direct_bytes) =
                    self.resolve_rgb_upload(&upload)?;
                let pixels = plane_stride as usize;
                self.ensure_out_cap(pixels);
                let params = MosaicParams::rgb(
                    width,
                    height,
                    plane_stride,
                    num_scenes,
                    selection,
                    rescale_min,
                    rescale_max,
                );
                self.queue
                    .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
                let binding = self.rgb_binding(&upload, direct_bytes);
                let bytes = self.run_pipeline_binding(
                    binding,
                    &self.rgb_rgba8_pipeline,
                    width.div_ceil(16),
                    height.div_ceil(16),
                    pixels * 4,
                )?;
                Ok(MosaicCompositeResult::Bytes(bytes))
            }
            MosaicOutput::RgbStyledJpeg {
                jpeg,
                quality,
                rescale_min,
                rescale_max,
            } => {
                let (width, height, plane_stride, num_scenes, direct_bytes) =
                    self.resolve_rgb_upload(&upload)?;
                self.ensure_out_cap(plane_stride as usize);
                let params = MosaicParams::rgb(
                    width,
                    height,
                    plane_stride,
                    num_scenes,
                    selection,
                    rescale_min,
                    rescale_max,
                );
                self.queue
                    .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
                let binding = self.rgb_binding(&upload, direct_bytes);
                let bytes = self.run_jpeg_binding(
                    binding,
                    &self.rgb_rgba8_pipeline,
                    width,
                    height,
                    jpeg,
                    quality,
                )?;
                Ok(MosaicCompositeResult::Bytes(bytes))
            }
        }
    }

    fn resolve_single_upload(
        &mut self,
        upload: &MosaicUpload<'_>,
    ) -> Result<(u32, u32, u32, u32, Option<u64>), GpuError> {
        match upload {
            MosaicUpload::Staged(stack) => Ok((
                stack.width,
                stack.height,
                stack.plane_stride,
                stack.num_scenes,
                None,
            )),
            MosaicUpload::Direct {
                planes,
                width,
                height,
            } => {
                let (plane_stride, stack_bytes) =
                    self.write_scene_planes(planes, *width, *height)?;
                Ok((
                    *width,
                    *height,
                    plane_stride as u32,
                    planes.len() as u32,
                    Some(stack_bytes),
                ))
            }
            _ => Err(GpuError::InvalidDimensions(
                "single-band mosaic output requires MosaicUpload::Staged or MosaicUpload::Direct"
                    .into(),
            )),
        }
    }

    fn single_binding<'a>(
        &'a self,
        upload: &'a MosaicUpload<'a>,
        direct_bytes: Option<u64>,
    ) -> wgpu::BufferBinding<'a> {
        match upload {
            MosaicUpload::Staged(stack) => stack.buffer.as_entire_buffer_binding(),
            MosaicUpload::Direct { .. } => wgpu::BufferBinding {
                buffer: &self.scene_buf,
                offset: 0,
                size: direct_bytes.and_then(std::num::NonZeroU64::new),
            },
            _ => unreachable!("resolve_single_upload rejected non-single upload"),
        }
    }

    fn resolve_rgb_upload(
        &mut self,
        upload: &MosaicUpload<'_>,
    ) -> Result<(u32, u32, u32, u32, Option<u64>), GpuError> {
        match upload {
            MosaicUpload::StagedRgb(stack) => Ok((
                stack.width,
                stack.height,
                stack.plane_stride,
                stack.num_scenes,
                None,
            )),
            MosaicUpload::DirectRgb {
                planes,
                width,
                height,
            } => {
                let (plane_stride, num_scenes, total_bytes) =
                    self.write_rgb_scene_planes(planes, *width, *height)?;
                Ok((
                    *width,
                    *height,
                    plane_stride as u32,
                    num_scenes as u32,
                    Some(total_bytes),
                ))
            }
            _ => Err(GpuError::InvalidDimensions(
                "RGB mosaic output requires MosaicUpload::StagedRgb or MosaicUpload::DirectRgb"
                    .into(),
            )),
        }
    }

    fn rgb_binding<'a>(
        &'a self,
        upload: &'a MosaicUpload<'a>,
        direct_bytes: Option<u64>,
    ) -> wgpu::BufferBinding<'a> {
        match upload {
            MosaicUpload::StagedRgb(stack) => stack.buffer.as_entire_buffer_binding(),
            MosaicUpload::DirectRgb { .. } => wgpu::BufferBinding {
                buffer: &self.scene_buf,
                offset: 0,
                size: direct_bytes.and_then(std::num::NonZeroU64::new),
            },
            _ => unreachable!("resolve_rgb_upload rejected non-RGB upload"),
        }
    }

    fn composite_bands_impl(
        &mut self,
        upload: MosaicUpload<'_>,
        selection: GpuPixelSelection,
    ) -> Result<MosaicCompositeResult, GpuError> {
        match upload {
            MosaicUpload::StagedBands(stacks) => {
                if stacks.is_empty() {
                    return Ok(MosaicCompositeResult::F32Bands(Vec::new()));
                }
                if stacks.len() == 1 {
                    let MosaicCompositeResult::F32(single) = self.composite(
                        MosaicUpload::Staged(stacks[0]),
                        selection,
                        MosaicOutput::F32,
                    )?
                    else {
                        unreachable!();
                    };
                    return Ok(MosaicCompositeResult::F32Bands(vec![single]));
                }
                let first = stacks[0];
                for (idx, s) in stacks.iter().enumerate().skip(1) {
                    if s.width != first.width
                        || s.height != first.height
                        || s.plane_stride != first.plane_stride
                        || s.num_scenes != first.num_scenes
                    {
                        return Err(GpuError::InvalidDimensions(format!(
                            "band {idx} stack mismatch: {}x{}x{} vs {}x{}x{}",
                            s.width,
                            s.height,
                            s.num_scenes,
                            first.width,
                            first.height,
                            first.num_scenes
                        )));
                    }
                }
                let pixels = (first.width as usize) * (first.height as usize);
                let total_words = pixels.checked_mul(stacks.len()).ok_or_else(|| {
                    GpuError::InvalidDimensions("batched mosaic words overflow".into())
                })?;
                self.ensure_out_cap(total_words);
                let params = MosaicParams::raw(
                    first.width,
                    first.height,
                    first.plane_stride,
                    first.num_scenes,
                    selection,
                );
                self.queue
                    .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
                let bindings: Vec<wgpu::BufferBinding<'_>> = stacks
                    .iter()
                    .map(|s| s.buffer.as_entire_buffer_binding())
                    .collect();
                let bands =
                    self.run_f32_bands_bindings(&bindings, first.width, first.height, pixels)?;
                Ok(MosaicCompositeResult::F32Bands(bands))
            }
            MosaicUpload::DirectBands {
                bands: band_planes,
                width,
                height,
            } => {
                if band_planes.is_empty() {
                    return Ok(MosaicCompositeResult::F32Bands(Vec::new()));
                }
                if band_planes.len() == 1 {
                    let MosaicCompositeResult::F32(single) = self.composite(
                        MosaicUpload::Direct {
                            planes: &band_planes[0],
                            width,
                            height,
                        },
                        selection,
                        MosaicOutput::F32,
                    )?
                    else {
                        unreachable!();
                    };
                    return Ok(MosaicCompositeResult::F32Bands(vec![single]));
                }
                let num_scenes = band_planes[0].len();
                if num_scenes == 0 || num_scenes > 32 || width == 0 || height == 0 {
                    return Err(GpuError::InvalidDimensions(format!(
                        "invalid batched scene stack: scenes={num_scenes}, {width}x{height}"
                    )));
                }
                let plane_stride = (width as usize)
                    .checked_mul(height as usize)
                    .ok_or_else(|| GpuError::InvalidDimensions("plane_stride overflow".into()))?;
                for (b, planes) in band_planes.iter().enumerate() {
                    if planes.len() != num_scenes {
                        return Err(GpuError::InvalidDimensions(format!(
                            "band {b} scene count {} != {num_scenes}",
                            planes.len()
                        )));
                    }
                    for (s, p) in planes.iter().enumerate() {
                        if p.len() < plane_stride {
                            return Err(GpuError::InvalidDimensions(format!(
                                "band {b} scene {s} len {} < {plane_stride}",
                                p.len()
                            )));
                        }
                    }
                }
                let plane_bytes = plane_stride
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| GpuError::InvalidDimensions("plane_bytes overflow".into()))?;
                let raw_band_bytes = plane_bytes
                    .checked_mul(num_scenes)
                    .ok_or_else(|| GpuError::InvalidDimensions("raw_band_bytes overflow".into()))?
                    as u64;
                let align = self.min_storage_align;
                let aligned_band_stride = raw_band_bytes.div_ceil(align) * align;
                let total_scene_bytes = (aligned_band_stride as usize)
                    .checked_mul(band_planes.len())
                    .ok_or_else(|| {
                        GpuError::InvalidDimensions("batched scene bytes overflow".into())
                    })?;
                if (total_scene_bytes as u64) > self.max_scene_bytes {
                    return Err(GpuError::InvalidDimensions(format!(
                        "batched scene stack {total_scene_bytes} B exceeds adapter limit {} B",
                        self.max_scene_bytes
                    )));
                }
                self.ensure_scene_cap(total_scene_bytes);

                for (b, planes) in band_planes.iter().enumerate() {
                    let band_base = (b as u64) * aligned_band_stride;
                    for (s, p) in planes.iter().enumerate() {
                        self.queue.write_buffer(
                            &self.scene_buf,
                            band_base + (s * plane_bytes) as u64,
                            bytemuck::cast_slice(&p[..plane_stride]),
                        );
                    }
                }

                let total_out_words = plane_stride
                    .checked_mul(band_planes.len())
                    .ok_or_else(|| {
                        GpuError::InvalidDimensions("batched out words overflow".into())
                    })?;
                self.ensure_out_cap(total_out_words);

                let params = MosaicParams::raw(
                    width,
                    height,
                    plane_stride as u32,
                    num_scenes as u32,
                    selection,
                );
                self.queue
                    .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

                let bindings: Vec<wgpu::BufferBinding<'_>> = (0..band_planes.len())
                    .map(|b| wgpu::BufferBinding {
                        buffer: &self.scene_buf,
                        offset: (b as u64) * aligned_band_stride,
                        size: std::num::NonZeroU64::new(raw_band_bytes),
                    })
                    .collect();
                let bands =
                    self.run_f32_bands_bindings(&bindings, width, height, plane_stride)?;
                Ok(MosaicCompositeResult::F32Bands(bands))
            }
            _ => Err(GpuError::InvalidDimensions(
                "F32Bands mosaic output requires MosaicUpload::StagedBands or MosaicUpload::DirectBands"
                    .into(),
            )),
        }
    }

    fn write_scene_planes(
        &mut self,
        planes: &[&[f32]],
        width: u32,
        height: u32,
    ) -> Result<(usize, u64), GpuError> {
        let (plane_stride, plane_bytes, total_bytes) =
            self.validate_scene_planes(planes, width, height)?;
        self.ensure_scene_cap(total_bytes);
        for (i, p) in planes.iter().enumerate() {
            self.queue.write_buffer(
                &self.scene_buf,
                (i * plane_bytes) as u64,
                bytemuck::cast_slice(&p[..plane_stride]),
            );
        }
        Ok((plane_stride, total_bytes as u64))
    }

    fn validate_rgb_scene_planes(
        &self,
        rgb_planes: &[&[&[f32]]; 3],
        width: u32,
        height: u32,
    ) -> Result<(usize, usize, usize, usize, usize), GpuError> {
        let num_scenes = rgb_planes[0].len();
        if num_scenes == 0
            || num_scenes > 32
            || width == 0
            || height == 0
            || rgb_planes[1].len() != num_scenes
            || rgb_planes[2].len() != num_scenes
        {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid RGB scene stack: scenes=[{},{},{}], {width}x{height}",
                rgb_planes[0].len(),
                rgb_planes[1].len(),
                rgb_planes[2].len()
            )));
        }
        let plane_stride = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| GpuError::InvalidDimensions("plane_stride overflow".into()))?;
        for (c, planes) in rgb_planes.iter().enumerate() {
            for (s, p) in planes.iter().enumerate() {
                if p.len() < plane_stride {
                    return Err(GpuError::InvalidDimensions(format!(
                        "RGB band {c} scene {s} len {} < {plane_stride}",
                        p.len()
                    )));
                }
            }
        }
        let plane_bytes = plane_stride
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| GpuError::InvalidDimensions("plane_bytes overflow".into()))?;
        let band_bytes = plane_bytes
            .checked_mul(num_scenes)
            .ok_or_else(|| GpuError::InvalidDimensions("band_bytes overflow".into()))?;
        let total_bytes = band_bytes
            .checked_mul(3)
            .ok_or_else(|| GpuError::InvalidDimensions("total_bytes overflow".into()))?;
        if (total_bytes as u64) > self.max_scene_bytes {
            return Err(GpuError::InvalidDimensions(format!(
                "RGB scene stack {total_bytes} B exceeds adapter limit {} B",
                self.max_scene_bytes
            )));
        }
        Ok((plane_stride, plane_bytes, band_bytes, total_bytes, num_scenes))
    }

    fn write_rgb_scene_planes(
        &mut self,
        rgb_planes: &[&[&[f32]]; 3],
        width: u32,
        height: u32,
    ) -> Result<(usize, usize, u64), GpuError> {
        let (plane_stride, plane_bytes, band_bytes, total_bytes, num_scenes) =
            self.validate_rgb_scene_planes(rgb_planes, width, height)?;
        self.ensure_scene_cap(total_bytes);
        for (c, planes) in rgb_planes.iter().enumerate() {
            let band_base = (c * band_bytes) as u64;
            for (s, p) in planes.iter().enumerate() {
                self.queue.write_buffer(
                    &self.scene_buf,
                    band_base + (s * plane_bytes) as u64,
                    bytemuck::cast_slice(&p[..plane_stride]),
                );
            }
        }
        Ok((plane_stride, num_scenes, total_bytes as u64))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_styled(
        &mut self,
        width: u32,
        height: u32,
        plane_stride: u32,
        num_scenes: u32,
        method: GpuPixelSelection,
        rescale_min: f32,
        rescale_max: f32,
        lut_rgba_u32: &[u32; 256],
        output: GpuPixelOutput,
    ) {
        let w = width as usize;
        let h = height as usize;
        let words_per_row = match output {
            GpuPixelOutput::Rgba8 => w,
            GpuPixelOutput::IndexedU8 => w.div_ceil(4),
        };
        self.ensure_out_cap(words_per_row * h);
        let params = MosaicParams::styled(
            width,
            height,
            plane_stride,
            num_scenes,
            method,
            rescale_min,
            rescale_max,
        );
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        if output == GpuPixelOutput::Rgba8 {
            self.queue
                .write_buffer(&self.lut_buf, 0, bytemuck::cast_slice(lut_rgba_u32));
        }
    }

    fn run_styled_binding(
        &self,
        scene_binding: wgpu::BufferBinding<'_>,
        width: u32,
        height: u32,
        output: GpuPixelOutput,
    ) -> Result<Vec<u8>, GpuError> {
        let w = width as usize;
        let h = height as usize;
        let words_per_row = match output {
            GpuPixelOutput::Rgba8 => w,
            GpuPixelOutput::IndexedU8 => w.div_ceil(4),
        };
        let (pipeline, wg_x, wg_y) = match output {
            GpuPixelOutput::Rgba8 => (&self.rgba8_pipeline, width.div_ceil(16), height.div_ceil(16)),
            GpuPixelOutput::IndexedU8 => (
                &self.indexed_pipeline,
                (words_per_row as u32).div_ceil(16),
                height.div_ceil(16),
            ),
        };
        let raw = self.run_pipeline_binding(
            scene_binding,
            pipeline,
            wg_x,
            wg_y,
            words_per_row * h * 4,
        )?;
        match output {
            GpuPixelOutput::Rgba8 => Ok(raw),
            GpuPixelOutput::IndexedU8 if w.is_multiple_of(4) => Ok(raw),
            GpuPixelOutput::IndexedU8 => Ok(crate::wgpu_util::unpack_indexed_u8(
                &raw,
                w,
                h,
                words_per_row,
            )),
        }
    }

    fn make_bind_group(&self, scene_binding: wgpu::BufferBinding<'_>) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-mosaic-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(scene_binding),
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
        })
    }

    fn run_f32_bands_bindings(
        &self,
        bindings: &[wgpu::BufferBinding<'_>],
        width: u32,
        height: u32,
        pixels: usize,
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        let wg_x = width.div_ceil(16);
        let wg_y = height.div_ceil(16);
        let band_bytes = (pixels * 4) as u64;
        let total_bytes = band_bytes * (bindings.len() as u64);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-mosaic-bands-enc"),
            });
        for (b, binding) in bindings.iter().enumerate() {
            let bind_group = self.make_bind_group(binding.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cogzilla-mosaic-band-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.f32_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            encoder.copy_buffer_to_buffer(
                &self.out_buf,
                0,
                &self.staging_buf,
                (b as u64) * band_bytes,
                band_bytes,
            );
        }
        self.queue.submit(Some(encoder.finish()));

        crate::wgpu_util::readback_staging_sync(
            &self.device,
            &self.staging_buf,
            total_bytes,
            |mapped| {
                let all_f32: &[f32] = bytemuck::cast_slice(mapped);
                all_f32.chunks_exact(pixels).map(|c| c.to_vec()).collect()
            },
        )
    }

    fn run_jpeg_binding(
        &self,
        scene_binding: wgpu::BufferBinding<'_>,
        pipeline: &wgpu::ComputePipeline,
        width: u32,
        height: u32,
        jpeg: &mut crate::GpuJpegEncoder,
        quality: u8,
    ) -> Result<Vec<u8>, GpuError> {
        let bind_group = self.make_bind_group(scene_binding);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-mosaic-jpeg-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-mosaic-jpeg-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }
        let jpeg_out_bytes = jpeg.record_encode_rgba8_from_buffer(
            &mut encoder,
            &self.out_buf,
            width,
            height,
            quality,
        )?;
        self.queue.submit(Some(encoder.finish()));
        jpeg.readback_jpeg(jpeg_out_bytes, width, height, quality)
    }

    fn run_pipeline_binding(
        &self,
        scene_binding: wgpu::BufferBinding<'_>,
        pipeline: &wgpu::ComputePipeline,
        wg_x: u32,
        wg_y: u32,
        out_bytes: usize,
    ) -> Result<Vec<u8>, GpuError> {
        let bind_group = self.make_bind_group(scene_binding);

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

        crate::wgpu_util::readback_staging_sync(
            &self.device,
            &self.staging_buf,
            out_bytes as u64,
            |mapped| mapped.to_vec(),
        )
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
            let MosaicCompositeResult::F32(staged_out) = comp
                .composite(MosaicUpload::Staged(&stack), method, MosaicOutput::F32)
                .unwrap()
            else {
                unreachable!();
            };
            let MosaicCompositeResult::F32(direct_out) = comp
                .composite(
                    MosaicUpload::Direct {
                        planes: &refs,
                        width: w,
                        height: h,
                    },
                    method,
                    MosaicOutput::F32,
                )
                .unwrap()
            else {
                unreachable!();
            };
            assert_eq!(staged_out.len(), direct_out.len());
            for i in 0..n {
                assert_eq!(
                    staged_out[i].to_bits(),
                    direct_out[i].to_bits(),
                    "{method:?} px {i}: Staged vs Direct bit mismatch"
                );
            }
            let mut max_err = 0.0f32;
            let mut mask_err = 0usize;
            for i in 0..n {
                let c = cpu_select(&planes, i, method);
                let g = staged_out[i];
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

    #[test]
    fn gpu_mosaic_batched_bands_and_fused_jpeg_match_reference() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU mosaic batched/JPEG test (no adapter): {e}");
                return;
            }
        };
        let mut comp = GpuMosaicCompositor::new(&ctx);
        let mut jpeg = crate::GpuJpegEncoder::new(&ctx);

        // 512x512 x 8 scenes = 8 MiB (> 2 MiB threshold, exercises parallel mapped_at_creation).
        let w = 512u32;
        let h = 512u32;
        let n = (w * h) as usize;
        let num_scenes = 8usize;
        let make_band = |seed: usize| -> Vec<Vec<f32>> {
            (0..num_scenes)
                .map(|s| {
                    (0..n)
                        .map(|i| {
                            if (i + s * 31 + seed * 17).is_multiple_of(9) {
                                f32::NAN
                            } else {
                                40.0 + ((i * 7 + s * 53 + seed * 29) % 200) as f32
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let b0 = make_band(0);
        let b1 = make_band(1);
        let b2 = make_band(2);
        let r0: Vec<&[f32]> = b0.iter().map(Vec::as_slice).collect();
        let r1: Vec<&[f32]> = b1.iter().map(Vec::as_slice).collect();
        let r2: Vec<&[f32]> = b2.iter().map(Vec::as_slice).collect();
        let s0 = comp.upload_scenes(&r0, w, h).unwrap();
        let s1 = comp.upload_scenes(&r1, w, h).unwrap();
        let s2 = comp.upload_scenes(&r2, w, h).unwrap();
        let direct_bands: Vec<Vec<&[f32]>> = vec![r0.clone(), r1.clone(), r2.clone()];

        for &method in &[
            GpuPixelSelection::Median,
            GpuPixelSelection::Mean,
            GpuPixelSelection::Stdev,
        ] {
            let MosaicCompositeResult::F32Bands(staged_batched) = comp
                .composite(
                    MosaicUpload::StagedBands(&[&s0, &s1, &s2]),
                    method,
                    MosaicOutput::F32Bands,
                )
                .unwrap()
            else {
                unreachable!();
            };
            let MosaicCompositeResult::F32Bands(direct_batched) = comp
                .composite(
                    MosaicUpload::DirectBands {
                        bands: &direct_bands,
                        width: w,
                        height: h,
                    },
                    method,
                    MosaicOutput::F32Bands,
                )
                .unwrap()
            else {
                unreachable!();
            };
            assert_eq!(staged_batched.len(), 3);
            assert_eq!(direct_batched.len(), 3);
            for (b_idx, ((stack, got_staged), got_direct)) in [&s0, &s1, &s2]
                .into_iter()
                .zip(&staged_batched)
                .zip(&direct_batched)
                .enumerate()
            {
                let MosaicCompositeResult::F32(single) = comp
                    .composite(MosaicUpload::Staged(stack), method, MosaicOutput::F32)
                    .unwrap()
                else {
                    unreachable!();
                };
                assert_eq!(got_staged.len(), single.len());
                assert_eq!(got_direct.len(), single.len());
                for i in 0..n {
                    assert_eq!(
                        got_staged[i].to_bits(),
                        single[i].to_bits(),
                        "{method:?} band {b_idx} px {i} staged mismatch"
                    );
                    assert_eq!(
                        got_direct[i].to_bits(),
                        single[i].to_bits(),
                        "{method:?} band {b_idx} px {i} direct mismatch"
                    );
                }
            }
        }

        // 1-pass fused Mosaic -> GPU JPEG and Styled (Rgba8 + IndexedU8) must be bit-identical
        // across Staged and Direct uploads, and fused JPEG must match 2-pass RGBA8 -> GPU JPEG.
        let mut lut = [0u32; 256];
        for (i, slot) in lut.iter_mut().enumerate() {
            let v = i as u32;
            *slot = v | ((255 - v) << 8) | (((v * 3) & 0xff) << 16) | (0xff << 24);
        }
        let MosaicCompositeResult::Bytes(staged_jpg) = comp
            .composite(
                MosaicUpload::Staged(&s0),
                GpuPixelSelection::Median,
                MosaicOutput::StyledJpeg {
                    lut: &lut,
                    jpeg: &mut jpeg,
                    quality: 85,
                    rescale: (40.0, 240.0),
                },
            )
            .unwrap()
        else {
            unreachable!();
        };
        let MosaicCompositeResult::Bytes(direct_jpg) = comp
            .composite(
                MosaicUpload::Direct {
                    planes: &r0,
                    width: w,
                    height: h,
                },
                GpuPixelSelection::Median,
                MosaicOutput::StyledJpeg {
                    lut: &lut,
                    jpeg: &mut jpeg,
                    quality: 85,
                    rescale: (40.0, 240.0),
                },
            )
            .unwrap()
        else {
            unreachable!();
        };
        assert!(staged_jpg.starts_with(&[0xff, 0xd8]) && staged_jpg.ends_with(&[0xff, 0xd9]));
        assert_eq!(
            staged_jpg, direct_jpg,
            "StyledJpeg must be bit-exact between Staged and Direct"
        );

        for out_kind in [GpuPixelOutput::Rgba8, GpuPixelOutput::IndexedU8] {
            let MosaicCompositeResult::Bytes(staged_styled) = comp
                .composite(
                    MosaicUpload::Staged(&s0),
                    GpuPixelSelection::Median,
                    MosaicOutput::Styled {
                        lut: &lut,
                        out: out_kind,
                        rescale: (40.0, 240.0),
                    },
                )
                .unwrap()
            else {
                unreachable!();
            };
            let MosaicCompositeResult::Bytes(direct_styled) = comp
                .composite(
                    MosaicUpload::Direct {
                        planes: &r0,
                        width: w,
                        height: h,
                    },
                    GpuPixelSelection::Median,
                    MosaicOutput::Styled {
                        lut: &lut,
                        out: out_kind,
                        rescale: (40.0, 240.0),
                    },
                )
                .unwrap()
            else {
                unreachable!();
            };
            assert_eq!(
                staged_styled, direct_styled,
                "Styled ({out_kind:?}) must be bit-exact between Staged and Direct"
            );
            if out_kind == GpuPixelOutput::Rgba8 {
                let ref_jpg = jpeg.encode_rgba8(&staged_styled, w, h, 85).unwrap();
                assert_eq!(
                    staged_jpg, ref_jpg,
                    "1-pass fused Mosaic->GPU JPEG must be bit-exact vs 2-pass RGBA8->GPU JPEG"
                );
            }
        }
    }

    #[test]
    fn gpu_mosaic_rgb_rgba8_and_fused_jpeg_match_cpu() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU RGB mosaic test (no adapter): {e}");
                return;
            }
        };
        let mut comp = GpuMosaicCompositor::new(&ctx);
        let mut jpeg = crate::GpuJpegEncoder::new(&ctx);

        let w = 256u32;
        let h = 256u32;
        let n = (w * h) as usize;
        let num_scenes = 10usize;
        let make_band = |seed: usize| -> Vec<Vec<f32>> {
            (0..num_scenes)
                .map(|s| {
                    (0..n)
                        .map(|i| {
                            // Force some all-NaN pixels in band 0, 1, or 2 to test any-band-NaN masking
                            if i == 17 + seed || (i + s * 41 + seed * 19).is_multiple_of(7) {
                                f32::NAN
                            } else {
                                10.0 * (seed as f32) + ((i * 11 + s * 59 + seed * 37) % 300) as f32
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let r_band = make_band(0);
        let g_band = make_band(1);
        let b_band = make_band(2);
        let r_refs: Vec<&[f32]> = r_band.iter().map(Vec::as_slice).collect();
        let g_refs: Vec<&[f32]> = g_band.iter().map(Vec::as_slice).collect();
        let b_refs: Vec<&[f32]> = b_band.iter().map(Vec::as_slice).collect();
        let rgb_planes: [&[&[f32]]; 3] = [&r_refs, &g_refs, &b_refs];
        let rgb_stack = comp.upload_rgb_scenes(&rgb_planes, w, h).unwrap();

        let rmin = [0.0f32, 10.0, 20.0];
        let rmax = [300.0f32, 310.0, 320.0];
        let MosaicCompositeResult::Bytes(rgba) = comp
            .composite(
                MosaicUpload::DirectRgb {
                    planes: &rgb_planes,
                    width: w,
                    height: h,
                },
                GpuPixelSelection::Median,
                MosaicOutput::RgbStyled {
                    rescale_min: rmin,
                    rescale_max: rmax,
                },
            )
            .unwrap()
        else {
            unreachable!();
        };
        let MosaicCompositeResult::Bytes(staged_rgba) = comp
            .composite(
                MosaicUpload::StagedRgb(&rgb_stack),
                GpuPixelSelection::Median,
                MosaicOutput::RgbStyled {
                    rescale_min: rmin,
                    rescale_max: rmax,
                },
            )
            .unwrap()
        else {
            unreachable!();
        };
        assert_eq!(rgba, staged_rgba, "RgbStyled must be bit-exact between DirectRgb and StagedRgb");
        assert_eq!(rgba.len(), n * 4);

        let inv = [
            255.0 / (rmax[0] - rmin[0]),
            255.0 / (rmax[1] - rmin[1]),
            255.0 / (rmax[2] - rmin[2]),
        ];
        let mut mask_err = 0usize;
        let mut max_diff = 0i32;
        for i in 0..n {
            let cr = cpu_select(&r_band, i, GpuPixelSelection::Median);
            let cg = cpu_select(&g_band, i, GpuPixelSelection::Median);
            let cb = cpu_select(&b_band, i, GpuPixelSelection::Median);
            let off = i * 4;
            if cr.is_finite() && cg.is_finite() && cb.is_finite() {
                if rgba[off + 3] != 255 {
                    mask_err += 1;
                } else {
                    let er = ((cr - rmin[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                    let eg = ((cg - rmin[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                    let eb = ((cb - rmin[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                    max_diff = max_diff
                        .max((er as i32 - rgba[off] as i32).abs())
                        .max((eg as i32 - rgba[off + 1] as i32).abs())
                        .max((eb as i32 - rgba[off + 2] as i32).abs());
                }
            } else if rgba[off..off + 4] != [0, 0, 0, 0] {
                mask_err += 1;
            }
        }
        assert_eq!(mask_err, 0, "RGB mosaic nodata mask mismatch");
        assert!(max_diff <= 1, "RGB mosaic max channel diff {max_diff} > 1");

        let MosaicCompositeResult::Bytes(fused_jpg) = comp
            .composite(
                MosaicUpload::DirectRgb {
                    planes: &rgb_planes,
                    width: w,
                    height: h,
                },
                GpuPixelSelection::Median,
                MosaicOutput::RgbStyledJpeg {
                    jpeg: &mut jpeg,
                    quality: 85,
                    rescale_min: rmin,
                    rescale_max: rmax,
                },
            )
            .unwrap()
        else {
            unreachable!();
        };
        let MosaicCompositeResult::Bytes(staged_jpg) = comp
            .composite(
                MosaicUpload::StagedRgb(&rgb_stack),
                GpuPixelSelection::Median,
                MosaicOutput::RgbStyledJpeg {
                    jpeg: &mut jpeg,
                    quality: 85,
                    rescale_min: rmin,
                    rescale_max: rmax,
                },
            )
            .unwrap()
        else {
            unreachable!();
        };
        assert_eq!(
            fused_jpg, staged_jpg,
            "RgbStyledJpeg must be bit-exact between DirectRgb and StagedRgb"
        );
        assert!(fused_jpg.starts_with(&[0xff, 0xd8]) && fused_jpg.ends_with(&[0xff, 0xd9]));
        let ref_jpg = jpeg.encode_rgba8(&rgba, w, h, 85).unwrap();
        assert_eq!(
            fused_jpg, ref_jpg,
            "1-pass fused RGB Mosaic->GPU JPEG must be bit-exact vs 2-pass RGBA8->GPU JPEG"
        );
    }

    #[test]
    fn pixel_selection_codes_match_mosaic_wgsl() {
        let mut wgsl_codes: Vec<u32> = Vec::new();
        let needle = "params.method == ";
        let mut rest = crate::shaders::MOSAIC_WGSL;
        while let Some(pos) = rest.find(needle) {
            rest = &rest[pos + needle.len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() {
                wgsl_codes.push(digits.parse().expect("valid u32 method literal"));
            }
        }
        wgsl_codes.sort_unstable();
        wgsl_codes.dedup();

        let rust_codes: Vec<u32> = [
            GpuPixelSelection::First,
            GpuPixelSelection::Highest,
            GpuPixelSelection::Lowest,
            GpuPixelSelection::Mean,
            GpuPixelSelection::Median,
            GpuPixelSelection::Stdev,
            GpuPixelSelection::Count,
        ]
        .into_iter()
        .map(GpuPixelSelection::as_u32)
        .collect();

        assert_eq!(
            wgsl_codes, rust_codes,
            "mosaic.wgsl params.method branches must match GpuPixelSelection::as_u32() exactly"
        );
    }
}
