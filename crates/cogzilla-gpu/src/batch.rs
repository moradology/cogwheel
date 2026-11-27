use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::colormap::{GpuColormapMode, GpuPixelOutput};
use crate::context::{GpuContext, GpuError};
use crate::resample::GpuSlab;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BatchParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    tile_w: u32,
    tile_h: u32,
    batch_size: u32,
    mode: u32,
    _pad0: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    _pad1: f32,
}

/// A single tile's separable 1D `(u_coords, v_coords)` inside a micro-batch.
pub struct TileCoordSlice<'a> {
    pub u_coords: &'a [f32],
    pub v_coords: &'a [f32],
}

/// Micro-batched 3D (`dispatch_workgroups(wg_x, wg_y, K)`) tile renderer.
///
/// Renders `K` tiles in a single command buffer + single readback, amortizing the synchronous
/// Metal (1.5 ms) / Vulkan (50–240 µs) submission floor across the entire batch.
pub struct GpuTileBatcher {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    rgba8_pipeline: wgpu::ComputePipeline,
    indexed_pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    lut_buf: wgpu::Buffer,
    u_buf: wgpu::Buffer,
    u_cap: usize,
    v_buf: wgpu::Buffer,
    v_cap: usize,
    out_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    out_words_cap: usize,
    u_scratch: Vec<f32>,
    v_scratch: Vec<f32>,
}

impl GpuTileBatcher {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-batch-tile-wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::BATCH_TILE_WGSL.into()),
        });

        use crate::wgpu_util::{bgl_storage_ro, bgl_storage_rw, bgl_uniform};
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-batch-bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_ro(3),
                bgl_storage_ro(4),
                bgl_storage_rw(5),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-batch-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let rgba8_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-batch-rgba8"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("batch_colormap_rgba8"),
            compilation_options: Default::default(),
            cache: None,
        });

        let indexed_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-batch-indexed-u8"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("batch_colormap_indexed_u8"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-batch-params"),
            size: std::mem::size_of::<BatchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-batch-lut"),
            size: (256 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_coord_cap = 256 * 64;
        let u_buf = Self::alloc_coord_buf(&device, "cogzilla-batch-u", init_coord_cap);
        let v_buf = Self::alloc_coord_buf(&device, "cogzilla-batch-v", init_coord_cap);

        let init_words = 64 * 256 * 64; // 64 tiles of 256x256 IndexedU8 = 4 MiB
        let (out_buf, staging_buf) = Self::alloc_out_pair(&device, init_words);

        Self {
            device,
            queue,
            bind_group_layout,
            rgba8_pipeline,
            indexed_pipeline,
            params_buf,
            lut_buf,
            u_buf,
            u_cap: init_coord_cap,
            v_buf,
            v_cap: init_coord_cap,
            out_buf,
            staging_buf,
            out_words_cap: init_words,
            u_scratch: Vec::with_capacity(init_coord_cap),
            v_scratch: Vec::with_capacity(init_coord_cap),
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
            label: Some("cogzilla-batch-out"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-batch-staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (out_buf, staging_buf)
    }

    fn ensure_capacities(&mut self, u_elems: usize, v_elems: usize, out_words: usize) {
        if u_elems > self.u_cap {
            self.u_cap = u_elems.next_power_of_two();
            self.u_buf = Self::alloc_coord_buf(&self.device, "cogzilla-batch-u", self.u_cap);
        }
        if v_elems > self.v_cap {
            self.v_cap = v_elems.next_power_of_two();
            self.v_buf = Self::alloc_coord_buf(&self.device, "cogzilla-batch-v", self.v_cap);
        }
        if out_words > self.out_words_cap {
            self.out_words_cap = out_words;
            let (out_buf, staging_buf) = Self::alloc_out_pair(&self.device, self.out_words_cap);
            self.out_buf = out_buf;
            self.staging_buf = staging_buf;
        }
    }

    /// Render a micro-batch of `tiles.len()` uniform-sized tiles from a GPU-resident `GpuSlab`
    /// in a single 3D compute dispatch and single readback.
    ///
    /// Returns a contiguous byte buffer containing `tiles.len()` back-to-back tile images:
    /// - `Rgba8`: `tile_w * tile_h * 4` bytes per tile
    /// - `IndexedU8`: `tile_w * tile_h` bytes per tile (requires `tile_w % 4 == 0` for zero-unpack)
    #[allow(clippy::too_many_arguments)]
    pub fn render_batch(
        &mut self,
        slab: &GpuSlab,
        tiles: &[TileCoordSlice<'_>],
        mode: GpuColormapMode,
        rescale_min: f32,
        rescale_max: f32,
        lut_rgba_u32: &[u32; 256],
        output: GpuPixelOutput,
    ) -> Result<Vec<u8>, GpuError> {
        let batch_size = tiles.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        let tile_w = tiles[0].u_coords.len();
        let tile_h = tiles[0].v_coords.len();
        if tile_w == 0 || tile_h == 0 {
            return Err(GpuError::InvalidDimensions(
                "tile dimensions must be > 0".into(),
            ));
        }
        for (i, t) in tiles.iter().enumerate() {
            if t.u_coords.len() != tile_w || t.v_coords.len() != tile_h {
                return Err(GpuError::InvalidDimensions(format!(
                    "batch tile {i} has mismatched dims {}x{} (expected {tile_w}x{tile_h})",
                    t.u_coords.len(),
                    t.v_coords.len()
                )));
            }
        }

        let words_per_row = match output {
            GpuPixelOutput::Rgba8 => tile_w,
            GpuPixelOutput::IndexedU8 => tile_w.div_ceil(4),
        };
        let tile_words = words_per_row * tile_h;
        let total_words = tile_words * batch_size;

        let u_total = tile_w * batch_size;
        let v_total = tile_h * batch_size;
        self.ensure_capacities(u_total, v_total, total_words);

        self.u_scratch.clear();
        self.v_scratch.clear();
        for t in tiles {
            self.u_scratch.extend_from_slice(t.u_coords);
            self.v_scratch.extend_from_slice(t.v_coords);
        }

        let range = rescale_max - rescale_min;
        let safe_range = if range.is_finite() && range.abs() > 1e-20 {
            range
        } else {
            1.0
        };
        let mode_u32 = match mode {
            GpuColormapMode::Bilinear => 0u32,
            GpuColormapMode::Cubic => 1u32,
            GpuColormapMode::Lanczos3 => 2u32,
            GpuColormapMode::Nearest => 3u32,
        };
        let params = BatchParams {
            src_w: slab.width,
            src_h: slab.height,
            src_stride: slab.stride,
            tile_w: tile_w as u32,
            tile_h: tile_h as u32,
            batch_size: batch_size as u32,
            mode: mode_u32,
            _pad0: 0,
            rescale_min,
            inv_range_255: 255.0 / safe_range,
            inv_range_254: 254.0 / safe_range,
            _pad1: 0.0,
        };

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.u_buf, 0, bytemuck::cast_slice(&self.u_scratch));
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(&self.v_scratch));
        if output == GpuPixelOutput::Rgba8 {
            self.queue
                .write_buffer(&self.lut_buf, 0, bytemuck::cast_slice(lut_rgba_u32));
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-batch-bg"),
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
                (tile_w as u32).div_ceil(16),
                (tile_h as u32).div_ceil(16),
            ),
            GpuPixelOutput::IndexedU8 => (
                &self.indexed_pipeline,
                (words_per_row as u32).div_ceil(16),
                (tile_h as u32).div_ceil(16),
            ),
        };

        let out_bytes = (total_words * std::mem::size_of::<u32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-batch-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-batch-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, batch_size as u32);
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
                if tile_w % 4 == 0 {
                    mapped.to_vec()
                } else {
                    let row_bytes = words_per_row * 4;
                    let tile_stride = row_bytes * tile_h;
                    let mut out = Vec::with_capacity(batch_size * tile_w * tile_h);
                    for t in 0..batch_size {
                        for py in 0..tile_h {
                            let start = t * tile_stride + py * row_bytes;
                            out.extend_from_slice(&mapped[start..start + tile_w]);
                        }
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
    use crate::colormap::pack_rgb_lut;
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
    use cogzilla_kernel::{BandSource, ColormapName, sample_cubic};

    #[test]
    fn batched_16_tiles_match_cpu_exact() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU batch test (no adapter): {e}");
                return;
            }
        };
        let mut batcher = GpuTileBatcher::new(&ctx);

        let src_w = 1024u32;
        let src_h = 1024u32;
        let mut src = vec![0.0f32; (src_w * src_h) as usize];
        for y in 0..src_h {
            for x in 0..src_w {
                let idx = (y * src_w + x) as usize;
                if x >= 300 && x < 340 && y >= 250 && y < 290 {
                    src[idx] = f32::NAN;
                } else {
                    src[idx] = 1000.0 + 500.0 * (x as f32 * 0.015).sin() * (y as f32 * 0.019).cos();
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
                min: 400.0,
                max: 1600.0,
            },
        };
        let blocks = [Some(&band)];
        let bs = BandSource::new(src_w, src_h, src_w, src_h, 0, 0, 1, 1, &blocks);
        let gpu_slab = GpuSlab::upload(&ctx.device, &src, src_w, src_h, src_w).unwrap();

        let tile_n = 256usize;
        let mut u_vecs = Vec::with_capacity(16);
        let mut v_vecs = Vec::with_capacity(16);
        for ty in 0..4 {
            for tx in 0..4 {
                let u0 = tx as f32 * 240.0 + 16.0;
                let v0 = ty as f32 * 240.0 + 16.0;
                u_vecs.push((0..tile_n).map(|i| u0 + i as f32 * 0.93).collect::<Vec<f32>>());
                v_vecs.push((0..tile_n).map(|i| v0 + i as f32 * 0.93).collect::<Vec<f32>>());
            }
        }
        let slices: Vec<TileCoordSlice<'_>> = u_vecs
            .iter()
            .zip(v_vecs.iter())
            .map(|(u, v)| TileCoordSlice {
                u_coords: u,
                v_coords: v,
            })
            .collect();

        let lut = ColormapName::Viridis.lut();
        let lut_u32 = pack_rgb_lut(&lut.rgb);
        let rmin = 500.0f32;
        let rmax = 1500.0f32;
        let inv254 = 254.0 / (rmax - rmin);

        let batch_out = batcher
            .render_batch(
                &gpu_slab,
                &slices,
                GpuColormapMode::Cubic,
                rmin,
                rmax,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();
        assert_eq!(batch_out.len(), 16 * tile_n * tile_n);

        let mut mask_err = 0usize;
        let mut max_diff = 0i32;
        for t in 0..16 {
            let tile_bytes = &batch_out[t * tile_n * tile_n..(t + 1) * tile_n * tile_n];
            for (py, &v) in v_vecs[t].iter().enumerate() {
                for (px, &u) in u_vecs[t].iter().enumerate() {
                    let val = sample_cubic(&bs, u, v);
                    let g = tile_bytes[py * tile_n + px];
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
        }
        assert_eq!(mask_err, 0, "batched nodata mask must match 100%");
        assert!(max_diff <= 1, "batched max diff {max_diff} > 1 LSB");
    }
}
