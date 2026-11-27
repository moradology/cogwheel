use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::context::{GpuContext, GpuError};
use crate::resample::GpuSlab;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPredictorKind {
    /// TIFF Predictor 2 (horizontal differencing, little-endian float32).
    Predictor2Le,
    /// TIFF Predictor 3 (Technical Note 3 floating-point horizontal differencing + byte shuffle).
    Predictor3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PredParams {
    tile_w: u32,
    tile_h: u32,
    batch_size: u32,
    has_nodata: u32,
    nodata_val: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU TIFF Predictor 2/3 unshuffle + nodata-normalization decoder.
///
/// Reconstructs batches of decompressed COG blocks directly into GPU-resident `f32` buffers
/// (zero intermediate scratch memory, 100% bit-exact against `cogzilla-codec`).
pub struct GpuPredictorDecoder {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    pred3_pipeline: wgpu::ComputePipeline,
    pred2_pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    raw_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    cap_words: usize,
}

impl GpuPredictorDecoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-predictor-wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::PREDICTOR_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-predictor-bgl"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-predictor-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pred3_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-unshuffle-pred3"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("unshuffle_pred3_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pred2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-reconstruct-pred2"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reconstruct_pred2_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-pred-params"),
            size: std::mem::size_of::<PredParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_words = 512 * 512 * 4;
        let (raw_buf, out_buf, staging_buf) = Self::alloc_buffers(&device, init_words);

        Self {
            device,
            queue,
            bind_group_layout,
            pred3_pipeline,
            pred2_pipeline,
            params_buf,
            raw_buf,
            out_buf,
            staging_buf,
            cap_words: init_words,
        }
    }

    fn alloc_buffers(
        device: &wgpu::Device,
        words: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let bytes = (words.max(1) * std::mem::size_of::<u32>()) as u64;
        let raw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-pred-raw"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-pred-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-pred-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (raw_buf, out_buf, staging_buf)
    }

    fn ensure_cap(&mut self, words: usize) {
        if words > self.cap_words {
            self.cap_words = words;
            let (raw_buf, out_buf, staging_buf) = Self::alloc_buffers(&self.device, self.cap_words);
            self.raw_buf = raw_buf;
            self.out_buf = out_buf;
            self.staging_buf = staging_buf;
        }
    }

    /// Reconstruct a batch of `batch_size` raw decompressed blocks (`raw_bytes.len() == batch_size * tile_w * tile_h * 4`)
    /// directly on the GPU, leaving the result resident in VRAM (no readback copy).
    pub fn decode_batch_resident(
        &mut self,
        raw_bytes: &[u8],
        tile_w: u32,
        tile_h: u32,
        batch_size: u32,
        kind: GpuPredictorKind,
        nodata: Option<f32>,
    ) -> Result<(), GpuError> {
        let total_words = (tile_w as usize) * (tile_h as usize) * (batch_size as usize);
        if tile_w == 0
            || tile_h == 0
            || batch_size == 0
            || tile_w % 4 != 0
            || raw_bytes.len() != total_words * 4
        {
            return Err(GpuError::InvalidDimensions(format!(
                "invalid predictor batch: {tile_w}x{tile_h} x {batch_size}, raw_bytes={}",
                raw_bytes.len()
            )));
        }
        self.ensure_cap(total_words);

        let finite_nd = nodata.filter(|v| !v.is_nan());
        let params = PredParams {
            tile_w,
            tile_h,
            batch_size,
            has_nodata: u32::from(finite_nd.is_some()),
            nodata_val: finite_nd.unwrap_or(0.0),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.raw_buf, 0, raw_bytes);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-pred-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.raw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline = match kind {
            GpuPredictorKind::Predictor3 => &self.pred3_pipeline,
            GpuPredictorKind::Predictor2Le => &self.pred2_pipeline,
        };

        let total_rows = batch_size * tile_h;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-pred-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-pred-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_rows.div_ceil(64), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    /// Dispatch Predictor 2/3 unshuffle over bytes already resident in `self.raw_buf`
    /// (models the post-`nvCOMP` in-VRAM pipeline with zero host->device upload).
    pub fn dispatch_uploaded_only(
        &self,
        _tile_w: u32,
        tile_h: u32,
        batch_size: u32,
        kind: GpuPredictorKind,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-pred-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.raw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });
        let pipeline = match kind {
            GpuPredictorKind::Predictor3 => &self.pred3_pipeline,
            GpuPredictorKind::Predictor2Le => &self.pred2_pipeline,
        };
        let total_rows = batch_size * tile_h;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-pred-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-pred-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_rows.div_ceil(64), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Reconstruct a single block directly into an owned `GpuSlab` resident in VRAM.
    pub fn decode_to_slab(
        &mut self,
        raw_bytes: &[u8],
        tile_w: u32,
        tile_h: u32,
        kind: GpuPredictorKind,
        nodata: Option<f32>,
    ) -> Result<GpuSlab, GpuError> {
        self.decode_batch_resident(raw_bytes, tile_w, tile_h, 1, kind, nodata)?;
        let bytes = (tile_w as u64) * (tile_h as u64) * 4;
        let slab_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-pred-slab"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-pred-copy-slab"),
            });
        enc.copy_buffer_to_buffer(&self.out_buf, 0, &slab_buf, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        Ok(GpuSlab {
            buffer: slab_buf,
            width: tile_w,
            height: tile_h,
            stride: tile_w,
        })
    }

    /// Reconstruct a batch and read back `f32` pixels to CPU (for verification / hybrid pipelines).
    pub fn decode_batch_readback(
        &mut self,
        raw_bytes: &[u8],
        tile_w: u32,
        tile_h: u32,
        batch_size: u32,
        kind: GpuPredictorKind,
        nodata: Option<f32>,
    ) -> Result<Vec<f32>, GpuError> {
        self.decode_batch_resident(raw_bytes, tile_w, tile_h, batch_size, kind, nodata)?;
        let out_bytes = ((tile_w as usize) * (tile_h as usize) * (batch_size as usize) * 4) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-pred-rb-enc"),
            });
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_codec::BandData;
    use cogzilla_codec::predictor::{reconstruct_predictor2, reconstruct_predictor3};
    use cogzilla_tiff::ByteOrder;

    fn encode_pred3_row(vals: &[f32]) -> Vec<u8> {
        let w = vals.len();
        let mut row = vec![0u8; w * 4];
        for (x, &v) in vals.iter().enumerate() {
            let b = v.to_be_bytes();
            row[x] = b[0];
            row[w + x] = b[1];
            row[2 * w + x] = b[2];
            row[3 * w + x] = b[3];
        }
        // Horizontal byte differencing
        for i in (1..row.len()).rev() {
            row[i] = row[i].wrapping_sub(row[i - 1]);
        }
        row
    }

    fn encode_pred2_le_row(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        let mut prev: u32 = 0;
        for &v in vals {
            let bits = v.to_bits();
            let diff = bits.wrapping_sub(prev);
            prev = bits;
            out.extend_from_slice(&diff.to_le_bytes());
        }
        out
    }

    #[test]
    fn gpu_predictor3_and_predictor2_are_bit_exact_vs_cpu() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU predictor test (no adapter): {e}");
                return;
            }
        };
        let mut dec = GpuPredictorDecoder::new(&ctx);

        let w = 256u32;
        let h = 256u32;
        let nodata = -9999.0f32;
        let mut src_vals = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                src_vals[i] = if x >= 40 && x < 60 && y >= 30 && y < 50 {
                    nodata
                } else {
                    -123.45 + (x as f32) * 0.75 + (y as f32) * 1.33
                };
            }
        }

        // 1. Predictor 3 bit-exact check
        let mut raw_p3 = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            raw_p3.extend_from_slice(&encode_pred3_row(&src_vals[y * w as usize..(y + 1) * w as usize]));
        }
        let mut cpu_buf = raw_p3.clone();
        let cpu_bands = reconstruct_predictor3(&mut cpu_buf, w, h, 1, Some(nodata));
        let BandData::F32(cpu_slab) = &cpu_bands[0].data else {
            panic!("expected F32 band");
        };
        let gpu_p3 = dec
            .decode_batch_readback(&raw_p3, w, h, 1, GpuPredictorKind::Predictor3, Some(nodata))
            .unwrap();
        assert_eq!(cpu_slab.len(), gpu_p3.len());
        for (i, (&c, &g)) in cpu_slab.as_slice().iter().zip(gpu_p3.iter()).enumerate() {
            if c.is_nan() {
                assert!(g.is_nan(), "pred3 pixel {i}: expected NaN, got {g}");
            } else {
                assert_eq!(
                    c.to_bits(),
                    g.to_bits(),
                    "pred3 pixel {i}: bit mismatch cpu={c} gpu={g}"
                );
            }
        }

        // 2. Predictor 2 LE bit-exact check
        let mut raw_p2 = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            raw_p2.extend_from_slice(&encode_pred2_le_row(&src_vals[y * w as usize..(y + 1) * w as usize]));
        }
        let cpu_p2 = reconstruct_predictor2(&raw_p2, w, h, 1, ByteOrder::Little, Some(nodata));
        let BandData::F32(cpu_p2_slab) = &cpu_p2[0].data else {
            panic!("expected F32 band");
        };
        let gpu_p2 = dec
            .decode_batch_readback(&raw_p2, w, h, 1, GpuPredictorKind::Predictor2Le, Some(nodata))
            .unwrap();
        for (i, (&c, &g)) in cpu_p2_slab.as_slice().iter().zip(gpu_p2.iter()).enumerate() {
            if c.is_nan() {
                assert!(g.is_nan(), "pred2 pixel {i}: expected NaN, got {g}");
            } else {
                assert_eq!(
                    c.to_bits(),
                    g.to_bits(),
                    "pred2 pixel {i}: bit mismatch cpu={c} gpu={g}"
                );
            }
        }
    }
}
