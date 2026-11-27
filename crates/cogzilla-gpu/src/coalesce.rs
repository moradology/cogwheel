//! Heterogeneous command-buffer micro-batcher (`GpuCoalescer`).
//!
//! Coalesces arbitrary concurrent GPU jobs — across different COGs, slabs, colormaps,
//! `CoordPlan`s (`Separable1D`, 17×17 `Grid2D`, `Exact2D`), output formats (`Rgba8`, `IndexedU8`,
//! `RawF32`), and shader families (single-band colormap + DEM Horn 3×3 / Terrarium / TerrainRGB)
//! — into **one `wgpu::CommandEncoder` → one `queue.submit()` → one `staging_buf.map_async()` →
//! one `device.poll(Wait)`**.
//!
//! This eliminates `(K - 1)` Metal/Vulkan synchronization fences across concurrent HTTP tile
//! requests without requiring callers to use a batch API or share the same source slab.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};

use crate::colormap::{GpuColormapMode, GpuPixelOutput};
use crate::context::{GpuContext, GpuError};
use crate::expr::{GpuColorFormula, GpuInstr, GpuMultiBandSlab};
use crate::resample::GpuSlab;
use crate::terrain::GpuDemAlgorithm;

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

/// Owned coordinate plan for submission to `GpuCoalescer`.
#[derive(Debug, Clone)]
pub enum OwnedCoordPlan {
    Separable1D { u: Vec<f32>, v: Vec<f32> },
    Grid2D { dst_w: usize, dst_h: usize, grid_u: [f32; 289], grid_v: [f32; 289] },
    Exact2D { dst_w: usize, dst_h: usize, u: Vec<f32>, v: Vec<f32> },
    /// 1D separable axes packed into `uv_packed = [u (dst_w), v (dst_h)]` + 1-bit-per-pixel mask.
    SeparableMasked {
        dst_w: usize,
        dst_h: usize,
        uv_packed: Vec<f32>,
        mask_bits: Vec<u32>,
    },
    /// 17×17 Grid2D packed into `uv_packed = [grid_u (289), grid_v (289)]` + 1-bit-per-pixel mask.
    Grid2DMasked {
        dst_w: usize,
        dst_h: usize,
        uv_packed: Vec<f32>,
        mask_bits: Vec<u32>,
    },
}

impl OwnedCoordPlan {
    fn dims(&self) -> (usize, usize) {
        match self {
            Self::Separable1D { u, v } => (u.len(), v.len()),
            Self::Grid2D { dst_w, dst_h, .. }
            | Self::Exact2D { dst_w, dst_h, .. }
            | Self::SeparableMasked { dst_w, dst_h, .. }
            | Self::Grid2DMasked { dst_w, dst_h, .. } => (*dst_w, *dst_h),
        }
    }

    fn dims_mode_slices(&self) -> Result<(usize, usize, u32, &[u8], &[u8]), GpuError> {
        match self {
            Self::Separable1D { u, v } => Ok((
                u.len(),
                v.len(),
                0,
                bytemuck::cast_slice(u.as_slice()),
                bytemuck::cast_slice(v.as_slice()),
            )),
            Self::Grid2D {
                dst_w,
                dst_h,
                grid_u,
                grid_v,
            } => Ok((
                *dst_w,
                *dst_h,
                1,
                bytemuck::cast_slice(grid_u.as_slice()),
                bytemuck::cast_slice(grid_v.as_slice()),
            )),
            Self::Exact2D { dst_w, dst_h, u, v } => {
                if u.len() != dst_w * dst_h || v.len() != dst_w * dst_h {
                    return Err(GpuError::InvalidDimensions(
                        "Exact2D u/v len mismatch".into(),
                    ));
                }
                Ok((
                    *dst_w,
                    *dst_h,
                    2,
                    bytemuck::cast_slice(u.as_slice()),
                    bytemuck::cast_slice(v.as_slice()),
                ))
            }
            Self::SeparableMasked {
                dst_w,
                dst_h,
                uv_packed,
                mask_bits,
            } => {
                let need_words = (dst_w * dst_h).div_ceil(32);
                if uv_packed.len() != dst_w + dst_h || mask_bits.len() < need_words {
                    return Err(GpuError::InvalidDimensions(
                        "SeparableMasked uv_packed/mask_bits len mismatch".into(),
                    ));
                }
                Ok((
                    *dst_w,
                    *dst_h,
                    3,
                    bytemuck::cast_slice(uv_packed.as_slice()),
                    bytemuck::cast_slice(&mask_bits[..need_words]),
                ))
            }
            Self::Grid2DMasked {
                dst_w,
                dst_h,
                uv_packed,
                mask_bits,
            } => {
                let need_words = (dst_w * dst_h).div_ceil(32);
                if uv_packed.len() != 578 || mask_bits.len() < need_words {
                    return Err(GpuError::InvalidDimensions(
                        "Grid2DMasked uv_packed/mask_bits len mismatch".into(),
                    ));
                }
                Ok((
                    *dst_w,
                    *dst_h,
                    4,
                    bytemuck::cast_slice(uv_packed.as_slice()),
                    bytemuck::cast_slice(&mask_bits[..need_words]),
                ))
            }
        }
    }
}

/// Output pixel format for a coalesced GPU job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalescedOutput {
    Rgba8,
    IndexedU8,
    RawF32,
}

impl From<GpuPixelOutput> for CoalescedOutput {
    fn from(v: GpuPixelOutput) -> Self {
        match v {
            GpuPixelOutput::Rgba8 => Self::Rgba8,
            GpuPixelOutput::IndexedU8 => Self::IndexedU8,
        }
    }
}

/// A heterogeneous GPU render job eligible for command-buffer coalescing.
pub enum CoalescedJob {
    Colormap {
        slab: Arc<GpuSlab>,
        plan: OwnedCoordPlan,
        mode: GpuColormapMode,
        rescale_min: f32,
        rescale_max: f32,
        lut_u32: [u32; 256],
        output: CoalescedOutput,
    },
    Terrain {
        slab: Arc<GpuSlab>,
        plan: OwnedCoordPlan,
        mode: GpuColormapMode,
        alg: GpuDemAlgorithm,
        pixel_size_x_m: f32,
        pixel_size_y_m: f32,
        rescale_min: f32,
        rescale_max: f32,
        lut_u32: [u32; 256],
        output: CoalescedOutput,
    },
    Expr {
        slab: Arc<GpuMultiBandSlab>,
        plan: OwnedCoordPlan,
        mode: GpuColormapMode,
        instrs: Vec<GpuInstr>,
        rescale_min: f32,
        rescale_max: f32,
        lut_u32: [u32; 256],
        output: CoalescedOutput,
    },
    Rgb {
        slab: Arc<GpuMultiBandSlab>,
        plan: OwnedCoordPlan,
        mode: GpuColormapMode,
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
        formula: GpuColorFormula,
    },
    SarRgb {
        slab: Arc<GpuMultiBandSlab>,
        plan: OwnedCoordPlan,
        mode: GpuColormapMode,
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
        formula: GpuColorFormula,
    },
    ExprRgb {
        slab: Arc<GpuMultiBandSlab>,
        plan: OwnedCoordPlan,
        mode: GpuColormapMode,
        instrs: Vec<GpuInstr>,
        lens: [u8; 3],
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
        formula: GpuColorFormula,
    },
    MultiBandF32 {
        slab: Arc<GpuMultiBandSlab>,
        plan: OwnedCoordPlan,
        mode: GpuColormapMode,
    },
    ExprRgbF32 {
        slab: Arc<GpuMultiBandSlab>,
        plan: OwnedCoordPlan,
        mode: GpuColormapMode,
        instrs: Vec<GpuInstr>,
        lens: [u8; 3],
    },
    /// Chains `jpeg_dct.wgsl` directly onto an RGBA8-producing inner job inside the same
    /// `CommandEncoder` (zero RGBA8 readback, zero RGBA8 re-upload), returning packed
    /// quantized `i16` MCU coefficients ready for `crate::jpeg::huffman_encode_coeffs`.
    WithJpeg {
        inner: Box<CoalescedJob>,
        quality: u8,
    },
    /// Chains the 3-stage `stats.wgsl` reducer directly onto a `RawF32`-producing inner job
    /// (`Colormap`, `Terrain`, `Expr`) inside the same `CommandEncoder` (zero `f32` readback,
    /// zero `f32` re-upload), returning a packed ~12 KB buffer ready for
    /// `crate::stats::decode_stats_readback`.
    WithStats {
        inner: Box<CoalescedJob>,
        nodata: Option<f32>,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
    },
}

impl CoalescedJob {
    /// Conservative upper bound on `(staging_bytes, max_slot_storage_bytes)` required by this job.
    /// Used to guard `wgpu::Device::limits()` before submission and cap per-batch staging size.
    pub(crate) fn estimated_staging_and_slot_bytes(&self) -> (u64, u64) {
        let plan_dims = |plan: &OwnedCoordPlan| -> (u64, u64) {
            let (w, h) = plan.dims();
            (w as u64, h as u64)
        };
        let out_bytes_for = |w: u64, h: u64, out: CoalescedOutput| -> u64 {
            match out {
                CoalescedOutput::IndexedU8 => w.div_ceil(4).saturating_mul(h).saturating_mul(4),
                CoalescedOutput::Rgba8 | CoalescedOutput::RawF32 => {
                    w.saturating_mul(h).saturating_mul(4)
                }
            }
        };
        match self {
            CoalescedJob::Colormap { plan, output, .. }
            | CoalescedJob::Terrain { plan, output, .. }
            | CoalescedJob::Expr { plan, output, .. } => {
                let (w, h) = plan_dims(plan);
                let b = out_bytes_for(w, h, *output);
                (b, b)
            }
            CoalescedJob::Rgb { plan, .. }
            | CoalescedJob::SarRgb { plan, .. }
            | CoalescedJob::ExprRgb { plan, .. } => {
                let (w, h) = plan_dims(plan);
                let b = w.saturating_mul(h).saturating_mul(4);
                (b, b)
            }
            CoalescedJob::MultiBandF32 { slab, plan, .. } => {
                let (w, h) = plan_dims(plan);
                let b = (slab.num_bands as u64)
                    .saturating_mul(w)
                    .saturating_mul(h)
                    .saturating_mul(4);
                (b, b)
            }
            CoalescedJob::ExprRgbF32 { plan, .. } => {
                let (w, h) = plan_dims(plan);
                let b = 3u64
                    .saturating_mul(w)
                    .saturating_mul(h)
                    .saturating_mul(4);
                (b, b)
            }
            CoalescedJob::WithJpeg { inner, .. } => {
                let (_, inner_slot) = inner.estimated_staging_and_slot_bytes();
                let (w, h) = match inner.as_ref() {
                    CoalescedJob::Colormap { plan, .. }
                    | CoalescedJob::Terrain { plan, .. }
                    | CoalescedJob::Expr { plan, .. }
                    | CoalescedJob::Rgb { plan, .. }
                    | CoalescedJob::SarRgb { plan, .. }
                    | CoalescedJob::ExprRgb { plan, .. }
                    | CoalescedJob::MultiBandF32 { plan, .. }
                    | CoalescedJob::ExprRgbF32 { plan, .. } => plan_dims(plan),
                    _ => (0, 0),
                };
                let jpeg_bytes = w
                    .div_ceil(16)
                    .saturating_mul(h.div_ceil(16))
                    .saturating_mul(192 * 4);
                (jpeg_bytes, inner_slot.max(jpeg_bytes))
            }
            CoalescedJob::WithStats { inner, .. } => {
                let (_, inner_slot) = inner.estimated_staging_and_slot_bytes();
                let stats_bytes = (std::mem::size_of::<crate::stats::ResolvedStats>()
                    + 2 * crate::stats::CDF_BINS * 4
                    + (crate::stats::MAX_WG as usize) * 4) as u64;
                (stats_bytes, inner_slot.max(stats_bytes))
            }
        }
    }
}

/// Telemetry counters for `GpuCoalescer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoalescerStats {
    pub batches_submitted: u64,
    pub jobs_executed: u64,
    pub max_batch_seen: usize,
}

type ReplyFn = Box<dyn FnOnce(Result<Vec<u8>, GpuError>) + Send>;

struct PendingRequest {
    job: CoalescedJob,
    reply: ReplyFn,
    cancelled: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl PendingRequest {
    #[inline]
    fn is_cancelled(&self) -> bool {
        self.cancelled
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Acquire))
    }
}

struct StatsSlotBuffers {
    params_buf: wgpu::Buffer,
    partials_buf: wgpu::Buffer,
    resolved_buf: wgpu::Buffer,
    hist_buf: wgpu::Buffer,
    var_buf: wgpu::Buffer,
}

impl StatsSlotBuffers {
    fn new(device: &wgpu::Device) -> Self {
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-stats-params"),
            size: std::mem::size_of::<crate::stats::StatsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-stats-partials"),
            size: (crate::stats::MAX_WG as usize * std::mem::size_of::<crate::stats::PartialStats>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let resolved_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-stats-resolved"),
            size: std::mem::size_of::<crate::stats::ResolvedStats>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-stats-hist"),
            size: (2 * crate::stats::CDF_BINS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-stats-var"),
            size: (crate::stats::MAX_WG as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            params_buf,
            partials_buf,
            resolved_buf,
            hist_buf,
            var_buf,
        }
    }
}

struct JobSlot {
    cm_params_buf: wgpu::Buffer,
    tr_params_buf: wgpu::Buffer,
    expr_params_buf: wgpu::Buffer,
    instr_buf: wgpu::Buffer,
    lut_buf: wgpu::Buffer,
    u_buf: wgpu::Buffer,
    u_cap: usize,
    v_buf: wgpu::Buffer,
    v_cap: usize,
    out_buf: wgpu::Buffer,
    out_words_cap: usize,
    jpeg_params_buf: wgpu::Buffer,
    jpeg_inv_q_buf: wgpu::Buffer,
    jpeg_out_buf: Option<wgpu::Buffer>,
    jpeg_out_cap_words: usize,
    stats_slot: Option<StatsSlotBuffers>,
}

impl JobSlot {
    fn new(device: &wgpu::Device) -> Self {
        let cm_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-cm-params"),
            size: std::mem::size_of::<TerrainParams>().max(std::mem::size_of::<ColormapParams>())
                as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tr_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-tr-params"),
            size: std::mem::size_of::<TerrainParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let expr_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-expr-params"),
            size: std::mem::size_of::<ExprParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instr_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-instrs"),
            size: (256 * std::mem::size_of::<GpuInstr>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-lut"),
            size: (256 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let init_coord = 512usize;
        let u_buf = Self::alloc_coord(device, init_coord);
        let v_buf = Self::alloc_coord(device, init_coord);
        let init_words = 256 * 256;
        let out_buf = Self::alloc_out(device, init_words);
        let jpeg_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-jpeg-params"),
            size: std::mem::size_of::<crate::jpeg::JpegParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let jpeg_inv_q_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-jpeg-inv-q"),
            size: (128 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            cm_params_buf,
            tr_params_buf,
            expr_params_buf,
            instr_buf,
            lut_buf,
            u_buf,
            u_cap: init_coord,
            v_buf,
            v_cap: init_coord,
            out_buf,
            out_words_cap: init_words,
            jpeg_params_buf,
            jpeg_inv_q_buf,
            jpeg_out_buf: None,
            jpeg_out_cap_words: 0,
            stats_slot: None,
        }
    }

    fn alloc_coord(device: &wgpu::Device, elems: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-coord"),
            size: (elems.max(1) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn alloc_out(device: &wgpu::Device, words: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-out"),
            size: (words.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn ensure_caps(&mut self, device: &wgpu::Device, u_len: usize, v_len: usize, out_words: usize) {
        let max_words = (device.limits().max_storage_buffer_binding_size / 4) as usize;
        if u_len > self.u_cap {
            self.u_cap = u_len.next_power_of_two().min(max_words.max(u_len));
            self.u_buf = Self::alloc_coord(device, self.u_cap);
        }
        if v_len > self.v_cap {
            self.v_cap = v_len.next_power_of_two().min(max_words.max(v_len));
            self.v_buf = Self::alloc_coord(device, self.v_cap);
        }
        if out_words > self.out_words_cap {
            self.out_words_cap = out_words.next_power_of_two().min(max_words.max(out_words));
            self.out_buf = Self::alloc_out(device, self.out_words_cap);
        }
    }

    fn ensure_jpeg_caps(&mut self, device: &wgpu::Device, jpeg_words: usize) {
        let max_words = (device.limits().max_storage_buffer_binding_size / 4) as usize;
        if jpeg_words > self.jpeg_out_cap_words || self.jpeg_out_buf.is_none() {
            self.jpeg_out_cap_words = jpeg_words
                .next_power_of_two()
                .min(max_words.max(jpeg_words));
            self.jpeg_out_buf = Some(Self::alloc_out(device, self.jpeg_out_cap_words));
        }
    }

    fn ensure_stats_slot(&mut self, device: &wgpu::Device) -> &StatsSlotBuffers {
        self.stats_slot
            .get_or_insert_with(|| StatsSlotBuffers::new(device))
    }
}

struct PreparedJob {
    slot_idx: usize,
    dst_w: usize,
    dst_h: usize,
    words_per_row: usize,
    out_words: usize,
    staging_byte_off: u64,
    output: CoalescedOutput,
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
    pipe_kind: u8,
    bind_group: wgpu::BindGroup,
    jpeg_pass: Option<(u32, u32, wgpu::BindGroup)>,
    stats_pass: Option<(u32, wgpu::BindGroup)>,
    reply: ReplyFn,
}

struct InFlightBatch {
    prepared: Vec<PreparedJob>,
    total_bytes: u64,
    sub_idx: wgpu::SubmissionIndex,
    map_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ready: Arc<AtomicBool>,
}

struct RingFrame {
    slots: Vec<JobSlot>,
    staging_buf: wgpu::Buffer,
    staging_words_cap: usize,
    in_flight: Option<InFlightBatch>,
}

impl RingFrame {
    fn new(device: &wgpu::Device) -> Self {
        let staging_words_cap = 512 * 512 * 4;
        let staging_buf = WorkerState::alloc_staging(device, staging_words_cap);
        Self {
            slots: Vec::new(),
            staging_buf,
            staging_words_cap,
            in_flight: None,
        }
    }
}

struct WorkerState {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bgl: wgpu::BindGroupLayout,
    expr_bgl: wgpu::BindGroupLayout,
    jpeg_bgl: wgpu::BindGroupLayout,
    stats_bgl: wgpu::BindGroupLayout,
    cm_rgba8: wgpu::ComputePipeline,
    cm_indexed: wgpu::ComputePipeline,
    cm_f32: wgpu::ComputePipeline,
    tr_rgba8: wgpu::ComputePipeline,
    tr_indexed: wgpu::ComputePipeline,
    tr_f32: wgpu::ComputePipeline,
    tr_rgb: wgpu::ComputePipeline,
    expr_rgba8: wgpu::ComputePipeline,
    expr_indexed: wgpu::ComputePipeline,
    expr_f32: wgpu::ComputePipeline,
    rgb_formula: wgpu::ComputePipeline,
    sar_rgb_formula: wgpu::ComputePipeline,
    expr_rgb_formula: wgpu::ComputePipeline,
    mb_resample_f32: wgpu::ComputePipeline,
    expr_rgb_f32: wgpu::ComputePipeline,
    jpeg_fdct: wgpu::ComputePipeline,
    stats_pass1: wgpu::ComputePipeline,
    stats_finalize1: wgpu::ComputePipeline,
    stats_pass2: wgpu::ComputePipeline,
    frames: Vec<RingFrame>,
    in_flight_order: VecDeque<usize>,
    free_frames: VecDeque<usize>,
}

impl WorkerState {
    fn new(ctx: &GpuContext, ring_depth: usize) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let cm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-coal-cm-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/colormap.wgsl").into()),
        });
        let tr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-coal-tr-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain.wgsl").into()),
        });
        let expr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-coal-expr-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/expr.wgsl").into()),
        });

        // Both colormap.wgsl and terrain.wgsl share the exact same 6-binding layout:
        // 0: uniform params, 1: slab, 2: u_coords, 3: v_coords, 4: lut, 5: dst_words
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-coal-bgl"),
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

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-coal-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let mk_pipe = |label: &str, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let cm_rgba8 = mk_pipe("coal-cm-rgba8", &cm_shader, "fused_colormap_rgba8");
        let cm_indexed = mk_pipe("coal-cm-idx", &cm_shader, "fused_colormap_indexed_u8");
        let cm_f32 = mk_pipe("coal-cm-f32", &cm_shader, "resample_f32");
        let tr_rgba8 = mk_pipe("coal-tr-rgba8", &tr_shader, "dem_horn_rgba8");
        let tr_indexed = mk_pipe("coal-tr-idx", &tr_shader, "dem_horn_indexed_u8");
        let tr_f32 = mk_pipe("coal-tr-f32", &tr_shader, "dem_horn_f32");
        let tr_rgb = mk_pipe("coal-tr-rgb", &tr_shader, "dem_terrain_rgb");

        // 7-binding layout for expr.wgsl:
        // 0: uniform ExprParams, 1: multi-band slab, 2: u, 3: v, 4: instrs, 5: lut, 6: dst_words
        let expr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-coal-expr-bgl"),
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
        let expr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-coal-expr-pl"),
            bind_group_layouts: &[&expr_bgl],
            push_constant_ranges: &[],
        });
        let mk_expr_pipe = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&expr_pl),
                module: &expr_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let expr_rgba8 = mk_expr_pipe("coal-expr-rgba8", "expr_eval_rgba8");
        let expr_indexed = mk_expr_pipe("coal-expr-idx", "expr_eval_indexed_u8");
        let expr_f32 = mk_expr_pipe("coal-expr-f32", "expr_eval_f32");
        let rgb_formula = mk_expr_pipe("coal-rgb-formula", "rgb_color_formula_rgba8");
        let sar_rgb_formula = mk_expr_pipe("coal-sar-rgb-formula", "sar_rgb_color_formula_rgba8");
        let expr_rgb_formula = mk_expr_pipe("coal-expr-rgb-formula", "expr_rgb_color_formula_rgba8");
        let mb_resample_f32 = mk_expr_pipe("coal-mb-f32", "multiband_resample_f32");
        let expr_rgb_f32 = mk_expr_pipe("coal-expr-rgb-f32", "expr_rgb_f32");

        let jpeg_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-coal-jpeg-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/jpeg_dct.wgsl").into()),
        });
        let jpeg_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-coal-jpeg-bgl"),
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
        let jpeg_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-coal-jpeg-pl"),
            bind_group_layouts: &[&jpeg_bgl],
            push_constant_ranges: &[],
        });
        let jpeg_fdct = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("coal-jpeg-fdct"),
            layout: Some(&jpeg_pl),
            module: &jpeg_shader,
            entry_point: Some("jpeg_mcu_420_fdct"),
            compilation_options: Default::default(),
            cache: None,
        });

        let stats_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-coal-stats-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stats.wgsl").into()),
        });
        let stats_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-coal-stats-bgl"),
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
        let stats_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-coal-stats-pl"),
            bind_group_layouts: &[&stats_bgl],
            push_constant_ranges: &[],
        });
        let mk_stats_pipe = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&stats_pl),
                module: &stats_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let stats_pass1 = mk_stats_pipe("coal-stats-pass1", "stats_reduce_pass1");
        let stats_finalize1 = mk_stats_pipe("coal-stats-finalize1", "stats_finalize_pass1");
        let stats_pass2 = mk_stats_pipe("coal-stats-pass2", "stats_hist_pass2");

        let depth = ring_depth.clamp(1, 4);
        let frames: Vec<RingFrame> = (0..depth).map(|_| RingFrame::new(&device)).collect();
        let free_frames: VecDeque<usize> = (0..depth).collect();

        Self {
            device,
            queue,
            bgl,
            expr_bgl,
            jpeg_bgl,
            stats_bgl,
            cm_rgba8,
            cm_indexed,
            cm_f32,
            tr_rgba8,
            tr_indexed,
            tr_f32,
            tr_rgb,
            expr_rgba8,
            expr_indexed,
            expr_f32,
            rgb_formula,
            sar_rgb_formula,
            expr_rgb_formula,
            mb_resample_f32,
            expr_rgb_f32,
            jpeg_fdct,
            stats_pass1,
            stats_finalize1,
            stats_pass2,
            frames,
            in_flight_order: VecDeque::with_capacity(depth),
            free_frames,
        }
    }

    fn alloc_staging(device: &wgpu::Device, words: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-coal-staging"),
            size: (words.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Non-blocking submission of a batch into `frame_idx`. Returns `true` if work was submitted
    /// to the GPU (so the caller should keep `frame_idx` in `in_flight_order`).
    fn submit_batch(&mut self, frame_idx: usize, batch: Vec<PendingRequest>) -> bool {
        if batch.is_empty() {
            return false;
        }
        let frame = &mut self.frames[frame_idx];
        while frame.slots.len() < batch.len() {
            frame.slots.push(JobSlot::new(&self.device));
        }

        let mut prepared = Vec::with_capacity(batch.len());
        let mut total_staging_words = 0usize;

        for (slot_idx, req) in batch.into_iter().enumerate() {
            let reply = req.reply;
            let (job, jpeg_quality, stats_cfg) = match req.job {
                CoalescedJob::WithJpeg { inner, quality } => (*inner, Some(quality), None),
                CoalescedJob::WithStats {
                    inner,
                    nodata,
                    num_bins,
                    hist_range,
                } => (*inner, None, Some((nodata, num_bins, hist_range))),
                other => (other, None, None),
            };
            let prev_len = prepared.len();
            match job {
                CoalescedJob::Colormap {
                    slab,
                    plan,
                    mode,
                    rescale_min,
                    rescale_max,
                    lut_u32,
                    output,
                } => {
                    let (dst_w, dst_h, coord_mode, u_slice, v_slice) = match plan.dims_mode_slices()
                    {
                        Ok(t) => t,
                        Err(e) => {
                            reply(Err(e));
                            continue;
                        }
                    };
                    if dst_w == 0 || dst_h == 0 {
                        reply(Err(GpuError::InvalidDimensions(
                            "zero dst dims".into(),
                        )));
                        continue;
                    }
                    let words_per_row = match output {
                        CoalescedOutput::Rgba8 | CoalescedOutput::RawF32 => dst_w,
                        CoalescedOutput::IndexedU8 => dst_w.div_ceil(4),
                    };
                    let out_words = words_per_row * dst_h;
                    let slot = &mut frame.slots[slot_idx];
                    slot.ensure_caps(&self.device, u_slice.len() / 4, v_slice.len() / 4, out_words);

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
                        .write_buffer(&slot.cm_params_buf, 0, bytemuck::bytes_of(&params));
                    self.queue.write_buffer(&slot.u_buf, 0, u_slice);
                    self.queue.write_buffer(&slot.v_buf, 0, v_slice);
                    if output == CoalescedOutput::Rgba8 {
                        self.queue
                            .write_buffer(&slot.lut_buf, 0, bytemuck::cast_slice(&lut_u32));
                    }

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("cogzilla-coal-cm-bg"),
                        layout: &self.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.cm_params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slab.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: slot.u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: slot.v_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: slot.lut_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: slot.out_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let (pipe_kind, wg_x, wg_y) = match output {
                        CoalescedOutput::Rgba8 => {
                            (0u8, (dst_w as u32).div_ceil(16), (dst_h as u32).div_ceil(16))
                        }
                        CoalescedOutput::IndexedU8 => (
                            1u8,
                            (words_per_row as u32).div_ceil(16),
                            (dst_h as u32).div_ceil(16),
                        ),
                        CoalescedOutput::RawF32 => {
                            (2u8, (dst_w as u32).div_ceil(16), (dst_h as u32).div_ceil(16))
                        }
                    };
                    let staging_byte_off = (total_staging_words * 4) as u64;
                    total_staging_words += out_words;
                    prepared.push(PreparedJob {
                        slot_idx,
                        dst_w,
                        dst_h,
                        words_per_row,
                        out_words,
                        staging_byte_off,
                        output,
                        wg_x,
                        wg_y,
                        wg_z: 1,
                        pipe_kind,
                        bind_group,
                        jpeg_pass: None,
                        stats_pass: None,
                        reply,
                    });
                }
                CoalescedJob::Terrain {
                    slab,
                    plan,
                    mode,
                    alg,
                    pixel_size_x_m,
                    pixel_size_y_m,
                    rescale_min,
                    rescale_max,
                    lut_u32,
                    output,
                } => {
                    let (dst_w, dst_h, coord_mode, u_slice, v_slice) = match plan.dims_mode_slices()
                    {
                        Ok(t) => t,
                        Err(e) => {
                            reply(Err(e));
                            continue;
                        }
                    };
                    if dst_w == 0 || dst_h == 0 {
                        reply(Err(GpuError::InvalidDimensions(
                            "zero dst dims".into(),
                        )));
                        continue;
                    }
                    let (alg_mode, sun_e, sun_n, sun_u, z_factor, is_terrain_rgb) = match alg {
                        GpuDemAlgorithm::Hillshade {
                            azimuth_deg,
                            altitude_deg,
                            z_factor,
                        } => {
                            let az = azimuth_deg.to_radians();
                            let alt = altitude_deg.to_radians();
                            (
                                0u32,
                                alt.cos() * az.sin(),
                                alt.cos() * az.cos(),
                                alt.sin(),
                                z_factor,
                                false,
                            )
                        }
                        GpuDemAlgorithm::Slope { z_factor } => (1u32, 0.0, 0.0, 1.0, z_factor, false),
                        GpuDemAlgorithm::Aspect => (2u32, 0.0, 0.0, 1.0, 1.0, false),
                        GpuDemAlgorithm::Terrarium => (3u32, 0.0, 0.0, 1.0, 1.0, true),
                        GpuDemAlgorithm::TerrainRgb => (4u32, 0.0, 0.0, 1.0, 1.0, true),
                    };
                    let eff_output = if is_terrain_rgb {
                        CoalescedOutput::Rgba8
                    } else {
                        output
                    };
                    let words_per_row = match eff_output {
                        CoalescedOutput::Rgba8 | CoalescedOutput::RawF32 => dst_w,
                        CoalescedOutput::IndexedU8 => dst_w.div_ceil(4),
                    };
                    let out_words = words_per_row * dst_h;
                    let slot = &mut frame.slots[slot_idx];
                    slot.ensure_caps(&self.device, u_slice.len() / 4, v_slice.len() / 4, out_words);

                    let range = rescale_max - rescale_min;
                    let safe_range = if range.is_finite() && range.abs() > 1e-20 {
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
                        resample_mode: mode.as_u32(),
                        alg_mode,
                        coord_mode,
                        inv_8dx: 1.0 / (8.0 * pixel_size_x_m.max(1e-3)),
                        inv_8dy: 1.0 / (8.0 * pixel_size_y_m.max(1e-3)),
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
                        .write_buffer(&slot.tr_params_buf, 0, bytemuck::bytes_of(&params));
                    self.queue.write_buffer(&slot.u_buf, 0, u_slice);
                    self.queue.write_buffer(&slot.v_buf, 0, v_slice);
                    if !is_terrain_rgb && eff_output == CoalescedOutput::Rgba8 {
                        self.queue
                            .write_buffer(&slot.lut_buf, 0, bytemuck::cast_slice(&lut_u32));
                    }

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("cogzilla-coal-tr-bg"),
                        layout: &self.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.tr_params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slab.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: slot.u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: slot.v_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: slot.lut_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: slot.out_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let pipe_kind = if is_terrain_rgb {
                        6u8
                    } else {
                        match eff_output {
                            CoalescedOutput::Rgba8 => 3u8,
                            CoalescedOutput::IndexedU8 => 4u8,
                            CoalescedOutput::RawF32 => 5u8,
                        }
                    };
                    let staging_byte_off = (total_staging_words * 4) as u64;
                    total_staging_words += out_words;
                    prepared.push(PreparedJob {
                        slot_idx,
                        dst_w,
                        dst_h,
                        words_per_row,
                        out_words,
                        staging_byte_off,
                        output: eff_output,
                        wg_x: (dst_w as u32).div_ceil(16),
                        wg_y: (dst_h as u32).div_ceil(16),
                        wg_z: 1,
                        pipe_kind,
                        bind_group,
                        jpeg_pass: None,
                        stats_pass: None,
                        reply,
                    });
                }
                CoalescedJob::Expr {
                    slab,
                    plan,
                    mode,
                    instrs,
                    rescale_min,
                    rescale_max,
                    lut_u32,
                    output,
                } => {
                    let (dst_w, dst_h, coord_mode, u_slice, v_slice) = match plan.dims_mode_slices()
                    {
                        Ok(t) => t,
                        Err(e) => {
                            reply(Err(e));
                            continue;
                        }
                    };
                    if dst_w == 0 || dst_h == 0 || instrs.is_empty() || instrs.len() > 256 {
                        reply(Err(GpuError::InvalidDimensions(
                            "expr job requires non-zero dims and 1..=256 instrs".into(),
                        )));
                        continue;
                    }
                    let words_per_row = match output {
                        CoalescedOutput::Rgba8 | CoalescedOutput::RawF32 => dst_w,
                        CoalescedOutput::IndexedU8 => dst_w.div_ceil(4),
                    };
                    let out_words = words_per_row * dst_h;
                    let slot = &mut frame.slots[slot_idx];
                    slot.ensure_caps(&self.device, u_slice.len() / 4, v_slice.len() / 4, out_words);

                    let range = rescale_max - rescale_min;
                    let safe_range = if range.is_finite() && range.abs() > 1e-20 {
                        range
                    } else {
                        1.0
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
                        resample_mode: mode.as_u32(),
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
                        .write_buffer(&slot.expr_params_buf, 0, bytemuck::bytes_of(&params));
                    self.queue
                        .write_buffer(&slot.instr_buf, 0, bytemuck::cast_slice(&instrs));
                    if !matches!(output, CoalescedOutput::RawF32) {
                        self.queue
                            .write_buffer(&slot.lut_buf, 0, bytemuck::cast_slice(&lut_u32));
                    }
                    self.queue.write_buffer(&slot.u_buf, 0, u_slice);
                    self.queue.write_buffer(&slot.v_buf, 0, v_slice);

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("coal-expr-bg"),
                        layout: &self.expr_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.expr_params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slab.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: slot.u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: slot.v_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: slot.instr_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: slot.lut_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: slot.out_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let staging_byte_off = (total_staging_words * 4) as u64;
                    total_staging_words += out_words;
                    let (wg_x, pipe_kind) = match output {
                        CoalescedOutput::Rgba8 => ((dst_w as u32).div_ceil(16), 7u8),
                        CoalescedOutput::IndexedU8 => ((words_per_row as u32).div_ceil(16), 8u8),
                        CoalescedOutput::RawF32 => ((dst_w as u32).div_ceil(16), 9u8),
                    };
                    prepared.push(PreparedJob {
                        slot_idx,
                        dst_w,
                        dst_h,
                        words_per_row,
                        out_words,
                        staging_byte_off,
                        output,
                        wg_x,
                        wg_y: (dst_h as u32).div_ceil(16),
                        wg_z: 1,
                        pipe_kind,
                        bind_group,
                        jpeg_pass: None,
                        stats_pass: None,
                        reply,
                    });
                }
                job @ (CoalescedJob::Rgb { .. } | CoalescedJob::SarRgb { .. }) => {
                    let (is_sar, slab, plan, mode, rescale_min, rescale_max, formula) = match job {
                        CoalescedJob::Rgb {
                            slab,
                            plan,
                            mode,
                            rescale_min,
                            rescale_max,
                            formula,
                        } => (false, slab, plan, mode, rescale_min, rescale_max, formula),
                        CoalescedJob::SarRgb {
                            slab,
                            plan,
                            mode,
                            rescale_min,
                            rescale_max,
                            formula,
                        } => (true, slab, plan, mode, rescale_min, rescale_max, formula),
                        _ => unreachable!(),
                    };
                    let (dst_w, dst_h, coord_mode, u_slice, v_slice) = match plan.dims_mode_slices()
                    {
                        Ok(t) => t,
                        Err(e) => {
                            reply(Err(e));
                            continue;
                        }
                    };
                    let min_bands = if is_sar { 2 } else { 3 };
                    if slab.num_bands < min_bands || dst_w == 0 || dst_h == 0 {
                        reply(Err(GpuError::InvalidDimensions(
                            "rgb/sar job requires sufficient bands and non-zero dims".into(),
                        )));
                        continue;
                    }
                    let out_words = dst_w * dst_h;
                    let slot = &mut frame.slots[slot_idx];
                    slot.ensure_caps(&self.device, u_slice.len() / 4, v_slice.len() / 4, out_words);

                    let inv = |min: f32, max: f32| {
                        let d = max - min;
                        255.0 / if d.is_finite() && d.abs() > 1e-20 { d } else { 1.0 }
                    };
                    let mut flags = 0u32;
                    if let Some(lut) = &formula.rgb_lut {
                        flags |= 1;
                        self.queue
                            .write_buffer(&slot.lut_buf, 0, bytemuck::cast_slice(lut));
                    }
                    let sat = if let Some(s) = formula.saturation {
                        flags |= 2;
                        s
                    } else {
                        1.0
                    };
                    let params = ExprParams {
                        src_w: slab.width,
                        src_h: slab.height,
                        src_stride: slab.stride,
                        band_stride: slab.band_stride,
                        dst_w: dst_w as u32,
                        dst_h: dst_h as u32,
                        num_bands: min_bands,
                        num_instrs: 0,
                        resample_mode: mode.as_u32(),
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
                        .write_buffer(&slot.expr_params_buf, 0, bytemuck::bytes_of(&params));
                    self.queue.write_buffer(&slot.u_buf, 0, u_slice);
                    self.queue.write_buffer(&slot.v_buf, 0, v_slice);

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(if is_sar { "coal-sar-rgb-bg" } else { "coal-rgb-bg" }),
                        layout: &self.expr_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.expr_params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slab.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: slot.u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: slot.v_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: slot.instr_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: slot.lut_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: slot.out_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let staging_byte_off = (total_staging_words * 4) as u64;
                    total_staging_words += out_words;
                    prepared.push(PreparedJob {
                        slot_idx,
                        dst_w,
                        dst_h,
                        words_per_row: dst_w,
                        out_words,
                        staging_byte_off,
                        output: CoalescedOutput::Rgba8,
                        wg_x: (dst_w as u32).div_ceil(16),
                        wg_y: (dst_h as u32).div_ceil(16),
                        wg_z: 1,
                        pipe_kind: if is_sar { 11u8 } else { 10u8 },
                        bind_group,
                        jpeg_pass: None,
                        stats_pass: None,
                        reply,
                    });
                }
                CoalescedJob::ExprRgb {
                    slab,
                    plan,
                    mode,
                    instrs,
                    lens,
                    rescale_min,
                    rescale_max,
                    formula,
                } => {
                    let (dst_w, dst_h, coord_mode, u_slice, v_slice) = match plan.dims_mode_slices()
                    {
                        Ok(t) => t,
                        Err(e) => {
                            reply(Err(e));
                            continue;
                        }
                    };
                    let total_len = lens[0] as usize + lens[1] as usize + lens[2] as usize;
                    if dst_w == 0
                        || dst_h == 0
                        || total_len == 0
                        || total_len > 256
                        || total_len != instrs.len()
                    {
                        reply(Err(GpuError::InvalidDimensions(
                            "expr_rgb job requires non-zero dims and valid 1..=256 packed instrs"
                                .into(),
                        )));
                        continue;
                    }
                    let out_words = dst_w * dst_h;
                    let slot = &mut frame.slots[slot_idx];
                    slot.ensure_caps(&self.device, u_slice.len() / 4, v_slice.len() / 4, out_words);

                    let inv = |min: f32, max: f32| {
                        let d = max - min;
                        255.0 / if d.is_finite() && d.abs() > 1e-20 { d } else { 1.0 }
                    };
                    let mut flags = 0u32;
                    if let Some(lut) = &formula.rgb_lut {
                        flags |= 1;
                        self.queue
                            .write_buffer(&slot.lut_buf, 0, bytemuck::cast_slice(lut));
                    }
                    let sat = if let Some(s) = formula.saturation {
                        flags |= 2;
                        s
                    } else {
                        1.0
                    };
                    let packed_lens = (lens[0] as u32)
                        | ((lens[1] as u32) << 8)
                        | ((lens[2] as u32) << 16);
                    let params = ExprParams {
                        src_w: slab.width,
                        src_h: slab.height,
                        src_stride: slab.stride,
                        band_stride: slab.band_stride,
                        dst_w: dst_w as u32,
                        dst_h: dst_h as u32,
                        num_bands: slab.num_bands,
                        num_instrs: packed_lens,
                        resample_mode: mode.as_u32(),
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
                        .write_buffer(&slot.expr_params_buf, 0, bytemuck::bytes_of(&params));
                    self.queue
                        .write_buffer(&slot.instr_buf, 0, bytemuck::cast_slice(&instrs));
                    self.queue.write_buffer(&slot.u_buf, 0, u_slice);
                    self.queue.write_buffer(&slot.v_buf, 0, v_slice);

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("coal-expr-rgb-bg"),
                        layout: &self.expr_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.expr_params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slab.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: slot.u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: slot.v_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: slot.instr_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: slot.lut_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: slot.out_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let staging_byte_off = (total_staging_words * 4) as u64;
                    total_staging_words += out_words;
                    prepared.push(PreparedJob {
                        slot_idx,
                        dst_w,
                        dst_h,
                        words_per_row: dst_w,
                        out_words,
                        staging_byte_off,
                        output: CoalescedOutput::Rgba8,
                        wg_x: (dst_w as u32).div_ceil(16),
                        wg_y: (dst_h as u32).div_ceil(16),
                        wg_z: 1,
                        pipe_kind: 12u8,
                        bind_group,
                        jpeg_pass: None,
                        stats_pass: None,
                        reply,
                    });
                }
                CoalescedJob::MultiBandF32 { slab, plan, mode } => {
                    let (dst_w, dst_h, coord_mode, u_slice, v_slice) = match plan.dims_mode_slices()
                    {
                        Ok(t) => t,
                        Err(e) => {
                            reply(Err(e));
                            continue;
                        }
                    };
                    if dst_w == 0 || dst_h == 0 || slab.num_bands == 0 || slab.num_bands > 8 {
                        reply(Err(GpuError::InvalidDimensions(
                            "multiband_f32 job requires non-zero dims and 1..=8 bands".into(),
                        )));
                        continue;
                    }
                    let out_words = dst_w * dst_h * (slab.num_bands as usize);
                    let slot = &mut frame.slots[slot_idx];
                    slot.ensure_caps(&self.device, u_slice.len() / 4, v_slice.len() / 4, out_words);

                    let params = ExprParams {
                        src_w: slab.width,
                        src_h: slab.height,
                        src_stride: slab.stride,
                        band_stride: slab.band_stride,
                        dst_w: dst_w as u32,
                        dst_h: dst_h as u32,
                        num_bands: slab.num_bands,
                        num_instrs: 0,
                        resample_mode: mode.as_u32(),
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
                        .write_buffer(&slot.expr_params_buf, 0, bytemuck::bytes_of(&params));
                    self.queue.write_buffer(&slot.u_buf, 0, u_slice);
                    self.queue.write_buffer(&slot.v_buf, 0, v_slice);

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("coal-mb-f32-bg"),
                        layout: &self.expr_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.expr_params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slab.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: slot.u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: slot.v_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: slot.instr_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: slot.lut_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: slot.out_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let staging_byte_off = (total_staging_words * 4) as u64;
                    total_staging_words += out_words;
                    prepared.push(PreparedJob {
                        slot_idx,
                        dst_w,
                        dst_h,
                        words_per_row: dst_w,
                        out_words,
                        staging_byte_off,
                        output: CoalescedOutput::RawF32,
                        wg_x: (dst_w as u32).div_ceil(16),
                        wg_y: (dst_h as u32).div_ceil(16),
                        wg_z: slab.num_bands,
                        pipe_kind: 13u8,
                        bind_group,
                        jpeg_pass: None,
                        stats_pass: None,
                        reply,
                    });
                }
                CoalescedJob::ExprRgbF32 {
                    slab,
                    plan,
                    mode,
                    instrs,
                    lens,
                } => {
                    let (dst_w, dst_h, coord_mode, u_slice, v_slice) = match plan.dims_mode_slices()
                    {
                        Ok(t) => t,
                        Err(e) => {
                            reply(Err(e));
                            continue;
                        }
                    };
                    let total_len = lens[0] as usize + lens[1] as usize + lens[2] as usize;
                    if dst_w == 0
                        || dst_h == 0
                        || total_len == 0
                        || total_len > 256
                        || total_len != instrs.len()
                    {
                        reply(Err(GpuError::InvalidDimensions(
                            "expr_rgb_f32 job requires non-zero dims and valid 1..=256 packed instrs"
                                .into(),
                        )));
                        continue;
                    }
                    let out_words = dst_w * dst_h * 3;
                    let slot = &mut frame.slots[slot_idx];
                    slot.ensure_caps(&self.device, u_slice.len() / 4, v_slice.len() / 4, out_words);

                    let packed_lens = (lens[0] as u32)
                        | ((lens[1] as u32) << 8)
                        | ((lens[2] as u32) << 16);
                    let params = ExprParams {
                        src_w: slab.width,
                        src_h: slab.height,
                        src_stride: slab.stride,
                        band_stride: slab.band_stride,
                        dst_w: dst_w as u32,
                        dst_h: dst_h as u32,
                        num_bands: slab.num_bands,
                        num_instrs: packed_lens,
                        resample_mode: mode.as_u32(),
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
                        .write_buffer(&slot.expr_params_buf, 0, bytemuck::bytes_of(&params));
                    self.queue
                        .write_buffer(&slot.instr_buf, 0, bytemuck::cast_slice(&instrs));
                    self.queue.write_buffer(&slot.u_buf, 0, u_slice);
                    self.queue.write_buffer(&slot.v_buf, 0, v_slice);

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("coal-expr-rgb-f32-bg"),
                        layout: &self.expr_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.expr_params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slab.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: slot.u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: slot.v_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: slot.instr_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: slot.lut_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: slot.out_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let staging_byte_off = (total_staging_words * 4) as u64;
                    total_staging_words += out_words;
                    prepared.push(PreparedJob {
                        slot_idx,
                        dst_w,
                        dst_h,
                        words_per_row: dst_w,
                        out_words,
                        staging_byte_off,
                        output: CoalescedOutput::RawF32,
                        wg_x: (dst_w as u32).div_ceil(16),
                        wg_y: (dst_h as u32).div_ceil(16),
                        wg_z: 1,
                        pipe_kind: 14u8,
                        bind_group,
                        jpeg_pass: None,
                        stats_pass: None,
                        reply,
                    });
                }
                CoalescedJob::WithJpeg { .. } | CoalescedJob::WithStats { .. } => {
                    reply(Err(GpuError::InvalidDimensions(
                        "nested WithJpeg/WithStats is not supported".into(),
                    )));
                    continue;
                }
            }
            if let Some(quality) = jpeg_quality {
                if prepared.len() > prev_len {
                    if prepared.last().unwrap().output != CoalescedOutput::Rgba8 {
                        let bad = prepared.pop().unwrap();
                        total_staging_words -= bad.out_words;
                        (bad.reply)(Err(GpuError::InvalidDimensions(
                            "WithJpeg requires an inner Rgba8 job".into(),
                        )));
                        continue;
                    }
                    let pj = prepared.last_mut().unwrap();
                    total_staging_words -= pj.out_words;
                    let mcu_w = (pj.dst_w as u32).div_ceil(16);
                    let mcu_h = (pj.dst_h as u32).div_ceil(16);
                    let jpeg_words = (mcu_w as usize) * (mcu_h as usize) * 192;
                    let slot = &mut frame.slots[pj.slot_idx];
                    slot.ensure_jpeg_caps(&self.device, jpeg_words);
                    let (_q_y, _q_c, inv_q) = crate::jpeg::build_inv_qtables(quality);
                    let params = crate::jpeg::JpegParams {
                        width: pj.dst_w as u32,
                        height: pj.dst_h as u32,
                        mcu_w,
                        mcu_h,
                    };
                    self.queue.write_buffer(
                        &slot.jpeg_params_buf,
                        0,
                        bytemuck::bytes_of(&params),
                    );
                    self.queue.write_buffer(
                        &slot.jpeg_inv_q_buf,
                        0,
                        bytemuck::cast_slice(&inv_q),
                    );
                    let jpeg_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("cogzilla-coal-jpeg-bg"),
                        layout: &self.jpeg_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.jpeg_params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slot.out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: slot.jpeg_inv_q_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: slot
                                    .jpeg_out_buf
                                    .as_ref()
                                    .unwrap()
                                    .as_entire_binding(),
                            },
                        ],
                    });
                    pj.out_words = jpeg_words;
                    pj.output = CoalescedOutput::RawF32;
                    pj.jpeg_pass = Some((mcu_w, mcu_h, jpeg_bg));
                    total_staging_words += jpeg_words;
                }
            }
            if let Some((nodata, num_bins, hist_range)) = stats_cfg {
                if prepared.len() > prev_len {
                    let valid_inner = {
                        let last = prepared.last().unwrap();
                        last.output == CoalescedOutput::RawF32
                            && last.wg_z == 1
                            && last.out_words == last.dst_w.saturating_mul(last.dst_h)
                    };
                    if !valid_inner {
                        let bad = prepared.pop().unwrap();
                        total_staging_words -= bad.out_words;
                        (bad.reply)(Err(GpuError::InvalidDimensions(
                            "WithStats requires an inner single-band RawF32 job".into(),
                        )));
                        continue;
                    }
                    let pj = prepared.last_mut().unwrap();
                    let len = pj.dst_w * pj.dst_h;
                    let len_u32 = len as u32;
                    let num_wg = len_u32.div_ceil(256 * 16).clamp(1, crate::stats::MAX_WG);
                    let n_bins = num_bins.clamp(1, crate::stats::CDF_BINS) as u32;
                    let (hr_min, hr_max) = hist_range.unwrap_or((0.0, 0.0));
                    let params = crate::stats::StatsParams {
                        len: len_u32,
                        num_wg,
                        has_nodata: u32::from(nodata.is_some()),
                        num_bins: n_bins,
                        has_hist_range: u32::from(hist_range.is_some()),
                        nodata_val: nodata.unwrap_or(0.0),
                        h_min: hr_min as f32,
                        h_max: hr_max as f32,
                    };
                    let slot = &mut frame.slots[pj.slot_idx];
                    slot.ensure_stats_slot(&self.device);
                    let sb = slot.stats_slot.as_ref().unwrap();
                    self.queue
                        .write_buffer(&sb.params_buf, 0, bytemuck::bytes_of(&params));
                    let stats_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("cogzilla-coal-stats-bg"),
                        layout: &self.stats_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: sb.params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slot.out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: sb.partials_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: sb.resolved_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: sb.hist_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: sb.var_buf.as_entire_binding(),
                            },
                        ],
                    });
                    total_staging_words -= pj.out_words;
                    let rb_words = std::mem::size_of::<crate::stats::ResolvedStats>() / 4;
                    let hb_words = 2 * crate::stats::CDF_BINS;
                    let stats_words = rb_words + hb_words + (num_wg as usize);
                    pj.out_words = stats_words;
                    pj.output = CoalescedOutput::RawF32;
                    pj.stats_pass = Some((num_wg, stats_bg));
                    total_staging_words += stats_words;
                }
            }
        }

        if prepared.is_empty() {
            return false;
        }

        if total_staging_words > frame.staging_words_cap {
            let max_words = (self.device.limits().max_buffer_size / 4) as usize;
            frame.staging_words_cap = total_staging_words
                .next_power_of_two()
                .min(max_words)
                .max(total_staging_words);
            frame.staging_buf = Self::alloc_staging(&self.device, frame.staging_words_cap);
        }

        // Encode ALL heterogeneous jobs into ONE CommandEncoder -> ONE non-blocking submit!
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-coal-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-coal-pass"),
                timestamp_writes: None,
            });
            for job in &prepared {
                let pipe = match job.pipe_kind {
                    0 => &self.cm_rgba8,
                    1 => &self.cm_indexed,
                    2 => &self.cm_f32,
                    3 => &self.tr_rgba8,
                    4 => &self.tr_indexed,
                    5 => &self.tr_f32,
                    6 => &self.tr_rgb,
                    7 => &self.expr_rgba8,
                    8 => &self.expr_indexed,
                    9 => &self.expr_f32,
                    10 => &self.rgb_formula,
                    11 => &self.sar_rgb_formula,
                    12 => &self.expr_rgb_formula,
                    13 => &self.mb_resample_f32,
                    _ => &self.expr_rgb_f32,
                };
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &job.bind_group, &[]);
                pass.dispatch_workgroups(job.wg_x, job.wg_y, job.wg_z);
            }
        }
        // Pass 2 (fused VRAM-only): chain jpeg_dct.wgsl directly onto slot.out_buf -> slot.jpeg_out_buf
        if prepared.iter().any(|j| j.jpeg_pass.is_some()) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-coal-jpeg-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.jpeg_fdct);
            for job in &prepared {
                if let Some((mcu_w, mcu_h, ref bg)) = job.jpeg_pass {
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(mcu_w, mcu_h, 1);
                }
            }
        }
        // Pass 2b (fused VRAM-only): chain 3-stage stats.wgsl directly onto slot.out_buf
        if prepared.iter().any(|j| j.stats_pass.is_some()) {
            for job in &prepared {
                if job.stats_pass.is_some() {
                    let sb = frame.slots[job.slot_idx]
                        .stats_slot
                        .as_ref()
                        .expect("stats_slot allocated");
                    encoder.clear_buffer(&sb.hist_buf, 0, None);
                }
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cogzilla-coal-stats-pass1"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.stats_pass1);
                for job in &prepared {
                    if let Some((num_wg, ref bg)) = job.stats_pass {
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(num_wg, 1, 1);
                    }
                }
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cogzilla-coal-stats-finalize1"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.stats_finalize1);
                for job in &prepared {
                    if let Some((_, ref bg)) = job.stats_pass {
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                }
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cogzilla-coal-stats-pass2"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.stats_pass2);
                for job in &prepared {
                    if let Some((num_wg, ref bg)) = job.stats_pass {
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(num_wg, 1, 1);
                    }
                }
            }
        }
        for job in &prepared {
            if let Some((num_wg, _)) = job.stats_pass {
                let sb = frame.slots[job.slot_idx]
                    .stats_slot
                    .as_ref()
                    .expect("stats_slot allocated");
                let rb = std::mem::size_of::<crate::stats::ResolvedStats>() as u64;
                let hb = (2 * crate::stats::CDF_BINS * std::mem::size_of::<u32>()) as u64;
                let vb = (num_wg as usize * std::mem::size_of::<f32>()) as u64;
                encoder.copy_buffer_to_buffer(
                    &sb.resolved_buf,
                    0,
                    &frame.staging_buf,
                    job.staging_byte_off,
                    rb,
                );
                encoder.copy_buffer_to_buffer(
                    &sb.hist_buf,
                    0,
                    &frame.staging_buf,
                    job.staging_byte_off + rb,
                    hb,
                );
                encoder.copy_buffer_to_buffer(
                    &sb.var_buf,
                    0,
                    &frame.staging_buf,
                    job.staging_byte_off + rb + hb,
                    vb,
                );
                continue;
            }
            let bytes = (job.out_words * 4) as u64;
            let src_buf = if job.jpeg_pass.is_some() {
                frame.slots[job.slot_idx]
                    .jpeg_out_buf
                    .as_ref()
                    .expect("jpeg_out_buf allocated")
            } else {
                &frame.slots[job.slot_idx].out_buf
            };
            encoder.copy_buffer_to_buffer(
                src_buf,
                0,
                &frame.staging_buf,
                job.staging_byte_off,
                bytes,
            );
        }

        let sub_idx = self.queue.submit(Some(encoder.finish()));

        let total_bytes = (total_staging_words * 4) as u64;
        let slice = frame.staging_buf.slice(..total_bytes);
        let (tx, map_rx) = mpsc::channel();
        let ready = Arc::new(AtomicBool::new(false));
        let ready_cb = Arc::clone(&ready);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
            ready_cb.store(true, Ordering::Release);
        });
        frame.in_flight = Some(InFlightBatch {
            prepared,
            total_bytes,
            sub_idx,
            map_rx,
            ready,
        });
        true
    }

    /// Drain completed in-flight ring frames. When `wait_oldest` is true and no frame is ready yet,
    /// blocks ONLY until the oldest in-flight submission completes (`WaitForSubmissionIndex`),
    /// allowing newer ring frames to keep executing on the GPU without a pipeline bubble.
    fn drain_completed(&mut self, wait_oldest: bool) -> usize {
        if self.in_flight_order.is_empty() {
            return 0;
        }
        self.device.poll(wgpu::Maintain::Poll);
        let mut drained = 0usize;
        loop {
            let Some(&oldest_idx) = self.in_flight_order.front() else {
                break;
            };
            let is_ready = self.frames[oldest_idx]
                .in_flight
                .as_ref()
                .map(|b| b.ready.load(Ordering::Acquire))
                .unwrap_or(true);
            if !is_ready {
                if drained == 0 && wait_oldest {
                    let sub_idx = self.frames[oldest_idx]
                        .in_flight
                        .as_ref()
                        .map(|b| b.sub_idx.clone());
                    if let Some(idx) = sub_idx {
                        self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(idx));
                        // Ensure map_async callback has fired
                        if !self.frames[oldest_idx]
                            .in_flight
                            .as_ref()
                            .map(|b| b.ready.load(Ordering::Acquire))
                            .unwrap_or(true)
                        {
                            self.device.poll(wgpu::Maintain::Wait);
                        }
                    }
                } else {
                    break;
                }
            }
            self.in_flight_order.pop_front();
            let frame = &mut self.frames[oldest_idx];
            if let Some(batch) = frame.in_flight.take() {
                let map_res = batch
                    .map_rx
                    .recv()
                    .map_err(|e| GpuError::BufferMap(e.to_string()))
                    .and_then(|r| r.map_err(|e| GpuError::BufferMap(e.to_string())));
                match map_res {
                    Err(e) => {
                        let msg = e.to_string();
                        for job in batch.prepared {
                            (job.reply)(Err(GpuError::BufferMap(msg.clone())));
                        }
                    }
                    Ok(()) => {
                        let slice = frame.staging_buf.slice(..batch.total_bytes);
                        let mapped = slice.get_mapped_range();
                        for job in batch.prepared {
                            let start = job.staging_byte_off as usize;
                            let len = job.out_words * 4;
                            let bytes = &mapped[start..start + len];
                            let out = match job.output {
                                CoalescedOutput::Rgba8 | CoalescedOutput::RawF32 => bytes.to_vec(),
                                CoalescedOutput::IndexedU8 => {
                                    if job.dst_w % 4 == 0 {
                                        bytes.to_vec()
                                    } else {
                                        let row_bytes = job.words_per_row * 4;
                                        let mut unpacked =
                                            Vec::with_capacity(job.dst_w * job.dst_h);
                                        for py in 0..job.dst_h {
                                            let rs = py * row_bytes;
                                            unpacked
                                                .extend_from_slice(&bytes[rs..rs + job.dst_w]);
                                        }
                                        unpacked
                                    }
                                }
                            };
                            (job.reply)(Ok(out));
                        }
                        drop(mapped);
                        frame.staging_buf.unmap();
                    }
                }
            }
            self.free_frames.push_back(oldest_idx);
            drained += 1;
        }
        drained
    }
}

/// Thread-safe heterogeneous GPU command-buffer micro-batcher.
///
/// Concurrent callers submit independent `CoalescedJob`s and await their individual results;
/// the background GPU worker packs whatever jobs arrive concurrently into a single command
/// encoder, single `queue.submit()`, and single `device.poll(Wait)`.
pub struct GpuCoalescer {
    tx: Option<mpsc::Sender<PendingRequest>>,
    worker: Option<JoinHandle<()>>,
    batches_submitted: Arc<AtomicU64>,
    jobs_executed: Arc<AtomicU64>,
    max_batch_seen: Arc<AtomicUsize>,
    pending_count: Arc<AtomicUsize>,
    max_pending: usize,
    max_frame_staging_bytes: u64,
    max_slot_storage_bytes: u64,
}

impl GpuCoalescer {
    /// Spawn a new heterogeneous GPU coalescer for `ctx`.
    ///
    /// `coalesce_window_us` controls the maximum time the worker waits to scoop concurrent burst
    /// arrivals before submitting (overridable via `COGZILLA_GPU_COALESCE_US`). When idle
    /// (`pending_count == 0` and no recent burst), isolated requests dispatch with **0 µs** wait.
    pub fn new(ctx: &GpuContext, max_batch: usize, coalesce_window_us: u64) -> Self {
        let max_batch = max_batch.clamp(1, 64);
        let env_us = std::env::var("COGZILLA_GPU_COALESCE_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok());
        let window = Duration::from_micros(env_us.unwrap_or(coalesce_window_us));
        let default_depth = if ctx.info.unified_memory { 2 } else { 3 };
        let ring_depth = std::env::var("COGZILLA_GPU_RING_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_depth)
            .clamp(1, 4);

        let max_pending = std::env::var("COGZILLA_GPU_MAX_QUEUED")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096)
            .max(max_batch);
        let limits = ctx.device.limits();
        let max_slot_storage_bytes =
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        // Cap per-batch staging buffer at half of max_buffer_size (max 256 MiB) so coalescing
        // multiple large 4096² bboxes can never exceed `Device::limits().max_buffer_size`.
        let max_frame_staging_bytes = (limits.max_buffer_size / 2).min(256 * 1024 * 1024);

        let (tx, rx) = mpsc::channel::<PendingRequest>();
        let batches_submitted = Arc::new(AtomicU64::new(0));
        let jobs_executed = Arc::new(AtomicU64::new(0));
        let max_batch_seen = Arc::new(AtomicUsize::new(0));
        let pending_count = Arc::new(AtomicUsize::new(0));

        let b_stat = Arc::clone(&batches_submitted);
        let j_stat = Arc::clone(&jobs_executed);
        let m_stat = Arc::clone(&max_batch_seen);
        let p_stat = Arc::clone(&pending_count);

        let mut state = WorkerState::new(ctx, ring_depth);
        let worker = std::thread::Builder::new()
            .name("cogzilla-gpu-coalescer".into())
            .spawn(move || {
                let mut batch: Vec<PendingRequest> = Vec::with_capacity(max_batch);
                let mut deferred: Option<PendingRequest> = None;
                let mut in_burst = false;
                let mut disconnected = false;
                loop {
                    // 1. Always harvest any GPU frames that have already finished (non-blocking).
                    state.drain_completed(false);

                    // 2. If all ring frames are currently in flight, wait ONLY for the oldest
                    //    submission so the other (N-1) frames keep executing on the GPU.
                    if state.free_frames.is_empty() {
                        state.drain_completed(true);
                        continue;
                    }

                    let mut batch_staging_bytes: u64 = 0;
                    // Returns `true` if `req` was added (or discarded as cancelled), `false` if
                    // adding it would exceed `max_frame_staging_bytes` (so caller defers it).
                    let try_push_if_live = |batch: &mut Vec<PendingRequest>,
                                            batch_bytes: &mut u64,
                                            deferred: &mut Option<PendingRequest>,
                                            req: PendingRequest|
                     -> bool {
                        if req.is_cancelled() {
                            (req.reply)(Err(GpuError::BufferMap("cancelled".into())));
                            return true;
                        }
                        let (req_bytes, _) = req.job.estimated_staging_and_slot_bytes();
                        if !batch.is_empty()
                            && batch_bytes.saturating_add(req_bytes) > max_frame_staging_bytes
                        {
                            *deferred = Some(req);
                            return false;
                        }
                        *batch_bytes = batch_bytes.saturating_add(req_bytes);
                        batch.push(req);
                        true
                    };

                    // 3. Acquire the next live job (first from `deferred`, then from `rx`).
                    batch.clear();
                    if let Some(prev) = deferred.take() {
                        let _ = try_push_if_live(
                            &mut batch,
                            &mut batch_staging_bytes,
                            &mut deferred,
                            prev,
                        );
                    } else if state.in_flight_order.is_empty() {
                        if disconnected {
                            break;
                        }
                        match rx.recv() {
                            Ok(first) => {
                                let _ = try_push_if_live(
                                    &mut batch,
                                    &mut batch_staging_bytes,
                                    &mut deferred,
                                    first,
                                );
                            }
                            Err(_) => break,
                        }
                    } else {
                        match rx.try_recv() {
                            Ok(first) => {
                                let _ = try_push_if_live(
                                    &mut batch,
                                    &mut batch_staging_bytes,
                                    &mut deferred,
                                    first,
                                );
                            }
                            Err(mpsc::TryRecvError::Empty) => {
                                // No new job ready yet; wait for oldest in-flight frame to finish.
                                state.drain_completed(true);
                                continue;
                            }
                            Err(mpsc::TryRecvError::Disconnected) => {
                                disconnected = true;
                                state.drain_completed(true);
                                continue;
                            }
                        }
                    }

                    // 4. Immediately scoop any other live jobs already sitting in the channel,
                    //    stopping if adding the next job would overflow `max_frame_staging_bytes`.
                    while batch.len() < max_batch && deferred.is_none() {
                        match rx.try_recv() {
                            Ok(next) => {
                                if !try_push_if_live(
                                    &mut batch,
                                    &mut batch_staging_bytes,
                                    &mut deferred,
                                    next,
                                ) {
                                    break;
                                }
                            }
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => {
                                disconnected = true;
                                break;
                            }
                        }
                    }

                    // If every drained request had already been cancelled by disconnected clients,
                    // loop without burning a GPU submit.
                    if batch.is_empty() {
                        continue;
                    }

                    // 5. Adaptive coalescing window: only wait briefly if the batch isn't full and
                    //    concurrent traffic is active (and never wait past a completed frame).
                    let active_waiters = p_stat.load(Ordering::Acquire);
                    if !disconnected
                        && deferred.is_none()
                        && !window.is_zero()
                        && batch.len() < max_batch
                        && batch_staging_bytes < max_frame_staging_bytes
                        && (batch.len() > 1 || active_waiters > 1 || in_burst)
                    {
                        let deadline = Instant::now() + window;
                        while batch.len() < max_batch && deferred.is_none() {
                            if !state.in_flight_order.is_empty() && state.drain_completed(false) > 0
                            {
                                // Harvest completed replies right away so Tokio PNG encoders start!
                            }
                            let now = Instant::now();
                            if now >= deadline {
                                break;
                            }
                            let slice_wait = (deadline - now).min(Duration::from_micros(100));
                            match rx.recv_timeout(slice_wait) {
                                Ok(next) => {
                                    if !try_push_if_live(
                                        &mut batch,
                                        &mut batch_staging_bytes,
                                        &mut deferred,
                                        next,
                                    ) {
                                        break;
                                    }
                                }
                                Err(mpsc::RecvTimeoutError::Timeout) => {
                                    if Instant::now() >= deadline {
                                        break;
                                    }
                                }
                                Err(mpsc::RecvTimeoutError::Disconnected) => {
                                    disconnected = true;
                                    break;
                                }
                            }
                        }
                    }

                    let k = batch.len();
                    in_burst = k > 1;
                    b_stat.fetch_add(1, Ordering::Relaxed);
                    j_stat.fetch_add(k as u64, Ordering::Relaxed);
                    m_stat.fetch_max(k, Ordering::Relaxed);

                    let frame_idx = state.free_frames.pop_front().expect("free frame checked");
                    let to_run = std::mem::take(&mut batch);
                    if state.submit_batch(frame_idx, to_run) {
                        state.in_flight_order.push_back(frame_idx);
                    } else {
                        state.free_frames.push_back(frame_idx);
                    }
                    // For ring_depth == 1 (synchronous A/B baseline), complete immediately.
                    if ring_depth == 1 {
                        state.drain_completed(true);
                    }
                }
                while !state.in_flight_order.is_empty() {
                    state.drain_completed(true);
                }
            })
            .expect("failed to spawn cogzilla-gpu-coalescer thread");

        Self {
            tx: Some(tx),
            worker: Some(worker),
            batches_submitted,
            jobs_executed,
            max_batch_seen,
            pending_count,
            max_pending,
            max_frame_staging_bytes,
            max_slot_storage_bytes,
        }
    }

    /// Submit a job asynchronously with a completion callback (non-blocking — ideal for
    /// Tokio `oneshot::Sender` so async HTTP handlers never block worker threads while
    /// waiting for the coalesced GPU command buffer).
    pub fn submit_with_callback(
        &self,
        job: CoalescedJob,
        on_done: impl FnOnce(Result<Vec<u8>, GpuError>) + Send + 'static,
    ) -> Result<(), GpuError> {
        self.submit_inner(job, None, on_done)
    }

    /// Submit a cancellable job asynchronously. If `cancelled` becomes `true` before the worker
    /// encodes the job into a GPU command buffer (e.g. an HTTP client disconnected or timed out),
    /// the job is discarded immediately without consuming GPU work.
    pub fn submit_cancellable(
        &self,
        job: CoalescedJob,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
        on_done: impl FnOnce(Result<Vec<u8>, GpuError>) + Send + 'static,
    ) -> Result<(), GpuError> {
        self.submit_inner(job, Some(cancelled), on_done)
    }

    fn submit_inner(
        &self,
        job: CoalescedJob,
        cancelled: Option<Arc<std::sync::atomic::AtomicBool>>,
        on_done: impl FnOnce(Result<Vec<u8>, GpuError>) + Send + 'static,
    ) -> Result<(), GpuError> {
        let Some(ref tx) = self.tx else {
            return Err(GpuError::BufferMap("coalescer shut down".into()));
        };
        let (staging_bytes, slot_bytes) = job.estimated_staging_and_slot_bytes();
        if staging_bytes > self.max_frame_staging_bytes || slot_bytes > self.max_slot_storage_bytes
        {
            return Err(GpuError::InvalidDimensions(format!(
                "coalesced job output ({staging_bytes} B staging / {slot_bytes} B slot) exceeds adapter buffer limit"
            )));
        }
        if self.pending_count.load(Ordering::Relaxed) >= self.max_pending {
            return Err(GpuError::BufferMap("coalescer queue full".into()));
        }
        let pending = Arc::clone(&self.pending_count);
        pending.fetch_add(1, Ordering::AcqRel);
        let send_res = tx.send(PendingRequest {
            job,
            reply: Box::new(move |res| {
                pending.fetch_sub(1, Ordering::AcqRel);
                on_done(res);
            }),
            cancelled,
        });
        if send_res.is_err() {
            self.pending_count.fetch_sub(1, Ordering::AcqRel);
            return Err(GpuError::BufferMap("coalescer worker exited".into()));
        }
        Ok(())
    }

    /// Submit a job and wait synchronously for its result. Concurrent callers across threads
    /// are automatically packed into shared GPU command buffers.
    pub fn submit(&self, job: CoalescedJob) -> Result<Vec<u8>, GpuError> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.submit_with_callback(job, move |res| {
            let _ = reply_tx.send(res);
        })?;
        reply_rx
            .recv()
            .map_err(|e| GpuError::BufferMap(format!("coalescer reply dropped: {e}")))?
    }

    pub fn stats(&self) -> CoalescerStats {
        CoalescerStats {
            batches_submitted: self.batches_submitted.load(Ordering::Relaxed),
            jobs_executed: self.jobs_executed.load(Ordering::Relaxed),
            max_batch_seen: self.max_batch_seen.load(Ordering::Relaxed),
        }
    }
}

impl Drop for GpuCoalescer {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colormap::{GpuColormapRenderer, GpuCoordPlan};
    use crate::resample::GpuResampler;
    use crate::terrain::GpuTerrainRenderer;

    #[test]
    fn heterogeneous_concurrent_burst_coalesces_and_matches_serial() {
        let Ok(ctx) = GpuContext::new_default() else {
            eprintln!("skipping GPU coalescer test (no adapter)");
            return;
        };
        let resampler = GpuResampler::new(&ctx);
        let mut cm_ref = GpuColormapRenderer::new(&ctx);
        let mut tr_ref = GpuTerrainRenderer::new(&ctx);

        // Two distinct slabs (simulating 2 different COGs)
        let make_slab = |phase: f32| {
            let w = 256usize;
            let h = 256usize;
            let mut data = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    data[y * w + x] = 500.0
                        + 300.0 * (x as f32 * 0.04 + phase).sin()
                        + 200.0 * (y as f32 * 0.05 - phase).cos();
                }
            }
            Arc::new(resampler.upload_slab(&data, w as u32, h as u32, w as u32).unwrap())
        };
        let slab_a = make_slab(0.0);
        let slab_b = make_slab(1.3);

        let n = 128usize;
        let u: Vec<f32> = (0..n).map(|i| 8.0 + (i as f32 / n as f32) * 238.0).collect();
        let v: Vec<f32> = (0..n).map(|i| 8.0 + (i as f32 / n as f32) * 238.0).collect();
        let lut = cogzilla_kernel::ColormapName::Viridis.lut();
        let lut_u32 = GpuColormapRenderer::pack_rgb_lut(&lut.rgb);

        // Serial ground truth for 3 distinct job types
        let ref_cm = cm_ref
            .render_slab(
                &slab_a,
                &u,
                &v,
                GpuColormapMode::Lanczos3,
                0.0,
                1200.0,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();
        let ref_f32 = cm_ref
            .render_slab_f32_with_plan(
                &slab_b,
                GpuCoordPlan::Separable1D {
                    u_coords: &u,
                    v_coords: &v,
                },
                GpuColormapMode::Lanczos3,
            )
            .unwrap();
        let ref_dem = tr_ref
            .render_slab(
                &slab_a,
                &u,
                &v,
                GpuColormapMode::Lanczos3,
                GpuDemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 1.0,
                },
                30.0,
                30.0,
                0.0,
                255.0,
                &lut_u32,
                GpuPixelOutput::Rgba8,
            )
            .unwrap();

        let coalescer = Arc::new(GpuCoalescer::new(&ctx, 32, 1000));
        let barrier = Arc::new(std::sync::Barrier::new(12));
        let mut handles = Vec::new();
        for i in 0..12usize {
            let c = Arc::clone(&coalescer);
            let b = Arc::clone(&barrier);
            let sa = Arc::clone(&slab_a);
            let sb = Arc::clone(&slab_b);
            let uu = u.clone();
            let vv = v.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                let job = match i % 3 {
                    0 => CoalescedJob::Colormap {
                        slab: sa,
                        plan: OwnedCoordPlan::Separable1D { u: uu, v: vv },
                        mode: GpuColormapMode::Lanczos3,
                        rescale_min: 0.0,
                        rescale_max: 1200.0,
                        lut_u32,
                        output: CoalescedOutput::IndexedU8,
                    },
                    1 => CoalescedJob::Colormap {
                        slab: sb,
                        plan: OwnedCoordPlan::Separable1D { u: uu, v: vv },
                        mode: GpuColormapMode::Lanczos3,
                        rescale_min: 0.0,
                        rescale_max: 1200.0,
                        lut_u32,
                        output: CoalescedOutput::RawF32,
                    },
                    _ => CoalescedJob::Terrain {
                        slab: sa,
                        plan: OwnedCoordPlan::Separable1D { u: uu, v: vv },
                        mode: GpuColormapMode::Lanczos3,
                        alg: GpuDemAlgorithm::Hillshade {
                            azimuth_deg: 315.0,
                            altitude_deg: 45.0,
                            z_factor: 1.0,
                        },
                        pixel_size_x_m: 30.0,
                        pixel_size_y_m: 30.0,
                        rescale_min: 0.0,
                        rescale_max: 255.0,
                        lut_u32,
                        output: CoalescedOutput::Rgba8,
                    },
                };
                (i % 3, c.submit(job).unwrap())
            }));
        }

        for h in handles {
            let (kind, out) = h.join().unwrap();
            match kind {
                0 => assert_eq!(out, ref_cm, "coalesced IndexedU8 must be bit-identical"),
                1 => assert_eq!(out, ref_f32, "coalesced RawF32 must be bit-identical"),
                _ => assert_eq!(out, ref_dem, "coalesced DEM Hillshade must be bit-identical"),
            }
        }
        let st = coalescer.stats();
        assert_eq!(st.jobs_executed, 12);
        assert!(
            st.batches_submitted < 12 && st.max_batch_seen >= 2,
            "expected coalescing across 12 concurrent threads, got {st:?}"
        );
    }

    #[test]
    fn five_family_multiband_burst_matches_serial() {
        let Ok(ctx) = GpuContext::new(crate::context::GpuOptions::default()) else {
            eprintln!("skipping five_family_multiband_burst_matches_serial (no GPU adapter)");
            return;
        };
        let resampler = crate::resample::GpuResampler::new(&ctx);
        let mut cm_ref = crate::colormap::GpuColormapRenderer::new(&ctx);
        let mut tr_ref = crate::terrain::GpuTerrainRenderer::new(&ctx);
        let mut ex_ref = crate::expr::GpuExprRenderer::new(&ctx);

        let w = 128u32;
        let h = 128u32;
        let b0: Vec<f32> = (0..(w * h))
            .map(|i| 0.05 + ((i % w) as f32 * 0.008) + ((i / w) as f32 * 0.004))
            .collect();
        let b1: Vec<f32> = (0..(w * h))
            .map(|i| 0.02 + ((i % w) as f32 * 0.003) + ((i / w) as f32 * 0.006))
            .collect();
        let b2: Vec<f32> = (0..(w * h))
            .map(|i| 0.01 + (((i % w) ^ (i / w)) as f32 * 0.005))
            .collect();

        let single_slab = Arc::new(resampler.upload_slab(&b0, w, h, w).unwrap());
        let mb3_slab = Arc::new(
            ex_ref
                .upload_bands(&[b0.as_slice(), b1.as_slice(), b2.as_slice()], w, h, w)
                .unwrap(),
        );
        let sar2_slab = Arc::new(
            ex_ref
                .upload_bands(&[b0.as_slice(), b1.as_slice()], w, h, w)
                .unwrap(),
        );

        let n = 64usize;
        let u: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * 1.8).collect();
        let v: Vec<f32> = (0..n).map(|i| 4.0 + i as f32 * 1.8).collect();
        let mut lut_rgb = [[0u8; 3]; 256];
        for i in 0..256 {
            lut_rgb[i] = [i as u8, (255 - i) as u8, ((i * 3) & 255) as u8];
        }
        let lut_u32 = crate::colormap::GpuColormapRenderer::pack_rgb_lut(&lut_rgb);

        // NDVI = (b0 - b1) / (b0 + b1)
        let ndvi_instrs = vec![
            GpuInstr::load_band(0),
            GpuInstr::load_band(1),
            GpuInstr::op(crate::expr::opcodes::SUB),
            GpuInstr::load_band(0),
            GpuInstr::load_band(1),
            GpuInstr::op(crate::expr::opcodes::ADD),
            GpuInstr::op(crate::expr::opcodes::DIV),
        ];
        let formula = GpuColorFormula::from_gamma_sigmoidal_sat(Some(1.2), Some((6.0, 0.5)), Some(1.3));

        let ref_cm = cm_ref
            .render_slab(
                &single_slab,
                &u,
                &v,
                GpuColormapMode::Lanczos3,
                0.0,
                2.0,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();
        let ref_dem = tr_ref
            .render_slab(
                &single_slab,
                &u,
                &v,
                GpuColormapMode::Cubic,
                GpuDemAlgorithm::Hillshade {
                    azimuth_deg: 315.0,
                    altitude_deg: 45.0,
                    z_factor: 1.0,
                },
                30.0,
                30.0,
                0.0,
                255.0,
                &lut_u32,
                GpuPixelOutput::Rgba8,
            )
            .unwrap();
        let ref_expr = ex_ref
            .render_expr(
                &mb3_slab,
                &u,
                &v,
                GpuColormapMode::Lanczos3,
                &ndvi_instrs,
                -1.0,
                1.0,
                &lut_u32,
                GpuPixelOutput::IndexedU8,
            )
            .unwrap();
        let ref_rgb = ex_ref
            .render_rgb_formula(
                &mb3_slab,
                &u,
                &v,
                GpuColormapMode::Lanczos3,
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
                &formula,
            )
            .unwrap();
        let ref_sar = ex_ref
            .render_sar_rgb_formula_with_plan(
                &sar2_slab,
                crate::colormap::GpuCoordPlan::Separable1D {
                    u_coords: &u,
                    v_coords: &v,
                },
                GpuColormapMode::Lanczos3,
                [-25.0, -30.0, 0.0],
                [0.0, -5.0, 15.0],
                &formula,
            )
            .unwrap();

        let coalescer = Arc::new(GpuCoalescer::new(&ctx, 32, 2000));
        let barrier = Arc::new(std::sync::Barrier::new(15));
        let mut handles = Vec::new();
        for i in 0..15usize {
            let c = Arc::clone(&coalescer);
            let b = Arc::clone(&barrier);
            let s1 = Arc::clone(&single_slab);
            let s3 = Arc::clone(&mb3_slab);
            let s2 = Arc::clone(&sar2_slab);
            let u_c = u.clone();
            let v_c = v.clone();
            let instrs = ndvi_instrs.clone();
            let form = formula.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                let plan = OwnedCoordPlan::Separable1D { u: u_c, v: v_c };
                let job = match i % 5 {
                    0 => CoalescedJob::Colormap {
                        slab: s1,
                        plan,
                        mode: GpuColormapMode::Lanczos3,
                        rescale_min: 0.0,
                        rescale_max: 2.0,
                        lut_u32,
                        output: CoalescedOutput::IndexedU8,
                    },
                    1 => CoalescedJob::Terrain {
                        slab: s1,
                        plan,
                        mode: GpuColormapMode::Cubic,
                        alg: GpuDemAlgorithm::Hillshade {
                            azimuth_deg: 315.0,
                            altitude_deg: 45.0,
                            z_factor: 1.0,
                        },
                        pixel_size_x_m: 30.0,
                        pixel_size_y_m: 30.0,
                        rescale_min: 0.0,
                        rescale_max: 255.0,
                        lut_u32,
                        output: CoalescedOutput::Rgba8,
                    },
                    2 => CoalescedJob::Expr {
                        slab: s3,
                        plan,
                        mode: GpuColormapMode::Lanczos3,
                        instrs,
                        rescale_min: -1.0,
                        rescale_max: 1.0,
                        lut_u32,
                        output: CoalescedOutput::IndexedU8,
                    },
                    3 => CoalescedJob::Rgb {
                        slab: s3,
                        plan,
                        mode: GpuColormapMode::Lanczos3,
                        rescale_min: [0.0, 0.0, 0.0],
                        rescale_max: [2.0, 2.0, 2.0],
                        formula: form,
                    },
                    _ => CoalescedJob::SarRgb {
                        slab: s2,
                        plan,
                        mode: GpuColormapMode::Lanczos3,
                        rescale_min: [-25.0, -30.0, 0.0],
                        rescale_max: [0.0, -5.0, 15.0],
                        formula: form,
                    },
                };
                (i % 5, c.submit(job).unwrap())
            }));
        }

        for h in handles {
            let (kind, out) = h.join().unwrap();
            match kind {
                0 => assert_eq!(out, ref_cm, "Colormap mismatch"),
                1 => assert_eq!(out, ref_dem, "Terrain mismatch"),
                2 => assert_eq!(out, ref_expr, "Expr NDVI mismatch"),
                3 => assert_eq!(out, ref_rgb, "Rgb+ColorFormula mismatch"),
                _ => assert_eq!(out, ref_sar, "SarRgb+ColorFormula mismatch"),
            }
        }
        let st = coalescer.stats();
        assert_eq!(st.jobs_executed, 15);
        assert!(
            st.batches_submitted < 15 && st.max_batch_seen >= 3,
            "expected multi-family coalescing across 15 threads, got {st:?}"
        );
    }
}
