//! Zarr chunk window reader (#365) and [`BandSource`] bridge (#367).
//!
//! Fetches covering chunks concurrently through [`ZarrStore`], decodes them into
//! [`cogzilla_codec::DecodedBand`] with exact edge dimensions and [`BlockStats`], maps unwritten
//! chunks (when `fill_value` is present) to `None` so the kernel's sparse all-nodata fast path
//! applies unchanged, and hands the result straight to [`cogzilla_kernel::BandSource::new`].

use std::sync::Arc;

use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
use cogzilla_kernel::BandSource;
use futures_util::stream::{self, StreamExt};

use crate::sharding::{ShardIndexCache, ShardingConfig, read_sharded_inner_chunk};
use crate::{ArrayMeta, SampleType, ZarrError, ZarrStore, decode_chunk_bytes};

/// Default concurrency limit for chunk fetches inside a single window read.
pub const DEFAULT_CHUNK_CONCURRENCY: usize = 16;

/// Default maximum number of chunks a single Zarr request may fan out to (#386).
pub const DEFAULT_MAX_CHUNKS_PER_REQUEST: usize = 256;

/// Default maximum total decoded bytes across all covering chunks in one Zarr request (256 MiB, #386).
pub const DEFAULT_MAX_DECODED_WINDOW_BYTES: usize = 256 * 1024 * 1024;

/// Pre-fetch fan-out and decoded-byte safety caps (#386).
#[derive(Debug, Clone, Copy)]
pub struct ZarrCaps {
    pub max_chunks_per_request: usize,
    pub max_decoded_window_bytes: usize,
}

impl Default for ZarrCaps {
    fn default() -> Self {
        Self {
            max_chunks_per_request: DEFAULT_MAX_CHUNKS_PER_REQUEST,
            max_decoded_window_bytes: DEFAULT_MAX_DECODED_WINDOW_BYTES,
        }
    }
}

impl ZarrCaps {
    /// Load caps from `COGZILLA_ZARR_MAX_CHUNKS_PER_REQUEST` and
    /// `COGZILLA_ZARR_MAX_DECODED_WINDOW_BYTES`, falling back to defaults.
    pub fn from_env() -> Self {
        let max_chunks = std::env::var("COGZILLA_ZARR_MAX_CHUNKS_PER_REQUEST")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_MAX_CHUNKS_PER_REQUEST);
        let max_bytes = std::env::var("COGZILLA_ZARR_MAX_DECODED_WINDOW_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_MAX_DECODED_WINDOW_BYTES);
        Self {
            max_chunks_per_request: max_chunks,
            max_decoded_window_bytes: max_bytes,
        }
    }
}

/// Rectangular chunk-grid window covering a requested pixel-space rectangle `[x0..x1, y0..y1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkWindow {
    pub ifd_w: u32,
    pub ifd_h: u32,
    pub tile_w: u32,
    pub tile_h: u32,
    pub tx0: u32,
    pub ty0: u32,
    pub grid_w: u32,
    pub grid_h: u32,
}

impl ChunkWindow {
    /// Compute the covering chunk grid for pixel window `[x0..x1, y0..y1]` on a 2-D or 3-D array.
    ///
    /// For 2-D arrays, axes are `(y, x)`; for 3-D arrays, axes are `(band/z, y, x)`.
    pub fn from_pixel_window(
        meta: &ArrayMeta,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> Result<Self, ZarrError> {
        meta.require_renderable_rank()?;
        let sharding = ShardingConfig::from_meta(meta)?;
        let effective_chunk_shape = sharding
            .as_ref()
            .map(|s| s.inner_chunk_shape.as_slice())
            .unwrap_or(&meta.chunk_shape);
        let (h_u64, w_u64, ch_u64, cw_u64) = spatial_dims(meta, effective_chunk_shape)?;
        let ifd_w = u32::try_from(w_u64).map_err(|_| ZarrError::InvalidSelection {
            detail: format!("array width {w_u64} exceeds u32"),
        })?;
        let ifd_h = u32::try_from(h_u64).map_err(|_| ZarrError::InvalidSelection {
            detail: format!("array height {h_u64} exceeds u32"),
        })?;
        let tile_w = u32::try_from(cw_u64).map_err(|_| ZarrError::InvalidSelection {
            detail: format!("chunk width {cw_u64} exceeds u32"),
        })?;
        let tile_h = u32::try_from(ch_u64).map_err(|_| ZarrError::InvalidSelection {
            detail: format!("chunk height {ch_u64} exceeds u32"),
        })?;
        if tile_w == 0 || tile_h == 0 || ifd_w == 0 || ifd_h == 0 {
            return Err(ZarrError::InvalidSelection {
                detail: "array or chunk spatial dimension is zero".into(),
            });
        }

        let cx0 = x0.min(ifd_w);
        let cy0 = y0.min(ifd_h);
        let cx1 = x1.min(ifd_w);
        let cy1 = y1.min(ifd_h);
        if cx0 >= cx1 || cy0 >= cy1 {
            return Ok(Self {
                ifd_w,
                ifd_h,
                tile_w,
                tile_h,
                tx0: 0,
                ty0: 0,
                grid_w: 0,
                grid_h: 0,
            });
        }

        let tx0 = cx0 / tile_w;
        let ty0 = cy0 / tile_h;
        let tx1 = (cx1 - 1) / tile_w;
        let ty1 = (cy1 - 1) / tile_h;
        Ok(Self {
            ifd_w,
            ifd_h,
            tile_w,
            tile_h,
            tx0,
            ty0,
            grid_w: tx1 - tx0 + 1,
            grid_h: ty1 - ty0 + 1,
        })
    }

    /// Total number of chunks in this window (`grid_w * grid_h`).
    pub fn chunk_count(&self) -> usize {
        (self.grid_w as usize) * (self.grid_h as usize)
    }

    /// Enforce fan-out and total decoded-byte caps from metadata BEFORE issuing any upstream GET (#386).
    pub fn check_caps(&self, meta: &ArrayMeta, caps: &ZarrCaps) -> Result<(), ZarrError> {
        let count = self.chunk_count();
        if count > caps.max_chunks_per_request {
            return Err(ZarrError::CapExceeded {
                kind: "covering chunk count",
                computed: count,
                cap: caps.max_chunks_per_request,
            });
        }
        let sharding = ShardingConfig::from_meta(meta)?;
        let chunk_shape = sharding
            .as_ref()
            .map(|s| s.inner_chunk_shape.as_slice())
            .unwrap_or(&meta.chunk_shape);
        let chunk_elems = chunk_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))
            .unwrap_or(usize::MAX);
        let chunk_bytes = chunk_elems.saturating_mul(meta.dtype.bytes_per_sample());
        let window_bytes = count.saturating_mul(chunk_bytes);
        if window_bytes > caps.max_decoded_window_bytes {
            return Err(ZarrError::CapExceeded {
                kind: "decoded window bytes",
                computed: window_bytes,
                cap: caps.max_decoded_window_bytes,
            });
        }
        Ok(())
    }

    /// Valid (edge-truncated) `(width, height)` for chunk `(tx, ty)`.
    pub fn valid_chunk_dims(&self, tx: u32, ty: u32) -> (u32, u32) {
        let ox = tx * self.tile_w;
        let oy = ty * self.tile_h;
        let w = self.ifd_w.saturating_sub(ox).min(self.tile_w);
        let h = self.ifd_h.saturating_sub(oy).min(self.tile_h);
        (w, h)
    }
}

/// Decoded chunk blocks for a window, ready to borrow as a [`BandSource`] with zero copies.
pub struct ZarrWindowBlocks {
    pub window: ChunkWindow,
    pub blocks: Vec<Option<DecodedBand>>,
    pub scale_offset: Option<(f32, f32)>,
}

impl ZarrWindowBlocks {
    /// Borrow the decoded chunks as a [`BandSource`] for `cogzilla-kernel`.
    ///
    /// Pass a temporary `Vec<Option<&DecodedBand>>` slice buffer (`refs`) owned by the caller so
    /// the returned [`BandSource`] borrows from `self` and `refs` without extra indirection.
    pub fn block_refs(&self) -> Vec<Option<&DecodedBand>> {
        self.blocks.iter().map(|b| b.as_ref()).collect()
    }

    /// Construct a [`BandSource`] pointing at the same kernel seam the COG reader uses.
    pub fn as_band_source<'a>(&self, block_refs: &'a [Option<&'a DecodedBand>]) -> BandSource<'a> {
        let mut src = BandSource::new(
            self.window.ifd_w,
            self.window.ifd_h,
            self.window.tile_w,
            self.window.tile_h,
            self.window.tx0,
            self.window.ty0,
            self.window.grid_w,
            self.window.grid_h,
            block_refs,
        );
        if let Some((scale, offset)) = self.scale_offset {
            src = src.with_scale_offset(scale, offset);
        }
        src
    }
}

/// Read all covering chunks for pixel window `[x0..x1, y0..y1]` of a 2-D array.
pub async fn read_chunk_window(
    store: &dyn ZarrStore,
    array_prefix: &str,
    meta: &ArrayMeta,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) -> Result<ZarrWindowBlocks, ZarrError> {
    read_chunk_window_for_band(
        store,
        array_prefix,
        meta,
        None,
        x0,
        y0,
        x1,
        y1,
        DEFAULT_CHUNK_CONCURRENCY,
    )
    .await
}

/// Read all covering chunks for pixel window `[x0..x1, y0..y1]` of a 2-D or 3-D array.
///
/// - For 2-D arrays, `band_idx` must be `None` (or `Some(0)`).
/// - For 3-D arrays `(depth, height, width)`, `band_idx` selects the 0-based index along axis 0
///   (defaulting to `0` if `None`).
#[allow(clippy::too_many_arguments)]
pub async fn read_chunk_window_for_band(
    store: &dyn ZarrStore,
    array_prefix: &str,
    meta: &ArrayMeta,
    band_idx: Option<u64>,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    max_concurrency: usize,
) -> Result<ZarrWindowBlocks, ZarrError> {
    read_chunk_window_for_band_with_caps(
        store,
        array_prefix,
        meta,
        band_idx,
        x0,
        y0,
        x1,
        y1,
        max_concurrency,
        &ZarrCaps::default(),
    )
    .await
}

/// Read all covering chunks for pixel window `[x0..x1, y0..y1]` of a 2-D or 3-D array, enforcing
/// `caps` on metadata before issuing any upstream GET (#386).
#[allow(clippy::too_many_arguments)]
pub async fn read_chunk_window_for_band_with_caps(
    store: &dyn ZarrStore,
    array_prefix: &str,
    meta: &ArrayMeta,
    band_idx: Option<u64>,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    max_concurrency: usize,
    caps: &ZarrCaps,
) -> Result<ZarrWindowBlocks, ZarrError> {
    let window = ChunkWindow::from_pixel_window(meta, x0, y0, x1, y1)?;
    window.check_caps(meta, caps)?;
    let scale_offset = meta.scale_offset();
    if window.chunk_count() == 0 {
        return Ok(ZarrWindowBlocks {
            window,
            blocks: Vec::new(),
            scale_offset,
        });
    }

    let sharding = ShardingConfig::from_meta(meta)?.map(Arc::new);
    let effective_chunk_shape: &[u64] = sharding
        .as_ref()
        .map(|s| s.inner_chunk_shape.as_slice())
        .unwrap_or(&meta.chunk_shape);

    let rank = meta.rank();
    let (tz, lz) = if rank == 3 {
        let depth = meta.shape[0];
        let cd = effective_chunk_shape[0];
        let b = band_idx.unwrap_or(0);
        if b >= depth {
            return Err(ZarrError::InvalidSelection {
                detail: format!("band index {b} out of range for depth {depth}"),
            });
        }
        if cd == 0 {
            return Err(ZarrError::InvalidSelection {
                detail: "3D array has chunk depth 0".into(),
            });
        }
        (Some(b / cd), (b % cd) as usize)
    } else {
        if let Some(b) = band_idx
            && b != 0
        {
            return Err(ZarrError::InvalidSelection {
                detail: format!("band index {b} invalid for 2D array"),
            });
        }
        (None, 0)
    };

    let nodata = meta.effective_nodata();
    let has_fill = meta.fill_value.is_some();
    let prefix = array_prefix.trim_matches('/');
    let concurrency = max_concurrency.max(1);
    let index_cache = Arc::new(ShardIndexCache::new());

    // Build ordered list of (slot_idx, tx, ty)
    let mut jobs = Vec::with_capacity(window.chunk_count());
    for gy in 0..window.grid_h {
        let ty = window.ty0 + gy;
        for gx in 0..window.grid_w {
            let tx = window.tx0 + gx;
            jobs.push(((gy * window.grid_w + gx) as usize, tx, ty));
        }
    }

    let results: Vec<Result<(usize, Option<DecodedBand>), ZarrError>> = stream::iter(jobs)
        .map(|(slot, tx, ty)| {
            let sharding = sharding.clone();
            let index_cache = Arc::clone(&index_cache);
            async move {
                let decoded_opt = if let Some(ref scfg) = sharding {
                    let idx: Vec<u64> = match tz {
                        Some(z) => vec![z, ty as u64, tx as u64],
                        None => vec![ty as u64, tx as u64],
                    };
                    read_sharded_inner_chunk(store, prefix, meta, scfg, &index_cache, &idx).await?
                } else {
                    let rel_key = match tz {
                        Some(z) => meta.chunk_key(&[z, ty as u64, tx as u64]),
                        None => meta.chunk_key(&[ty as u64, tx as u64]),
                    };
                    let full_key = if prefix.is_empty() {
                        rel_key
                    } else {
                        format!("{prefix}/{rel_key}")
                    };
                    let Some(raw) = store.get(&full_key).await? else {
                        if has_fill {
                            return Ok((slot, None));
                        }
                        return Err(ZarrError::MissingChunk {
                            chunk_key: full_key,
                        });
                    };
                    Some(decode_chunk_bytes(meta, &raw, &full_key)?)
                };

                let Some(decoded_bytes) = decoded_opt else {
                    return Ok((slot, None));
                };
                let plane_bytes =
                    extract_2d_plane(meta, window.tile_h as usize, window.tile_w as usize, &decoded_bytes, lz)?;
                let (valid_w, valid_h) = window.valid_chunk_dims(tx, ty);
                let band = bytes_to_decoded_band(
                    plane_bytes,
                    meta.dtype,
                    window.tile_w,
                    window.tile_h,
                    valid_w,
                    valid_h,
                    nodata,
                );
                Ok((slot, Some(band)))
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    let mut blocks: Vec<Option<DecodedBand>> = (0..window.chunk_count()).map(|_| None).collect();
    for res in results {
        let (slot, band_opt) = res?;
        blocks[slot] = band_opt;
    }

    Ok(ZarrWindowBlocks {
        window,
        blocks,
        scale_offset,
    })
}

fn spatial_dims(meta: &ArrayMeta, chunk_shape: &[u64]) -> Result<(u64, u64, u64, u64), ZarrError> {
    match meta.rank() {
        2 => Ok((meta.shape[0], meta.shape[1], chunk_shape[0], chunk_shape[1])),
        3 => Ok((meta.shape[1], meta.shape[2], chunk_shape[1], chunk_shape[2])),
        other => Err(ZarrError::UnsupportedRank(other)),
    }
}

pub(crate) fn extract_2d_plane<'a>(
    meta: &ArrayMeta,
    ch: usize,
    cw: usize,
    decoded: &'a [u8],
    lz: usize,
) -> Result<&'a [u8], ZarrError> {
    if meta.rank() == 2 {
        return Ok(decoded);
    }
    let plane_bytes = ch * cw * meta.dtype.bytes_per_sample();
    let start = lz * plane_bytes;
    let end = start + plane_bytes;
    if end > decoded.len() {
        return Err(ZarrError::Codec {
            codec: "chunk".into(),
            chunk_key: format!("slice[{lz}]"),
            detail: format!(
                "3D slice {lz} out of bounds for decoded chunk len {}",
                decoded.len()
            ),
        });
    }
    Ok(&decoded[start..end])
}

/// Convert native-endian C-order 2-D chunk bytes (stride `tile_w`, rows `tile_h`) into a
/// [`DecodedBand`] with exact edge dimensions `(valid_w, valid_h)` and accurate [`BlockStats`]
/// computed strictly over `0..valid_h, 0..valid_w`.
pub(crate) fn bytes_to_decoded_band(
    bytes: &[u8],
    dtype: SampleType,
    tile_w: u32,
    tile_h: u32,
    valid_w: u32,
    valid_h: u32,
    nodata: Option<f64>,
) -> DecodedBand {
    let tw = tile_w as usize;
    let th = tile_h as usize;
    let vw = valid_w as usize;
    let vh = valid_h as usize;
    let total = tw * th;

    match dtype {
        SampleType::Uint8 | SampleType::Bool => {
            let nd_u8 = nodata.and_then(|v| {
                if v.is_finite() && (0.0..=255.0).contains(&v) && v.fract() == 0.0 {
                    Some(v as u8)
                } else {
                    None
                }
            });
            let data: Box<[u8]> = bytes[..total].into();
            let stats = compute_stats_u8(&data, tw, vw, vh, nd_u8);
            DecodedBand {
                data: BandData::U8 { data, nodata: nd_u8 },
                width: valid_w,
                height: valid_h,
                stats,
            }
        }
        SampleType::Uint16 => {
            let nd_u16 = nodata.and_then(|v| {
                if v.is_finite() && (0.0..=65535.0).contains(&v) && v.fract() == 0.0 {
                    Some(v as u16)
                } else {
                    None
                }
            });
            let mut data = Vec::with_capacity(total);
            for chunk in bytes[..total * 2].chunks_exact(2) {
                data.push(u16::from_ne_bytes([chunk[0], chunk[1]]));
            }
            let data = data.into_boxed_slice();
            let stats = compute_stats_u16(&data, tw, vw, vh, nd_u16);
            DecodedBand {
                data: BandData::U16 { data, nodata: nd_u16 },
                width: valid_w,
                height: valid_h,
                stats,
            }
        }
        SampleType::Float32 | SampleType::Float16 | SampleType::Int8 | SampleType::Int16 => {
            let mut slab = AlignedF32Slab::new_uninit(total);
            let dst = slab.as_mut_slice();
            let nd_f32 = nodata.and_then(|v| if v.is_nan() { None } else { Some(v as f32) });
            match dtype {
                SampleType::Float32 => {
                    for (d, c) in dst.iter_mut().zip(bytes[..total * 4].chunks_exact(4)) {
                        let v = f32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
                        *d = if nd_f32 == Some(v) { f32::NAN } else { v };
                    }
                }
                SampleType::Float16 => {
                    for (d, c) in dst.iter_mut().zip(bytes[..total * 2].chunks_exact(2)) {
                        let v = f16_to_f32(u16::from_ne_bytes([c[0], c[1]]));
                        *d = if nd_f32 == Some(v) { f32::NAN } else { v };
                    }
                }
                SampleType::Int8 => {
                    for (d, &b) in dst.iter_mut().zip(&bytes[..total]) {
                        let v = (b as i8) as f32;
                        *d = if nd_f32 == Some(v) { f32::NAN } else { v };
                    }
                }
                SampleType::Int16 => {
                    for (d, c) in dst.iter_mut().zip(bytes[..total * 2].chunks_exact(2)) {
                        let v = i16::from_ne_bytes([c[0], c[1]]) as f32;
                        *d = if nd_f32 == Some(v) { f32::NAN } else { v };
                    }
                }
                _ => unreachable!(),
            }
            let stats = compute_stats_f32(slab.as_slice(), tw, vw, vh);
            DecodedBand {
                data: BandData::F32(slab),
                width: valid_w,
                height: valid_h,
                stats,
            }
        }
        // Wide dtypes stay exact in `BandData::F64` (never truncated through `f32`).
        SampleType::Int32
        | SampleType::Uint32
        | SampleType::Int64
        | SampleType::Uint64
        | SampleType::Float64 => {
            let mut data = Vec::with_capacity(total);
            let nd_f64 = nodata.filter(|v| !v.is_nan());
            match dtype {
                SampleType::Int32 => {
                    for c in bytes[..total * 4].chunks_exact(4) {
                        let v = i32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f64;
                        data.push(if nd_f64 == Some(v) { f64::NAN } else { v });
                    }
                }
                SampleType::Uint32 => {
                    for c in bytes[..total * 4].chunks_exact(4) {
                        let v = u32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f64;
                        data.push(if nd_f64 == Some(v) { f64::NAN } else { v });
                    }
                }
                SampleType::Int64 => {
                    for c in bytes[..total * 8].chunks_exact(8) {
                        let v = i64::from_ne_bytes(c.try_into().unwrap()) as f64;
                        data.push(if nd_f64 == Some(v) { f64::NAN } else { v });
                    }
                }
                SampleType::Uint64 => {
                    for c in bytes[..total * 8].chunks_exact(8) {
                        let v = u64::from_ne_bytes(c.try_into().unwrap()) as f64;
                        data.push(if nd_f64 == Some(v) { f64::NAN } else { v });
                    }
                }
                SampleType::Float64 => {
                    for c in bytes[..total * 8].chunks_exact(8) {
                        let v = f64::from_ne_bytes(c.try_into().unwrap());
                        data.push(if nd_f64 == Some(v) { f64::NAN } else { v });
                    }
                }
                _ => unreachable!(),
            }
            let data = data.into_boxed_slice();
            let stats = compute_stats_f64(&data, tw, vw, vh);
            DecodedBand {
                data: BandData::F64(data),
                width: valid_w,
                height: valid_h,
                stats,
            }
        }
        SampleType::Uint1 | SampleType::Uint4 | SampleType::Uint12 => {
            unreachable!("Zarr stores never produce bit-packed TIFF sample types")
        }
    }
}

fn compute_stats_u8(data: &[u8], stride: usize, vw: usize, vh: usize, nd: Option<u8>) -> BlockStats {
    let mut all_nodata = true;
    let mut no_nodata = true;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for r in 0..vh {
        let row = &data[r * stride..r * stride + vw];
        for &v in row {
            if nd == Some(v) {
                no_nodata = false;
            } else {
                all_nodata = false;
                let vf = v as f32;
                if vf < min {
                    min = vf;
                }
                if vf > max {
                    max = vf;
                }
            }
        }
    }
    if all_nodata {
        BlockStats::ALL_NODATA
    } else {
        BlockStats {
            all_nodata,
            no_nodata,
            min,
            max,
        }
    }
}

fn compute_stats_u16(
    data: &[u16],
    stride: usize,
    vw: usize,
    vh: usize,
    nd: Option<u16>,
) -> BlockStats {
    let mut all_nodata = true;
    let mut no_nodata = true;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for r in 0..vh {
        let row = &data[r * stride..r * stride + vw];
        for &v in row {
            if nd == Some(v) {
                no_nodata = false;
            } else {
                all_nodata = false;
                let vf = v as f32;
                if vf < min {
                    min = vf;
                }
                if vf > max {
                    max = vf;
                }
            }
        }
    }
    if all_nodata {
        BlockStats::ALL_NODATA
    } else {
        BlockStats {
            all_nodata,
            no_nodata,
            min,
            max,
        }
    }
}

fn compute_stats_f32(data: &[f32], stride: usize, vw: usize, vh: usize) -> BlockStats {
    let mut all_nodata = true;
    let mut no_nodata = true;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for r in 0..vh {
        let row = &data[r * stride..r * stride + vw];
        for &v in row {
            if v.is_nan() {
                no_nodata = false;
            } else {
                all_nodata = false;
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
    }
    if all_nodata {
        BlockStats::ALL_NODATA
    } else {
        BlockStats {
            all_nodata,
            no_nodata,
            min,
            max,
        }
    }
}

fn compute_stats_f64(data: &[f64], stride: usize, vw: usize, vh: usize) -> BlockStats {
    let mut all_nodata = true;
    let mut no_nodata = true;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for r in 0..vh {
        let row = &data[r * stride..r * stride + vw];
        for &v in row {
            if v.is_nan() {
                no_nodata = false;
            } else {
                all_nodata = false;
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
    }
    if all_nodata {
        BlockStats::ALL_NODATA
    } else {
        BlockStats {
            all_nodata,
            no_nodata,
            min: min as f32,
            max: max as f32,
        }
    }
}

/// Convert IEEE 754 binary16 bits to `f32`.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;
    let f_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChunkKeyEncoding, Endianness, MemoryStore, ZarrVersion};

    fn make_2d_meta(h: u64, w: u64, ch: u64, cw: u64, dtype: SampleType, fill: Option<f64>) -> ArrayMeta {
        ArrayMeta {
            version: ZarrVersion::V3,
            shape: vec![h, w],
            chunk_shape: vec![ch, cw],
            dtype,
            endianness: Endianness::Little,
            fill_value: fill,
            codecs: vec![],
            chunk_key: ChunkKeyEncoding::Default { separator: '/' },
            dimension_names: Some(vec![Some("y".into()), Some("x".into())]),
            attributes: Default::default(),
            order_f: false,
        }
    }

    #[tokio::test]
    async fn edge_chunk_returns_truncated_dims_and_exact_stats() {
        // 10x10 array with 8x8 chunks -> chunk (1,1) has valid dims (2, 2)
        let meta = make_2d_meta(10, 10, 8, 8, SampleType::Float32, Some(f64::NAN));
        let mut store = MemoryStore::new();
        let mut chunk = vec![999.0f32; 64]; // overhang sentinel must NOT pollute stats
        chunk[0] = 10.0;
        chunk[1] = 20.0;
        chunk[8] = 30.0;
        chunk[9] = f32::NAN;
        let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
        store.insert("arr/c/1/1", bytes);

        let wb = read_chunk_window(&store, "arr", &meta, 8, 8, 10, 10).await.unwrap();
        assert_eq!(wb.blocks.len(), 1);
        let blk = wb.blocks[0].as_ref().unwrap();
        assert_eq!((blk.width, blk.height), (2, 2));
        assert!(!blk.stats.all_nodata);
        assert!(!blk.stats.no_nodata);
        assert_eq!(blk.stats.min, 10.0);
        assert_eq!(blk.stats.max, 30.0);

        let refs = wb.block_refs();
        let src = wb.as_band_source(&refs);
        assert_eq!(src.sample_checked(8, 8), 10.0);
        assert_eq!(src.sample_checked(9, 8), 20.0);
        assert_eq!(src.sample_checked(8, 9), 30.0);
        assert!(src.sample_checked(9, 9).is_nan());
        assert!(src.sample_checked(10, 8).is_nan()); // out of array bounds
    }

    #[tokio::test]
    async fn missing_chunk_with_no_fill_value_is_error() {
        let meta = make_2d_meta(16, 16, 8, 8, SampleType::Float32, None);
        let store = MemoryStore::new();
        let err = read_chunk_window(&store, "arr", &meta, 0, 0, 8, 8)
            .await
            .err()
            .expect("expected MissingChunk");
        assert!(matches!(err, ZarrError::MissingChunk { .. }), "{err}");
    }

    #[tokio::test]
    async fn wide_dtype_int32_stays_exact_in_f64_band_data() {
        let meta = make_2d_meta(4, 4, 4, 4, SampleType::Int32, Some(-9999.0));
        let mut store = MemoryStore::new();
        // Value > 2^24 that would lose precision if cast through f32!
        let exact_val: i32 = 16_777_217;
        let mut vals = [exact_val; 16];
        vals[0] = -9999;
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        store.insert("wide/c/0/0", bytes);

        let wb = read_chunk_window(&store, "wide", &meta, 0, 0, 4, 4).await.unwrap();
        let blk = wb.blocks[0].as_ref().unwrap();
        assert!(matches!(blk.data, BandData::F64(_)));
        let refs = wb.block_refs();
        let src = wb.as_band_source(&refs);
        assert!(src.sample_f64_checked(0, 0).is_nan());
        assert_eq!(src.sample_f64_checked(1, 0), exact_val as f64);
    }

    #[tokio::test]
    async fn caps_refuse_oversized_fanout_before_first_fetch() {
        // 1024x1024 array with 16x16 chunks = 64x64 = 4096 chunks
        let meta = make_2d_meta(1024, 1024, 16, 16, SampleType::Float32, Some(f64::NAN));
        let store = MemoryStore::new();
        let caps = ZarrCaps {
            max_chunks_per_request: 16,
            max_decoded_window_bytes: 1024 * 1024,
        };
        let err = read_chunk_window_for_band_with_caps(
            &store, "arr", &meta, None, 0, 0, 1024, 1024, 4, &caps,
        )
        .await
        .err()
        .expect("expected CapExceeded");
        assert!(
            matches!(err, ZarrError::CapExceeded { computed: 4096, cap: 16, .. }),
            "{err}"
        );
        assert_eq!(
            store.get_count(),
            0,
            "cap refusal must happen before any upstream GET"
        );
    }
}
