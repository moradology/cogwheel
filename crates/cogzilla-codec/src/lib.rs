pub mod lzw;
pub mod predictor;
pub mod slab;

use std::cell::RefCell;

use cogzilla_tiff::{
    ByteOrder, Compression, IfdMeta, MaskIfdMeta, PlanarConfig, Predictor, SampleType,
};
use thiserror::Error;

pub use slab::{AlignedF32Slab, BandData, BlockStats, DecodedBand, DecodedTile};

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("deflate decompression failed: {0}")]
    Deflate(String),
    #[error("zstd decompression failed: code {0}")]
    Zstd(usize),
    #[error("LZW decompression failed: {0}")]
    Lzw(&'static str),
    #[error("JPEG decompression failed: {0}")]
    Jpeg(String),
    #[error("WebP decompression failed")]
    Webp,
    #[error("uncompressed tile size mismatch: expected {expected} bytes, got {got}")]
    SizeMismatch { expected: usize, got: usize },
}

struct ThreadCodecState {
    deflate: libdeflater::Decompressor,
    zstd_dctx: *mut zstd_sys::ZSTD_DCtx,
    lzw: lzw::LzwDecoder,
    scratch: Vec<u8>,
}

impl ThreadCodecState {
    fn new() -> Self {
        let zstd_dctx = unsafe { zstd_sys::ZSTD_createDCtx() };
        assert!(!zstd_dctx.is_null(), "failed to allocate ZSTD_DCtx");
        Self {
            deflate: libdeflater::Decompressor::new(),
            zstd_dctx,
            lzw: lzw::LzwDecoder::new(),
            scratch: Vec::with_capacity(512 * 512 * 4),
        }
    }
}

impl Drop for ThreadCodecState {
    fn drop(&mut self) {
        if !self.zstd_dctx.is_null() {
            unsafe {
                zstd_sys::ZSTD_freeDCtx(self.zstd_dctx);
            }
        }
    }
}

thread_local! {
    static CODEC_STATE: RefCell<ThreadCodecState> = RefCell::new(ThreadCodecState::new());
}

/// Hard upper bound on a single decoded TIFF tile/strip (128 MiB) to prevent integer wrap
/// overflows and zip-bomb OOMs on hostile remote GeoTIFFs.
const MAX_DECODED_TILE_BYTES: usize = 128 * 1024 * 1024;

/// Decompress and reconstruct a single internal COG tile into planar 64-byte-aligned `f32` bands
/// with precomputed `BlockStats`, reusing thread-local codec contexts and scratch memory.
pub fn decode_tile(
    compressed: &[u8],
    ifd: &IfdMeta,
    byte_order: ByteOrder,
    nodata: Option<f32>,
) -> Result<DecodedTile, CodecError> {
    decode_tile_rows(compressed, ifd, byte_order, nodata, ifd.tile_height)
}

/// Decompress and reconstruct a single TIFF tile or strip with an explicit row count `actual_rows`
/// (which is `< ifd.tile_height` for the final partial strip of a stripped TIFF).
///
/// Decoding only `actual_rows` avoids trailing zero-padding in `BlockStats`, prevents false
/// `alpha == 0` detection on stripped RGBA rasters, and eliminates stale scratch bytes on WebP strips.
pub fn decode_tile_rows(
    compressed: &[u8],
    ifd: &IfdMeta,
    byte_order: ByteOrder,
    nodata: Option<f32>,
    actual_rows: u32,
) -> Result<DecodedTile, CodecError> {
    let tile_h = actual_rows.min(ifd.tile_height).max(1);
    let bands_in_tile = match ifd.planar_config {
        PlanarConfig::Chunky => ifd.samples_per_pixel as usize,
        PlanarConfig::Planar => 1,
    };
    let expected_bytes = (ifd.tile_width as usize)
        .checked_mul(tile_h as usize)
        .and_then(|n| n.checked_mul(bands_in_tile))
        .and_then(|n| n.checked_mul(ifd.sample_type.bytes_per_sample()))
        .filter(|&b| b > 0 && b <= MAX_DECODED_TILE_BYTES)
        .ok_or(CodecError::SizeMismatch {
            expected: MAX_DECODED_TILE_BYTES,
            got: compressed.len(),
        })?;

    // Fast path: Uncompressed + Predictor::None avoids touching thread-local scratch completely!
    if ifd.compression == Compression::None
        && ifd.predictor == Predictor::None
        && compressed.len() >= expected_bytes
    {
        let mut bands = match ifd.sample_type {
            SampleType::Float32 => predictor::reconstruct_predictor1(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                nodata,
            ),
            SampleType::Uint16 => predictor::reconstruct_u16(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                nodata,
            ),
            SampleType::Int16 => predictor::reconstruct_i16(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                nodata,
            ),
            SampleType::Uint8 => predictor::reconstruct_u8(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                false,
                nodata,
            ),
            SampleType::Int8 => predictor::reconstruct_i8(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                false,
                nodata,
            ),
            SampleType::Uint32 => predictor::reconstruct_int32(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                false,
                nodata,
            ),
            SampleType::Int32 => predictor::reconstruct_int32(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                true,
                false,
                nodata,
            ),
            SampleType::Float64 => {
                let mut copy = compressed[..expected_bytes].to_vec();
                predictor::reconstruct_f64(
                    &mut copy,
                    ifd.tile_width,
                    tile_h,
                    bands_in_tile,
                    byte_order,
                    Predictor::None,
                    nodata,
                )
            }
        };
        apply_rgba_alpha_if_present(&mut bands, ifd);
        return Ok(DecodedTile { bands });
    }

    CODEC_STATE.with(|cell| {
        struct ScratchShrinkGuard<'a>(&'a mut ThreadCodecState);
        impl Drop for ScratchShrinkGuard<'_> {
            fn drop(&mut self) {
                if self.0.scratch.capacity() > 4 * 1024 * 1024 {
                    self.0.scratch = Vec::with_capacity(512 * 512 * 4);
                }
            }
        }
        let mut guard = cell.borrow_mut();
        let st_guard = ScratchShrinkGuard(&mut *guard);
        let st = &mut *st_guard.0;
        if st.scratch.len() < expected_bytes {
            st.scratch.resize(expected_bytes, 0);
        }
        let raw_slice = &mut st.scratch[..expected_bytes];

        match ifd.compression {
            Compression::None => {
                if compressed.len() < expected_bytes {
                    if ifd.is_stripped {
                        raw_slice[..compressed.len()].copy_from_slice(compressed);
                        raw_slice[compressed.len()..].fill(0);
                    } else {
                        return Err(CodecError::SizeMismatch {
                            expected: expected_bytes,
                            got: compressed.len(),
                        });
                    }
                } else {
                    raw_slice.copy_from_slice(&compressed[..expected_bytes]);
                }
            }
            Compression::Deflate => {
                // TIFF Adobe Deflate (code 8 / 32946) uses zlib framing; fall back to raw deflate if needed.
                let written = match st.deflate.zlib_decompress(compressed, raw_slice) {
                    Ok(n) => n,
                    Err(e1) => st
                        .deflate
                        .deflate_decompress(compressed, raw_slice)
                        .map_err(|_| CodecError::Deflate(format!("{e1:?}")))?,
                };
                if written < expected_bytes {
                    raw_slice[written..].fill(0);
                }
            }
            Compression::Zstd => {
                let code = unsafe {
                    zstd_sys::ZSTD_decompressDCtx(
                        st.zstd_dctx,
                        raw_slice.as_mut_ptr() as *mut std::ffi::c_void,
                        expected_bytes,
                        compressed.as_ptr() as *const std::ffi::c_void,
                        compressed.len(),
                    )
                };
                if unsafe { zstd_sys::ZSTD_isError(code) } != 0 {
                    return Err(CodecError::Zstd(code));
                }
                if code < expected_bytes {
                    raw_slice[code..].fill(0);
                }
            }
            Compression::Lzw => {
                let written = st.lzw.decode(compressed, raw_slice)?;
                if written < expected_bytes {
                    raw_slice[written..].fill(0);
                }
            }
            Compression::Jpeg => {
                // Merge shared TIFF JPEGTables (tag 347) with abbreviated tile bitstream if present
                let merged_buf: Vec<u8>;
                let jpeg_bytes: &[u8] = if let Some(tables) = ifd.jpeg_tables.as_deref() {
                    if tables.len() >= 4 && compressed.len() >= 2 {
                        let t_end = if tables.ends_with(&[0xFF, 0xD9]) {
                            tables.len() - 2
                        } else {
                            tables.len()
                        };
                        let c_start = if compressed.starts_with(&[0xFF, 0xD8]) {
                            2
                        } else {
                            0
                        };
                        merged_buf = [tables[..t_end].as_ref(), compressed[c_start..].as_ref()].concat();
                        &merged_buf
                    } else {
                        compressed
                    }
                } else {
                    compressed
                };

                use zune_jpeg::zune_core::colorspace::ColorSpace;
                use zune_jpeg::zune_core::options::DecoderOptions;

                let opts = DecoderOptions::default()
                    .set_max_width(ifd.tile_width as usize)
                    .set_max_height(tile_h as usize);
                let mut dec = zune_jpeg::JpegDecoder::new_with_options(jpeg_bytes, opts);
                dec.decode_headers()
                    .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                let out_cs = match (bands_in_tile, ifd.photometric) {
                    (1, _) => ColorSpace::Luma,
                    // Photometric=2 (RGB): match the stream's own input colorspace (RGB if an Adobe
                    // APP14 transform=0 marker is present, YCbCr if stripped) so zune-jpeg performs
                    // zero colorspace conversion instead of failing on RGB->YCbCr.
                    (3, 2) => dec.get_input_colorspace().unwrap_or(ColorSpace::YCbCr),
                    _ => ColorSpace::RGB,
                };
                dec.set_options(opts.jpeg_set_out_colorspace(out_cs));
                let out_len = dec.output_buffer_size().unwrap_or(expected_bytes);
                if out_len <= expected_bytes {
                    dec.decode_into(&mut raw_slice[..out_len])
                        .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                    if out_len < expected_bytes {
                        raw_slice[out_len..expected_bytes].fill(0);
                    }
                } else {
                    let pixels = dec
                        .decode()
                        .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                    raw_slice[..expected_bytes].copy_from_slice(&pixels[..expected_bytes]);
                }
            }
            Compression::Webp => {
                let stride = (ifd.tile_width as i32) * (bands_in_tile as i32);
                let ptr = unsafe {
                    if bands_in_tile == 4 {
                        libwebp_sys::WebPDecodeRGBAInto(
                            compressed.as_ptr(),
                            compressed.len(),
                            raw_slice.as_mut_ptr(),
                            expected_bytes,
                            stride,
                        )
                    } else {
                        libwebp_sys::WebPDecodeRGBInto(
                            compressed.as_ptr(),
                            compressed.len(),
                            raw_slice.as_mut_ptr(),
                            expected_bytes,
                            stride,
                        )
                    }
                };
                if ptr.is_null() {
                    return Err(CodecError::Webp);
                }
            }
        }

        let mut bands = match (ifd.sample_type, ifd.predictor) {
            (SampleType::Float32, Predictor::None) => predictor::reconstruct_predictor1(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                nodata,
            ),
            (SampleType::Float32, Predictor::HorizontalInt) => predictor::reconstruct_predictor2(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                nodata,
            ),
            (SampleType::Float32, Predictor::FloatingPoint) => predictor::reconstruct_predictor3(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                nodata,
            ),
            (SampleType::Uint16, Predictor::None) => predictor::reconstruct_u16(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                nodata,
            ),
            (SampleType::Uint16, Predictor::HorizontalInt) => predictor::reconstruct_u16(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                true,
                nodata,
            ),
            (SampleType::Uint16, Predictor::FloatingPoint) => predictor::reconstruct_u16(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                nodata,
            ),
            (SampleType::Int16, p) => predictor::reconstruct_i16(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Uint8, p) => predictor::reconstruct_u8(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Int8, p) => predictor::reconstruct_i8(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Uint32, p) => predictor::reconstruct_int32(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Int32, p) => predictor::reconstruct_int32(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                true,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Float64, p) => predictor::reconstruct_f64(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                p,
                nodata,
            ),
        };

        apply_rgba_alpha_if_present(&mut bands, ifd);
        Ok(DecodedTile { bands })
    })
}

/// Return the 0-based alpha channel index from `ExtraSamples` (tag 338 = 1 or 2), supporting
/// both 2-band Gray+Alpha (`samples_per_pixel == 2`) and 4-band RGBA (`samples_per_pixel >= 4`).
pub fn alpha_channel_index(samples_per_pixel: usize, extra_samples: &[u16]) -> Option<usize> {
    if samples_per_pixel < 2 || extra_samples.is_empty() || extra_samples.len() >= samples_per_pixel
    {
        return None;
    }
    let pos = extra_samples.iter().position(|&v| v == 1 || v == 2)?;
    let base_samples = samples_per_pixel - extra_samples.len();
    Some(base_samples + pos)
}

/// If a chunky multi-band tile carries an alpha channel via `ExtraSamples` (tag 338 = 1 or 2,
/// e.g. 2-band Gray+Alpha or 4-band RGBA), mask transparent pixels (`alpha == 0`) to `f32::NAN`
/// in the color bands and recompute their `BlockStats` so every downstream kernel renders alpha
/// identically to TiTiler.
fn apply_rgba_alpha_if_present(bands: &mut [DecodedBand], ifd: &IfdMeta) {
    if ifd.planar_config != PlanarConfig::Chunky {
        return;
    }
    let Some(alpha_idx) = alpha_channel_index(bands.len(), &ifd.extra_samples) else {
        return;
    };
    // O(1) fast-path: if the alpha band has no nodata and min > 0, every pixel is opaque!
    let alpha_stats = bands[alpha_idx].stats;
    if alpha_stats.no_nodata && alpha_stats.min > 0.0 {
        return;
    }
    let n = bands[alpha_idx].data.len();
    let alpha: Vec<f32> = (0..n).map(|i| bands[alpha_idx].data.get_f32(i)).collect();
    if alpha.iter().all(|&a| a > 0.0) {
        return;
    }
    for (idx, b) in bands.iter_mut().enumerate() {
        if idx != alpha_idx {
            apply_alpha_slice_to_band(b, &alpha);
        }
    }
}

/// Mask transparent pixels (`alpha == 0` or `NaN`) to `f32::NAN` in a single-band decoded tile
/// (used for `PlanarConfig::Planar` Gray+Alpha / RGBA COGs where the alpha channel lives in a separate plane).
pub fn apply_alpha_band_to_tile(tile: &mut DecodedTile, alpha: &DecodedBand) {
    if alpha.stats.no_nodata && alpha.stats.min > 0.0 {
        return;
    }
    let n = alpha.data.len();
    let alpha_vals: Vec<f32> = (0..n).map(|i| alpha.data.get_f32(i)).collect();
    if alpha_vals.iter().all(|&a| a > 0.0) {
        return;
    }
    for b in &mut tile.bands {
        apply_alpha_slice_to_band(b, &alpha_vals);
    }
}

fn apply_alpha_slice_to_band(b: &mut DecodedBand, alpha: &[f32]) {
    let n = alpha.len();
    let slice = b.data.make_f32_mut();
    let mut all_nd = true;
    let mut no_nd = true;
    for i in 0..n.min(slice.len()) {
        if alpha[i] == 0.0 || alpha[i].is_nan() {
            slice[i] = f32::NAN;
            no_nd = false;
        } else if slice[i].is_nan() {
            no_nd = false;
        } else {
            all_nd = false;
        }
    }
    b.stats.all_nodata = all_nd;
    b.stats.no_nodata = no_nd;
}

/// Decompress a 1-bit (packed MSB-first per row) or 8-bit internal transparency mask tile
/// (`NewSubfileType & 4 != 0`) into a `tile_width * tile_height` byte mask (`0` = transparent, `255` = opaque).
pub fn decode_mask_tile(compressed: &[u8], mask: &MaskIfdMeta) -> Result<Vec<u8>, CodecError> {
    decode_mask_tile_rows(compressed, mask, mask.tile_height)
}

/// Decompress a 1-bit or 8-bit internal transparency mask tile or strip with an explicit row count
/// `actual_rows` (which is `< mask.tile_height` for the final partial strip of a stripped mask IFD).
pub fn decode_mask_tile_rows(
    compressed: &[u8],
    mask: &MaskIfdMeta,
    actual_rows: u32,
) -> Result<Vec<u8>, CodecError> {
    let w = mask.tile_width as usize;
    let h = actual_rows.min(mask.tile_height).max(1) as usize;
    let row_bytes = if mask.bits_per_sample == 1 {
        w.div_ceil(8)
    } else {
        w
    };
    let expected = row_bytes
        .checked_mul(h)
        .filter(|&b| b > 0 && b <= MAX_DECODED_TILE_BYTES)
        .ok_or(CodecError::SizeMismatch {
            expected: MAX_DECODED_TILE_BYTES,
            got: compressed.len(),
        })?;

    let raw: Vec<u8> = CODEC_STATE.with(|cell| {
        let mut guard = cell.borrow_mut();
        let st = &mut *guard;
        let mut buf = vec![0u8; expected];
        match mask.compression {
            Compression::None => {
                let n = compressed.len().min(expected);
                buf[..n].copy_from_slice(&compressed[..n]);
            }
            Compression::Deflate => {
                if let Err(e1) = st.deflate.zlib_decompress(compressed, &mut buf) {
                    st.deflate
                        .deflate_decompress(compressed, &mut buf)
                        .map_err(|_| CodecError::Deflate(format!("{e1:?}")))?;
                }
            }
            Compression::Zstd => {
                let code = unsafe {
                    zstd_sys::ZSTD_decompressDCtx(
                        st.zstd_dctx,
                        buf.as_mut_ptr() as *mut std::ffi::c_void,
                        expected,
                        compressed.as_ptr() as *const std::ffi::c_void,
                        compressed.len(),
                    )
                };
                if unsafe { zstd_sys::ZSTD_isError(code) } != 0 {
                    return Err(CodecError::Zstd(code));
                }
            }
            Compression::Lzw => {
                st.lzw.decode(compressed, &mut buf)?;
            }
            _ => {
                return Err(CodecError::Deflate(
                    "unsupported compression in internal mask IFD".into(),
                ));
            }
        }
        Ok(buf)
    })?;

    if mask.bits_per_sample == 1 {
        let mut out = vec![0u8; w * h];
        for r in 0..h {
            let src_row = &raw[r * row_bytes..(r + 1) * row_bytes];
            let dst_row = &mut out[r * w..(r + 1) * w];
            for c in 0..w {
                let bit = (src_row[c >> 3] >> (7 - (c & 7))) & 1;
                dst_row[c] = if bit != 0 { 255 } else { 0 };
            }
        }
        Ok(out)
    } else {
        let mut out = raw;
        if mask.predictor == Predictor::HorizontalInt {
            for r in 0..h {
                let row = &mut out[r * w..(r + 1) * w];
                let mut acc = 0u8;
                for px in row.iter_mut() {
                    acc = acc.wrapping_add(*px);
                    *px = acc;
                }
            }
        }
        Ok(out)
    }
}

/// Apply an unpacked `0`/`255` transparency mask to every band of a `DecodedTile`,
/// setting masked pixels to `f32::NAN` and updating `BlockStats`.
pub fn apply_mask_to_tile(tile: &mut DecodedTile, mask: &[u8]) {
    if mask.iter().all(|&m| m != 0) {
        return;
    }
    for b in &mut tile.bands {
        let slice = b.data.make_f32_mut();
        let n = slice.len().min(mask.len());
        let mut all_nd = true;
        let mut no_nd = true;
        for i in 0..n {
            if mask[i] == 0 {
                slice[i] = f32::NAN;
                no_nd = false;
            } else if slice[i].is_nan() {
                no_nd = false;
            } else {
                all_nd = false;
            }
        }
        b.stats.all_nodata = all_nd;
        b.stats.no_nodata = no_nd;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_tiff::Affine2D;

    #[test]
    fn hostile_tile_dimension_overflow_is_rejected_without_allocation() {
        // 2_147_418_113 * 2_147_549_185 * 1 * 4 wraps mod 2^64 to 4 bytes in release mode!
        let ident = Affine2D {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 1.0,
            f: 0.0,
        };
        let ifd = IfdMeta {
            width: 2_147_418_113,
            height: 2_147_549_185,
            tile_width: 2_147_418_113,
            tile_height: 2_147_549_185,
            tiles_across: 1,
            tiles_down: 1,
            samples_per_pixel: 1,
            sample_type: SampleType::Float32,
            compression: Compression::None,
            predictor: Predictor::None,
            planar_config: PlanarConfig::Chunky,
            photometric: 1,
            extra_samples: Vec::new(),
            is_stripped: false,
            tile_offsets: Box::new([0]),
            tile_byte_counts: Box::new([4]),
            jpeg_tables: None,
            mask_ifd: None,
            pixel_to_crs: ident,
            crs_to_pixel: ident,
        };
        assert!(matches!(
            decode_tile(&[0u8; 4], &ifd, ByteOrder::Little, None),
            Err(CodecError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn apply_alpha_band_to_tile_masks_zero_alpha_and_updates_stats() {
        let mut tile = DecodedTile {
            bands: vec![DecodedBand {
                width: 2,
                height: 2,
                data: BandData::U8 {
                    data: vec![10u8, 20, 30, 40].into_boxed_slice(),
                    nodata: None,
                },
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 10.0,
                    max: 40.0,
                },
            }],
        };
        let alpha = DecodedBand {
            width: 2,
            height: 2,
            data: BandData::U8 {
                data: vec![255u8, 0, 255, 0].into_boxed_slice(),
                nodata: None,
            },
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 0.0,
                max: 255.0,
            },
        };
        apply_alpha_band_to_tile(&mut tile, &alpha);
        let b = &tile.bands[0];
        assert!(!b.stats.all_nodata);
        assert!(!b.stats.no_nodata);
        assert_eq!(b.data.get_f32(0), 10.0);
        assert!(b.data.get_f32(1).is_nan());
        assert_eq!(b.data.get_f32(2), 30.0);
        assert!(b.data.get_f32(3).is_nan());
    }

    #[test]
    fn decode_mask_tile_undoes_horizontal_predictor_on_8bit_mask_and_gray_alpha_index() {
        // Horizontal-differenced opaque 4x2 8-bit mask: [255, 0, 0, 0] per row -> [255, 255, 255, 255]
        let mask_meta = MaskIfdMeta {
            width: 4,
            height: 2,
            tile_width: 4,
            tile_height: 2,
            tiles_across: 1,
            tiles_down: 1,
            bits_per_sample: 8,
            compression: Compression::None,
            predictor: Predictor::HorizontalInt,
            is_stripped: false,
            tile_offsets: Box::new([0]),
            tile_byte_counts: Box::new([8]),
        };
        let raw_diff = [255u8, 0, 0, 0, 0, 255, 0, 1];
        let decoded = decode_mask_tile(&raw_diff, &mask_meta).unwrap();
        assert_eq!(decoded, vec![255, 255, 255, 255, 0, 255, 255, 0]);

        // 2-band Gray+Alpha and 4-band RGBA ExtraSamples index detection
        assert_eq!(alpha_channel_index(2, &[2]), Some(1));
        assert_eq!(alpha_channel_index(4, &[1]), Some(3));
        assert_eq!(alpha_channel_index(4, &[0]), None);
    }

    #[test]
    fn decode_tile_rows_clamps_partial_final_strip_stats_and_rgba_alpha() {
        // 2x5 stripped RGBA image with tile_height=4: final strip has actual_rows=1 (2 pixels = 8 bytes).
        // All alpha values in that 1 row are 255 (opaque), so stats must stay no_nodata=true
        // and min=100.0 (not polluted by zero-padded rows 1..4!).
        let ifd = IfdMeta {
            width: 2,
            height: 5,
            tile_width: 2,
            tile_height: 4,
            tiles_across: 1,
            tiles_down: 2,
            samples_per_pixel: 4,
            sample_type: SampleType::Uint8,
            planar_config: PlanarConfig::Chunky,
            compression: Compression::None,
            predictor: Predictor::None,
            photometric: 2,
            extra_samples: vec![2],
            jpeg_tables: None,
            mask_ifd: None,
            is_stripped: true,
            pixel_to_crs: cogzilla_tiff::Affine2D {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 1.0,
                f: 0.0,
            },
            crs_to_pixel: cogzilla_tiff::Affine2D {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 1.0,
                f: 0.0,
            },
            tile_offsets: Box::new([0, 32]),
            tile_byte_counts: Box::new([32, 8]),
        };
        assert_eq!(ifd.block_height(0), 4);
        assert_eq!(ifd.block_height(1), 1);

        let raw_last_strip = [100u8, 120, 140, 255, 200, 210, 220, 255];
        let tile = decode_tile_rows(
            &raw_last_strip,
            &ifd,
            ByteOrder::Little,
            None,
            ifd.block_height(1),
        )
        .unwrap();
        assert_eq!(tile.bands[0].height, 1);
        assert_eq!(tile.bands[0].data.len(), 2);
        assert!(tile.bands[0].stats.no_nodata);
        assert_eq!(tile.bands[0].stats.min, 100.0);
        assert_eq!(tile.bands[0].stats.max, 200.0);
        assert_eq!(tile.bands[3].stats.min, 255.0);
    }
}
