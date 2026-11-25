pub mod lzw;
pub mod predictor;
pub mod slab;

use std::cell::RefCell;

use cogzilla_tiff::{
    ByteOrder, Compression, IfdMeta, MaskIfdMeta, PlanarConfig, Predictor, SampleType,
};
use thiserror::Error;

pub use slab::{AlignedF32Slab, BandData, BlockStats, DecodedBand, DecodedTile};

/// Decode TIFF PackBits (compression code 32773) RLE stream into `dst`.
fn decode_packbits(src: &[u8], dst: &mut [u8]) -> Result<usize, CodecError> {
    let mut si = 0usize;
    let mut di = 0usize;
    while si < src.len() && di < dst.len() {
        let hdr = src[si] as i8;
        si += 1;
        if hdr >= 0 {
            let count = (hdr as usize) + 1;
            if si + count > src.len() {
                return Err(CodecError::Deflate("truncated PackBits literal run".into()));
            }
            let copy = count.min(dst.len() - di);
            dst[di..di + copy].copy_from_slice(&src[si..si + copy]);
            si += count;
            di += copy;
        } else if hdr != -128 {
            let count = (1i16 - hdr as i16) as usize;
            if si >= src.len() {
                return Err(CodecError::Deflate("truncated PackBits repeat run".into()));
            }
            let byte = src[si];
            si += 1;
            let copy = count.min(dst.len() - di);
            dst[di..di + copy].fill(byte);
            di += copy;
        }
    }
    Ok(di)
}

/// Compress a byte slice into a gzip stream (RFC 1952) using `libdeflater` at default level.
pub fn compress_gzip(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
    let max_sz = compressor.gzip_compress_bound(input.len());
    let mut out = vec![0u8; max_sz];
    let n = compressor
        .gzip_compress(input, &mut out)
        .map_err(|e| format!("gzip compress error: {e:?}"))?;
    out.truncate(n);
    Ok(out)
}

/// Decompress a gzip-encoded buffer using `libdeflater` with an upper bound on decompressed size
/// to guard against decompression bombs.
pub fn decompress_gzip_limited(input: &[u8], max_out_bytes: usize) -> Result<Vec<u8>, String> {
    if input.len() < 18 || input[0] != 0x1f || input[1] != 0x8b {
        return Err("invalid gzip header".into());
    }
    let isize_hint = u32::from_le_bytes([
        input[input.len() - 4],
        input[input.len() - 3],
        input[input.len() - 2],
        input[input.len() - 1],
    ]) as usize;
    let mut cap = isize_hint
        .max(input.len().saturating_mul(2))
        .clamp(4096, max_out_bytes);
    let mut decompressor = libdeflater::Decompressor::new();
    loop {
        let mut out = vec![0u8; cap];
        match decompressor.gzip_decompress(input, &mut out) {
            Ok(n) => {
                out.truncate(n);
                return Ok(out);
            }
            Err(libdeflater::DecompressionError::InsufficientSpace) => {
                if cap >= max_out_bytes {
                    return Err(format!(
                        "gzip payload exceeds {} MiB limit",
                        max_out_bytes / (1024 * 1024)
                    ));
                }
                cap = cap.saturating_mul(2).min(max_out_bytes);
            }
            Err(e) => return Err(format!("gzip decompression failed: {e:?}")),
        }
    }
}

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
    jpeg_merge: Vec<u8>,
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
            jpeg_merge: Vec::new(),
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

    // Never apply GDAL_NODATA to an ExtraSamples alpha channel (e.g. GDAL_NODATA=255 on RGBA
    // would turn opaque alpha=255 into NaN and blank every pixel). For U8/U16, raw samples stay
    // intact so apply_rgba_alpha_if_present simply clears nodata on the alpha band; for
    // F32-backed dtypes we defer nodata masking to the non-alpha color bands only.
    let chunky_alpha_idx = if ifd.planar_config == PlanarConfig::Chunky {
        alpha_channel_index(bands_in_tile, &ifd.extra_samples)
    } else {
        None
    };
    let defer_color_nodata = chunky_alpha_idx.is_some()
        && nodata.is_some()
        && !matches!(ifd.sample_type, SampleType::Uint8 | SampleType::Uint16);
    let recon_nodata = if defer_color_nodata { None } else { nodata };
    let deferred_nodata = if defer_color_nodata { nodata } else { None };

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
                recon_nodata,
            ),
            SampleType::Uint16 => predictor::reconstruct_u16(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                recon_nodata,
            ),
            SampleType::Int16 => predictor::reconstruct_i16(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                recon_nodata,
            ),
            SampleType::Uint8 => predictor::reconstruct_u8(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                false,
                recon_nodata,
            ),
            SampleType::Int8 => predictor::reconstruct_i8(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                false,
                recon_nodata,
            ),
            SampleType::Uint32 => predictor::reconstruct_int32(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                false,
                recon_nodata,
            ),
            SampleType::Int32 => predictor::reconstruct_int32(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                true,
                false,
                recon_nodata,
            ),
            SampleType::Float64 => predictor::reconstruct_f64_uncompressed(
                &compressed[..expected_bytes],
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                recon_nodata,
            ),
        };
        apply_rgba_alpha_if_present(&mut bands, ifd, deferred_nodata);
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
            Compression::PackBits => {
                let written = decode_packbits(compressed, raw_slice)?;
                if written < expected_bytes {
                    if ifd.is_stripped {
                        raw_slice[written..].fill(0);
                    } else {
                        return Err(CodecError::SizeMismatch {
                            expected: expected_bytes,
                            got: written,
                        });
                    }
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
                if ifd.sample_type != SampleType::Uint8 {
                    return Err(CodecError::Jpeg(format!(
                        "JPEG compression requires uint8 samples, got {:?}",
                        ifd.sample_type
                    )));
                }
                // Merge shared TIFF JPEGTables (tag 347) with abbreviated tile bitstream if present
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
                        st.jpeg_merge.clear();
                        st.jpeg_merge.reserve(t_end + (compressed.len() - c_start));
                        st.jpeg_merge.extend_from_slice(&tables[..t_end]);
                        st.jpeg_merge.extend_from_slice(&compressed[c_start..]);
                        &st.jpeg_merge
                    } else {
                        compressed
                    }
                } else {
                    compressed
                };

                use zune_jpeg::zune_core::colorspace::ColorSpace;
                use zune_jpeg::zune_core::options::DecoderOptions;

                let opts = DecoderOptions::default()
                    .set_max_width((ifd.tile_width as usize).next_multiple_of(16))
                    .set_max_height((tile_h as usize).next_multiple_of(16));
                let mut dec = zune_jpeg::JpegDecoder::new_with_options(jpeg_bytes, opts);
                dec.decode_headers()
                    .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                let out_cs = match (bands_in_tile, ifd.photometric) {
                    (1, _) => ColorSpace::Luma,
                    // Photometric=2 (RGB): match the stream's own input colorspace (RGB if an Adobe
                    // APP14 transform=0 marker is present, YCbCr if stripped) so zune-jpeg performs
                    // zero colorspace conversion instead of failing on RGB->YCbCr.
                    (3, 2) => dec.get_input_colorspace().unwrap_or(ColorSpace::YCbCr),
                    (3, _) => ColorSpace::RGB,
                    (4, _) => dec.get_input_colorspace().unwrap_or(ColorSpace::RGBA),
                    _ => {
                        return Err(CodecError::Jpeg(format!(
                            "unsupported JPEG band count: {bands_in_tile}"
                        )));
                    }
                };
                if out_cs.num_components() != bands_in_tile {
                    return Err(CodecError::Jpeg(format!(
                        "JPEG colorspace {out_cs:?} has {} components, expected {bands_in_tile}",
                        out_cs.num_components()
                    )));
                }
                let (jw, jh) = dec.dimensions().ok_or_else(|| {
                    CodecError::Jpeg("JPEG header missing dimensions".into())
                })?;
                let decoded_jpeg_bytes = jw
                    .checked_mul(jh)
                    .and_then(|px| px.checked_mul(bands_in_tile))
                    .filter(|&b| b > 0 && b <= MAX_DECODED_TILE_BYTES)
                    .ok_or_else(|| {
                        CodecError::Jpeg(format!(
                            "JPEG dimensions {jw}x{jh}x{bands_in_tile} exceed max decoded tile size"
                        ))
                    })?;
                let _ = decoded_jpeg_bytes;
                dec.set_options(opts.jpeg_set_out_colorspace(out_cs));
                let tile_w = ifd.tile_width as usize;
                let tile_h_usize = tile_h as usize;
                if jw == tile_w {
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
                        if pixels.len() < expected_bytes {
                            return Err(CodecError::Jpeg(format!(
                                "decoded JPEG buffer too small: {} < {expected_bytes}",
                                pixels.len()
                            )));
                        }
                        raw_slice[..expected_bytes].copy_from_slice(&pixels[..expected_bytes]);
                    }
                } else {
                    // MCU-padded or narrower JPEG stream: copy row-by-row so rows don't shear.
                    let pixels = dec
                        .decode()
                        .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                    let src_stride = jw * bands_in_tile;
                    let dst_stride = tile_w * bands_in_tile;
                    let copy_bytes = jw.min(tile_w) * bands_in_tile;
                    let copy_rows = jh.min(tile_h_usize);
                    raw_slice[..expected_bytes].fill(0);
                    for y in 0..copy_rows {
                        let src_off = y * src_stride;
                        if src_off + copy_bytes > pixels.len() {
                            break;
                        }
                        let dst_off = y * dst_stride;
                        raw_slice[dst_off..dst_off + copy_bytes]
                            .copy_from_slice(&pixels[src_off..src_off + copy_bytes]);
                    }
                }
            }
            Compression::Webp => {
                if ifd.sample_type != SampleType::Uint8 || (bands_in_tile != 3 && bands_in_tile != 4)
                {
                    return Err(CodecError::Webp);
                }
                let mut webp_w: std::ffi::c_int = 0;
                let mut webp_h: std::ffi::c_int = 0;
                let info_ok = unsafe {
                    libwebp_sys::WebPGetInfo(
                        compressed.as_ptr(),
                        compressed.len(),
                        &mut webp_w,
                        &mut webp_h,
                    )
                };
                if info_ok == 0
                    || webp_w <= 0
                    || webp_h <= 0
                    || webp_w as u32 > ifd.tile_width
                    || webp_h as u32 > tile_h
                {
                    return Err(CodecError::Webp);
                }
                // Zero thread-local scratch if WebP dimensions are smaller than the tile/strip so
                // unwritten rows/columns never leak bytes from prior tiles on this thread.
                if webp_w as u32 != ifd.tile_width || webp_h as u32 != tile_h {
                    raw_slice.fill(0);
                }
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
                recon_nodata,
            ),
            (SampleType::Float32, Predictor::HorizontalInt) => predictor::reconstruct_predictor2(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                recon_nodata,
            ),
            (SampleType::Float32, Predictor::FloatingPoint) => predictor::reconstruct_predictor3(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                recon_nodata,
            ),
            (SampleType::Uint16, Predictor::None) => predictor::reconstruct_u16(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                recon_nodata,
            ),
            (SampleType::Uint16, Predictor::HorizontalInt) => predictor::reconstruct_u16(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                true,
                recon_nodata,
            ),
            (SampleType::Uint16, Predictor::FloatingPoint) => predictor::reconstruct_u16(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                recon_nodata,
            ),
            (SampleType::Int16, p) => predictor::reconstruct_i16(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                p == Predictor::HorizontalInt,
                recon_nodata,
            ),
            (SampleType::Uint8, p) => predictor::reconstruct_u8(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                p == Predictor::HorizontalInt,
                recon_nodata,
            ),
            (SampleType::Int8, p) => predictor::reconstruct_i8(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                p == Predictor::HorizontalInt,
                recon_nodata,
            ),
            (SampleType::Uint32, p) => predictor::reconstruct_int32(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                false,
                p == Predictor::HorizontalInt,
                recon_nodata,
            ),
            (SampleType::Int32, p) => predictor::reconstruct_int32(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                true,
                p == Predictor::HorizontalInt,
                recon_nodata,
            ),
            (SampleType::Float64, p) => predictor::reconstruct_f64(
                raw_slice,
                ifd.tile_width,
                tile_h,
                bands_in_tile,
                byte_order,
                p,
                recon_nodata,
            ),
        };

        apply_rgba_alpha_if_present(&mut bands, ifd, deferred_nodata);
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
fn apply_rgba_alpha_if_present(
    bands: &mut [DecodedBand],
    ifd: &IfdMeta,
    deferred_color_nodata: Option<f32>,
) {
    if ifd.planar_config != PlanarConfig::Chunky {
        return;
    }
    let Some(alpha_idx) = alpha_channel_index(bands.len(), &ifd.extra_samples) else {
        return;
    };
    // Clear any GDAL_NODATA metadata accidentally attached to a U8/U16 alpha band so e.g.
    // GDAL_NODATA=255 does not treat opaque alpha=255 as NaN.
    match &mut bands[alpha_idx].data {
        BandData::U8 { data, nodata } if nodata.is_some() => {
            *nodata = None;
            let mut mn = u8::MAX;
            let mut mx = u8::MIN;
            for &v in data.iter() {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            bands[alpha_idx].stats = BlockStats {
                all_nodata: data.is_empty(),
                no_nodata: true,
                min: if data.is_empty() { f32::NAN } else { mn as f32 },
                max: if data.is_empty() { f32::NAN } else { mx as f32 },
            };
        }
        BandData::U16 { data, nodata } if nodata.is_some() => {
            *nodata = None;
            let mut mn = u16::MAX;
            let mut mx = u16::MIN;
            for &v in data.iter() {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            bands[alpha_idx].stats = BlockStats {
                all_nodata: data.is_empty(),
                no_nodata: true,
                min: if data.is_empty() { f32::NAN } else { mn as f32 },
                max: if data.is_empty() { f32::NAN } else { mx as f32 },
            };
        }
        BandData::RgbU8Channel { rgb, channel, nodata } if nodata.is_some() => {
            *nodata = None;
            let ch = *channel as usize;
            let n = rgb.len() / 3;
            let mut mn = u8::MAX;
            let mut mx = u8::MIN;
            for i in 0..n {
                let v = rgb[i * 3 + ch];
                mn = mn.min(v);
                mx = mx.max(v);
            }
            bands[alpha_idx].stats = BlockStats {
                all_nodata: n == 0,
                no_nodata: true,
                min: if n == 0 { f32::NAN } else { mn as f32 },
                max: if n == 0 { f32::NAN } else { mx as f32 },
            };
        }
        _ => {}
    }

    // Apply deferred GDAL_NODATA to non-alpha F32-backed color bands only.
    if let Some(nd_f32) = deferred_color_nodata {
        for (idx, b) in bands.iter_mut().enumerate() {
            if idx == alpha_idx {
                continue;
            }
            let slice = b.data.make_f32_mut();
            let mut all_nd = true;
            let mut no_nd = true;
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for v in slice.iter_mut() {
                if v.is_nan() || (!nd_f32.is_nan() && *v == nd_f32) {
                    *v = f32::NAN;
                    no_nd = false;
                } else {
                    all_nd = false;
                    if *v < mn {
                        mn = *v;
                    }
                    if *v > mx {
                        mx = *v;
                    }
                }
            }
            b.stats = BlockStats {
                all_nodata: all_nd,
                no_nodata: no_nd,
                min: if all_nd { f32::NAN } else { mn },
                max: if all_nd { f32::NAN } else { mx },
            };
        }
    }

    // O(1) fast-path: if the alpha band has no nodata and min > 0, every pixel is opaque!
    let alpha_stats = bands[alpha_idx].stats;
    if alpha_stats.no_nodata && alpha_stats.min > 0.0 {
        return;
    }
    let n = bands[alpha_idx].data.len();
    if (0..n).all(|i| bands[alpha_idx].data.get_f32(i) > 0.0) {
        return;
    }
    let (before, rest) = bands.split_at_mut(alpha_idx);
    let (alpha_band, after) = rest.split_first_mut().unwrap();
    for b in before.iter_mut().chain(after.iter_mut()) {
        apply_alpha_data_to_band(b, &alpha_band.data);
    }
}

/// Mask transparent pixels (`alpha == 0` or `NaN`) to `f32::NAN` in a single-band decoded tile
/// (used for `PlanarConfig::Planar` Gray+Alpha / RGBA COGs where the alpha channel lives in a separate plane).
pub fn apply_alpha_band_to_tile(tile: &mut DecodedTile, alpha: &DecodedBand) {
    if alpha.stats.no_nodata && alpha.stats.min > 0.0 {
        return;
    }
    let n = alpha.data.len();
    if (0..n).all(|i| alpha.data.get_f32(i) > 0.0) {
        return;
    }
    for b in &mut tile.bands {
        apply_alpha_data_to_band(b, &alpha.data);
    }
}

fn apply_alpha_data_to_band(b: &mut DecodedBand, alpha: &BandData) {
    let n = alpha.len();
    let slice = b.data.make_f32_mut();
    let mut all_nd = true;
    let mut no_nd = true;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for i in 0..n.min(slice.len()) {
        let a = alpha.get_f32(i);
        if a == 0.0 || a.is_nan() {
            slice[i] = f32::NAN;
            no_nd = false;
        } else {
            let v = slice[i];
            if v.is_nan() {
                no_nd = false;
            } else {
                all_nd = false;
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
        }
    }
    b.stats.all_nodata = all_nd;
    b.stats.no_nodata = no_nd;
    if all_nd {
        b.stats.min = f32::NAN;
        b.stats.max = f32::NAN;
    } else {
        b.stats.min = mn;
        b.stats.max = mx;
    }
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
    // Enforce MAX_DECODED_TILE_BYTES on the unpacked output pixel count (1 byte/px),
    // not just packed 1-bit row_bytes (which would allow 8x larger 1 GiB allocations).
    let unpacked_bytes = w
        .checked_mul(h)
        .filter(|&px| px > 0 && px <= MAX_DECODED_TILE_BYTES)
        .ok_or(CodecError::SizeMismatch {
            expected: MAX_DECODED_TILE_BYTES,
            got: compressed.len(),
        })?;
    let row_bytes = if mask.bits_per_sample == 1 {
        w.div_ceil(8)
    } else {
        w
    };
    let expected = row_bytes
        .checked_mul(h)
        .filter(|&b| b > 0 && b <= unpacked_bytes)
        .ok_or(CodecError::SizeMismatch {
            expected: MAX_DECODED_TILE_BYTES,
            got: compressed.len(),
        })?;

    let decompress_into = |st: &mut ThreadCodecState, dst: &mut [u8]| -> Result<(), CodecError> {
        let written = match mask.compression {
            Compression::None => {
                let n = compressed.len().min(dst.len());
                dst[..n].copy_from_slice(&compressed[..n]);
                n
            }
            Compression::Deflate => match st.deflate.zlib_decompress(compressed, dst) {
                Ok(n) => n,
                Err(e1) => st
                    .deflate
                    .deflate_decompress(compressed, dst)
                    .map_err(|_| CodecError::Deflate(format!("{e1:?}")))?,
            },
            Compression::Zstd => {
                let code = unsafe {
                    zstd_sys::ZSTD_decompressDCtx(
                        st.zstd_dctx,
                        dst.as_mut_ptr() as *mut std::ffi::c_void,
                        dst.len(),
                        compressed.as_ptr() as *const std::ffi::c_void,
                        compressed.len(),
                    )
                };
                if unsafe { zstd_sys::ZSTD_isError(code) } != 0 {
                    return Err(CodecError::Zstd(code));
                }
                code
            }
            Compression::Lzw => st.lzw.decode(compressed, dst)?,
            Compression::PackBits => decode_packbits(compressed, dst)?,
            _ => {
                return Err(CodecError::Deflate(
                    "unsupported compression in internal mask IFD".into(),
                ));
            }
        };
        if written < dst.len() {
            dst[written..].fill(0);
        }
        Ok(())
    };

    if mask.bits_per_sample == 1 {
        CODEC_STATE.with(|cell| {
            let mut guard = cell.borrow_mut();
            let st = &mut *guard;
            if st.scratch.len() < expected {
                st.scratch.resize(expected, 0);
            }
            let mut tmp_scratch = std::mem::take(&mut st.scratch);
            let res = decompress_into(st, &mut tmp_scratch[..expected]);
            let unpacked = res.map(|()| {
                let packed = &tmp_scratch[..expected];
                if w.is_multiple_of(8) && packed.iter().all(|&b| b == 0xFF) {
                    return vec![255u8; w * h];
                }
                let mut out = vec![0u8; w * h];
                for r in 0..h {
                    let src_row = &packed[r * row_bytes..(r + 1) * row_bytes];
                    let dst_row = &mut out[r * w..(r + 1) * w];
                    for c in 0..w {
                        let bit = (src_row[c >> 3] >> (7 - (c & 7))) & 1;
                        dst_row[c] = if bit != 0 { 255 } else { 0 };
                    }
                }
                out
            });
            st.scratch = if tmp_scratch.capacity() > 4 * 1024 * 1024 {
                Vec::with_capacity(512 * 512 * 4)
            } else {
                tmp_scratch
            };
            unpacked
        })
    } else {
        let mut out = vec![0u8; expected];
        CODEC_STATE.with(|cell| {
            let mut guard = cell.borrow_mut();
            decompress_into(&mut guard, &mut out)
        })?;
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
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for i in 0..n {
            if mask[i] == 0 {
                slice[i] = f32::NAN;
                no_nd = false;
            } else {
                let v = slice[i];
                if v.is_nan() {
                    no_nd = false;
                } else {
                    all_nd = false;
                    if v < mn {
                        mn = v;
                    }
                    if v > mx {
                        mx = v;
                    }
                }
            }
        }
        b.stats.all_nodata = all_nd;
        b.stats.no_nodata = no_nd;
        if all_nd {
            b.stats.min = f32::NAN;
            b.stats.max = f32::NAN;
        } else {
            b.stats.min = mn;
            b.stats.max = mx;
        }
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

    #[test]
    fn one_bit_mask_enforces_unpacked_byte_limit_and_recomputes_stats() {
        let huge_mask = MaskIfdMeta {
            width: 32768,
            height: 32768,
            tile_width: 32768,
            tile_height: 32768,
            tiles_across: 1,
            tiles_down: 1,
            bits_per_sample: 1,
            compression: Compression::None,
            predictor: Predictor::None,
            is_stripped: false,
            tile_offsets: Box::new([0]),
            tile_byte_counts: Box::new([1]),
        };
        assert!(
            matches!(decode_mask_tile(&[0u8], &huge_mask), Err(CodecError::SizeMismatch { .. })),
            "32768x32768 1-bit mask unpacks to 1 GiB and must be rejected"
        );

        let mut slab = AlignedF32Slab::new_uninit(4);
        slab.as_mut_slice().copy_from_slice(&[1.0, 10.0, 20.0, 999.0]);
        let mut tile = DecodedTile {
            bands: vec![DecodedBand {
                width: 2,
                height: 2,
                data: BandData::F32(slab),
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 1.0,
                    max: 999.0,
                },
            }],
        };
        apply_mask_to_tile(&mut tile, &[0, 255, 255, 0]);
        assert!(!tile.bands[0].stats.all_nodata);
        assert!(!tile.bands[0].stats.no_nodata);
        assert_eq!(tile.bands[0].stats.min, 10.0);
        assert_eq!(tile.bands[0].stats.max, 20.0);

        apply_mask_to_tile(&mut tile, &[0, 0, 0, 0]);
        assert!(tile.bands[0].stats.all_nodata);
        assert!(tile.bands[0].stats.min.is_nan());
        assert!(tile.bands[0].stats.max.is_nan());
    }

    #[test]
    fn chunky_rgba_and_gray_alpha_ignore_gdal_nodata_on_alpha_channel() {
        let mut ifd_u8 = IfdMeta {
            width: 2,
            height: 1,
            tile_width: 2,
            tile_height: 1,
            tiles_across: 1,
            tiles_down: 1,
            samples_per_pixel: 4,
            sample_type: SampleType::Uint8,
            planar_config: PlanarConfig::Chunky,
            compression: Compression::None,
            predictor: Predictor::None,
            photometric: 2,
            extra_samples: vec![2],
            jpeg_tables: None,
            mask_ifd: None,
            is_stripped: false,
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
            tile_offsets: Box::new([0]),
            tile_byte_counts: Box::new([8]),
        };
        // Pixel 0: opaque (R=10, G=20, B=30, A=255), Pixel 1: transparent (R=40, G=50, B=60, A=0)
        // With GDAL_NODATA = 255, opaque alpha=255 MUST NOT become NaN and blank pixel 0!
        let raw_u8 = [10u8, 20, 30, 255, 40, 50, 60, 0];
        let tile_u8 = decode_tile(&raw_u8, &ifd_u8, ByteOrder::Little, Some(255.0)).unwrap();
        assert_eq!(tile_u8.bands[0].data.get_f32(0), 10.0);
        assert_eq!(tile_u8.bands[1].data.get_f32(0), 20.0);
        assert_eq!(tile_u8.bands[2].data.get_f32(0), 30.0);
        assert!(tile_u8.bands[0].data.get_f32(1).is_nan());

        // Float32 Gray+Alpha with GDAL_NODATA = 1.0: opaque alpha=1.0 must stay opaque,
        // while color value 1.0 is masked to NaN.
        ifd_u8.samples_per_pixel = 2;
        ifd_u8.sample_type = SampleType::Float32;
        ifd_u8.photometric = 1;
        let mut raw_f32 = Vec::new();
        for v in [42.0f32, 1.0f32, 1.0f32, 1.0f32] {
            raw_f32.extend_from_slice(&v.to_le_bytes());
        }
        let tile_f32 = decode_tile(&raw_f32, &ifd_u8, ByteOrder::Little, Some(1.0)).unwrap();
        assert_eq!(tile_f32.bands[0].data.get_f32(0), 42.0);
        assert!(tile_f32.bands[0].data.get_f32(1).is_nan());
    }
}
