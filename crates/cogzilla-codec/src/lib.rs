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

/// Decompress and reconstruct a single internal COG tile into planar 64-byte-aligned `f32` bands
/// with precomputed `BlockStats`, reusing thread-local codec contexts and scratch memory.
pub fn decode_tile(
    compressed: &[u8],
    ifd: &IfdMeta,
    byte_order: ByteOrder,
    nodata: Option<f32>,
) -> Result<DecodedTile, CodecError> {
    let bands_in_tile = match ifd.planar_config {
        PlanarConfig::Chunky => ifd.samples_per_pixel as usize,
        PlanarConfig::Planar => 1,
    };
    let expected_bytes = (ifd.tile_width as usize)
        * (ifd.tile_height as usize)
        * bands_in_tile
        * ifd.sample_type.bytes_per_sample();

    // Fast path: Uncompressed + Predictor::None avoids touching thread-local scratch completely!
    if ifd.compression == Compression::None
        && ifd.predictor == Predictor::None
        && compressed.len() >= expected_bytes
    {
        let bands = match ifd.sample_type {
            SampleType::Float32 => predictor::reconstruct_predictor1(
                &compressed[..expected_bytes],
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                nodata,
            ),
            SampleType::Uint16 => predictor::reconstruct_u16(
                &compressed[..expected_bytes],
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                false,
                nodata,
            ),
            SampleType::Int16 => predictor::reconstruct_i16(
                &compressed[..expected_bytes],
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                false,
                nodata,
            ),
            SampleType::Uint8 => predictor::reconstruct_u8(
                &compressed[..expected_bytes],
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                false,
                nodata,
            ),
            SampleType::Int8 => predictor::reconstruct_i8(
                &compressed[..expected_bytes],
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                false,
                nodata,
            ),
            SampleType::Uint32 => predictor::reconstruct_int32(
                &compressed[..expected_bytes],
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                false,
                false,
                nodata,
            ),
            SampleType::Int32 => predictor::reconstruct_int32(
                &compressed[..expected_bytes],
                ifd.tile_width,
                ifd.tile_height,
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
                    ifd.tile_height,
                    bands_in_tile,
                    byte_order,
                    Predictor::None,
                    nodata,
                )
            }
        };
        return Ok(DecodedTile { bands });
    }

    CODEC_STATE.with(|cell| {
        let mut guard = cell.borrow_mut();
        let st = &mut *guard;
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
                let res = st.deflate.zlib_decompress(compressed, raw_slice);
                if let Err(e1) = res {
                    st.deflate
                        .deflate_decompress(compressed, raw_slice)
                        .map_err(|_| CodecError::Deflate(format!("{e1:?}")))?;
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
            }
            Compression::Lzw => {
                st.lzw.decode(compressed, raw_slice)?;
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

                let out_cs = match (bands_in_tile, ifd.photometric) {
                    (1, _) => ColorSpace::Luma,
                    // Photometric=2 (RGB) stored without YCbCr transform: skip YCbCr->RGB in decoder
                    (3, 2) => ColorSpace::YCbCr,
                    _ => ColorSpace::RGB,
                };
                let opts = DecoderOptions::default().jpeg_set_out_colorspace(out_cs);
                let mut dec = zune_jpeg::JpegDecoder::new_with_options(jpeg_bytes, opts);
                let pixels = dec
                    .decode()
                    .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                let copy_len = pixels.len().min(expected_bytes);
                raw_slice[..copy_len].copy_from_slice(&pixels[..copy_len]);
                if copy_len < expected_bytes {
                    raw_slice[copy_len..expected_bytes].fill(0);
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
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                nodata,
            ),
            (SampleType::Float32, Predictor::HorizontalInt) => predictor::reconstruct_predictor2(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                nodata,
            ),
            (SampleType::Float32, Predictor::FloatingPoint) => predictor::reconstruct_predictor3(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                nodata,
            ),
            (SampleType::Uint16, Predictor::None) => predictor::reconstruct_u16(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                false,
                nodata,
            ),
            (SampleType::Uint16, Predictor::HorizontalInt) => predictor::reconstruct_u16(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                true,
                nodata,
            ),
            (SampleType::Uint16, Predictor::FloatingPoint) => predictor::reconstruct_u16(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                false,
                nodata,
            ),
            (SampleType::Int16, p) => predictor::reconstruct_i16(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Uint8, p) => predictor::reconstruct_u8(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Int8, p) => predictor::reconstruct_i8(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Uint32, p) => predictor::reconstruct_int32(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                false,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Int32, p) => predictor::reconstruct_int32(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
                bands_in_tile,
                byte_order,
                true,
                p == Predictor::HorizontalInt,
                nodata,
            ),
            (SampleType::Float64, p) => predictor::reconstruct_f64(
                raw_slice,
                ifd.tile_width,
                ifd.tile_height,
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

/// If a chunky multi-band tile carries an alpha channel via `ExtraSamples` (tag 338 = 1 or 2,
/// e.g. 4-band RGBA), mask transparent pixels (`alpha == 0`) to `f32::NAN` in the color bands
/// and recompute their `BlockStats` so every downstream kernel renders alpha identically to TiTiler.
fn apply_rgba_alpha_if_present(bands: &mut [DecodedBand], ifd: &IfdMeta) {
    if ifd.planar_config != PlanarConfig::Chunky
        || bands.len() < 4
        || ifd.extra_samples.is_empty()
    {
        return;
    }
    let has_alpha = ifd.extra_samples.iter().any(|&v| v == 1 || v == 2);
    if !has_alpha {
        return;
    }
    let alpha_idx = bands.len() - 1;
    let n = bands[alpha_idx].data.len();
    let alpha: Vec<f32> = (0..n).map(|i| bands[alpha_idx].data.get_f32(i)).collect();
    if alpha.iter().all(|&a| a > 0.0) {
        return;
    }
    for b in &mut bands[..alpha_idx] {
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
}

/// Decompress a 1-bit (packed MSB-first per row) or 8-bit internal transparency mask tile
/// (`NewSubfileType & 4 != 0`) into a `tile_width * tile_height` byte mask (`0` = transparent, `255` = opaque).
pub fn decode_mask_tile(compressed: &[u8], mask: &MaskIfdMeta) -> Result<Vec<u8>, CodecError> {
    let w = mask.tile_width as usize;
    let h = mask.tile_height as usize;
    let row_bytes = if mask.bits_per_sample == 1 {
        w.div_ceil(8)
    } else {
        w
    };
    let expected = row_bytes * h;

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
        Ok(raw)
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
