pub mod lzw;
pub mod predictor;
pub mod slab;

use std::cell::RefCell;

use cogwheel_tiff::{ByteOrder, Compression, IfdMeta, PlanarConfig, Predictor, SampleType};
use thiserror::Error;

pub use slab::{AlignedF32Slab, BlockStats, DecodedBand, DecodedTile};

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("deflate decompression failed: {0}")]
    Deflate(String),
    #[error("zstd decompression failed: code {0}")]
    Zstd(usize),
    #[error("LZW decompression failed: {0}")]
    Lzw(&'static str),
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
    if ifd.compression == Compression::None && ifd.predictor == Predictor::None {
        if compressed.len() < expected_bytes {
            return Err(CodecError::SizeMismatch {
                expected: expected_bytes,
                got: compressed.len(),
            });
        }
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
                    return Err(CodecError::SizeMismatch {
                        expected: expected_bytes,
                        got: compressed.len(),
                    });
                }
                raw_slice.copy_from_slice(&compressed[..expected_bytes]);
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
        }

        let bands = match (ifd.sample_type, ifd.predictor) {
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

        Ok(DecodedTile { bands })
    })
}
