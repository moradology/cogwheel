use std::cell::RefCell;
use std::sync::OnceLock;

use bytes::Bytes;
use jpeg_encoder::{ColorType, Encoder};

thread_local! {
    static JPEG_RGB_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn encode_jpeg_rgb_raw(width: u32, height: u32, rgb: &[u8], quality: u8) -> Bytes {
    let (Ok(w), Ok(h)) = (u16::try_from(width), u16::try_from(height)) else {
        return Bytes::new();
    };
    let mut out = Vec::with_capacity(rgb.len() / 6);
    let enc = Encoder::new(&mut out, quality.clamp(1, 100));
    if enc.encode(rgb, w, h, ColorType::Rgb).is_err() {
        return Bytes::new();
    }
    Bytes::from(out)
}

/// Encode an RGB8 buffer to JPEG via pure-Rust SIMD `jpeg-encoder`.
/// Automatically emits 1-component `Luma` JPEG when all pixels have `r == g == b`.
pub fn encode_jpeg_rgb(width: u32, height: u32, rgb: &[u8], quality: u8) -> Bytes {
    let pixels = (width as usize).saturating_mul(height as usize);
    let is_gray = rgb
        .chunks_exact(3)
        .take(pixels)
        .all(|p| p[0] == p[1] && p[1] == p[2]);
    if is_gray {
        return JPEG_RGB_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.clear();
            scratch.resize(pixels, 0);
            for (src, dst) in rgb.chunks_exact(3).take(pixels).zip(scratch.iter_mut()) {
                *dst = src[0];
            }
            let encoded = encode_jpeg_gray(width, height, &scratch, quality);
            if scratch.capacity() > 4 * 1024 * 1024 {
                *scratch = Vec::new();
            }
            encoded
        });
    }
    encode_jpeg_rgb_raw(width, height, rgb, quality)
}

/// Encode a 1-channel Grayscale (`Luma`) buffer to JPEG via pure-Rust SIMD `jpeg-encoder`.
pub fn encode_jpeg_gray(width: u32, height: u32, gray: &[u8], quality: u8) -> Bytes {
    let (Ok(w), Ok(h)) = (u16::try_from(width), u16::try_from(height)) else {
        return Bytes::new();
    };
    let mut out = Vec::with_capacity(gray.len() / 4);
    let enc = Encoder::new(&mut out, quality.clamp(1, 100));
    if enc.encode(gray, w, h, ColorType::Luma).is_err() {
        return Bytes::new();
    }
    Bytes::from(out)
}

/// Encode an RGBA8 buffer to JPEG (stripping alpha onto black background, matching TiTiler).
/// Automatically emits a 1-component `Luma` JPEG when all opaque pixels are grayscale (`r==g==b`),
/// matching TiTiler's 1-band greyscale JPEG output and reducing size by ~30%.
/// Reuses a thread-local scratch buffer to avoid per-tile heap allocations.
pub fn encode_jpeg_rgba(width: u32, height: u32, rgba: &[u8], quality: u8) -> Bytes {
    encode_jpeg_rgba_as(width, height, rgba, quality, true)
}

/// [`encode_jpeg_rgba`], always in three components: for a caller that knows the picture
/// is of three bands, whatever its pixels happen to hold.
pub fn encode_jpeg_rgba_colour(width: u32, height: u32, rgba: &[u8], quality: u8) -> Bytes {
    encode_jpeg_rgba_as(width, height, rgba, quality, false)
}

fn encode_jpeg_rgba_as(width: u32, height: u32, rgba: &[u8], quality: u8, grey_if_grey: bool) -> Bytes {
    let pixels = (width as usize).saturating_mul(height as usize);
    let is_gray = grey_if_grey
        && rgba
        .chunks_exact(4)
        .take(pixels)
        .all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
    JPEG_RGB_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        let encoded = if is_gray {
            scratch.resize(pixels, 0);
            for (src, dst) in rgba.chunks_exact(4).take(pixels).zip(scratch.iter_mut()) {
                *dst = if src[3] != 0 { src[0] } else { 0 };
            }
            encode_jpeg_gray(width, height, &scratch, quality)
        } else {
            let needed = pixels.saturating_mul(3);
            scratch.resize(needed, 0);
            for (src, dst) in rgba
                .chunks_exact(4)
                .take(pixels)
                .zip(scratch.chunks_exact_mut(3))
            {
                if src[3] != 0 {
                    dst[0] = src[0];
                    dst[1] = src[1];
                    dst[2] = src[2];
                }
            }
            encode_jpeg_rgb_raw(width, height, &scratch, quality)
        };
        if scratch.capacity() > 4 * 1024 * 1024 {
            *scratch = Vec::new();
        }
        encoded
    })
}

/// Pre-encoded static black JPEG for empty/out-of-bounds tiles.
pub fn empty_black_jpeg(size: u32) -> Bytes {
    match size {
        256 => {
            static JPG_256: OnceLock<Bytes> = OnceLock::new();
            JPG_256
                .get_or_init(|| encode_jpeg_rgb(256, 256, &vec![0u8; 256 * 256 * 3], 85))
                .clone()
        }
        512 => {
            static JPG_512: OnceLock<Bytes> = OnceLock::new();
            JPG_512
                .get_or_init(|| encode_jpeg_rgb(512, 512, &vec![0u8; 512 * 512 * 3], 85))
                .clone()
        }
        other => encode_jpeg_rgb(
            other,
            other,
            &vec![0u8; (other as usize) * (other as usize) * 3],
            85,
        ),
    }
}
