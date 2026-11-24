use std::cell::RefCell;
use std::sync::OnceLock;

use bytes::Bytes;
use jpeg_encoder::{ColorType, Encoder};

thread_local! {
    static JPEG_RGB_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Encode an RGB8 buffer to JPEG via pure-Rust SIMD `jpeg-encoder`.
pub fn encode_jpeg_rgb(width: u32, height: u32, rgb: &[u8], quality: u8) -> Bytes {
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

/// Encode an RGBA8 buffer to JPEG (stripping alpha onto black background, matching TiTiler).
/// Reuses a thread-local RGB scratch buffer to avoid per-tile heap allocations.
pub fn encode_jpeg_rgba(width: u32, height: u32, rgba: &[u8], quality: u8) -> Bytes {
    let pixels = (width as usize).saturating_mul(height as usize);
    let needed = pixels.saturating_mul(3);
    JPEG_RGB_SCRATCH.with(|cell| {
        let mut rgb = cell.borrow_mut();
        rgb.clear();
        rgb.resize(needed, 0);
        for (src, dst) in rgba
            .chunks_exact(4)
            .take(pixels)
            .zip(rgb.chunks_exact_mut(3))
        {
            if src[3] != 0 {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
            }
        }
        let encoded = encode_jpeg_rgb(width, height, &rgb, quality);
        if rgb.capacity() > 4 * 1024 * 1024 {
            *rgb = Vec::new();
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
