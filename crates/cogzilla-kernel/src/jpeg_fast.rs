use std::sync::OnceLock;

use bytes::Bytes;
use jpeg_encoder::{ColorType, Encoder};

/// Encode an RGB8 buffer to JPEG via pure-Rust SIMD `jpeg-encoder`.
pub fn encode_jpeg_rgb(width: u32, height: u32, rgb: &[u8], quality: u8) -> Bytes {
    let mut out = Vec::with_capacity(rgb.len() / 6);
    let enc = Encoder::new(&mut out, quality.clamp(1, 100));
    let _ = enc.encode(rgb, width as u16, height as u16, ColorType::Rgb);
    Bytes::from(out)
}

/// Encode an RGBA8 buffer to JPEG (stripping alpha onto black background, matching TiTiler).
pub fn encode_jpeg_rgba(width: u32, height: u32, rgba: &[u8], quality: u8) -> Bytes {
    let pixels = (width as usize) * (height as usize);
    let mut rgb = Vec::with_capacity(pixels * 3);
    for i in 0..pixels {
        let a = rgba[i * 4 + 3];
        if a == 0 {
            rgb.push(0);
            rgb.push(0);
            rgb.push(0);
        } else {
            rgb.push(rgba[i * 4]);
            rgb.push(rgba[i * 4 + 1]);
            rgb.push(rgba[i * 4 + 2]);
        }
    }
    encode_jpeg_rgb(width, height, &rgb, quality)
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
