use std::sync::OnceLock;

use bytes::Bytes;

/// Encode an RGB8 buffer to WebP via native C `libwebp`.
/// `quality`: `None` -> lossless, `Some(q)` -> lossy quality `0.0..=100.0`.
pub fn encode_webp_rgb(width: u32, height: u32, rgb: &[u8], quality: Option<f32>) -> Bytes {
    let mut out_ptr: *mut u8 = std::ptr::null_mut();
    let stride = (width * 3) as i32;
    let len = unsafe {
        match quality {
            Some(q) => libwebp_sys::WebPEncodeRGB(
                rgb.as_ptr(),
                width as i32,
                height as i32,
                stride,
                q,
                &mut out_ptr,
            ),
            None => libwebp_sys::WebPEncodeLosslessRGB(
                rgb.as_ptr(),
                width as i32,
                height as i32,
                stride,
                &mut out_ptr,
            ),
        }
    };
    if len == 0 || out_ptr.is_null() {
        return Bytes::new();
    }
    let bytes = unsafe {
        let slice = std::slice::from_raw_parts(out_ptr, len);
        let b = Bytes::copy_from_slice(slice);
        libwebp_sys::WebPFree(out_ptr as *mut std::ffi::c_void);
        b
    };
    bytes
}

/// Encode an RGBA8 buffer to WebP via native C `libwebp`.
pub fn encode_webp_rgba(width: u32, height: u32, rgba: &[u8], quality: Option<f32>) -> Bytes {
    let mut out_ptr: *mut u8 = std::ptr::null_mut();
    let stride = (width * 4) as i32;
    let len = unsafe {
        match quality {
            Some(q) => libwebp_sys::WebPEncodeRGBA(
                rgba.as_ptr(),
                width as i32,
                height as i32,
                stride,
                q,
                &mut out_ptr,
            ),
            None => libwebp_sys::WebPEncodeLosslessRGBA(
                rgba.as_ptr(),
                width as i32,
                height as i32,
                stride,
                &mut out_ptr,
            ),
        }
    };
    if len == 0 || out_ptr.is_null() {
        return Bytes::new();
    }
    unsafe {
        let slice = std::slice::from_raw_parts(out_ptr, len);
        let b = Bytes::copy_from_slice(slice);
        libwebp_sys::WebPFree(out_ptr as *mut std::ffi::c_void);
        b
    }
}

/// Pre-encoded static transparent WebP for instant empty/nodata tile responses.
pub fn empty_transparent_webp(size: u32) -> Bytes {
    match size {
        256 => {
            static WEBP_256: OnceLock<Bytes> = OnceLock::new();
            WEBP_256
                .get_or_init(|| {
                    let raw = vec![0u8; 256 * 256 * 4];
                    encode_webp_rgba(256, 256, &raw, None)
                })
                .clone()
        }
        512 => {
            static WEBP_512: OnceLock<Bytes> = OnceLock::new();
            WEBP_512
                .get_or_init(|| {
                    let raw = vec![0u8; 512 * 512 * 4];
                    encode_webp_rgba(512, 512, &raw, None)
                })
                .clone()
        }
        other => {
            let raw = vec![0u8; (other as usize) * (other as usize) * 4];
            encode_webp_rgba(other, other, &raw, None)
        }
    }
}
