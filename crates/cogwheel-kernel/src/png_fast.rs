use std::cell::RefCell;
use std::sync::OnceLock;

use bytes::{BufMut, Bytes, BytesMut};
use libdeflater::{CompressionLvl, Compressor};

const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
const PNG_IEND: [u8; 12] = [0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82];

struct ThreadPngEncoder {
    compressor: Compressor,
    zlib_buf: Vec<u8>,
}

impl ThreadPngEncoder {
    fn new() -> Self {
        Self {
            compressor: Compressor::new(CompressionLvl::new(1).expect("valid level 1")),
            zlib_buf: Vec::with_capacity(512 * 512 * 4 + 1024),
        }
    }
}

thread_local! {
    static PNG_ENC: RefCell<ThreadPngEncoder> = RefCell::new(ThreadPngEncoder::new());
}

#[inline(always)]
fn write_chunk(out: &mut BytesMut, chunk_type: [u8; 4], data: &[u8]) {
    out.put_u32(data.len() as u32);
    out.put_slice(&chunk_type);
    out.put_slice(data);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&chunk_type);
    hasher.update(data);
    out.put_u32(hasher.finalize());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_sub_rgba_avx2(src: &[u8], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let len = src.len();
    let mut i = 0usize;
    let mut prev = _mm_setzero_si128();
    while i + 16 <= len {
        let cur = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let left = _mm_alignr_epi8::<12>(cur, prev);
        let diff = _mm_sub_epi8(cur, left);
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, diff);
        prev = cur;
        i += 16;
    }
    while i < len {
        let left = if i >= 4 { src[i - 4] } else { 0 };
        out[i] = src[i].wrapping_sub(left);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_sub_rgb_avx2(src: &[u8], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let len = src.len();
    let mut i = 0usize;
    let mut prev = _mm_setzero_si128();
    while i + 16 <= len {
        let cur = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let left = _mm_alignr_epi8::<13>(cur, prev);
        let diff = _mm_sub_epi8(cur, left);
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, diff);
        prev = cur;
        i += 16;
    }
    while i < len {
        let left = if i >= 3 { src[i - 3] } else { 0 };
        out[i] = src[i].wrapping_sub(left);
        i += 1;
    }
}

/// Apply PNG Sub filter (type = 1) to an RGBA8 scanline (`bpp = 4`) using ARM64 NEON or x86_64 AVX2.
#[inline(always)]
pub fn filter_sub_rgba_row(src_rgba: &[u8], dst_filtered: &mut [u8]) {
    dst_filtered[0] = 1;
    let out = &mut dst_filtered[1..1 + src_rgba.len()];
    let len = src_rgba.len();

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut i = 0usize;
        let mut prev = unsafe { vdupq_n_u8(0) };
        while i + 16 <= len {
            unsafe {
                let cur = vld1q_u8(src_rgba.as_ptr().add(i));
                let left = vextq_u8::<12>(prev, cur);
                let diff = vsubq_u8(cur, left);
                vst1q_u8(out.as_mut_ptr().add(i), diff);
                prev = cur;
            }
            i += 16;
        }
        while i < len {
            let left = if i >= 4 { src_rgba[i - 4] } else { 0 };
            out[i] = src_rgba[i].wrapping_sub(left);
            i += 1;
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { filter_sub_rgba_avx2(src_rgba, out) };
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        for i in 0..4.min(len) {
            out[i] = src_rgba[i];
        }
        for i in 4..len {
            out[i] = src_rgba[i].wrapping_sub(src_rgba[i - 4]);
        }
    }
}

/// Apply PNG Sub filter (type = 1) to an opaque RGB8 scanline (`bpp = 3`) using NEON or AVX2.
#[inline(always)]
pub fn filter_sub_rgb_row(src_rgb: &[u8], dst_filtered: &mut [u8]) {
    dst_filtered[0] = 1;
    let out = &mut dst_filtered[1..1 + src_rgb.len()];
    let len = src_rgb.len();

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut i = 0usize;
        let mut prev = unsafe { vdupq_n_u8(0) };
        while i + 16 <= len {
            unsafe {
                let cur = vld1q_u8(src_rgb.as_ptr().add(i));
                let left = vextq_u8::<13>(prev, cur);
                let diff = vsubq_u8(cur, left);
                vst1q_u8(out.as_mut_ptr().add(i), diff);
                prev = cur;
            }
            i += 16;
        }
        while i < len {
            let left = if i >= 3 { src_rgb[i - 3] } else { 0 };
            out[i] = src_rgb[i].wrapping_sub(left);
            i += 1;
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { filter_sub_rgb_avx2(src_rgb, out) };
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        for i in 0..3.min(len) {
            out[i] = src_rgb[i];
        }
        for i in 3..len {
            out[i] = src_rgb[i].wrapping_sub(src_rgb[i - 3]);
        }
    }
}

/// Compress already-filtered PNG scanlines in a single whole-buffer `libdeflate` call
/// and frame into a complete valid PNG `Bytes` buffer.
/// `color_type`: `2` = RGB8, `6` = RGBA8.
pub fn encode_filtered_png(width: u32, height: u32, color_type: u8, filtered: &[u8]) -> Bytes {
    PNG_ENC.with(|cell| {
        let mut guard = cell.borrow_mut();
        let enc = &mut *guard;
        let bound = enc.compressor.zlib_compress_bound(filtered.len());
        if enc.zlib_buf.len() < bound {
            enc.zlib_buf.resize(bound, 0);
        }
        let written = enc
            .compressor
            .zlib_compress(filtered, &mut enc.zlib_buf[..bound])
            .expect("libdeflate zlib compress");

        let mut out = BytesMut::with_capacity(8 + 25 + 12 + written + 12);
        out.put_slice(&PNG_SIG);

        let mut ihdr = [0u8; 13];
        ihdr[0..4].copy_from_slice(&width.to_be_bytes());
        ihdr[4..8].copy_from_slice(&height.to_be_bytes());
        ihdr[8] = 8;
        ihdr[9] = color_type;
        ihdr[10] = 0;
        ihdr[11] = 0;
        ihdr[12] = 0;
        write_chunk(&mut out, *b"IHDR", &ihdr);
        write_chunk(&mut out, *b"IDAT", &enc.zlib_buf[..written]);
        out.put_slice(&PNG_IEND);
        out.freeze()
    })
}

/// Pre-encoded static transparent RGBA8 PNG for instant `<200ns` empty/nodata tile responses.
pub fn empty_transparent_png(size: u32) -> Bytes {
    match size {
        256 => {
            static PNG_256: OnceLock<Bytes> = OnceLock::new();
            PNG_256
                .get_or_init(|| {
                    let raw = vec![0u8; 256 * (1 + 256 * 4)];
                    encode_filtered_png(256, 256, 6, &raw)
                })
                .clone()
        }
        512 => {
            static PNG_512: OnceLock<Bytes> = OnceLock::new();
            PNG_512
                .get_or_init(|| {
                    let raw = vec![0u8; 512 * (1 + 512 * 4)];
                    encode_filtered_png(512, 512, 6, &raw)
                })
                .clone()
        }
        other => {
            let raw = vec![0u8; (other as usize) * (1 + (other as usize) * 4)];
            encode_filtered_png(other, other, 6, &raw)
        }
    }
}
