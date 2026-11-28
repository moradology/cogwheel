//! NumPy v1.0 `.npy` array serialization and retyping.

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_raster::SampleType;

/// Encode a 64-byte-aligned NumPy v1.0 `.npy` header for a C-contiguous array of the
/// dtype `descr` (`<f4`, `|u1`, ...) and of `shape`: the magic, the version, the length of
/// what follows, then the dict padded with spaces and ended by a newline so that the data
/// starts on a multiple of 64 bytes. Every `.npy` this crate writes begins with it.
pub fn encode_npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    use std::fmt::Write;
    let mut dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': (");
    for (i, dim) in shape.iter().enumerate() {
        if i > 0 {
            dict.push_str(", ");
        }
        let _ = write!(dict, "{dim}");
    }
    // A tuple of one is written `(n,)`.
    dict.push_str(if shape.len() == 1 { ",), }" } else { "), }" });
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    for _ in 0..pad {
        dict.push(' ');
    }
    dict.push('\n');
    let hlen = dict.len() as u16;
    let mut hdr = Vec::with_capacity(10 + dict.len());
    hdr.extend_from_slice(b"\x93NUMPY\x01\x00");
    hdr.extend_from_slice(&hlen.to_le_bytes());
    hdr.extend_from_slice(dict.as_bytes());
    hdr
}

/// Encode a 64-byte-aligned NumPy v1.0 `.npy` header for a C-contiguous `<f4` array
/// of shape `(bands, height, width)`.
pub fn encode_npy_f32_header(bands: usize, height: u32, width: u32) -> Vec<u8> {
    encode_npy_header("<f4", &[bands, height as usize, width as usize])
}

/// The NumPy dtype string of a sample type: little-endian, and `|` for the single bytes.
#[inline(always)]
pub const fn npy_descr(kind: SampleType) -> &'static str {
    kind.npy_descr()
}

fn write_npy_typed_planes(
    out: &mut Vec<u8>,
    kind: SampleType,
    data: impl Iterator<Item = f64>,
    mask_bits: Option<impl Iterator<Item = bool>>,
) {
    macro_rules! write_planes {
        ($t:ty, $cast:expr) => {{
            let cast = $cast;
            for v in data {
                let x: $t = cast(v);
                out.extend_from_slice(&x.to_le_bytes());
            }
            if let Some(bits) = mask_bits {
                for there in bits {
                    let x: $t = if there { <$t>::MAX } else { <$t>::MIN };
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }};
    }
    match kind {
        SampleType::Bool => {
            for v in data {
                out.push(u8::from(v != 0.0 && !v.is_nan()));
            }
            if let Some(bits) = mask_bits {
                for there in bits {
                    out.push(u8::from(there));
                }
            }
        }
        SampleType::Uint1 | SampleType::Uint4 | SampleType::Uint8 => {
            write_planes!(u8, |v: f64| v as u8)
        }
        SampleType::Int8 => write_planes!(i8, |v: f64| v as i8),
        SampleType::Uint12 | SampleType::Uint16 => write_planes!(u16, |v: f64| v as u16),
        SampleType::Int16 => write_planes!(i16, |v: f64| v as i16),
        SampleType::Uint32 => write_planes!(u32, |v: f64| v as u32),
        SampleType::Int32 => write_planes!(i32, |v: f64| v as i32),
        SampleType::Uint64 => write_planes!(u64, |v: f64| v as u64),
        SampleType::Int64 => write_planes!(i64, |v: f64| v as i64),
        SampleType::Float16 => {
            for v in data {
                out.extend_from_slice(&f32_to_f16_bits(v as f32).to_le_bytes());
            }
            if let Some(bits) = mask_bits {
                for there in bits {
                    // Max finite f16 = 65504.0 (0x7BFF), min finite f16 = -65504.0 (0xFBFF).
                    let bits: u16 = if there { 0x7BFF } else { 0xFBFF };
                    out.extend_from_slice(&bits.to_le_bytes());
                }
            }
        }
        SampleType::Float32 => write_planes!(f32, |v: f64| (v as f32).clamp(f32::MIN, f32::MAX)),
        SampleType::Float64 => write_planes!(f64, |v: f64| v.clamp(f64::MIN, f64::MAX)),
    }
}

/// Convert `f32` to IEEE-754 binary16 bits (preserving `NaN` as canonical quiet NaN `0x7E00`
/// and clamping infinities to `±65504.0`).
fn f32_to_f16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7E00;
    }
    let clamped = v.clamp(-65504.0, 65504.0);
    let bits = clamped.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = bits & 0x007F_FFFF;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (frac | 0x0080_0000) >> (1 - exp + 13);
        return sign | (m as u16);
    }
    if exp >= 31 {
        return sign | 0x7BFF;
    }
    sign | ((exp as u16) << 10) | ((frac >> 13) as u16)
}

/// A `.npy` array of `bands` planes of `height` by `width`, written in the dtype of `kind`
/// from CHW float32 `values` in which NaN marks a value that is not there.
///
/// A value that is not there is written as `absent`, or as zero without one. Integers take
/// the value truncated toward zero and held to the type's range (a NaN `absent` is zero);
/// the float types hold an infinity to their largest finite number. With `mask`, one more
/// plane follows the bands, as rio-tiler's `ImageData.mask` is: the type's largest number
/// (`255` for `u1`, `65535` for `u2`, `f32::MAX` for `f4`, ...) where any band has a value,
/// and its smallest elsewhere (`0` for the unsigned types, `-32768` for `i2`, `f32::MIN`
/// for `f4`, ...). Only the first
/// `bands * height * width` of `values` are written, and a shorter slice makes a shorter
/// body under a header that still names every plane.
///
/// This is not the float32 path of [`wrap_raw_f32_as_npy_with_mask`], which copies its
/// planes as they are, counts an infinity as absent and writes `255.0` in its mask.
pub fn encode_npy_typed(
    values: &[f32],
    bands: usize,
    height: usize,
    width: usize,
    kind: SampleType,
    absent: Option<f32>,
    mask: bool,
) -> Bytes {
    let pixels = width * height;
    let planes = bands + usize::from(mask);
    let hdr = encode_npy_header(npy_descr(kind), &[planes, height, width]);
    let mut out = Vec::with_capacity(hdr.len() + planes * pixels * kind.bytes_per_sample());
    out.extend_from_slice(&hdr);
    let absent = f64::from(absent.unwrap_or(0.0));
    let data = &values[..(bands * pixels).min(values.len())];
    let vals = data
        .iter()
        .map(|v| if v.is_nan() { absent } else { f64::from(*v) });
    let mask_bits = mask.then(|| {
        (0..pixels).map(|i| {
            (0..bands).any(|band| data.get(band * pixels + i).is_some_and(|v| !v.is_nan()))
        })
    });
    write_npy_typed_planes(&mut out, kind, vals, mask_bits);
    Bytes::from(out)
}

/// The shape of a float32 `.npy` this crate wrote (`<f4`, C order), and where its data starts.
pub fn f32_npy_shape(npy: &[u8]) -> Option<(Vec<usize>, usize)> {
    if npy.len() < 10 || &npy[..8] != b"\x93NUMPY\x01\x00" {
        return None;
    }
    let start = 10 + usize::from(u16::from_le_bytes([npy[8], npy[9]]));
    let dict = std::str::from_utf8(npy.get(10..start)?).ok()?;
    if !dict.contains("'descr': '<f4'") || !dict.contains("'fortran_order': False") {
        return None;
    }
    let from = dict.find("'shape': (")? + "'shape': (".len();
    let shape = dict[from..from + dict[from..].find(')')?]
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| dim.parse().ok())
        .collect::<Option<Vec<usize>>>()?;
    Some((shape, start))
}

/// A float32 `.npy` of CHW planes rewritten in the dtype of `kind`, or `None` when `npy` is
/// not such an array or its body is not the size its header names.
///
/// The values are cast as [`encode_npy_typed`] casts them: integers truncated toward zero
/// and held to the type's range (NaN as zero), floats with an infinity held to their largest
/// finite number. With `mask` the last plane is a validity mask (any value but zero where a
/// pixel is there), written as the type's largest number there and its smallest elsewhere,
/// as rio-tiler's `ImageData.mask` is. The values under the mask are kept as they are: what
/// the renderer put there is the raster's own fill.
pub fn retype_f32_npy(npy: &[u8], kind: SampleType, mask: bool) -> Option<Bytes> {
    retype_f32_npy_with(npy, kind, mask, |v| v)
}

/// [`retype_f32_npy`], each value (not the mask) passed through `prepare` before it is cast:
/// how an algorithm rounds or wraps it into `kind`.
pub fn retype_f32_npy_with(
    npy: &[u8],
    kind: SampleType,
    mask: bool,
    prepare: impl Fn(f64) -> f64,
) -> Option<Bytes> {
    let (shape, start) = f32_npy_shape(npy)?;
    let [planes, height, width] = shape[..] else { return None };
    let body = &npy[start..];
    let pixels = height.checked_mul(width)?;
    if planes == 0 || body.len() != planes.checked_mul(pixels)?.checked_mul(4)? {
        return None;
    }
    let bands = planes - usize::from(mask);
    let data_bytes = bands * pixels * 4;
    let hdr = encode_npy_header(npy_descr(kind), &[planes, height, width]);
    let mut out = Vec::with_capacity(hdr.len() + planes * pixels * kind.bytes_per_sample());
    out.extend_from_slice(&hdr);
    let data = body[..data_bytes]
        .chunks_exact(4)
        .map(|v| prepare(f64::from(f32::from_le_bytes([v[0], v[1], v[2], v[3]]))));
    let mask_bits = mask.then(|| {
        body[data_bytes..].chunks_exact(4).map(|v| {
            let m = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
            m.is_finite() && m != 0.0
        })
    });
    write_npy_typed_planes(&mut out, kind, data, mask_bits);
    Some(Bytes::from(out))
}

/// Like [`encode_npy_typed`], taking full-precision CHW `f64` values (preserving exact
/// `uint32`/`int32`/`int64` up to ±2^53 and 17-digit `float64` without `f32` quantization).
pub fn encode_npy_typed_f64(
    values: &[f64],
    bands: usize,
    height: usize,
    width: usize,
    kind: SampleType,
    absent: Option<f64>,
    mask: bool,
) -> Bytes {
    let pixels = width * height;
    let planes = bands + usize::from(mask);
    let hdr = encode_npy_header(npy_descr(kind), &[planes, height, width]);
    let mut out = Vec::with_capacity(hdr.len() + planes * pixels * kind.bytes_per_sample());
    out.extend_from_slice(&hdr);
    let absent = absent.unwrap_or(if kind.storable_range().is_none() {
        f64::NAN
    } else {
        0.0
    });
    let data = &values[..(bands * pixels).min(values.len())];
    let vals = data.iter().map(|&v| if v.is_nan() { absent } else { v });
    let mask_bits = mask.then(|| {
        (0..pixels).map(|i| {
            (0..bands).any(|band| data.get(band * pixels + i).is_some_and(|v| !v.is_nan()))
        })
    });
    write_npy_typed_planes(&mut out, kind, vals, mask_bits);
    Bytes::from(out)
}

/// Prepend a NumPy v1.0 `.npy` `<f4` header to raw CHW little-endian float32 bytes.
pub fn wrap_raw_f32_as_npy(raw: Bytes, bands: usize, height: u32, width: u32) -> Bytes {
    wrap_raw_f32_as_npy_with_mask(raw, bands, height, width, false)
}

/// Prepend a NumPy v1.0 `.npy` `<f4` header to raw CHW little-endian float32 bytes,
/// optionally stacking a `(1, H, W)` float32 validity mask band (`255.0` valid / `0.0` nodata)
/// when `return_mask` is `true` (matching TiTiler / `rio-tiler` `.npy` output).
pub fn wrap_raw_f32_as_npy_with_mask(
    raw: Bytes,
    bands: usize,
    height: u32,
    width: u32,
    return_mask: bool,
) -> Bytes {
    if !return_mask {
        let hdr = encode_npy_f32_header(bands, height, width);
        let mut out = BytesMut::with_capacity(hdr.len() + raw.len());
        out.put_slice(&hdr);
        out.put_slice(&raw);
        return out.freeze();
    }
    let n_px = (height as usize).saturating_mul(width as usize);
    let plane_bytes = n_px * 4;
    let hdr = encode_npy_f32_header(bands + 1, height, width);
    let mut out = BytesMut::with_capacity(hdr.len() + raw.len() + plane_bytes);
    out.put_slice(&hdr);
    out.put_slice(&raw);
    let valid_le = 255.0f32.to_le_bytes();
    let zero_le = 0.0f32.to_le_bytes();
    for px in 0..n_px {
        let px_off = px * 4;
        let mut any_valid = false;
        for b in 0..bands {
            let off = b * plane_bytes + px_off;
            if off + 4 <= raw.len() {
                let v = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                if v.is_finite() {
                    any_valid = true;
                    break;
                }
            }
        }
        out.put_slice(if any_valid { &valid_le } else { &zero_le });
    }
    out.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_body(bytes: &[u8]) -> &[u8] {
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let hdr_len = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        &bytes[10 + hdr_len..]
    }

    #[test]
    fn float_dtypes_write_nan_for_absent_pixels_in_npy() {
        // Float16: absent pixel -> canonical binary16 quiet NaN (0x7E00); finite 1.0 -> 0x3C00.
        let f16_npy = encode_npy_typed_f64(&[1.0, f64::NAN], 1, 1, 2, SampleType::Float16, None, false);
        let body = npy_body(&f16_npy);
        assert_eq!(body.len(), 4);
        assert_eq!(u16::from_le_bytes([body[0], body[1]]), 0x3C00);
        let nan_bits = u16::from_le_bytes([body[2], body[3]]);
        assert_eq!(nan_bits & 0x7C00, 0x7C00, "Float16 absent pixel must have NaN exponent");
        assert_ne!(nan_bits & 0x03FF, 0, "Float16 absent pixel must have non-zero mantissa (NaN, not Inf)");

        // Float32: absent pixel -> NaN.
        let f32_npy = encode_npy_typed_f64(&[1.0, f64::NAN], 1, 1, 2, SampleType::Float32, None, false);
        let body = npy_body(&f32_npy);
        assert_eq!(f32::from_le_bytes(body[0..4].try_into().unwrap()), 1.0);
        assert!(f32::from_le_bytes(body[4..8].try_into().unwrap()).is_nan());

        // Float64: absent pixel -> NaN.
        let f64_npy = encode_npy_typed_f64(&[1.0, f64::NAN], 1, 1, 2, SampleType::Float64, None, false);
        let body = npy_body(&f64_npy);
        assert_eq!(f64::from_le_bytes(body[0..8].try_into().unwrap()), 1.0);
        assert!(f64::from_le_bytes(body[8..16].try_into().unwrap()).is_nan());

        // Integer/Bool dtypes: absent pixel -> 0.
        let u8_npy = encode_npy_typed_f64(&[7.0, f64::NAN], 1, 1, 2, SampleType::Uint8, None, false);
        assert_eq!(npy_body(&u8_npy), &[7u8, 0u8]);
        let bool_npy = encode_npy_typed_f64(&[1.0, f64::NAN], 1, 1, 2, SampleType::Bool, None, false);
        assert_eq!(npy_body(&bool_npy), &[1u8, 0u8]);
    }
}
