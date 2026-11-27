//! Zarr chunk codec chain execution: bytes -> native-endian C-order sample bytes.
//!
//! Reuses `libdeflater` (`gzip`, `zlib`) and `zstd-sys` (`zstd`) already linked by the workspace,
//! plus pure-Rust bounds-checked `lz4` (raw numcodecs block + frame), `shuffle`, `bitshuffle`,
//! `bytes` endianness, and `transpose`.

use serde_json::Value;

use crate::{ArrayMeta, SampleType, Endianness, ZarrError, ZarrVersion};

/// Hard upper bound on a single decoded chunk (128 MiB), matching `cogzilla-codec`'s
/// `MAX_DECODED_TILE_BYTES` to prevent hostile chunk headers from triggering OOM allocations.
pub const MAX_DECODED_CHUNK_BYTES: usize = 128 * 1024 * 1024;

/// Human-readable list of codecs supported by this build (included in `UnsupportedCodec` errors).
pub const SUPPORTED_CODECS: &str =
    "bytes, transpose, gzip, zlib, zstd, lz4, blosc, shuffle, bitshuffle";

/// Compute the expected uncompressed byte length of one chunk, enforcing `MAX_DECODED_CHUNK_BYTES`.
pub fn expected_chunk_bytes(
    chunk_shape: &[u64],
    dtype: SampleType,
    chunk_key: &str,
) -> Result<usize, ZarrError> {
    let mut elems: u64 = 1;
    for &d in chunk_shape {
        elems = elems.checked_mul(d).ok_or(ZarrError::ChunkTooLarge {
            chunk_key: chunk_key.to_string(),
            bytes: usize::MAX,
            cap: MAX_DECODED_CHUNK_BYTES,
        })?;
    }
    let bytes_u64 = elems
        .checked_mul(dtype.bytes_per_sample() as u64)
        .ok_or(ZarrError::ChunkTooLarge {
            chunk_key: chunk_key.to_string(),
            bytes: usize::MAX,
            cap: MAX_DECODED_CHUNK_BYTES,
        })?;
    let bytes = usize::try_from(bytes_u64).map_err(|_| ZarrError::ChunkTooLarge {
        chunk_key: chunk_key.to_string(),
        bytes: usize::MAX,
        cap: MAX_DECODED_CHUNK_BYTES,
    })?;
    if bytes > MAX_DECODED_CHUNK_BYTES {
        return Err(ZarrError::ChunkTooLarge {
            chunk_key: chunk_key.to_string(),
            bytes,
            cap: MAX_DECODED_CHUNK_BYTES,
        });
    }
    Ok(bytes)
}

/// Decode a raw chunk payload through `meta.codecs` (in reverse encode order) into native-endian
/// C-order sample bytes of exact length `expected_chunk_bytes(meta.chunk_shape, meta.dtype)`.
pub fn decode_chunk_bytes(
    meta: &ArrayMeta,
    raw: &[u8],
    chunk_key: &str,
) -> Result<Vec<u8>, ZarrError> {
    decode_chunk_with_shape(
        meta,
        &meta.chunk_shape,
        &meta.codecs,
        raw,
        chunk_key,
    )
}

/// Decode a raw chunk payload with an explicit chunk shape and codec list (also usable by
/// inner chunks of a v3 shard).
pub fn decode_chunk_with_shape(
    meta: &ArrayMeta,
    chunk_shape: &[u64],
    codecs: &[crate::CodecSpec],
    raw: &[u8],
    chunk_key: &str,
) -> Result<Vec<u8>, ZarrError> {
    let expected = expected_chunk_bytes(chunk_shape, meta.dtype, chunk_key)?;
    let mut buf = raw.to_vec();
    let mut endian_handled = false;

    // Decode runs in reverse of declared (encode) order: outermost bytes-to-bytes first, then
    // array-to-bytes (`bytes`), then array-to-array (`transpose`).
    for codec in codecs.iter().rev() {
        match codec.name.as_str() {
            "gzip" => {
                buf = decompress_gzip(&buf, expected, chunk_key)?;
            }
            "zlib" => {
                buf = decompress_zlib(&buf, expected, chunk_key)?;
            }
            "zstd" => {
                buf = decompress_zstd(&buf, expected, chunk_key)?;
            }
            "lz4" => {
                buf = decompress_lz4(&buf, expected, chunk_key)?;
            }
            "blosc" => {
                buf = crate::blosc::decompress_blosc(&buf, expected, chunk_key)?;
            }
            "shuffle" => {
                let elem_size = codec
                    .configuration
                    .get("elementsize")
                    .and_then(Value::as_u64)
                    .map(|v| v as usize)
                    .unwrap_or_else(|| meta.dtype.bytes_per_sample());
                buf = unshuffle_bytes(&buf, elem_size, chunk_key)?;
            }
            "bitshuffle" => {
                let elem_size = codec
                    .configuration
                    .get("elementsize")
                    .and_then(Value::as_u64)
                    .map(|v| v as usize)
                    .unwrap_or_else(|| meta.dtype.bytes_per_sample());
                buf = unbitshuffle_bytes(&buf, elem_size, chunk_key)?;
            }
            "bytes" => {
                let endian = codec
                    .configuration
                    .get("endian")
                    .and_then(Value::as_str)
                    .map(|e| {
                        if e == "big" {
                            Endianness::Big
                        } else {
                            Endianness::Little
                        }
                    })
                    .unwrap_or(Endianness::Little);
                ensure_exact_len(&buf, expected, "bytes", chunk_key)?;
                convert_endian_to_native(&mut buf, meta.dtype.bytes_per_sample(), endian);
                endian_handled = true;
            }
            "transpose" => {
                ensure_exact_len(&buf, expected, "transpose", chunk_key)?;
                let order = parse_transpose_order(&codec.configuration, chunk_shape.len(), chunk_key)?;
                buf = untranspose(&buf, chunk_shape, &order, meta.dtype.bytes_per_sample(), chunk_key)?;
            }
            other => {
                return Err(ZarrError::UnsupportedCodec {
                    codec: other.to_string(),
                    chunk_key: chunk_key.to_string(),
                    supported: SUPPORTED_CODECS,
                });
            }
        }
    }

    // Verify final decoded length matches declared chunk shape.
    ensure_exact_len(&buf, expected, "chunk", chunk_key)?;

    // Zarr v2 carries endianness in dtype (not a `bytes` codec) and optional Fortran order.
    if meta.version == ZarrVersion::V2 && !endian_handled {
        convert_endian_to_native(&mut buf, meta.dtype.bytes_per_sample(), meta.endianness);
    }
    if meta.order_f && chunk_shape.len() > 1 {
        let f_order: Vec<usize> = (0..chunk_shape.len()).rev().collect();
        buf = untranspose(&buf, chunk_shape, &f_order, meta.dtype.bytes_per_sample(), chunk_key)?;
    }

    Ok(buf)
}

fn ensure_exact_len(
    buf: &[u8],
    expected: usize,
    codec: &str,
    chunk_key: &str,
) -> Result<(), ZarrError> {
    if buf.len() != expected {
        return Err(ZarrError::Codec {
            codec: codec.to_string(),
            chunk_key: chunk_key.to_string(),
            detail: format!("expected {expected} decoded bytes, got {}", buf.len()),
        });
    }
    Ok(())
}

fn convert_endian_to_native(buf: &mut [u8], elem_size: usize, stored: Endianness) {
    if elem_size <= 1 {
        return;
    }
    let needs_swap = match stored {
        Endianness::Little => cfg!(target_endian = "big"),
        Endianness::Big => cfg!(target_endian = "little"),
    };
    if !needs_swap {
        return;
    }
    for chunk in buf.chunks_exact_mut(elem_size) {
        chunk.reverse();
    }
}

fn decompress_gzip(src: &[u8], expected: usize, chunk_key: &str) -> Result<Vec<u8>, ZarrError> {
    cogzilla_codec::inflate_to_vec(src, expected, cogzilla_codec::DeflateWrapper::Gzip).map_err(
        |e| ZarrError::Codec {
            codec: "gzip".into(),
            chunk_key: chunk_key.into(),
            detail: e.to_string(),
        },
    )
}

fn decompress_zlib(src: &[u8], expected: usize, chunk_key: &str) -> Result<Vec<u8>, ZarrError> {
    cogzilla_codec::inflate_to_vec(src, expected, cogzilla_codec::DeflateWrapper::Zlib).map_err(
        |e| ZarrError::Codec {
            codec: "zlib".into(),
            chunk_key: chunk_key.into(),
            detail: e.to_string(),
        },
    )
}

fn decompress_zstd(src: &[u8], expected: usize, chunk_key: &str) -> Result<Vec<u8>, ZarrError> {
    cogzilla_codec::zstd_to_vec(src, expected).map_err(|e| ZarrError::Codec {
        codec: "zstd".into(),
        chunk_key: chunk_key.into(),
        detail: e.to_string(),
    })
}

/// Decompress LZ4: supports both standard LZ4 frames (magic `0x184D2204`) and numcodecs's
/// 4-byte little-endian uncompressed-length-prefixed raw LZ4 block.
pub fn decompress_lz4(src: &[u8], expected: usize, chunk_key: &str) -> Result<Vec<u8>, ZarrError> {
    if src.len() < 4 {
        return Err(ZarrError::Codec {
            codec: "lz4".into(),
            chunk_key: chunk_key.into(),
            detail: "truncated lz4 payload".into(),
        });
    }
    // Standard LZ4 frame magic: 0x184D2204 little-endian
    if src[0..4] == [0x04, 0x22, 0x4D, 0x18] {
        return decompress_lz4_frame(src, expected, chunk_key);
    }
    // Numcodecs format: 4-byte LE uncompressed size + raw LZ4 block
    let declared = u32::from_le_bytes([src[0], src[1], src[2], src[3]]) as usize;
    if declared > expected || declared > MAX_DECODED_CHUNK_BYTES {
        return Err(ZarrError::Codec {
            codec: "lz4".into(),
            chunk_key: chunk_key.into(),
            detail: format!(
                "lz4 header declares {declared} bytes, exceeding expected chunk size {expected}"
            ),
        });
    }
    let mut out = Vec::with_capacity(declared);
    decompress_lz4_block(&src[4..], &mut out, declared, chunk_key)?;
    if out.len() != declared {
        return Err(ZarrError::Codec {
            codec: "lz4".into(),
            chunk_key: chunk_key.into(),
            detail: format!("lz4 decoded {} bytes, header declared {declared}", out.len()),
        });
    }
    Ok(out)
}

/// Pure-Rust bounds-checked LZ4 raw block decoder.
pub fn decompress_lz4_block(
    src: &[u8],
    out: &mut Vec<u8>,
    max_out: usize,
    chunk_key: &str,
) -> Result<(), ZarrError> {
    let mut i = 0usize;
    let start_len = out.len();
    let limit = start_len.saturating_add(max_out);
    while i < src.len() {
        let token = src[i];
        i += 1;

        // Literal length
        let mut lit_len = (token >> 4) as usize;
        if lit_len == 15 {
            loop {
                if i >= src.len() {
                    return Err(ZarrError::Codec {
                        codec: "lz4".into(),
                        chunk_key: chunk_key.into(),
                        detail: "truncated literal length extension".into(),
                    });
                }
                let b = src[i] as usize;
                i += 1;
                lit_len = lit_len.saturating_add(b);
                if b != 255 {
                    break;
                }
            }
        }
        if i + lit_len > src.len() {
            return Err(ZarrError::Codec {
                codec: "lz4".into(),
                chunk_key: chunk_key.into(),
                detail: "truncated literal run".into(),
            });
        }
        if out.len() + lit_len > limit {
            return Err(ZarrError::Codec {
                codec: "lz4".into(),
                chunk_key: chunk_key.into(),
                detail: "lz4 output exceeds declared bound".into(),
            });
        }
        out.extend_from_slice(&src[i..i + lit_len]);
        i += lit_len;

        // End of block (last sequence has no match)
        if i >= src.len() {
            break;
        }

        // Match offset (2 bytes little-endian)
        if i + 2 > src.len() {
            return Err(ZarrError::Codec {
                codec: "lz4".into(),
                chunk_key: chunk_key.into(),
                detail: "truncated match offset".into(),
            });
        }
        let offset = u16::from_le_bytes([src[i], src[i + 1]]) as usize;
        i += 2;
        if offset == 0 || offset > out.len() - start_len {
            return Err(ZarrError::Codec {
                codec: "lz4".into(),
                chunk_key: chunk_key.into(),
                detail: format!("invalid lz4 match offset {offset}"),
            });
        }

        // Match length
        let mut match_len = ((token & 0x0F) as usize) + 4;
        if (token & 0x0F) == 15 {
            loop {
                if i >= src.len() {
                    return Err(ZarrError::Codec {
                        codec: "lz4".into(),
                        chunk_key: chunk_key.into(),
                        detail: "truncated match length extension".into(),
                    });
                }
                let b = src[i] as usize;
                i += 1;
                match_len = match_len.saturating_add(b);
                if b != 255 {
                    break;
                }
            }
        }
        if out.len() + match_len > limit {
            return Err(ZarrError::Codec {
                codec: "lz4".into(),
                chunk_key: chunk_key.into(),
                detail: "lz4 match exceeds declared bound".into(),
            });
        }
        // Byte-by-byte copy handles overlapping matches (offset < match_len) safely.
        for _ in 0..match_len {
            let b = out[out.len() - offset];
            out.push(b);
        }
    }
    Ok(())
}

fn decompress_lz4_frame(
    src: &[u8],
    expected: usize,
    chunk_key: &str,
) -> Result<Vec<u8>, ZarrError> {
    if src.len() < 7 {
        return Err(ZarrError::Codec {
            codec: "lz4".into(),
            chunk_key: chunk_key.into(),
            detail: "truncated lz4 frame header".into(),
        });
    }
    let flg = src[4];
    let version = (flg >> 6) & 0b11;
    if version != 0b01 {
        return Err(ZarrError::Codec {
            codec: "lz4".into(),
            chunk_key: chunk_key.into(),
            detail: format!("unsupported lz4 frame version {version}"),
        });
    }
    let has_block_checksum = (flg & 0x10) != 0;
    let has_content_size = (flg & 0x08) != 0;
    let has_dict_id = (flg & 0x01) != 0;
    let mut pos = 6usize; // magic(4) + FLG(1) + BD(1)
    if has_content_size {
        if pos + 8 > src.len() {
            return Err(ZarrError::Codec {
                codec: "lz4".into(),
                chunk_key: chunk_key.into(),
                detail: "truncated lz4 frame content size".into(),
            });
        }
        let csize = u64::from_le_bytes(src[pos..pos + 8].try_into().unwrap());
        if csize > expected as u64 {
            return Err(ZarrError::Codec {
                codec: "lz4".into(),
                chunk_key: chunk_key.into(),
                detail: format!("lz4 frame content size {csize} exceeds expected {expected}"),
            });
        }
        pos += 8;
    }
    if has_dict_id {
        pos += 4;
    }
    pos += 1; // HC byte
    if pos > src.len() {
        return Err(ZarrError::Codec {
            codec: "lz4".into(),
            chunk_key: chunk_key.into(),
            detail: "truncated lz4 frame descriptor".into(),
        });
    }

    let mut out = Vec::with_capacity(expected);
    loop {
        if pos + 4 > src.len() {
            return Err(ZarrError::Codec {
                codec: "lz4".into(),
                chunk_key: chunk_key.into(),
                detail: "truncated lz4 block size".into(),
            });
        }
        let raw_sz = u32::from_le_bytes(src[pos..pos + 4].try_into().unwrap());
        pos += 4;
        if raw_sz == 0 {
            break; // EndMark
        }
        let uncompressed = (raw_sz & 0x8000_0000) != 0;
        let block_sz = (raw_sz & 0x7FFF_FFFF) as usize;
        if pos + block_sz > src.len() {
            return Err(ZarrError::Codec {
                codec: "lz4".into(),
                chunk_key: chunk_key.into(),
                detail: "truncated lz4 frame block".into(),
            });
        }
        let block = &src[pos..pos + block_sz];
        pos += block_sz;
        if has_block_checksum {
            if pos + 4 > src.len() {
                return Err(ZarrError::Codec {
                    codec: "lz4".into(),
                    chunk_key: chunk_key.into(),
                    detail: "truncated lz4 block checksum".into(),
                });
            }
            pos += 4;
        }
        let remaining = expected.saturating_sub(out.len());
        if uncompressed {
            if block.len() > remaining {
                return Err(ZarrError::Codec {
                    codec: "lz4".into(),
                    chunk_key: chunk_key.into(),
                    detail: "lz4 uncompressed block exceeds expected size".into(),
                });
            }
            out.extend_from_slice(block);
        } else {
            decompress_lz4_block(block, &mut out, remaining, chunk_key)?;
        }
    }
    Ok(out)
}

/// Byte-unshuffle (inverse of `numcodecs.Shuffle` and Blosc byte-shuffle).
pub fn unshuffle_bytes(
    src: &[u8],
    elem_size: usize,
    chunk_key: &str,
) -> Result<Vec<u8>, ZarrError> {
    if elem_size == 0 {
        return Err(ZarrError::Codec {
            codec: "shuffle".into(),
            chunk_key: chunk_key.into(),
            detail: "elementsize must be >= 1".into(),
        });
    }
    if elem_size == 1 || src.is_empty() {
        return Ok(src.to_vec());
    }
    let count = src.len() / elem_size;
    let rem = src.len() % elem_size;
    let mut out = vec![0u8; src.len()];
    for i in 0..count {
        for b in 0..elem_size {
            out[i * elem_size + b] = src[b * count + i];
        }
    }
    if rem > 0 {
        let tail = count * elem_size;
        out[tail..].copy_from_slice(&src[tail..]);
    }
    Ok(out)
}

/// Bit-unshuffle (inverse of Blosc/numcodecs bitshuffle).
///
/// Groups of 8 elements (`8 * elem_size` bytes) have their bits transposed across the 8 elements;
/// trailing elements `< 8` are byte-unshuffled, and any remaining bytes `< elem_size` are copied.
pub fn unbitshuffle_bytes(
    src: &[u8],
    elem_size: usize,
    chunk_key: &str,
) -> Result<Vec<u8>, ZarrError> {
    if elem_size == 0 {
        return Err(ZarrError::Codec {
            codec: "bitshuffle".into(),
            chunk_key: chunk_key.into(),
            detail: "elementsize must be >= 1".into(),
        });
    }
    if src.is_empty() {
        return Ok(Vec::new());
    }
    let n_elems = src.len() / elem_size;
    let block_elems = (n_elems / 8) * 8;
    let block_bytes = block_elems * elem_size;
    let mut out = vec![0u8; src.len()];

    // Bit-untranspose the main 8-element-aligned region:
    // Bit `bit_idx` (0..8*elem_size) of element `elem_idx` (0..block_elems) is stored at
    // bit `(elem_idx % 8)` of byte `(bit_idx * (block_elems / 8) + elem_idx / 8)`.
    let n_groups = block_elems / 8;
    let n_bits = elem_size * 8;
    for g in 0..n_groups {
        let base_elem = g * 8;
        for bit in 0..n_bits {
            let src_byte = src[bit * n_groups + g];
            if src_byte == 0 {
                continue;
            }
            let dst_byte_in_elem = bit / 8;
            let dst_bit_mask = 1u8 << (bit % 8);
            for lane in 0..8 {
                if (src_byte & (1u8 << lane)) != 0 {
                    out[(base_elem + lane) * elem_size + dst_byte_in_elem] |= dst_bit_mask;
                }
            }
        }
    }

    // Trailing elements (< 8) are byte-shuffled by Blosc/bitshuffle.
    let rem_elems = n_elems - block_elems;
    if rem_elems > 0 {
        let rem_bytes = rem_elems * elem_size;
        let tail_src = &src[block_bytes..block_bytes + rem_bytes];
        let tail_out = unshuffle_bytes(tail_src, elem_size, chunk_key)?;
        out[block_bytes..block_bytes + rem_bytes].copy_from_slice(&tail_out);
    }

    // Any leftover bytes (< elem_size) are copied verbatim.
    let covered = n_elems * elem_size;
    if covered < src.len() {
        out[covered..].copy_from_slice(&src[covered..]);
    }
    Ok(out)
}

fn parse_transpose_order(
    cfg: &Value,
    ndim: usize,
    chunk_key: &str,
) -> Result<Vec<usize>, ZarrError> {
    let Some(order_val) = cfg.get("order") else {
        return Err(ZarrError::Codec {
            codec: "transpose".into(),
            chunk_key: chunk_key.into(),
            detail: "missing transpose configuration.order".into(),
        });
    };
    if let Some(s) = order_val.as_str() {
        return match s {
            "C" => Ok((0..ndim).collect()),
            "F" => Ok((0..ndim).rev().collect()),
            other => Err(ZarrError::Codec {
                codec: "transpose".into(),
                chunk_key: chunk_key.into(),
                detail: format!("invalid transpose order string {other:?}"),
            }),
        };
    }
    let arr = order_val.as_array().ok_or_else(|| ZarrError::Codec {
        codec: "transpose".into(),
        chunk_key: chunk_key.into(),
        detail: "transpose order must be an array of axis indices".into(),
    })?;
    if arr.len() != ndim {
        return Err(ZarrError::Codec {
            codec: "transpose".into(),
            chunk_key: chunk_key.into(),
            detail: format!("transpose order len {} != chunk ndim {ndim}", arr.len()),
        });
    }
    let mut order = Vec::with_capacity(ndim);
    let mut seen = vec![false; ndim];
    for v in arr {
        let ax = v.as_u64().ok_or_else(|| ZarrError::Codec {
            codec: "transpose".into(),
            chunk_key: chunk_key.into(),
            detail: format!("non-integer axis {v}"),
        })? as usize;
        if ax >= ndim || seen[ax] {
            return Err(ZarrError::Codec {
                codec: "transpose".into(),
                chunk_key: chunk_key.into(),
                detail: format!("invalid or duplicate axis {ax} for ndim {ndim}"),
            });
        }
        seen[ax] = true;
        order.push(ax);
    }
    Ok(order)
}

/// Invert a Zarr v3 `transpose` codec whose `order` describes the stored axis permutation.
///
/// Stored shape is `stored_shape[k] = chunk_shape[order[k]]`. For every logical coordinate
/// `coords[0..ndim]`, its stored index is `[coords[order[0]], coords[order[1]], ...]`.
fn untranspose(
    src: &[u8],
    chunk_shape: &[u64],
    order: &[usize],
    elem_size: usize,
    chunk_key: &str,
) -> Result<Vec<u8>, ZarrError> {
    let ndim = chunk_shape.len();
    if ndim <= 1 || order.iter().enumerate().all(|(i, &ax)| i == ax) {
        return Ok(src.to_vec());
    }
    let dims: Vec<usize> = chunk_shape.iter().map(|&d| d as usize).collect();
    let total_elems: usize = dims.iter().product();
    if total_elems * elem_size != src.len() {
        return Err(ZarrError::Codec {
            codec: "transpose".into(),
            chunk_key: chunk_key.into(),
            detail: format!(
                "transpose buffer len {} != {} * {elem_size}",
                src.len(),
                total_elems
            ),
        });
    }

    // Compute C-order strides for the stored (transposed) layout.
    let stored_dims: Vec<usize> = order.iter().map(|&ax| dims[ax]).collect();
    let mut stored_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        stored_strides[i] = stored_strides[i + 1] * stored_dims[i + 1];
    }
    // Map each logical axis `ax` to its stride in `src`.
    let mut logical_to_src_stride = vec![0usize; ndim];
    for (stored_pos, &ax) in order.iter().enumerate() {
        logical_to_src_stride[ax] = stored_strides[stored_pos];
    }

    let mut out = vec![0u8; src.len()];
    let mut coords = vec![0usize; ndim];
    for dst_elem in 0..total_elems {
        let mut src_elem = 0usize;
        for ax in 0..ndim {
            src_elem += coords[ax] * logical_to_src_stride[ax];
        }
        let s0 = src_elem * elem_size;
        let d0 = dst_elem * elem_size;
        out[d0..d0 + elem_size].copy_from_slice(&src[s0..s0 + elem_size]);

        // Increment C-order logical coordinates.
        for ax in (0..ndim).rev() {
            coords[ax] += 1;
            if coords[ax] < dims[ax] {
                break;
            }
            coords[ax] = 0;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChunkKeyEncoding, CodecSpec};

    fn make_meta(shape: Vec<u64>, dtype: SampleType, endian: Endianness, codecs: Vec<CodecSpec>) -> ArrayMeta {
        ArrayMeta {
            version: ZarrVersion::V3,
            shape: shape.clone(),
            chunk_shape: shape,
            dtype,
            endianness: endian,
            fill_value: None,
            codecs,
            chunk_key: ChunkKeyEncoding::Default { separator: '/' },
            dimension_names: None,
            attributes: Default::default(),
            order_f: false,
        }
    }

    fn shuffle_bytes(src: &[u8], elem_size: usize) -> Vec<u8> {
        let count = src.len() / elem_size;
        let mut out = vec![0u8; src.len()];
        for i in 0..count {
            for b in 0..elem_size {
                out[b * count + i] = src[i * elem_size + b];
            }
        }
        out
    }

    fn compress_zstd(src: &[u8]) -> Vec<u8> {
        cogzilla_codec::compress_zstd(src, 3).unwrap()
    }

    #[test]
    fn three_codec_chain_bytes_shuffle_zstd_decodes_exact_samples() {
        let samples: Vec<f32> = (0..64).map(|i| i as f32 * 1.25 - 10.0).collect();
        // Encode order: bytes(big) -> shuffle(4) -> zstd
        let mut be_bytes = Vec::with_capacity(samples.len() * 4);
        for &v in &samples {
            be_bytes.extend_from_slice(&v.to_be_bytes());
        }
        let shuffled = shuffle_bytes(&be_bytes, 4);
        let compressed = compress_zstd(&shuffled);

        let meta = make_meta(
            vec![8, 8],
            SampleType::Float32,
            Endianness::Big,
            vec![
                CodecSpec {
                    name: "bytes".into(),
                    configuration: serde_json::json!({"endian": "big"}),
                },
                CodecSpec {
                    name: "shuffle".into(),
                    configuration: serde_json::json!({"elementsize": 4}),
                },
                CodecSpec {
                    name: "zstd".into(),
                    configuration: serde_json::json!({"level": 3}),
                },
            ],
        );

        let decoded = decode_chunk_bytes(&meta, &compressed, "c/0/0").unwrap();
        let got: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, samples);
    }

    #[test]
    fn zlib_and_lz4_block_and_frame_decode_cleanly() {
        let payload: Vec<u8> = (0..256u32).flat_map(|v| v.to_le_bytes()).collect();

        // zlib round-trip
        let zbuf = cogzilla_codec::compress_zlib(&payload).unwrap();
        let meta_z = make_meta(
            vec![16, 16],
            SampleType::Uint32,
            Endianness::Little,
            vec![CodecSpec {
                name: "zlib".into(),
                configuration: Value::Null,
            }],
        );
        assert_eq!(decode_chunk_bytes(&meta_z, &zbuf, "0.0").unwrap(), payload);

        // numcodecs LZ4 raw block (literal-only block: 16 bytes)
        let small: Vec<u8> = (0..16u8).collect();
        let mut lz4_raw = Vec::new();
        lz4_raw.extend_from_slice(&(16u32).to_le_bytes());
        lz4_raw.push(0xF0); // lit_len = 15 + next byte
        lz4_raw.push(1); // +1 = 16 literals, no match
        lz4_raw.extend_from_slice(&small);
        let meta_lz4 = make_meta(
            vec![4, 4],
            SampleType::Uint8,
            Endianness::Little,
            vec![CodecSpec {
                name: "lz4".into(),
                configuration: Value::Null,
            }],
        );
        assert_eq!(decode_chunk_bytes(&meta_lz4, &lz4_raw, "0.0").unwrap(), small);

        // LZ4 frame with uncompressed block
        let mut lz4_frame = vec![0x04, 0x22, 0x4D, 0x18, 0x60, 0x40, 0x82];
        lz4_frame.extend_from_slice(&(0x8000_0000u32 | 16).to_le_bytes());
        lz4_frame.extend_from_slice(&small);
        lz4_frame.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(decode_chunk_bytes(&meta_lz4, &lz4_frame, "0.0").unwrap(), small);
    }

    #[test]
    fn unknown_codec_is_named_error_listing_supported() {
        let meta = make_meta(
            vec![4, 4],
            SampleType::Uint8,
            Endianness::Little,
            vec![CodecSpec {
                name: "mystery_codec".into(),
                configuration: Value::Null,
            }],
        );
        let err = decode_chunk_bytes(&meta, &[0u8; 16], "c/1/2").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mystery_codec"), "{msg}");
        assert!(msg.contains("c/1/2"), "{msg}");
        assert!(msg.contains("zstd"), "{msg}");
    }

    #[test]
    fn truncated_and_oversized_payloads_never_panic_or_overallocate() {
        let meta_zstd = make_meta(
            vec![4, 4],
            SampleType::Float32,
            Endianness::Little,
            vec![CodecSpec {
                name: "zstd".into(),
                configuration: Value::Null,
            }],
        );
        // Truncated zstd
        assert!(decode_chunk_bytes(&meta_zstd, &[0x28, 0xb5], "c/0/0").is_err());

        // Hostile LZ4 header claiming 2 GiB
        let meta_lz4 = make_meta(
            vec![4, 4],
            SampleType::Float32,
            Endianness::Little,
            vec![CodecSpec {
                name: "lz4".into(),
                configuration: Value::Null,
            }],
        );
        let hostile = 0x7FFF_FFFFu32.to_le_bytes();
        assert!(decode_chunk_bytes(&meta_lz4, &hostile, "c/0/0").is_err());

        // Chunk shape exceeding MAX_DECODED_CHUNK_BYTES (e.g. 10000 x 10000 f32 = 400 MB)
        let huge = make_meta(vec![10000, 10000], SampleType::Float32, Endianness::Little, vec![]);
        let err = decode_chunk_bytes(&huge, &[], "c/0/0").unwrap_err();
        assert!(matches!(err, ZarrError::ChunkTooLarge { .. }), "{err}");
    }
}
