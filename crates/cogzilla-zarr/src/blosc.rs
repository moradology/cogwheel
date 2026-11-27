//! Pure-Rust Blosc 1.x container decoder.
//!
//! Parses the 16-byte Blosc header and block offset table, dispatches each block/stream to the
//! workspace's existing decoders (`zstd-sys`, `libdeflater` zlib, pure-Rust `lz4`, and pure-Rust
//! `blosclz`), and applies the header's `shuffle` or `bitshuffle` inverse filter.

use crate::codecs::{
    MAX_DECODED_CHUNK_BYTES, decompress_lz4_block, decompress_zlib_raw, decompress_zstd_raw,
    unbitshuffle_bytes, unshuffle_bytes,
};
use crate::ZarrError;

const BLOSC_MIN_HEADER_LEN: usize = 16;

// Header flag bits (Blosc 1.x)
const BLOSC_DOSHUFFLE: u8 = 0x01;
const BLOSC_MEMCPYED: u8 = 0x02;
const BLOSC_DOBITSHUFFLE: u8 = 0x04;
const BLOSC_DODELTA: u8 = 0x08;
// Bit 4 (0x10) is set by c-blosc when a block is NOT split (e.g. zstd, zlib, or nosplit);
// when bit 4 is clear AND BLOSC_DOSHUFFLE is set (blosclz/lz4), the block is split into
// `typesize` streams.
const BLOSC_NON_SPLIT: u8 = 0x10;

// Compressor codes stored in bits 5..7 of `flags`
const COMP_BLOSCLZ: u8 = 0;
const COMP_LZ4: u8 = 1; // covers both lz4 and lz4hc
const COMP_SNAPPY: u8 = 2;
const COMP_ZLIB: u8 = 3;
const COMP_ZSTD: u8 = 4;

/// Parsed 16-byte Blosc 1.x header.
#[derive(Debug, Clone, Copy)]
pub struct BloscHeader {
    pub version: u8,
    pub versionlz: u8,
    pub flags: u8,
    pub typesize: usize,
    pub nbytes: usize,
    pub blocksize: usize,
    pub cbytes: usize,
}

impl BloscHeader {
    pub fn parse(src: &[u8], chunk_key: &str) -> Result<Self, ZarrError> {
        if src.len() < BLOSC_MIN_HEADER_LEN {
            return Err(blosc_err(
                chunk_key,
                format!("truncated blosc header ({} < 16 bytes)", src.len()),
            ));
        }
        let version = src[0];
        let versionlz = src[1];
        let flags = src[2];
        let typesize = src[3] as usize;
        let nbytes = u32::from_le_bytes(src[4..8].try_into().unwrap()) as usize;
        let blocksize = u32::from_le_bytes(src[8..12].try_into().unwrap()) as usize;
        let cbytes = u32::from_le_bytes(src[12..16].try_into().unwrap()) as usize;

        if typesize == 0 {
            return Err(blosc_err(chunk_key, "blosc header has typesize 0"));
        }
        if cbytes < BLOSC_MIN_HEADER_LEN || cbytes > src.len() {
            return Err(blosc_err(
                chunk_key,
                format!(
                    "blosc cbytes {cbytes} out of bounds for buffer of {} bytes",
                    src.len()
                ),
            ));
        }

        Ok(Self {
            version,
            versionlz,
            flags,
            typesize,
            nbytes,
            blocksize,
            cbytes,
        })
    }

    pub fn compcode(&self) -> u8 {
        self.flags >> 5
    }
}

fn blosc_err(chunk_key: &str, detail: impl Into<String>) -> ZarrError {
    ZarrError::Codec {
        codec: "blosc".into(),
        chunk_key: chunk_key.into(),
        detail: detail.into(),
    }
}

/// Decompress a Blosc 1.x buffer, enforcing `expected` and `MAX_DECODED_CHUNK_BYTES` before
/// allocating.
pub fn decompress_blosc(
    src: &[u8],
    expected: usize,
    chunk_key: &str,
) -> Result<Vec<u8>, ZarrError> {
    let hdr = BloscHeader::parse(src, chunk_key)?;
    if hdr.nbytes > expected || hdr.nbytes > MAX_DECODED_CHUNK_BYTES {
        return Err(blosc_err(
            chunk_key,
            format!(
                "blosc header declares {} bytes, exceeding cap {expected}",
                hdr.nbytes
            ),
        ));
    }
    if hdr.nbytes == 0 {
        return Ok(Vec::new());
    }

    // Fast path: raw uncompressed memcpy'd payload right after the 16-byte header.
    if (hdr.flags & BLOSC_MEMCPYED) != 0 {
        let end = BLOSC_MIN_HEADER_LEN + hdr.nbytes;
        if end > src.len() {
            return Err(blosc_err(
                chunk_key,
                format!("truncated blosc memcpy payload (need {end}, have {})", src.len()),
            ));
        }
        return Ok(src[BLOSC_MIN_HEADER_LEN..end].to_vec());
    }

    if hdr.blocksize == 0 {
        return Err(blosc_err(chunk_key, "blosc header has blocksize 0"));
    }

    let nblocks = hdr.nbytes.div_ceil(hdr.blocksize);
    let bstarts_end = BLOSC_MIN_HEADER_LEN
        .checked_add(nblocks.checked_mul(4).ok_or_else(|| blosc_err(chunk_key, "nblocks overflow"))?)
        .ok_or_else(|| blosc_err(chunk_key, "bstarts overflow"))?;
    if bstarts_end > hdr.cbytes {
        return Err(blosc_err(
            chunk_key,
            format!("truncated blosc block index (need {bstarts_end}, cbytes {})", hdr.cbytes),
        ));
    }

    let compcode = hdr.compcode();
    let do_shuffle = (hdr.flags & BLOSC_DOSHUFFLE) != 0;
    let do_bitshuffle = (hdr.flags & BLOSC_DOBITSHUFFLE) != 0;
    let do_delta = (hdr.flags & BLOSC_DODELTA) != 0;
    let split = do_shuffle && (hdr.flags & BLOSC_NON_SPLIT) == 0;

    let mut out = vec![0u8; hdr.nbytes];
    let mut block_scratch = Vec::with_capacity(hdr.blocksize);
    let mut stream_scratch = Vec::new();

    for j in 0..nblocks {
        let off_pos = BLOSC_MIN_HEADER_LEN + j * 4;
        let bstart = i32::from_le_bytes(src[off_pos..off_pos + 4].try_into().unwrap());
        if bstart < 0 || (bstart as usize) < bstarts_end || (bstart as usize) > hdr.cbytes {
            return Err(blosc_err(
                chunk_key,
                format!("invalid blosc block {j} offset {bstart}"),
            ));
        }
        let bsize = if j + 1 == nblocks {
            hdr.nbytes - j * hdr.blocksize
        } else {
            hdr.blocksize
        };

        let mut pos = bstart as usize;
        block_scratch.clear();

        // Blosc 1.x splits a full block into `typesize` independent streams when BLOSC_SPLIT is
        // set; the trailing partial block (`bsize < hdr.blocksize`) is never split.
        let is_split_block = split
            && hdr.typesize > 1
            && bsize == hdr.blocksize
            && bsize >= hdr.typesize
            && bsize % hdr.typesize == 0;
        let nstreams = if is_split_block { hdr.typesize } else { 1 };
        let stream_uncomp = bsize / nstreams;

        for _s in 0..nstreams {
            if pos + 4 > hdr.cbytes {
                return Err(blosc_err(chunk_key, format!("truncated stream header in block {j}")));
            }
            let ctbytes_raw = i32::from_le_bytes(src[pos..pos + 4].try_into().unwrap());
            pos += 4;
            if ctbytes_raw < 0 {
                return Err(blosc_err(
                    chunk_key,
                    format!("negative stream cbytes {ctbytes_raw} in block {j}"),
                ));
            }
            let ctbytes = ctbytes_raw as usize;
            if ctbytes == 0 {
                // Zero-run optimization: stream is all zero bytes.
                block_scratch.resize(block_scratch.len() + stream_uncomp, 0u8);
                continue;
            }
            if pos + ctbytes > hdr.cbytes {
                return Err(blosc_err(
                    chunk_key,
                    format!("truncated stream payload in block {j} (need {ctbytes})"),
                ));
            }
            let comp_slice = &src[pos..pos + ctbytes];
            pos += ctbytes;

            if ctbytes == stream_uncomp {
                // Uncompressed stream (stored verbatim because compression did not shrink it).
                block_scratch.extend_from_slice(comp_slice);
            } else {
                stream_scratch.clear();
                decompress_blosc_stream(
                    compcode,
                    comp_slice,
                    stream_uncomp,
                    &mut stream_scratch,
                    chunk_key,
                )?;
                if stream_scratch.len() != stream_uncomp {
                    return Err(blosc_err(
                        chunk_key,
                        format!(
                            "block {j} stream decoded {} bytes, expected {stream_uncomp}",
                            stream_scratch.len()
                        ),
                    ));
                }
                block_scratch.extend_from_slice(&stream_scratch);
            }
        }

        // Apply inverse filter(s) to the decoded block.
        let filtered = if do_bitshuffle {
            unbitshuffle_bytes(&block_scratch, hdr.typesize, chunk_key)?
        } else if do_shuffle {
            unshuffle_bytes(&block_scratch, hdr.typesize, chunk_key)?
        } else {
            block_scratch.clone()
        };

        let dst_start = j * hdr.blocksize;
        out[dst_start..dst_start + bsize].copy_from_slice(&filtered);
    }

    if do_delta {
        // Blosc delta filter stores differences relative to the first block's reference bytes.
        // Undo per-block delta against the first block (if multiple blocks).
        let ref_len = hdr.blocksize.min(hdr.nbytes);
        for j in 1..nblocks {
            let start = j * hdr.blocksize;
            let len = (hdr.nbytes - start).min(ref_len);
            for k in 0..len {
                out[start + k] = out[start + k].wrapping_add(out[k]);
            }
        }
    }

    Ok(out)
}

fn decompress_blosc_stream(
    compcode: u8,
    src: &[u8],
    expected: usize,
    out: &mut Vec<u8>,
    chunk_key: &str,
) -> Result<(), ZarrError> {
    match compcode {
        COMP_BLOSCLZ => decompress_blosclz(src, expected, out, chunk_key),
        COMP_LZ4 => {
            out.reserve(expected);
            decompress_lz4_block(src, out, expected, chunk_key)
        }
        COMP_ZLIB => {
            let dec = decompress_zlib_raw(src, expected, chunk_key)?;
            out.extend_from_slice(&dec);
            Ok(())
        }
        COMP_ZSTD => {
            let dec = decompress_zstd_raw(src, expected, chunk_key)?;
            out.extend_from_slice(&dec);
            Ok(())
        }
        COMP_SNAPPY => Err(blosc_err(
            chunk_key,
            "blosc snappy sub-codec is not supported (supported: blosclz, lz4, zlib, zstd)",
        )),
        other => Err(blosc_err(
            chunk_key,
            format!("unknown blosc sub-codec id {other}"),
        )),
    }
}

/// Pure-Rust bounds-checked BloscLZ decoder (Blosc1 format).
fn decompress_blosclz(
    src: &[u8],
    expected: usize,
    out: &mut Vec<u8>,
    chunk_key: &str,
) -> Result<(), ZarrError> {
    out.clear();
    out.reserve(expected);
    if src.is_empty() {
        return Ok(());
    }
    let mut ip = 0usize;
    let mut first = true;
    while ip < src.len() {
        // The top 3 bits of the very first byte in a BloscLZ/FastLZ stream carry the format
        // level; masking with 31 yields the initial literal run count.
        let ctrl = if first {
            first = false;
            src[ip] & 31
        } else {
            src[ip]
        };
        ip += 1;
        if ctrl < 32 {
            // Literal run: (ctrl + 1) bytes
            let lit_len = (ctrl as usize) + 1;
            if ip + lit_len > src.len() {
                return Err(blosc_err(chunk_key, "blosclz truncated literal run"));
            }
            if out.len() + lit_len > expected {
                return Err(blosc_err(chunk_key, "blosclz literal run exceeds block size"));
            }
            out.extend_from_slice(&src[ip..ip + lit_len]);
            ip += lit_len;
        } else {
            // Match copy
            let mut match_len = (ctrl >> 5) as usize;
            let mut ofs = ((ctrl & 31) as usize) << 8;
            if match_len == 7 {
                loop {
                    if ip >= src.len() {
                        return Err(blosc_err(chunk_key, "blosclz truncated match length"));
                    }
                    let code = src[ip] as usize;
                    ip += 1;
                    match_len = match_len.saturating_add(code);
                    if code != 255 {
                        break;
                    }
                }
            }
            match_len += 2;
            if ip >= src.len() {
                return Err(blosc_err(chunk_key, "blosclz truncated match offset"));
            }
            let ofs_lo = src[ip] as usize;
            ip += 1;
            ofs += ofs_lo;
            // BloscLZ extended 16-bit big-endian offset when low 13 bits are all ones (0x1FFF).
            if ofs == 0x1FFF {
                if ip + 2 > src.len() {
                    return Err(blosc_err(chunk_key, "blosclz truncated extended offset"));
                }
                ofs += ((src[ip] as usize) << 8) | (src[ip + 1] as usize);
                ip += 2;
            }
            let dist = ofs + 1;
            if dist > out.len() {
                return Err(blosc_err(
                    chunk_key,
                    format!("blosclz invalid match distance {dist} at output len {}", out.len()),
                ));
            }
            if out.len() + match_len > expected {
                return Err(blosc_err(chunk_key, "blosclz match exceeds block size"));
            }
            for _ in 0..match_len {
                let b = out[out.len() - dist];
                out.push(b);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blosc_memcpy_and_hostile_headers_are_handled() {
        // Valid memcpy'd blosc buffer (4 float32s = 16 bytes)
        let payload: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut buf = vec![0u8; 16 + payload.len()];
        buf[0] = 2; // version
        buf[1] = 1; // versionlz
        buf[2] = BLOSC_MEMCPYED;
        buf[3] = 4; // typesize
        buf[4..8].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        buf[8..12].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        let total_len = buf.len() as u32;
        buf[12..16].copy_from_slice(&total_len.to_le_bytes());
        buf[16..].copy_from_slice(&payload);

        let dec = decompress_blosc(&buf, payload.len(), "c/0/0").unwrap();
        assert_eq!(dec, payload);

        // Hostile header claiming 1 GiB nbytes
        let mut hostile = buf.clone();
        hostile[4..8].copy_from_slice(&(1024u32 * 1024 * 1024).to_le_bytes());
        assert!(decompress_blosc(&hostile, payload.len(), "c/0/0").is_err());

        // Truncated header
        assert!(decompress_blosc(&[2, 1, 0, 4], 16, "c/0/0").is_err());
    }

    #[test]
    fn blosc_trailing_partial_block_is_not_split() {
        // Construct a valid Blosc 1.x split-shuffle payload with 2 blocks:
        // block 0: full blocksize=16 (typesize=4 -> 4 split streams of 4 uncompressed bytes each)
        // block 1: trailing partial bsize=8 < blocksize=16 (NOT split -> 1 stream of 8 bytes)
        let typesize = 4usize;
        let blocksize = 16usize;
        let nbytes = 24usize;
        let b0_shuffled = [
            0u8, 4, 8, 12, // stream 0
            1, 5, 9, 13, // stream 1
            2, 6, 10, 14, // stream 2
            3, 7, 11, 15, // stream 3
        ];
        let b1_shuffled = [16u8, 20, 17, 21, 18, 22, 19, 23];

        let mut buf = vec![0u8; 16 + 8]; // 16-byte header + 2 * 4-byte bstarts
        let b0_start = buf.len() as i32;
        for s in 0..4 {
            buf.extend_from_slice(&4i32.to_le_bytes()); // uncompressed stream
            buf.extend_from_slice(&b0_shuffled[s * 4..(s + 1) * 4]);
        }
        let b1_start = buf.len() as i32;
        buf.extend_from_slice(&8i32.to_le_bytes()); // single unsplit stream for partial block
        buf.extend_from_slice(&b1_shuffled);

        let cbytes = buf.len() as i32;
        buf[0] = 2; // version
        buf[1] = 1; // versionlz
        buf[2] = BLOSC_DOSHUFFLE | 0x20; // shuffle + split (BLOSC_NON_SPLIT=0) + LZ4
        buf[3] = typesize as u8;
        buf[4..8].copy_from_slice(&(nbytes as i32).to_le_bytes());
        buf[8..12].copy_from_slice(&(blocksize as i32).to_le_bytes());
        buf[12..16].copy_from_slice(&cbytes.to_le_bytes());
        buf[16..20].copy_from_slice(&b0_start.to_le_bytes());
        buf[20..24].copy_from_slice(&b1_start.to_le_bytes());

        let decoded = decompress_blosc(&buf, nbytes, "era5/0.0.0").unwrap();
        let expected: Vec<u8> = (0u8..24).collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    #[ignore]
    fn bench_blosc_throughput_4mb() {
        let dir = std::path::Path::new("/tmp/blosc_bench");
        for name in ["zstd", "lz4", "blosclz", "zlib"] {
            let enc = std::fs::read(dir.join(format!("{name}.blosc"))).unwrap();
            let expected = 1024 * 1024 * 4;
            for _ in 0..3 {
                let d = decompress_blosc(&enc, expected, name).unwrap();
                assert_eq!(d.len(), expected);
            }
            let iters = 50usize;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let d = decompress_blosc(&enc, expected, name).unwrap();
                std::hint::black_box(d);
            }
            let dt = t0.elapsed().as_secs_f64();
            let mb_s = (expected * iters) as f64 / 1e6 / dt;
            eprintln!("rust-blosc {name:8}: {mb_s:7.1} MB/s (1-thread)");
        }
    }
}
