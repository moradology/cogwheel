use crate::CodecError;

const CLEAR_CODE: u16 = 256;
const EOI_CODE: u16 = 257;
const FIRST_CODE: u16 = 258;
const MAX_TABLE: usize = 4096;

/// Reusable thread-local flat-table state for high-throughput MSB-first TIFF LZW decoding.
///
/// Compact 8-byte dictionary entries (`[u64; 4096]` = 32 KiB, 100% L1d-resident on Zen 4 & M5):
/// - Bits `48..64`: decoded string length (`1..=4096`).
/// - If `len <= 6`: bits `0..48` hold the exact 1..6 decoded bytes inline (little-endian),
///   emitted with a single unaligned 8-byte store (zero loads from `out`, zero prefix chasing).
/// - If `len > 6`: bits `0..8` hold `first_byte`, bits `16..48` hold the `u32` byte offset in `out`
///   where this string was previously written, emitted via a linear non-overlapping copy.
/// - In both cases, `entry as u8` is always `first_byte`.
pub struct LzwDecoder {
    entries: [u64; MAX_TABLE],
}

impl LzwDecoder {
    pub fn new() -> Self {
        let mut entries = [0u64; MAX_TABLE];
        for (i, e) in entries.iter_mut().take(256).enumerate() {
            *e = (1u64 << 48) | (i as u64);
        }
        Self { entries }
    }

    /// Decode a TIFF MSB-first (early-change) LZW stream directly into `out`.
    /// Returns the number of decoded bytes written.
    #[inline(never)]
    pub fn decode(&mut self, src: &[u8], out: &mut [u8]) -> Result<usize, CodecError> {
        let src_len = src.len();
        let src_ptr = src.as_ptr();
        let src_bits = src_len * 8;

        let out_len = out.len();
        if out_len > u32::MAX as usize {
            return Err(CodecError::Lzw("LZW output buffer exceeds 4 GiB limit"));
        }
        let out_ptr = out.as_mut_ptr();

        let mut bit_pos: usize = 0;
        let mut out_pos: usize = 0;

        let mut code_bits: u32 = 9;
        let mut max_code: u16 = (1u16 << 9) - 1;
        let mut next_code: u16 = FIRST_CODE;
        let mut prev_code: u16 = CLEAR_CODE;
        let mut prev_out_pos: usize = 0;
        let mut prev_entry: u64 = 0;

        let entries = &mut self.entries;

        #[inline(always)]
        unsafe fn read_code_slow(
            src_ptr: *const u8,
            src_len: usize,
            bit_pos: &mut usize,
            code_bits: u32,
        ) -> Option<u16> {
            let bp = *bit_pos;
            if bp + (code_bits as usize) > src_len * 8 {
                return None;
            }
            let byte_pos = bp >> 3;
            let bit_off = (bp & 7) as u32;
            let mut chunk: u64 = 0;
            let avail = (src_len - byte_pos).min(8);
            for k in 0..avail {
                chunk |= (unsafe { *src_ptr.add(byte_pos + k) } as u64) << (56 - k * 8);
            }
            *bit_pos = bp + (code_bits as usize);
            Some(((chunk << bit_off) >> (64 - code_bits)) as u16)
        }

        // Emit one normal code (`code < next_code && (code & !1) != CLEAR_CODE`) when
        // `prev_code != CLEAR_CODE` is already guaranteed.
        macro_rules! emit_normal_12 {
            ($code:expr, $insert_table:expr) => {{
                let entry = unsafe { *entries.get_unchecked($code as usize) };
                let len = (entry >> 48) as usize;
                let first = entry as u8;
                if len <= 6 {
                    if out_pos + 8 <= out_len {
                        unsafe {
                            (out_ptr.add(out_pos) as *mut u64).write_unaligned(entry);
                        }
                    } else {
                        let avail = (out_len - out_pos).min(len);
                        let bytes = entry.to_le_bytes();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                out_ptr.add(out_pos),
                                avail,
                            );
                        }
                        if avail < len {
                            return Ok(out_len);
                        }
                    }
                } else if out_pos + len <= out_len {
                    let off = ((entry >> 16) & 0xFFFF_FFFF) as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            out_ptr.add(off),
                            out_ptr.add(out_pos),
                            len,
                        );
                    }
                } else {
                    let avail = out_len - out_pos;
                    let off = ((entry >> 16) & 0xFFFF_FFFF) as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            out_ptr.add(off),
                            out_ptr.add(out_pos),
                            avail,
                        );
                    }
                    return Ok(out_len);
                }
                if $insert_table && (next_code as usize) < MAX_TABLE {
                    let prev_len = prev_entry >> 48;
                    let new_entry = if prev_len < 6 {
                        (prev_entry + (1u64 << 48)) | ((first as u64) << (prev_len * 8))
                    } else {
                        ((prev_len + 1) << 48)
                            | ((prev_out_pos as u64) << 16)
                            | (prev_entry & 0xFF)
                    };
                    unsafe { *entries.get_unchecked_mut(next_code as usize) = new_entry };
                    next_code += 1;
                }
                prev_out_pos = out_pos;
                prev_entry = entry;
                out_pos += len;
            }};
        }

        // Fast inner loop: guaranteed `byte_pos + 8 <= src_len` and `out_pos + 16 <= out_len`
        // so bit reading is completely branchless and inline 8-byte stores need zero bounds checks.
        // Crucially, literals `0..256` and dictionary entries `258..next_code` share the EXACT
        // same `entries[code]` fast path, eliminating the 50/50 unpredictable `code < 256` branch
        // misprediction storm on high-entropy `LZW_P1` blocks!
        'main: while (bit_pos >> 3) + 8 <= src_len && out_pos + 16 <= out_len {
            // Specialized 4-code 12-bit batch fast path: 4 * 12 + (<=7) = 55 <= 64 bits, so a
            // single `u64::from_be_bytes` load feeds 4 codes with zero refill branches, constant
            // shifts, and no `code_bits` growth checks.
            if code_bits == 12 && prev_code != CLEAR_CODE {
                while (bit_pos >> 3) + 8 <= src_len && out_pos + 40 <= out_len {
                    let byte_pos = bit_pos >> 3;
                    let bit_off = (bit_pos & 7) as u32;
                    let aligned = u64::from_be_bytes(unsafe {
                        (src_ptr.add(byte_pos) as *const [u8; 8]).read_unaligned()
                    }) << bit_off;

                    let c0 = (aligned >> 52) as u16;
                    let c1 = ((aligned >> 40) & 0xFFF) as u16;
                    let c2 = ((aligned >> 28) & 0xFFF) as u16;
                    let c3 = ((aligned >> 16) & 0xFFF) as u16;

                    if c0 < next_code
                        && (c0 & !1) != CLEAR_CODE
                        && c1 < next_code
                        && (c1 & !1) != CLEAR_CODE
                        && c2 < next_code
                        && (c2 & !1) != CLEAR_CODE
                        && c3 < next_code
                        && (c3 & !1) != CLEAR_CODE
                    {
                        if (next_code as usize) == MAX_TABLE {
                            emit_normal_12!(c0, false);
                            emit_normal_12!(c1, false);
                            emit_normal_12!(c2, false);
                            emit_normal_12!(c3, false);
                        } else {
                            emit_normal_12!(c0, true);
                            emit_normal_12!(c1, true);
                            emit_normal_12!(c2, true);
                            emit_normal_12!(c3, true);
                        }
                        prev_code = c3;
                        bit_pos += 48;
                        continue;
                    }
                    break;
                }
                if (bit_pos >> 3) + 8 > src_len || out_pos + 16 > out_len {
                    break 'main;
                }
            }

            let byte_pos = bit_pos >> 3;
            let bit_off = (bit_pos & 7) as u32;
            let chunk = u64::from_be_bytes(unsafe {
                (src_ptr.add(byte_pos) as *const [u8; 8]).read_unaligned()
            });
            let code = ((chunk << bit_off) >> (64 - code_bits)) as u16;
            bit_pos += code_bits as usize;

            if code < next_code && (code & !1) != CLEAR_CODE {
                let entry = unsafe { *entries.get_unchecked(code as usize) };
                let len = (entry >> 48) as usize;
                let first = entry as u8;

                if len <= 6 {
                    if out_pos + 8 <= out_len {
                        unsafe {
                            (out_ptr.add(out_pos) as *mut u64).write_unaligned(entry);
                        }
                    } else {
                        let avail = (out_len - out_pos).min(len);
                        let bytes = entry.to_le_bytes();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                out_ptr.add(out_pos),
                                avail,
                            );
                        }
                        if avail < len {
                            return Ok(out_len);
                        }
                    }
                } else if out_pos + len <= out_len {
                    let off = ((entry >> 16) & 0xFFFF_FFFF) as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            out_ptr.add(off),
                            out_ptr.add(out_pos),
                            len,
                        );
                    }
                } else {
                    let avail = out_len - out_pos;
                    let off = ((entry >> 16) & 0xFFFF_FFFF) as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            out_ptr.add(off),
                            out_ptr.add(out_pos),
                            avail,
                        );
                    }
                    return Ok(out_len);
                }

                if prev_code != CLEAR_CODE && (next_code as usize) < MAX_TABLE {
                    let prev_len = prev_entry >> 48;
                    let new_entry = if prev_len < 6 {
                        (prev_entry + (1u64 << 48)) | ((first as u64) << (prev_len * 8))
                    } else {
                        ((prev_len + 1) << 48)
                            | ((prev_out_pos as u64) << 16)
                            | (prev_entry & 0xFF)
                    };
                    unsafe { *entries.get_unchecked_mut(next_code as usize) = new_entry };
                    next_code += 1;
                    if next_code == max_code && code_bits < 12 {
                        code_bits += 1;
                        max_code = (1u16 << code_bits) - 1;
                    }
                }
                prev_out_pos = out_pos;
                prev_entry = entry;
                prev_code = code;
                out_pos += len;
                continue;
            }

            if code == CLEAR_CODE {
                code_bits = 9;
                max_code = (1u16 << 9) - 1;
                next_code = FIRST_CODE;
                prev_code = CLEAR_CODE;
                continue;
            }

            if code == EOI_CODE {
                // Scrub up to 8 bytes past `out_pos` that the previous 8-byte `write_unaligned`
                // fast-path entry may have written (bits 48..64 carry entry length).
                let clear_end = (out_pos + 8).min(out_len);
                out[out_pos..clear_end].fill(0);
                return Ok(out_pos);
            }

            if code == next_code && prev_code != CLEAR_CODE {
                let first = prev_entry as u8;
                let prev_len = prev_entry >> 48;
                let new_len = (prev_len as usize) + 1;
                let new_entry = if prev_len < 6 {
                    (prev_entry + (1u64 << 48)) | ((first as u64) << (prev_len * 8))
                } else {
                    ((prev_len + 1) << 48) | ((out_pos as u64) << 16) | (first as u64)
                };

                if new_len <= 6 {
                    unsafe {
                        (out_ptr.add(out_pos) as *mut u64).write_unaligned(new_entry);
                    }
                } else if out_pos + new_len <= out_len {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            out_ptr.add(prev_out_pos),
                            out_ptr.add(out_pos),
                            prev_len as usize,
                        );
                        *out_ptr.add(out_pos + prev_len as usize) = first;
                    }
                } else {
                    let avail = out_len - out_pos;
                    let copy_prev = avail.min(prev_len as usize);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            out_ptr.add(prev_out_pos),
                            out_ptr.add(out_pos),
                            copy_prev,
                        );
                        if avail > prev_len as usize {
                            *out_ptr.add(out_pos + prev_len as usize) = first;
                        }
                    }
                    return Ok(out_len);
                }

                if (next_code as usize) < MAX_TABLE {
                    unsafe { *entries.get_unchecked_mut(next_code as usize) = new_entry };
                    next_code += 1;
                    if next_code == max_code && code_bits < 12 {
                        code_bits += 1;
                        max_code = (1u16 << code_bits) - 1;
                    }
                }
                prev_out_pos = out_pos;
                prev_entry = new_entry;
                prev_code = code;
                out_pos += new_len;
                continue;
            }

            return Err(CodecError::Lzw("out-of-range LZW code"));
        }

        // Tail loop (last <= 8 input bytes or last <= 16 output bytes)
        while bit_pos + (code_bits as usize) <= src_bits && out_pos < out_len {
            let Some(code) = (unsafe { read_code_slow(src_ptr, src_len, &mut bit_pos, code_bits) })
            else {
                break;
            };
            if code == EOI_CODE {
                break;
            }
            if code == CLEAR_CODE {
                code_bits = 9;
                max_code = (1u16 << 9) - 1;
                next_code = FIRST_CODE;
                prev_code = CLEAR_CODE;
                continue;
            }
            if prev_code == CLEAR_CODE {
                if code >= 256 {
                    return Err(CodecError::Lzw("first code after CLEAR must be literal"));
                }
                unsafe { *out_ptr.add(out_pos) = code as u8 };
                prev_out_pos = out_pos;
                prev_entry = (1u64 << 48) | (code as u64);
                prev_code = code;
                out_pos += 1;
                continue;
            }
            if code < 256 {
                unsafe { *out_ptr.add(out_pos) = code as u8 };
                if (next_code as usize) < MAX_TABLE {
                    let prev_len = prev_entry >> 48;
                    let new_entry = if prev_len < 6 {
                        (prev_entry + (1u64 << 48)) | ((code as u64) << (prev_len * 8))
                    } else {
                        ((prev_len + 1) << 48)
                            | ((prev_out_pos as u64) << 16)
                            | (prev_entry & 0xFF)
                    };
                    unsafe { *entries.get_unchecked_mut(next_code as usize) = new_entry };
                    next_code += 1;
                    if next_code == max_code && code_bits < 12 {
                        code_bits += 1;
                        max_code = (1u16 << code_bits) - 1;
                    }
                }
                prev_out_pos = out_pos;
                prev_entry = (1u64 << 48) | (code as u64);
                prev_code = code;
                out_pos += 1;
            } else if code < next_code {
                let entry = unsafe { *entries.get_unchecked(code as usize) };
                let len = (entry >> 48) as usize;
                let first = entry as u8;
                let avail = (out_len - out_pos).min(len);
                if len <= 6 {
                    let bytes = entry.to_le_bytes();
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr.add(out_pos), avail);
                    }
                } else {
                    let off = ((entry >> 16) & 0xFFFF_FFFF) as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(out_ptr.add(off), out_ptr.add(out_pos), avail);
                    }
                }
                if avail < len {
                    out_pos += avail;
                    break;
                }
                if (next_code as usize) < MAX_TABLE {
                    let prev_len = prev_entry >> 48;
                    let new_entry = if prev_len < 6 {
                        (prev_entry + (1u64 << 48)) | ((first as u64) << (prev_len * 8))
                    } else {
                        ((prev_len + 1) << 48)
                            | ((prev_out_pos as u64) << 16)
                            | (prev_entry & 0xFF)
                    };
                    unsafe { *entries.get_unchecked_mut(next_code as usize) = new_entry };
                    next_code += 1;
                    if next_code == max_code && code_bits < 12 {
                        code_bits += 1;
                        max_code = (1u16 << code_bits) - 1;
                    }
                }
                prev_out_pos = out_pos;
                prev_entry = entry;
                prev_code = code;
                out_pos += len;
            } else if code == next_code {
                let first = prev_entry as u8;
                let prev_len = prev_entry >> 48;
                let new_len = (prev_len as usize) + 1;
                let avail = (out_len - out_pos).min(new_len);
                let copy_prev = avail.min(prev_len as usize);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        out_ptr.add(prev_out_pos),
                        out_ptr.add(out_pos),
                        copy_prev,
                    );
                    if avail > prev_len as usize {
                        *out_ptr.add(out_pos + prev_len as usize) = first;
                    }
                }
                if avail < new_len {
                    out_pos += avail;
                    break;
                }
                let new_entry = if prev_len < 6 {
                    (prev_entry + (1u64 << 48)) | ((first as u64) << (prev_len * 8))
                } else {
                    ((prev_len + 1) << 48) | ((out_pos as u64) << 16) | (first as u64)
                };
                if (next_code as usize) < MAX_TABLE {
                    unsafe { *entries.get_unchecked_mut(next_code as usize) = new_entry };
                    next_code += 1;
                    if next_code == max_code && code_bits < 12 {
                        code_bits += 1;
                        max_code = (1u16 << code_bits) - 1;
                    }
                }
                prev_out_pos = out_pos;
                prev_entry = new_entry;
                prev_code = code;
                out_pos += new_len;
            } else {
                return Err(CodecError::Lzw("out-of-range LZW code"));
            }
        }

        let clear_end = (out_pos + 8).min(out_len);
        out[out_pos..clear_end].fill(0);
        Ok(out_pos)
    }
}

impl Default for LzwDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a sequence of MSB-first TIFF LZW codes at variable bit widths (9..=12).
    fn pack_lzw_codes(codes: &[(u16, u8)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut acc: u64 = 0;
        let mut bits: u32 = 0;
        for &(code, width) in codes {
            acc = (acc << width) | (code as u64);
            bits += width as u32;
            while bits >= 8 {
                bits -= 8;
                out.push(((acc >> bits) & 0xFF) as u8);
            }
        }
        if bits > 0 {
            out.push(((acc << (8 - bits)) & 0xFF) as u8);
        }
        // Pad with 16 trailing bytes so `(bit_pos >> 3) + 8 <= src_len` holds in the 12-bit batch loop.
        out.extend_from_slice(&[0u8; 16]);
        out
    }

    #[test]
    fn lzw_12bit_batch_long_followed_by_short_never_overflows_or_underflows() {
        // Build a valid TIFF LZW stream that reaches 12-bit mode with a long entry (len = 36)
        // and a short entry (len = 2), then emits [long(36), short(2), short(2), short(2)]
        // starting right at `out_pos = out_len - 40` (inside the 4-code 12-bit batch loop).
        // Previously, `c0` advanced `out_pos` to `out_len - 4` and `c1..c3` wrote 8 bytes
        // unaligned past `out_len` (heap OOB write) and could underflow `out_len - out_pos`.
        let mut codes: Vec<(u16, u8)> = Vec::new();
        let mut width: u8 = 9;
        let mut next_code: u16 = 258;
        codes.push((256, width)); // CLEAR

        // First emit a run of 'A' (65) to grow a long dictionary string up to length 36:
        // code 258 = AA (len 2), 259 = AAA (len 3), ..., 292 = A*36 (len 36).
        let mut emitted: usize = 0;
        codes.push((65, width));
        emitted += 1;
        let mut long_code: u16 = 65;
        for _ in 0..35 {
            // Re-emit previous code to grow length by 1, then emit literal 65 to step steadily
            codes.push((long_code, width));
            emitted += long_code.saturating_sub(256).max(1) as usize;
            long_code = next_code;
            next_code += 1;
            if next_code == (1u16 << width) - 1 && width < 12 {
                width += 1;
            }
        }
        // Actually compute exact lengths by decoding into a generous buffer first.
        // Fill table past 2047 so code_bits reaches 12.
        let short_code: u16 = 258; // len 2 ("AA")
        let mut sym: u16 = 0;
        while width < 12 {
            codes.push((sym & 0xFF, width));
            emitted += 1;
            sym = sym.wrapping_add(17);
            next_code += 1;
            if next_code == (1u16 << width) - 1 && width < 12 {
                width += 1;
            }
        }
        // Ensure we are strictly inside 12-bit mode (`width == 12`) with `prev_code != CLEAR`.
        for _ in 0..4 {
            codes.push((66, width));
            emitted += 1;
        }
        // Now measure exact decoded length up to this prefix.
        let mut dec = LzwDecoder::new();
        let prefix_bytes = pack_lzw_codes(&codes);
        let mut probe = vec![0u8; emitted + 4096];
        let mut prefix_with_eoi = codes.clone();
        prefix_with_eoi.push((257, width));
        let prefix_len = dec.decode(&pack_lzw_codes(&prefix_with_eoi), &mut probe).unwrap();
        let _ = prefix_bytes;

        // Append the 4-code 12-bit batch: [long_code (len 36), short_code (2), short_code (2), long_code (36)]
        // Case A: out_len = prefix_len + 40 -> `c0` leaves 4 bytes, `c1` (len 2) leaves 2 bytes,
        // `c2` (len 2) fills to exact `out_len`, `c3` (len 36) hits `avail = 0`.
        // Case B: out_len = prefix_len + 39 -> `c0` leaves 3 bytes, `c1` (len 2) leaves 1 byte,
        // `c2` (len 2) overflows by 1 byte -> must clamp & return Ok(out_len) without OOB write or underflow!
        codes.push((long_code, 12));
        codes.push((short_code, 12));
        codes.push((short_code, 12));
        codes.push((long_code, 12));
        codes.push((257, 12)); // EOI
        let stream = pack_lzw_codes(&codes);

        for extra in [40usize, 39, 38, 37, 41] {
            let mut out = vec![0xCDu8; prefix_len + extra];
            let got = dec.decode(&stream, &mut out).unwrap();
            assert_eq!(got, prefix_len + extra);
        }
    }
}
