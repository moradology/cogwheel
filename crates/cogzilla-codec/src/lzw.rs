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

        // Fast inner loop: guaranteed `byte_pos + 8 <= src_len` and `out_pos + 16 <= out_len`
        // so bit reading is completely branchless and inline 8-byte stores need zero bounds checks.
        // Crucially, literals `0..256` and dictionary entries `258..next_code` share the EXACT
        // same `entries[code]` fast path, eliminating the 50/50 unpredictable `code < 256` branch
        // misprediction storm on high-entropy `LZW_P1` blocks!
        while (bit_pos >> 3) + 8 <= src_len && out_pos + 16 <= out_len {
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
                    unsafe {
                        (out_ptr.add(out_pos) as *mut u64).write_unaligned(entry);
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

        Ok(out_pos)
    }
}

impl Default for LzwDecoder {
    fn default() -> Self {
        Self::new()
    }
}
