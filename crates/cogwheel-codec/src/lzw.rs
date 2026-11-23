use crate::CodecError;

const CLEAR_CODE: u16 = 256;
const EOI_CODE: u16 = 257;
const FIRST_CODE: u16 = 258;
const MAX_TABLE: usize = 4096;

/// Reusable thread-local flat-table state for high-throughput MSB-first TIFF LZW decoding.
/// Stores `(prefix, first_byte, last_byte, length)` so strings are written directly backwards
/// into the output buffer without an intermediate stack reversal buffer.
pub struct LzwDecoder {
    prefix: [u16; MAX_TABLE],
    first_byte: [u8; MAX_TABLE],
    last_byte: [u8; MAX_TABLE],
    length: [u16; MAX_TABLE],
}

impl LzwDecoder {
    pub fn new() -> Self {
        let mut d = Self {
            prefix: [0; MAX_TABLE],
            first_byte: [0; MAX_TABLE],
            last_byte: [0; MAX_TABLE],
            length: [1; MAX_TABLE],
        };
        for i in 0..256 {
            d.first_byte[i] = i as u8;
            d.last_byte[i] = i as u8;
            d.length[i] = 1;
        }
        d
    }

    /// Decode a TIFF MSB-first (early-change) LZW stream directly into `out`.
    /// Returns the number of decoded bytes written.
    #[inline(never)]
    pub fn decode(&mut self, src: &[u8], out: &mut [u8]) -> Result<usize, CodecError> {
        let mut bit_buf: u64 = 0;
        let mut bits_in_buf: u32 = 0;
        let mut src_pos: usize = 0;
        let src_len = src.len();

        let mut out_pos: usize = 0;
        let out_len = out.len();

        let mut code_bits: u32 = 9;
        let mut next_code: u16 = FIRST_CODE;
        let mut prev_code: u16 = CLEAR_CODE;

        macro_rules! read_code {
            () => {{
                if bits_in_buf < code_bits {
                    // Refill up to 56 bits when possible using a single unaligned 8-byte BE load
                    if src_pos + 8 <= src_len {
                        let bytes = unsafe { *(src.as_ptr().add(src_pos) as *const [u8; 8]) };
                        let chunk = u64::from_be_bytes(bytes);
                        let consume_bytes = ((64 - bits_in_buf) >> 3) as usize;
                        bit_buf |= chunk >> bits_in_buf;
                        bits_in_buf += (consume_bytes as u32) << 3;
                        src_pos += consume_bytes;
                    } else {
                        while bits_in_buf <= 56 && src_pos < src_len {
                            let b = unsafe { *src.get_unchecked(src_pos) } as u64;
                            src_pos += 1;
                            bit_buf |= b << (56 - bits_in_buf);
                            bits_in_buf += 8;
                        }
                    }
                    if bits_in_buf < code_bits {
                        break;
                    }
                }
                let code = (bit_buf >> (64 - code_bits)) as u16;
                bit_buf <<= code_bits;
                bits_in_buf -= code_bits;
                code
            }};
        }

        loop {
            let code = read_code!();
            if code == EOI_CODE {
                break;
            }
            if code == CLEAR_CODE {
                code_bits = 9;
                next_code = FIRST_CODE;
                prev_code = CLEAR_CODE;
                continue;
            }

            if prev_code == CLEAR_CODE {
                if code >= 256 {
                    return Err(CodecError::Lzw("first code after CLEAR must be literal"));
                }
                if out_pos < out_len {
                    unsafe { *out.get_unchecked_mut(out_pos) = code as u8 };
                    out_pos += 1;
                }
                prev_code = code;
                continue;
            }

            // Fast path: literal byte
            if code < 256 {
                if out_pos < out_len {
                    unsafe { *out.get_unchecked_mut(out_pos) = code as u8 };
                    out_pos += 1;
                }
                if (next_code as usize) < MAX_TABLE {
                    let nc = next_code as usize;
                    let pc = prev_code as usize;
                    self.prefix[nc] = prev_code;
                    self.first_byte[nc] = self.first_byte[pc];
                    self.last_byte[nc] = code as u8;
                    self.length[nc] = self.length[pc] + 1;
                    next_code += 1;
                    // TIFF LZW early-change rule: bump bit width at 511, 1023, 2047
                    if code_bits < 12 && next_code == ((1u16 << code_bits) - 1) {
                        code_bits += 1;
                    }
                }
                prev_code = code;
                continue;
            }

            if code < next_code {
                let c = code as usize;
                let len = self.length[c] as usize;
                let first = self.first_byte[c];

                // Write string directly backwards into `out`
                if out_pos + len <= out_len {
                    let mut cur = code;
                    let mut wp = out_pos + len;
                    while cur >= 256 {
                        wp -= 1;
                        unsafe {
                            *out.get_unchecked_mut(wp) = *self.last_byte.get_unchecked(cur as usize);
                            cur = *self.prefix.get_unchecked(cur as usize);
                        }
                    }
                    wp -= 1;
                    unsafe { *out.get_unchecked_mut(wp) = cur as u8 };
                    out_pos += len;
                } else {
                    // Truncated tail safety (should not happen for valid TIFF tiles)
                    let mut tmp = [0u8; MAX_TABLE];
                    let mut cur = code;
                    let mut wp = len;
                    while cur >= 256 {
                        wp -= 1;
                        tmp[wp] = self.last_byte[cur as usize];
                        cur = self.prefix[cur as usize];
                    }
                    tmp[0] = cur as u8;
                    let avail = out_len.saturating_sub(out_pos);
                    out[out_pos..out_pos + avail].copy_from_slice(&tmp[..avail]);
                    out_pos += avail;
                }

                if (next_code as usize) < MAX_TABLE {
                    let nc = next_code as usize;
                    let pc = prev_code as usize;
                    self.prefix[nc] = prev_code;
                    self.first_byte[nc] = self.first_byte[pc];
                    self.last_byte[nc] = first;
                    self.length[nc] = self.length[pc] + 1;
                    next_code += 1;
                    if code_bits < 12 && next_code == ((1u16 << code_bits) - 1) {
                        code_bits += 1;
                    }
                }
                prev_code = code;
            } else if code == next_code {
                // KwKwK special case
                let pc = prev_code as usize;
                let first = self.first_byte[pc];
                let len = (self.length[pc] as usize) + 1;

                if out_pos + len <= out_len {
                    let mut wp = out_pos + len - 1;
                    unsafe { *out.get_unchecked_mut(wp) = first };
                    let mut cur = prev_code;
                    while cur >= 256 {
                        wp -= 1;
                        unsafe {
                            *out.get_unchecked_mut(wp) = *self.last_byte.get_unchecked(cur as usize);
                            cur = *self.prefix.get_unchecked(cur as usize);
                        }
                    }
                    wp -= 1;
                    unsafe { *out.get_unchecked_mut(wp) = cur as u8 };
                    out_pos += len;
                } else {
                    let mut tmp = [0u8; MAX_TABLE];
                    tmp[len - 1] = first;
                    let mut cur = prev_code;
                    let mut wp = len - 1;
                    while cur >= 256 {
                        wp -= 1;
                        tmp[wp] = self.last_byte[cur as usize];
                        cur = self.prefix[cur as usize];
                    }
                    tmp[0] = cur as u8;
                    let avail = out_len.saturating_sub(out_pos);
                    out[out_pos..out_pos + avail].copy_from_slice(&tmp[..avail]);
                    out_pos += avail;
                }

                if (next_code as usize) < MAX_TABLE {
                    let nc = next_code as usize;
                    self.prefix[nc] = prev_code;
                    self.first_byte[nc] = first;
                    self.last_byte[nc] = first;
                    self.length[nc] = len as u16;
                    next_code += 1;
                    if code_bits < 12 && next_code == ((1u16 << code_bits) - 1) {
                        code_bits += 1;
                    }
                }
                prev_code = code;
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
