//! GPU-accelerated Baseline JPEG (4:2:0 JFIF) encoder (`GpuJpegEncoder`).
//!
//! Offloads RGBA8 -> alpha-strip -> BT.601 YCbCr 4:2:0 subsampling -> separable 8×8 2D Forward
//! DCT -> quantization -> zigzag reordering onto a single WGSL compute pass (`jpeg_dct.wgsl`),
//! then runs a zero-allocation table-driven Annex K Huffman bitstream writer on CPU.

use std::sync::{Arc, OnceLock, mpsc};

use bytemuck::{Pod, Zeroable};

use crate::context::{GpuContext, GpuError};

// Standard JPEG Annex K quantization tables (natural raster order)
const STD_LUMA_Q: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];

const STD_CHROMA_Q: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

// Zigzag order: ZZ_TO_NAT[zz] = natural raster index
const ZZ_TO_NAT: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

// Standard JPEG Annex K Huffman specifications
const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_LUMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_CHROMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHROMA_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

#[derive(Clone)]
struct HuffTable {
    code: [u16; 256],
    len: [u8; 256],
}

impl HuffTable {
    fn build(bits: &[u8; 16], vals: &[u8]) -> Self {
        let mut code = [0u16; 256];
        let mut len = [0u8; 256];
        let mut cur_code: u16 = 0;
        let mut idx = 0usize;
        for (bit_len, &count) in bits.iter().enumerate() {
            let bl = (bit_len + 1) as u8;
            for _ in 0..count {
                let sym = vals[idx] as usize;
                code[sym] = cur_code;
                len[sym] = bl;
                cur_code = cur_code.wrapping_add(1);
                idx += 1;
            }
            cur_code <<= 1;
        }
        Self { code, len }
    }
}

struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    bit_buf: u64,
    bit_cnt: u32,
}

impl<'a> BitWriter<'a> {
    #[inline]
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    #[inline(always)]
    fn write_bits(&mut self, bits: u32, len: u32) {
        self.bit_buf = (self.bit_buf << len) | (bits as u64);
        self.bit_cnt += len;
        if self.bit_cnt >= 32 {
            self.bit_cnt -= 32;
            let word = (self.bit_buf >> self.bit_cnt) as u32;
            let inv = !word;
            let has_ff = (inv.wrapping_sub(0x0101_0101) & word & 0x8080_8080) != 0;
            let bytes = word.to_be_bytes();
            if !has_ff {
                self.out.extend_from_slice(&bytes);
            } else {
                for b in bytes {
                    self.out.push(b);
                    if b == 0xff {
                        self.out.push(0x00);
                    }
                }
            }
        }
    }

    #[inline]
    fn flush_align(&mut self) {
        while self.bit_cnt >= 8 {
            self.bit_cnt -= 8;
            let byte = ((self.bit_buf >> self.bit_cnt) & 0xff) as u8;
            self.out.push(byte);
            if byte == 0xff {
                self.out.push(0x00);
            }
        }
        if self.bit_cnt > 0 {
            let pad = 8 - self.bit_cnt;
            let byte = (((self.bit_buf << pad) | ((1u64 << pad) - 1)) & 0xff) as u8;
            self.bit_cnt = 0;
            self.out.push(byte);
            if byte == 0xff {
                self.out.push(0x00);
            }
        }
    }
}

#[inline(always)]
fn bit_size_and_code(val: i32) -> (u32, u32) {
    if val == 0 {
        return (0, 0);
    }
    let abs_v = val.unsigned_abs();
    let nbits = 32 - abs_v.leading_zeros();
    let code = if val > 0 {
        abs_v
    } else {
        (val + ((1i32 << nbits) - 1)) as u32
    };
    (nbits, code)
}

#[inline(always)]
fn encode_block(
    bw: &mut BitWriter<'_>,
    zz: &[i16; 64],
    prev_dc: &mut i32,
    dc_tbl: &HuffTable,
    ac_tbl: &HuffTable,
) {
    let dc = zz[0] as i32;
    let diff = dc - *prev_dc;
    *prev_dc = dc;
    let (dc_nbits, dc_code) = bit_size_and_code(diff);
    let sym = dc_nbits as usize;
    let h_len = dc_tbl.len[sym] as u32;
    let h_code = dc_tbl.code[sym] as u32;
    bw.write_bits((h_code << dc_nbits) | dc_code, h_len + dc_nbits);

    // Fast backward scan in 8×i16 (128-bit) chunks to skip trailing AC zeros.
    let mut last_chunk = 7usize;
    while last_chunk > 0 && zz[last_chunk * 8..(last_chunk + 1) * 8] == [0i16; 8] {
        last_chunk -= 1;
    }
    let mut last_nz = last_chunk * 8 + 7;
    while last_nz > 0 && zz[last_nz] == 0 {
        last_nz -= 1;
    }
    if last_nz == 0 {
        bw.write_bits(ac_tbl.code[0x00] as u32, ac_tbl.len[0x00] as u32);
        return;
    }

    let mut run = 0usize;
    for &c in &zz[1..=last_nz] {
        if c == 0 {
            run += 1;
            continue;
        }
        while run >= 16 {
            bw.write_bits(ac_tbl.code[0xf0] as u32, ac_tbl.len[0xf0] as u32);
            run -= 16;
        }
        let (nbits, code) = bit_size_and_code(c as i32);
        let rs = (run << 4) | (nbits as usize);
        let ac_len = ac_tbl.len[rs] as u32;
        let ac_code = ac_tbl.code[rs] as u32;
        bw.write_bits((ac_code << nbits) | code, ac_len + nbits);
        run = 0;
    }
    if last_nz < 63 {
        bw.write_bits(ac_tbl.code[0x00] as u32, ac_tbl.len[0x00] as u32);
    }
}

pub(crate) fn build_inv_qtables(quality: u8) -> ([u8; 64], [u8; 64], [f32; 128]) {
    let q_y = scale_qtable(&STD_LUMA_Q, quality);
    let q_c = scale_qtable(&STD_CHROMA_Q, quality);
    let mut inv_q = [0.0f32; 128];
    for i in 0..64 {
        inv_q[i] = 1.0 / (q_y[i] as f32);
        inv_q[64 + i] = 1.0 / (q_c[i] as f32);
    }
    (q_y, q_c, inv_q)
}

struct StdHuffTables {
    dc_luma: HuffTable,
    dc_chroma: HuffTable,
    ac_luma: HuffTable,
    ac_chroma: HuffTable,
}

fn std_huff_tables() -> &'static StdHuffTables {
    static TABLES: OnceLock<StdHuffTables> = OnceLock::new();
    TABLES.get_or_init(|| StdHuffTables {
        dc_luma: HuffTable::build(&DC_LUMA_BITS, &DC_LUMA_VALS),
        dc_chroma: HuffTable::build(&DC_CHROMA_BITS, &DC_CHROMA_VALS),
        ac_luma: HuffTable::build(&AC_LUMA_BITS, &AC_LUMA_VALS),
        ac_chroma: HuffTable::build(&AC_CHROMA_BITS, &AC_CHROMA_VALS),
    })
}

/// Huffman-encode quantized zigzagged `i16` MCU coefficients produced by `jpeg_dct.wgsl`
/// (e.g. from `CoalescedJob::WithJpeg`) into a complete Baseline JFIF `.jpg` byte vector.
///
/// Lock-free and uses SWAR 32-bit `BitWriter` + 128-bit zero-tail scan + multi-threaded
/// `DRI`/`RST0..RST7` encoding for large images (`>= 8192` MCUs).
pub fn huffman_encode_coeffs(coeffs_bytes: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    let mcu_w = width.div_ceil(16) as usize;
    let mcu_h = height.div_ceil(16) as usize;
    let total_mcus = mcu_w * mcu_h;
    let coeffs: &[i16] = bytemuck::cast_slice(coeffs_bytes);
    let q_y = scale_qtable(&STD_LUMA_Q, quality);
    let q_c = scale_qtable(&STD_CHROMA_Q, quality);
    let tbls = std_huff_tables();

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let use_parallel = total_mcus >= 8192 && n_threads >= 2;
    let restart_interval: u16 = if use_parallel {
        total_mcus.div_ceil(n_threads).clamp(512, 65535) as u16
    } else {
        0
    };

    let mut jpg = Vec::with_capacity((width as usize * height as usize) / 4 + 1024);
    write_jfif_headers(
        &mut jpg,
        width as u16,
        height as u16,
        &q_y,
        &q_c,
        restart_interval,
    );

    let dc_luma = &tbls.dc_luma;
    let ac_luma = &tbls.ac_luma;
    let dc_chroma = &tbls.dc_chroma;
    let ac_chroma = &tbls.ac_chroma;

    if restart_interval > 0 {
        let ri = restart_interval as usize;
        let n_chunks = total_mcus.div_ceil(ri);
        let mut chunks: Vec<Vec<u8>> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_chunks);
            for chunk_idx in 0..n_chunks {
                let start_mcu = chunk_idx * ri;
                let end_mcu = (start_mcu + ri).min(total_mcus);
                let chunk_coeffs = &coeffs[start_mcu * 384..end_mcu * 384];
                let is_last = chunk_idx + 1 == n_chunks;
                handles.push(s.spawn(move || {
                    let mcu_count = end_mcu - start_mcu;
                    let mut buf = Vec::with_capacity(mcu_count * 32 + 64);
                    encode_mcu_slice(
                        &mut buf,
                        chunk_coeffs,
                        mcu_count,
                        dc_luma,
                        ac_luma,
                        dc_chroma,
                        ac_chroma,
                    );
                    if !is_last {
                        buf.push(0xff);
                        buf.push(0xd0 + ((chunk_idx & 7) as u8));
                    }
                    buf
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("jpeg huffman worker"))
                .collect()
        });
        for c in chunks.drain(..) {
            jpg.extend_from_slice(&c);
        }
    } else {
        encode_mcu_slice(
            &mut jpg,
            coeffs,
            total_mcus,
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        );
    }
    jpg.extend_from_slice(&[0xff, 0xd9]); // EOI
    jpg
}

fn scale_qtable(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [1u8; 64];
    for i in 0..64 {
        let v = ((base[i] as u32 * scale + 50) / 100).clamp(1, 255);
        out[i] = v as u8;
    }
    out
}

fn write_jfif_headers(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    q_y: &[u8; 64],
    q_c: &[u8; 64],
    restart_interval: u16,
) {
    // SOI
    out.extend_from_slice(&[0xff, 0xd8]);
    // APP0 JFIF
    out.extend_from_slice(&[
        0xff, 0xe0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00,
        0x01, 0x00, 0x00,
    ]);
    // DQT (Luma table 0, zigzag order)
    out.extend_from_slice(&[0xff, 0xdb, 0x00, 0x43, 0x00]);
    for &nat in &ZZ_TO_NAT {
        out.push(q_y[nat]);
    }
    // DQT (Chroma table 1, zigzag order)
    out.extend_from_slice(&[0xff, 0xdb, 0x00, 0x43, 0x01]);
    for &nat in &ZZ_TO_NAT {
        out.push(q_c[nat]);
    }
    // SOF0 (Baseline DCT, 8-bit, 3 components, 4:2:0)
    out.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        0x11,
        8,
        (height >> 8) as u8,
        (height & 0xff) as u8,
        (width >> 8) as u8,
        (width & 0xff) as u8,
        3,
        1,
        0x22,
        0, // Y: 2x2 sampling, Q0
        2,
        0x11,
        1, // Cb: 1x1 sampling, Q1
        3,
        0x11,
        1, // Cr: 1x1 sampling, Q1
    ]);
    // DHT tables
    let write_dht = |out: &mut Vec<u8>, tc_th: u8, bits: &[u8; 16], vals: &[u8]| {
        let len = (2 + 1 + 16 + vals.len()) as u16;
        out.extend_from_slice(&[0xff, 0xc4, (len >> 8) as u8, (len & 0xff) as u8, tc_th]);
        out.extend_from_slice(bits);
        out.extend_from_slice(vals);
    };
    write_dht(out, 0x00, &DC_LUMA_BITS, &DC_LUMA_VALS);
    write_dht(out, 0x10, &AC_LUMA_BITS, &AC_LUMA_VALS);
    write_dht(out, 0x01, &DC_CHROMA_BITS, &DC_CHROMA_VALS);
    write_dht(out, 0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALS);
    // Optional DRI (Define Restart Interval) for multi-threaded RST0..RST7 encoding
    if restart_interval > 0 {
        out.extend_from_slice(&[
            0xff,
            0xdd,
            0x00,
            0x04,
            (restart_interval >> 8) as u8,
            (restart_interval & 0xff) as u8,
        ]);
    }
    // SOS
    out.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x11, 3, 0x11, 0x00, 0x3f, 0x00,
    ]);
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct JpegParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mcu_w: u32,
    pub(crate) mcu_h: u32,
}

/// GPU-accelerated Baseline JFIF (4:2:0) encoder.
pub struct GpuJpegEncoder {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    inv_q_buf: wgpu::Buffer,
    src_buf: wgpu::Buffer,
    src_cap_bytes: usize,
    out_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    out_cap_words: usize,
}

impl GpuJpegEncoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-jpeg-dct-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/jpeg_dct.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-jpeg-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-jpeg-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cogzilla-jpeg-fdct"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("jpeg_mcu_420_fdct"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-jpeg-params"),
            size: std::mem::size_of::<JpegParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let inv_q_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-jpeg-inv-q"),
            size: (128 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let src_cap_bytes = 1024 * 1024 * 4;
        let src_buf = Self::alloc_src(&device, src_cap_bytes);
        let out_cap_words = 64 * 64 * 192;
        let (out_buf, staging_buf) = Self::alloc_out(&device, out_cap_words);

        Self {
            device,
            queue,
            bgl,
            pipeline,
            params_buf,
            inv_q_buf,
            src_buf,
            src_cap_bytes,
            out_buf,
            staging_buf,
            out_cap_words,
        }
    }

    fn alloc_src(device: &wgpu::Device, bytes: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-jpeg-src"),
            size: bytes.max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn alloc_out(device: &wgpu::Device, words: usize) -> (wgpu::Buffer, wgpu::Buffer) {
        let bytes = (words.max(1) * 4) as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-jpeg-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-jpeg-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (out, staging)
    }

    /// Upload an RGBA8 image to the GPU, run WGSL 4:2:0 YCbCr + 8×8 FDCT + Quantize + Zigzag,
    /// and emit a standard Baseline JFIF `.jpg` (using multi-threaded `DRI`/`RST0..RST7`
    /// Huffman encoding for images with `>= 1024` MCUs).
    pub fn encode_rgba8(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        quality: u8,
    ) -> Result<Vec<u8>, GpuError> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| GpuError::InvalidDimensions("JPEG dims overflow".into()))?;
        if width == 0 || height == 0 || rgba.len() < expected || width > 65535 || height > 65535 {
            return Err(GpuError::InvalidDimensions(
                "invalid RGBA8 buffer for GPU JPEG".into(),
            ));
        }
        let max_binding = self.device.limits().max_storage_buffer_binding_size as usize;
        if expected > max_binding {
            return Err(GpuError::InvalidDimensions(format!(
                "JPEG RGBA8 input ({expected} bytes) exceeds adapter max_storage_buffer_binding_size ({max_binding} bytes)"
            )));
        }
        if expected > self.src_cap_bytes {
            self.src_cap_bytes = expected.next_power_of_two().min(max_binding);
            self.src_buf = Self::alloc_src(&self.device, self.src_cap_bytes);
        }
        self.queue.write_buffer(&self.src_buf, 0, &rgba[..expected]);
        let src_buf = self.src_buf.clone();
        self.encode_rgba8_from_buffer_inner(&src_buf, width, height, quality)
    }

    /// Encode an RGBA8 image already resident in a GPU `STORAGE` buffer (zero CPU→GPU upload).
    pub fn encode_rgba8_from_buffer(
        &mut self,
        src_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
        quality: u8,
    ) -> Result<Vec<u8>, GpuError> {
        if width == 0 || height == 0 || width > 65535 || height > 65535 {
            return Err(GpuError::InvalidDimensions(
                "invalid RGBA8 dimensions for GPU JPEG".into(),
            ));
        }
        self.encode_rgba8_from_buffer_inner(src_buf, width, height, quality)
    }

    /// Record the GPU 4:2:0 YCbCr + 8×8 FDCT + Quantize + Zigzag compute pass and staging copy
    /// directly into an existing `CommandEncoder` reading from a GPU-resident RGBA8 `STORAGE`
    /// buffer. Returns the staging byte count to pass to [`Self::readback_jpeg`] after submitting
    /// the encoder.
    pub fn record_encode_rgba8_from_buffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        src_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
        quality: u8,
    ) -> Result<u64, GpuError> {
        if width == 0 || height == 0 || width > 65535 || height > 65535 {
            return Err(GpuError::InvalidDimensions(
                "invalid RGBA8 dimensions for GPU JPEG".into(),
            ));
        }
        let mcu_w = width.div_ceil(16);
        let mcu_h = height.div_ceil(16);
        let total_mcus = (mcu_w as usize) * (mcu_h as usize);
        let total_words = total_mcus * 192;
        if total_words > self.out_cap_words {
            self.out_cap_words = total_words.next_power_of_two();
            let (o, s) = Self::alloc_out(&self.device, self.out_cap_words);
            self.out_buf = o;
            self.staging_buf = s;
        }

        let (_q_y, _q_c, inv_q) = build_inv_qtables(quality);
        let params = JpegParams {
            width,
            height,
            mcu_w,
            mcu_h,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.inv_q_buf, 0, bytemuck::cast_slice(&inv_q));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-jpeg-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.inv_q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });

        let out_bytes = (total_words * 4) as u64;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-jpeg-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(mcu_w, mcu_h, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging_buf, 0, out_bytes);
        Ok(out_bytes)
    }

    /// Map the JPEG staging buffer after submitting an encoder recorded with
    /// [`Self::record_encode_rgba8_from_buffer`] and emit the Baseline JFIF `.jpg` bytes.
    pub fn readback_jpeg(
        &self,
        out_bytes: u64,
        width: u32,
        height: u32,
        quality: u8,
    ) -> Result<Vec<u8>, GpuError> {
        let slice = self.staging_buf.slice(..out_bytes);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::BufferMap(e.to_string()))?
            .map_err(|e| GpuError::BufferMap(e.to_string()))?;

        let mapped = slice.get_mapped_range();
        let jpg = huffman_encode_coeffs(&mapped, width, height, quality);
        drop(mapped);
        self.staging_buf.unmap();
        Ok(jpg)
    }

    fn encode_rgba8_from_buffer_inner(
        &mut self,
        src_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
        quality: u8,
    ) -> Result<Vec<u8>, GpuError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-jpeg-enc"),
            });
        let out_bytes =
            self.record_encode_rgba8_from_buffer(&mut encoder, src_buf, width, height, quality)?;
        self.queue.submit(Some(encoder.finish()));
        self.readback_jpeg(out_bytes, width, height, quality)
    }
}

fn encode_mcu_slice(
    out: &mut Vec<u8>,
    coeffs: &[i16],
    mcu_count: usize,
    dc_luma: &HuffTable,
    ac_luma: &HuffTable,
    dc_chroma: &HuffTable,
    ac_chroma: &HuffTable,
) {
    let mut bw = BitWriter::new(out);
    let mut prev_y = 0i32;
    let mut prev_cb = 0i32;
    let mut prev_cr = 0i32;
    for mcu in 0..mcu_count {
        let base = mcu * 384;
        for b in 0..4 {
            let blk: &[i16; 64] = coeffs[base + b * 64..base + (b + 1) * 64]
                .try_into()
                .unwrap();
            encode_block(&mut bw, blk, &mut prev_y, dc_luma, ac_luma);
        }
        let cb_blk: &[i16; 64] = coeffs[base + 4 * 64..base + 5 * 64].try_into().unwrap();
        encode_block(&mut bw, cb_blk, &mut prev_cb, dc_chroma, ac_chroma);
        let cr_blk: &[i16; 64] = coeffs[base + 5 * 64..base + 6 * 64].try_into().unwrap();
        encode_block(&mut bw, cr_blk, &mut prev_cr, dc_chroma, ac_chroma);
    }
    bw.flush_align();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuOptions;

    #[test]
    fn gpu_jpeg_encodes_valid_jfif_with_high_psnr() {
        let Ok(ctx) = GpuContext::new(GpuOptions::default()) else {
            eprintln!("skipping gpu_jpeg test (no GPU adapter)");
            return;
        };
        let mut enc = GpuJpegEncoder::new(&ctx);
        for &(w, h, q) in &[
            (256u32, 256u32, 85u8),
            (333, 217, 90),
            (512, 512, 85),
            (2048, 1024, 85),
        ] {
            let mut rgba = Vec::with_capacity((w * h * 4) as usize);
            for y in 0..h {
                for x in 0..w {
                    let r = (128.0 + 100.0 * (x as f32 * 0.04).sin()) as u8;
                    let g = (128.0 + 90.0 * (y as f32 * 0.05).cos()) as u8;
                    let b = (128.0 + 80.0 * ((x + y) as f32 * 0.03).sin()) as u8;
                    let a = if x < 16 && y < 16 { 0 } else { 255 };
                    rgba.extend_from_slice(&[r, g, b, a]);
                }
            }
            let jpg = enc.encode_rgba8(&rgba, w, h, q).expect("gpu jpeg encode");
            assert!(jpg.starts_with(&[0xff, 0xd8, 0xff, 0xe0]), "missing JFIF SOI/APP0");
            assert!(jpg.ends_with(&[0xff, 0xd9]), "missing EOI");

            // Decode with zune-jpeg and verify PSNR against alpha-stripped RGB.
            let mut dec = zune_jpeg::JpegDecoder::new(&jpg);
            let decoded = dec.decode().expect("zune-jpeg decode of GPU JFIF");
            let info = dec.info().expect("jpeg info");
            assert_eq!((info.width as u32, info.height as u32), (w, h));
            assert_eq!(decoded.len(), (w * h * 3) as usize);

            let mut mse = 0.0f64;
            for i in 0..(w * h) as usize {
                let a = rgba[i * 4 + 3];
                let (er, eg, eb) = if a == 0 {
                    (0.0, 0.0, 0.0)
                } else {
                    (rgba[i * 4] as f64, rgba[i * 4 + 1] as f64, rgba[i * 4 + 2] as f64)
                };
                let dr = decoded[i * 3] as f64 - er;
                let dg = decoded[i * 3 + 1] as f64 - eg;
                let db = decoded[i * 3 + 2] as f64 - eb;
                mse += dr * dr + dg * dg + db * db;
            }
            mse /= w as f64 * h as f64 * 3.0;
            let psnr = 10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10();
            eprintln!("GPU JPEG {w}x{h} Q={q}: {} bytes, PSNR={psnr:.2} dB", jpg.len());
            assert!(psnr > 35.0, "PSNR too low for {w}x{h} Q={q}: {psnr:.2} dB");
        }
    }
}
