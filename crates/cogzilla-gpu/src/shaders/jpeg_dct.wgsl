// GPU JPEG 4:2:0 YCbCr + 8x8 2D Forward DCT + Quantization + Zigzag compute shader.
//
// Dispatched with 1 workgroup of `@workgroup_size(8, 8, 1)` per 16x16 MCU:
// `dispatch_workgroups(mcu_w, mcu_h, 1)`.
// Reads packed RGBA8 pixels (`src_rgba_words`), strips transparent pixels onto black,
// converts to level-shifted BT.601 YCbCr (4:2:0 subsampled), runs separable 8x8 2D FDCT
// in workgroup shared memory, quantizes & zigzags, and emits 6 x 32 = 192 packed `u32`
// words per MCU (2 signed `i16` zigzagged coefficients per `u32`).

struct JpegParams {
    width: u32,
    height: u32,
    mcu_w: u32,
    mcu_h: u32,
}

@group(0) @binding(0) var<uniform> params: JpegParams;
@group(0) @binding(1) var<storage, read> src_rgba: array<u32>;
// 0..64: reciprocal luma quant table (1.0 / q_y[i]), 64..128: reciprocal chroma quant table
@group(0) @binding(2) var<storage, read> inv_qtables: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_mcu_words: array<u32>;

// Orthonormal 8x8 DCT-II basis: DCT_MAT[k * 8 + n] = alpha(k) * cos(pi * (2n + 1) * k / 16)
const DCT_MAT = array<f32, 64>(
     0.35355338,  0.35355338,  0.35355338,  0.35355338,  0.35355338,  0.35355338,  0.35355338,  0.35355338,
     0.49039263,  0.41573480,  0.27778512,  0.09754516, -0.09754516, -0.27778512, -0.41573480, -0.49039263,
     0.46193975,  0.19134171, -0.19134171, -0.46193975, -0.46193975, -0.19134171,  0.19134171,  0.46193975,
     0.41573480, -0.09754516, -0.49039263, -0.27778512,  0.27778512,  0.49039263,  0.09754516, -0.41573480,
     0.35355338, -0.35355338, -0.35355338,  0.35355338,  0.35355338, -0.35355338, -0.35355338,  0.35355338,
     0.27778512, -0.49039263,  0.09754516,  0.41573480, -0.41573480, -0.09754516,  0.49039263, -0.27778512,
     0.19134171, -0.46193975,  0.46193975, -0.19134171, -0.19134171,  0.46193975, -0.46193975,  0.19134171,
     0.09754516, -0.27778512,  0.41573480, -0.49039263,  0.49039263, -0.41573480,  0.27778512, -0.09754516,
);

// Maps natural raster index `ly * 8 + lx` -> zigzag index `0..64`
const NAT_TO_ZZ = array<u32, 64>(
     0u,  1u,  5u,  6u, 14u, 15u, 27u, 28u,
     2u,  4u,  7u, 13u, 16u, 26u, 29u, 42u,
     3u,  8u, 12u, 17u, 25u, 30u, 41u, 43u,
     9u, 11u, 18u, 24u, 31u, 40u, 44u, 53u,
    10u, 19u, 23u, 32u, 39u, 45u, 52u, 54u,
    20u, 22u, 33u, 38u, 46u, 51u, 55u, 60u,
    21u, 34u, 37u, 47u, 50u, 56u, 59u, 61u,
    35u, 36u, 48u, 49u, 57u, 58u, 62u, 63u,
);

var<workgroup> sh_blk: array<f32, 64>;
var<workgroup> sh_tmp: array<f32, 64>;
var<workgroup> sh_zz: array<i32, 64>;
var<workgroup> sh_cb: array<f32, 64>;
var<workgroup> sh_cr: array<f32, 64>;

fn load_rgb_clamped(px: u32, py: u32) -> vec3<f32> {
    let cx = min(px, params.width - 1u);
    let cy = min(py, params.height - 1u);
    let w = src_rgba[cy * params.width + cx];
    let a = (w >> 24u) & 0xffu;
    if (a == 0u) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return vec3<f32>(
        f32(w & 0xffu),
        f32((w >> 8u) & 0xffu),
        f32((w >> 16u) & 0xffu),
    );
}

fn run_dct_quant_store(
    lx: u32,
    ly: u32,
    tid: u32,
    q_offset: u32,
    out_word_base: u32,
) {
    workgroupBarrier();
    // Row 1D DCT
    var sum_r: f32 = 0.0;
    let row_base = ly * 8u;
    let mat_x_base = lx * 8u;
    for (var n: u32 = 0u; n < 8u; n = n + 1u) {
        sum_r = sum_r + sh_blk[row_base + n] * DCT_MAT[mat_x_base + n];
    }
    sh_tmp[tid] = sum_r;
    workgroupBarrier();

    // Column 1D DCT
    var sum_c: f32 = 0.0;
    let mat_y_base = ly * 8u;
    for (var m: u32 = 0u; m < 8u; m = m + 1u) {
        sum_c = sum_c + sh_tmp[m * 8u + lx] * DCT_MAT[mat_y_base + m];
    }

    let q_val = clamp(i32(round(sum_c * inv_qtables[q_offset + tid])), -2047, 2047);
    sh_zz[NAT_TO_ZZ[tid]] = q_val;
    workgroupBarrier();

    if (tid < 32u) {
        let c0 = u32(sh_zz[tid * 2u] & 0xffff);
        let c1 = u32(sh_zz[tid * 2u + 1u] & 0xffff);
        out_mcu_words[out_word_base + tid] = c0 | (c1 << 16u);
    }
    workgroupBarrier();
}

@compute @workgroup_size(8, 8, 1)
fn jpeg_mcu_420_fdct(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let mx = wgid.x;
    let my = wgid.y;
    if (mx >= params.mcu_w || my >= params.mcu_h) {
        return;
    }
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 8u + lx;

    let base_x = mx * 16u;
    let base_y = my * 16u;
    let mcu_out_base = (my * params.mcu_w + mx) * 192u;

    // Precompute 4:2:0 averaged Cb & Cr for this thread's 2x2 quad in the 16x16 MCU
    let qx = base_x + lx * 2u;
    let qy = base_y + ly * 2u;
    let p00 = load_rgb_clamped(qx, qy);
    let p10 = load_rgb_clamped(qx + 1u, qy);
    let p01 = load_rgb_clamped(qx, qy + 1u);
    let p11 = load_rgb_clamped(qx + 1u, qy + 1u);
    let avg = (p00 + p10 + p01 + p11) * 0.25;
    sh_cb[tid] = -0.168736 * avg.x - 0.331264 * avg.y + 0.5 * avg.z;
    sh_cr[tid] =  0.5 * avg.x - 0.418688 * avg.y - 0.081312 * avg.z;

    // 4 Luma (Y) 8x8 blocks: (0,0), (8,0), (0,8), (8,8)
    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        let bx = (b & 1u) * 8u;
        let by = (b >> 1u) * 8u;
        let rgb = load_rgb_clamped(base_x + bx + lx, base_y + by + ly);
        sh_blk[tid] = 0.299 * rgb.x + 0.587 * rgb.y + 0.114 * rgb.z - 128.0;
        run_dct_quant_store(lx, ly, tid, 0u, mcu_out_base + b * 32u);
    }

    // Cb 8x8 block (q_offset = 64)
    sh_blk[tid] = sh_cb[tid];
    run_dct_quant_store(lx, ly, tid, 64u, mcu_out_base + 4u * 32u);

    // Cr 8x8 block (q_offset = 64)
    sh_blk[tid] = sh_cr[tid];
    run_dct_quant_store(lx, ly, tid, 64u, mcu_out_base + 5u * 32u);
}
