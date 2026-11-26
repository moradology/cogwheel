struct PredParams {
    /// Pixels per row (must be a multiple of 4)
    tile_w: u32,
    /// Rows per block
    tile_h: u32,
    /// Number of blocks in batch
    batch_size: u32,
    /// 1 if nodata normalization is enabled, else 0
    has_nodata: u32,
    nodata_val: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: PredParams;
/// Raw decompressed Predictor 2 or Predictor 3 bytes packed as little-endian u32 words.
/// Total words = batch_size * tile_h * tile_w (since 1 f32 = 4 bytes = 1 u32).
@group(0) @binding(1) var<storage, read> raw_words: array<u32>;
/// Reconstructed f32 pixels (batch_size * tile_h * tile_w), directly usable as a GpuSlab!
@group(0) @binding(2) var<storage, read_write> dst_f32: array<f32>;

fn nan_f32() -> f32 {
    return bitcast<f32>(0x7fc00000u);
}

fn sum_bytes_u32(w: u32) -> u32 {
    return (w & 0xffu) + ((w >> 8u) & 0xffu) + ((w >> 16u) & 0xffu) + (w >> 24u);
}

/// Zero-scratch, single-pass, bit-exact TIFF Predictor 3 (Tech Note 3) float32 unshuffle.
/// Each invocation reconstructs one entire row (`tile_w` float32 pixels = `4 * tile_w` bytes).
@compute @workgroup_size(64, 1, 1)
fn unshuffle_pred3_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row_global = gid.x;
    let total_rows = params.batch_size * params.tile_h;
    if (row_global >= total_rows) {
        return;
    }

    let w = params.tile_w;
    let plane_words = w >> 2u; // w / 4 u32 words per byte-plane
    let row_word_base = row_global * w;

    let p0_base = row_word_base;
    let p1_base = row_word_base + plane_words;
    let p2_base = row_word_base + plane_words * 2u;
    let p3_base = row_word_base + plane_words * 3u;

    // Pass 1: compute the 3 byte-plane boundary carries (mod 256)
    var s0: u32 = 0u;
    var s1: u32 = 0u;
    var s2: u32 = 0u;
    for (var k: u32 = 0u; k < plane_words; k = k + 1u) {
        s0 += sum_bytes_u32(raw_words[p0_base + k]);
        s1 += sum_bytes_u32(raw_words[p1_base + k]);
        s2 += sum_bytes_u32(raw_words[p2_base + k]);
    }

    var acc0: u32 = 0u;
    var acc1: u32 = s0 & 0xffu;
    var acc2: u32 = (s0 + s1) & 0xffu;
    var acc3: u32 = (s0 + s1 + s2) & 0xffu;

    let dst_base = row_global * w;
    let check_nd = params.has_nodata != 0u;
    let nd = params.nodata_val;

    // Pass 2: stream 4 bytes from each plane in lockstep and emit 4 reconstructed f32 pixels
    for (var k: u32 = 0u; k < plane_words; k = k + 1u) {
        let w0 = raw_words[p0_base + k];
        let w1 = raw_words[p1_base + k];
        let w2 = raw_words[p2_base + k];
        let w3 = raw_words[p3_base + k];

        let out_k = dst_base + (k << 2u);
        for (var b: u32 = 0u; b < 4u; b = b + 1u) {
            let shift = b * 8u;
            acc0 = (acc0 + ((w0 >> shift) & 0xffu)) & 0xffu;
            acc1 = (acc1 + ((w1 >> shift) & 0xffu)) & 0xffu;
            acc2 = (acc2 + ((w2 >> shift) & 0xffu)) & 0xffu;
            acc3 = (acc3 + ((w3 >> shift) & 0xffu)) & 0xffu;

            let f_bits = (acc0 << 24u) | (acc1 << 16u) | (acc2 << 8u) | acc3;
            var val = bitcast<f32>(f_bits);
            if (check_nd && val == nd) {
                val = nan_f32();
            }
            dst_f32[out_k + b] = val;
        }
    }
}

/// Bit-exact TIFF Predictor 2 (horizontal differencing, little-endian float32) reconstruction.
/// Each invocation reconstructs one entire row (`tile_w` float32 pixels).
@compute @workgroup_size(64, 1, 1)
fn reconstruct_pred2_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row_global = gid.x;
    let total_rows = params.batch_size * params.tile_h;
    if (row_global >= total_rows) {
        return;
    }

    let w = params.tile_w;
    let base = row_global * w;
    let check_nd = params.has_nodata != 0u;
    let nd = params.nodata_val;

    var acc: u32 = 0u;
    for (var x: u32 = 0u; x < w; x = x + 1u) {
        acc = acc + raw_words[base + x];
        var val = bitcast<f32>(acc);
        if (check_nd && val == nd) {
            val = nan_f32();
        }
        dst_f32[base + x] = val;
    }
}
