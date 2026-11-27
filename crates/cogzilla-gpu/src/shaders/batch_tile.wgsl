struct BatchParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    tile_w: u32,
    tile_h: u32,
    batch_size: u32,
    /// 0 = Bilinear, 1 = Cubic, 2 = Lanczos3
    mode: u32,
    _pad0: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> src_slab: array<f32>;
/// Contiguous per-tile U coordinates: length = batch_size * tile_w
@group(0) @binding(2) var<storage, read> u_batch: array<f32>;
/// Contiguous per-tile V coordinates: length = batch_size * tile_h
@group(0) @binding(3) var<storage, read> v_batch: array<f32>;
/// 256 packed RGBA8 u32 entries
@group(0) @binding(4) var<storage, read> cmap_lut: array<u32>;
/// Contiguous per-tile output words (tile_idx * tile_words + local_word)
@group(0) @binding(5) var<storage, read_write> dst_words: array<u32>;

fn sample_nearest(u: f32, v: f32) -> f32 {
    if (!is_finite_f32(u) || !is_finite_f32(v)
        || u < -0.5 || v < -0.5
        || u > f32(params.src_w) - 0.5
        || v > f32(params.src_h) - 0.5) {
        return nan_f32();
    }
    let xi = clamp(i32(floor(u + 0.5)), 0, i32(params.src_w) - 1);
    let yi = clamp(i32(floor(v + 0.5)), 0, i32(params.src_h) - 1);
    return sample_checked(xi, yi);
}

fn sample_mode(u: f32, v: f32) -> f32 {
    if (params.mode == 3u) {
        return sample_nearest(u, v);
    } else if (params.mode == 0u) {
        return sample_bilinear(u, v);
    } else if (params.mode == 1u) {
        return sample_cubic(u, v);
    } else {
        return sample_lanczos3(u, v);
    }
}

@compute @workgroup_size(16, 16, 1)
fn batch_colormap_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let tile_idx = gid.z;
    if (px >= params.tile_w || py >= params.tile_h || tile_idx >= params.batch_size) {
        return;
    }
    let u = u_batch[tile_idx * params.tile_w + px];
    let v = v_batch[tile_idx * params.tile_h + py];
    let val = sample_mode(u, v);
    let out_idx = (tile_idx * params.tile_h + py) * params.tile_w + px;
    if (!is_finite_f32(val)) {
        dst_words[out_idx] = 0u;
    } else {
        let scaled = clamp((val - params.rescale_min) * params.inv_range_255, 0.0, 255.0);
        dst_words[out_idx] = cmap_lut[u32(scaled)];
    }
}

/// 3D batched IndexedU8 dispatch: gid.z = tile_idx, each invocation packs 4 horizontal pixels into 1 u32.
@compute @workgroup_size(16, 16, 1)
fn batch_colormap_indexed_u8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_x = gid.x;
    let py = gid.y;
    let tile_idx = gid.z;
    let words_per_row = (params.tile_w + 3u) / 4u;
    if (word_x >= words_per_row || py >= params.tile_h || tile_idx >= params.batch_size) {
        return;
    }

    let u_base = tile_idx * params.tile_w;
    let v = v_batch[tile_idx * params.tile_h + py];
    let base_x = word_x * 4u;
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let px = base_x + k;
        if (px < params.tile_w) {
            let val = sample_mode(u_batch[u_base + px], v);
            var idx: u32 = 0u;
            if (is_finite_f32(val)) {
                let scaled = clamp((val - params.rescale_min) * params.inv_range_254, 0.0, 254.0);
                idx = 1u + u32(scaled);
            }
            packed = packed | ((idx & 0xffu) << (k * 8u));
        }
    }
    let tile_words = words_per_row * params.tile_h;
    dst_words[tile_idx * tile_words + py * words_per_row + word_x] = packed;
}
