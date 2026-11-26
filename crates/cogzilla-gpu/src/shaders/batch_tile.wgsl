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

const PI: f32 = 3.14159265358979323846;

fn nan_f32() -> f32 {
    return bitcast<f32>(0x7fc00000u);
}

fn is_nan_f32(v: f32) -> bool {
    let bits = bitcast<u32>(v);
    return ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) != 0u);
}

fn is_finite_f32(v: f32) -> bool {
    let bits = bitcast<u32>(v);
    return (bits & 0x7f800000u) != 0x7f800000u;
}

fn sample_checked(x: i32, y: i32) -> f32 {
    if (x < 0 || y < 0 || x >= i32(params.src_w) || y >= i32(params.src_h)) {
        return nan_f32();
    }
    return src_slab[u32(y) * params.src_stride + u32(x)];
}

fn sample_bilinear_raw(u: f32, v: f32) -> f32 {
    let x0_f = floor(u);
    let y0_f = floor(v);
    let x0 = i32(x0_f);
    let y0 = i32(y0_f);
    let fx = u - x0_f;
    let fy = v - y0_f;

    let v00 = sample_checked(x0, y0);
    let v10 = sample_checked(x0 + 1, y0);
    let v01 = sample_checked(x0, y0 + 1);
    let v11 = sample_checked(x0 + 1, y0 + 1);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    var sum = 0.0;
    var wsum = 0.0;
    if (!is_nan_f32(v00)) { sum += v00 * w00; wsum += w00; }
    if (!is_nan_f32(v10)) { sum += v10 * w10; wsum += w10; }
    if (!is_nan_f32(v01)) { sum += v01 * w01; wsum += w01; }
    if (!is_nan_f32(v11)) { sum += v11 * w11; wsum += w11; }

    if (wsum > 0.0) {
        return sum / wsum;
    }
    return nan_f32();
}

fn cubic_weights(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        (-0.5 * t3) + t2 - (0.5 * t),
        (1.5 * t3) - (2.5 * t2) + 1.0,
        (-1.5 * t3) + (2.0 * t2) + (0.5 * t),
        (0.5 * t3) - (0.5 * t2),
    );
}

fn sample_cubic(u: f32, v: f32) -> f32 {
    if (!is_finite_f32(u) || !is_finite_f32(v)
        || u < -2.0 || v < -2.0
        || u > f32(params.src_w) + 1.0
        || v > f32(params.src_h) + 1.0) {
        return nan_f32();
    }
    let x0_f = floor(u);
    let y0_f = floor(v);
    let x0 = i32(x0_f);
    let y0 = i32(y0_f);
    let wx = cubic_weights(u - x0_f);
    let wy = cubic_weights(v - y0_f);

    var acc = 0.0;
    for (var j: i32 = 0; j < 4; j = j + 1) {
        let ry = y0 + j - 1;
        var row_acc = 0.0;
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let s = sample_checked(x0 + i - 1, ry);
            if (is_nan_f32(s)) {
                return sample_bilinear_raw(u, v);
            }
            row_acc += s * wx[i];
        }
        acc += row_acc * wy[j];
    }
    return acc;
}

fn lanczos3_weight(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1e-6) {
        return 1.0;
    }
    if (ax >= 3.0) {
        return 0.0;
    }
    let px = PI * ax;
    let px3 = px * (1.0 / 3.0);
    return (sin(px) / px) * (sin(px3) / px3);
}

fn sample_lanczos3(u: f32, v: f32) -> f32 {
    if (!is_finite_f32(u) || !is_finite_f32(v)
        || u < -0.5 || v < -0.5
        || u > f32(params.src_w) - 0.5
        || v > f32(params.src_h) - 0.5) {
        return nan_f32();
    }
    let x0_f = floor(u);
    let y0_f = floor(v);
    let x0 = i32(x0_f);
    let y0 = i32(y0_f);
    let fx = u - x0_f;
    let fy = v - y0_f;

    var wx: array<f32, 6>;
    var wy: array<f32, 6>;
    for (var k: i32 = 0; k < 6; k = k + 1) {
        let d = f32(k - 2);
        wx[k] = lanczos3_weight(d - fx);
        wy[k] = lanczos3_weight(d - fy);
    }

    var sum = 0.0;
    var wsum = 0.0;
    for (var j: i32 = 0; j < 6; j = j + 1) {
        let ry = y0 + j - 2;
        let wyj = wy[j];
        for (var i: i32 = 0; i < 6; i = i + 1) {
            let s = sample_checked(x0 + i - 2, ry);
            if (!is_nan_f32(s)) {
                let w = wx[i] * wyj;
                sum += s * w;
                wsum += w;
            }
        }
    }

    if (wsum >= 0.5) {
        return sum / wsum;
    }
    return sample_bilinear_raw(u, v);
}

fn sample_mode(u: f32, v: f32) -> f32 {
    if (params.mode == 0u) {
        if (!is_finite_f32(u) || !is_finite_f32(v)
            || u < -0.5 || v < -0.5
            || u > f32(params.src_w) - 0.5
            || v > f32(params.src_h) - 0.5) {
            return nan_f32();
        }
        return sample_bilinear_raw(u, v);
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
