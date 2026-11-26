struct ColormapParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    /// 0 = Bilinear, 1 = Cubic, 2 = Lanczos3
    mode: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    /// 0 = Separable1D, 1 = Grid2D (17x17 control grid), 2 = Exact2D (dst_w * dst_h)
    coord_mode: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ColormapParams;
@group(0) @binding(1) var<storage, read> src_slab: array<f32>;
@group(0) @binding(2) var<storage, read> u_coords: array<f32>;
@group(0) @binding(3) var<storage, read> v_coords: array<f32>;
/// 256 packed RGBA8 u32 entries: r | (g << 8) | (b << 16) | (a << 24)
@group(0) @binding(4) var<storage, read> cmap_lut: array<u32>;
/// Output words:
/// - rgba8 entry point: 1 u32 per pixel (packed RGBA8)
/// - indexed_u8 entry point: 1 u32 per 4 horizontal pixels (packed little-endian 4 x u8)
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

fn sample_bilinear(u: f32, v: f32) -> f32 {
    if (!is_finite_f32(u) || !is_finite_f32(v)
        || u < -0.5 || v < -0.5
        || u > f32(params.src_w) - 0.5
        || v > f32(params.src_h) - 0.5) {
        return nan_f32();
    }
    return sample_bilinear_raw(u, v);
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
        return sample_bilinear(u, v);
    } else if (params.mode == 1u) {
        return sample_cubic(u, v);
    } else if (params.mode == 3u) {
        if (!is_finite_f32(u) || !is_finite_f32(v)) {
            return nan_f32();
        }
        return sample_checked(i32(floor(u + 0.5)), i32(floor(v + 0.5)));
    } else {
        return sample_lanczos3(u, v);
    }
}

fn value_to_rgba8(val: f32) -> u32 {
    if (!is_finite_f32(val)) {
        return 0u;
    }
    let scaled = clamp((val - params.rescale_min) * params.inv_range_255, 0.0, 255.0);
    let idx = u32(scaled);
    return cmap_lut[idx];
}

fn value_to_indexed_u8(val: f32) -> u32 {
    if (!is_finite_f32(val)) {
        return 0u;
    }
    let scaled = clamp((val - params.rescale_min) * params.inv_range_254, 0.0, 254.0);
    return 1u + u32(scaled);
}

fn eval_uv(px: u32, py: u32) -> vec2<f32> {
    if (params.coord_mode == 0u) {
        return vec2<f32>(u_coords[px], v_coords[py]);
    } else if (params.coord_mode == 1u) {
        let cell_px = max(params.dst_w / 16u, 1u);
        let inv_cell = 1.0 / f32(cell_px);
        let gx = min(px / cell_px, 15u);
        let gy = min(py / cell_px, 15u);
        let tx = f32(px - gx * cell_px) * inv_cell;
        let ty = f32(py - gy * cell_px) * inv_cell;
        let r0 = gy * 17u + gx;
        let r1 = (gy + 1u) * 17u + gx;
        let u_l = u_coords[r0] + (u_coords[r1] - u_coords[r0]) * ty;
        let u_r = u_coords[r0 + 1u] + (u_coords[r1 + 1u] - u_coords[r0 + 1u]) * ty;
        let v_l = v_coords[r0] + (v_coords[r1] - v_coords[r0]) * ty;
        let v_r = v_coords[r0 + 1u] + (v_coords[r1 + 1u] - v_coords[r0 + 1u]) * ty;
        return vec2<f32>(u_l + (u_r - u_l) * tx, v_l + (v_r - v_l) * tx);
    } else {
        let idx = py * params.dst_w + px;
        return vec2<f32>(u_coords[idx], v_coords[idx]);
    }
}

@compute @workgroup_size(16, 16, 1)
fn fused_colormap_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }
    let uv = eval_uv(px, py);
    let val = sample_mode(uv.x, uv.y);
    dst_words[py * params.dst_w + px] = value_to_rgba8(val);
}

@compute @workgroup_size(16, 16, 1)
fn resample_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }
    let uv = eval_uv(px, py);
    let val = sample_mode(uv.x, uv.y);
    dst_words[py * params.dst_w + px] = bitcast<u32>(val);
}

/// Each invocation computes 4 horizontally adjacent pixels and packs them into 1 u32,
/// shrinking GPU -> CPU readback bandwidth by 4x vs RGBA8/f32.
@compute @workgroup_size(16, 16, 1)
fn fused_colormap_indexed_u8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_x = gid.x;
    let py = gid.y;
    let words_per_row = (params.dst_w + 3u) / 4u;
    if (word_x >= words_per_row || py >= params.dst_h) {
        return;
    }

    let base_x = word_x * 4u;
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let px = base_x + k;
        if (px < params.dst_w) {
            let uv = eval_uv(px, py);
            let val = sample_mode(uv.x, uv.y);
            let idx = value_to_indexed_u8(val);
            packed = packed | ((idx & 0xffu) << (k * 8u));
        }
    }
    dst_words[py * words_per_row + word_x] = packed;
}
