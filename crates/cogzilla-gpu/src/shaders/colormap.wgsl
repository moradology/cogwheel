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
