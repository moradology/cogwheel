struct MosaicParams {
    width: u32,
    height: u32,
    plane_stride: u32,
    num_scenes: u32,
    /// 0=First, 1=Highest, 2=Lowest, 3=Mean, 4=Median, 5=Stdev, 6=Count
    method: u32,
    /// For 3-band RGB entry point: element stride between R, G, and B stacks (`num_scenes * plane_stride`).
    band_stride: u32,
    _pad1: u32,
    _pad2: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    _pad3: f32,
    rescale_min_g: f32,
    inv_range_255_g: f32,
    rescale_min_b: f32,
    inv_range_255_b: f32,
}

@group(0) @binding(0) var<uniform> params: MosaicParams;
/// Stacked scene planes: scene s of band b at offset b * params.band_stride + s * params.plane_stride
@group(0) @binding(1) var<storage, read> scene_planes: array<f32>;
@group(0) @binding(2) var<storage, read> cmap_lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_words: array<u32>;

fn select_pixel_at(base_off: u32, px_idx: u32) -> f32 {
    let n = min(params.num_scenes, 32u);

    if (params.method == 0u) {
        // First
        for (var s: u32 = 0u; s < n; s = s + 1u) {
            let v = scene_planes[base_off + s * params.plane_stride + px_idx];
            if (is_finite_f32(v)) {
                return v;
            }
        }
        return nan_f32();
    }

    if (params.method == 1u) {
        // Highest
        var best = -3.4028234e38;
        var count: u32 = 0u;
        for (var s: u32 = 0u; s < n; s = s + 1u) {
            let v = scene_planes[base_off + s * params.plane_stride + px_idx];
            if (is_finite_f32(v)) {
                if (count == 0u || v > best) {
                    best = v;
                }
                count += 1u;
            }
        }
        return select(nan_f32(), best, count > 0u);
    }

    if (params.method == 2u) {
        // Lowest
        var best = 3.4028234e38;
        var count: u32 = 0u;
        for (var s: u32 = 0u; s < n; s = s + 1u) {
            let v = scene_planes[base_off + s * params.plane_stride + px_idx];
            if (is_finite_f32(v)) {
                if (count == 0u || v < best) {
                    best = v;
                }
                count += 1u;
            }
        }
        return select(nan_f32(), best, count > 0u);
    }

    if (params.method == 3u) {
        // Mean
        var sum = 0.0;
        var count: u32 = 0u;
        for (var s: u32 = 0u; s < n; s = s + 1u) {
            let v = scene_planes[base_off + s * params.plane_stride + px_idx];
            if (is_finite_f32(v)) {
                sum += v;
                count += 1u;
            }
        }
        return select(nan_f32(), sum / f32(count), count > 0u);
    }

    if (params.method == 4u) {
        // Median (in-register insertion sort over up to 32 valid values)
        var vals: array<f32, 32>;
        var count: u32 = 0u;
        for (var s: u32 = 0u; s < n; s = s + 1u) {
            let v = scene_planes[base_off + s * params.plane_stride + px_idx];
            if (is_finite_f32(v)) {
                var j = count;
                while (j > 0u && vals[j - 1u] > v) {
                    vals[j] = vals[j - 1u];
                    j -= 1u;
                }
                vals[j] = v;
                count += 1u;
            }
        }
        if (count == 0u) {
            return nan_f32();
        }
        let mid = count >> 1u;
        if ((count & 1u) == 1u) {
            return vals[mid];
        }
        return 0.5 * (vals[mid - 1u] + vals[mid]);
    }

    if (params.method == 5u) {
        // Stdev (Welford's online algorithm for numerical stability in f32)
        var mean = 0.0;
        var m2 = 0.0;
        var count: u32 = 0u;
        for (var s: u32 = 0u; s < n; s = s + 1u) {
            let v = scene_planes[base_off + s * params.plane_stride + px_idx];
            if (is_finite_f32(v)) {
                count += 1u;
                let delta = v - mean;
                mean += delta / f32(count);
                let delta2 = v - mean;
                m2 += delta * delta2;
            }
        }
        if (count == 0u) {
            return nan_f32();
        }
        return sqrt(max(m2 / f32(count), 0.0));
    }

    // method == 6u: Count
    var count: u32 = 0u;
    for (var s: u32 = 0u; s < n; s = s + 1u) {
        let v = scene_planes[base_off + s * params.plane_stride + px_idx];
        if (is_finite_f32(v)) {
            count += 1u;
        }
    }
    return select(nan_f32(), f32(count), count > 0u);
}

fn select_pixel(px_idx: u32) -> f32 {
    return select_pixel_at(0u, px_idx);
}

@compute @workgroup_size(16, 16, 1)
fn mosaic_select_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    let idx = py * params.width + px;
    dst_words[idx] = bitcast<u32>(select_pixel(idx));
}

@compute @workgroup_size(16, 16, 1)
fn mosaic_select_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    let idx = py * params.width + px;
    let val = select_pixel(idx);
    if (!is_finite_f32(val)) {
        dst_words[idx] = 0u;
    } else {
        let scaled = clamp((val - params.rescale_min) * params.inv_range_255, 0.0, 255.0);
        dst_words[idx] = cmap_lut[u32(scaled)];
    }
}

@compute @workgroup_size(16, 16, 1)
fn mosaic_select_indexed_u8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_x = gid.x;
    let py = gid.y;
    let words_per_row = (params.width + 3u) / 4u;
    if (word_x >= words_per_row || py >= params.height) {
        return;
    }
    let base_x = word_x * 4u;
    let row_off = py * params.width;
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let px = base_x + k;
        if (px < params.width) {
            let val = select_pixel(row_off + px);
            var idx: u32 = 0u;
            if (is_finite_f32(val)) {
                let scaled = clamp((val - params.rescale_min) * params.inv_range_254, 0.0, 254.0);
                idx = 1u + u32(scaled);
            }
            packed = packed | ((idx & 0xffu) << (k * 8u));
        }
    }
    dst_words[py * words_per_row + word_x] = packed;
}

@compute @workgroup_size(16, 16, 1)
fn mosaic_select_rgb_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    let idx = py * params.width + px;
    let r = select_pixel_at(0u, idx);
    if (!is_finite_f32(r)) {
        dst_words[idx] = 0u;
        return;
    }
    let g = select_pixel_at(params.band_stride, idx);
    if (!is_finite_f32(g)) {
        dst_words[idx] = 0u;
        return;
    }
    let b = select_pixel_at(2u * params.band_stride, idx);
    if (!is_finite_f32(b)) {
        dst_words[idx] = 0u;
        return;
    }
    let r8 = u32(clamp((r - params.rescale_min) * params.inv_range_255, 0.0, 255.0));
    let g8 = u32(clamp((g - params.rescale_min_g) * params.inv_range_255_g, 0.0, 255.0));
    let b8 = u32(clamp((b - params.rescale_min_b) * params.inv_range_255_b, 0.0, 255.0));
    dst_words[idx] = r8 | (g8 << 8u) | (b8 << 16u) | 0xff000000u;
}
