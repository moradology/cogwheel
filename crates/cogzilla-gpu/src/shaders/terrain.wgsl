struct TerrainParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    /// Resample mode: 0 = Bilinear, 1 = Cubic, 2 = Lanczos3
    resample_mode: u32,
    /// Algorithm: 0 = Hillshade, 1 = Slope, 2 = Aspect, 3 = Terrarium, 4 = TerrainRgb
    alg_mode: u32,
    /// Coordinate mode: 0 = Separable1D, 1 = Grid2D (17x17), 2 = Exact2D (dst_w * dst_h)
    coord_mode: u32,
    inv_8dx: f32,
    inv_8dy: f32,
    z_factor: f32,
    sun_e: f32,
    sun_n: f32,
    sun_u: f32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> params: TerrainParams;
@group(0) @binding(1) var<storage, read> src_slab: array<f32>;
@group(0) @binding(2) var<storage, read> u_coords: array<f32>;
@group(0) @binding(3) var<storage, read> v_coords: array<f32>;
@group(0) @binding(4) var<storage, read> cmap_lut: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst_words: array<u32>;

const RAD_TO_DEG: f32 = 57.29577951308232;

// 18x18 workgroup shared memory tile for 16x16 workgroup + 1-pixel halo (324 floats = 1.3 KiB).
var<workgroup> shared_elev: array<f32, 324>;
// For packed IndexedU8 output (4 pixels per u32) inside a 16x16 workgroup.
var<workgroup> shared_idx: array<u32, 256>;

fn sample_elev(u: f32, v: f32) -> f32 {
    if (params.resample_mode == 0u) {
        return sample_bilinear(u, v);
    } else if (params.resample_mode == 1u) {
        return sample_cubic(u, v);
    } else if (params.resample_mode == 3u) {
        if (!is_finite_f32(u) || !is_finite_f32(v)) {
            return nan_f32();
        }
        return sample_checked(i32(floor(u + 0.5)), i32(floor(v + 0.5)));
    } else {
        return sample_lanczos3(u, v);
    }
}

fn eval_uv_extrap(ix: i32, iy: i32) -> vec2<f32> {
    let w = i32(params.dst_w);
    let h = i32(params.dst_h);
    let cx = u32(clamp(ix, 0, w - 1));
    let cy = u32(clamp(iy, 0, h - 1));
    var uv = eval_uv(cx, cy);
    if (ix < 0 && w >= 2) {
        uv = 2.0 * eval_uv(0u, cy) - eval_uv(1u, cy);
    } else if (ix >= w && w >= 2) {
        uv = 2.0 * eval_uv(u32(w - 1), cy) - eval_uv(u32(w - 2), cy);
    }
    if (iy < 0 && h >= 2) {
        let uv0 = eval_uv(cx, 0u);
        let uv1 = eval_uv(cx, 1u);
        uv = uv + (uv0 - uv1);
    } else if (iy >= h && h >= 2) {
        let uv0 = eval_uv(cx, u32(h - 1));
        let uv1 = eval_uv(cx, u32(h - 2));
        uv = uv + (uv0 - uv1);
    }
    return uv;
}

fn eval_horn_from_shared(lx: u32, ly: u32) -> f32 {
    let cx = lx + 1u;
    let cy = ly + 1u;
    let z_c = shared_elev[cy * 18u + cx];
    if (is_nan_f32(z_c)) {
        return nan_f32();
    }

    let r0 = (cy - 1u) * 18u;
    let r1 = cy * 18u;
    let r2 = (cy + 1u) * 18u;

    let z_nw = shared_elev[r0 + cx - 1u];
    let z_n  = shared_elev[r0 + cx];
    let z_ne = shared_elev[r0 + cx + 1u];
    let z_w  = shared_elev[r1 + cx - 1u];
    let z_e  = shared_elev[r1 + cx + 1u];
    let z_sw = shared_elev[r2 + cx - 1u];
    let z_s  = shared_elev[r2 + cx];
    let z_se = shared_elev[r2 + cx + 1u];

    let nw = select(z_nw, z_c, is_nan_f32(z_nw));
    let n  = select(z_n,  z_c, is_nan_f32(z_n));
    let ne = select(z_ne, z_c, is_nan_f32(z_ne));
    let w  = select(z_w,  z_c, is_nan_f32(z_w));
    let e  = select(z_e,  z_c, is_nan_f32(z_e));
    let sw = select(z_sw, z_c, is_nan_f32(z_sw));
    let s  = select(z_s,  z_c, is_nan_f32(z_s));
    let se = select(z_se, z_c, is_nan_f32(z_se));

    let dz_dx = ((ne + 2.0 * e + se) - (nw + 2.0 * w + sw)) * params.inv_8dx;
    let dz_dy = ((sw + 2.0 * s + se) - (nw + 2.0 * n + ne)) * params.inv_8dy;

    if (params.alg_mode == 0u) {
        // Hillshade [0, 255]
        let nx = -params.z_factor * dz_dx;
        let ny = params.z_factor * dz_dy;
        let inv_len = inverseSqrt(nx * nx + ny * ny + 1.0);
        let dot = (nx * params.sun_e + ny * params.sun_n + params.sun_u) * inv_len;
        return clamp(max(dot, 0.0) * 255.0, 0.0, 255.0);
    } else if (params.alg_mode == 1u) {
        // Slope in degrees [0, 90]
        let grad = sqrt(dz_dx * dz_dx + dz_dy * dz_dy) * params.z_factor;
        return atan(grad) * RAD_TO_DEG;
    } else {
        // Aspect in degrees [0, 360) clockwise from North
        if (abs(dz_dx) < 1e-12 && abs(dz_dy) < 1e-12) {
            return nan_f32();
        }
        var bearing = atan2(-dz_dx, dz_dy) * RAD_TO_DEG;
        if (bearing < 0.0) {
            bearing += 360.0;
        }
        return bearing;
    }
}

fn load_shared_tile(wg_id: vec3<u32>, lid: vec3<u32>) {
    let base_x = i32(wg_id.x * 16u) - 1;
    let base_y = i32(wg_id.y * 16u) - 1;
    let flat = lid.y * 16u + lid.x;

    // 256 threads cooperatively fill 324 (18x18) elements in <= 2 iterations.
    var idx = flat;
    loop {
        if (idx >= 324u) {
            break;
        }
        let ty = idx / 18u;
        let tx = idx - ty * 18u;
        let uv = eval_uv_extrap(base_x + i32(tx), base_y + i32(ty));
        shared_elev[idx] = sample_elev(uv.x, uv.y);
        idx += 256u;
    }
    workgroupBarrier();
}

/// Fused resample + 18x18 shared-memory Horn 3x3 (Hillshade / Slope / Aspect) -> RGBA8.
@compute @workgroup_size(16, 16, 1)
fn dem_horn_rgba8(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    load_shared_tile(wg_id, lid);

    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let val = eval_horn_from_shared(lid.x, lid.y);
    if (!is_finite_f32(val)) {
        dst_words[py * params.dst_w + px] = 0u;
    } else {
        let scaled = clamp((val - params.rescale_min) * params.inv_range_255, 0.0, 255.0);
        dst_words[py * params.dst_w + px] = cmap_lut[u32(scaled)];
    }
}

/// Fused resample + 18x18 shared-memory Horn 3x3 (Hillshade / Slope / Aspect) -> raw f32.
@compute @workgroup_size(16, 16, 1)
fn dem_horn_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    load_shared_tile(wg_id, lid);

    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let val = eval_horn_from_shared(lid.x, lid.y);
    dst_words[py * params.dst_w + px] = bitcast<u32>(val);
}

/// Fused resample + 18x18 shared-memory Horn 3x3 -> packed IndexedU8 (4 pixels per u32, 4x smaller readback).
@compute @workgroup_size(16, 16, 1)
fn dem_horn_indexed_u8(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    load_shared_tile(wg_id, lid);

    let px = gid.x;
    let py = gid.y;
    var q: u32 = 0u;
    if (px < params.dst_w && py < params.dst_h) {
        let val = eval_horn_from_shared(lid.x, lid.y);
        if (is_finite_f32(val)) {
            let scaled = clamp((val - params.rescale_min) * params.inv_range_254, 0.0, 254.0);
            q = 1u + u32(scaled);
        }
    }
    shared_idx[lid.y * 16u + lid.x] = q;
    workgroupBarrier();

    // Every 4th lane packs 4 horizontal bytes into 1 u32 word and writes coalesced.
    if ((lid.x & 3u) == 0u && py < params.dst_h) {
        let word_x = px >> 2u;
        let words_per_row = (params.dst_w + 3u) >> 2u;
        if (word_x < words_per_row) {
            let base = lid.y * 16u + lid.x;
            let p0 = shared_idx[base];
            let p1 = shared_idx[base + 1u];
            let p2 = shared_idx[base + 2u];
            let p3 = shared_idx[base + 3u];
            dst_words[py * words_per_row + word_x] =
                (p0 & 0xffu) | ((p1 & 0xffu) << 8u) | ((p2 & 0xffu) << 16u) | ((p3 & 0xffu) << 24u);
        }
    }
}

/// Fused resample -> Terrarium RGB (alg_mode == 3) or Mapbox Terrain-RGB (alg_mode == 4) packed RGBA8.
@compute @workgroup_size(16, 16, 1)
fn dem_terrain_rgb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let uv = eval_uv(px, py);
    let elev = sample_elev(uv.x, uv.y);
    let out_idx = py * params.dst_w + px;
    if (!is_finite_f32(elev)) {
        dst_words[out_idx] = 0u;
        return;
    }

    if (params.alg_mode == 3u) {
        // Terrarium: v = clamp(elev + 32768.0, 0.0, 65535.99609375)
        let v = clamp(elev + 32768.0, 0.0, 65535.996);
        let vf = floor(v);
        let r = u32(floor(v / 256.0)) & 0xffu;
        let g = u32(vf) & 0xffu;
        let b = u32(clamp(floor((v - vf) * 256.0), 0.0, 255.0)) & 0xffu;
        dst_words[out_idx] = r | (g << 8u) | (b << 16u) | (0xffu << 24u);
    } else {
        // Mapbox Terrain-RGB: enc = round((elev + 10000.0) * 10.0)
        let enc = u32(clamp(round((elev + 10000.0) * 10.0), 0.0, 16777215.0));
        let r = (enc >> 16u) & 0xffu;
        let g = (enc >> 8u) & 0xffu;
        let b = enc & 0xffu;
        dst_words[out_idx] = r | (g << 8u) | (b << 16u) | (0xffu << 24u);
    }
}
