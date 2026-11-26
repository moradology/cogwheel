struct ResampleParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ResampleParams;
@group(0) @binding(1) var<storage, read> src_slab: array<f32>;
@group(0) @binding(2) var<storage, read> u_coords: array<f32>;
@group(0) @binding(3) var<storage, read> v_coords: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_out: array<f32>;

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

fn sample_bilinear_fallback(u: f32, v: f32) -> f32 {
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

@compute @workgroup_size(16, 16, 1)
fn resample_cubic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let u = u_coords[px];
    let v = v_coords[py];
    let out_idx = py * params.dst_w + px;

    if (!is_finite_f32(u) || !is_finite_f32(v)
        || u < -2.0 || v < -2.0
        || u > f32(params.src_w) + 1.0
        || v > f32(params.src_h) + 1.0) {
        dst_out[out_idx] = nan_f32();
        return;
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
                dst_out[out_idx] = sample_bilinear_fallback(u, v);
                return;
            }
            row_acc += s * wx[i];
        }
        acc += row_acc * wy[j];
    }

    dst_out[out_idx] = acc;
}

@compute @workgroup_size(16, 16, 1)
fn resample_lanczos3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let u = u_coords[px];
    let v = v_coords[py];
    let out_idx = py * params.dst_w + px;

    if (!is_finite_f32(u) || !is_finite_f32(v)
        || u < -0.5 || v < -0.5
        || u > f32(params.src_w) - 0.5
        || v > f32(params.src_h) - 0.5) {
        dst_out[out_idx] = nan_f32();
        return;
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
        dst_out[out_idx] = sum / wsum;
    } else {
        dst_out[out_idx] = sample_bilinear_fallback(u, v);
    }
}

@compute @workgroup_size(1, 1, 1)
fn noop_dispatch() {
    // Minimal kernel to measure pure command submission + sync overhead.
}
