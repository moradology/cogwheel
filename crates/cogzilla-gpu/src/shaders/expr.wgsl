struct ExprParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    band_stride: u32,
    dst_w: u32,
    dst_h: u32,
    num_bands: u32,
    num_instrs: u32,
    /// 0 = Bilinear, 1 = Cubic, 2 = Lanczos3
    resample_mode: u32,
    /// Bit 0: apply RGB LUT, Bit 1: apply LCH saturation
    formula_flags: u32,
    /// 0 = Separable1D, 1 = Grid2D (17x17 bilinear), 2 = Exact2D (dst_w * dst_h)
    coord_mode: u32,
    _pad1: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    saturation: f32,
    rgb_min: vec4<f32>,
    rgb_inv_255: vec4<f32>,
}

struct GpuInstr {
    opcode: u32,
    imm_u32: u32,
    imm_f32: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: ExprParams;
/// Stacked planar f32 slabs: band b at offset b * params.band_stride
@group(0) @binding(1) var<storage, read> bands_slab: array<f32>;
@group(0) @binding(2) var<storage, read> u_coords: array<f32>;
@group(0) @binding(3) var<storage, read> v_coords: array<f32>;
@group(0) @binding(4) var<storage, read> instrs: array<GpuInstr>;
/// 256 packed RGBA8 u32 entries (for expr colormap) OR 256 packed RGB-LUT u32 entries (r_lut | g_lut<<8 | b_lut<<16)
@group(0) @binding(5) var<storage, read> lut_u32: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst_words: array<u32>;

const PI: f32 = 3.14159265358979323846;
const INV_LN10: f32 = 0.4342944819032518;

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

fn sample_checked(band: u32, x: i32, y: i32) -> f32 {
    if (x < 0 || y < 0 || x >= i32(params.src_w) || y >= i32(params.src_h)) {
        return nan_f32();
    }
    return bands_slab[band * params.band_stride + u32(y) * params.src_stride + u32(x)];
}

fn sample_bilinear_raw(band: u32, u: f32, v: f32) -> f32 {
    let x0_f = floor(u);
    let y0_f = floor(v);
    let x0 = i32(x0_f);
    let y0 = i32(y0_f);
    let fx = u - x0_f;
    let fy = v - y0_f;

    let v00 = sample_checked(band, x0, y0);
    let v10 = sample_checked(band, x0 + 1, y0);
    let v01 = sample_checked(band, x0, y0 + 1);
    let v11 = sample_checked(band, x0 + 1, y0 + 1);

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

fn sample_cubic(band: u32, u: f32, v: f32) -> f32 {
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
            let s = sample_checked(band, x0 + i - 1, ry);
            if (is_nan_f32(s)) {
                return sample_bilinear_raw(band, u, v);
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

fn sample_lanczos3(band: u32, u: f32, v: f32) -> f32 {
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
            let s = sample_checked(band, x0 + i - 2, ry);
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
    return sample_bilinear_raw(band, u, v);
}

fn sample_band(band: u32, u: f32, v: f32) -> f32 {
    if (params.resample_mode == 0u) {
        if (!is_finite_f32(u) || !is_finite_f32(v)
            || u < -0.5 || v < -0.5
            || u > f32(params.src_w) - 0.5
            || v > f32(params.src_h) - 0.5) {
            return nan_f32();
        }
        return sample_bilinear_raw(band, u, v);
    } else if (params.resample_mode == 1u) {
        return sample_cubic(band, u, v);
    } else if (params.resample_mode == 3u) {
        if (!is_finite_f32(u) || !is_finite_f32(v)
            || u < -0.5 || v < -0.5
            || u > f32(params.src_w) - 0.5
            || v > f32(params.src_h) - 0.5) {
            return nan_f32();
        }
        return sample_checked(band, i32(floor(u + 0.5)), i32(floor(v + 0.5)));
    } else {
        return sample_lanczos3(band, u, v);
    }
}

fn eval_expr_vm_range(
    bvals: ptr<function, array<f32, 8>>,
    pc_start: u32,
    pc_end: u32,
) -> f32 {
    var stack: array<f32, 32>;
    var sp: u32 = 0u;

    for (var pc: u32 = pc_start; pc < pc_end; pc = pc + 1u) {
        let ins = instrs[pc];
        switch ins.opcode {
            case 0u: { // PushConst
                stack[sp] = ins.imm_f32;
                sp += 1u;
            }
            case 1u: { // LoadBand
                stack[sp] = (*bvals)[ins.imm_u32 & 7u];
                sp += 1u;
            }
            case 2u: { // Add
                sp -= 1u;
                stack[sp - 1u] = stack[sp - 1u] + stack[sp];
            }
            case 3u: { // Sub
                sp -= 1u;
                stack[sp - 1u] = stack[sp - 1u] - stack[sp];
            }
            case 4u: { // Mul
                sp -= 1u;
                stack[sp - 1u] = stack[sp - 1u] * stack[sp];
            }
            case 5u: { // Div
                sp -= 1u;
                let denom = stack[sp];
                if (abs(denom) < 1e-12 || is_nan_f32(denom)) {
                    stack[sp - 1u] = nan_f32();
                } else {
                    stack[sp - 1u] = stack[sp - 1u] / denom;
                }
            }
            case 6u: { // Pow (safe for a <= 0.0, matching f32::powf)
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                if (is_nan_f32(a) || is_nan_f32(b)) {
                    stack[sp - 1u] = nan_f32();
                } else if (b == 0.0) {
                    stack[sp - 1u] = 1.0;
                } else if (a > 0.0) {
                    stack[sp - 1u] = pow(a, b);
                } else if (a == 0.0) {
                    stack[sp - 1u] = select(nan_f32(), 0.0, b > 0.0);
                } else if (trunc(b) == b) {
                    let mag = pow(-a, b);
                    let is_odd = abs(b - 2.0 * floor(b * 0.5)) >= 0.5;
                    stack[sp - 1u] = select(mag, -mag, is_odd);
                } else {
                    stack[sp - 1u] = nan_f32();
                }
            }
            case 7u: { // Neg
                stack[sp - 1u] = -stack[sp - 1u];
            }
            case 8u: { // Log10
                let x = stack[sp - 1u];
                stack[sp - 1u] = select(nan_f32(), log(x) * INV_LN10, x > 0.0);
            }
            case 9u: { // Ln
                let x = stack[sp - 1u];
                stack[sp - 1u] = select(nan_f32(), log(x), x > 0.0);
            }
            case 10u: { // Log2
                let x = stack[sp - 1u];
                stack[sp - 1u] = select(nan_f32(), log2(x), x > 0.0);
            }
            case 11u: { // Db (10 * log10(x) for x > 0 else NaN)
                let x = stack[sp - 1u];
                stack[sp - 1u] = select(nan_f32(), 10.0 * log(x) * INV_LN10, x > 0.0);
            }
            case 12u: { // Exp
                stack[sp - 1u] = exp(stack[sp - 1u]);
            }
            case 13u: { // Sqrt
                let x = stack[sp - 1u];
                stack[sp - 1u] = select(nan_f32(), sqrt(x), x >= 0.0);
            }
            case 14u: { // Abs
                stack[sp - 1u] = abs(stack[sp - 1u]);
            }
            case 15u: { // Sin
                stack[sp - 1u] = sin(stack[sp - 1u]);
            }
            case 16u: { // Cos
                stack[sp - 1u] = cos(stack[sp - 1u]);
            }
            case 17u: { // Tan
                stack[sp - 1u] = tan(stack[sp - 1u]);
            }
            case 18u: { // Min (NaN-propagating)
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(min(a, b), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 19u: { // Max (NaN-propagating)
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(max(a, b), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 20u: { // Clip (val, lo, hi) — NaN-propagating, handles lo > hi
                sp -= 2u;
                let x = stack[sp - 1u];
                let lo = stack[sp];
                let hi = stack[sp + 1u];
                if (is_nan_f32(x) || is_nan_f32(lo) || is_nan_f32(hi)) {
                    stack[sp - 1u] = nan_f32();
                } else {
                    let a = min(lo, hi);
                    let b = max(lo, hi);
                    stack[sp - 1u] = clamp(x, a, b);
                }
            }
            case 21u: { // Lt
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(select(0.0, 1.0, a < b), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 22u: { // Le
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(select(0.0, 1.0, a <= b), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 23u: { // Gt
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(select(0.0, 1.0, a > b), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 24u: { // Ge
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(select(0.0, 1.0, a >= b), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 25u: { // Eq
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(select(0.0, 1.0, a == b), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 26u: { // Ne
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(select(0.0, 1.0, a != b), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 27u: { // Where (cond, a, b)
                sp -= 2u;
                let cond = stack[sp - 1u];
                let a = stack[sp];
                let b = stack[sp + 1u];
                if (is_nan_f32(cond)) {
                    stack[sp - 1u] = nan_f32();
                } else if (cond != 0.0) {
                    stack[sp - 1u] = a;
                } else {
                    stack[sp - 1u] = b;
                }
            }
            case 28u: { // Ceil
                stack[sp - 1u] = ceil(stack[sp - 1u]);
            }
            case 29u: { // Floor
                stack[sp - 1u] = floor(stack[sp - 1u]);
            }
            case 30u: { // Round
                stack[sp - 1u] = round(stack[sp - 1u]);
            }
            default: {}
        }
    }

    if (sp == 0u) {
        return nan_f32();
    }
    return stack[sp - 1u];
}

fn eval_expr_vm(u: f32, v: f32) -> f32 {
    var bvals: array<f32, 8>;
    let nb = min(params.num_bands, 8u);
    for (var b: u32 = 0u; b < nb; b = b + 1u) {
        bvals[b] = sample_band(b, u, v);
    }
    return eval_expr_vm_range(&bvals, 0u, params.num_instrs);
}

fn eval_uv(px: u32, py: u32) -> vec2<f32> {
    if (params.coord_mode == 0u) {
        return vec2<f32>(u_coords[px], v_coords[py]);
    }
    if (params.coord_mode == 2u) {
        let idx = py * params.dst_w + px;
        return vec2<f32>(u_coords[idx], v_coords[idx]);
    }
    if (params.coord_mode == 3u) {
        let idx = py * params.dst_w + px;
        let word = bitcast<u32>(v_coords[idx >> 5u]);
        if ((word & (1u << (idx & 31u))) == 0u) {
            return vec2<f32>(nan_f32(), nan_f32());
        }
        return vec2<f32>(u_coords[px], u_coords[params.dst_w + py]);
    }
    if (params.coord_mode == 4u) {
        let idx = py * params.dst_w + px;
        let word = bitcast<u32>(v_coords[idx >> 5u]);
        if ((word & (1u << (idx & 31u))) == 0u) {
            return vec2<f32>(nan_f32(), nan_f32());
        }
        let gx_f = f32(px) * 16.0 / f32(params.dst_w);
        let gy_f = f32(py) * 16.0 / f32(params.dst_h);
        let gx0 = min(u32(gx_f), 15u);
        let gy0 = min(u32(gy_f), 15u);
        let fx = gx_f - f32(gx0);
        let fy = gy_f - f32(gy0);
        let i00 = gy0 * 17u + gx0;
        let i10 = i00 + 1u;
        let i01 = i00 + 17u;
        let i11 = i01 + 1u;
        let u0 = u_coords[i00] + (u_coords[i10] - u_coords[i00]) * fx;
        let u1 = u_coords[i01] + (u_coords[i11] - u_coords[i01]) * fx;
        let v0 = u_coords[289u + i00] + (u_coords[289u + i10] - u_coords[289u + i00]) * fx;
        let v1 = u_coords[289u + i01] + (u_coords[289u + i11] - u_coords[289u + i01]) * fx;
        return vec2<f32>(u0 + (u1 - u0) * fy, v0 + (v1 - v0) * fy);
    }
    // Grid2D (17x17 coarse warp grid matching cogzilla-proj::CoordPlan::Grid2D)
    let gx_f = f32(px) * 16.0 / f32(params.dst_w);
    let gy_f = f32(py) * 16.0 / f32(params.dst_h);
    let gx0 = min(u32(gx_f), 15u);
    let gy0 = min(u32(gy_f), 15u);
    let fx = gx_f - f32(gx0);
    let fy = gy_f - f32(gy0);
    let i00 = gy0 * 17u + gx0;
    let i10 = i00 + 1u;
    let i01 = i00 + 17u;
    let i11 = i01 + 1u;
    let u0 = u_coords[i00] + (u_coords[i10] - u_coords[i00]) * fx;
    let u1 = u_coords[i01] + (u_coords[i11] - u_coords[i01]) * fx;
    let v0 = v_coords[i00] + (v_coords[i10] - v_coords[i00]) * fx;
    let v1 = v_coords[i01] + (v_coords[i11] - v_coords[i01]) * fx;
    return vec2<f32>(u0 + (u1 - u0) * fy, v0 + (v1 - v0) * fy);
}

@compute @workgroup_size(16, 16, 1)
fn expr_eval_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }
    let uv = eval_uv(px, py);
    let out_idx = py * params.dst_w + px;
    if (!is_finite_f32(uv.x) || !is_finite_f32(uv.y)) {
        dst_words[out_idx] = 0u;
        return;
    }
    let val = eval_expr_vm(uv.x, uv.y);
    if (!is_finite_f32(val)) {
        dst_words[out_idx] = 0u;
    } else {
        let scaled = clamp((val - params.rescale_min) * params.inv_range_255, 0.0, 255.0);
        dst_words[out_idx] = lut_u32[u32(scaled)];
    }
}

@compute @workgroup_size(16, 16, 1)
fn expr_eval_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }
    let uv = eval_uv(px, py);
    if (!is_finite_f32(uv.x) || !is_finite_f32(uv.y)) {
        dst_words[py * params.dst_w + px] = bitcast<u32>(nan_f32());
        return;
    }
    let val = eval_expr_vm(uv.x, uv.y);
    dst_words[py * params.dst_w + px] = bitcast<u32>(val);
}

@compute @workgroup_size(16, 16, 1)
fn expr_eval_indexed_u8(@builtin(global_invocation_id) gid: vec3<u32>) {
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
            var idx: u32 = 0u;
            if (is_finite_f32(uv.x) && is_finite_f32(uv.y)) {
                let val = eval_expr_vm(uv.x, uv.y);
                if (is_finite_f32(val)) {
                    let scaled = clamp((val - params.rescale_min) * params.inv_range_254, 0.0, 254.0);
                    idx = 1u + u32(scaled);
                }
            }
            packed = packed | ((idx & 0xffu) << (k * 8u));
        }
    }
    dst_words[py * words_per_row + word_x] = packed;
}

// ---- CIE LCH (D65) Saturation matching cogzilla-kernel::color_formula ----

fn srgb_to_lin(u: f32) -> f32 {
    if (u <= 0.04045) {
        return u * (1.0 / 12.92);
    }
    return pow((u + 0.055) * (1.0 / 1.055), 2.4);
}

fn lin_to_srgb_u32(u: f32) -> u32 {
    let c = clamp(u, 0.0, 1.0);
    var s: f32;
    if (c <= 0.0031308) {
        s = c * 12.92;
    } else {
        s = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    }
    return u32(clamp(round(s * 255.0), 0.0, 255.0));
}

fn lab_f(t: f32) -> f32 {
    if (t > 0.0088564516) {
        return pow(t, 1.0 / 3.0);
    }
    return 7.787037 * t + (16.0 / 116.0);
}

fn lab_f_inv(t: f32) -> f32 {
    if (t > 0.20689655) {
        return t * t * t;
    }
    return (t - (16.0 / 116.0)) * (1.0 / 7.787037);
}

fn apply_lch_sat(r: u32, g: u32, b: u32, sat: f32) -> vec3<u32> {
    let rl = srgb_to_lin(f32(r) * (1.0 / 255.0));
    let gl = srgb_to_lin(f32(g) * (1.0 / 255.0));
    let bl = srgb_to_lin(f32(b) * (1.0 / 255.0));

    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;

    let fx = lab_f(x * (1.0 / 0.95047));
    let fy = lab_f(y);
    let fz = lab_f(z * (1.0 / 1.08883));
    let a = 500.0 * (fx - fy) * sat;
    let b_lab = 200.0 * (fy - fz) * sat;

    let fx2 = fy + a * (1.0 / 500.0);
    let fz2 = fy - b_lab * (1.0 / 200.0);
    let x2 = 0.95047 * lab_f_inv(fx2);
    let y2 = lab_f_inv(fy);
    let z2 = 1.08883 * lab_f_inv(fz2);

    let rl2 = 3.2404542 * x2 - 1.5371385 * y2 - 0.4985314 * z2;
    let gl2 = -0.9692660 * x2 + 1.8760108 * y2 + 0.0415560 * z2;
    let bl2 = 0.0556434 * x2 - 0.2040259 * y2 + 1.0572252 * z2;

    return vec3<u32>(lin_to_srgb_u32(rl2), lin_to_srgb_u32(gl2), lin_to_srgb_u32(bl2));
}

/// Fused 3-band RGB resample -> per-channel rescale -> Gamma/Sigmoidal LUT -> LCH Saturation -> RGBA8.
@compute @workgroup_size(16, 16, 1)
fn rgb_color_formula_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let uv = eval_uv(px, py);
    let vr = sample_band(0u, uv.x, uv.y);
    let vg = sample_band(1u, uv.x, uv.y);
    let vb = sample_band(2u, uv.x, uv.y);
    let out_idx = py * params.dst_w + px;

    if (!is_finite_f32(vr) || !is_finite_f32(vg) || !is_finite_f32(vb)) {
        dst_words[out_idx] = 0u;
        return;
    }

    var r = u32(clamp((vr - params.rgb_min.x) * params.rgb_inv_255.x, 0.0, 255.0));
    var g = u32(clamp((vg - params.rgb_min.y) * params.rgb_inv_255.y, 0.0, 255.0));
    var b = u32(clamp((vb - params.rgb_min.z) * params.rgb_inv_255.z, 0.0, 255.0));

    if ((params.formula_flags & 1u) != 0u) {
        r = lut_u32[r] & 0xffu;
        g = (lut_u32[g] >> 8u) & 0xffu;
        b = (lut_u32[b] >> 16u) & 0xffu;
    }
    if ((params.formula_flags & 2u) != 0u) {
        let sat_rgb = apply_lch_sat(r, g, b, params.saturation);
        r = sat_rgb.x;
        g = sat_rgb.y;
        b = sat_rgb.z;
    }

    dst_words[out_idx] = r | (g << 8u) | (b << 16u) | (0xffu << 24u);
}

/// Fused Dual-Pol SAR False-Color RGB ([VV_dB, VH_dB, VV_dB - VH_dB]) -> rescale ->
/// Gamma/Sigmoidal LUT -> CIE LCH Saturation -> RGBA8.
@compute @workgroup_size(16, 16, 1)
fn sar_rgb_color_formula_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let uv = eval_uv(px, py);
    let p_vv = sample_band(0u, uv.x, uv.y);
    let p_vh = sample_band(1u, uv.x, uv.y);
    let out_idx = py * params.dst_w + px;

    if (!is_finite_f32(p_vv) || !is_finite_f32(p_vh) || p_vv <= 0.0 || p_vh <= 0.0) {
        dst_words[out_idx] = 0u;
        return;
    }

    let vv = 10.0 * log(p_vv) * INV_LN10;
    let vh = 10.0 * log(p_vh) * INV_LN10;
    let diff = vv - vh;

    var r = u32(clamp((vv - params.rgb_min.x) * params.rgb_inv_255.x, 0.0, 255.0));
    var g = u32(clamp((vh - params.rgb_min.y) * params.rgb_inv_255.y, 0.0, 255.0));
    var b = u32(clamp((diff - params.rgb_min.z) * params.rgb_inv_255.z, 0.0, 255.0));

    if ((params.formula_flags & 1u) != 0u) {
        r = lut_u32[r] & 0xffu;
        g = (lut_u32[g] >> 8u) & 0xffu;
        b = (lut_u32[b] >> 16u) & 0xffu;
    }
    if ((params.formula_flags & 2u) != 0u) {
        let sat_rgb = apply_lch_sat(r, g, b, params.saturation);
        r = sat_rgb.x;
        g = sat_rgb.y;
        b = sat_rgb.z;
    }

    dst_words[out_idx] = r | (g << 8u) | (b << 16u) | (0xffu << 24u);
}

/// Fused 3-expression CustomExprRgb (`[prog_r, prog_g, prog_b]`) -> rescale ->
/// Gamma/Sigmoidal LUT -> CIE LCH Saturation -> RGBA8.
/// `params.num_instrs` packs `len_r | (len_g << 8u) | (len_b << 16u)`.
@compute @workgroup_size(16, 16, 1)
fn expr_rgb_color_formula_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let uv = eval_uv(px, py);
    let out_idx = py * params.dst_w + px;
    if (!is_finite_f32(uv.x) || !is_finite_f32(uv.y)) {
        dst_words[out_idx] = 0u;
        return;
    }
    var bvals: array<f32, 8>;
    let nb = min(params.num_bands, 8u);
    for (var b: u32 = 0u; b < nb; b = b + 1u) {
        bvals[b] = sample_band(b, uv.x, uv.y);
    }

    let len0 = params.num_instrs & 0xffu;
    let len1 = (params.num_instrs >> 8u) & 0xffu;
    let len2 = (params.num_instrs >> 16u) & 0xffu;
    let vr = eval_expr_vm_range(&bvals, 0u, len0);
    let vg = eval_expr_vm_range(&bvals, len0, len0 + len1);
    let vb = eval_expr_vm_range(&bvals, len0 + len1, len0 + len1 + len2);

    if (!is_finite_f32(vr) || !is_finite_f32(vg) || !is_finite_f32(vb)) {
        dst_words[out_idx] = 0u;
        return;
    }

    var r = u32(clamp((vr - params.rgb_min.x) * params.rgb_inv_255.x, 0.0, 255.0));
    var g = u32(clamp((vg - params.rgb_min.y) * params.rgb_inv_255.y, 0.0, 255.0));
    var b = u32(clamp((vb - params.rgb_min.z) * params.rgb_inv_255.z, 0.0, 255.0));

    if ((params.formula_flags & 1u) != 0u) {
        r = lut_u32[r] & 0xffu;
        g = (lut_u32[g] >> 8u) & 0xffu;
        b = (lut_u32[b] >> 16u) & 0xffu;
    }
    if ((params.formula_flags & 2u) != 0u) {
        let sat_rgb = apply_lch_sat(r, g, b, params.saturation);
        r = sat_rgb.x;
        g = sat_rgb.y;
        b = sat_rgb.z;
    }

    dst_words[out_idx] = r | (g << 8u) | (b << 16u) | (0xffu << 24u);
}

/// Multi-band planar `(num_bands, dst_h, dst_w)` Float32 resampling for scientific `.npy` / `.tif`
/// ML tensor cubes (3D Z-dispatch over `gid.z in 0..params.num_bands`).
@compute @workgroup_size(16, 16, 1)
fn multiband_resample_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let band = gid.z;
    if (px >= params.dst_w || py >= params.dst_h || band >= params.num_bands) {
        return;
    }
    let uv = eval_uv(px, py);
    let val = sample_band(band, uv.x, uv.y);
    let plane_pix = params.dst_w * params.dst_h;
    dst_words[band * plane_pix + py * params.dst_w + px] = bitcast<u32>(val);
}

/// Planar `(3, dst_h, dst_w)` Float32 evaluation for 3-expression `CustomExprRgb` `.npy` / `.tif`.
@compute @workgroup_size(16, 16, 1)
fn expr_rgb_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }
    let uv = eval_uv(px, py);
    let plane_pix = params.dst_w * params.dst_h;
    let pix = py * params.dst_w + px;
    if (!is_finite_f32(uv.x) || !is_finite_f32(uv.y)) {
        let nan_bits = bitcast<u32>(nan_f32());
        dst_words[pix] = nan_bits;
        dst_words[plane_pix + pix] = nan_bits;
        dst_words[2u * plane_pix + pix] = nan_bits;
        return;
    }
    var bvals: array<f32, 8>;
    let nb = min(params.num_bands, 8u);
    for (var b: u32 = 0u; b < nb; b = b + 1u) {
        bvals[b] = sample_band(b, uv.x, uv.y);
    }
    let len0 = params.num_instrs & 0xffu;
    let len1 = (params.num_instrs >> 8u) & 0xffu;
    let len2 = (params.num_instrs >> 16u) & 0xffu;
    let vr = eval_expr_vm_range(&bvals, 0u, len0);
    let vg = eval_expr_vm_range(&bvals, len0, len0 + len1);
    let vb = eval_expr_vm_range(&bvals, len0 + len1, len0 + len1 + len2);
    dst_words[pix] = bitcast<u32>(vr);
    dst_words[plane_pix + pix] = bitcast<u32>(vg);
    dst_words[2u * plane_pix + pix] = bitcast<u32>(vb);
}
