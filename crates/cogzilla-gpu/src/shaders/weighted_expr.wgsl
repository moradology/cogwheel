// Direct 1:1 native-window GPU expression VM pass chained into `weighted_stats.wgsl`.
// Evaluates `GpuInstr` bytecode only on pixels with `cov[i] > cov_eps`, writing directly
// into a `WeightedBandSlot::src_buf` with zero CPU expression evaluation and zero extra readback.

const INV_LN10: f32 = 0.4342944819032518;

struct WeightedExprParams {
    len: u32,
    /// Bitmask of which stacked `raw_bands` slots (`0..8`) this expression references.
    band_mask: u32,
    pc_start: u32,
    pc_end: u32,
    cov_eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuInstr {
    opcode: u32,
    imm_u32: u32,
    imm_f32: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: WeightedExprParams;
/// Stacked input raw bands: band `b` at `b * params.len + i` (up to 8 bands).
@group(0) @binding(1) var<storage, read> raw_bands: array<f32>;
@group(0) @binding(2) var<storage, read> cov: array<f32>;
@group(0) @binding(3) var<storage, read> instrs: array<GpuInstr>;
@group(0) @binding(4) var<storage, read_write> dst_expr: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn weighted_expr_eval(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * 256u;
    for (var i: u32 = gid.x; i < params.len; i = i + stride) {
        if (cov[i] <= params.cov_eps) {
            dst_expr[i] = nan_f32();
            continue;
        }

        var bvals: array<f32, 8>;
        var all_valid = true;
        let mask = params.band_mask;
        for (var b: u32 = 0u; b < 8u; b = b + 1u) {
            if ((mask & (1u << b)) != 0u) {
                let v = raw_bands[b * params.len + i];
                if (!is_finite_f32(v)) {
                    dst_expr[i] = nan_f32();
                    all_valid = false;
                    break;
                }
                bvals[b] = v;
            }
        }
        if (!all_valid) {
            continue;
        }

    var stack: array<f32, 32>;
    var sp: u32 = 0u;
    for (var pc: u32 = params.pc_start; pc < params.pc_end; pc = pc + 1u) {
        let ins = instrs[pc];
        switch ins.opcode {
            case 0u: { // PushConst
                stack[sp] = ins.imm_f32;
                sp += 1u;
            }
            case 1u: { // LoadBand
                stack[sp] = bvals[ins.imm_u32 & 7u];
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
            case 11u: { // Db
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
            case 20u: { // Clip (NaN-propagating, handles lo > hi)
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
            dst_expr[i] = nan_f32();
        } else {
            dst_expr[i] = stack[sp - 1u];
        }
    }
}
