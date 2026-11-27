const INV_LN10: f32 = 0.4342944819032518;

struct GpuInstr {
    opcode: u32,
    imm_u32: u32,
    imm_f32: f32,
    _pad: u32,
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
            case 31u: { // Mod (floored remainder matching Python/NumPy & CPU)
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                if (is_nan_f32(a) || is_nan_f32(b) || abs(b) < 1e-12) {
                    stack[sp - 1u] = nan_f32();
                } else {
                    let r = a - b * trunc(a / b);
                    stack[sp - 1u] = select(r, r + b, r != 0.0 && (r < 0.0) != (b < 0.0));
                }
            }
            case 32u: { // ModPow2 (floored remainder by constant power of two in imm_f32)
                let a = stack[sp - 1u];
                let c = ins.imm_f32;
                stack[sp - 1u] = a - c * floor(a / c);
            }
            case 33u: { // BitAnd
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(f32(i32(a) & i32(b)), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 34u: { // BitOr
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(f32(i32(a) | i32(b)), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 35u: { // BitNot
                let x = stack[sp - 1u];
                stack[sp - 1u] = select(select(0.0, 1.0, x == 0.0), nan_f32(), is_nan_f32(x));
            }
            case 36u: { // Log1p
                let x = stack[sp - 1u];
                if (!(x > -1.0)) {
                    stack[sp - 1u] = nan_f32();
                } else if (abs(x) < 1e-4) {
                    stack[sp - 1u] = x - 0.5 * x * x;
                } else {
                    stack[sp - 1u] = log(1.0 + x);
                }
            }
            case 37u: { // Expm1
                let x = stack[sp - 1u];
                if (is_nan_f32(x)) {
                    stack[sp - 1u] = nan_f32();
                } else if (abs(x) < 1e-4) {
                    stack[sp - 1u] = x + 0.5 * x * x;
                } else {
                    stack[sp - 1u] = exp(x) - 1.0;
                }
            }
            case 38u: { // Sign
                let x = stack[sp - 1u];
                if (is_nan_f32(x)) {
                    stack[sp - 1u] = nan_f32();
                } else if (x > 0.0) {
                    stack[sp - 1u] = 1.0;
                } else if (x < 0.0) {
                    stack[sp - 1u] = -1.0;
                } else {
                    stack[sp - 1u] = 0.0;
                }
            }
            case 39u: { // Arcsin
                let x = stack[sp - 1u];
                stack[sp - 1u] = select(nan_f32(), asin(x), x >= -1.0 && x <= 1.0);
            }
            case 40u: { // Arccos
                let x = stack[sp - 1u];
                stack[sp - 1u] = select(nan_f32(), acos(x), x >= -1.0 && x <= 1.0);
            }
            case 41u: { // Arctan
                stack[sp - 1u] = atan(stack[sp - 1u]);
            }
            case 42u: { // Arctan2
                sp -= 1u;
                let a = stack[sp - 1u];
                let b = stack[sp];
                stack[sp - 1u] = select(atan2(a, b), nan_f32(), is_nan_f32(a) || is_nan_f32(b));
            }
            case 43u: { // Sinh
                stack[sp - 1u] = sinh(stack[sp - 1u]);
            }
            case 44u: { // Cosh
                stack[sp - 1u] = cosh(stack[sp - 1u]);
            }
            case 45u: { // Tanh
                stack[sp - 1u] = tanh(stack[sp - 1u]);
            }
            case 46u: { // Trunc
                stack[sp - 1u] = trunc(stack[sp - 1u]);
            }
            default: {}
        }
    }

    if (sp == 0u) {
        return nan_f32();
    }
    return stack[sp - 1u];
}
