use crate::bandmath::fast_db_from_power;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    PushConst(f32),
    LoadBand(u8),
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
    Log10,
    Ln,
    Log2,
    Db,
    Exp,
    Sqrt,
    Abs,
    Min,
    Max,
    Clip,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    Where,
    Ceil,
    Floor,
    Round,
    Median(u8),
    Std(u8),
    Var(u8),
}

/// Compiled flat bytecode for any TiTiler-style band-math expression (e.g. `(b1-b2)/(b1+b2)`).
#[derive(Debug, Clone)]
pub struct ExprProgram {
    pub ops: Vec<Op>,
    /// Unique 1-based band indices referenced by the expression, in slot order `0..used_bands.len()`.
    pub used_bands: Vec<u16>,
    /// Peak VM stack depth computed at compile time (<= 32).
    pub max_depth: u8,
}

const MAX_PARSE_DEPTH: usize = 64;
const MAX_VM_STACK: usize = 32;
const MAX_EXPR_BANDS: usize = 16;

impl ExprProgram {
    pub fn compile(input: &str) -> Result<Self, String> {
        const MAX_EXPR_CHARS: usize = 4096;
        const MAX_EXPR_OPS: usize = 256;
        if input.len() > MAX_EXPR_CHARS {
            return Err(format!(
                "expression length ({}) exceeds limit ({MAX_EXPR_CHARS})",
                input.len()
            ));
        }
        let tokens = tokenize(input)?;
        let mut p = Parser {
            tokens: &tokens,
            pos: 0,
            depth: 0,
            ops: Vec::with_capacity(tokens.len()),
            used_bands: Vec::new(),
        };
        p.parse_expr()?;
        if p.pos < p.tokens.len() {
            return Err(format!("unexpected trailing token at position {}", p.pos));
        }
        if p.ops.len() > MAX_EXPR_OPS {
            return Err(format!(
                "expression has {} ops (max {MAX_EXPR_OPS})",
                p.ops.len()
            ));
        }
        if p.used_bands.len() > MAX_EXPR_BANDS {
            return Err(format!(
                "expression references {} distinct bands (max {MAX_EXPR_BANDS})",
                p.used_bands.len()
            ));
        }
        // Validate peak VM stack depth so eval() can never overflow its fixed 32-slot stack.
        let mut max_depth = 0usize;
        let mut depth = 0usize;
        for &op in &p.ops {
            match op {
                Op::PushConst(_) | Op::LoadBand(_) => {
                    depth += 1;
                    max_depth = max_depth.max(depth);
                }
                Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Pow
                | Op::Min
                | Op::Max
                | Op::Lt
                | Op::Le
                | Op::Gt
                | Op::Ge
                | Op::Eq
                | Op::Ne => {
                    depth = depth.saturating_sub(1);
                }
                Op::Clip | Op::Where => {
                    depth = depth.saturating_sub(2);
                }
                Op::Median(n) | Op::Std(n) | Op::Var(n) => {
                    depth = depth.saturating_sub((n as usize).saturating_sub(1));
                }
                _ => {}
            }
        }
        if max_depth > MAX_VM_STACK {
            return Err(format!(
                "expression nesting depth ({max_depth}) exceeds limit ({MAX_VM_STACK})"
            ));
        }
        Ok(Self {
            ops: p.ops,
            used_bands: p.used_bands,
            max_depth: max_depth as u8,
        })
    }

    /// Zero-alloc in-register stack VM evaluation for a single pixel's sampled band values.
    #[inline(always)]
    pub fn eval(&self, bands: &[f32]) -> f32 {
        let mut stack = [0.0f32; MAX_VM_STACK];
        let mut sp = 0usize;
        for &op in &self.ops {
            match op {
                Op::PushConst(c) => {
                    stack[sp] = c;
                    sp += 1;
                }
                Op::LoadBand(slot) => {
                    let v = bands.get(slot as usize).copied().unwrap_or(f32::NAN);
                    stack[sp] = v;
                    sp += 1;
                }
                Op::Add => {
                    sp -= 1;
                    stack[sp - 1] += stack[sp];
                }
                Op::Sub => {
                    sp -= 1;
                    stack[sp - 1] -= stack[sp];
                }
                Op::Mul => {
                    sp -= 1;
                    stack[sp - 1] *= stack[sp];
                }
                Op::Div => {
                    sp -= 1;
                    let d = stack[sp];
                    stack[sp - 1] = if d.abs() < 1e-12 {
                        f32::NAN
                    } else {
                        stack[sp - 1] / d
                    };
                }
                Op::Pow => {
                    sp -= 1;
                    let a = stack[sp - 1];
                    let b = stack[sp];
                    stack[sp - 1] = if a.is_nan() || b.is_nan() {
                        f32::NAN
                    } else {
                        a.powf(b)
                    };
                }
                Op::Neg => {
                    stack[sp - 1] = -stack[sp - 1];
                }
                Op::Log10 => {
                    let x = stack[sp - 1];
                    stack[sp - 1] = if x > 0.0 { x.log10() } else { f32::NAN };
                }
                Op::Ln => {
                    let x = stack[sp - 1];
                    stack[sp - 1] = if x > 0.0 { x.ln() } else { f32::NAN };
                }
                Op::Log2 => {
                    let x = stack[sp - 1];
                    stack[sp - 1] = if x > 0.0 { x.log2() } else { f32::NAN };
                }
                Op::Db => {
                    stack[sp - 1] = fast_db_from_power(stack[sp - 1]);
                }
                Op::Exp => {
                    stack[sp - 1] = stack[sp - 1].exp();
                }
                Op::Sqrt => {
                    let x = stack[sp - 1];
                    stack[sp - 1] = if x >= 0.0 { x.sqrt() } else { f32::NAN };
                }
                Op::Abs => {
                    stack[sp - 1] = stack[sp - 1].abs();
                }
                Op::Min => {
                    sp -= 1;
                    let a = stack[sp - 1];
                    let b = stack[sp];
                    stack[sp - 1] = if a.is_nan() || b.is_nan() {
                        f32::NAN
                    } else {
                        a.min(b)
                    };
                }
                Op::Max => {
                    sp -= 1;
                    let a = stack[sp - 1];
                    let b = stack[sp];
                    stack[sp - 1] = if a.is_nan() || b.is_nan() {
                        f32::NAN
                    } else {
                        a.max(b)
                    };
                }
                Op::Clip => {
                    let hi = stack[sp - 1];
                    let lo = stack[sp - 2];
                    sp -= 2;
                    let x = stack[sp - 1];
                    stack[sp - 1] = if x.is_nan() || lo.is_nan() || hi.is_nan() {
                        f32::NAN
                    } else {
                        let (a, b) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                        x.clamp(a, b)
                    };
                }
                Op::Lt => {
                    sp -= 1;
                    let a = stack[sp - 1];
                    let b = stack[sp];
                    stack[sp - 1] = if a.is_nan() || b.is_nan() { f32::NAN } else if a < b { 1.0 } else { 0.0 };
                }
                Op::Le => {
                    sp -= 1;
                    let a = stack[sp - 1];
                    let b = stack[sp];
                    stack[sp - 1] = if a.is_nan() || b.is_nan() { f32::NAN } else if a <= b { 1.0 } else { 0.0 };
                }
                Op::Gt => {
                    sp -= 1;
                    let a = stack[sp - 1];
                    let b = stack[sp];
                    stack[sp - 1] = if a.is_nan() || b.is_nan() { f32::NAN } else if a > b { 1.0 } else { 0.0 };
                }
                Op::Ge => {
                    sp -= 1;
                    let a = stack[sp - 1];
                    let b = stack[sp];
                    stack[sp - 1] = if a.is_nan() || b.is_nan() { f32::NAN } else if a >= b { 1.0 } else { 0.0 };
                }
                Op::Eq => {
                    sp -= 1;
                    let a = stack[sp - 1];
                    let b = stack[sp];
                    stack[sp - 1] = if a.is_nan() || b.is_nan() { f32::NAN } else if a == b { 1.0 } else { 0.0 };
                }
                Op::Ne => {
                    sp -= 1;
                    let a = stack[sp - 1];
                    let b = stack[sp];
                    stack[sp - 1] = if a.is_nan() || b.is_nan() { f32::NAN } else if a != b { 1.0 } else { 0.0 };
                }
                Op::Where => {
                    let b = stack[sp - 1];
                    let a = stack[sp - 2];
                    sp -= 2;
                    let cond = stack[sp - 1];
                    stack[sp - 1] = if cond.is_nan() {
                        f32::NAN
                    } else if cond != 0.0 {
                        a
                    } else {
                        b
                    };
                }
                Op::Ceil => {
                    stack[sp - 1] = stack[sp - 1].ceil();
                }
                Op::Floor => {
                    stack[sp - 1] = stack[sp - 1].floor();
                }
                Op::Round => {
                    stack[sp - 1] = stack[sp - 1].round();
                }
                Op::Median(n) => {
                    let k = n as usize;
                    let base = sp - k;
                    let mut tmp = [0.0f32; MAX_EXPR_BANDS];
                    let mut has_nan = false;
                    for i in 0..k {
                        let v = stack[base + i];
                        if v.is_nan() {
                            has_nan = true;
                            break;
                        }
                        tmp[i] = v;
                    }
                    sp = base + 1;
                    stack[base] = if has_nan || k == 0 {
                        f32::NAN
                    } else {
                        let slice = &mut tmp[..k];
                        slice.sort_unstable_by(f32::total_cmp);
                        if k % 2 == 1 {
                            slice[k / 2]
                        } else {
                            0.5 * (slice[k / 2 - 1] + slice[k / 2])
                        }
                    };
                }
                Op::Var(n) | Op::Std(n) => {
                    let k = n as usize;
                    let base = sp - k;
                    let mut sum = 0.0f64;
                    let mut has_nan = false;
                    for i in 0..k {
                        let v = stack[base + i];
                        if v.is_nan() {
                            has_nan = true;
                            break;
                        }
                        sum += v as f64;
                    }
                    sp = base + 1;
                    stack[base] = if has_nan || k == 0 {
                        f32::NAN
                    } else if k == 1 {
                        0.0
                    } else {
                        let mean = sum / (k as f64);
                        let mut sq = 0.0f64;
                        for i in 0..k {
                            let d = stack[base + i] as f64 - mean;
                            sq += d * d;
                        }
                        let var = sq / ((k - 1) as f64);
                        if matches!(op, Op::Std(_)) {
                            var.sqrt() as f32
                        } else {
                            var as f32
                        }
                    };
                }
            }
        }
        if sp > 0 { stack[sp - 1] } else { f32::NAN }
    }

    /// Scanline-vectorized bytecode VM evaluation across `N` contiguous pixels (`[f32; N]`),
    /// reusing a caller-provided scratch stack buffer to avoid zeroing 8–32 KiB per scanline.
    pub fn eval_row_with_scratch<const N: usize>(
        &self,
        band_rows: &[[f32; N]],
        scratch: &mut Vec<[f32; N]>,
        out: &mut [f32; N],
    ) {
        const MAX_STACK: usize = 16;
        let depth = self.max_depth as usize;
        if depth > MAX_STACK || self.ops.is_empty() {
            let mut vals = [0.0f32; 16];
            let nb = band_rows.len().min(16);
            for px in 0..N {
                for b in 0..nb {
                    vals[b] = band_rows[b][px];
                }
                out[px] = self.eval(&vals[..nb]);
            }
            return;
        }
        let needed = depth.max(1);
        if scratch.len() < needed {
            scratch.resize(needed, [0.0f32; N]);
        }
        self.eval_row_with_slice(band_rows, &mut scratch[..needed], out);
    }

    /// Scanline-vectorized bytecode VM evaluation across `N` contiguous pixels (`[f32; N]`).
    /// Uses a compact 4-slot stack when `max_depth <= 4` (NDVI, NDWI, EVI, etc. — 4× less stack
    /// zeroing per scanline), falling back to 16 slots for deeper expressions.
    pub fn eval_row<const N: usize>(&self, band_rows: &[[f32; N]], out: &mut [f32; N]) {
        const MAX_STACK: usize = 16;
        if (self.max_depth as usize) > MAX_STACK || self.ops.is_empty() {
            let mut vals = [0.0f32; 16];
            let nb = band_rows.len().min(16);
            for px in 0..N {
                for b in 0..nb {
                    vals[b] = band_rows[b][px];
                }
                out[px] = self.eval(&vals[..nb]);
            }
            return;
        }
        if self.max_depth <= 4 {
            let mut stack = [[0.0f32; N]; 4];
            self.eval_row_with_slice(band_rows, &mut stack, out);
        } else {
            let mut stack = [[0.0f32; N]; MAX_STACK];
            self.eval_row_with_slice(band_rows, &mut stack, out);
        }
    }

    #[inline(always)]
    fn eval_row_with_slice<const N: usize>(
        &self,
        band_rows: &[[f32; N]],
        stack: &mut [[f32; N]],
        out: &mut [f32; N],
    ) {
        let mut sp = 0usize;

        for &op in &self.ops {
            match op {
                Op::PushConst(c) => {
                    stack[sp].fill(c);
                    sp += 1;
                }
                Op::LoadBand(slot) => {
                    if let Some(row) = band_rows.get(slot as usize) {
                        stack[sp] = *row;
                    } else {
                        stack[sp].fill(f32::NAN);
                    }
                    sp += 1;
                }
                Op::Add => {
                    sp -= 1;
                    let (lo, hi) = stack.split_at_mut(sp);
                    let lhs = &mut lo[sp - 1];
                    let rhs = &hi[0];
                    for i in 0..N {
                        lhs[i] += rhs[i];
                    }
                }
                Op::Sub => {
                    sp -= 1;
                    let (lo, hi) = stack.split_at_mut(sp);
                    let lhs = &mut lo[sp - 1];
                    let rhs = &hi[0];
                    for i in 0..N {
                        lhs[i] -= rhs[i];
                    }
                }
                Op::Mul => {
                    sp -= 1;
                    let (lo, hi) = stack.split_at_mut(sp);
                    let lhs = &mut lo[sp - 1];
                    let rhs = &hi[0];
                    for i in 0..N {
                        lhs[i] *= rhs[i];
                    }
                }
                Op::Div => {
                    sp -= 1;
                    let (lo, hi) = stack.split_at_mut(sp);
                    let lhs = &mut lo[sp - 1];
                    let rhs = &hi[0];
                    for i in 0..N {
                        let d = rhs[i];
                        lhs[i] = if d.abs() < 1e-12 { f32::NAN } else { lhs[i] / d };
                    }
                }
                Op::Pow => {
                    sp -= 1;
                    let (lo, hi) = stack.split_at_mut(sp);
                    let lhs = &mut lo[sp - 1];
                    let rhs = &hi[0];
                    for i in 0..N {
                        let a = lhs[i];
                        let b = rhs[i];
                        lhs[i] = if a.is_nan() || b.is_nan() {
                            f32::NAN
                        } else {
                            a.powf(b)
                        };
                    }
                }
                Op::Neg => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        s[i] = -s[i];
                    }
                }
                Op::Log10 => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        let x = s[i];
                        s[i] = if x > 0.0 { x.log10() } else { f32::NAN };
                    }
                }
                Op::Ln => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        let x = s[i];
                        s[i] = if x > 0.0 { x.ln() } else { f32::NAN };
                    }
                }
                Op::Log2 => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        let x = s[i];
                        s[i] = if x > 0.0 { x.log2() } else { f32::NAN };
                    }
                }
                Op::Db => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        s[i] = fast_db_from_power(s[i]);
                    }
                }
                Op::Exp => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        s[i] = s[i].exp();
                    }
                }
                Op::Sqrt => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        let x = s[i];
                        s[i] = if x >= 0.0 { x.sqrt() } else { f32::NAN };
                    }
                }
                Op::Abs => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        s[i] = s[i].abs();
                    }
                }
                Op::Min => {
                    sp -= 1;
                    let (lo, hi) = stack.split_at_mut(sp);
                    let lhs = &mut lo[sp - 1];
                    let rhs = &hi[0];
                    for i in 0..N {
                        let a = lhs[i];
                        let b = rhs[i];
                        lhs[i] = if a.is_nan() || b.is_nan() { f32::NAN } else { a.min(b) };
                    }
                }
                Op::Max => {
                    sp -= 1;
                    let (lo, hi) = stack.split_at_mut(sp);
                    let lhs = &mut lo[sp - 1];
                    let rhs = &hi[0];
                    for i in 0..N {
                        let a = lhs[i];
                        let b = rhs[i];
                        lhs[i] = if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) };
                    }
                }
                Op::Clip => {
                    let (lo_part, hi_part) = stack.split_at_mut(sp - 2);
                    let target = &mut lo_part[sp - 3];
                    let lo_v = &hi_part[0];
                    let hi_v = &hi_part[1];
                    for i in 0..N {
                        let x = target[i];
                        let a = lo_v[i];
                        let b = hi_v[i];
                        target[i] = if x.is_nan() || a.is_nan() || b.is_nan() {
                            f32::NAN
                        } else {
                            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                            x.clamp(lo, hi)
                        };
                    }
                    sp -= 2;
                }
                Op::Lt | Op::Le | Op::Gt | Op::Ge | Op::Eq | Op::Ne => {
                    sp -= 1;
                    let (lo, hi) = stack.split_at_mut(sp);
                    let lhs = &mut lo[sp - 1];
                    let rhs = &hi[0];
                    for i in 0..N {
                        let a = lhs[i];
                        let b = rhs[i];
                        lhs[i] = if a.is_nan() || b.is_nan() {
                            f32::NAN
                        } else {
                            let ok = match op {
                                Op::Lt => a < b,
                                Op::Le => a <= b,
                                Op::Gt => a > b,
                                Op::Ge => a >= b,
                                Op::Eq => a == b,
                                Op::Ne => a != b,
                                _ => unreachable!(),
                            };
                            if ok { 1.0 } else { 0.0 }
                        };
                    }
                }
                Op::Where => {
                    let (lo_part, hi_part) = stack.split_at_mut(sp - 2);
                    let cond = &mut lo_part[sp - 3];
                    let a_v = &hi_part[0];
                    let b_v = &hi_part[1];
                    for i in 0..N {
                        let c = cond[i];
                        cond[i] = if c.is_nan() {
                            f32::NAN
                        } else if c != 0.0 {
                            a_v[i]
                        } else {
                            b_v[i]
                        };
                    }
                    sp -= 2;
                }
                Op::Ceil => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        s[i] = s[i].ceil();
                    }
                }
                Op::Floor => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        s[i] = s[i].floor();
                    }
                }
                Op::Round => {
                    let s = &mut stack[sp - 1];
                    for i in 0..N {
                        s[i] = s[i].round();
                    }
                }
                Op::Median(n) => {
                    let k = n as usize;
                    let base = sp - k;
                    let mut tmp = [0.0f32; MAX_EXPR_BANDS];
                    for i in 0..N {
                        let mut has_nan = false;
                        for j in 0..k {
                            let v = stack[base + j][i];
                            if v.is_nan() {
                                has_nan = true;
                                break;
                            }
                            tmp[j] = v;
                        }
                        stack[base][i] = if has_nan || k == 0 {
                            f32::NAN
                        } else {
                            let slice = &mut tmp[..k];
                            slice.sort_unstable_by(f32::total_cmp);
                            if k % 2 == 1 {
                                slice[k / 2]
                            } else {
                                0.5 * (slice[k / 2 - 1] + slice[k / 2])
                            }
                        };
                    }
                    sp = base + 1;
                }
                Op::Var(n) | Op::Std(n) => {
                    let k = n as usize;
                    let base = sp - k;
                    let is_std = matches!(op, Op::Std(_));
                    for i in 0..N {
                        let mut sum = 0.0f64;
                        let mut has_nan = false;
                        for j in 0..k {
                            let v = stack[base + j][i];
                            if v.is_nan() {
                                has_nan = true;
                                break;
                            }
                            sum += v as f64;
                        }
                        stack[base][i] = if has_nan || k == 0 {
                            f32::NAN
                        } else if k == 1 {
                            0.0
                        } else {
                            let mean = sum / (k as f64);
                            let mut sq = 0.0f64;
                            for j in 0..k {
                                let d = stack[base + j][i] as f64 - mean;
                                sq += d * d;
                            }
                            let var = sq / ((k - 1) as f64);
                            if is_std { var.sqrt() as f32 } else { var as f32 }
                        };
                    }
                    sp = base + 1;
                }
            }
        }

        if sp > 0 {
            *out = stack[sp - 1];
        } else {
            out.fill(f32::NAN);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f32),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    BangEq,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    out.push(Tok::Caret);
                    i += 2;
                } else {
                    out.push(Tok::Star);
                    i += 1;
                }
            }
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            }
            '^' => {
                out.push(Tok::Caret);
                i += 1;
            }
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            '<' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Tok::Le);
                    i += 2;
                } else {
                    out.push(Tok::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Tok::Ge);
                    i += 2;
                } else {
                    out.push(Tok::Gt);
                    i += 1;
                }
            }
            '=' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Tok::EqEq);
                    i += 2;
                } else {
                    return Err("expected '==' in expression".to_string());
                }
            }
            '!' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Tok::BangEq);
                    i += 2;
                } else {
                    return Err("expected '!=' in expression".to_string());
                }
            }
            '0'..='9' | '.' => {
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_digit() || bytes[i] == b'.' || bytes[i] == b'e' || bytes[i] == b'E')
                {
                    if (bytes[i] == b'e' || bytes[i] == b'E')
                        && i + 1 < bytes.len()
                        && (bytes[i + 1] == b'+' || bytes[i + 1] == b'-')
                    {
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                let val = s[start..i]
                    .parse::<f32>()
                    .map_err(|_| format!("invalid number '{}'", &s[start..i]))?;
                out.push(Tok::Num(val));
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                }
                out.push(Tok::Ident(s[start..i].to_ascii_lowercase()));
            }
            other => return Err(format!("unexpected character '{other}' in expression")),
        }
    }
    Ok(out)
}

struct Parser<'a> {
    tokens: &'a [Tok],
    pos: usize,
    depth: usize,
    ops: Vec<Op>,
    used_bands: Vec<u16>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(format!(
                "expression nesting exceeds maximum depth ({MAX_PARSE_DEPTH})"
            ));
        }
        Ok(())
    }

    fn band_slot(&mut self, band_1based: u16) -> u8 {
        if let Some(idx) = self.used_bands.iter().position(|&b| b == band_1based) {
            idx as u8
        } else {
            let idx = self.used_bands.len() as u8;
            self.used_bands.push(band_1based);
            idx
        }
    }

    fn parse_expr(&mut self) -> Result<(), String> {
        self.enter()?;
        let res = (|| {
            self.parse_add_sub()?;
            while let Some(tok) = self.peek().cloned() {
                let op = match tok {
                    Tok::Lt => Op::Lt,
                    Tok::Le => Op::Le,
                    Tok::Gt => Op::Gt,
                    Tok::Ge => Op::Ge,
                    Tok::EqEq => Op::Eq,
                    Tok::BangEq => Op::Ne,
                    _ => break,
                };
                self.pos += 1;
                self.parse_add_sub()?;
                self.ops.push(op);
            }
            Ok(())
        })();
        self.depth -= 1;
        res
    }

    fn parse_add_sub(&mut self) -> Result<(), String> {
        self.parse_term()?;
        while let Some(tok) = self.peek().cloned() {
            match tok {
                Tok::Plus => {
                    self.pos += 1;
                    self.parse_term()?;
                    self.ops.push(Op::Add);
                }
                Tok::Minus => {
                    self.pos += 1;
                    self.parse_term()?;
                    self.ops.push(Op::Sub);
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn parse_term(&mut self) -> Result<(), String> {
        self.parse_unary()?;
        while let Some(tok) = self.peek().cloned() {
            match tok {
                Tok::Star => {
                    self.pos += 1;
                    self.parse_unary()?;
                    self.ops.push(Op::Mul);
                }
                Tok::Slash => {
                    self.pos += 1;
                    self.parse_unary()?;
                    self.ops.push(Op::Div);
                }
                _ => break,
            }
        }
        Ok(())
    }

    // Standard math/Python precedence: `-x^2 == -(x^2)` and `2^-3 == 2^(-3)`.
    fn parse_unary(&mut self) -> Result<(), String> {
        self.enter()?;
        let res = match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                self.parse_unary()?;
                self.ops.push(Op::Neg);
                Ok(())
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                self.parse_unary()
            }
            _ => self.parse_power(),
        };
        self.depth -= 1;
        res
    }

    fn parse_power(&mut self) -> Result<(), String> {
        self.parse_primary()?;
        if matches!(self.peek(), Some(Tok::Caret)) {
            self.pos += 1;
            self.parse_unary()?;
            self.ops.push(Op::Pow);
        }
        Ok(())
    }

    fn parse_primary(&mut self) -> Result<(), String> {
        let tok = self
            .peek()
            .cloned()
            .ok_or_else(|| "unexpected end of expression".to_string())?;
        match tok {
            Tok::Num(n) => {
                self.pos += 1;
                self.ops.push(Op::PushConst(n));
                Ok(())
            }
            Tok::LParen => {
                self.pos += 1;
                self.parse_expr()?;
                if !matches!(self.peek(), Some(Tok::RParen)) {
                    return Err("expected ')'".to_string());
                }
                self.pos += 1;
                Ok(())
            }
            Tok::Ident(name) => {
                self.pos += 1;
                if matches!(self.peek(), Some(Tok::LParen)) {
                    // Function call
                    self.pos += 1;
                    let mut argc = 0usize;
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        loop {
                            self.parse_expr()?;
                            argc += 1;
                            if matches!(self.peek(), Some(Tok::Comma)) {
                                self.pos += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        return Err(format!("expected ')' after {name}(...)"));
                    }
                    self.pos += 1;
                    let op = match (name.as_str(), argc) {
                        ("log10", 1) => Op::Log10,
                        ("ln" | "log", 1) => Op::Ln,
                        ("log2", 1) => Op::Log2,
                        ("db", 1) => Op::Db,
                        ("exp", 1) => Op::Exp,
                        ("sqrt", 1) => Op::Sqrt,
                        ("abs", 1) => Op::Abs,
                        ("ceil", 1) => Op::Ceil,
                        ("floor", 1) => Op::Floor,
                        ("round", 1) => Op::Round,
                        ("min" | "minimum", 2) => Op::Min,
                        ("max" | "maximum", 2) => Op::Max,
                        ("clip", 3) => Op::Clip,
                        ("where", 3) => Op::Where,
                        ("median", n) if (1..=MAX_EXPR_BANDS).contains(&n) => Op::Median(n as u8),
                        ("std", n) if (1..=MAX_EXPR_BANDS).contains(&n) => Op::Std(n as u8),
                        ("var", n) if (1..=MAX_EXPR_BANDS).contains(&n) => Op::Var(n as u8),
                        _ => return Err(format!("unknown function '{name}' with {argc} args")),
                    };
                    self.ops.push(op);
                    Ok(())
                } else if let Some(rest) = name.strip_prefix('b') {
                    let b = rest
                        .parse::<u16>()
                        .map_err(|_| format!("invalid band identifier '{name}'"))?;
                    if b == 0 {
                        return Err("band indices are 1-based (b1, b2, ...)".to_string());
                    }
                    let slot = self.band_slot(b);
                    self.ops.push(Op::LoadBand(slot));
                    Ok(())
                } else {
                    Err(format!("unknown identifier '{name}'"))
                }
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_and_evaluates_titiler_expressions() {
        let prog = ExprProgram::compile("(b4 - b3) / (b4 + b3)").unwrap();
        assert_eq!(prog.used_bands, vec![4, 3]);
        let v = prog.eval(&[0.6, 0.2]);
        assert!((v - 0.5).abs() < 1e-6);

        let evi = ExprProgram::compile("2.5 * (b4 - b3) / (b4 + 6 * b3 - 7.5 * b1 + 1)").unwrap();
        assert_eq!(evi.used_bands, vec![4, 3, 1]);
        let ev = evi.eval(&[0.5, 0.1, 0.05]);
        let expected = 2.5 * (0.5 - 0.1) / (0.5 + 0.6 - 0.375 + 1.0);
        assert!((ev - expected).abs() < 1e-5);

        let db = ExprProgram::compile("10 * log10(b1)").unwrap();
        assert!((db.eval(&[0.01]) - (-20.0)).abs() < 1e-4);
    }

    #[test]
    fn expr_vm_safety_precedence_and_nan_propagation() {
        // -b1^2 must be -(b1^2) = -9, not (-3)^2 = +9; 2^-3 must be 0.125
        let p = ExprProgram::compile("-b1^2").unwrap();
        assert!((p.eval(&[3.0]) - (-9.0)).abs() < 1e-6);
        let p2 = ExprProgram::compile("2^-3").unwrap();
        assert!((p2.eval(&[]) - 0.125).abs() < 1e-6);

        // clip with inverted bounds (lo > hi) must not panic
        let pc = ExprProgram::compile("clip(b1, 10, 5)").unwrap();
        assert!((pc.eval(&[7.0]) - 7.0).abs() < 1e-6);
        assert!((pc.eval(&[2.0]) - 5.0).abs() < 1e-6);

        // Comparisons, where(cond, a, b), ceil/floor/round (scalar & row parity)
        let pw = ExprProgram::compile("where(b1 > 0.2, ceil(b1 * 10), floor(b2))").unwrap();
        assert!((pw.eval(&[0.25, 3.9]) - 3.0).abs() < 1e-6);
        assert!((pw.eval(&[0.1, 3.9]) - 3.0).abs() < 1e-6);
        assert!((pw.eval(&[0.1, 4.9]) - 4.0).abs() < 1e-6);
        let row_b1 = [0.25f32, 0.1, 0.5, f32::NAN];
        let row_b2 = [3.9f32, 4.9, 1.2, 2.0];
        let mut out = [0.0f32; 4];
        let mut scratch = Vec::new();
        pw.eval_row_with_scratch(&[row_b1, row_b2], &mut scratch, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
        assert!((out[2] - 5.0).abs() < 1e-6);
        assert!(out[3].is_nan());

        let pr = ExprProgram::compile("where(b1 == 2, round(b2), b1 != 3)").unwrap();
        assert!((pr.eval(&[2.0, 2.6]) - 3.0).abs() < 1e-6);
        assert!((pr.eval(&[3.0, 2.6]) - 0.0).abs() < 1e-6);
        assert!((pr.eval(&[4.0, 2.6]) - 1.0).abs() < 1e-6);

        // min/max/pow must propagate NaN instead of swallowing it
        let pm = ExprProgram::compile("min(b1 / (b1 - b1), 5.0)").unwrap();
        assert!(pm.eval(&[1.0]).is_nan());

        // >16 distinct bands must be rejected at compile time
        let many_bands = (1..=17).map(|i| format!("b{i}")).collect::<Vec<_>>().join("+");
        assert!(ExprProgram::compile(&many_bands).is_err());

        // Deep nesting must be rejected cleanly instead of overflowing stack / VM
        let deep = format!("{}1{}", "(".repeat(100), ")".repeat(100));
        assert!(ExprProgram::compile(&deep).is_err());
    }

    #[test]
    fn eval_row_matches_scalar_eval_across_all_ops() {
        let exprs = [
            "(b1 - b2) / (b1 + b2)",
            "2.5 * (b1 - b2) / (b1 + 6 * b2 - 7.5 * b3 + 1)",
            "clip(db(b1) - min(b2, max(b3, 0.1)), -30, 10)",
            "sqrt(abs(b1 - b2)) + exp(-b3)",
            "median(b1, b2, b3, b1 + b2)",
            "std(b1, b2, b3)",
            "var(b1, b2, b3)",
            "std(b1)",
        ];
        // Verify exact 4-band median (differs from mean!) and 1-band std NaN masking
        let med4 = ExprProgram::compile("median(b1, b2, b3, b4)").unwrap();
        assert!((med4.eval(&[1.0, 2.0, 3.0, 100.0]) - 2.5).abs() < 1e-6);
        assert!(med4.eval(&[1.0, f32::NAN, 3.0, 4.0]).is_nan());
        let std1 = ExprProgram::compile("std(b1)").unwrap();
        assert_eq!(std1.eval(&[42.0]), 0.0);
        assert!(std1.eval(&[f32::NAN]).is_nan());
        let mut rows = [[0.0f32; 256]; 3];
        for i in 0..256 {
            rows[0][i] = if i % 17 == 0 { f32::NAN } else if i == 5 { 0.0 } else { (i as f32 + 1.0) * 0.01 };
            rows[1][i] = if i % 23 == 0 { f32::NAN } else if i == 5 { 0.0 } else { (256 - i) as f32 * 0.005 };
            rows[2][i] = (i as f32) * 0.002 + 0.01;
        }
        for expr in exprs {
            let prog = ExprProgram::compile(expr).unwrap();
            let nb = prog.used_bands.len();
            let mut out = [0.0f32; 256];
            prog.eval_row::<256>(&rows[..nb], &mut out);
            for i in 0..256 {
                let mut vals = [0.0f32; 3];
                for b in 0..nb {
                    vals[b] = rows[b][i];
                }
                let expected = prog.eval(&vals[..nb]);
                if expected.is_nan() {
                    assert!(out[i].is_nan(), "expr={expr} i={i} expected NaN, got {}", out[i]);
                } else {
                    assert!(
                        (out[i] - expected).abs() < 1e-5,
                        "expr={expr} i={i} mismatch: row={} vs scalar={}",
                        out[i],
                        expected
                    );
                }
            }
        }
    }
}
