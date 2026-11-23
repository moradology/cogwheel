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
}

/// Compiled flat bytecode for any TiTiler-style band-math expression (e.g. `(b1-b2)/(b1+b2)`).
#[derive(Debug, Clone)]
pub struct ExprProgram {
    pub ops: Vec<Op>,
    /// Unique 1-based band indices referenced by the expression, in slot order `0..used_bands.len()`.
    pub used_bands: Vec<u16>,
}

impl ExprProgram {
    pub fn compile(input: &str) -> Result<Self, String> {
        let tokens = tokenize(input)?;
        let mut p = Parser {
            tokens: &tokens,
            pos: 0,
            ops: Vec::with_capacity(tokens.len()),
            used_bands: Vec::new(),
        };
        p.parse_expr()?;
        if p.pos < p.tokens.len() {
            return Err(format!("unexpected trailing token at position {}", p.pos));
        }
        Ok(Self {
            ops: p.ops,
            used_bands: p.used_bands,
        })
    }

    /// Zero-alloc in-register stack VM evaluation for a single pixel's sampled band values.
    #[inline(always)]
    pub fn eval(&self, bands: &[f32]) -> f32 {
        let mut stack = [0.0f32; 32];
        let mut sp = 0usize;
        for &op in &self.ops {
            match op {
                Op::PushConst(c) => {
                    stack[sp] = c;
                    sp += 1;
                }
                Op::LoadBand(slot) => {
                    let v = unsafe { *bands.get_unchecked(slot as usize) };
                    if v.is_nan() {
                        return f32::NAN;
                    }
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
                    stack[sp - 1] = stack[sp - 1].powf(stack[sp]);
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
                    stack[sp - 1] = stack[sp - 1].min(stack[sp]);
                }
                Op::Max => {
                    sp -= 1;
                    stack[sp - 1] = stack[sp - 1].max(stack[sp]);
                }
                Op::Clip => {
                    let hi = stack[sp - 1];
                    let lo = stack[sp - 2];
                    sp -= 2;
                    stack[sp - 1] = stack[sp - 1].clamp(lo, hi);
                }
            }
        }
        if sp > 0 { stack[sp - 1] } else { f32::NAN }
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
    ops: Vec<Op>,
    used_bands: Vec<u16>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
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
        self.parse_power()?;
        while let Some(tok) = self.peek().cloned() {
            match tok {
                Tok::Star => {
                    self.pos += 1;
                    self.parse_power()?;
                    self.ops.push(Op::Mul);
                }
                Tok::Slash => {
                    self.pos += 1;
                    self.parse_power()?;
                    self.ops.push(Op::Div);
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn parse_power(&mut self) -> Result<(), String> {
        self.parse_unary()?;
        if matches!(self.peek(), Some(Tok::Caret)) {
            self.pos += 1;
            self.parse_power()?;
            self.ops.push(Op::Pow);
        }
        Ok(())
    }

    fn parse_unary(&mut self) -> Result<(), String> {
        match self.peek() {
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
            _ => self.parse_primary(),
        }
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
                        ("min" | "minimum", 2) => Op::Min,
                        ("max" | "maximum", 2) => Op::Max,
                        ("clip", 3) => Op::Clip,
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
}
