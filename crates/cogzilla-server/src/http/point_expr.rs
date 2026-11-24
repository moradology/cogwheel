//! TiTiler's band math on point values: expressions over `b1..bn`, the stacked bands of the
//! assets that were asked for, evaluated in double precision as numexpr does.
//!
//! `b1+b2;b2-b1` gives two values. Each result is described by the expression with every
//! `bN` replaced by that band's description (`red_b1+nir_b1`).

/// Evaluate `expression` over one pixel. A band with no value makes every result that
/// touches it empty.
pub(super) fn apply(
    expression: &str,
    values: &[Option<f64>],
    descriptions: &[String],
) -> Result<(Vec<Option<f64>>, Vec<String>), String> {
    let mut out = Vec::new();
    let mut described = Vec::new();
    for part in expression.split(';').map(str::trim).filter(|p| !p.is_empty()) {
        let mut parser = Parser { src: part.as_bytes(), pos: 0, values };
        let v = parser.sum()?;
        parser.skip_space();
        if parser.pos != parser.src.len() {
            return Err(format!("Invalid expression '{part}'"));
        }
        out.push(v.is_finite().then_some(v));
        described.push(describe(part, descriptions));
    }
    if out.is_empty() {
        return Err("Invalid expression ''".into());
    }
    Ok((out, described))
}

/// `b3` -> `Some(2)`.
fn band_index(ident: &str) -> Option<usize> {
    let n: usize = ident.strip_prefix('b')?.parse().ok()?;
    n.checked_sub(1)
}

fn describe(part: &str, descriptions: &[String]) -> String {
    let bytes = part.as_bytes();
    let mut out = String::with_capacity(part.len() * 2);
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let ident = &part[start..i];
            match band_index(ident).and_then(|n| descriptions.get(n)) {
                Some(d) => out.push_str(d),
                None => out.push_str(ident),
            }
        } else {
            // Not an identifier start: copy one whole character.
            let ch = part[i..].chars().next().unwrap_or(' ');
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    values: &'a [Option<f64>],
}

impl Parser<'_> {
    fn skip_space(&mut self) {
        while self.src.get(self.pos).is_some_and(u8::is_ascii_whitespace) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_space();
        if self.src[self.pos..].starts_with(token.as_bytes()) {
            self.pos += token.len();
            true
        } else {
            false
        }
    }

    fn sum(&mut self) -> Result<f64, String> {
        let mut v = self.product()?;
        loop {
            if self.eat("+") {
                v += self.product()?;
            } else if self.eat("-") {
                v -= self.product()?;
            } else {
                return Ok(v);
            }
        }
    }

    fn product(&mut self) -> Result<f64, String> {
        let mut v = self.unary()?;
        loop {
            self.skip_space();
            // `**` belongs to the power rule, not to multiplication.
            if self.src[self.pos..].starts_with(b"**") {
                return Ok(v);
            }
            if self.eat("*") {
                v *= self.unary()?;
            } else if self.eat("/") {
                v /= self.unary()?;
            } else if self.eat("%") {
                v %= self.unary()?;
            } else {
                return Ok(v);
            }
        }
    }

    fn unary(&mut self) -> Result<f64, String> {
        if self.eat("-") {
            return Ok(-self.unary()?);
        }
        if self.eat("+") {
            return self.unary();
        }
        let base = self.primary()?;
        if self.eat("**") {
            // Right-associative, and binds tighter than a leading minus on its right.
            return Ok(base.powf(self.unary()?));
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<f64, String> {
        self.skip_space();
        let start = self.pos;
        let Some(&c) = self.src.get(self.pos) else {
            return Err("Invalid expression: unexpected end".into());
        };
        if c == b'(' {
            self.pos += 1;
            let v = self.sum()?;
            if !self.eat(")") {
                return Err("Invalid expression: missing ')'".into());
            }
            return Ok(v);
        }
        if c.is_ascii_digit() || c == b'.' {
            while self.src.get(self.pos).is_some_and(|b| b.is_ascii_digit() || *b == b'.') {
                self.pos += 1;
            }
            // An exponent: 1e-3, 2E5.
            if self.src.get(self.pos).is_some_and(|b| matches!(b, b'e' | b'E')) {
                let mark = self.pos;
                self.pos += 1;
                if self.src.get(self.pos).is_some_and(|b| matches!(b, b'+' | b'-')) {
                    self.pos += 1;
                }
                if self.src.get(self.pos).is_some_and(u8::is_ascii_digit) {
                    while self.src.get(self.pos).is_some_and(u8::is_ascii_digit) {
                        self.pos += 1;
                    }
                } else {
                    self.pos = mark;
                }
            }
            let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
            return text.parse::<f64>().map_err(|_| format!("Invalid number '{text}'"));
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            while self.src.get(self.pos).is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_') {
                self.pos += 1;
            }
            let ident = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
            if self.eat("(") {
                let arg = self.sum()?;
                if !self.eat(")") {
                    return Err("Invalid expression: missing ')'".into());
                }
                return match ident {
                    "sqrt" => Ok(arg.sqrt()),
                    "abs" => Ok(arg.abs()),
                    "exp" => Ok(arg.exp()),
                    "log" => Ok(arg.ln()),
                    "log10" => Ok(arg.log10()),
                    "sin" => Ok(arg.sin()),
                    "cos" => Ok(arg.cos()),
                    "tan" => Ok(arg.tan()),
                    "arcsin" => Ok(arg.asin()),
                    "arccos" => Ok(arg.acos()),
                    "arctan" => Ok(arg.atan()),
                    "floor" => Ok(arg.floor()),
                    "ceil" => Ok(arg.ceil()),
                    other => Err(format!("function '{other}' is not supported in point expressions")),
                };
            }
            return match band_index(ident).and_then(|n| self.values.get(n)) {
                Some(v) => Ok(v.unwrap_or(f64::NAN)),
                None => Err(format!("Invalid band/asset name '{ident}'")),
            };
        }
        Err(format!("Invalid expression: unexpected '{}'", c as char))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(expr: &str, values: &[Option<f64>]) -> Result<(Vec<Option<f64>>, Vec<String>), String> {
        let names: Vec<String> = ["red_b1", "nir_b1", "blue_b1"].iter().map(|s| s.to_string()).collect();
        apply(expr, values, &names[..values.len().min(3)])
    }

    #[test]
    fn band_math_matches_titiler() {
        // Values and descriptions titiler-pgstac returned for red=1060, nir=2386.
        let v = [Some(1060.0), Some(2386.0)];
        let (out, names) = run("(b2-b1)/(b2+b1)", &v).unwrap();
        assert_eq!(out, [Some(0.3847939640162507)]);
        assert_eq!(names, ["(nir_b1-red_b1)/(nir_b1+red_b1)"]);
        let (out, names) = run("b1+b2;b2-b1", &v).unwrap();
        assert_eq!(out, [Some(3446.0), Some(1326.0)]);
        assert_eq!(names, ["red_b1+nir_b1", "nir_b1-red_b1"]);
        assert_eq!(run("b1*2", &v).unwrap().0, [Some(2120.0)]);
    }

    #[test]
    fn precedence_functions_and_numbers() {
        let v = [Some(2.0), Some(3.0), Some(4.0)];
        let one = |e: &str| run(e, &v).unwrap().0[0].unwrap();
        assert_eq!(one("b1 + b2 * b3"), 14.0);
        assert_eq!(one("-b1 ** 2"), -4.0);
        assert_eq!(one("b1 ** b2 ** 2"), 512.0);
        assert_eq!(one("(b1 + b2) * -b3"), -20.0);
        assert_eq!(one("sqrt(b3) + abs(-b1) + 1e1 + .5"), 14.5);
        assert_eq!(one("b3 % b2"), 1.0);
        // b10 is not b1 followed by a zero.
        assert_eq!(run("b10", &v).unwrap_err(), "Invalid band/asset name 'b10'");
    }

    #[test]
    fn empty_bands_and_bad_input() {
        // A band with no value empties the results that use it, and only those.
        let (out, _) = run("b1+b2;b2*2", &[None, Some(3.0)]).unwrap();
        assert_eq!(out, [None, Some(6.0)]);
        assert_eq!(run("b1/b2", &[Some(1.0), Some(0.0)]).unwrap().0, [None]);
        assert_eq!(run("nir-red", &[Some(1.0)]).unwrap_err(), "Invalid band/asset name 'nir'");
        assert!(run("b1 +", &[Some(1.0)]).is_err());
        assert!(run("(b1", &[Some(1.0)]).is_err());
        assert!(run("b1 b1", &[Some(1.0)]).is_err());
        assert!(run("where(b1)", &[Some(1.0)]).is_err());
        assert!(run(";", &[Some(1.0)]).is_err());
    }
}
