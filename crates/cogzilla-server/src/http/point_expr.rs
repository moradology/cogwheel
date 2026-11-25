//! TiTiler's band math on pixel values: expressions over `b1..bn`, the stacked bands of the
//! assets that were asked for, evaluated in double precision as numexpr does.
//!
//! `b1+b2;b2-b1` gives two values. Each result is described by the expression with every
//! `bN` replaced by that band's description (`red_b1+nir_b1`).

/// A parsed expression: one tree per `;`-separated part.
pub(super) struct Program {
    /// Each part's tree, its text, and the bands it reads, by index.
    parts: Vec<(Node, String, Box<[usize]>)>,
}

#[derive(Debug, Clone)]
enum Node {
    Num(f64),
    Band(usize),
    Neg(Box<Node>),
    Bin(u8, Box<Node>, Box<Node>),
    Call(fn(f64) -> f64, Box<Node>),
}

impl Node {
    fn eval(&self, bands: &[f64]) -> f64 {
        match self {
            Node::Num(v) => *v,
            // A pixel with fewer bands than the expression was parsed for has no value there.
            Node::Band(i) => bands.get(*i).copied().unwrap_or(f64::NAN),
            Node::Neg(a) => -a.eval(bands),
            Node::Bin(op, a, b) => {
                let (a, b) = (a.eval(bands), b.eval(bands));
                match op {
                    b'+' => a + b,
                    b'-' => a - b,
                    b'*' => a * b,
                    b'/' => a / b,
                    // numexpr's modulo takes the sign of the divisor: (-2) % 3 is 1.
                    b'%' => a - (a / b).floor() * b,
                    _ => a.powf(b),
                }
            }
            Node::Call(f, a) => f(a.eval(bands)),
        }
    }

    /// The bands this tree reads.
    fn reads(&self, into: &mut Vec<usize>) {
        match self {
            Node::Num(_) => {}
            Node::Band(i) => {
                if !into.contains(i) {
                    into.push(*i);
                }
            }
            Node::Neg(a) | Node::Call(_, a) => a.reads(into),
            Node::Bin(_, a, b) => {
                a.reads(into);
                b.reads(into);
            }
        }
    }
}

impl Program {
    /// Parse `expression` for a pixel of `bands` stacked bands.
    pub(super) fn parse(expression: &str, bands: usize) -> Result<Self, String> {
        // An expression comes from a URL: its length, its results and its nesting are held
        // to what band math needs, so that none of them can be used against the server.
        if expression.len() > MAX_EXPRESSION_BYTES {
            return Err(format!("expression is longer than {MAX_EXPRESSION_BYTES} bytes"));
        }
        let mut parts = Vec::new();
        for part in expression.split(';').map(str::trim).filter(|p| !p.is_empty()) {
            if parts.len() == MAX_RESULTS {
                return Err(format!("expression has more than {MAX_RESULTS} results"));
            }
            let mut parser = Parser { src: part.as_bytes(), pos: 0, bands, depth: 0 };
            let node = parser.sum()?;
            parser.skip_space();
            if parser.pos != parser.src.len() {
                return Err(format!("Invalid expression '{part}'"));
            }
            let mut reads = Vec::new();
            node.reads(&mut reads);
            parts.push((node, part.to_string(), reads.into_boxed_slice()));
        }
        if parts.is_empty() {
            return Err("Invalid expression ''".into());
        }
        Ok(Self { parts })
    }

    /// How many values the expression gives.
    pub(super) fn len(&self) -> usize {
        self.parts.len()
    }

    /// Result `part` for one pixel; a band with no value is NaN, and so is what uses it.
    ///
    /// Where every band read has a value and the arithmetic has none to give, the result
    /// is a number all the same, as in rio-tiler, which passes it through numpy's
    /// `nan_to_num`: zero for `0/0`, and the largest number there is, of either sign, for
    /// a division by zero.
    pub(super) fn eval(&self, part: usize, bands: &[f64]) -> f64 {
        let (tree, _, reads) = &self.parts[part];
        let v = tree.eval(bands);
        if v.is_finite() {
            return v;
        }
        for &i in reads.iter() {
            match bands.get(i) {
                Some(band) if !band.is_nan() => {}
                _ => return f64::NAN,
            }
        }
        if v.is_nan() {
            0.0
        } else if v > 0.0 {
            f64::MAX
        } else {
            f64::MIN
        }
    }

    /// The text of each part, as it was written.
    pub(super) fn sources(&self) -> impl Iterator<Item = &str> {
        self.parts.iter().map(|(_, text, _)| text.as_str())
    }

    /// Each part with every `bN` replaced by that band's description.
    pub(super) fn describe(&self, descriptions: &[String]) -> Vec<String> {
        self.parts.iter().map(|(_, text, _)| describe(text, descriptions)).collect()
    }
}

/// Evaluate `expression` over one pixel. A band with no value makes every result that
/// touches it empty.
pub(super) fn apply(
    expression: &str,
    values: &[Option<f64>],
    descriptions: &[String],
) -> Result<(Vec<Option<f64>>, Vec<String>), String> {
    let program = Program::parse(expression, values.len())?;
    let bands: Vec<f64> = values.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let out = (0..program.len())
        .map(|i| Some(program.eval(i, &bands)).filter(|v| v.is_finite()))
        .collect();
    Ok((out, program.describe(descriptions)))
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

/// The longest expression taken, the most results one may have, and how deep it may nest.
pub(super) const MAX_EXPRESSION_BYTES: usize = 4096;
pub(super) const MAX_RESULTS: usize = 16;
const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    bands: usize,
    /// How many operands are open around the one being read.
    depth: usize,
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

    fn sum(&mut self) -> Result<Node, String> {
        let mut v = self.product()?;
        loop {
            if self.eat("+") {
                v = Node::Bin(b'+', Box::new(v), Box::new(self.product()?));
            } else if self.eat("-") {
                v = Node::Bin(b'-', Box::new(v), Box::new(self.product()?));
            } else {
                return Ok(v);
            }
        }
    }

    fn product(&mut self) -> Result<Node, String> {
        let mut v = self.unary()?;
        loop {
            self.skip_space();
            // `**` belongs to the power rule, not to multiplication.
            if self.src[self.pos..].starts_with(b"**") {
                return Ok(v);
            }
            if self.eat("*") {
                v = Node::Bin(b'*', Box::new(v), Box::new(self.unary()?));
            } else if self.eat("/") {
                v = Node::Bin(b'/', Box::new(v), Box::new(self.unary()?));
            } else if self.eat("%") {
                v = Node::Bin(b'%', Box::new(v), Box::new(self.unary()?));
            } else {
                return Ok(v);
            }
        }
    }

    /// Every way the parser calls itself goes through here, so this is where depth is counted.
    fn unary(&mut self) -> Result<Node, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(format!("expression nests deeper than {MAX_DEPTH}"));
        }
        let node = self.operand();
        self.depth -= 1;
        node
    }

    fn operand(&mut self) -> Result<Node, String> {
        if self.eat("-") {
            return Ok(Node::Neg(Box::new(self.unary()?)));
        }
        if self.eat("+") {
            return self.unary();
        }
        let base = self.primary()?;
        if self.eat("**") {
            // Right-associative, and binds tighter than a leading minus on its right.
            return Ok(Node::Bin(b'^', Box::new(base), Box::new(self.unary()?)));
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<Node, String> {
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
            return text.parse::<f64>().map(Node::Num).map_err(|_| format!("Invalid number '{text}'"));
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
                let function: fn(f64) -> f64 = match ident {
                    "sqrt" => f64::sqrt,
                    "abs" => f64::abs,
                    "exp" => f64::exp,
                    "log" => f64::ln,
                    "log10" => f64::log10,
                    "sin" => f64::sin,
                    "cos" => f64::cos,
                    "tan" => f64::tan,
                    "arcsin" => f64::asin,
                    "arccos" => f64::acos,
                    "arctan" => f64::atan,
                    "floor" => f64::floor,
                    "ceil" => f64::ceil,
                    other => return Err(format!("function '{other}' is not supported in expressions")),
                };
                return Ok(Node::Call(function, Box::new(arg)));
            }
            return match band_index(ident).filter(|n| *n < self.bands) {
                Some(n) => Ok(Node::Band(n)),
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
        // As numexpr: the sign of the divisor.
        assert_eq!(one("(-b2) % b3"), 1.0);
        assert_eq!(one("b2 % (-b3)"), -1.0);
        // b10 is not b1 followed by a zero.
        assert_eq!(run("b10", &v).unwrap_err(), "Invalid band/asset name 'b10'");
    }

    #[test]
    fn empty_bands_and_bad_input() {
        // A band with no value empties the results that use it, and only those.
        let (out, _) = run("b1+b2;b2*2", &[None, Some(3.0)]).unwrap();
        assert_eq!(out, [None, Some(6.0)]);
        // Arithmetic with no answer is a number all the same, as numpy's nan_to_num makes it.
        assert_eq!(run("b1/b2", &[Some(1.0), Some(0.0)]).unwrap().0, [Some(f64::MAX)]);
        assert_eq!(run("-b1/b2;b2/b2;sqrt(b2-b1)", &[Some(1.0), Some(0.0)]).unwrap().0, [Some(f64::MIN), Some(0.0), Some(0.0)]);
        // Not where a band it reads has no value, whatever the other bands hold.
        assert_eq!(run("b1/b2;b3/b3", &[None, Some(0.0), Some(0.0)]).unwrap().0, [None, Some(0.0)]);
        assert_eq!(run("nir-red", &[Some(1.0)]).unwrap_err(), "Invalid band/asset name 'nir'");
        assert!(run("b1 +", &[Some(1.0)]).is_err());
        assert!(run("(b1", &[Some(1.0)]).is_err());
        assert!(run("b1 b1", &[Some(1.0)]).is_err());
        assert!(run("where(b1)", &[Some(1.0)]).is_err());
        assert!(run(";", &[Some(1.0)]).is_err());
    }
}
