//! TiTiler's band math on pixel values: expressions over `b1..bn`, the stacked bands of the
//! assets that were asked for, evaluated in double precision as numexpr does.
//!
//! `b1+b2;b2-b1` gives two values. Each result is described by the expression with every
//! `bN` replaced by that band's description (`red_b1+nir_b1`).
//!
//! Beyond arithmetic, numexpr's grammar is read as numexpr reads it: comparisons (true is 1),
//! `&`, `|` and `~` (on comparisons, logic; on whole numbers, bits), `//`, `where(c, a, b)`,
//! and its functions of one and two arguments. A program that uses any of that says so
//! ([`Program::plain`]): band math over planes, whose arrays are typed from the expression's
//! text, reads plain arithmetic only.

/// A parsed expression: one tree per `;`-separated part.
pub(super) struct Program {
    /// Each part's tree, its text, and the bands it reads, by index.
    parts: Vec<(Node, String, Box<[usize]>)>,
    /// Whether any part goes beyond arithmetic and the first functions.
    extended: bool,
}

#[derive(Debug, Clone)]
enum Node {
    Num(f64),
    Band(usize),
    Neg(Box<Node>),
    Bin(u8, Box<Node>, Box<Node>),
    Call(fn(f64) -> f64, Box<Node>),
    /// Anything beyond arithmetic, boxed so that a node stays as small as plain band math,
    /// whose loop over a plane's pixels walks these trees, needs it.
    More(Box<More>),
}

#[derive(Debug, Clone)]
enum More {
    Call2(fn(f64, f64) -> f64, Node, Node),
    /// A comparison, 1 where it holds and 0 where not.
    Cmp(Cmp, Node, Node),
    /// `&` (`true`) or `|` (`false`): logic over comparisons, bits over whole numbers.
    Bits(bool, Node, Node),
    /// `<<` (`true`) or `>>` (`false`): bit shifts over whole numbers.
    Shift(bool, Node, Node),
    /// `~`: not, over a comparison; the bits turned over, over a whole number.
    Not(Node),
    /// Python's floor division, which numexpr follows.
    FloorDiv(Node, Node),
    Where(Node, Node, Node),
}

/// The type numexpr gives a result over bands of whole numbers, narrowest first: numpy
/// stacks a program's results in the widest of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kind {
    Bool,
    Whole,
    Double,
}

#[derive(Debug, Clone, Copy)]
enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// `x` as a whole number for numexpr's bit operations, which take no fractions.
fn whole(x: f64) -> Option<i64> {
    (x.is_finite() && x.fract() == 0.0 && x.abs() < 9.0e15).then_some(x as i64)
}

/// numpy's `minimum` and `maximum`: a NaN on either side is the answer.
fn minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
}

fn maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
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
            Node::More(more) => more.eval(bands),
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
            Node::More(more) => more.reads(into),
        }
    }

    /// What numexpr makes of this tree over bands of whole numbers: sums, differences,
    /// products, `%`, `//`, `**`, bits and `where` keep them whole; a division, a function
    /// (`abs` included) and a literal with a fraction make doubles; a comparison is a bool.
    /// The tree is one parsed with `typing`, where a literal with a fraction is marked.
    fn kind(&self) -> Kind {
        match self {
            Node::Num(_) | Node::Band(_) => Kind::Whole,
            Node::Call(..) => Kind::Double,
            Node::Neg(a) => a.kind().max(Kind::Whole),
            Node::Bin(b'/', ..) => Kind::Double,
            Node::Bin(_, a, b) => a.kind().max(b.kind()).max(Kind::Whole),
            Node::More(more) => match more.as_ref() {
                More::FloorDiv(a, b) | More::Shift(_, a, b) => a.kind().max(b.kind()).max(Kind::Whole),
                More::Bits(_, a, b) | More::Where(_, a, b) => a.kind().max(b.kind()),
                More::Not(a) => a.kind(),
                More::Cmp(..) => Kind::Bool,
                More::Call2(..) => Kind::Double,
            },
        }
    }
}

impl More {
    /// [`Node::eval`] for what goes beyond arithmetic and the first functions: worked out
    /// apart, and never folded into that loop.
    #[inline(never)]
    fn eval(&self, bands: &[f64]) -> f64 {
        match self {
            More::FloorDiv(a, b) => (a.eval(bands) / b.eval(bands)).floor(),
            More::Call2(f, a, b) => f(a.eval(bands), b.eval(bands)),
            More::Cmp(op, a, b) => {
                let (a, b) = (a.eval(bands), b.eval(bands));
                if a.is_nan() || b.is_nan() {
                    return f64::NAN;
                }
                let holds = match op {
                    Cmp::Lt => a < b,
                    Cmp::Le => a <= b,
                    Cmp::Gt => a > b,
                    Cmp::Ge => a >= b,
                    Cmp::Eq => a == b,
                    Cmp::Ne => a != b,
                };
                f64::from(u8::from(holds))
            }
            More::Bits(and, a, b) => match (whole(a.eval(bands)), whole(b.eval(bands))) {
                (Some(a), Some(b)) => (if *and { a & b } else { a | b }) as f64,
                _ => f64::NAN,
            },
            More::Shift(left, a, b) => match (whole(a.eval(bands)), whole(b.eval(bands))) {
                (Some(a), Some(b)) if b >= 0 => {
                    if b >= 64 {
                        if *left || a >= 0 { 0.0 } else { -1.0 }
                    } else if *left {
                        a.wrapping_shl(b as u32) as f64
                    } else {
                        a.wrapping_shr(b as u32) as f64
                    }
                }
                _ => f64::NAN,
            },
            More::Not(a) => {
                let v = a.eval(bands);
                if is_logical(a) {
                    if v.is_nan() { v } else { 1.0 - v }
                } else {
                    whole(v).map_or(f64::NAN, |v| !v as f64)
                }
            }
            More::Where(c, a, b) => {
                let c = c.eval(bands);
                if c.is_nan() {
                    f64::NAN
                } else if c != 0.0 {
                    a.eval(bands)
                } else {
                    b.eval(bands)
                }
            }
        }
    }

    fn reads(&self, into: &mut Vec<usize>) {
        match self {
            More::Not(a) => a.reads(into),
            More::Call2(_, a, b)
            | More::Cmp(_, a, b)
            | More::Bits(_, a, b)
            | More::Shift(_, a, b)
            | More::FloorDiv(a, b) => {
                a.reads(into);
                b.reads(into);
            }
            More::Where(c, a, b) => {
                c.reads(into);
                a.reads(into);
                b.reads(into);
            }
        }
    }
}

/// Whether a tree is true or false rather than a number: what `~` turns over as logic.
fn is_logical(node: &Node) -> bool {
    match node {
        Node::More(more) => match more.as_ref() {
            More::Cmp(..) => true,
            More::Bits(_, a, b) => is_logical(a) && is_logical(b),
            More::Not(a) => is_logical(a),
            _ => false,
        },
        _ => false,
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
        let mut extended = false;
        for part in expression.split(';').map(str::trim).filter(|p| !p.is_empty()) {
            if parts.len() == MAX_RESULTS {
                return Err(format!("expression has more than {MAX_RESULTS} results"));
            }
            let mut parser = Parser { src: part.as_bytes(), pos: 0, bands, depth: 0, extended: false, typing: false };
            let node = parser.comparison()?;
            extended |= parser.extended;
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
        Ok(Self { parts, extended })
    }

    /// Whether the expression is arithmetic and the first functions only (`sqrt`, `log`,
    /// `floor`...), whose type the shared typing of band math reads from the text.
    pub(super) fn plain(&self) -> bool {
        !self.extended
    }

    /// The type of the results stacked, over bands of whole numbers, as numexpr and numpy
    /// give it: bools if every result is a comparison, doubles if any is, whole otherwise.
    ///
    /// Each part is parsed again to read it: the tree that is worked out per pixel does not
    /// tell `2.0` from `2`, so that the loop over a plane's pixels stays as it is.
    pub(super) fn kind(&self) -> Kind {
        self.parts
            .iter()
            .map(|(_, text, _)| {
                let mut parser = Parser { src: text.as_bytes(), pos: 0, bands: usize::MAX, depth: 0, extended: false, typing: true };
                parser.comparison().map_or(Kind::Double, |tree| tree.kind())
            })
            .max()
            .unwrap_or(Kind::Double)
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

/// `b3` (or `B3`, as rio-tiler takes it) -> `Some(2)`.
fn band_index(ident: &str) -> Option<usize> {
    let n: usize = ident.strip_prefix(['b', 'B'])?.parse().ok()?;
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
    /// Whether anything beyond arithmetic and the first functions was read.
    extended: bool,
    /// Whether the tree is read for its type ([`Program::kind`]): a literal with a fraction
    /// or an exponent, a double to numexpr, is then a call, which types as one.
    typing: bool,
}

/// What a literal with a fraction is wrapped in when a tree is read for its type.
fn real(x: f64) -> f64 {
    x
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

    /// One comparison at most, as numexpr has them: `a < b < c` is not read.
    fn comparison(&mut self) -> Result<Node, String> {
        let left = self.bit_or()?;
        self.skip_space();
        let rest = &self.src[self.pos..];
        let op = if rest.starts_with(b"<=") {
            Some((Cmp::Le, 2))
        } else if rest.starts_with(b">=") {
            Some((Cmp::Ge, 2))
        } else if rest.starts_with(b"==") {
            Some((Cmp::Eq, 2))
        } else if rest.starts_with(b"!=") {
            Some((Cmp::Ne, 2))
        } else if rest.starts_with(b"<") && !rest.starts_with(b"<<") {
            Some((Cmp::Lt, 1))
        } else if rest.starts_with(b">") && !rest.starts_with(b">>") {
            Some((Cmp::Gt, 1))
        } else {
            None
        };
        let Some((op, len)) = op else { return Ok(left) };
        self.pos += len;
        self.extended = true;
        let right = self.bit_or()?;
        Ok(Node::More(Box::new(More::Cmp(op, left, right))))
    }

    fn bit_or(&mut self) -> Result<Node, String> {
        let mut v = self.bit_and()?;
        while self.eat("|") {
            self.extended = true;
            v = Node::More(Box::new(More::Bits(false, v, self.bit_and()?)));
        }
        Ok(v)
    }

    fn bit_and(&mut self) -> Result<Node, String> {
        let mut v = self.shift()?;
        while self.eat("&") {
            self.extended = true;
            v = Node::More(Box::new(More::Bits(true, v, self.shift()?)));
        }
        Ok(v)
    }

    fn shift(&mut self) -> Result<Node, String> {
        let mut v = self.sum()?;
        loop {
            if self.eat("<<") {
                self.extended = true;
                v = Node::More(Box::new(More::Shift(true, v, self.sum()?)));
            } else if self.eat(">>") {
                self.extended = true;
                v = Node::More(Box::new(More::Shift(false, v, self.sum()?)));
            } else {
                return Ok(v);
            }
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
            } else if self.eat("//") {
                self.extended = true;
                v = Node::More(Box::new(More::FloorDiv(v, self.unary()?)));
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
        if self.eat("~") {
            self.extended = true;
            return Ok(Node::More(Box::new(More::Not(self.unary()?))));
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
            let v = self.comparison()?;
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
            let v = text.parse::<f64>().map_err(|_| format!("Invalid number '{text}'"))?;
            return Ok(if self.typing && text.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
                Node::Call(real, Box::new(Node::Num(v)))
            } else {
                Node::Num(v)
            });
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            while self.src.get(self.pos).is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_') {
                self.pos += 1;
            }
            let ident = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
            if self.eat("(") {
                let mut args = vec![self.comparison()?];
                while self.eat(",") {
                    if args.len() == 3 {
                        return Err(format!("function '{ident}' takes at most three arguments"));
                    }
                    args.push(self.comparison()?);
                }
                if !self.eat(")") {
                    return Err("Invalid expression: missing ')'".into());
                }
                return self.call(ident, args);
            }
            return match band_index(ident).filter(|n| *n < self.bands) {
                Some(n) => Ok(Node::Band(n)),
                None => Err(format!("Invalid band/asset name '{ident}'")),
            };
        }
        Err(format!("Invalid expression: unexpected '{}'", c as char))
    }

    /// A call of `ident` on `args`, by numexpr's functions.
    fn call(&mut self, ident: &str, mut args: Vec<Node>) -> Result<Node, String> {
        let arity = |n: usize| {
            if args.len() == n {
                Ok(())
            } else {
                Err(format!("function '{ident}' takes {n} argument{}", if n == 1 { "" } else { "s" }))
            }
        };
        let two: Option<fn(f64, f64) -> f64> = match ident {
            "arctan2" => Some(f64::atan2),
            "fmod" => Some(|a, b| a % b),
            "minimum" => Some(minimum),
            "maximum" => Some(maximum),
            _ => None,
        };
        if let Some(function) = two {
            arity(2)?;
            self.extended = true;
            let b = args.pop().unwrap_or(Node::Num(f64::NAN));
            let a = args.pop().unwrap_or(Node::Num(f64::NAN));
            return Ok(Node::More(Box::new(More::Call2(function, a, b))));
        }
        if ident == "where" {
            arity(3)?;
            self.extended = true;
            let b = args.pop().unwrap_or(Node::Num(f64::NAN));
            let a = args.pop().unwrap_or(Node::Num(f64::NAN));
            let c = args.pop().unwrap_or(Node::Num(f64::NAN));
            return Ok(Node::More(Box::new(More::Where(c, a, b))));
        }
        arity(1)?;
        let arg = args.pop().unwrap_or(Node::Num(f64::NAN));
        let later: Option<fn(f64) -> f64> = match ident {
            "log1p" => Some(f64::ln_1p),
            "expm1" => Some(f64::exp_m1),
            "sinh" => Some(f64::sinh),
            "cosh" => Some(f64::cosh),
            "tanh" => Some(f64::tanh),
            "arcsinh" => Some(f64::asinh),
            "arccosh" => Some(f64::acosh),
            "arctanh" => Some(f64::atanh),
            // numpy rounds a half to the even neighbour.
            "round" => Some(f64::round_ties_even),
            "real" => Some(|x| x),
            // One where the argument has a value; nowhere, as everything, where it has none.
            "ones_like" => Some(|x| if x.is_nan() { x } else { 1.0 }),
            _ => None,
        };
        if let Some(function) = later {
            self.extended = true;
            return Ok(Node::Call(function, Box::new(arg)));
        }
        // The functions every expression reader here has always taken.
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
        Ok(Node::Call(function, Box::new(arg)))
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
    fn whole_numbers_as_numexpr_types_them() {
        // Read from numexpr itself, over uint16 bands.
        let kind = |e: &str| Program::parse(e, 2).unwrap().kind();
        for e in ["b1*2", "b1+b2;b2-b1", "-b1", "b1%3", "b1//2", "b1**2", "where(b1>5,b1,0)", "where(b1>1.5,b1,0)", "b1&3", "b1>5;b1*2"] {
            assert_eq!(kind(e), Kind::Whole, "{e}");
        }
        for e in ["b1/1", "abs(b1)", "b1*2.0", "b1*1e1", ".5*b1", "minimum(b1,b2)", "b1*2;b1/2", "where(b1>5,b1,0.5)"] {
            assert_eq!(kind(e), Kind::Double, "{e}");
        }
        for e in ["b1>5", "(b1>5)&(b2<3)", "~(b1>5);b2==1"] {
            assert_eq!(kind(e), Kind::Bool, "{e}");
        }
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
    fn numexpr_beyond_arithmetic() {
        // Values titiler-pgstac returned for a pixel of 77, 81 (visual_b1, visual_b2).
        let v = [Some(77.0), Some(81.0)];
        let one = |e: &str| run(e, &v).unwrap().0[0].unwrap();
        assert_eq!(one("b1>=b2"), 0.0);
        assert_eq!(one("b1!=b2"), 1.0);
        assert_eq!(one("~(b1>100)"), 1.0);
        assert_eq!(one("(b1>1)&(b2<9000)"), 1.0);
        assert_eq!(one("b1//7"), 11.0);
        assert_eq!(one("round(b1/7)"), 11.0);
        assert_eq!(one("log1p(b1)"), 4.356708826689592);
        assert_eq!(one("where(b1>b2,b1,b2)"), 81.0);
        assert_eq!(one("arctan2(b1,b2)"), 0.7600871142546);
        assert_eq!(one("minimum(b1,b2)"), 77.0);
        assert_eq!(one("fmod(b1,7)"), 0.0);
        assert_eq!(one("B1+B2"), 158.0);
        assert_eq!(one("ones_like(b1)"), 1.0);
        // Bits over whole numbers, as numexpr's integers.
        assert_eq!(one("b1&b2"), f64::from(77 & 81));
        assert_eq!(one("b1|b2"), f64::from(77 | 81));
        assert_eq!(one("~b1"), -78.0);
        // numpy rounds a half to the even neighbour.
        assert_eq!(one("round(2.5)+b1*0"), 2.0);
        // A comparison binds looser than `|`, so this is `b1 > (100|b2) > 5`: a chain, which
        // numexpr fails on too.
        assert!(run("b1>100|b2>5", &v).is_err());
        assert!(run("b1<b2<3", &v).is_err());
        assert_eq!(one("(b1>100)|(b2>5)"), 1.0);
        assert!(run("where(b1,b2)", &v).unwrap_err().contains("takes 3 arguments"));
        assert!(run("minimum(b1)", &v).unwrap_err().contains("takes 2 arguments"));
        assert!(run("f(b1,b2,b1,b2)", &v).unwrap_err().contains("at most three"));
        // Bit shifts over whole numbers (`<<` and `>>`), with precedence between `&` and `+`.
        assert_eq!(one("b1<<2"), f64::from(77 << 2));
        assert_eq!(one("b1>>2"), f64::from(77 >> 2));
        assert_eq!(one("1+2<<1"), 6.0);
        assert_eq!(one("b1&(3<<2)"), f64::from(77 & (3 << 2)));
        assert_eq!(one("b1&3<<2"), f64::from(77 & (3 << 2)));
        assert_eq!(one("b1<<64"), 0.0);
        assert_eq!(one("b1>>64"), 0.0);
        assert_eq!(one("(-b1)>>64"), -1.0);
        // Negative shift count produces NaN -> 0.0 via nan_to_num when all bands are present.
        assert_eq!(one("b1<<(-1)"), 0.0);
        assert_eq!(Program::parse("b1<<2", 2).unwrap().kind(), Kind::Whole);
        assert!(!Program::parse("b1<<2", 2).unwrap().plain());
        // A band with no value is no value through all of it.
        assert_eq!(run("where(b1>0,1,0);~(b1>0)", &[None, Some(1.0)]).unwrap().0, [None, None]);
        // Who reads what: plain arithmetic, or more.
        assert!(Program::parse("(b2-b1)/(b2+b1)*sqrt(b1)", 2).unwrap().plain());
        for e in ["b1>1", "b1//2", "~b1", "where(b1>1,b1,0)", "log1p(b1)", "b1&b2", "minimum(b1,b2)"] {
            assert!(!Program::parse(e, 2).unwrap().plain(), "{e}");
        }
        assert_eq!(run("B1+b2", &v).unwrap().1, ["red_b1+nir_b1"]);
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
