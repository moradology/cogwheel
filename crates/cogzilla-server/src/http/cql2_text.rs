//! CQL2 text to CQL2 JSON, written to give the JSON titiler-pgstac's converter (the `cql2`
//! library, 0.6) gives for the same text, so that a collection filtered by a text filter is
//! registered in pgSTAC under the same hash.
//!
//! What matters for that: every number is a float (`20` is `20.0`), chains of `AND` or of
//! `OR` are one flat list, `NOT LIKE`/`NOT IN`/`NOT BETWEEN`/`IS NOT NULL` are a `not`
//! around the plain operator, a leading minus on anything but a number is a product with
//! `-1.0`, timestamps are brought to UTC, and WKT becomes GeoJSON.
//!
//! A filter that parses is then checked against the shape the CQL2 schema asks for, as the
//! library's `validate()` does; what fails either step is an error for the caller to turn
//! into a `422`. Where the library accepts something by accident (a WKT it cannot read
//! becomes a function call named `POINT`), this parser rejects it instead.

use serde_json::{Value, json};

/// The CQL2 JSON for a CQL2 text filter.
pub(super) fn to_json(text: &str) -> Result<Value, String> {
    let mut parser = Parser { src: text, pos: 0, depth: 0 };
    let expr = parser.expr()?;
    parser.skip_space();
    if parser.pos != parser.src.len() {
        return Err(parser.unexpected());
    }
    if !expr.is_object() || !is_boolean(&expr) {
        return Err("the filter is not a valid CQL2 boolean expression".into());
    }
    Ok(expr)
}

/// A filter given as CQL2 JSON where text was expected, as the reference's converter
/// writes it back: operators in lower case, timestamps in UTC, every number a float, and
/// nothing kept in an expression but what CQL2 puts there.
pub(super) fn json_to_json(filter: &Value) -> Result<Value, String> {
    let expr = tidy(filter, 0)?;
    if op_of(&expr).is_none() || !is_boolean(&expr) {
        return Err("the filter is not a valid CQL2 boolean expression".into());
    }
    Ok(expr)
}

/// Every number in a JSON value as a float: `50` is written `50.0`.
fn numbers_as_floats(value: &mut Value) {
    match value {
        Value::Number(n) if !n.is_f64() => {
            if let Some(f) = n.as_f64().and_then(serde_json::Number::from_f64) {
                *n = f;
            }
        }
        Value::Array(items) => items.iter_mut().for_each(numbers_as_floats),
        Value::Object(fields) => fields.values_mut().for_each(numbers_as_floats),
        _ => {}
    }
}

fn tidy(v: &Value, depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return Err("the filter is nested too deeply".into());
    }
    let instant = |text: &str| -> Result<Value, String> {
        // A timestamp is known by what it is once brought to UTC.
        let utc = normalize_timestamp(text);
        if text == ".." || is_date(text) {
            Ok(json!(text))
        } else if is_timestamp(&utc) {
            Ok(json!(utc))
        } else {
            Err(format!("'{text}' is not a date or a timestamp"))
        }
    };
    match v {
        Value::Null => Err("null is not a CQL2 expression".into()),
        Value::Bool(_) | Value::String(_) => Ok(v.clone()),
        Value::Number(_) => {
            let mut number = v.clone();
            numbers_as_floats(&mut number);
            Ok(number)
        }
        Value::Array(items) => Ok(Value::Array(items.iter().map(|i| tidy(i, depth + 1)).collect::<Result<_, _>>()?)),
        Value::Object(fields) => {
            if let Some(name) = fields.get("op") {
                let (Some(name), Some(args)) = (name.as_str(), fields.get("args").and_then(Value::as_array)) else {
                    return Err("an operation needs an `op` and a list of `args`".into());
                };
                let lower = name.to_lowercase();
                let name = if lower == "isnull" { "isNull" } else { lower.as_str() };
                let args = args.iter().map(|a| tidy(a, depth + 1)).collect::<Result<Vec<_>, _>>()?;
                Ok(op(name, args))
            } else if let Some(name) = fields.get("property") {
                let name = name.as_str().ok_or("a property is named by a string")?;
                Ok(json!({ "property": name }))
            } else if let Some(text) = fields.get("timestamp") {
                let utc = text.as_str().map(normalize_timestamp).filter(|t| is_timestamp(t)).ok_or("not a timestamp")?;
                Ok(json!({ "timestamp": utc }))
            } else if let Some(text) = fields.get("date") {
                let text = text.as_str().filter(|t| is_date(t)).ok_or("not a date")?;
                Ok(json!({ "date": text }))
            } else if let Some(ends) = fields.get("interval") {
                let ends = ends.as_array().filter(|e| e.len() == 2).ok_or("an interval has two ends")?;
                let ends = ends
                    .iter()
                    .map(|end| match end {
                        Value::String(text) => instant(text),
                        other => tidy(other, depth + 1),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(json!({ "interval": ends }))
            } else if fields.contains_key("bbox") || fields.get("type").is_some_and(|t| t.is_string() && t != "GeometryCollection") {
                // A box or a GeoJSON geometry is kept whole.
                let mut shape = v.clone();
                numbers_as_floats(&mut shape);
                Ok(shape)
            } else {
                Err("not a CQL2 expression".into())
            }
        }
    }
}

/// Nesting allowed before the parser gives up, so a hostile filter cannot exhaust the stack.
const MAX_DEPTH: usize = 64;

const SPATIAL: [&str; 8] = [
    "s_contains", "s_crosses", "s_disjoint", "s_equals", "s_intersects", "s_overlaps",
    "s_touches", "s_within",
];
const TEMPORAL: [&str; 15] = [
    "t_after", "t_before", "t_contains", "t_disjoint", "t_during", "t_equals", "t_finishedBy",
    "t_finishes", "t_intersects", "t_meets", "t_metBy", "t_overlappedBy", "t_overlaps",
    "t_startedBy", "t_starts",
];
const ARRAY: [&str; 4] = ["a_containedBy", "a_contains", "a_equals", "a_overlaps"];
const COMPARISON: [&str; 6] = ["=", "<>", "<", "<=", ">", ">="];
const ARITHMETIC: [&str; 7] = ["+", "-", "*", "/", "%", "^", "div"];
const WKT_TYPES: [(&str, &str); 7] = [
    ("point", "Point"),
    ("linestring", "LineString"),
    ("polygon", "Polygon"),
    ("multipoint", "MultiPoint"),
    ("multilinestring", "MultiLineString"),
    ("multipolygon", "MultiPolygon"),
    ("geometrycollection", "GeometryCollection"),
];

fn op(name: &str, args: Vec<Value>) -> Value {
    json!({ "op": name, "args": args })
}

struct Parser<'a> {
    src: &'a str,
    pos: usize,
    depth: usize,
}

impl Parser<'_> {
    fn rest(&self) -> &str {
        &self.src[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn skip_space(&mut self) {
        while let Some(c) = self.peek().filter(|c| c.is_whitespace()) {
            self.pos += c.len_utf8();
        }
    }

    fn unexpected(&self) -> String {
        match self.peek() {
            Some(c) => format!("unexpected '{c}' at position {}", self.pos + 1),
            None => "unexpected end of the filter".into(),
        }
    }

    /// Take `token` (after white space) if it is next.
    fn eat(&mut self, token: &str) -> bool {
        self.skip_space();
        if self.rest().starts_with(token) {
            self.pos += token.len();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, token: &str) -> Result<(), String> {
        if self.eat(token) { Ok(()) } else { Err(self.unexpected()) }
    }

    /// Take a keyword in operator position, whatever its case. As in the library's grammar
    /// it does not have to be followed by a space: `a=1AND b=2` is two comparisons.
    fn eat_keyword(&mut self, keyword: &str) -> bool {
        self.skip_space();
        let rest = self.rest();
        if rest.len() >= keyword.len()
            && rest.is_char_boundary(keyword.len())
            && rest[..keyword.len()].eq_ignore_ascii_case(keyword)
        {
            self.pos += keyword.len();
            true
        } else {
            false
        }
    }

    /// An identifier: a letter, then letters, digits, `_`, `:` and `.`.
    fn identifier(&mut self) -> Option<&str> {
        let rest = &self.src[self.pos..];
        let mut end = 0;
        for (i, c) in rest.char_indices() {
            let ok = if i == 0 { c.is_alphabetic() } else { c.is_alphanumeric() || matches!(c, '_' | ':' | '.') };
            if !ok {
                break;
            }
            end = i + c.len_utf8();
        }
        if end == 0 {
            return None;
        }
        self.pos += end;
        Some(&rest[..end])
    }

    fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T, String>) -> Result<T, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err("the filter is nested too deeply".into());
        }
        let out = f(self);
        self.depth -= 1;
        out
    }

    fn expr(&mut self) -> Result<Value, String> {
        self.nested(|p| p.chain("OR", "or", Self::and))
    }

    fn and(&mut self) -> Result<Value, String> {
        self.chain("AND", "and", Self::not)
    }

    /// `a AND b AND c`: one operator over all of them, also when some were in parentheses.
    fn chain(
        &mut self,
        keyword: &str,
        name: &str,
        operand: fn(&mut Self) -> Result<Value, String>,
    ) -> Result<Value, String> {
        let first = operand(self)?;
        if !self.eat_keyword(keyword) {
            return Ok(first);
        }
        let mut args = Vec::new();
        let mut push = |v: Value| match v {
            Value::Object(mut o) if o.get("op").and_then(Value::as_str) == Some(name) => {
                match o.remove("args") {
                    Some(Value::Array(inner)) => args.extend(inner),
                    other => args.extend(other),
                }
            }
            v => args.push(v),
        };
        push(first);
        loop {
            push(operand(self)?);
            if !self.eat_keyword(keyword) {
                break;
            }
        }
        Ok(op(name, args))
    }

    fn not(&mut self) -> Result<Value, String> {
        self.skip_space();
        let start = self.pos;
        if self.identifier().is_some_and(|w| w.eq_ignore_ascii_case("not")) {
            let inner = self.nested(Self::not)?;
            return Ok(op("not", vec![inner]));
        }
        self.pos = start;
        self.predicate()
    }

    fn predicate(&mut self) -> Result<Value, String> {
        let left = self.arith()?;
        self.skip_space();
        for (token, name) in [("<>", "<>"), ("!=", "<>"), ("<=", "<="), (">=", ">="), ("=", "="), ("<", "<"), (">", ">")] {
            if self.eat(token) {
                let right = self.arith()?;
                return Ok(op(name, vec![left, right]));
            }
        }
        if self.eat_keyword("IS") {
            let negated = self.eat_keyword("NOT");
            if !self.eat_keyword("NULL") {
                return Err(self.unexpected());
            }
            let test = op("isNull", vec![left]);
            return Ok(if negated { op("not", vec![test]) } else { test });
        }
        let before_not = self.pos;
        let negated = self.eat_keyword("NOT");
        let test = if self.eat_keyword("LIKE") {
            let pattern = self.arith()?;
            op("like", vec![left, pattern])
        } else if self.eat_keyword("BETWEEN") {
            let low = self.arith()?;
            if !self.eat_keyword("AND") {
                return Err(self.unexpected());
            }
            let high = self.arith()?;
            op("between", vec![left, low, high])
        } else if self.eat_keyword("IN") {
            self.expect("(")?;
            let items = self.list()?;
            op("in", vec![left, Value::Array(items)])
        } else if negated {
            self.pos = before_not;
            return Err(self.unexpected());
        } else {
            return Ok(left);
        };
        Ok(if negated { op("not", vec![test]) } else { test })
    }

    /// One or more expressions up to the closing parenthesis, the opening one being taken.
    fn list(&mut self) -> Result<Vec<Value>, String> {
        let mut items = vec![self.expr()?];
        while self.eat(",") {
            items.push(self.expr()?);
        }
        self.expect(")")?;
        Ok(items)
    }

    fn arith(&mut self) -> Result<Value, String> {
        let mut left = self.term()?;
        loop {
            let name = if self.eat("+") {
                "+"
            } else if self.eat("-") {
                "-"
            } else {
                return Ok(left);
            };
            left = op(name, vec![left, self.term()?]);
        }
    }

    fn term(&mut self) -> Result<Value, String> {
        let mut left = self.power()?;
        loop {
            let name = if self.eat("*") {
                "*"
            } else if self.eat("/") {
                "/"
            } else if self.eat("%") {
                "%"
            } else if self.eat_keyword("div") {
                "div"
            } else {
                return Ok(left);
            };
            left = op(name, vec![left, self.power()?]);
        }
    }

    fn power(&mut self) -> Result<Value, String> {
        let mut left = self.unary()?;
        while self.eat("^") {
            left = op("^", vec![left, self.unary()?]);
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Value, String> {
        if self.eat("-") {
            let inner = self.nested(Self::unary)?;
            return Ok(match inner.as_f64() {
                Some(n) => json!(-n),
                None => op("*", vec![json!(-1.0), inner]),
            });
        }
        if self.eat("+") {
            // A plus sign is only taken in front of a number.
            return if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.number()
            } else {
                Err(self.unexpected())
            };
        }
        self.primary()
    }

    fn number(&mut self) -> Result<Value, String> {
        let bytes = self.rest().as_bytes();
        let digits = |from: usize| bytes[from..].iter().take_while(|b| b.is_ascii_digit()).count();
        let mut end = digits(0);
        if end == 0 {
            return Err(self.unexpected());
        }
        if bytes.get(end) == Some(&b'.') {
            end += 1 + digits(end + 1);
        }
        if matches!(bytes.get(end), Some(b'e' | b'E')) {
            let sign = usize::from(matches!(bytes.get(end + 1), Some(b'+' | b'-')));
            let exponent = digits(end + 1 + sign);
            if exponent > 0 {
                end += 1 + sign + exponent;
            }
        }
        let text = &self.rest()[..end];
        let value: f64 = text.parse().map_err(|_| format!("'{text}' is not a number"))?;
        if !value.is_finite() {
            return Err(format!("'{text}' is not a finite number"));
        }
        self.pos += end;
        Ok(json!(value))
    }

    /// A quoted run with the quote doubled to stand for itself, the opening quote being next.
    fn quoted(&mut self, quote: char) -> Result<String, String> {
        let mut out = String::new();
        let mut chars = self.rest().char_indices().skip(1).peekable();
        while let Some((i, c)) = chars.next() {
            if c != quote {
                out.push(c);
            } else if chars.peek().is_some_and(|(_, next)| *next == quote) {
                out.push(quote);
                chars.next();
            } else {
                self.pos += i + 1;
                return Ok(out);
            }
        }
        Err("a quoted value is not closed".into())
    }

    fn primary(&mut self) -> Result<Value, String> {
        self.skip_space();
        let Some(c) = self.peek() else {
            return Err(self.unexpected());
        };
        if c.is_ascii_digit() {
            return self.number();
        }
        if c == '\'' {
            return self.quoted('\'').map(Value::String);
        }
        if c == '"' {
            return self.quoted('"').map(|name| json!({ "property": name }));
        }
        if c == '(' {
            self.pos += 1;
            let mut items = self.list()?;
            // One expression in parentheses is that expression; several are an array.
            return Ok(if items.len() == 1 { items.remove(0) } else { Value::Array(items) });
        }
        let Some(word) = self.identifier().map(str::to_string) else {
            return Err(self.unexpected());
        };
        let lower = word.to_ascii_lowercase();
        match lower.as_str() {
            "true" => return Ok(json!(true)),
            "false" => return Ok(json!(false)),
            "null" => return Ok(Value::Null),
            _ => {}
        }
        // A WKT type, alone (`POINT (..)`, `POINT Z (..)`, `POLYGON EMPTY`) or with its
        // measure attached (`POINTZ(..)`).
        let wkt = WKT_TYPES.iter().find_map(|(name, geojson_type)| {
            let measure = match lower.strip_prefix(name)? {
                "" => None,
                "z" => Some(Measure::Z),
                "m" => Some(Measure::M),
                "zm" => Some(Measure::Zm),
                _ => return None,
            };
            Some((*geojson_type, measure))
        });
        if let Some((geojson_type, attached)) = wkt {
            let mark = self.pos;
            self.skip_space();
            let tagged = self.rest().chars().next().is_some_and(|c| matches!(c, 'Z' | 'z' | 'M' | 'm' | 'E' | 'e'));
            if (tagged && attached.is_none()) || self.rest().starts_with('(') {
                return self.wkt(geojson_type, attached);
            }
            self.pos = mark;
        }
        if !self.eat("(") {
            return Ok(json!({ "property": word }));
        }
        let args = if self.eat(")") { Vec::new() } else { self.call_args(&lower)? };
        match lower.as_str() {
            "timestamp" | "date" => {
                let [Value::String(text)] = args.as_slice() else {
                    return Err(format!("{} takes one quoted value", word.to_ascii_uppercase()));
                };
                let text = if lower == "timestamp" { normalize_timestamp(text) } else { text.clone() };
                Ok(json!({ lower: text }))
            }
            "interval" => {
                let ends = args
                    .into_iter()
                    .map(|v| match v {
                        Value::String(s) => Value::String(normalize_timestamp(&s)),
                        other => other,
                    })
                    .collect::<Vec<_>>();
                Ok(json!({ "interval": ends }))
            }
            "bbox" => Ok(json!({ "bbox": args })),
            _ => {
                let canonical = SPATIAL
                    .iter()
                    .chain(&TEMPORAL)
                    .chain(&ARRAY)
                    .chain(&["casei", "accenti"])
                    .find(|name| name.eq_ignore_ascii_case(&word));
                Ok(op(canonical.copied().unwrap_or(&word), args))
            }
        }
    }

    /// The arguments of a function call, up to its closing parenthesis. In the array
    /// functions a value in parentheses is an array even when it is alone, and so is a
    /// bare literal.
    fn call_args(&mut self, lower_name: &str) -> Result<Vec<Value>, String> {
        let array_function = ARRAY.iter().any(|name| name.eq_ignore_ascii_case(lower_name));
        let mut args = Vec::new();
        loop {
            self.skip_space();
            let arg = if array_function && self.rest().starts_with('(') {
                self.pos += 1;
                Value::Array(self.list()?)
            } else {
                let value = self.expr()?;
                match value {
                    Value::Number(_) | Value::String(_) | Value::Bool(_) if array_function => json!([value]),
                    other => other,
                }
            };
            args.push(arg);
            if !self.eat(",") {
                break;
            }
        }
        self.expect(")")?;
        Ok(args)
    }

    /// A WKT geometry, its type name being taken already, and its measure too when that
    /// was written onto the name.
    fn wkt(&mut self, geojson_type: &str, attached: Option<Measure>) -> Result<Value, String> {
        self.skip_space();
        let tag = self.identifier().map(str::to_ascii_lowercase);
        let measure = match (attached, tag.as_deref()) {
            (Some(measure), None) => measure,
            (None, None) => Measure::Plain,
            (None, Some("z")) => Measure::Z,
            (None, Some("m")) => Measure::M,
            (None, Some("zm")) => Measure::Zm,
            // An empty geometry has no coordinates; a point cannot be written that way.
            (None, Some("empty")) if geojson_type == "GeometryCollection" => {
                return Ok(json!({ "type": geojson_type, "geometries": [] }));
            }
            (None, Some("empty")) if geojson_type != "Point" => {
                return Ok(json!({ "type": geojson_type, "coordinates": [] }));
            }
            _ => return Err("this WKT geometry cannot be read".into()),
        };
        self.expect("(")?;
        let body = if geojson_type == "GeometryCollection" {
            let mut geometries = Vec::new();
            loop {
                self.skip_space();
                let word = self.identifier().map(str::to_ascii_lowercase).unwrap_or_default();
                let inner = WKT_TYPES
                    .iter()
                    .find(|(name, t)| *name == word && *t != "GeometryCollection")
                    .ok_or_else(|| "a GEOMETRYCOLLECTION takes simple geometries".to_string())?;
                geometries.push(self.wkt(inner.1, None)?);
                if !self.eat(",") {
                    break;
                }
            }
            json!({ "type": geojson_type, "geometries": geometries })
        } else {
            let coordinates = match geojson_type {
                "Point" => self.position(measure)?,
                "LineString" => self.positions(measure)?,
                "MultiPoint" => {
                    // Both `(0 0, 1 1)` and `((0 0), (1 1))`.
                    let mut points = Vec::new();
                    loop {
                        let wrapped = self.eat("(");
                        points.push(self.position(measure)?);
                        if wrapped {
                            self.expect(")")?;
                        }
                        if !self.eat(",") {
                            break;
                        }
                    }
                    Value::Array(points)
                }
                "Polygon" | "MultiLineString" => self.sequence(|p| p.ring(measure))?,
                _ => self.sequence(|p| {
                    p.expect("(")?;
                    let rings = p.sequence(|p| p.ring(measure))?;
                    p.expect(")")?;
                    Ok(rings)
                })?,
            };
            // Every position of a geometry has the same number of coordinates.
            fn sizes(v: &Value, out: &mut Vec<usize>) {
                match v.as_array() {
                    Some(items) if items.first().is_some_and(Value::is_number) => out.push(items.len()),
                    Some(items) => items.iter().for_each(|i| sizes(i, out)),
                    None => {}
                }
            }
            let mut seen = Vec::new();
            sizes(&coordinates, &mut seen);
            if seen.windows(2).any(|w| w[0] != w[1]) {
                return Err("the positions of a WKT geometry differ in dimension".into());
            }
            json!({ "type": geojson_type, "coordinates": coordinates })
        };
        self.expect(")")?;
        Ok(body)
    }

    fn sequence(&mut self, mut item: impl FnMut(&mut Self) -> Result<Value, String>) -> Result<Value, String> {
        let mut items = vec![item(self)?];
        while self.eat(",") {
            items.push(item(self)?);
        }
        Ok(Value::Array(items))
    }

    fn ring(&mut self, measure: Measure) -> Result<Value, String> {
        self.expect("(")?;
        let points = self.positions(measure)?;
        self.expect(")")?;
        Ok(points)
    }

    fn positions(&mut self, measure: Measure) -> Result<Value, String> {
        self.sequence(|p| p.position(measure))
    }

    /// One position: two to four plain decimal numbers separated by spaces.
    fn position(&mut self, measure: Measure) -> Result<Value, String> {
        let mut numbers: Vec<f64> = Vec::with_capacity(4);
        loop {
            self.skip_space();
            let bytes = self.rest().as_bytes();
            let sign = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
            let digits = |from: usize| bytes[from..].iter().take_while(|b| b.is_ascii_digit()).count();
            let whole = digits(sign);
            if whole == 0 {
                break;
            }
            let mut end = sign + whole;
            if bytes.get(end) == Some(&b'.') {
                end += 1 + digits(end + 1);
            }
            if matches!(bytes.get(end), Some(b'e' | b'E')) {
                return Err("WKT coordinates with an exponent are not supported".into());
            }
            let text = self.rest()[..end].trim_start_matches('+').trim_end_matches('.');
            numbers.push(text.parse().map_err(|_| format!("'{text}' is not a number"))?);
            self.pos += end;
        }
        let keep = match (measure, numbers.len()) {
            (Measure::Plain, n @ (2 | 3)) => n,
            (Measure::Z, 3) => 3,
            (Measure::M, 3) => 2,
            (Measure::Zm, 4) => 3,
            _ => return Err("a WKT position has the wrong number of coordinates".into()),
        };
        Ok(json!(&numbers[..keep]))
    }
}

#[derive(Clone, Copy)]
enum Measure {
    Plain,
    Z,
    M,
    Zm,
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        _ => 28,
    }
}

/// Days since 1970-01-01 of a civil date, and back.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// A timestamp the way the library writes it: in UTC with a `Z`, seconds always there, and
/// no trailing zeros in the fraction. A value that is not a real date and time is left as
/// it is, for the check that follows to judge.
fn normalize_timestamp(text: &str) -> String {
    let parsed = (|| {
        let b = text.as_bytes();
        let num = |from: usize, len: usize| -> Option<i64> {
            let part = b.get(from..from + len)?;
            part.iter().all(u8::is_ascii_digit).then(|| part.iter().fold(0i64, |n, d| n * 10 + i64::from(d - b'0')))
        };
        let (year, month, day) = (num(0, 4)?, num(5, 2)?, num(8, 2)?);
        if b.get(4) != Some(&b'-') || b.get(7) != Some(&b'-') || !matches!(b.get(10), Some(b'T' | b' ')) {
            return None;
        }
        let (hour, minute) = (num(11, 2)?, num(14, 2)?);
        if b.get(13) != Some(&b':') {
            return None;
        }
        let mut at = 16;
        let mut second = 0;
        let mut fraction = "";
        if b.get(at) == Some(&b':') {
            second = num(at + 1, 2)?;
            at += 3;
            if b.get(at) == Some(&b'.') {
                let len = b[at + 1..].iter().take_while(|d| d.is_ascii_digit()).count();
                if len == 0 || len > 9 {
                    return None;
                }
                fraction = text[at + 1..at + 1 + len].trim_end_matches('0');
                at += 1 + len;
            }
        }
        let offset_minutes = match &text[at..] {
            "" | "Z" | "z" => 0,
            zone => {
                let z = zone.as_bytes();
                let sign = match z.first()? {
                    b'+' => 1,
                    b'-' => -1,
                    _ => return None,
                };
                if z.len() != 6 || z[3] != b':' {
                    return None;
                }
                let (h, m) = (num(at + 1, 2)?, num(at + 4, 2)?);
                if h > 23 || m > 59 {
                    return None;
                }
                sign * (h * 60 + m)
            }
        };
        let real = (1..=12).contains(&month)
            && (1..=days_in_month(year, month)).contains(&day)
            && hour < 24
            && minute < 60
            && second < 60;
        if !real {
            return None;
        }
        let minutes = days_from_civil(year, month, day) * 1440 + hour * 60 + minute - offset_minutes;
        let (y, mo, d) = civil_from_days(minutes.div_euclid(1440));
        let of_day = minutes.rem_euclid(1440);
        if !(0..=9999).contains(&y) {
            return None;
        }
        let dot = if fraction.is_empty() { "" } else { "." };
        Some(format!(
            "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{second:02}{dot}{fraction}Z",
            of_day / 60,
            of_day % 60
        ))
    })();
    parsed.unwrap_or_else(|| text.to_string())
}

fn all_digits(text: &str, lengths: &[usize], separators: &[u8]) -> bool {
    let b = text.as_bytes();
    let mut at = 0;
    for (i, len) in lengths.iter().enumerate() {
        if b.len() < at + len || !b[at..at + len].iter().all(u8::is_ascii_digit) {
            return false;
        }
        at += len;
        if let Some(sep) = separators.get(i) {
            if b.get(at) != Some(sep) {
                return false;
            }
            at += 1;
        }
    }
    at == b.len()
}

fn is_date(text: &str) -> bool {
    all_digits(text, &[4, 2, 2], b"--")
}

/// `YYYY-MM-DDTHH:MM:SS[.fff]Z`, the only form the CQL2 schema takes.
fn is_timestamp(text: &str) -> bool {
    let Some(body) = text.strip_suffix('Z') else {
        return false;
    };
    let (seconds, fraction) = body.split_once('.').map_or((body, None), |(s, f)| (s, Some(f)));
    all_digits(seconds, &[4, 2, 2, 2, 2, 2], b"--T::")
        && fraction.is_none_or(|f| !f.is_empty() && f.bytes().all(|d| d.is_ascii_digit()))
}

fn op_of(v: &Value) -> Option<(&str, &[Value])> {
    Some((v.get("op")?.as_str()?, v.get("args")?.as_array()?))
}

fn is_property(v: &Value) -> bool {
    v.get("property").is_some_and(Value::is_string)
}

/// A function call that is not one of the language's own operators.
fn is_function(v: &Value) -> bool {
    op_of(v).is_some_and(|(name, args)| {
        let reserved = ["and", "or", "not", "like", "between", "in", "isNull", "casei", "accenti"];
        !reserved.contains(&name)
            && !COMPARISON.contains(&name)
            && !ARITHMETIC.contains(&name)
            && !SPATIAL.contains(&name)
            && !TEMPORAL.contains(&name)
            && !ARRAY.contains(&name)
            && args.iter().all(|a| !a.is_null())
    })
}

fn is_numeric(v: &Value) -> bool {
    v.is_number()
        || is_property(v)
        || is_function(v)
        || op_of(v).is_some_and(|(name, args)| ARITHMETIC.contains(&name) && args.len() == 2 && args.iter().all(is_numeric))
}

/// A string-valued expression: a string, a property, a function, or one of those in
/// `casei` or `accenti`.
fn is_character(v: &Value) -> bool {
    v.is_string()
        || is_property(v)
        || is_function(v)
        || op_of(v).is_some_and(|(name, args)| matches!(name, "casei" | "accenti") && args.len() == 1 && is_character(&args[0]))
}

fn is_pattern(v: &Value) -> bool {
    v.is_string()
        || op_of(v).is_some_and(|(name, args)| matches!(name, "casei" | "accenti") && args.len() == 1 && is_pattern(&args[0]))
}

fn is_instant(v: &Value) -> bool {
    v.get("timestamp").and_then(Value::as_str).is_some_and(is_timestamp)
        || v.get("date").and_then(Value::as_str).is_some_and(is_date)
}

fn is_interval(v: &Value) -> bool {
    v.get("interval").and_then(Value::as_array).is_some_and(|ends| {
        ends.len() == 2
            && ends.iter().all(|e| match e {
                Value::String(s) => s == ".." || is_date(s) || is_timestamp(s),
                other => is_property(other) || is_function(other),
            })
    })
}

fn is_scalar(v: &Value) -> bool {
    v.is_boolean() || is_numeric(v) || is_character(v) || is_instant(v)
}

fn is_geometry(v: &Value) -> bool {
    fn min_points(coordinates: &Value, depth: usize, min: usize) -> bool {
        let Some(items) = coordinates.as_array() else {
            return false;
        };
        if depth == 0 { items.len() >= min } else { items.iter().all(|i| min_points(i, depth - 1, min)) }
    }
    if let Some(numbers) = v.get("bbox").and_then(Value::as_array) {
        return matches!(numbers.len(), 4 | 6) && numbers.iter().all(Value::is_number);
    }
    let Some(kind) = v.get("type").and_then(Value::as_str) else {
        return false;
    };
    let Some(coordinates) = v.get("coordinates") else {
        return v
            .get("geometries")
            .and_then(Value::as_array)
            .is_some_and(|g| g.len() >= 2 && g.iter().all(is_geometry));
    };
    match kind {
        "LineString" => min_points(coordinates, 0, 2),
        "MultiLineString" => min_points(coordinates, 1, 2),
        "Polygon" => min_points(coordinates, 1, 4),
        "MultiPolygon" => min_points(coordinates, 2, 4),
        _ => true,
    }
}

/// Whether a value is something the CQL2 schema takes where a truth value is expected.
fn is_boolean(v: &Value) -> bool {
    if v.is_boolean() || is_function(v) {
        return true;
    }
    let Some((name, args)) = op_of(v) else {
        return false;
    };
    match name {
        "and" | "or" => args.len() >= 2 && args.iter().all(is_boolean),
        "not" => args.len() == 1 && is_boolean(&args[0]),
        "like" => args.len() == 2 && is_character(&args[0]) && is_pattern(&args[1]),
        "between" => args.len() == 3 && args.iter().all(is_numeric),
        "in" => {
            args.len() == 2
                && is_scalar(&args[0])
                && args[1].as_array().is_some_and(|items| !items.is_empty() && items.iter().all(is_scalar))
        }
        "isNull" => args.len() == 1,
        _ if COMPARISON.contains(&name) => args.len() == 2 && args.iter().all(is_scalar),
        _ if SPATIAL.contains(&name) => {
            args.len() == 2 && args.iter().all(|a| is_property(a) || is_function(a) || is_geometry(a))
        }
        _ if TEMPORAL.contains(&name) => {
            args.len() == 2
                && args.iter().all(|a| is_property(a) || is_function(a) || is_instant(a) || is_interval(a))
        }
        _ if ARRAY.contains(&name) => {
            args.len() == 2 && args.iter().all(|a| a.is_array() || is_property(a) || is_function(a))
        }
        _ => false,
    }
}

/// Used by the tests: the JSON whether or not it passes the schema check.
#[cfg(test)]
fn parse_only(text: &str) -> Result<(Value, bool), String> {
    let mut parser = Parser { src: text, pos: 0, depth: 0 };
    let expr = parser.expr()?;
    parser.skip_space();
    if parser.pos != parser.src.len() {
        return Err(parser.unexpected());
    }
    let valid = is_boolean(&expr);
    Ok((expr, valid))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text, and what the `cql2` library (0.6.0, as shipped with titiler-pgstac 3.2.2) gives
    /// for it: its JSON and whether its `validate()` passes, or `null` where it cannot parse.
    const VECTORS: &str = include_str!("cql2_text_vectors.json");

    #[test]
    fn json_given_as_text_is_written_as_the_reference_writes_it() {
        let tidy = |text: &str| json_to_json(&serde_json::from_str(text).unwrap()).map(|v| v.to_string());
        let p = r#"{"property":"eo:cloud_cover"}"#;
        // What titiler-pgstac 3.2.2 registered for each, read back from its `info`.
        for (given, written) in [
            (
                format!(r#"{{"op":"AND","args":[{{"op":"<","args":[{p},50]}},{{"op":"Not","args":[{{"op":"IsNull","args":[{p}]}}]}}]}}"#),
                format!(r#"{{"args":[{{"args":[{p},50.0],"op":"<"}},{{"args":[{{"args":[{p}],"op":"isNull"}}],"op":"not"}}],"op":"and"}}"#),
            ),
            (format!(r#"{{"op":"IN","args":[{p},[1,2]]}}"#), format!(r#"{{"args":[{p},[1.0,2.0]],"op":"in"}}"#)),
            (format!(r#"{{"op":"BETWEEN","args":[{p},1,2]}}"#), format!(r#"{{"args":[{p},1.0,2.0],"op":"between"}}"#)),
            (format!(r#"{{"op":"<","args":[{p},50],"extra":1}}"#), format!(r#"{{"args":[{p},50.0],"op":"<"}}"#)),
            (format!(r#"{{"op":"=","args":[{p},true]}}"#), format!(r#"{{"args":[{p},true],"op":"="}}"#)),
            (
                r#"{"op":"T_AFTER","args":[{"property":"datetime"},{"timestamp":"2020-01-01T00:00:00+02:00"}]}"#.to_string(),
                r#"{"args":[{"property":"datetime"},{"timestamp":"2019-12-31T22:00:00Z"}],"op":"t_after"}"#.to_string(),
            ),
            (
                r#"{"op":"t_during","args":[{"property":"datetime"},{"interval":["2020-01-01","2021-01-01T00:00:00.120Z"]}]}"#.to_string(),
                r#"{"args":[{"property":"datetime"},{"interval":["2020-01-01","2021-01-01T00:00:00.12Z"]}],"op":"t_during"}"#.to_string(),
            ),
            (
                r#"{"op":"S_INTERSECTS","args":[{"property":"geometry"},{"type":"Point","coordinates":[8,45],"extra":1}]}"#.to_string(),
                r#"{"args":[{"property":"geometry"},{"coordinates":[8.0,45.0],"extra":1.0,"type":"Point"}],"op":"s_intersects"}"#.to_string(),
            ),
            (
                r#"{"op":"LIKE","args":[{"property":"platform"},"sen%"]}"#.to_string(),
                r#"{"args":[{"property":"platform"},"sen%"],"op":"like"}"#.to_string(),
            ),
        ] {
            assert_eq!(tidy(&given).as_deref(), Ok(written.as_str()), "{given}");
        }
        // What it answered 422 for.
        for bad in [
            "{}".to_string(),
            r#"{"foo":1}"#.to_string(),
            r#"{"property":"x"}"#.to_string(),
            "true".to_string(),
            format!(r#"{{"op":"=","args":[{p}]}}"#),
            format!(r#"{{"op":"=","args":[{p},null]}}"#),
            r#"{"op":"and","args":[true]}"#.to_string(),
            format!(r#"{{"op":"=","args":[{p},{{"timestamp":"nope"}}]}}"#),
            format!(r#"{{"op":"=","args":[{p},{{"function":{{"name":"f","args":[1]}}}}]}}"#),
            r#"{"op":"s_intersects","args":[{"property":"geometry"},{"type":"GeometryCollection","geometries":[]}]}"#.to_string(),
        ] {
            assert!(tidy(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn matches_the_reference_converter() {
        let vectors: Vec<Value> = serde_json::from_str(VECTORS).unwrap();
        assert!(vectors.len() > 300);
        let mut wrong = Vec::new();
        for case in &vectors {
            let text = case["text"].as_str().unwrap();
            let ours = to_json(text).ok();
            // What the reference accepts and validates has to come out the same; what it
            // rejects, at either step, has to be rejected.
            let expected = (case["valid"] == true).then(|| case["json"].clone()).filter(Value::is_object);
            if case["quirk"] == true {
                assert!(ours.is_none(), "{text}: {ours:?}");
            } else if ours != expected {
                wrong.push(format!("{text}\n   ours     {}\n   expected {}", json!(ours), json!(expected)));
            }
        }
        assert!(wrong.is_empty(), "{} of {} differ:\n{}", wrong.len(), vectors.len(), wrong.join("\n"));
    }

    #[test]
    fn numbers_are_floats_in_the_json_text() {
        let json = to_json("eo:cloud_cover < 20 AND gsd = 10.5").unwrap();
        assert_eq!(
            json.to_string(),
            r#"{"args":[{"args":[{"property":"eo:cloud_cover"},20.0],"op":"<"},{"args":[{"property":"gsd"},10.5],"op":"="}],"op":"and"}"#
        );
    }

    #[test]
    fn timestamps_are_brought_to_utc() {
        for (given, expected) in [
            ("2024-12-01T00:00:00Z", "2024-12-01T00:00:00Z"),
            ("2024-12-01T00:00:00.120Z", "2024-12-01T00:00:00.12Z"),
            ("2024-12-01T00:00:00.000Z", "2024-12-01T00:00:00Z"),
            ("2024-12-01 00:00Z", "2024-12-01T00:00:00Z"),
            ("2024-01-01T00:00:00+01:00", "2023-12-31T23:00:00Z"),
            ("2024-02-29T23:30:00-05:30", "2024-03-01T05:00:00Z"),
            ("2024-12-01T00:00:00", "2024-12-01T00:00:00Z"),
            ("2024-02-30T00:00:00Z", "2024-02-30T00:00:00Z"),
            ("nope", "nope"),
        ] {
            assert_eq!(normalize_timestamp(given), expected, "{given}");
        }
        assert_eq!(civil_from_days(days_from_civil(2000, 2, 29)), (2000, 2, 29));
        assert_eq!(days_from_civil(1970, 1, 1), 0);
    }

    #[test]
    fn bad_filters_are_errors() {
        for text in ["", "a =", "a = 1 AND", "(a = 1", "a = 'x", "a", "1", "TRUE", "a + 1", "a = NULL"] {
            assert!(to_json(text).is_err(), "{text}");
        }
        // Deep nesting is refused, not followed.
        let deep = format!("{}a = 1{}", "(".repeat(500), ")".repeat(500));
        assert_eq!(to_json(&deep).unwrap_err(), "the filter is nested too deeply");
        assert!(parse_only("a").is_ok_and(|(_, valid)| !valid));
    }
}
