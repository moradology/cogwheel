//! The type an array of data is written in, as TiTiler writes it: the raster's own type,
//! the type that holds every band when several are stacked, and what numexpr makes of band
//! math. The renderers work in single precision; an array they write is rewritten in this
//! type before it is answered.

use cogzilla_tiff::SampleType;

use super::ParsedQuery;

/// The type that holds the numbers of both, as numpy promotes them when it stacks bands.
pub(crate) fn wider_type(a: SampleType, b: SampleType) -> SampleType {
    use SampleType as T;
    if a == b {
        return a;
    }
    let float = |t| matches!(t, T::Float32 | T::Float64);
    let signed = |t| matches!(t, T::Int8 | T::Int16 | T::Int32 | T::Int64);
    let bits = |t| match t {
        T::Uint1 | T::Uint4 | T::Uint8 | T::Int8 => 8u32,
        T::Uint16 | T::Int16 => 16,
        T::Uint32 | T::Int32 | T::Float32 => 32,
        T::Uint64 | T::Int64 | T::Float64 => 64,
    };
    if float(a) || float(b) {
        // float32 holds whole numbers of 16 bits and no more.
        let narrow = |t| if float(t) { bits(t) == 32 } else { bits(t) <= 16 };
        return if narrow(a) && narrow(b) { T::Float32 } else { T::Float64 };
    }
    let sized = |kinds: [SampleType; 4], width: u32| kinds[match width { 8 => 0, 16 => 1, 32 => 2, _ => 3 }];
    match (signed(a), signed(b)) {
        (false, false) => sized([T::Uint8, T::Uint16, T::Uint32, T::Uint64], bits(a).max(bits(b))),
        (true, true) => sized([T::Int8, T::Int16, T::Int32, T::Int64], bits(a).max(bits(b))),
        // A signed type one size up from the unsigned one, or the signed one's own if larger.
        _ => {
            let (own, unsigned) = if signed(a) { (bits(a), bits(b)) } else { (bits(b), bits(a)) };
            if own == 64 && unsigned < 64 {
                T::Int64
            } else {
                match (unsigned * 2).max(own) {
                    16 => T::Int16,
                    32 => T::Int32,
                    _ => T::Float64,
                }
            }
        }
    }
}

/// The type band math comes out in, as numexpr has it in TiTiler, from the type `read`
/// of the bands it is worked on: whole numbers of up to 16 bits stay whole, as 32-bit
/// ones, through sums, differences and products, and 32-bit floats stay so; a division, a
/// function or a fraction makes doubles.
pub(crate) fn band_math_type<'a>(expressions: impl Iterator<Item = &'a String> + Clone, read: SampleType) -> SampleType {
    let whole = |e: &String| {
        let mut rest = e.as_str();
        while let Some(at) = rest.find('b') {
            let digits = rest[at + 1..].bytes().take_while(u8::is_ascii_digit).count();
            if digits == 0 {
                return false;
            }
            rest = &rest[at + 1 + digits..];
        }
        !e.contains("**")
            && !e
                .bytes()
                .any(|c| c == b'/' || c == b'.' || c == b'^' || (c.is_ascii_alphabetic() && c != b'b'))
    };
    // float32 stays float32 through operations and functions alike; a literal with a
    // fraction or an exponent is a double and makes the result one.
    let plain = |e: &String| {
        let e = e.as_bytes();
        !e.contains(&b'.') && !e.windows(2).any(|w| w[0].is_ascii_digit() && matches!(w[1], b'e' | b'E'))
    };
    if read == SampleType::Float32 && expressions.clone().all(plain) {
        SampleType::Float32
    } else if !expressions.clone().all(whole) {
        SampleType::Float64
    } else {
        match read {
            SampleType::Uint1
            | SampleType::Uint4
            | SampleType::Uint8
            | SampleType::Int8
            | SampleType::Uint16
            | SampleType::Int16 => SampleType::Int32,
            _ => SampleType::Float64,
        }
    }
}

/// Whether a request turns values into a picture's bytes: a rescale, a colour map, a colour
/// formula. TiTiler's array of such a request is those bytes.
pub(super) fn shown(pq: &ParsedQuery) -> bool {
    !pq.rescale.is_empty()
        || pq.colormap_explicit
        || pq.colormap_name_set
        || pq.custom_colormap_json.is_some()
        || pq.color_formula.is_some()
}

/// What decides the type of a `/cog` or `/stac` array, read from the request.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ArrayRule {
    /// The bands as they were read, in the type that holds them all; float32 once unscaled.
    Data { unscale: bool },
    /// Band math over them.
    Math { unscale: bool },
}

impl ArrayRule {
    /// The rule for a request, or none when what was asked turns the values into a
    /// picture's bytes (a rescale, a colour map, a colour formula) or into an algorithm's
    /// output, or does band math on single assets: those arrays are written as they are.
    pub(super) fn of(pq: &ParsedQuery) -> Option<Self> {
        if shown(pq) || pq.algorithm.is_some() || !pq.asset_exprs.is_empty() {
            return None;
        }
        Some(match pq.expr_raw {
            Some(_) => Self::Math { unscale: pq.unscale },
            None => Self::Data { unscale: pq.unscale },
        })
    }

    /// The type for bands read as `reads`, with `expression` the band math as worked on
    /// (`b1`, `b2`, ... after any asset names are rewritten), several separated by `;`.
    pub(super) fn kind(&self, reads: &[SampleType], expression: Option<&str>) -> Option<SampleType> {
        let (unscale, math) = match self {
            Self::Data { unscale } => (*unscale, false),
            Self::Math { unscale } => (*unscale, true),
        };
        let first = *reads.first()?;
        let read = if unscale { SampleType::Float32 } else { reads[1..].iter().fold(first, |a, b| wider_type(a, *b)) };
        if !math {
            return Some(read);
        }
        let expressions: Vec<String> = expression?.split(';').map(str::to_string).collect();
        Some(band_math_type(expressions.iter(), read))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_bands_take_the_type_that_holds_them_all() {
        use SampleType as T;
        for (a, b, both) in [
            (T::Uint8, T::Uint16, T::Uint16), (T::Uint16, T::Int16, T::Int32), (T::Uint8, T::Int8, T::Int16),
            (T::Uint8, T::Int16, T::Int16), (T::Uint8, T::Int32, T::Int32), (T::Int8, T::Int16, T::Int16),
            (T::Uint16, T::Float32, T::Float32), (T::Uint32, T::Float32, T::Float64), (T::Float32, T::Float64, T::Float64),
            (T::Uint32, T::Int32, T::Float64), (T::Uint8, T::Uint8, T::Uint8), (T::Uint16, T::Uint32, T::Uint32),
        ] {
            assert_eq!(wider_type(a, b), both, "{a:?} with {b:?}");
            assert_eq!(wider_type(b, a), both, "{b:?} with {a:?}");
        }
    }

    #[test]
    fn band_math_keeps_whole_numbers_whole_until_it_divides() {
        use SampleType as T;
        let kind = |e: &str, read| band_math_type(std::iter::once(&e.to_string()), read);
        assert_eq!(kind("b1*2", T::Uint16), T::Int32);
        assert_eq!(kind("b1+b2", T::Int16), T::Int32);
        assert_eq!(kind("b1/2", T::Uint16), T::Float64);
        assert_eq!(kind("b1*0.5", T::Uint8), T::Float64);
        assert_eq!(kind("sqrt(b1)", T::Uint8), T::Float64);
        assert_eq!(kind("b1/2", T::Float32), T::Float32);
        assert_eq!(kind("b1*0.5", T::Float32), T::Float64);
        assert_eq!(kind("b1*2", T::Int32), T::Float64);
        assert_eq!(kind("b1**-1", T::Uint16), T::Float64);
        assert_eq!(kind("b1^-1", T::Uint8), T::Float64);
        assert_eq!(kind("b1**3", T::Uint16), T::Float64);
        assert_eq!(kind("b1**-1", T::Float32), T::Float32);
    }

    #[test]
    fn a_request_says_which_rule_its_array_follows() {
        let rule = |q: &str| ArrayRule::of(&super::super::parse_query(Some(q)));
        assert_eq!(rule("url=a.tif"), Some(ArrayRule::Data { unscale: false }));
        assert_eq!(rule("url=a.tif&unscale=true"), Some(ArrayRule::Data { unscale: true }));
        assert_eq!(rule("url=a.tif&expression=b1*2"), Some(ArrayRule::Math { unscale: false }));
        for shown in ["rescale=0,100", "colormap_name=viridis", "color_formula=gamma%20r%202", "algorithm=hillshade", "assets=a%7Cexpression=b1*2"] {
            assert_eq!(rule(&format!("url=a.tif&{shown}")), None, "{shown}");
        }
        use SampleType as T;
        assert_eq!(ArrayRule::Data { unscale: false }.kind(&[T::Uint16], None), Some(T::Uint16));
        assert_eq!(ArrayRule::Data { unscale: false }.kind(&[T::Uint8, T::Uint16, T::Int16], None), Some(T::Int32));
        assert_eq!(ArrayRule::Data { unscale: true }.kind(&[T::Int16], None), Some(T::Float32));
        assert_eq!(ArrayRule::Math { unscale: false }.kind(&[T::Uint16, T::Uint16], Some("b1+b2;b1*2")), Some(T::Int32));
        assert_eq!(ArrayRule::Math { unscale: false }.kind(&[T::Uint16], Some("(b2-b1)/(b2+b1)")), Some(T::Float64));
        assert_eq!(ArrayRule::Math { unscale: false }.kind(&[T::Uint16], None), None);
        assert_eq!(ArrayRule::Data { unscale: false }.kind(&[], None), None);
    }
}
