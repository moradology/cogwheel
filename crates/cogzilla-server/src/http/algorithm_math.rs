//! TiTiler's one-output algorithms as band math for the expression VM, worked as TiTiler
//! works them: `std` and `var` over the bands as a sample (`ddof=1`), `grayscale` truncated
//! back to the whole numbers it was given, and `cast`, `ceil` and `floor` wrapped into
//! their integer type as numpy's `astype` wraps (300 is 44 and -1 is 255 in uint8).
//!
//! `whole` is the type of the bands when they are read as a raster's own whole numbers
//! (an integer type, not unscaled), and `None` when they may have fractions (a float
//! raster, unscaled values, or a type not known): only then do they need rounding.

use cogzilla_tiff::SampleType;

use super::array_type::{cast_type, integer_range};

/// The type `whole` asks for: `source` when its bands are read as its own whole numbers.
pub(super) fn whole_numbers(source: SampleType, unscale: bool) -> Option<SampleType> {
    (!unscale && integer_range(source).is_some()).then_some(source)
}

/// `b{band}` as TiTiler's `cast` (to `dtype`, uint8 when none is named), `ceil` or
/// `floor` leaves it: rounded (`cast` truncates toward zero), then wrapped into a type of up
/// to 16 bits as [`super::array_type::astype`] wraps, and the type's range, which a picture
/// of it spans. A `cast` to a float type, or to a name that is not a type, leaves the band
/// as it is, with no range of its own.
pub(super) fn integer_band(alg: &str, band: u16, whole: Option<SampleType>, dtype: Option<&str>) -> (String, Option<(f32, f32)>) {
    let b = format!("b{band}");
    let kind = if alg == "cast" { cast_type(dtype) } else { Some(SampleType::Uint8) };
    let Some((kind, (lo, hi))) = kind.and_then(|kind| Some((kind, integer_range(kind)?))) else {
        return (b, None);
    };
    let rounded = match (whole, alg) {
        (Some(_), _) => b,
        (None, "ceil") => format!("ceil({b})"),
        (None, "floor") => format!("floor({b})"),
        (None, _) => format!("trunc({b})"),
    };
    // Whole numbers already inside the type's range are what `astype` leaves them, and
    // 32-bit types are only truncated, as `astype` does.
    let fits = whole.and_then(integer_range).is_some_and(|(from, to)| from >= lo && to <= hi);
    let expr = if fits || kind.bytes_per_sample() > 2 {
        rounded
    } else if lo == 0.0 {
        format!("{rounded} % {}", hi + 1.0)
    } else {
        format!("({rounded} + {}) % {} - {}", -lo, hi - lo + 1.0, -lo)
    };
    (expr, Some((lo as f32, hi as f32)))
}

/// The band math of TiTiler's one-output algorithm `alg` (lowercase) over `bands`, its
/// `cast` to `cast_dtype` and its `bitonal` split at `threshold`; an error for an algorithm
/// that is not band math, or `normalizedIndex` over fewer than two bands.
pub(super) fn band_math(
    alg: &str,
    bands: &[u16],
    whole: Option<SampleType>,
    cast_dtype: Option<&str>,
    threshold: f32,
) -> Result<String, String> {
    let list = |sep: &str| bands.iter().map(|b| format!("b{b}")).collect::<Vec<_>>().join(sep);
    // TiTiler's ITU-R 601-2 luma, as floats; `grayscale` truncates it for whole numbers.
    let luma = |[r, g, b]: [u16; 3]| format!("0.299 * b{r} + 0.587 * b{g} + 0.114 * b{b}");
    let rgb = (bands.len() >= 3).then(|| [bands[0], bands[1], bands[2]]);
    Ok(match alg {
        "normalizedindex" => {
            let [b1, b2] = match bands {
                [b1, b2, ..] => [b1, b2],
                _ => return Err("algorithm 'normalizedIndex' requires at least 2 bands".into()),
            };
            format!("(b{b2} - b{b1}) / (b{b2} + b{b1})")
        }
        "min" | "max" => bands[1..].iter().fold(format!("b{}", bands[0]), |acc, b| format!("{alg}({acc}, b{b})")),
        "sum" => list(" + "),
        "mean" => format!("({}) / {}", list(" + "), bands.len()),
        "median" => format!("median({})", list(", ")),
        // A sample's spread, numpy's `ddof=1`: the population's scaled by n / (n - 1).
        "var" | "std" if bands.len() >= 2 => {
            let n = bands.len() as f64;
            let scale = if alg == "var" { n / (n - 1.0) } else { (n / (n - 1.0)).sqrt() };
            format!("{alg}({}) * {scale}", list(", "))
        }
        "var" | "std" => format!("{alg}({})", list(", ")),
        "grayscale" => match (rgb, whole) {
            (None, _) => format!("b{}", bands[0]),
            (Some(rgb), None) => luma(rgb),
            // In whole numbers the sum is exact and one division rounds it once, as close to
            // numpy's float64 as single precision comes; then truncated toward zero.
            (Some([r, g, b]), Some(_)) => format!("trunc((b{r} * 299 + b{g} * 587 + b{b} * 114) / 1000)"),
        },
        "bitonal" => {
            let src = rgb.map_or_else(|| format!("b{}", bands[0]), |rgb| format!("({})", luma(rgb)));
            format!("where({src} > {threshold}, 255, 0)")
        }
        "cast" | "ceil" | "floor" => integer_band(alg, bands[0], whole, cast_dtype).0,
        _ => return Err(format!("algorithm '{alg}' is not band math")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_kernel::ExprProgram;

    fn eval(expr: &str, values: &[f32]) -> f32 {
        ExprProgram::compile(expr).unwrap().eval(values)
    }

    #[test]
    fn cast_ceil_and_floor_wrap_into_uint8_as_astype_does() {
        use SampleType as T;
        let float = |alg: &str, v: f32| eval(&integer_band(alg, 1, None, None).0, &[v]);
        assert_eq!(float("cast", 300.0), 44.0);
        assert_eq!(float("cast", -1.0), 255.0);
        assert_eq!(float("cast", -300.7), 212.0);
        assert_eq!(float("cast", 255.9), 255.0);
        assert_eq!(float("ceil", 255.2), 0.0);
        assert_eq!(float("floor", -0.5), 255.0);
        assert!(float("cast", f32::NAN).is_nan());
        // Whole numbers need no rounding, and those that fit need no wrapping either.
        assert_eq!(integer_band("cast", 2, Some(T::Uint16), None).0, "b2 % 256");
        assert_eq!(integer_band("ceil", 1, Some(T::Uint8), None), ("b1".to_string(), Some((0.0, 255.0))));
        assert_eq!(eval(&integer_band("floor", 1, Some(T::Int16), None).0, &[-1.0]), 255.0);
        let int16 = integer_band("cast", 1, None, Some("int16"));
        assert_eq!(int16.1, Some((-32768.0, 32767.0)));
        assert_eq!(eval(&int16.0, &[40_000.0]), -25_536.0);
        assert_eq!(eval(&int16.0, &[-2.5]), -2.0);
        assert_eq!(integer_band("cast", 1, Some(T::Int16), Some("int32")).0, "b1");
        assert_eq!(integer_band("cast", 1, None, Some("float32")), ("b1".to_string(), None));
    }

    #[test]
    fn std_and_var_spread_over_a_sample() {
        let var = band_math("var", &[1, 2, 3], None, None, 127.0).unwrap();
        let std = band_math("std", &[1, 2, 3], None, None, 127.0).unwrap();
        // numpy.var([1, 2, 6], ddof=1) == 7.0
        assert!((eval(&var, &[1.0, 2.0, 6.0]) - 7.0).abs() < 1e-5);
        assert!((eval(&std, &[1.0, 2.0, 6.0]) - 7.0f32.sqrt()).abs() < 1e-5);
        assert!(eval(&var, &[1.0, f32::NAN, 6.0]).is_nan());
    }

    #[test]
    fn grayscale_truncates_whole_numbers_back() {
        use SampleType as T;
        let gray = |whole| band_math("grayscale", &[1, 2, 3], whole, None, 127.0).unwrap();
        // 100*0.299 + 200*0.587 + 50*0.114 == 153.0 exactly; 101 makes 153.299.
        assert_eq!(eval(&gray(Some(T::Uint8)), &[100.0, 200.0, 50.0]), 153.0);
        assert_eq!(eval(&gray(Some(T::Uint8)), &[101.0, 200.0, 50.0]), 153.0);
        assert_eq!(eval(&gray(Some(T::Int16)), &[-101.0, -200.0, -50.0]), -153.0);
        assert!((eval(&gray(None), &[101.0, 200.0, 50.0]) - 153.299).abs() < 1e-3);
        assert_eq!(band_math("grayscale", &[2], Some(T::Uint8), None, 127.0).unwrap(), "b2");
    }

    #[test]
    fn other_algorithms_are_their_band_math() {
        assert_eq!(band_math("normalizedindex", &[1, 2], None, None, 0.0).unwrap(), "(b2 - b1) / (b2 + b1)");
        assert!(band_math("normalizedindex", &[1], None, None, 0.0).is_err());
        assert_eq!(band_math("max", &[1, 2, 3], None, None, 0.0).unwrap(), "max(max(b1, b2), b3)");
        assert_eq!(band_math("mean", &[1, 2], None, None, 0.0).unwrap(), "(b1 + b2) / 2");
        assert_eq!(band_math("bitonal", &[1], None, None, 127.0).unwrap(), "where(b1 > 127, 255, 0)");
        assert!(band_math("hillshade", &[1], None, None, 0.0).is_err());
    }
}
