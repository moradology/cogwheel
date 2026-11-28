//! TiTiler's one-output algorithms as band math for the expression VM, worked as TiTiler
//! works them: `std` and `var` over the bands as a sample (`ddof=1`), `grayscale` truncated
//! back to the whole numbers it was given, and `cast`, `ceil` and `floor` wrapped into
//! their integer type as numpy's `astype` wraps (300 is 44 and -1 is 255 in uint8).
//!
//! `whole` is the type of the bands when they are read as a raster's own whole numbers
//! (an integer type, not unscaled), and `None` when they may have fractions (a float
//! raster, unscaled values, or a type not known): only then do they need rounding.

use cogzilla_raster::SampleType;

use super::array_type::cast_type;

/// The type `whole` asks for: `source` when its bands are read as its own whole numbers.
pub(super) fn whole_numbers(source: SampleType, unscale: bool) -> Option<SampleType> {
    (!unscale && source.storable_range().is_some()).then_some(source)
}

/// `b{band}` as TiTiler's `cast` (to `dtype`, uint8 when none is named), `ceil` or
/// `floor` leaves it: rounded (`cast` truncates toward zero), then wrapped into a type of up
/// to 16 bits as [`super::array_type::astype`] wraps, and the type's range, which a picture
/// of it spans. A `cast` to a float type, or to a name that is not a type, leaves the band
/// as it is, with no range of its own.
pub(super) fn integer_band(alg: &str, band: u16, whole: Option<SampleType>, dtype: Option<&str>) -> (String, Option<(f32, f32)>) {
    let b = format!("b{band}");
    let kind = if alg == "cast" { cast_type(dtype) } else { Some(SampleType::Uint8) };
    let Some((kind, (lo, hi))) = kind.and_then(|kind| Some((kind, kind.storable_range()?))) else {
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
    let fits = whole.and_then(SampleType::storable_range).is_some_and(|(from, to)| from >= lo && to <= hi);
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
        // One sample has no spread under `ddof=1`: numpy's NaN, which TiTiler masks.
        "var" | "std" => format!("(b{0} - b{0}) / 0", bands[0]),
        // TiTiler fails (a `500`) on fewer than three bands; its words, as a `400`.
        "grayscale" if rgb.is_none() => {
            return Err(format!("Cannot apply `grayscale` algorithm on image with {} bands.", bands.len()));
        }
        "bitonal" if bands.len() == 2 => {
            return Err("Cannot apply `bitonal` algorithm on image with 2 bands.".into());
        }
        "grayscale" => match (rgb, whole) {
            (None, _) => unreachable!("fewer than three bands is refused above"),
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

/// What an algorithm worked over band math gives ([`over_results`]).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Composed {
    /// Each output's description and band math.
    pub(super) outputs: Vec<(String, String)>,
    /// The ops the outputs may compile to: as many as the band math they are made of took,
    /// twice over where the algorithm reads an output twice (`normalizedIndex`).
    pub(super) max_ops: usize,
    /// The type TiTiler's algorithm writes its outputs in, when the band math's is known.
    pub(super) kind: Option<SampleType>,
}

/// Band math and TiTiler's one-output algorithm `alg` together, as rio-tiler works them: the
/// expression's outputs first, then the algorithm over them as its bands (a post-process of
/// what was read). `parts` are each output's description and band math, and what comes back
/// are the algorithm's outputs, described as TiTiler describes them: `cast`, `ceil` and
/// `floor` keep each output and its description, `normalizedIndex` writes its formula over
/// the descriptions, and the rest are one output named after the algorithm. `None` for an
/// algorithm that is not band math (a terrain or DEM algorithm draws from the bands).
///
/// Every part is checked as band math of its own first, whether the algorithm reads it or
/// not (`normalizedIndex` reads two). `typed` is the type of the parts' values, when it is
/// known (as numexpr types them, after any asset's own band math): whole numbers are what
/// `grayscale` truncates back to and what needs no rounding.
pub(super) fn over_results(
    alg: &str,
    typed: Option<SampleType>,
    cast_dtype: Option<&str>,
    threshold: f32,
    parts: &[(String, String)],
) -> Result<Option<Composed>, super::EngineError> {
    let alg = alg.to_ascii_lowercase();
    if parts.is_empty() {
        return Ok(None);
    }
    let one_output =
        matches!(alg.as_str(), "normalizedindex" | "min" | "max" | "sum" | "mean" | "median" | "var" | "std" | "grayscale" | "bitonal");
    if !one_output && !matches!(alg.as_str(), "cast" | "ceil" | "floor") {
        return Ok(None);
    }
    // As many outputs as band math may have, before any are folded into one.
    if parts.len() > 16 {
        return Err(super::EngineError::BadRequest("expression exceeds max 16 semicolon-separated outputs".into()));
    }
    let mut ops = 0usize;
    for (_, text) in parts {
        ops += cogzilla_kernel::ExprProgram::compile(text).map_err(super::EngineError::BadRequest)?.ops.len();
    }
    let max_ops = 2 * ops + 64;
    let whole = typed.and_then(|kind| whole_numbers(kind, false));
    let kind = typed.and_then(|kind| {
        let cast = if alg == "cast" { super::array_type::cast_type(cast_dtype) } else { None };
        crate::algorithms::lookup_algorithm(&alg)?.out_dtype.resolve(cast, kind)
    });
    if !one_output {
        let one = integer_band(&alg, 1, whole, cast_dtype).0;
        let outputs = parts
            .iter()
            .map(|(desc, text)| Ok((desc.clone(), super::compose_over_results(&one, std::slice::from_ref(text))?)))
            .collect::<Result<Vec<_>, super::EngineError>>()?;
        return Ok(Some(Composed { outputs, max_ops, kind }));
    }
    let texts: Vec<String> = parts.iter().map(|(_, text)| text.clone()).collect();
    let bands: Vec<u16> = (1..=parts.len().min(usize::from(u16::MAX)) as u16).collect();
    let expr = band_math(&alg, &bands, whole, cast_dtype, threshold).map_err(super::EngineError::BadRequest)?;
    let desc = if alg == "normalizedindex" {
        super::substitute_band_tokens(&expr, |k| parts.get(k.wrapping_sub(1)).map(|(desc, _)| desc.clone()))
    } else {
        alg.clone()
    };
    Ok(Some(Composed { outputs: vec![(desc, super::compose_over_results(&expr, &texts)?)], max_ops, kind }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_kernel::ExprProgram;

    fn eval(expr: &str, values: &[f32]) -> f32 {
        ExprProgram::compile_with_limit(expr, 1024).unwrap().eval(values)
    }

    #[test]
    fn an_algorithm_works_over_band_math_results() {
        let parts = |exprs: &[&str]| exprs.iter().map(|e| (e.to_string(), e.to_string())).collect::<Vec<_>>();
        let over = |alg: &str, exprs: &[&str]| over_results(alg, None, None, 127.0, &parts(exprs)).unwrap().unwrap().outputs;
        // floor(77 / 2) == 38, each output kept with its description.
        let floor = over("floor", &["b1/2", "b1*300"]);
        assert_eq!(floor.iter().map(|(d, _)| d.as_str()).collect::<Vec<_>>(), ["b1/2", "b1*300"]);
        assert_eq!(eval(&floor[0].1, &[77.0]), 38.0);
        assert_eq!(eval(&floor[1].1, &[1.0]), 44.0);
        // One output, named after the algorithm; normalizedIndex's formula over the parts.
        let max = over("max", &["b1/2", "b1*3"]);
        assert_eq!(max.len(), 1);
        assert_eq!(max[0].0, "max");
        assert_eq!(eval(&max[0].1, &[10.0]), 30.0);
        let ndi = over("normalizedIndex", &["b1/2", "b1"]);
        assert_eq!(ndi[0].0, "(b1 - b1/2) / (b1 + b1/2)");
        assert!((eval(&ndi[0].1, &[6.0]) - 1.0 / 3.0).abs() < 1e-6);
        // Not band math, and nothing to work over.
        assert!(over_results("hillshade", None, None, 127.0, &parts(&["b1"])).unwrap().is_none());
        assert!(over_results("floor", None, None, 127.0, &[]).unwrap().is_none());
        assert!(over_results("normalizedIndex", None, None, 127.0, &parts(&["b1"])).is_err());
        // A part the algorithm does not read is still band math to be checked.
        assert!(over_results("normalizedIndex", None, None, 127.0, &parts(&["b1", "b2", "bogus("])).is_err());
        // Over whole numbers, `grayscale` truncates back to them, as TiTiler casts to the
        // type it was given; over fractions it does not.
        let gray = |typed| over_results("grayscale", typed, None, 127.0, &parts(&["b1", "b2", "b3"])).unwrap().unwrap();
        assert_eq!(eval(&gray(Some(SampleType::Uint8)).outputs[0].1, &[101.0, 200.0, 50.0]), 153.0);
        assert!((eval(&gray(Some(SampleType::Float64)).outputs[0].1, &[101.0, 200.0, 50.0]) - 153.299).abs() < 1e-3);
        assert!((eval(&gray(None).outputs[0].1, &[101.0, 200.0, 50.0]) - 153.299).abs() < 1e-3);
        // The algorithm's own type: normalizedIndex over uint16 band math is float32.
        let typed = over_results("normalizedIndex", Some(SampleType::Uint16), None, 127.0, &parts(&["b1", "b2"])).unwrap().unwrap();
        assert_eq!(typed.kind, Some(SampleType::Float32));
        // Room for an output read twice: two parts of 65 ops each make 263 under normalizedIndex.
        let long = format!("b1{}", "+1".repeat(32));
        let long2 = long.replace("b1", "b2");
        let ndi = over_results("normalizedIndex", None, None, 127.0, &parts(&[&long, &long2])).unwrap().unwrap();
        let program = ExprProgram::compile_with_limit(&ndi.outputs[0].1, ndi.max_ops).unwrap();
        assert!(program.ops.len() > ExprProgram::MAX_OPS);
        // No more parts than band math may have outputs, even where they fold into one.
        let many = vec!["b1"; 17];
        assert!(over_results("sum", None, None, 127.0, &parts(&many)).is_err());
        assert!(over_results("sum", None, None, 127.0, &parts(&many[..16])).unwrap().is_some());
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
        // TiTiler's words for what it fails on, and one sample's spread masked.
        assert_eq!(band_math("grayscale", &[2], Some(T::Uint8), None, 127.0).unwrap_err(), "Cannot apply `grayscale` algorithm on image with 1 bands.");
        assert_eq!(band_math("bitonal", &[1, 2], None, None, 127.0).unwrap_err(), "Cannot apply `bitonal` algorithm on image with 2 bands.");
        assert!(eval(&band_math("std", &[1], None, None, 127.0).unwrap(), &[5.0]).is_nan());
        assert!(eval(&band_math("var", &[1], None, None, 127.0).unwrap(), &[5.0]).is_nan());
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
