//! TiTiler's algorithms' parameters as their pydantic models check them: each algorithm reads
//! its own fields, in pydantic's lax mode (`"45"`, `45.0` and `true` are all the integer they
//! spell), within its bounds, and passes over any other key. What it refuses TiTiler answers
//! `400` with pydantic's words, which [`check`] writes the same way (but for the line that
//! links to pydantic's documentation of its version). One field is wider than TiTiler's:
//! `contours`' `increment` may be a fraction.

use serde_json::Value;

use crate::algorithms::{ParamDef, ParamKind, lookup_algorithm};

/// A value as Python writes it in pydantic's messages, and the name of its type.
fn python(v: &Value) -> (String, &'static str) {
    match v {
        Value::Null => ("None".into(), "NoneType"),
        Value::Bool(b) => ((if *b { "True" } else { "False" }).into(), "bool"),
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            (if f.fract() == 0.0 && f.abs() < 1e16 { format!("{f:.1}") } else { format!("{f}") }, "float")
        }
        Value::Number(n) => (n.to_string(), "int"),
        Value::String(s) => (format!("'{s}'"), "str"),
        Value::Array(_) => (v.to_string(), "list"),
        Value::Object(_) => (v.to_string(), "dict"),
    }
}

/// A bound as pydantic prints it: a whole number without its point, `1e-06` as `0.000001`.
fn bound(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 { format!("{}", x as i64) } else { format!("{x}") }
}

/// The number a field holds, as pydantic's lax mode reads it, or what it says instead: the
/// message and its error type.
fn lax(kind: ParamKind, v: &Value) -> Result<Option<f64>, (String, &'static str)> {
    let int = matches!(kind, ParamKind::Int);
    let number = match v {
        Value::Null if matches!(kind, ParamKind::OptionalFloat) => return Ok(None),
        Value::Null if int => return Err(("Input should be a valid integer".into(), "int_type")),
        Value::Null => return Err(("Input should be a valid number".into(), "float_type")),
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) => {
            let s = s.trim();
            let parsed = if int { s.parse::<i64>().ok().map(|n| n as f64) } else { s.parse::<f64>().ok() };
            match parsed {
                Some(n) => n,
                None if int => return Err(("Input should be a valid integer, unable to parse string as an integer".into(), "int_parsing")),
                None => return Err(("Input should be a valid number, unable to parse string as a number".into(), "float_parsing")),
            }
        }
        _ if int => return Err(("Input should be a valid integer".into(), "int_type")),
        _ => return Err(("Input should be a valid number".into(), "float_type")),
    };
    if int && number.fract() != 0.0 {
        return Err(("Input should be a valid integer, got a number with a fractional part".into(), "int_from_float"));
    }
    Ok(Some(number))
}

/// What TiTiler's algorithm `name` (case-insensitive, aliases accepted) says of `params`, a JSON
/// object: `Ok` when its model takes them, else pydantic's message for every field it refuses.
pub(crate) fn check(name: &str, params: &serde_json::Map<String, Value>) -> Result<(), String> {
    let Some(def) = lookup_algorithm(name) else { return Ok(()) };
    let Some(title) = def.pydantic_name else { return Ok(()) };
    let mut errors = Vec::new();
    for ParamDef { name: key, kind, min: lo, max: hi, .. } in def.params {
        let Some(v) = params.get(*key) else { continue };
        let refused = match lax(*kind, v) {
            Err(refused) => Some(refused),
            Ok(Some(n)) if n < *lo => Some((format!("Input should be greater than or equal to {}", bound(*lo)), "greater_than_equal")),
            Ok(Some(n)) if n > *hi => Some((format!("Input should be less than or equal to {}", bound(*hi)), "less_than_equal")),
            Ok(_) => None,
        };
        if let Some((msg, kind)) = refused {
            let (value, ty) = python(v);
            errors.push(format!("{key}\n  {msg} [type={kind}, input_value={value}, input_type={ty}]"));
        }
    }
    if errors.is_empty() {
        return Ok(());
    }
    let count = if errors.len() == 1 { "1 validation error".to_string() } else { format!("{} validation errors", errors.len()) };
    Err(format!("{count} for {title}\n{}", errors.join("\n")))
}

/// A parameter's number as pydantic's lax mode reads it (`"45"`, `true`), for the values the
/// renderer takes; `None` where [`check`] refuses it or the key is not there.
pub(crate) fn number(params: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
    lax(ParamKind::Float, params.get(key)?).ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(name: &str, json: &str) -> Result<(), String> {
        check(name, serde_json::from_str::<Value>(json).unwrap().as_object().unwrap())
    }

    #[test]
    fn titilers_words_for_what_it_refuses() {
        // As TiTiler 2.4 and titiler-pgstac answer them, but for the documentation link.
        assert_eq!(
            run("hillshade", r#"{"azimuth":400}"#).unwrap_err(),
            "1 validation error for HillShade\nazimuth\n  Input should be less than or equal to 360 [type=less_than_equal, input_value=400, input_type=int]"
        );
        assert_eq!(
            run("hillshade", r#"{"buffer":null}"#).unwrap_err(),
            "1 validation error for HillShade\nbuffer\n  Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]"
        );
        assert_eq!(
            run("slope", r#"{"z_exaggeration":0}"#).unwrap_err(),
            "1 validation error for Slope\nz_exaggeration\n  Input should be greater than or equal to 0.000001 [type=greater_than_equal, input_value=0, input_type=int]"
        );
        assert!(run("hillshade", r#"{"angle_altitude":"x"}"#).unwrap_err().contains("[type=float_parsing, input_value='x', input_type=str]"));
        assert!(run("contours", r#"{"thickness":11,"increment":1000}"#).unwrap_err().starts_with("2 validation errors for Contours\nincrement"));
        // A fractional increment is taken, beyond TiTiler.
        assert!(run("contours", r#"{"increment":0.1}"#).is_ok());
        assert!(run("terrarium", r#"{"nodata_height":100000}"#).is_err());
        assert!(run("terrainrgb", r#"{"interval":2}"#).is_err());
        assert!(run("hillshade", r#"{"azimuth":45.5}"#).unwrap_err().contains("int_from_float"));
    }

    #[test]
    fn what_pydantic_takes_and_passes_over() {
        for ok in [r#"{"azimuth":"45"}"#, r#"{"azimuth":45.0}"#, r#"{"azimuth":true}"#, r#"{"foo":1}"#, r#"{"angle_altitude":-90}"#] {
            assert!(run("hillshade", ok).is_ok(), "{ok}");
        }
        assert!(run("terrarium", r#"{"nodata_height":null}"#).is_ok());
        // An algorithm without parameters takes whatever it is given.
        assert!(run("cast", r#"{"azimuth":400}"#).is_ok());
        let params = serde_json::from_str::<Value>(r#"{"azimuth":"30","buffer":true,"x":null}"#).unwrap();
        let params = params.as_object().unwrap();
        assert_eq!((number(params, "azimuth"), number(params, "buffer"), number(params, "x")), (Some(30.0), Some(1.0), None));
    }
}
