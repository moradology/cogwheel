//! titiler-pgstac's `/searches` and `/collections/{id}` mosaic routes, on the compatibility path,
//! and the item layer in front of the `/stac` handlers for `/collections/{id}/items/{id}`.
//!
//! The items behind a tile come from pgSTAC's own `geojsonsearch`, called with the
//! arguments titiler-pgstac passes, and are rendered by the existing mosaic engine.
//! Responses follow titiler-pgstac's shapes: JSON `{"detail": ...}` errors, `422` for a
//! missing required parameter, `204` for a tile no item covers.

// A rejected request is returned as the finished HTTP response, built once and moved once.
#![allow(clippy::result_large_err)]

use std::sync::Arc;

use bytes::Bytes;
use cogzilla_kernel::{ColormapName, OutputFormat, ResampleMode};
use cogzilla_pgstac::{
    Lookup, MosaicItem, PgStac, PgStacError, SampleGrid, SearchField, SearchLimits, SearchOrder, SearchRecord,
    jsonb_key_order,
};
use cogzilla_proj::mercator::lonlat_deg_to_mercator;
use cogzilla_proj::{CrsModel, TileCoord, mercator_to_lonlat_deg, tile_bounds_3857};
use cogzilla_raster::SampleType;
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CONTENT_TYPE, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{Value, json};

use super::{
    ParsedQuery, apply_unscale_cogs, build_band_request, build_tile_request, parse_bbox4,
    parse_ext_format, parse_y_scale_ext, percent_decode, read_limited_body,
    to_http_response_cached,
};
use super::array_type::{band_math_type, wider_type};
use super::paint::{Paint, formula_levels, rescale_byte};
use super::pixel_selection::{Fold, Selected, Selection};
use super::point_expr::{Kind, Program};
use super::zonal::{self, Bins, StatsOptions};
use crate::engine::{BandRequest, EngineError, MOSAIC_SCENES_AHEAD, TileEngine};
use crate::stac::{StacItem, stac_item_from_json};
mod tile;

use self::tile::{render_non_mercator_tms_tile, tile};
use crate::stats::{
    PolygonMask, build_map_html, encode_rgba_output_colour, query_point, render_crs_bbox, render_wgs84_bbox,
};

/// Most scenes one tile may read. titiler-pgstac's default `items_limit` is 100.
const MAX_SCENES_PER_TILE: usize = 256;
/// The engine composites one to three co-registered assets per scene.
const MAX_ASSETS_PER_SCENE: usize = 3;
/// The most assets of one item a tile or statistics will read, each on its own grid.
const MAX_ASSETS_PER_ITEM: usize = 16;

/// The parts of the request the routes need.
pub(super) struct Ctx<'a> {
    pub engine: &'a Arc<TileEngine>,
    /// `scheme://host[/prefix]`, for links in responses.
    pub host: &'a str,
    pub forwarded_prefix: &'a str,
    pub path: &'a str,
    pub raw_query: &'a str,
    pub if_none_match: Option<&'a str>,
    /// The one item a picture is made of, when the request is for an item and not a
    /// search: the search's routes then draw that item and look nothing up.
    pub only: Option<&'a Arc<Vec<MosaicItem>>>,
}

/// The items a picture over `geometry` is made of: the request's own item, or what the
/// search finds there.
#[allow(clippy::too_many_arguments)]
async fn pixel_items(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    geometry: &Value,
    hash: &str,
    limits: SearchLimits,
    names: &[String],
    grid: Option<SampleGrid>,
    lookup: Lookup,
) -> Result<Arc<Vec<MosaicItem>>, cogzilla_pgstac::PgStacError> {
    match ctx.only {
        Some(only) => Ok(only.clone()),
        None => pg.items_for_pixels(geometry, hash, limits, names, grid, lookup).await,
    }
}

type Resp = Response<Full<Bytes>>;

fn json_resp(status: StatusCode, body: &Value) -> Resp {
    super::plain_resp(status, body.to_string(), "application/json")
}

fn detail(status: StatusCode, msg: impl Into<String>) -> Resp {
    super::json_error_resp(status, &msg.into())
}

/// FastAPI's body for a required query parameter that is absent.
fn missing_query_param(name: &str) -> Resp {
    json_resp(
        StatusCode::UNPROCESSABLE_ENTITY,
        &json!({ "detail": [{ "type": "missing", "loc": ["query", name], "msg": "Field required", "input": null }] }),
    )
}

fn invalid_param(location: &str, name: &str, msg: impl Into<String>) -> Resp {
    json_resp(
        StatusCode::UNPROCESSABLE_ENTITY,
        &json!({ "detail": [{ "type": "value_error", "loc": [location, name], "msg": msg.into() }] }),
    )
}

/// A 405 that says which method the route does take.
fn method_not_allowed(takes: Option<Method>) -> Resp {
    super::method_not_allowed_opt(takes.as_ref().map(Method::as_str))
}

fn no_content() -> Resp {
    super::error_to_http_response(&EngineError::NoContent)
}

fn db_error(err: &PgStacError) -> Resp {
    match err {
        PgStacError::ReadOnly => detail(StatusCode::BAD_REQUEST, err.to_string()),
        PgStacError::Unavailable(_) | PgStacError::Config(_) => {
            tracing::warn!(error = %err, "pgSTAC database unavailable");
            detail(StatusCode::SERVICE_UNAVAILABLE, "pgSTAC database is unavailable")
        }
        // What Postgres raised describes the request (a bad filter, an unknown search).
        // Anything else stays in the log.
        _ => match err.server_message() {
            Some(msg) => {
                tracing::warn!(error = %err, "pgSTAC query failed");
                detail(StatusCode::INTERNAL_SERVER_ERROR, msg)
            }
            None => {
                tracing::warn!(error = %err, "pgSTAC query failed");
                detail(StatusCode::INTERNAL_SERVER_ERROR, "pgSTAC query failed")
            }
        },
    }
}

/// A `400` in titiler-pgstac's error shape.
pub(super) fn bad_request(message: String) -> Resp {
    detail(StatusCode::BAD_REQUEST, message)
}

/// The shared answer to an engine's error, but for a method a route does not take: an
/// item's routes are each for one method, and say so as Starlette does.
pub(super) fn engine_error(err: &EngineError) -> Resp {
    match err {
        EngineError::Shared(inner) => engine_error(inner),
        EngineError::MethodNotAllowed(allow) => {
            method_not_allowed(Some(if allow.contains("GET") { Method::GET } else { Method::POST }))
        }
        other => super::error_to_http_response(other),
    }
}

/// Decoded `key=value` pairs in request order.
pub(super) fn query_pairs(raw_query: &str) -> Vec<(String, String)> {
    raw_query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (percent_decode(k), percent_decode(v))
        })
        .collect()
}

/// The query with its numbers written plainly. titiler-pgstac reads a number as Python
/// does, with space around it, `_` between digits and a whole one as `256.0`; the shared
/// parser reads only the plain spelling, so that is what it is given. A value that is not
/// a number is left for the route to refuse.
pub(super) fn plain_numbers(raw_query: &str) -> String {
    let plain = |part: &str| -> Option<String> {
        let (key, value) = part.split_once('=')?;
        let key = super::query_key(key);
        let key = key.as_ref();
        let value = percent_decode(value);
        let written = match key {
            "tilesize" | "padding" | "items_limit" | "scan_limit" | "time_limit" | "width" | "height" | "max_size"
                if py_whole(&value) =>
            {
                let n = py_number(&value)?;
                // More padding than the widest kernel reads changes no pixel.
                let n = if key == "padding" { n.min(64.0) } else { n };
                (n.abs() < 1e15).then(|| format!("{}", n as i64))?
            }
            "buffer" => py_number(&value).filter(|n| n.is_finite()).map(|n| format!("{n}"))?,
            _ => return None,
        };
        (written != value).then(|| format!("{key}={written}"))
    };
    // A size of zero is a size not given, as rio-tiler reads it: the pair is dropped.
    let unset = |part: &str| {
        part.split_once('=').is_some_and(|(key, value)| {
            matches!(super::query_key(key).as_ref(), "width" | "height" | "max_size") && py_whole(&percent_decode(value)) && py_number(&percent_decode(value)) == Some(0.0)
        })
    };
    let parts: Vec<String> =
        raw_query.split('&').filter(|part| !unset(part)).map(|part| plain(part).unwrap_or_else(|| part.to_string())).collect();
    parts.join("&")
}

fn query_value<'a>(pairs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    pairs.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// Python's `urllib.parse.quote_plus`, which is how TiTiler writes query strings back out.
fn quote_plus(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(b as char),
            b' ' => out.push('+'),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

use super::query::{looks_like_rfc3339_datetime as looks_like_datetime, parse_bool};

/// A number as Python reads one from text: space around it and `_` between digits allowed.
fn py_number(s: &str) -> Option<f64> {
    let t = s.trim();
    let b = t.as_bytes();
    let between = |i: usize| i > 0 && b[i - 1].is_ascii_digit() && b.get(i + 1).is_some_and(u8::is_ascii_digit);
    if !(0..b.len()).all(|i| b[i] != b'_' || between(i)) {
        return None;
    }
    t.replace('_', "").parse::<f64>().ok()
}

/// A whole number as pydantic reads one from text: [`py_number`]'s spelling, digits only
/// but for a fraction of zeros.
fn py_whole(s: &str) -> bool {
    let t = s.trim();
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, "0"));
    !whole.is_empty()
        && whole.bytes().all(|c| c.is_ascii_digit() || c == b'_')
        && !fraction.is_empty()
        && fraction.bytes().all(|c| c == b'0')
        && py_number(whole).is_some()
}

/// Whether the tile engine gathers too little of a raster for this method at this zoom. A
/// method that gathers a pixel's footprint is right there only while the raster has an
/// overview about as coarse as the tile (`sum` comes out at half past that); such a tile
/// is read as a box over its bounds is, which gathers the whole footprint.
fn engine_gathers_too_little(crs: &CrsModel, cog: &cogzilla_io::OpenedCog, resample: ResampleMode, zoom: u8) -> bool {
    if !matches!(
        resample,
        ResampleMode::Average | ResampleMode::Mode | ResampleMode::Gauss | ResampleMode::Rms | ResampleMode::Min
            | ResampleMode::Max | ResampleMode::Med | ResampleMode::Q1 | ResampleMode::Q3 | ResampleMode::Sum
    ) {
        return false;
    }
    let full = cog.index.full_res();
    let coarsest = cog.index.ifds.iter().map(|ifd| ifd.width).min().unwrap_or(full.width).max(1);
    let covered = warped_resolution(crs, cog, true).map(|res| res * f64::from(full.width) / f64::from(coarsest));
    let of_tile = 40_075_016.685_578_49 / (256.0 * f64::from(1u32 << u32::from(zoom).min(30)));
    covered.is_none_or(|covered| of_tile > 2.0 * covered)
}

/// A tile's `tilesize`, `padding` and `buffer` are more than zero, as titiler-pgstac has them.
fn tile_bounds_error(pairs: &[(String, String)]) -> Option<Resp> {
    // Of a parameter given twice, the last is the one read.
    for key in ["tilesize", "padding", "buffer"] {
        let Some((_, value)) = pairs.iter().rev().find(|(k, _)| k == key) else {
            continue;
        };
        let positive = py_number(value).is_some_and(|n| n > 0.0);
        let wrong = match key {
            "tilesize" | "padding" => (!(py_whole(value) && positive)).then_some("Input should be a whole number greater than 0"),
            "buffer" => (!positive).then_some("Input should be a number greater than 0"),
            _ => None,
        };
        if let Some(msg) = wrong {
            return Some(invalid_param("query", key, msg));
        }
    }
    None
}

/// What a point reads of a picture's parameters, checked as a picture's are.
fn point_query_error(pairs: &[(String, String)]) -> Option<Resp> {
    let read: Vec<(String, String)> = pairs
        .iter()
        .filter(|(k, _)| matches!(k.as_str(), "resampling" | "nodata" | "unscale"))
        .cloned()
        .collect();
    render_query_error(&read)
}

/// What FastAPI refuses of a picture's parameters before anything is drawn. The routes
/// that only describe a mosaic (`tilejson.json`, `map.html`, the WMTS document) take the
/// same parameters as its tiles, and titiler-pgstac checks them there too.
fn render_query_error(pairs: &[(String, String)]) -> Option<Resp> {
    let bad = |name: &str, msg: &str| Some(invalid_param("query", name, msg));
    if let Some(resp) = tile_bounds_error(pairs) {
        return Some(resp);
    }
    if let Some(map) = query_value(pairs, "colormap")
        && !map.is_empty()
        && !matches!(map.trim(), "NaN" | "Infinity" | "-Infinity")
        && serde_json::from_str::<Value>(map).map_or(true, |v| v.is_string())
    {
        return Some(detail(StatusCode::BAD_REQUEST, "Could not parse the colormap value."));
    }
    if let Some(resp) = colormap_name_error(pairs) {
        return Some(resp);
    }
    for (key, value) in pairs {
        let number = || py_number(value).is_some();
        let whole = || py_whole(value);
        let wrong = match key.as_str() {
            "rescale" => {
                // Either `min,max` or the same as JSON: `[min,max]`, or a list of those.
                let pair = |v: &Value| v.as_array().is_some_and(|a| a.len() == 2 && a.iter().all(Value::is_number));
                let good = if value.trim_start().starts_with('[') {
                    serde_json::from_str::<Value>(value)
                        .is_ok_and(|v| pair(&v) || v.as_array().is_some_and(|a| !a.is_empty() && a.iter().all(pair)))
                } else {
                    let parts: Vec<&str> = value.split(',').collect();
                    parts.len() == 2 && parts.iter().all(|p| py_number(p).is_some())
                };
                (!good).then_some("Input should be a range, as min,max")
            }
            "resampling" | "reproject" => resample_method(key, value).is_none().then_some("Input should be a resampling method"),
            "pixel_selection" => Selection::parse(value).is_none().then_some("Input should be a pixel selection method"),
            // Only the operation is looked at here; what follows it is the tile's to refuse.
            "color_formula" => (value.is_empty()
                || value.starts_with(char::is_whitespace)
                || (cogzilla_kernel::ColorFormula::parse(value).is_err()
                    && !value.split_whitespace().next().is_some_and(|op| {
                        matches!(op.to_ascii_lowercase().as_str(), "gamma" | "sigmoidal" | "saturation")
                    })))
            .then_some("Input should be a colour formula"),
            "nodata" => (!number()).then_some("Input should be a valid number"),
            "unscale" | "return_mask" | "asset_as_band" | "exitwhenfull" | "skipcovered" => {
                parse_bool(value).is_none().then_some("Input should be a valid boolean")
            }
            "items_limit" | "time_limit" | "scan_limit" => (!whole()).then_some("Input should be a valid integer"),
            "algorithm" => super::get_algorithm_json(value).is_err().then_some("Input should be the name of an algorithm"),
            "algorithm_params" => serde_json::from_str::<Value>(value).is_err().then_some("Input should be valid JSON"),
            _ => None,
        };
        if let Some(msg) = wrong {
            return bad(key, msg);
        }
    }
    None
}

/// `scan_limit`, `items_limit`, `time_limit`, `exitwhenfull` and `skipcovered`.
fn parse_limits(pairs: &[(String, String)]) -> Result<SearchLimits, Resp> {
    let mut limits = SearchLimits::default();
    let int = |name: &str, max_allowed: i32, slot: &mut i32| -> Result<(), Resp> {
        if let Some(v) = query_value(pairs, name) {
            let given = v
                .parse::<i32>()
                .ok()
                .filter(|n| *n >= 0)
                .ok_or_else(|| invalid_param("query", name, "Input should be a valid integer"))?;
            if given > max_allowed {
                return Err(invalid_param(
                    "query",
                    name,
                    format!("Input should be less than or equal to {max_allowed}"),
                ));
            }
            // Zero is no limit given, as in titiler-pgstac: the default stands.
            if given > 0 {
                *slot = given;
            }
        }
        Ok(())
    };
    int("scan_limit", 100_000, &mut limits.scan_limit)?;
    int("items_limit", 10_000, &mut limits.items_limit)?;
    int("time_limit", 60, &mut limits.time_limit_secs)?;
    let boolean = |name: &str, slot: &mut bool| -> Result<(), Resp> {
        if let Some(v) = query_value(pairs, name) {
            *slot = parse_bool(v)
                .ok_or_else(|| invalid_param("query", name, "Input should be a valid boolean"))?;
        }
        Ok(())
    };
    boolean("exitwhenfull", &mut limits.exitwhenfull)?;
    boolean("skipcovered", &mut limits.skipcovered)?;
    Ok(limits)
}

/// `item_lookup`, a `cogzilla` parameter: which lookup lists the items of this request.
fn parse_lookup(pairs: &[(String, String)]) -> Result<Lookup, Resp> {
    match query_value(pairs, "item_lookup") {
        None => Ok(Lookup::Configured),
        Some("lean") => Ok(Lookup::Lean),
        Some("geojsonsearch") => Ok(Lookup::Reference),
        Some(other) => Err(detail(
            StatusCode::BAD_REQUEST,
            format!("item_lookup '{other}' is not known (use lean or geojsonsearch)"),
        )),
    }
}

/// Where a map tile samples its outline.
fn tile_grid(scale: Option<u32>, pq: &ParsedQuery) -> Option<SampleGrid> {
    let side = scale.map_or(pq.tile_size, |s| s.saturating_mul(256));
    SampleGrid::for_image(side, side, true)
}

/// Where a `bbox` or `feature` image samples its box. `None` when the size follows the
/// data's own resolution, which is not known before the items are.
fn part_grid(size: PartSize, bbox: [f64; 4]) -> Option<SampleGrid> {
    let ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]);
    if !(ratio.is_finite() && ratio > 0.0) {
        return None;
    }
    // rio-tiler takes the ratio from whole pixel counts; the box's own is that but for
    // rounding, which must not tip a size over to the next pixel.
    let scaled = |side: u32, by: f64| (f64::from(side) * by - 1e-6).ceil().clamp(1.0, f64::from(u32::MAX)) as u32;
    let (width, height) = match size {
        PartSize::Exact(w, h) | PartSize::Query { width: Some(w), height: Some(h), .. } => (w, h),
        PartSize::Query { width: Some(w), .. } => (w, scaled(w, ratio)),
        PartSize::Query { height: Some(h), .. } => (scaled(h, 1.0 / ratio), h),
        PartSize::Query { max_size: Some(m), .. } if ratio > 1.0 => (scaled(m, 1.0 / ratio), m),
        PartSize::Query { max_size: Some(m), .. } => (m, scaled(m, ratio)),
        PartSize::Query { .. } => return None,
    };
    SampleGrid::for_image(width, height, false)
}

/// The tile's outline in WGS84 as a GeoJSON polygon, which is what `geojsonsearch` takes.
fn tile_polygon(tile: TileCoord) -> Value {
    let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
    let (west, south) = mercator_to_lonlat_deg(min_x, min_y);
    let (east, north) = mercator_to_lonlat_deg(max_x, max_y);
    bbox_polygon([west, south, east, north])
}

fn search_json(rec: &SearchRecord) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("hash".into(), json!(rec.hash));
    out.insert("search".into(), rec.search.clone());
    if let Some(w) = &rec.where_clause {
        out.insert("_where".into(), json!(w));
    }
    if let Some(o) = &rec.orderby {
        out.insert("orderby".into(), json!(o));
    }
    out.insert("lastused".into(), json!(rec.lastused));
    out.insert("usecount".into(), json!(rec.usecount.unwrap_or(0)));
    out.insert("metadata".into(), rec.metadata.clone());
    Value::Object(out)
}

/// Python's `str()` of a JSON value, which is what `urlencode` writes into a link.
fn py_str(v: &Value) -> String {
    fn repr(v: &Value) -> String {
        match v {
            Value::String(s) => format!("'{s}'"),
            other => py_str(other),
        }
    }
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(a) => format!("[{}]", a.iter().map(repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => {
            let mut keys: Vec<String> = o.keys().cloned().collect();
            jsonb_key_order(&mut keys);
            let body: Vec<String> = keys.iter().map(|k| format!("'{k}': {}", repr(&o[k]))).collect();
            format!("{{{}}}", body.join(", "))
        }
    }
}

/// Python's `json.dumps` with its default separators, keys in Postgres' jsonb order.
fn py_json_dumps(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("[{}]", a.iter().map(py_json_dumps).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => {
            let mut keys: Vec<String> = o.keys().cloned().collect();
            jsonb_key_order(&mut keys);
            let body: Vec<String> = keys
                .iter()
                .map(|k| format!("{}: {}", Value::String(k.clone()), py_json_dumps(&o[k])))
                .collect();
            format!("{{{}}}", body.join(", "))
        }
        scalar => scalar.to_string(),
    }
}

/// Python truthiness of a JSON value.
fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The layers a mosaic's `metadata.defaults` describe, each as a name and the query string
/// titiler-pgstac puts on the layer's links. A layer without `assets` cannot make a tile
/// request and gets no link. `adapt` folds `asset_bidx` and `asset_expression` into the
/// asset names (`name|bidx=1`), as the info routes do.
fn layer_queries(metadata: &Value, adapt: bool) -> Vec<(String, String)> {
    let Some(defaults) = metadata.get("defaults").and_then(Value::as_object) else {
        return Vec::new();
    };
    let as_list = |v: &Value| match v {
        Value::Array(a) => a.clone(),
        other => vec![other.clone()],
    };
    let mut names: Vec<String> = defaults.keys().cloned().collect();
    jsonb_key_order(&mut names);
    let mut out = Vec::new();
    for name in names {
        let Some(values) = defaults[&name].as_object() else {
            continue;
        };
        // Python reads the keys in jsonb order, then moves the ones it rewrites to the end.
        let mut keys: Vec<String> = values.keys().cloned().collect();
        jsonb_key_order(&mut keys);
        let mut params: Vec<(String, Value)> =
            keys.into_iter().map(|k| (k.clone(), values[&k].clone())).collect();
        for key in ["rescale", "colormap", "assets", "asset_bidx"] {
            let Some(pos) = params.iter().position(|(k, _)| k == key) else {
                continue;
            };
            let (_, value) = params.remove(pos);
            if !py_truthy(&value) {
                continue;
            }
            let value = match key {
                "rescale" => Value::Array(
                    as_list(&value)
                        .iter()
                        .map(|r| match r {
                            Value::Array(pair) => {
                                json!(pair.iter().map(py_str).collect::<Vec<_>>().join(","))
                            }
                            other => other.clone(),
                        })
                        .collect(),
                ),
                "colormap" if !value.is_string() => json!(py_json_dumps(&value)),
                "assets" | "asset_bidx" if value.is_string() => json!([value]),
                _ => value,
            };
            params.push((key.to_string(), value));
        }
        let assets: Vec<String> = params
            .iter()
            .find(|(k, _)| k == "assets")
            .map(|(_, v)| as_list(v).iter().map(py_str).collect())
            .unwrap_or_default();
        if assets.is_empty() {
            continue;
        }
        if adapt {
            let mut options: Vec<Vec<String>> = vec![Vec::new(); assets.len()];
            for (key, label) in [("asset_bidx", "bidx"), ("asset_expression", "expression")] {
                let Some(pos) = params.iter().position(|(k, _)| k == key) else {
                    continue;
                };
                let (_, value) = params.remove(pos);
                for entry in as_list(&value) {
                    let entry = py_str(&entry);
                    if let Some((asset, opt)) = entry.split_once('|')
                        && let Some(i) = assets.iter().position(|a| a == asset)
                    {
                        options[i].push(format!("{label}={opt}"));
                    }
                }
            }
            let merged: Vec<Value> = assets
                .iter()
                .zip(&options)
                .map(|(a, o)| if o.is_empty() { json!(a) } else { json!(format!("{a}|{}", o.join("|"))) })
                .collect();
            if let Some(slot) = params.iter_mut().find(|(k, _)| k == "assets") {
                slot.1 = Value::Array(merged);
            }
        }
        let mut qs = Vec::new();
        for (k, v) in &params {
            match v {
                Value::Array(items) => {
                    qs.extend(items.iter().map(|i| format!("{}={}", quote_plus(k), quote_plus(&py_str(i)))));
                }
                other => qs.push(format!("{}={}", quote_plus(k), quote_plus(&py_str(other)))),
            }
        }
        out.push((name, qs.join("&")));
    }
    out
}

fn register_links(host: &str, rec: &SearchRecord) -> Value {
    let id = &rec.hash;
    let tilejson = format!("{host}/searches/{id}/{{tileMatrixSetId}}/tilejson.json");
    let mut links = vec![
        json!({ "href": format!("{host}/searches/{id}/info"), "rel": "metadata", "title": "Mosaic metadata" }),
        json!({ "href": tilejson, "rel": "tilejson", "templated": true, "title": "Link for TileJSON (Template URL)" }),
        json!({ "href": format!("{host}/searches/{id}/{{tileMatrixSetId}}/map.html"), "rel": "map", "templated": true, "title": "Link for Map viewer (Template URL)" }),
        json!({ "href": format!("{host}/searches/{id}/WMTSCapabilities.xml"), "rel": "wmts", "title": "WMTS Capabilities link." }),
    ];
    for (name, qs) in layer_queries(&rec.metadata, false) {
        links.push(json!({ "href": format!("{tilejson}?{qs}"), "rel": "tilejson", "templated": true, "title": format!("TileJSON link for `{name}` layer (Template URL).") }));
    }
    Value::Array(links)
}

/// `base` is the mosaic's own path: `/searches/{hash}` or `/collections/{id}`.
fn info_links(host: &str, base: &str, rec: &SearchRecord) -> Value {
    let layers = layer_queries(&rec.metadata, true);
    let tilejson = format!("{host}{base}/{{tileMatrixSetId}}/tilejson.json");
    let map = format!("{host}{base}/{{tileMatrixSetId}}/map.html");
    let mut links = vec![
        json!({ "href": format!("{host}{base}/info"), "rel": "self", "title": "Mosaic metadata" }),
        json!({ "href": tilejson, "rel": "tilejson", "templated": true, "title": "TileJSON link (Template URL)." }),
    ];
    for (name, qs) in &layers {
        links.push(json!({ "href": format!("{tilejson}?{qs}"), "rel": "tilejson", "templated": true, "title": format!("TileJSON link for `{name}` layer (Template URL).") }));
    }
    links.push(json!({ "href": map, "rel": "map", "templated": true, "title": "Map viewer link (Template URL)." }));
    for (name, qs) in &layers {
        links.push(json!({ "href": format!("{map}?{qs}"), "rel": "map", "templated": true, "title": format!("Map viewer link for `{name}` layer (Template URL).") }));
    }
    links.push(json!({ "href": format!("{host}{base}/WMTSCapabilities.xml"), "rel": "wmts", "title": "WMTS Capabilities link." }));
    Value::Array(links)
}

fn list_item_links(host: &str, id: &str) -> Value {
    json!([
        { "href": format!("{host}/searches/{id}/info"), "rel": "metadata", "title": "Mosaic metadata" },
        { "href": format!("{host}/searches/{id}/{{tileMatrixSetId}}/tilejson.json"), "rel": "tilejson", "templated": true, "title": "Link for TileJSON (Template URL)" }
    ])
}

/// Split a `POST /searches/register` body into the search pgSTAC stores and its metadata,
/// the way titiler-pgstac's model does.
fn split_register_body(body: Value) -> Result<(Value, Value), Resp> {
    let Value::Object(mut obj) = body else {
        return Err(invalid_param("body", "", "Input should be a valid dictionary or object"));
    };
    let metadata = match obj.remove("metadata") {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(m)) => m,
        Some(_) => return Err(invalid_param("body", "metadata", "Input should be a valid dictionary")),
    };
    let mut metadata: serde_json::Map<String, Value> =
        metadata.into_iter().filter(|(_, v)| !v.is_null()).collect();
    metadata
        .entry("type".to_string())
        .or_insert_with(|| json!("mosaic"));

    obj.retain(|_, v| !v.is_null());
    if obj.contains_key("bbox") && obj.contains_key("intersects") {
        return Err(invalid_param(
            "body",
            "intersects",
            "Value error, intersects and bbox parameters are mutually exclusive",
        ));
    }
    match obj.get("filter-lang").and_then(Value::as_str) {
        None | Some("cql2-json") => {}
        Some(_) => {
            return Err(invalid_param("body", "filter-lang", "Input should be 'cql2-json'"));
        }
    }
    // titiler-pgstac always stores the language, and the stored JSON is what gets hashed.
    obj.insert("filter-lang".into(), json!("cql2-json"));
    Ok((Value::Object(obj), Value::Object(metadata)))
}

async fn register<B>(ctx: &Ctx<'_>, pg: &PgStac, req: Request<B>) -> Resp
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let body = match read_limited_body(req.into_body()).await {
        Ok(b) => b,
        Err(e) => return engine_error(&e),
    };
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return invalid_param("body", "", "JSON decode error"),
    };
    let (search, metadata) = match split_register_body(body) {
        Ok(parts) => parts,
        Err(resp) => return resp,
    };
    match pg.register_search(&search, &metadata).await {
        Ok(rec) => json_resp(
            StatusCode::OK,
            &json!({ "id": rec.hash, "links": register_links(ctx.host, &rec) }),
        ),
        Err(e) => db_error(&e),
    }
}

async fn list(ctx: &Ctx<'_>, pg: &PgStac, pairs: &[(String, String)]) -> Resp {
    let limit = match query_value(pairs, "limit").map(str::parse::<i64>) {
        None => 10,
        Some(Ok(n)) if (1..=1000).contains(&n) => n,
        Some(_) => return invalid_param("query", "limit", "Input should be between 1 and 1000"),
    };
    let offset = match query_value(pairs, "offset").map(str::parse::<i64>) {
        None => 0,
        Some(Ok(n)) if n >= 0 => n,
        Some(_) => return invalid_param("query", "offset", "Input should be greater than or equal to 0"),
    };
    // Sorted by `lastused` or `usecount`, or by any key of the searches' metadata.
    let mut order = Vec::new();
    for spec in query_value(pairs, "sortby").unwrap_or("").split(',').map(str::trim_start).filter(|s| !s.is_empty()) {
        let (descending, field) = match spec.as_bytes()[0] {
            b'-' => (true, &spec[1..]),
            b'+' => (false, &spec[1..]),
            _ => (false, spec),
        };
        let field = match field {
            "lastused" => SearchField::LastUsed,
            "usecount" => SearchField::UseCount,
            other => SearchField::Metadata(other.to_string()),
        };
        order.push(SearchOrder { field, descending });
    }
    if order.len() > 64 {
        return invalid_param("query", "sortby", "sortby takes at most 64 fields");
    }
    // Every other parameter is metadata a search must have been registered with; given
    // twice, the last value is the one meant.
    let mut wanted: Vec<(String, String)> = Vec::new();
    for (k, v) in pairs {
        if matches!(k.to_ascii_lowercase().as_str(), "limit" | "offset" | "properties" | "sortby") {
            continue;
        }
        match wanted.iter_mut().find(|(seen, _)| seen == k) {
            Some(slot) => slot.1 = v.clone(),
            None => wanted.push((k.clone(), v.clone())),
        }
    }
    let (searches, matched) = match pg.list_searches(limit, offset, &order, &wanted).await {
        Ok(page) => page,
        Err(e) => return db_error(&e),
    };
    // A page's link is the request's own parameters with the page's `offset`, and the
    // `limit` used when the request gave none.
    let page_url = |off: i64| {
        let mut params: Vec<(String, String)> = Vec::with_capacity(pairs.len() + 2);
        for (k, v) in pairs {
            match params.iter_mut().find(|(seen, _)| seen == k) {
                Some(slot) => slot.1 = v.clone(),
                None => params.push((k.clone(), v.clone())),
            }
        }
        for (k, v) in [("limit", limit.to_string()), ("offset", off.to_string())] {
            match params.iter_mut().find(|(seen, _)| seen == k) {
                Some(slot) => slot.1 = v,
                None => params.push((k.to_string(), v)),
            }
        }
        let query: Vec<String> = params.iter().map(|(k, v)| format!("{}={}", quote_plus(k), quote_plus(v))).collect();
        format!("{}/searches/list?{}", ctx.host, query.join("&"))
    };
    // As titiler-pgstac has it: a next page whenever this one is not all there is,
    // starting after what this one returned.
    let returned = searches.len() as i64;
    let mut links = vec![json!({ "href": page_url(offset), "rel": "self" })];
    if returned < matched {
        links.push(json!({ "href": page_url(offset + returned), "rel": "next" }));
    }
    if offset > 0 {
        links.push(json!({ "href": page_url((offset - limit).max(0)), "rel": "prev" }));
    }
    let entries: Vec<Value> = searches
        .iter()
        .map(|rec| json!({ "search": search_json(rec), "links": list_item_links(ctx.host, &rec.hash) }))
        .collect();
    json_resp(
        StatusCode::OK,
        &json!({
            "searches": entries,
            "links": links,
            "context": { "returned": searches.len(), "limit": limit, "matched": matched }
        }),
    )
}

async fn find_search(pg: &PgStac, id: &str) -> Result<Arc<SearchRecord>, Resp> {
    match pg.get_search_cached(id).await {
        Ok(Some(rec)) => Ok(rec),
        Ok(None) => Err(detail(StatusCode::NOT_FOUND, format!("SearchId `{id}` not found"))),
        Err(e) => Err(db_error(&e)),
    }
}

/// The pgSTAC routes: path, method and summary.
const OPENAPI_ROUTES: [(&str, &str, &str); 63] = [
    ("/searches/list", "get", "List registered pgSTAC searches"),
    ("/searches/{search_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}", "get", "Render pgSTAC search mosaic tile, format from the pixels"),
    ("/searches/{search_id}/bbox/{minx},{miny},{maxx},{maxy}/{width}x{height}.{format}", "get", "Image of a pgSTAC search mosaic over a bounding box, at a size"),
    ("/searches/{search_id}/feature", "post", "Image of a pgSTAC search mosaic clipped to a GeoJSON feature, format from the pixels"),
    ("/searches/{search_id}/feature/{width}x{height}.{format}", "post", "Image of a pgSTAC search mosaic clipped to a GeoJSON feature, at a size"),
    ("/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}", "get", "Render pgSTAC collection mosaic tile, format from the pixels"),
    ("/collections/{collection_id}/bbox/{minx},{miny},{maxx},{maxy}/{width}x{height}.{format}", "get", "Image of a pgSTAC collection mosaic over a bounding box, at a size"),
    ("/collections/{collection_id}/feature", "post", "Image of a pgSTAC collection mosaic clipped to a GeoJSON feature, format from the pixels"),
    ("/collections/{collection_id}/feature/{width}x{height}.{format}", "post", "Image of a pgSTAC collection mosaic clipped to a GeoJSON feature, at a size"),
    ("/collections/{collection_id}/items/{item_id}/statistics", "post", "pgSTAC item statistics under GeoJSON features"),
    ("/collections/{collection_id}/items/{item_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}", "get", "Render pgSTAC item tile, format from the pixels"),
    ("/collections/{collection_id}/items/{item_id}/preview", "get", "pgSTAC item preview, format from the pixels"),
    ("/collections/{collection_id}/items/{item_id}/preview/{width}x{height}.{format}", "get", "pgSTAC item preview, at a size"),
    ("/collections/{collection_id}/items/{item_id}/bbox/{minx},{miny},{maxx},{maxy}/{width}x{height}.{format}", "get", "pgSTAC item image for a bounding box, at a size"),
    ("/collections/{collection_id}/items/{item_id}/feature", "post", "pgSTAC item image for a GeoJSON feature, format from the pixels"),
    ("/collections/{collection_id}/items/{item_id}/feature/{width}x{height}.{format}", "post", "pgSTAC item image for a GeoJSON feature, at a size"),
    ("/searches/{search_id}/statistics", "post", "Statistics of a pgSTAC search mosaic under GeoJSON features"),
    ("/collections/{collection_id}/statistics", "post", "Statistics of a pgSTAC collection mosaic under GeoJSON features"),
    ("/collections/{collection_id}/items/{item_id}/renders/{render_id}", "get", "One of a pgSTAC item's renders"),
    ("/searches/{search_id}/bbox/{minx},{miny},{maxx},{maxy}.{format}", "get", "Image of a pgSTAC search mosaic over a bounding box"),
    ("/searches/{search_id}/feature.{format}", "post", "Image of a pgSTAC search mosaic clipped to a GeoJSON feature"),
    ("/collections/{collection_id}/bbox/{minx},{miny},{maxx},{maxy}.{format}", "get", "Image of a pgSTAC collection mosaic over a bounding box"),
    ("/collections/{collection_id}/feature.{format}", "post", "Image of a pgSTAC collection mosaic clipped to a GeoJSON feature"),
    ("/searches/{search_id}/point/{lon},{lat}", "get", "Values of the items under a point"),
    ("/collections/{collection_id}/point/{lon},{lat}", "get", "Values of the items under a point"),
    ("/searches/{search_id}/WMTSCapabilities.xml", "get", "pgSTAC search WMTS capabilities"),
    ("/collections/{collection_id}/WMTSCapabilities.xml", "get", "pgSTAC collection WMTS capabilities"),
    ("/collections/{collection_id}/items/{item_id}/WMTSCapabilities.xml", "get", "pgSTAC item WMTS capabilities"),
    ("/collections/{collection_id}/items/{item_id}/assets", "get", "Assets of a pgSTAC item"),
    ("/collections/{collection_id}/items/{item_id}/info", "get", "pgSTAC item asset metadata"),
    ("/collections/{collection_id}/items/{item_id}/info.geojson", "get", "pgSTAC item asset metadata as GeoJSON"),
    ("/collections/{collection_id}/items/{item_id}/statistics", "get", "pgSTAC item statistics"),
    ("/collections/{collection_id}/items/{item_id}/asset_statistics", "get", "pgSTAC item per-asset statistics"),
    ("/collections/{collection_id}/items/{item_id}/tiles", "get", "pgSTAC item tilesets"),
    ("/collections/{collection_id}/items/{item_id}/tiles/{tileMatrixSetId}", "get", "pgSTAC item tileset description"),
    ("/collections/{collection_id}/items/{item_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}", "get", "Render pgSTAC item tile"),
    ("/collections/{collection_id}/items/{item_id}/{tileMatrixSetId}/tilejson.json", "get", "pgSTAC item TileJSON 3.0"),
    ("/collections/{collection_id}/items/{item_id}/{tileMatrixSetId}/map.html", "get", "pgSTAC item map viewer"),
    ("/collections/{collection_id}/items/{item_id}/point/{lon},{lat}", "get", "pgSTAC item values at a point"),
    ("/collections/{collection_id}/items/{item_id}/preview.{format}", "get", "pgSTAC item preview"),
    ("/collections/{collection_id}/items/{item_id}/bbox/{minx},{miny},{maxx},{maxy}.{format}", "get", "pgSTAC item image for a bounding box"),
    ("/collections/{collection_id}/items/{item_id}/feature.{format}", "post", "pgSTAC item image for a GeoJSON feature"),
    ("/collections/{collection_id}/items/{item_id}/renders", "get", "pgSTAC item renders"),
    ("/searches/register", "post", "Register a pgSTAC search"),
    ("/searches/", "get", "List registered pgSTAC searches"),
    ("/searches/{search_id}/info", "get", "pgSTAC search metadata"),
    ("/searches/{search_id}/{tileMatrixSetId}/tilejson.json", "get", "pgSTAC search TileJSON 3.0"),
    ("/searches/{search_id}/{tileMatrixSetId}/map.html", "get", "pgSTAC search map viewer"),
    ("/searches/{search_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}", "get", "Render pgSTAC search mosaic tile"),
    ("/searches/{search_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}/assets", "get", "Items covering a tile"),
    ("/searches/{search_id}/tiles", "get", "pgSTAC search tilesets"),
    ("/searches/{search_id}/tiles/{tileMatrixSetId}", "get", "pgSTAC search tileset description"),
    ("/searches/{search_id}/bbox/{minx},{miny},{maxx},{maxy}/assets", "get", "Items intersecting a bounding box"),
    ("/searches/{search_id}/point/{lon},{lat}/assets", "get", "Items covering a point"),
    ("/collections/{collection_id}/info", "get", "pgSTAC collection mosaic metadata"),
    ("/collections/{collection_id}/{tileMatrixSetId}/tilejson.json", "get", "pgSTAC collection TileJSON 3.0"),
    ("/collections/{collection_id}/{tileMatrixSetId}/map.html", "get", "pgSTAC collection map viewer"),
    ("/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}", "get", "Render pgSTAC collection mosaic tile"),
    ("/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}/assets", "get", "Items covering a tile"),
    ("/collections/{collection_id}/tiles", "get", "pgSTAC collection tilesets"),
    ("/collections/{collection_id}/tiles/{tileMatrixSetId}", "get", "pgSTAC collection tileset description"),
    ("/collections/{collection_id}/bbox/{minx},{miny},{maxx},{maxy}/assets", "get", "Items intersecting a bounding box"),
    ("/collections/{collection_id}/point/{lon},{lat}/assets", "get", "Items covering a point"),
];

/// The pgSTAC routes, as entries for the OpenAPI document's `paths`.
pub(super) fn openapi_paths() -> Value {
    let mut paths = serde_json::Map::new();
    for (path, method, summary) in OPENAPI_ROUTES {
        // A path may take more than one method.
        let entry = paths.entry(path.to_string()).or_insert_with(|| json!({}));
        entry[method] = json!({ "summary": summary });
    }
    Value::Object(paths)
}

/// Returns true for content-negotiation query parameters (`f=html`/`f=json` or
/// `format=html`/`format=json`) that select the metadata representation itself and must not
/// be forwarded into generated tile URL templates.
fn is_doc_negotiation_param(k: &str, v: &str) -> bool {
    k.eq_ignore_ascii_case("f")
        || (k.eq_ignore_ascii_case("format")
            && matches!(v.to_ascii_lowercase().as_str(), "html" | "json"))
}

/// The request's query string, re-encoded the way TiTiler writes it into links.
fn encoded_query(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .filter(|(k, v)| !is_doc_negotiation_param(k, v))
        .map(|(k, v)| format!("{}={}", quote_plus(k), quote_plus(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// The search behind `/collections/{id}`: the collection, narrowed by the STAC search
/// parameters in the query string, exactly as titiler-pgstac builds it.
fn collection_search(
    id: &str,
    collection: &Value,
    pairs: &[(String, String)],
) -> Result<(Value, Value), Resp> {
    let mut search = serde_json::Map::new();
    search.insert("collections".into(), json!([id]));
    search.insert("filter-lang".into(), json!("cql2-json"));
    let mut bbox: Option<Vec<f64>> = None;
    if let Some(v) = query_value(pairs, "ids") {
        search.insert("ids".into(), json!(v.split(',').collect::<Vec<_>>()));
    }
    if let Some(v) = query_value(pairs, "bbox") {
        let nums: Option<Vec<f64>> = v.split(',').map(|n| n.trim().parse::<f64>().ok()).collect();
        match nums {
            Some(n) if n.len() == 4 || n.len() == 6 => {
                search.insert("bbox".into(), json!(n));
                bbox = Some(n);
            }
            _ => {
                return Err(invalid_param(
                    "query",
                    "bbox",
                    "Value error, Bounding box must have 4 or 6 numbers",
                ));
            }
        }
    }
    if let Some(v) = query_value(pairs, "datetime") {
        // A date pgSTAC cannot read is the request's fault, and is said so before it is asked.
        // (Digits alone are a timestamp to the reference; pgSTAC is left to read those.)
        let open = |part: &str| matches!(part, "" | "..");
        let dated = |part: &str| looks_like_datetime(part) || (!part.is_empty() && part.bytes().all(|c| c.is_ascii_digit()));
        if !v.split('/').all(|part| open(part) || dated(part)) || v.split('/').all(open) || v.split('/').count() > 2 {
            return Err(invalid_param("query", "datetime", "Input should be a valid datetime"));
        }
        search.insert("datetime".into(), json!(v));
    }
    if let Some(v) = query_value(pairs, "sortby") {
        let fields: Vec<Value> = v
            .split(',')
            .filter(|f| !f.is_empty())
            .map(|f| match f.as_bytes()[0] {
                b'-' => json!({ "field": &f[1..], "direction": "desc" }),
                b'+' => json!({ "field": &f[1..], "direction": "asc" }),
                _ => json!({ "field": f, "direction": "asc" }),
            })
            .collect();
        search.insert("sortby".into(), json!(fields));
    }
    if let Some(v) = query_value(pairs, "query") {
        let parsed: Value = serde_json::from_str(v)
            .map_err(|_| invalid_param("query", "query", "Input should be valid JSON"))?;
        search.insert("query".into(), parsed);
    }
    // As in titiler-pgstac a filter is CQL2 text unless `filter-lang` says JSON, and text
    // is turned into the JSON its converter gives, so the search has the same hash.
    let filter_lang = query_value(pairs, "filter-lang").unwrap_or("cql2-text");
    if !matches!(filter_lang, "cql2-text" | "cql2-json") {
        return Err(invalid_param("query", "filter-lang", "Input should be 'cql2-text' or 'cql2-json'"));
    }
    if let Some(v) = query_value(pairs, "filter") {
        let parsed: Value = if filter_lang == "cql2-json" {
            serde_json::from_str(v)
                .map_err(|_| invalid_param("query", "filter", "Input should be valid JSON"))?
        } else if v.starts_with('{') {
            // titiler-pgstac's converter reads CQL2 JSON too where it expects text, and
            // writes it back as it writes text.
            let given: Value = serde_json::from_str(v).map_err(|e| {
                detail(StatusCode::UNPROCESSABLE_ENTITY, format!("invalid CQL2 JSON filter: {e}"))
            })?;
            super::cql2_text::json_to_json(&given).map_err(|e| {
                detail(StatusCode::UNPROCESSABLE_ENTITY, format!("invalid CQL2 JSON filter: {e}"))
            })?
        } else {
            super::cql2_text::to_json(v).map_err(|e| {
                detail(StatusCode::UNPROCESSABLE_ENTITY, format!("invalid CQL2 text filter: {e}"))
            })?
        };
        search.insert("filter".into(), parsed);
    }

    let mut metadata = serde_json::Map::new();
    metadata.insert("type".into(), json!("mosaic"));
    metadata.insert("name".into(), json!(format!("Mosaic for '{id}' Collection")));
    let extent = collection
        .pointer("/extent/spatial/bbox/0")
        .and_then(Value::as_array)
        .and_then(|a| a.iter().map(Value::as_f64).collect::<Option<Vec<f64>>>());
    if let Some(bounds) = bbox.or(extent) {
        metadata.insert("bounds".into(), json!(bounds));
    }
    if let Some(item_assets) = collection.get("item_assets").and_then(Value::as_object) {
        // The list is hashed with the search, so it has to come out in Postgres' key order.
        let mut keys: Vec<String> = item_assets.keys().cloned().collect();
        jsonb_key_order(&mut keys);
        metadata.insert("assets".into(), json!(keys));
    }
    // The STAC render extension: each render becomes a set of default tile parameters,
    // minus the keys titiler-pgstac does not take.
    if let Some(renders) = collection.get("renders").and_then(Value::as_object) {
        let mut defaults = serde_json::Map::new();
        for (name, render) in renders {
            let Some(render) = render.as_object() else {
                continue;
            };
            let mut render = render.clone();
            render.remove("title");
            render.remove("tilematrixsets");
            if let Some(zooms) = render.remove("minmax_zoom")
                && let Some([min, max]) = zooms.as_array().map(Vec::as_slice)
            {
                render.insert("minzoom".into(), min.clone());
                render.insert("maxzoom".into(), max.clone());
            }
            defaults.insert(name.clone(), Value::Object(render));
        }
        metadata.insert("defaults".into(), Value::Object(defaults));
    }
    Ok((Value::Object(search), Value::Object(metadata)))
}

async fn collection_mosaic(
    pg: &PgStac,
    id: &str,
    pairs: &[(String, String)],
) -> Result<Arc<SearchRecord>, Resp> {
    // One request always derives the same search, so its registration is remembered under
    // the parameters that shape it.
    let narrowing: Vec<String> = ["ids", "bbox", "datetime", "sortby", "query", "filter", "filter-lang"]
        .iter()
        .filter_map(|k| query_value(pairs, k).map(|v| format!("{k}={v}")))
        .collect();
    let key = format!("collection|{id}|{}", narrowing.join("&"));
    let collection = match pg.get_collection_cached(id).await {
        Ok(Some(c)) => c,
        Ok(None) => {
            return Err(detail(StatusCode::NOT_FOUND, format!("CollectionId `{id}` not found")));
        }
        Err(e) => return Err(db_error(&e)),
    };
    let (search, metadata) = collection_search(id, &collection, pairs)?;
    pg.register_search_cached(&key, &search, &metadata)
        .await
        .map_err(|e| db_error(&e))
}

/// Zoom range and bounds of a mosaic: the query string, then the search metadata, then
/// the whole tile matrix set.
fn mosaic_extent(
    metadata: &Value,
    pairs: &[(String, String)],
) -> Result<(i64, i64, [f64; 4]), Resp> {
    let zoom = |name: &str| -> Result<Option<i64>, Resp> {
        match query_value(pairs, name) {
            None => Ok(None),
            Some(v) => py_number(v)
                .filter(|_| py_whole(v))
                .map(|n| Some(n as i64))
                .ok_or_else(|| invalid_param("query", name, "Input should be a valid integer")),
        }
    };
    let meta_i64 = |k: &str| metadata.get(k).and_then(Value::as_i64);
    let minzoom = zoom("minzoom")?.or_else(|| meta_i64("minzoom")).unwrap_or(0);
    let maxzoom = zoom("maxzoom")?.or_else(|| meta_i64("maxzoom")).unwrap_or(24);
    let bounds: [f64; 4] = metadata
        .get("bounds")
        .and_then(Value::as_array)
        .filter(|a| a.len() == 4)
        .and_then(|a| Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?, a[3].as_f64()?]))
        .unwrap_or([-180.0, -85.0511287798066, 180.0, 85.0511287798066]);
    Ok((minzoom, maxzoom, bounds))
}

/// Rows and columns of Web Mercator tiles that touch `bounds` at zoom `z`.
fn tile_limits(z: i64, bounds: [f64; 4]) -> Value {
    const EPS: f64 = 1e-11;
    const MAX_LAT: f64 = 85.0511287798066;
    let n = (1u64 << z) as f64;
    let last = (1u64 << z) - 1;
    let col = |lon: f64| (((lon + 180.0) / 360.0 * n).floor().max(0.0) as u64).min(last);
    let row = |lat: f64| {
        let lat = lat.clamp(-MAX_LAT, MAX_LAT).to_radians();
        let y = (1.0 - lat.tan().asinh() / std::f64::consts::PI) / 2.0 * n;
        (y.floor().max(0.0) as u64).min(last)
    };
    json!({
        "tileMatrix": z.to_string(),
        "minTileRow": row(bounds[3]),
        "maxTileRow": row(bounds[1] + EPS),
        "minTileCol": col(bounds[0]),
        "maxTileCol": col(bounds[2] - EPS),
    })
}

fn bounding_box(bounds: [f64; 4]) -> Value {
    json!({
        "lowerLeft": [bounds[0], bounds[1]],
        "upperRight": [bounds[2], bounds[3]],
        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
    })
}

/// `GET {base}/tiles/{tms}`: the OGC tileset description for one tile matrix set.
fn tileset(
    ctx: &Ctx<'_>,
    base: &str,
    metadata: &Value,
    tms: &str,
    pairs: &[(String, String)],
) -> Resp {
    let Some(tms_def) = crate::tms::lookup_tms(tms) else {
        return invalid_param("path", "tileMatrixSetId", "Input should be a supported TileMatrixSet");
    };
    let (minzoom, maxzoom, bounds) = match mosaic_extent(metadata, pairs) {
        Ok(e) => e,
        Err(r) => return r,
    };
    let host = ctx.host;
    let qs = encoded_query(pairs);
    let with_qs = |url: String| if qs.is_empty() { url } else { format!("{url}?{qs}") };
    let limits: Vec<Value> = if tms == "WebMercatorQuad" {
        (minzoom.clamp(0, 30)..=maxzoom.clamp(0, 30))
            .map(|z| tile_limits(z, bounds))
            .collect()
    } else {
        let lo = minzoom.clamp(0, 30);
        let hi = maxzoom.clamp(0, 30);
        tms_def
            .matrices
            .iter()
            .filter_map(|m| {
                let z = m.id.parse::<i64>().ok()?;
                if !(lo..=hi).contains(&z) {
                    return None;
                }
                let (c0, c1, r0, r1) = tms_def.tile_limits_for_wgs84_bounds(m, bounds);
                Some(json!({
                    "tileMatrix": m.id,
                    "minTileRow": r0,
                    "maxTileRow": r1,
                    "minTileCol": c0,
                    "maxTileCol": c1,
                }))
            })
            .collect()
    };
    json_resp(
        StatusCode::OK,
        &json!({
            "title": format!("tileset tiled using {tms} TileMatrixSet"),
            "dataType": "map",
            "crs": tms_def.crs_uri,
            "links": [
                { "href": format!("{host}{base}/tiles/{tms}"), "rel": "self", "type": "application/json", "title": format!("Tileset tiled using {tms} TileMatrixSet") },
                { "href": with_qs(format!("{host}{base}/tiles/{tms}/%7Bz%7D/%7Bx%7D/%7By%7D")), "rel": "tile", "templated": true, "title": "Templated link for retrieving Raster tiles" },
                { "href": format!("{host}/tileMatrixSets/{tms}"), "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme", "type": "application/json", "title": format!("Definition of '{tms}' tileMatrixSet") },
                { "href": with_qs(format!("{host}{base}/{tms}/map.html")), "rel": "data", "type": "text/html", "title": format!("Map viewer for '{tms}' tileMatrixSet") }
            ],
            "tileMatrixSetLimits": limits,
            "boundingBox": bounding_box(bounds),
            "accessConstraints": "unclassified"
        }),
    )
}

/// `GET {base}/tiles`: all thirteen OGC tile matrix sets on offer (WebMercatorQuad first).
fn tilesets(ctx: &Ctx<'_>, base: &str, metadata: &Value, pairs: &[(String, String)]) -> Resp {
    // The list has no zooms in it, and titiler-pgstac does not read the request's.
    let (_, _, bounds) = match mosaic_extent(metadata, &[]) {
        Ok(e) => e,
        Err(r) => return r,
    };
    let host = ctx.host;
    let qs = encoded_query(pairs);
    let with_qs = |url: String| if qs.is_empty() { url } else { format!("{url}?{qs}") };
    let ordered = crate::tms::lookup_tms("WebMercatorQuad")
        .into_iter()
        .chain(crate::tms::ALL_TMS.iter().filter(|t| t.id != "WebMercatorQuad"));
    let list: Vec<Value> = ordered
        .map(|t| {
            let tms = t.id;
            json!({
                "title": format!("tileset tiled using {tms} TileMatrixSet"),
                "dataType": "map",
                "crs": t.crs_uri,
                "links": [
                    { "href": with_qs(format!("{host}{base}/tiles/{tms}")), "rel": "self", "type": "application/json", "title": format!("Tileset tiled using {tms} TileMatrixSet") },
                    { "href": with_qs(format!("{host}{base}/tiles/{tms}/%7Bz%7D/%7Bx%7D/%7By%7D")), "rel": "tile", "title": "Templated link for retrieving Raster tiles" },
                    { "href": format!("{host}/tileMatrixSets/{tms}"), "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme", "type": "application/json", "title": format!("Definition of '{tms}' tileMatrixSet") }
                ],
                "boundingBox": bounding_box(bounds),
                "accessConstraints": "unclassified"
            })
        })
        .collect();
    json_resp(StatusCode::OK, &json!({ "tilesets": list }))
}

/// What the WMTS document needs to know about the thing being tiled.
struct WmtsSource<'a> {
    /// Prefix of every layer name: the search hash, or `TiTiler` for an item.
    title: &'a str,
    bounds: [f64; 4],
    minzoom: i64,
    maxzoom: i64,
    /// Layers from `defaults`, as name and tile query string.
    default_layers: Vec<(String, String)>,
}

/// `GET {base}/WMTSCapabilities.xml`: the RESTful WMTS description TiTiler writes, for the
/// one tile matrix set these routes serve. Each layer in `defaults` is a WMTS layer, and so
/// is the request's own query string when it can make a tile.
fn wmts(ctx: &Ctx<'_>, base: &str, source: WmtsSource<'_>, pairs: &[(String, String)]) -> Resp {
    const TMS: &str = "WebMercatorQuad";
    // The tiles' padding and buffer are checked here; their size is the document's own.
    let of_tiles: Vec<(String, String)> = pairs.iter().filter(|(k, _)| k != "tilesize").cloned().collect();
    if tile_bounds_error(&of_tiles).is_some() {
        return detail(StatusCode::BAD_REQUEST, "a parameter of the tiles this document describes is not valid");
    }
    let tile_format = query_value(pairs, "tile_format").unwrap_or("png");
    let media_type = match tile_format {
        "png" => "image/png",
        "jpeg" => "image/jpeg",
        "jpg" => "image/jpg",
        "webp" => "image/webp",
        "npy" => "application/x-binary",
        // The tiles are served in these too.
        "tif" | "tiff" => "image/tiff; application=geotiff",
        "pngraw" => "image/png",
        "jp2" => {
            return detail(
                StatusCode::BAD_REQUEST,
                format!("tile_format={tile_format} is not supported on pgSTAC routes (use png, pngraw, jpg, jpeg, webp, tif, tiff or npy)"),
            );
        }
        _ => {
            return invalid_param(
                "query",
                "tile_format",
                "Input should be 'png', 'npy', 'tif', 'tiff', 'jpeg', 'jpg', 'jp2', 'webp' or 'pngraw'",
            );
        }
    };
    let use_epsg = match query_value(pairs, "use_epsg").map(parse_bool) {
        None => false,
        Some(Some(b)) => b,
        Some(None) => {
            return invalid_param("query", "use_epsg", "Input should be a valid boolean");
        }
    };
    // `zooms=WebMercatorQuad::min,max` overrides the zoom range of the query-string layer.
    let mut zoom_override = None;
    for (_, v) in pairs.iter().filter(|(k, _)| k == "zooms") {
        let parsed = v.split_once("::").and_then(|(tms, range)| {
            let (lo, hi) = range.split_once(',')?;
            Some((tms, lo.trim().parse::<i64>().ok()?, hi.trim().parse::<i64>().ok()?))
        });
        match parsed {
            Some((TMS, lo, hi)) => zoom_override = Some((lo, hi)),
            Some(_) => {}
            None => {
                return invalid_param(
                    "query",
                    "zooms",
                    "Value error, invalid zooms: expected tileMatrixSetId::minzoom,maxzoom",
                );
            }
        }
    }

    // Layers: (name, tile query string, zoom range).
    let mut layers: Vec<(String, String, (i64, i64))> = source
        .default_layers
        .into_iter()
        .map(|(name, qs)| {
            // The tile size is the service's to choose, not the layer's.
            let qs: Vec<&str> = qs.split('&').filter(|kv| kv.split_once('=').is_none_or(|(k, _)| super::query_key(k) != "tilesize")).collect();
            (name, qs.join("&"), (source.minzoom, source.maxzoom))
        })
        .collect();
    const NOT_TILE_PARAMS: [&str; 10] = [
        "tile_format", "use_epsg", "tilesize", "zooms", "service", "request", "acceptversions",
        "sections", "updatesequence", "acceptformats",
    ];
    // `assets=` with nothing after it is a layer all the same, as in titiler-pgstac.
    if query_value(pairs, "assets").is_some() {
        let qs: Vec<String> = pairs
            .iter()
            .filter(|(k, _)| !NOT_TILE_PARAMS.contains(&k.to_ascii_lowercase().as_str()))
            .map(|(k, v)| format!("{}={}", quote_plus(k), quote_plus(v)))
            .collect();
        layers.push((
            "default".into(),
            qs.join("&"),
            zoom_override.unwrap_or((source.minzoom, source.maxzoom)),
        ));
    }
    if layers.is_empty() {
        return detail(
            StatusCode::BAD_REQUEST,
            "Could not find any valid layers in metadata or construct one from Query Parameters.",
        );
    }

    let host = ctx.host;
    let this_url = crate::stats::xml_escape(&if ctx.raw_query.is_empty() {
        format!("{host}{base}/WMTSCapabilities.xml")
    } else {
        format!("{host}{base}/WMTSCapabilities.xml?{}", ctx.raw_query)
    });
    let operation = |name: &str| {
        format!(
            r#"        <ows:Operation name="{name}">
            <ows:DCP>
                <ows:HTTP>
                    <ows:Get xlink:href="{this_url}">
                        <ows:Constraint name="GetEncoding">
                            <ows:AllowedValues>
                                <ows:Value>RESTful</ows:Value>
                            </ows:AllowedValues>
                        </ows:Constraint>
                    </ows:Get>
                </ows:HTTP>
            </ows:DCP>
        </ows:Operation>
"#
        )
    };
    let mut xml = String::with_capacity(16 * 1024);
    xml.push_str(r#"<Capabilities xmlns="http://www.opengis.net/wmts/1.0" xmlns:ows="http://www.opengis.net/ows/1.1" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:gml="http://www.opengis.net/gml" xsi:schemaLocation="http://www.opengis.net/wmts/1.0 http://schemas.opengis.net/wmts/1.0/wmtsGetCapabilities_response.xsd" version="1.0.0">
    <ows:ServiceIdentification>
        <ows:Title>Web Map Tile Service by TiTiler</ows:Title>
        <ows:ServiceType>OGC WMTS</ows:ServiceType>
        <ows:ServiceTypeVersion>1.0.0</ows:ServiceTypeVersion>
    </ows:ServiceIdentification>
    <ows:ServiceProvider>
        <ows:ProviderName>TiTiler</ows:ProviderName>
        <ows:ProviderSite xlink:href="https://developmentseed.org/titiler/"/>
        <ows:ServiceContact/>
    </ows:ServiceProvider>
    <ows:OperationsMetadata>
"#);
    xml.push_str(&operation("GetCapabilities"));
    xml.push_str(&operation("GetTile"));
    xml.push_str("    </ows:OperationsMetadata>\n    <Contents>\n");
    let [w, s, e, n] = source.bounds;
    for (name, qs, (minzoom, maxzoom)) in &layers {
        let id = crate::stats::xml_escape(&format!("{}_{TMS}_{name}", source.title));
        let template = crate::stats::xml_escape(&if qs.is_empty() {
            format!("{host}{base}/tiles/{TMS}/{{TileMatrix}}/{{TileCol}}/{{TileRow}}.{tile_format}")
        } else {
            format!("{host}{base}/tiles/{TMS}/{{TileMatrix}}/{{TileCol}}/{{TileRow}}.{tile_format}?{qs}")
        });
        xml.push_str(&format!(
            r#"        <Layer>
            <ows:Title>{id}</ows:Title>
            <ows:Identifier>{id}</ows:Identifier>
            <ows:WGS84BoundingBox crs="urn:ogc:def:crs:OGC:2:84">
                <ows:LowerCorner>{} {}</ows:LowerCorner>
                <ows:UpperCorner>{} {}</ows:UpperCorner>
            </ows:WGS84BoundingBox>
            <Style isDefault="true">
                <ows:Title>{id}</ows:Title>
                <ows:Identifier>{id}</ows:Identifier>
            </Style>
            <Format>{media_type}</Format>
            <TileMatrixSetLink>
                <TileMatrixSet>{TMS}</TileMatrixSet>
                <TileMatrixSetLimits>
"#,
            python_float(w),
            python_float(s),
            python_float(e),
            python_float(n)
        ));
        for z in (*minzoom).clamp(0, 30)..=(*maxzoom).clamp(0, 30) {
            let l = tile_limits(z, source.bounds);
            xml.push_str(&format!(
                "                    <TileMatrixLimits>\n                        <TileMatrix>{z}</TileMatrix>\n                        <MinTileRow>{}</MinTileRow>\n                        <MaxTileRow>{}</MaxTileRow>\n                        <MinTileCol>{}</MinTileCol>\n                        <MaxTileCol>{}</MaxTileCol>\n                    </TileMatrixLimits>\n",
                l["minTileRow"], l["maxTileRow"], l["minTileCol"], l["maxTileCol"]
            ));
        }
        xml.push_str(&format!(
            "                </TileMatrixSetLimits>\n            </TileMatrixSetLink>\n            <ResourceURL format=\"{media_type}\" resourceType=\"tile\" template=\"{template}\" />\n        </Layer>\n"
        ));
    }
    let crs = if use_epsg { "EPSG:3857" } else { "http://www.opengis.net/def/crs/EPSG/0/3857" };
    xml.push_str(&format!(
        "        <TileMatrixSet>\n            <ows:Identifier>{TMS}</ows:Identifier>\n            <ows:SupportedCRS>{crs}</ows:SupportedCRS>\n"
    ));
    for z in 0..=24u32 {
        let side = 1u64 << z;
        xml.push_str(&format!(
            "            <TileMatrix>\n                <ows:Identifier>{z}</ows:Identifier>\n                <ScaleDenominator>{}</ScaleDenominator>\n                <TopLeftCorner>-20037508.342789244 20037508.342789244</TopLeftCorner>\n                <TileWidth>256</TileWidth>\n                <TileHeight>256</TileHeight>\n                <MatrixWidth>{side}</MatrixWidth>\n                <MatrixHeight>{side}</MatrixHeight>\n            </TileMatrix>\n",
            python_float(559_082_264.028_717_8 / side as f64)
        ));
    }
    xml.push_str(&format!(
        "        </TileMatrixSet>\n    </Contents>\n    <ServiceMetadataURL xlink:href=\"{this_url}\" />\n</Capabilities>"
    ));
    let mut resp = Response::new(Full::new(Bytes::from(xml)));
    resp.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static("application/xml"));
    resp.headers_mut().insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    resp
}

/// Only WGS84 coordinates are taken on the bbox and point routes for now.
fn check_coord_crs(pq: &ParsedQuery, pairs: &[(String, String)]) -> Result<(), Resp> {
    // As the shared parser read it, in any spelling of degrees it takes.
    match pq.coord_crs_epsg {
        None | Some(4326) => Ok(()),
        Some(_) => Err(detail(
            StatusCode::BAD_REQUEST,
            format!(
                "coord_crs '{}' is not supported on pgSTAC mosaic routes (use epsg:4326)",
                query_value(pairs, "coord_crs").unwrap_or_default().to_ascii_lowercase()
            ),
        )),
    }
}

/// A position given in the request's `coord_crs`, in degrees: Web Mercator metres are
/// turned into them, and any other system is refused.
fn in_degrees(pq: &ParsedQuery, pairs: &[(String, String)], x: f64, y: f64) -> Result<(f64, f64), Resp> {
    if pq.coord_crs_3857 {
        return Ok(mercator_to_lonlat_deg(x, y));
    }
    check_coord_crs(pq, pairs).map(|()| (x, y))
}

/// `GET {base}/bbox/{minx},{miny},{maxx},{maxy}/assets` and `GET {base}/point/{lon},{lat}/assets`.
async fn geometry_assets(
    pg: &PgStac,
    hash: &str,
    geometry: Value,
    limits: SearchLimits,
    pairs: &[(String, String)],
    grid: Option<SampleGrid>,
) -> Resp {
    match listed_items(pg, hash, &geometry, limits, pairs, grid).await {
        Ok(items) => json_resp(
            StatusCode::OK,
            &Value::Array(items.iter().map(MosaicItem::to_json).collect()),
        ),
        Err(r) => r,
    }
}

/// The items of an `assets` route. They are `geojsonsearch`'s, whole, as titiler-pgstac
/// lists them. With `item_lookup=lean` they are instead what the lean lookup would hand a
/// picture sampled on `grid`: ids and the hrefs of the assets named, nothing else.
async fn listed_items(
    pg: &PgStac,
    hash: &str,
    geometry: &Value,
    limits: SearchLimits,
    pairs: &[(String, String)],
    grid: Option<SampleGrid>,
) -> Result<Arc<Vec<MosaicItem>>, Resp> {
    let found = if parse_lookup(pairs)? == Lookup::Lean {
        let names: Vec<String> = parse_assets(pairs)?.into_iter().map(|a| a.name).collect();
        pg.items_for_pixels(geometry, hash, limits, &names, grid, Lookup::Lean).await
    } else {
        pg.items_for_geometry(geometry, hash, limits).await
    };
    found.map_err(|e| db_error(&e))
}

/// All thirteen built-in OGC TileMatrixSets are supported on pgSTAC mosaic routes.
/// An unknown name returns TiTiler's `422` validation error.
fn check_tms(tms: &str) -> Result<(), Resp> {
    if crate::tms::lookup_tms(tms).is_some() {
        Ok(())
    } else {
        Err(invalid_param("path", "tileMatrixSetId", "Input should be a supported TileMatrixSet"))
    }
}

fn tilejson(
    ctx: &Ctx<'_>,
    base: &str,
    rec: &SearchRecord,
    tms: &str,
    pairs: &[(String, String)],
) -> Resp {
    if !pairs.iter().any(|(k, _)| k == "assets") {
        return missing_query_param("assets");
    }
    let (mut minzoom, mut maxzoom, bounds) = match mosaic_extent(&rec.metadata, pairs) {
        Ok(e) => e,
        Err(r) => return r,
    };
    if tms != "WebMercatorQuad"
        && let Some(tms_def) = crate::tms::lookup_tms(tms)
    {
        let tms_min = tms_def.matrices.iter().filter_map(|m| m.id.parse::<i64>().ok()).min().unwrap_or(0);
        let tms_max = tms_def.matrices.iter().filter_map(|m| m.id.parse::<i64>().ok()).max().unwrap_or(24);
        minzoom = minzoom.clamp(tms_min, tms_max);
        maxzoom = maxzoom.clamp(minzoom, tms_max);
    }

    // TiTiler drops the parameters that shape the TileJSON itself and pins the tile size.
    let dropped = ["tilematrixsetid", "tile_format", "tile_scale", "minzoom", "maxzoom"];
    let mut qs: Vec<String> = pairs
        .iter()
        .filter(|(k, v)| {
            !dropped.contains(&k.to_ascii_lowercase().as_str())
                && !is_doc_negotiation_param(k, v)
        })
        .map(|(k, v)| format!("{}={}", quote_plus(k), quote_plus(v)))
        .collect();
    if query_value(pairs, "tilesize").is_none() {
        qs.push("tilesize=512".to_string());
    }
    let ext = match query_value(pairs, "tile_format") {
        Some(f) if !f.is_empty() => format!(".{f}"),
        _ => String::new(),
    };
    let tiles = format!(
        "{}{base}/tiles/{tms}/{{z}}/{{x}}/{{y}}{ext}?{}",
        ctx.host,
        qs.join("&")
    );
    let mut out = serde_json::Map::new();
    out.insert("tilejson".into(), json!("3.0.0"));
    out.insert("version".into(), json!("1.0.0"));
    out.insert("scheme".into(), json!("xyz"));
    out.insert("tiles".into(), json!([tiles]));
    out.insert("minzoom".into(), json!(minzoom));
    out.insert("maxzoom".into(), json!(maxzoom));
    out.insert("bounds".into(), json!(bounds));
    out.insert(
        "center".into(),
        json!([(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0, minzoom]),
    );
    out.insert("raster_layers".into(), json!({}));
    json_resp(StatusCode::OK, &Value::Object(out))
}

/// What a tile request asks of the items, beyond the search itself.
/// One entry of `assets`: a name and, after a `|`, TiTiler's options for that asset.
#[derive(Debug, Clone, PartialEq)]
struct AssetSpec {
    name: String,
    /// `bidx=1,2`: the bands to take, counted from one.
    indexes: Option<Vec<u16>>,
    /// `expression=b1*2`: band math over this asset's own bands. It wins over `bidx`.
    expression: Option<String>,
}

/// `assets=red`, `assets=visual|bidx=3,2,1`, `assets=red|expression=b1*2`, each `assets`
/// parameter being one asset. As in TiTiler an option is `key=value`, and the keys it does
/// not know are passed over. An entry without options may still list several names with
/// commas, which the other routes of this server take.
fn parse_assets(pairs: &[(String, String)]) -> Result<Vec<AssetSpec>, Resp> {
    let bad = |msg: &str| invalid_param("query", "assets", format!("Value error, {msg}"));
    pairs
        .iter()
        .filter(|(k, _)| k == "assets")
        .flat_map(|(_, v)| {
            let whole = v.contains('|');
            v.split(move |c| c == ',' && !whole).filter(|part| !part.is_empty())
        })
        .map(|entry| {
            let (name, options) = entry.split_once('|').map_or((entry, None), |(n, o)| (n, Some(o)));
            if name.is_empty() {
                return Err(detail(StatusCode::BAD_REQUEST, "an asset has no name"));
            }
            let mut spec = AssetSpec { name: name.to_string(), indexes: None, expression: None };
            for option in options.into_iter().flat_map(|o| o.split('|')) {
                let (key, value) = option
                    .split_once('=')
                    .ok_or_else(|| bad("not enough values to unpack (expected 2, got 1)"))?;
                match key {
                    "bidx" => {
                        // Bands are counted from one.
                        let indexes: Option<Vec<u16>> =
                            value.split(',').map(|i| i.parse().ok().filter(|b| *b > 0)).collect();
                        spec.indexes = Some(indexes.ok_or_else(|| {
                            bad(&format!("invalid literal for int() with base 10: '{value}'"))
                        })?);
                    }
                    "expression" => spec.expression = Some(value.to_string()),
                    _ => {}
                }
            }
            Ok(spec)
        })
        .collect()
}

/// The bands each `assets` entry chooses, in [`parse_assets`]'s order: its own `|bidx=` or
/// `|indexes=` (the last of them), else what `asset_bidx=name|...` gives its asset, as the
/// shared parser reads them. `None` for every band.
fn entry_selections(pairs: &[(String, String)]) -> Vec<Option<Vec<u16>>> {
    let by_name: Vec<(String, Vec<u16>)> = pairs
        .iter()
        .filter(|(k, _)| k == "asset_bidx")
        .filter_map(|(_, v)| {
            let (name, bands) = v.split_once('|')?;
            Some((name.trim().to_string(), bands.split(',').filter_map(|b| b.trim().parse().ok()).collect()))
        })
        .collect();
    pairs
        .iter()
        .filter(|(k, _)| k == "assets")
        .flat_map(|(_, v)| {
            let whole = v.contains('|');
            v.split(move |c| c == ',' && !whole).filter(|part| !part.is_empty())
        })
        .map(|entry| {
            let mut options = entry.split('|');
            let name = options.next().unwrap_or_default();
            let own = options
                .filter_map(|o| o.strip_prefix("bidx=").or_else(|| o.strip_prefix("indexes=")))
                .next_back()
                .map(|bands| bands.split(',').filter_map(|b| b.trim().parse().ok()).collect::<Vec<u16>>());
            own.or_else(|| {
                let of_it: Vec<u16> = by_name.iter().filter(|(n, _)| n == name).flat_map(|(_, b)| b.clone()).collect();
                (!of_it.is_empty()).then_some(of_it)
            })
        })
        .collect()
}

/// What an asset gives to the stacked bands of an item.
enum AssetBands {
    /// These of its bands, counted from zero.
    Pick(Vec<usize>),
    /// The results of band math over its own bands.
    Math(Program),
}

/// The bands an asset with `bands` bands contributes, and how TiTiler describes them:
/// `visual_b3` for a band, the expression after the asset's name for band math, and the
/// bare name with `asset_as_band`, which only single-band assets can be used with.
fn asset_bands(spec: &AssetSpec, bands: usize, as_band: bool) -> Result<(AssetBands, Vec<String>), String> {
    let (out, mut descriptions) = if let Some(expression) = &spec.expression {
        let program = Program::parse(expression, bands)?;
        let descriptions: Vec<String> =
            program.sources().map(|part| format!("{}_{part}", spec.name)).collect();
        (AssetBands::Math(program), descriptions)
    } else {
        let picked: Vec<usize> = match &spec.indexes {
            None => (0..bands).collect(),
            Some(indexes) => indexes
                .iter()
                .map(|&i| {
                    usize::from(i).checked_sub(1).filter(|i| *i < bands).ok_or_else(|| {
                        let known: Vec<String> = (1..=bands).map(|b| b.to_string()).collect();
                        let tail = if bands == 1 { "," } else { "" };
                        format!("band index {i} out of range (not in ({}{tail}))", known.join(", "))
                    })
                })
                .collect::<Result<_, _>>()?,
        };
        let descriptions = picked.iter().map(|i| format!("{}_b{}", spec.name, i + 1)).collect();
        (AssetBands::Pick(picked), descriptions)
    };
    if as_band {
        if descriptions.len() != 1 {
            return Err("Can't use `asset_as_band` for multibands asset".into());
        }
        descriptions = vec![spec.name.clone()];
    }
    Ok((out, descriptions))
}

impl AssetBands {
    /// The contribution for whole planes, one per band of the asset.
    fn planes(&self, bands: &[Vec<f32>]) -> Vec<Vec<f64>> {
        match self {
            AssetBands::Pick(picked) => {
                picked.iter().map(|&i| bands[i].iter().map(|v| f64::from(*v)).collect()).collect()
            }
            AssetBands::Math(program) => {
                let len = bands.first().map_or(0, Vec::len);
                let mut out = vec![Vec::with_capacity(len); program.len()];
                let mut pixel = vec![0f64; bands.len()];
                for i in 0..len {
                    // A pixel with no band at all, which is most of an empty tile, has
                    // no result either: nothing is worked out for it.
                    let mut any = false;
                    for (value, band) in pixel.iter_mut().zip(bands) {
                        *value = f64::from(band[i]);
                        any |= !band[i].is_nan();
                    }
                    for (part, plane) in out.iter_mut().enumerate() {
                        let v = if any { program.eval(part, &pixel) } else { f64::NAN };
                        plane.push(if v.is_finite() { v } else { f64::NAN });
                    }
                }
                out
            }
        }
    }

    /// The contribution for one pixel, given every band of the asset there. A band with
    /// no value is NaN, and so is band math that does not come out finite.
    fn pixel(&self, bands: &[f64], out: &mut Vec<f64>) {
        match self {
            AssetBands::Pick(picked) => out.extend(picked.iter().map(|&i| bands[i])),
            AssetBands::Math(program) => out.extend((0..program.len()).map(|part| {
                let v = program.eval(part, bands);
                if v.is_finite() { v } else { f64::NAN }
            })),
        }
    }
}

struct MosaicQuery {
    /// The assets asked for, with their options.
    assets: Vec<AssetSpec>,
    /// `asset_as_band`: bands are named after their assets.
    asset_as_band: bool,
    limits: SearchLimits,
    /// Method for a source already in the tile's projection (`resampling`).
    resampling: ResampleMode,
    /// Method for a source that has to be reprojected (`reproject`).
    reproject: ResampleMode,
    /// Which lookup lists the items (`item_lookup`).
    lookup: Lookup,
    /// `pixel_selection`, when it is not `first`.
    selection: Option<Selection>,
}

/// The most a `median` may hold of its scenes at once.
const MAX_HELD_BYTES: usize = 256 * 1024 * 1024;

/// The most a pixel selection may work in, beside the scenes it reads.
const MAX_WORKING_BYTES: usize = 512 * 1024 * 1024;

/// Refuse a pixel selection over more bands and pixels than it may work in.
/// A fold for a pixel selection, refused before anything is read when holding every scene
/// (the median does) or working through them would take too much.
fn fold_within_limits<T: super::pixel_selection::Sample>(
    method: Selection,
    scenes: usize,
    bands: usize,
    pixels: usize,
    integer: bool,
    of_a_tile: bool,
) -> Result<Fold<T>, EngineError> {
    if Fold::<T>::held_bytes(method, scenes, bands, pixels) > MAX_HELD_BYTES {
        return Err(EngineError::BadRequest(if of_a_tile {
            format!("{scenes} items are too many for a median at this tile size; narrow the search or lower items_limit")
        } else {
            format!("{scenes} items are too many for a median at this size; narrow the search, lower items_limit or pass max_size")
        }));
    }
    check_working_bytes(Fold::<T>::working_bytes(method, bands, pixels))?;
    Ok(Fold::new(method, bands, pixels, integer))
}

fn check_working_bytes(bytes: usize) -> Result<(), EngineError> {
    if bytes > MAX_WORKING_BYTES {
        return Err(EngineError::BadRequest(
            "the image is too large for this pixel_selection; pass max_size, or width and height".into(),
        ));
    }
    Ok(())
}

/// TiTiler's two resampling parameters. Both default to nearest neighbour.
fn parse_resample(pairs: &[(String, String)], name: &str) -> Result<ResampleMode, Resp> {
    match query_value(pairs, name) {
        None => Ok(ResampleMode::Nearest),
        Some(method) => resample_method(name, method).ok_or_else(|| invalid_param("query", name, "Input should be a resampling method")),
    }
}

/// The method a `resampling` or `reproject` names, of those rasterio has for each: delegates
/// to [`super::query::parse_resample_mode_strict`].
fn resample_method(parameter: &str, method: &str) -> Option<ResampleMode> {
    super::query::parse_resample_mode_strict(parameter, method)
}

/// A tileset list or tileset asked for in a format other than `html` and `json`, as FastAPI
/// refuses it.
fn tileset_format_error(pairs: &[(String, String)]) -> Option<Resp> {
    let f = query_value(pairs, "f").filter(|f| !matches!(*f, "html" | "json"))?;
    Some(json_resp(
        StatusCode::UNPROCESSABLE_ENTITY,
        &json!({ "detail": [{
            "type": "literal_error",
            "loc": ["query", "f"],
            "msg": "Input should be 'html' or 'json'",
            "input": f,
            "ctx": { "expected": "'html' or 'json'" },
        }] }),
    ))
}

/// A `colormap_name` nobody has, or one not in lower case, as FastAPI refuses it.
fn colormap_name_error(pairs: &[(String, String)]) -> Option<Resp> {
    let name = query_value(pairs, "colormap_name")?;
    (ColormapName::from_str_opt(name).is_none() || name.chars().any(char::is_uppercase)).then(|| {
        json_resp(
            StatusCode::UNPROCESSABLE_ENTITY,
            &json!({ "detail": [{
                "type": "literal_error",
                "loc": ["query", "colormap_name"],
                "msg": "Input should be the name of a colour map (see /colorMaps)",
                "input": name,
            }] }),
        )
    })
}

fn check_mosaic_query(pq: &ParsedQuery, pairs: &[(String, String)]) -> Result<MosaicQuery, Resp> {
    if pq.assets.is_empty() {
        return Err(missing_query_param("assets"));
    }
    // `colormap` is a map written out, never a name; a name nobody has is FastAPI's to refuse.
    if let Some(map) = query_value(pairs, "colormap")
        && !map.is_empty()
        && !map.trim_start().starts_with(['{', '['])
    {
        return Err(detail(StatusCode::BAD_REQUEST, "Could not parse the colormap value."));
    }
    if let Some(resp) = colormap_name_error(pairs) {
        return Err(resp);
    }
    if let Some(e) = pq.errors.iter().find(|e| !own_error(e)) {
        return Err(parse_error(e));
    }
    // The shared parser reads any word as a flag; FastAPI reads only the ones that are.
    for flag in ["unscale", "return_mask"] {
        if query_value(pairs, flag).is_some_and(|v| parse_bool(v).is_none()) {
            return Err(invalid_param("query", flag, "Input should be a valid boolean"));
        }
    }
    let assets = parse_assets(pairs)?;
    if assets.is_empty() {
        return Err(missing_query_param("assets"));
    }
    let asset_as_band = match query_value(pairs, "asset_as_band") {
        None => false,
        Some(v) => parse_bool(v)
            .ok_or_else(|| invalid_param("query", "asset_as_band", "Input should be a valid boolean"))?,
    };
    if assets.len() > MAX_ASSETS_PER_ITEM {
        return Err(detail(
            StatusCode::BAD_REQUEST,
            format!("at most {MAX_ASSETS_PER_ITEM} assets per item are supported, got {}", assets.len()),
        ));
    }
    let selection = match query_value(pairs, "pixel_selection") {
        None => None,
        Some(name) => match Selection::parse(name) {
            Some(Selection::First) => None,
            Some(method) => Some(method),
            None => {
                return Err(invalid_param(
                    "query",
                    "pixel_selection",
                    "Input should be 'first', 'highest', 'lowest', 'mean', 'median', 'stdev', 'lastbandlow', 'lastbandhigh' or 'count'",
                ));
            }
        },
    };
    // A tile's `buffer` is taken before this is asked; anything else has none.
    if pq.buffer > 0.0 {
        return Err(detail(StatusCode::BAD_REQUEST, "parameter 'buffer' is for tiles"));
    }
    if pq.unscale && assets.len() > 1 {
        return Err(detail(
            StatusCode::BAD_REQUEST,
            "parameter 'unscale' is not supported with more than one asset yet",
        ));
    }
    Ok(MosaicQuery {
        selection,
        lookup: parse_lookup(pairs)?,
        assets,
        asset_as_band,
        limits: parse_limits(pairs)?,
        resampling: parse_resample(pairs, "resampling")?,
        reproject: parse_resample(pairs, "reproject")?,
    })
}

/// What a cached tile was made from, for its cache key: the hrefs read, in order.
pub(super) fn source_identity<T: std::hash::Hash>(sources: &T) -> u64 {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    sources.hash(&mut hasher);
    hasher.finish()
}

/// Each asset's own band indexes, as the shared query holds them.
pub(super) type AssetBidx = Vec<(String, Vec<u16>)>;

/// The asset options of an item request, as the `/stac` handlers take them. The shared
/// parser adds every asset's `|bidx=` to one list, which is right for one asset and reads
/// the wrong bands for several; here each asset keeps its own. Band math on one asset is
/// not something those handlers do. A tile or a box with it never gets here: the search
/// routes draw those (`item_picture_by_asset`). A point reads every band of such an asset
/// and works it out afterwards (`item_fixup`); a preview is not planned here, but stacked
/// entry by entry as rio-tiler stacks an item (`expand_stac_asset_bidx`); the statistics read it as one
/// expression over the stacked bands (`item_asset_math`). On what is left, an image over a
/// posted shape or the info, it is refused, not left out.
///
/// Returns each asset's bands and the request's own `bidx`, or `None` when no asset has
/// options. Only the info, statistics and point handlers read bands asset by asset; on
/// the others several assets with bands of their own are refused too.
pub(super) fn item_asset_options(rest: &[&str], raw_query: &str) -> Result<Option<(AssetBidx, Vec<u16>)>, Resp> {
    // A preview and the statistics read each entry's own options as the shared parser
    // records them, and stack the assets (`expand_stac_asset_bidx`).
    if item_preview_route(rest) || matches!(rest, ["statistics"]) {
        return Ok(None);
    }
    let pairs = query_pairs(raw_query);
    let mut specs = parse_assets(&pairs)?;
    // Bands chosen on an asset with band math, which the shared parser has kept: dropped
    // here, so the options are given back even when no other asset chooses any.
    let mut dropped = false;
    if let Some(spec) = specs.iter().find(|s| s.expression.is_some()) {
        if !matches!(rest, ["point", _]) {
            return Err(detail(
                StatusCode::BAD_REQUEST,
                format!("assets={}|expression= is not supported on this item route yet; use `expression`", spec.name),
            ));
        }
        // An asset's band math is over all of its bands, and wins over its `bidx`.
        for spec in specs.iter_mut().filter(|s| s.expression.is_some()) {
            dropped |= spec.indexes.take().is_some();
        }
    }
    if !dropped && specs.iter().all(|s| s.indexes.is_none()) {
        return Ok(None);
    }
    let by_asset = matches!(rest, ["info" | "info.geojson" | "asset_statistics"] | ["point", _]);
    if specs.len() > 1 && !by_asset {
        return Err(detail(
            StatusCode::BAD_REQUEST,
            "`|bidx=` on several assets is not supported on this item route yet",
        ));
    }
    let own: Vec<u16> = pairs
        .iter()
        .filter(|(k, _)| k == "bidx")
        .flat_map(|(_, v)| v.split(','))
        .filter_map(|b| b.trim().parse().ok())
        .collect();
    Ok(Some((specs.into_iter().filter_map(|s| s.indexes.map(|i| (s.name, i))).collect(), own)))
}

/// Whether an `assets` option may be band math (`red|expression=b1*2`): a look at the
/// decoded values that spares the requests without it a parse of their assets.
pub(super) fn asset_math_asked(pairs: &[(String, String)]) -> bool {
    pairs.iter().any(|(k, v)| k == "assets" && v.split('|').skip(1).any(|option| option.starts_with("expression=")))
}

/// An item's preview, which the `/stac` preview stacks as rio-tiler does, each asset's own
/// options with it (`expand_stac_asset_bidx`).
pub(super) fn item_preview_route(rest: &[&str]) -> bool {
    match rest {
        ["preview", ..] => true,
        [one] => one.starts_with("preview."),
        _ => false,
    }
}

/// The parameters the shared parser checks that FastAPI validates on titiler-pgstac's
/// routes, where a value it cannot read is a 422.
const CHECKED: [&str; 11] =
    ["coord_crs", "dst_crs", "crs", "max_size", "width", "height", "nodata", "unscale", "return_mask", "buffer", "padding"];

/// Whether an item's route reads `name`, one of [`CHECKED`], as titiler-pgstac's routes do
/// (each asked with each, 2026-11-26): what a route reads FastAPI validates, a 422 when it
/// cannot be read, before the item is looked for; what it does not read it passes over.
fn item_route_reads(method: &Method, rest: &[&str], name: &str) -> bool {
    if matches!(name, "algorithm" | "algorithm_params") {
        return item_applies_algorithm(rest);
    }
    const TILE: [&str; 5] = ["nodata", "unscale", "return_mask", "buffer", "padding"];
    const PICTURE: [&str; 8] = ["coord_crs", "dst_crs", "max_size", "width", "height", "nodata", "unscale", "return_mask"];
    let reads = |list: &[&str]| list.contains(&name);
    // A size in the path is the size: `width` and `height` in the query are not read.
    let sized = |list: &[&str]| reads(list) && !matches!(name, "width" | "height");
    let named = |seg: &str, route: &str| seg == route || seg.strip_prefix(route).is_some_and(|ext| ext.starts_with('.'));
    match rest {
        ["tiles", _, _, _, _] | [_, "tilejson.json" | "map.html"] => reads(&TILE),
        ["point", _] => matches!(name, "coord_crs" | "nodata" | "unscale"),
        ["bbox", _, _] => sized(&PICTURE),
        ["bbox", _] => reads(&PICTURE),
        ["feature", _] => sized(&PICTURE),
        [one] if named(one, "feature") => reads(&PICTURE),
        // A preview is of the whole item, in no system but the one it is drawn in.
        ["preview", _] => sized(&PICTURE[1..]),
        [one] if named(one, "preview") => reads(&PICTURE[1..]),
        ["statistics"] if *method == Method::POST => matches!(name, "coord_crs" | "dst_crs" | "max_size" | "width" | "height" | "nodata" | "unscale"),
        ["statistics" | "asset_statistics"] => matches!(name, "max_size" | "width" | "height" | "nodata" | "unscale"),
        _ => false,
    }
}

/// Whether an item's route applies `algorithm`: as a mosaic's routes, and its preview and
/// statistics; not its point, `asset_statistics` or `info`.
fn item_applies_algorithm(rest: &[&str]) -> bool {
    match rest {
        ["preview" | "statistics", ..] => true,
        [one] if one.starts_with("preview.") => true,
        _ => applies_algorithm(rest),
    }
}

/// Whether an item's route passes over the parameter `name`, one the shared parser checks
/// (or `algorithm` and its parameters) but the route does not read: the request goes on
/// without it, as in titiler-pgstac.
pub(super) fn item_passes_over(method: &Method, rest: &[&str], name: &str) -> bool {
    (CHECKED.contains(&name) || matches!(name, "algorithm" | "algorithm_params")) && !item_route_reads(method, rest, name)
}

/// Whether the shared parser's complaint `e` is about a parameter [`item_passes_over`].
pub(super) fn item_passes_over_error(method: &Method, rest: &[&str], e: &str) -> bool {
    error_parameter(e).is_some_and(|name| item_passes_over(method, rest, name))
}

/// The answer to a complaint of the shared parser. A value it could not read, of a
/// parameter FastAPI checks in titiler-pgstac, is a 422 as there; anything else (what is
/// not supported here, above all) stays the 400 it was.
fn parse_error(e: &str) -> Resp {
    match error_parameter(e) {
        Some(name) if e.starts_with("invalid ") && CHECKED.contains(&name) => {
            let msg = if name.ends_with("crs") { "Value error, invalid CRS format" } else { "Input should be a valid value" };
            invalid_param("query", name, msg)
        }
        _ => detail(StatusCode::BAD_REQUEST, e.to_string()),
    }
}

/// Whether the shared parser's complaint is about something these routes read for
/// themselves and answer for: `reproject`, whose methods are checked here, and
/// `asset_as_band`, `histogram_bins` and the options of `assets`, which get a 422.
fn own_error(e: &str) -> bool {
    // The selection is read here, and one nobody has is a 422 as titiler-pgstac answers it.
    // A `colormap` with nothing in it is passed over.
    if e.starts_with("unsupported pixel_selection") || e.starts_with("invalid colormap=:") {
        return true;
    }
    match error_parameter(e) {
        Some("reproject" | "resampling" | "asset_as_band" | "histogram_bins") => true,
        Some("assets") => e.ends_with("in |bidx="),
        _ => false,
    }
}

/// Whether the shared parser's complaint is about a `width` or `height` it could not read.
/// Routes that take those answer for them themselves, with a 422.
fn own_size_error(e: &str) -> bool {
    matches!(error_parameter(e), Some("width" | "height"))
}

/// The parameter one of the shared parser's complaints is about. They read
/// `invalid name=value: why` or `parameter 'name' is not supported by cogzilla`; the name
/// is taken from where it stands, never looked for in the value.
fn error_parameter(e: &str) -> Option<&str> {
    if let Some(rest) = e.strip_prefix("invalid ") {
        return rest.split_once('=').map(|(name, _)| name);
    }
    e.strip_prefix("parameter '")?.split_once('\'').map(|(name, _)| name)
}

/// The request's `expression`, under any of the names the shared parser takes for it; as
/// there, the last one given is the one used.
fn request_expression(pairs: &[(String, String)]) -> Option<&str> {
    pairs.iter().rev().find(|(k, _)| matches!(k.as_str(), "expression" | "expr" | "index")).map(|(_, v)| v.as_str())
}

/// The asset names of a request, once its `expression` is known not to use them: band math
/// is over `b1..bn`, never over asset names.
fn expression_names(pq: &ParsedQuery, query: &MosaicQuery) -> Result<Vec<String>, Resp> {
    let names: Vec<String> = query.assets.iter().map(|a| a.name.clone()).collect();
    if let Some(expression) = &pq.expr_raw {
        let words = expression.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
        if let Some(name) = words.filter(|w| !w.is_empty()).find(|w| names.iter().any(|n| n == w)) {
            return Err(detail(StatusCode::BAD_REQUEST, format!("Invalid band/asset name '{name}'")));
        }
    }
    Ok(names)
}

/// Whether a tile has to be made from planes, each asset read on its own, rather than by
/// the renderer's single pass over the bands of one raster or the first band of up to
/// three on one grid. Assets on different grids are found out later, from the rasters.
fn wants_planes(pq: &ParsedQuery, query: &MosaicQuery) -> bool {
    let math = pq.expr_raw.as_deref();
    let plain = |a: &AssetSpec| a.expression.is_none() && a.indexes.as_ref().is_none_or(|i| i == &[1]);
    query.selection.is_some_and(Selection::by_last_band)
        || math.is_some_and(|e| e.contains(';'))
        || match query.assets.as_slice() {
            [one] => match (&one.expression, &one.indexes) {
                (Some(own), _) => math.is_some() || own.contains(';'),
                (None, Some(indexes)) => math.is_some() || !matches!(indexes.len(), 1 | 3),
                (None, None) => false,
            },
            // Two bands are a picture of their own in TiTiler, not the first of them in grey.
            several => {
                several.len() > MAX_ASSETS_PER_SCENE
                    || !several.iter().all(plain)
                    || (several.len() == 2 && math.is_none())
            }
        }
}

/// The most values, bands times pixels, one item's assets may be read into for an image.
const MAX_PLANE_VALUES: usize = 64 * 1024 * 1024;

/// The most bands the assets of one item may stack up to on a tile.
const MAX_BANDS_PER_ITEM: usize = 32;

/// Whether band math over planes reads every expression of a request. The renderer's own
/// engine knows names and functions it does not (`ndvi`, `^`, `max(a, b)`); and planes read
/// plain arithmetic only, whose arrays' type the shared typing of band math knows.
fn planes_read_all(pq: &ParsedQuery, query: &MosaicQuery) -> bool {
    let reads = |e: &String| Program::parse(e, usize::from(u16::MAX)).is_ok_and(|p| p.plain());
    pq.expr_raw.as_ref().is_none_or(reads) && query.assets.iter().all(|a| a.expression.as_ref().is_none_or(reads))
}

/// A tile made from planes: what to read of each item and what to make of it.
struct PlanesTile {
    assets: Vec<AssetSpec>,
    as_band: bool,
    expression: Option<String>,
    selection: Selection,
    resampling: ResampleMode,
    reproject: ResampleMode,
    window: Window,
    /// The value that stands for no data, when the request names one: it is then that
    /// value and not the raster's own, as GDAL has it.
    nodata: Option<f32>,
    /// Whether a NumPy array is asked for, which holds any number of bands where a
    /// picture holds three.
    array: bool,
    /// Whether the per-pixel work goes to the pool for heavy work: when there is more
    /// than one raster to work out, or a picture larger than a tile.
    pool: bool,
    /// Whether the band math is `normalizedIndex`'s, whose array TiTiler writes as float32.
    index: bool,
}

/// A polygon an image is cut to, and which of the picture's pixels are inside it once
/// that has been worked out.
type PartMask = (Arc<PolygonMask>, tokio::sync::OnceCell<Vec<bool>>);

/// What a mosaic of planes is read over.
enum Window {
    /// A map tile: the tile, its size and `unscale`; the rest is set for each read.
    Tile(crate::engine::TileRequest),
    /// A box at a size in pixels, optionally cut to a polygon: in degrees, or with
    /// `mercator` in Web Mercator metres. `size` is what was asked for, from which each
    /// raster's own size follows.
    Part { bbox: [f64; 4], mercator: bool, width: u32, height: u32, size: PartSize, mask: Option<PartMask>, unscale: bool },
}

impl Window {
    fn pixels(&self) -> usize {
        match self {
            Window::Tile(request) => (request.tile_size as usize).pow(2),
            Window::Part { width, height, .. } => *width as usize * *height as usize,
        }
    }

    fn unscale(&self) -> bool {
        match self {
            Window::Tile(request) => request.unscale,
            Window::Part { unscale, .. } => *unscale,
        }
    }

    /// Bands of one raster over the window, each a plane of `f32` on the window's grid
    /// whatever the raster's own, NaN where it has no data; none asked for is all of them.
    #[allow(clippy::too_many_arguments)]
    async fn read(
        &self,
        engine: &TileEngine,
        href: &str,
        bands_1based: &[u16],
        resampling: ResampleMode,
        reproject: ResampleMode,
        max_values: usize,
        pool: bool,
        nodata: Option<f32>,
    ) -> Result<(Vec<Vec<f32>>, SampleType), EngineError> {
        let (bbox, mercator, width, height, size, mask, unscale) = match self {
            Window::Tile(request) => {
                let cog = engine.store.open_uri(href).await.map(|c| engine.store.with_nodata(&c, nodata))?;
                let crs = engine.get_or_build_crs(&cog)?;
                let resample = if matches!(crs, CrsModel::Epsg3857) { resampling } else { reproject };
                let Some(res) = warped_resolution(&crs, &cog, true).filter(|_| resample == ResampleMode::Nearest) else {
                    // The tile engine reads the raster's own `nodata` and tiles of 256 and
                    // 512; a `nodata` of the request's, or another size, is read as a box
                    // over the tile's bounds is.
                    if nodata.is_some()
                        || !matches!(request.tile_size, 256 | 512)
                        || engine_gathers_too_little(&crs, &cog, resample, request.tile.z)
                    {
                        let full = cog.index.full_res();
                        let wanted = TileEngine::wanted_bands(full.samples_per_pixel, full.extra_samples.len(), bands_1based)?;
                        let (w, s, e, n) = tile_bounds_3857(request.tile);
                        let side = request.tile_size as usize;
                        if wanted.len() * side * side > max_values {
                            return Err(EngineError::BadRequest(format!(
                                "{} bands of an asset are too many to hold for a tile; choose fewer with `bidx`",
                                wanted.len()
                            )));
                        }
                        let planes = match resampled_planes(engine, &cog, [w, s, e, n], true, (side, side), &wanted, None, resample, request.unscale, nodata).await {
                            Ok(planes) => planes,
                            // The item's footprint meets the tile but this raster does not.
                            Err(EngineError::NotFound(_)) => vec![vec![f32::NAN; side * side]; wanted.len()],
                            Err(e) => return Err(e),
                        };
                        return Ok((in_raster_type(planes, full.sample_type, request.unscale), full.sample_type));
                    }
                    let (planes, sample_type) = engine.raster_planes(href, bands_1based, request, resampling, reproject, nodata).await?;
                    return Ok((in_raster_type(planes, sample_type, request.unscale), sample_type));
                };
                // The way GDAL's warp reads a tile: from the overview it would pick for
                // the tile's span at the raster's own resolution.
                let full = cog.index.full_res();
                let wanted = TileEngine::wanted_bands(full.samples_per_pixel, full.extra_samples.len(), bands_1based)?;
                let (w, s, e, n) = tile_bounds_3857(request.tile);
                let side = request.tile_size as usize;
                if wanted.len() * side * side > max_values {
                    return Err(EngineError::BadRequest(format!(
                        "{} bands of an asset are too many to hold for a tile; choose fewer with `bidx`",
                        wanted.len()
                    )));
                }
                let own = |span: f64| ((span / res).round() as u64).max(1);
                // One road whatever bands are asked for. GDAL takes another when only some
                // are, and a band then comes from other pixels than it does in the whole
                // raster; a band is the same band here either way.
                let read =
                    warp_nearest(engine, &cog, &crs, [w, s, e, n], true, (side, side), (own(e - w), own(n - s)), WarpPath::Direct, request.unscale, pool, &wanted, true, nodata)
                        .await?;
                let mut planes = Vec::with_capacity(wanted.len());
                for band in &wanted {
                    planes.push(read.get(usize::from(*band) - 1).cloned().ok_or_else(|| {
                        EngineError::BadRequest("unexpected number of planes from the reader".into())
                    })?);
                }
                return Ok((planes, full.sample_type));
            }
            Window::Part { bbox, mercator, width, height, size, mask, unscale } => (*bbox, *mercator, *width, *height, *size, mask, *unscale),
        };
        let cog = engine.store.open_uri(href).await.map(|c| engine.store.with_nodata(&c, nodata))?;
        let full = cog.index.full_res();
        let sample_type = full.sample_type;
        let wanted = TileEngine::wanted_bands(full.samples_per_pixel, full.extra_samples.len(), bands_1based)?;
        let pixels = self.pixels();
        // Known from the header alone, so nothing is read of an image too large to hold.
        if wanted.len() * pixels > max_values {
            return Err(EngineError::BadRequest(format!(
                "{} bands of an asset at {width}x{height} pixels are too many to hold; pass max_size, or width and height",
                wanted.len()
            )));
        }
        // As on tiles: `resampling` for a source already in the output's projection,
        // `reproject` for one that has to be warped into it.
        let resample = match engine.get_or_build_crs(&cog)? {
            CrsModel::Geographic if !mercator => resampling,
            CrsModel::Epsg3857 if mercator => resampling,
            _ => reproject,
        };
        // The renderer burns a line one pixel to a column and leaves points out; GDAL burns
        // every pixel a line crosses and the one a point is in, as the shape is burned below.
        let lines_or_points = mask.as_ref().is_some_and(|(m, _)| !m.lines.is_empty() || !m.points.is_empty());
        if resample == ResampleMode::Nearest || mercator || lines_or_points {
            // As rio-tiler does: the raster is read at the size its own resolution gives
            // for what was asked, the way GDAL's warp reads it, and brought to the
            // picture's size afterwards.
            let out = (width as usize, height as usize);
            let mut planes: Vec<Vec<f32>> = if resample != ResampleMode::Nearest {
                // Smoothed on a grid in metres, or under lines or points: the box renderer,
                // and the shape burned below.
                match resampled_planes(engine, &cog, bbox, mercator, out, &wanted, None, resample, unscale, nodata).await {
                    Ok(planes) => in_raster_type(planes, sample_type, unscale),
                    // The item's footprint meets the box but this raster does not.
                    Err(EngineError::NotFound(_)) => vec![vec![f32::NAN; out.0 * out.1]; wanted.len()],
                    Err(e) => return Err(e),
                }
            } else {
            let crs = engine.get_or_build_crs(&cog)?;
            let res = warped_resolution(&crs, &cog, mercator)
                .ok_or_else(|| EngineError::BadRequest("cannot place an asset in WGS84".into()))?;
            let native = part_dims(PartSize::Query { max_size: None, width: None, height: None }, bbox, res);
            let (w, h) = part_dims(size, bbox, res);
            let all = usize::from(full.samples_per_pixel);
            // A raster finer than the first item's, which set the picture's size, would be
            // read larger than an image may be: it is read at the picture's size instead.
            let own = if w > u64::from(MAX_PART_SIDE) || h > u64::from(MAX_PART_SIDE) || all as u64 * w * h > max_values as u64 {
                (width as usize, height as usize)
            } else {
                (w as usize, h as usize)
            };
            if all * own.0 * own.1 > max_values {
                return Err(EngineError::BadRequest(format!(
                    "{all} bands of an asset at {}x{} pixels are too many to hold; pass max_size, or width and height",
                    own.0, own.1
                )));
            }
            // One road whatever bands are asked for: see the tile's reading above.
            let read = match warp_nearest(engine, &cog, &crs, bbox, mercator, own, native, WarpPath::Direct, unscale, true, &[], true, nodata).await {
                Ok(read) => read,
                // The item's footprint meets the box but this raster does not.
                Err(EngineError::NotFound(_)) => vec![vec![f32::NAN; own.0 * own.1]; all],
                Err(e) => return Err(e),
            };
            let mut planes: Vec<Vec<f32>> = Vec::with_capacity(wanted.len());
            for band in &wanted {
                let plane = read.get(usize::from(*band) - 1).cloned().ok_or_else(|| {
                    EngineError::BadRequest("unexpected number of planes from the reader".into())
                })?;
                planes.push(resize_nearest(plane, own, out));
            }
            planes
            };
            if let Some((mask, inside)) = mask {
                // Worked out once for the picture, and not on the workers that serve requests.
                let inside = inside
                    .get_or_try_init(|| async {
                        let mask = mask.clone();
                        engine
                            .store
                            .run_heavy_cpu(move || {
                                // GDAL's all-touched rule, which is how rio-tiler burns a
                                // shape: a pixel is inside when its centre is, or when an
                                // edge passes through it. Each polygon is burned on its own,
                                // so two that overlap are their union.
                                let shape = rings_on_grid(&mask, bbox, out, mercator);
                                let burned = zonal::shape_cover_counts(&shape, out.0, out.1, 1).map_err(EngineError::BadRequest)?;
                                let inside: Vec<bool> = burned.into_iter().map(|n| n > 0).collect();
                                Ok::<_, EngineError>(inside)
                            })
                            .await?
                    })
                    .await?;
                for plane in &mut planes {
                    for (v, inside) in plane.iter_mut().zip(inside) {
                        if !inside {
                            *v = f32::NAN;
                        }
                    }
                }
            }
            return Ok((planes, sample_type));
        }
        // The renderer takes the request's own `nodata` in place of the raster's, and burns the shape.
        let shape = mask.as_ref().map(|(mask, _)| mask.as_ref());
        let planes =
            resampled_planes(engine, &cog, bbox, false, (width as usize, height as usize), &wanted, shape, resample, unscale, nodata).await?;
        Ok((in_raster_type(planes, sample_type, unscale), sample_type))
    }
}

/// The bands one item brings to a tile.
struct ItemBands {
    bands: Vec<Vec<f32>>,
    /// How many bands were read to make them.
    stacked: usize,
    /// Whether they are the rasters' own whole numbers.
    integer: bool,
    sample_type: SampleType,
    /// The type that holds the numbers of every asset read; `sample_type` is the first's.
    widest: SampleType,
}

/// What one asset brings: its planes, its sample type, and whether they are its own numbers.
type AssetRead = (Vec<Vec<f32>>, SampleType, bool);

/// A read under way.
type Reading<'a, T> = std::pin::Pin<Box<dyn Future<Output = Result<T, EngineError>> + Send + 'a>>;

/// Band math over whole planes, in double precision; a result that is not a number is no data.
fn band_math(expression: &str, bands: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, EngineError> {
    let program = Program::parse(expression, bands.len()).map_err(EngineError::BadRequest)?;
    if !program.plain() {
        return Err(EngineError::BadRequest(format!(
            "expression '{expression}' is read on points and statistics; pictures and arrays take arithmetic and the functions of one argument"
        )));
    }
    let len = bands.first().map_or(0, Vec::len);
    let mut out = vec![vec![f32::NAN; len]; program.len()];
    let mut pixel = vec![0f64; bands.len()];
    for i in 0..len {
        // A pixel with no band at all, which is most of an empty tile, has no result
        // either: nothing is worked out for it.
        let mut any = false;
        for (value, band) in pixel.iter_mut().zip(bands) {
            *value = f64::from(band[i]);
            any |= !band[i].is_nan();
        }
        if !any {
            continue;
        }
        for (part, plane) in out.iter_mut().enumerate() {
            let v = program.eval(part, &pixel);
            if v.is_finite() {
                plane[i] = v as f32;
            }
        }
    }
    Ok(out)
}

impl PlanesTile {
    /// One asset's bands over the tile, read on its own grid, with its own options applied.
    async fn asset(
        &self,
        engine: &TileEngine,
        href: &str,
        spec: &AssetSpec,
    ) -> Result<AssetRead, EngineError> {
        let wanted: &[u16] = match (&spec.expression, &spec.indexes) {
            (None, Some(indexes)) => indexes,
            _ => &[],
        };
        // Each asset may take its share of what an item may hold.
        let (planes, sample_type) =
            self.window
                .read(engine, href, wanted, self.resampling, self.reproject, MAX_PLANE_VALUES / self.assets.len().max(1), self.pool, self.nodata)
                .await?;
        let planes = match &spec.expression {
            Some(expression) => band_math(expression, &planes)?,
            None => planes,
        };
        if self.as_band && planes.len() != 1 {
            return Err(EngineError::BadRequest("Can't use `asset_as_band` for multibands asset".into()));
        }
        Ok((planes, sample_type, spec.expression.is_none()))
    }

    /// One item's bands over the tile: its assets read together, stacked in the order asked
    /// for, then the request's band math.
    async fn item(&self, engine: &TileEngine, hrefs: &[String]) -> Result<ItemBands, EngineError> {
        let mut reads: Vec<Reading<'_, AssetRead>> = Vec::with_capacity(hrefs.len());
        for (href, spec) in hrefs.iter().zip(&self.assets) {
            reads.push(Box::pin(self.asset(engine, href, spec)));
        }
        let parts = futures_util::future::try_join_all(reads).await?;
        let sample_type = parts.first().map_or(SampleType::Uint8, |(_, st, _)| *st);
        // The one type that holds every asset's numbers, as numpy stacks them.
        let widest = parts.iter().map(|(_, st, _)| *st).reduce(wider_type).unwrap_or(sample_type);
        let integer = self.expression.is_none()
            && !self.window.unscale()
            && parts.iter().all(|(_, st, own)| *own && !matches!(st, SampleType::Float32 | SampleType::Float64));
        let stacked: usize = parts.iter().map(|(planes, _, _)| planes.len()).sum();
        if stacked > MAX_BANDS_PER_ITEM {
            return Err(EngineError::BadRequest(format!(
                "{stacked} bands per item are too many for a tile (at most {MAX_BANDS_PER_ITEM}); choose bands with `bidx`"
            )));
        }
        let mut bands: Vec<Vec<f32>> = parts.into_iter().flat_map(|(planes, _, _)| planes).collect();
        if let Some(expression) = &self.expression {
            bands = band_math(expression, &bands)?;
        }
        Ok(ItemBands { bands, stacked, integer, sample_type, widest })
    }

    /// The mosaic of the items' bands, and the first raster's sample type. Items are read
    /// up to eight at once and taken in order; only `first` stops before the last of them.
    async fn mosaic(
        &self,
        engine: &TileEngine,
        scenes: &[Vec<String>],
    ) -> Result<Option<(Selected, SampleType, SampleType)>, EngineError> {
        use futures_util::StreamExt;
        let pixels = self.window.pixels();
        let mut reading: futures_util::stream::FuturesOrdered<Reading<'_, ItemBands>> =
            futures_util::stream::FuturesOrdered::new();
        let mut next = 0;
        let mut fold: Option<(Fold, SampleType, SampleType)> = None;
        // A tile's planes are small, and its items are read eight at once from the start.
        // An image's can be large: its first item is read alone, and how many follow at
        // once is set by what that one took.
        // So can a tile's above 512 pixels a side, which is read as an image is.
        let small = matches!(self.window, Window::Tile(_)) && pixels <= 512 * 512;
        let mut ahead = if small { MOSAIC_SCENES_AHEAD } else { 1 };
        loop {
            while reading.len() < ahead && next < scenes.len() {
                reading.push_back(Box::pin(self.item(engine, &scenes[next])));
                next += 1;
            }
            let Some(item) = reading.next().await else {
                break;
            };
            let item = item?;
            if !small {
                ahead = (2 * MAX_PLANE_VALUES / (item.stacked.max(item.bands.len()).max(1) * pixels).max(1))
                    .clamp(1, MOSAIC_SCENES_AHEAD);
            }
            let (fold, _, _) = match &mut fold {
                Some(fold) => fold,
                None => {
                    let bands = item.bands.len();
                    if bands == 0 || (self.selection.by_last_band() && bands < 2) {
                        return Err(EngineError::BadRequest(
                            "pixel_selection by the last band needs at least two bands".into(),
                        ));
                    }
                    let new = fold_within_limits::<f32>(self.selection, scenes.len(), bands, pixels, item.integer, true)?;
                    fold.insert((new.absent_as(self.nodata.unwrap_or(0.0)), item.sample_type, item.widest))
                }
            };
            if item.bands.len() != fold.bands() {
                return Err(EngineError::BadRequest(
                    "the items under the tile do not have the same number of bands".into(),
                ));
            }
            fold.add(item.bands.iter().flatten().copied());
            // More bands than a picture has are refused whatever the other items hold, so
            // none of them is read; the first is kept to say how the refusal is worded.
            let left = match self.selection {
                Selection::Count => 1,
                method if method.by_last_band() => fold.bands() - 1,
                _ => fold.bands(),
            };
            if fold.is_done() || (left > 3 && !self.array) {
                break;
            }
        }
        Ok(fold.and_then(|(fold, sample_type, widest)| fold.finish().map(|selected| (selected, sample_type, widest))))
    }
}

/// How the bands a mosaic of planes leaves are to be shown.
struct Look<'a> {
    rescale: &'a [(f32, f32)],
    colormap: ColormapName,
    /// The request's own colour map, given as JSON: classes or intervals of values.
    custom: Option<&'a MosaicColormap>,
    formula: Option<&'a cogzilla_kernel::ColorFormula>,
    /// The format asked for and its media type; `None` leaves it to the mask.
    asked: Option<(OutputFormat, &'static str)>,
    jpeg_quality: u8,
    return_mask: bool,
    /// Whether a colour map was named: one band is then a picture in colour, not a grey one.
    mapped: bool,
}

impl PlanesTile {
    /// The mosaic as a picture of `width` by `height`, which is the window's size.
    async fn picture(
        &self,
        engine: &TileEngine,
        scenes: &[Vec<String>],
        width: usize,
        height: usize,
        look: &Look<'_>,
    ) -> Result<(Bytes, &'static str), EngineError> {
        // An asset's band math, as the request's own (see `band_math`), is plain arithmetic
        // on a picture or an array.
        for expression in self.assets.iter().filter_map(|a| a.expression.as_deref()) {
            if Program::parse(expression, usize::from(u16::MAX)).is_ok_and(|p| !p.plain()) {
                return Err(EngineError::BadRequest(format!(
                    "expression '{expression}' is read on points and statistics; pictures and arrays take arithmetic and the functions of one argument"
                )));
            }
        }
        let pixels = width * height;
        let mut formula = look.formula;
        let (rgba, bands, full) = match self.mosaic(engine, scenes).await? {
            None => (vec![0u8; pixels * 4], 1, false),
            Some((selected, sample_type, widest)) => {
                // Without `rescale`: the type's own range for a raster's numbers, and band math
                // as the renderer draws it: three results (`b3;b2;b1`) in the range of the
                // raster they were read from, one (an index such as `(b2-b1)/(b2+b1)`) from -1
                // to 1. (TiTiler, without `rescale`, stretches the result type's whole range:
                // every pixel 127, or 0.)
                let math = self.expression.is_some() || self.assets.iter().any(|a| a.expression.is_some());
                let default = match (math, selected.bands) {
                    (false, bands) => super::default_rescale_for_dtype(sample_type, bands == 1),
                    (true, 3) => super::default_rescale_for_dtype(widest, false),
                    (true, _) => (-1.0, 1.0),
                };
                // A pixel is there when any band of it is.
                let per_band = selected.values.len() / selected.bands.max(1);
                let full =
                    (0..per_band).all(|i| selected.values.iter().skip(i).step_by(per_band).any(|v| !v.is_nan()));
                // The driver TiTiler would have tried: the one asked for, or JPEG when
                // every pixel is there and PNG when not.
                let driver = match look.asked {
                    Some((OutputFormat::Jpeg { .. }, _)) => "JPEG",
                    Some((OutputFormat::Webp { .. }, _)) => "WEBP",
                    Some(_) => "PNG",
                    None if full => "JPEG",
                    None => "PNG",
                };
                if look.custom.is_some() && selected.bands != 1 {
                    return Err(EngineError::BadRequest("Source data must be 1 band".into()));
                }
                // An array of the values as they are, when nothing was asked that turns
                // them into a picture's bytes.
                let shown = !look.rescale.is_empty() || look.mapped || look.custom.is_some() || look.formula.is_some();
                // The type those values are in.
                let kind = || {
                    let math_of = || self.expression.iter().chain(self.assets.iter().filter_map(|a| a.expression.as_ref()));
                    // What the bands are before any math: floats once unscaled.
                    let read = if self.window.unscale() { SampleType::Float32 } else { widest };
                    if matches!(self.selection, Selection::Stdev) {
                        SampleType::Float64
                    } else if matches!(self.selection, Selection::Count) {
                        SampleType::Uint8
                    } else if self.index {
                        SampleType::Float32
                    } else if !math {
                        read
                    } else {
                        band_math_type(math_of(), read)
                    }
                };
                // A PNG holds 16 bits as well as 8: bands of 16-bit whole numbers, asked for
                // as a PNG and as they are, are written so, as GDAL writes them for TiTiler.
                // It has room for four samples a pixel, the mask among them.
                if let (Some((OutputFormat::Png, content_type)), false, 1..=4) =
                    (look.asked, shown, selected.bands + usize::from(look.return_mask))
                    && kind() == SampleType::Uint16
                {
                    return Ok((png_of_16_bits(&selected.values, selected.bands, width, height, look.return_mask), content_type));
                }
                if let (Some((OutputFormat::Npy, content_type)), false) = (look.asked, shown) {
                    let kind = kind();
                    return Ok((npy_array(&selected.values, selected.bands, width, height, kind, look.return_mask, self.nodata), content_type));
                }
                // An array of rescaled values and nothing else: each band's bytes, however
                // many bands there are.
                if matches!(look.asked, Some((OutputFormat::Npy, _))) && !look.mapped && look.custom.is_none() && look.formula.is_none() {
                    // One range for each band when there are as many; the first for all otherwise.
                    let range = |band: usize| match look.rescale {
                        [] => default,
                        several if several.len() == selected.bands => several[band],
                        several => several[0],
                    };
                    let values: Vec<f32> = selected
                        .values
                        .chunks(per_band.max(1))
                        .enumerate()
                        .flat_map(|(band, plane)| {
                            let (min, max) = range(band);
                            plane.iter().map(move |v| if v.is_nan() { f32::NAN } else { f32::from(rescale_byte(*v, min, max)) })
                        })
                        .collect();
                    let content_type = look.asked.map_or("application/x-binary", |(_, ct)| ct);
                    return Ok((npy_array(&values, selected.bands, width, height, SampleType::Uint8, look.return_mask, None), content_type));
                }
                // One band under a colour map takes the formula before the map; the
                // formula is then done with.
                let before_map = formula.filter(|_| {
                    selected.bands == 1 && (look.mapped || look.custom.is_some())
                });
                if before_map.is_some() {
                    formula = None;
                }
                let rgba = paint_bands(&selected, pixels, look.rescale, default, look.colormap, look.custom, before_map).map_err(|bands| {
                    EngineError::BadRequest(format!(
                        "Could not encode array of shape ({bands},{height},{width}) and of datatype `uint8` using {driver} driver"
                    ))
                })?;
                // Under a colour map of the request's own, a pixel is there when its colour
                // is not wholly transparent: that is the mask TiTiler picks a format by.
                let full = if look.custom.is_some() { rgba.chunks_exact(4).all(|p| p[3] != 0) } else { full };
                (rgba, selected.bands, full)
            }
        };
        // With no format asked for, the mask decides it. Two bands carry theirs as a third,
        // so their picture is never transparent and cannot say.
        let asked = match look.asked {
            Some(asked) => asked,
            None => parse_ext_format(if full { "jpg" } else { "png" }, None, look.jpeg_quality)?,
        };
        // Two bands and a mask are three bands to GDAL, and JPEG has no such picture.
        if bands == 2 && matches!(asked.0, OutputFormat::Jpeg { .. }) {
            return Err(EngineError::BadRequest(format!(
                "Could not encode array of shape (2,{height},{width}) and of datatype `uint8` using JPEG driver"
            )));
        }
        let grey = bands == 1 && !look.mapped && look.custom.is_none();
        // An array of a picture: its bytes, band by band, and the mask as a band of its own.
        if let (OutputFormat::Npy, content_type) = asked {
            let mut rgba = rgba;
            if let Some(formula) = formula {
                formula.apply_buffer_in_place(&mut rgba, 4);
            }
            let shown = if grey { 1 } else { 3 };
            let mut values = Vec::with_capacity(shown * pixels);
            for band in 0..shown {
                values.extend(rgba.chunks_exact(4).map(|p| if p[3] == 0 { f32::NAN } else { f32::from(p[band]) }));
            }
            return Ok((npy_array(&values, shown, width, height, SampleType::Uint8, look.return_mask, None), content_type));
        }
        encode_picture(rgba, width, height, formula, Some(asked), look.jpeg_quality, look.return_mask, grey)
    }
}

/// A NumPy array of `bands` planes of `width` by `height`, as TiTiler writes one: in the
/// type `kind`, a value that is not there as `absent` (the no-data value the request
/// named) or zero, and after the bands, with `mask`, one more that holds the type's
/// largest number where any band has a value and its smallest elsewhere.
fn npy_array(
    values: &[f32],
    bands: usize,
    width: usize,
    height: usize,
    kind: SampleType,
    mask: bool,
    absent: Option<f32>,
) -> Bytes {
    cogzilla_kernel::encode_npy_typed(values, bands, height, width, kind, absent, mask)
}

/// Encode an RGBA buffer as a NumPy `|u1` array of shape `(C, H, W)` (or `(C + 1, H, W)` with mask),
/// where `C` is `1` when `grey` is true and `3` otherwise. Short buffers are zero-padded to `H * W`.
fn rgba_to_npy_u8(rgba: &[u8], width: usize, height: usize, grey: bool, return_mask: bool) -> Bytes {
    let pixels = width.saturating_mul(height);
    let shown = if grey { 1 } else { 3 };
    let planes = shown + usize::from(return_mask);
    let hdr = cogzilla_kernel::encode_npy_header("|u1", &[planes, height, width]);
    let body = planes.saturating_mul(pixels);
    let mut out = Vec::with_capacity(hdr.len().saturating_add(body));
    out.extend_from_slice(&hdr);
    for band in 0..shown {
        out.extend(
            rgba.chunks_exact(4)
                .map(|px| px[band])
                .chain(std::iter::repeat(0u8))
                .take(pixels),
        );
    }
    if return_mask {
        out.extend(
            rgba.chunks_exact(4)
                .map(|px| if px[3] == 0 { 0 } else { 255 })
                .chain(std::iter::repeat(0u8))
                .take(pixels),
        );
    }
    Bytes::from(out)
}

/// A colour map given in the request as JSON, read the way rio-tiler applies one.
struct MosaicColormap {
    map: cogzilla_kernel::CustomColormap,
    /// Classes for every byte, 0 to 255: rio-tiler then takes the value as a byte and
    /// looks that up, where it otherwise looks for the value itself.
    every_byte: bool,
}

/// The most intervals a colour map may have: each pixel is tried against them all.
const MAX_COLORMAP_INTERVALS: usize = 256;

impl MosaicColormap {
    /// `None` for a map with nothing in it, which TiTiler takes as no map at all.
    fn parse(json: &str) -> Result<Option<Self>, String> {
        let given: Value = serde_json::from_str(json).map_err(|e| format!("invalid colormap JSON: {e}"))?;
        let every_byte = match &given {
            Value::Object(classes) if classes.is_empty() => return Ok(None),
            Value::Array(spans) if spans.is_empty() => return Ok(None),
            Value::Array(spans) if spans.len() > MAX_COLORMAP_INTERVALS => {
                return Err(format!("a colormap takes at most {MAX_COLORMAP_INTERVALS} intervals"));
            }
            Value::Object(classes) => classes.len() == 256 && (0..256).all(|k| classes.contains_key(&k.to_string())),
            _ => false,
        };
        let map = cogzilla_kernel::CustomColormap::from_json_str(json, false)?;
        Ok(Some(Self { map, every_byte }))
    }

    /// The colour of a value, with the range `rescale` gave when it gave one. Nothing,
    /// which is transparent black, for a value the map does not name.
    fn colour(&self, v: f32, scaled: Option<(f32, f32)>) -> [u8; 4] {
        const NOTHING: [u8; 4] = [0; 4];
        match &self.map {
            cogzilla_kernel::CustomColormap::Discrete { lut, extra, .. } => {
                let key = match scaled {
                    Some((min, max)) => i64::from(rescale_byte(v, min, max)),
                    // The value cast to a byte, as NumPy casts: towards zero, and round the clock.
                    None if self.every_byte => (v.trunc() as i64).rem_euclid(256),
                    // A class is for the value that equals it, not for those near it.
                    None if v.fract() == 0.0 => v as i64,
                    None => return NOTHING,
                };
                match usize::try_from(key) {
                    Ok(byte) if byte < 256 => lut.rgba_u32[byte].to_le_bytes(),
                    _ => extra.get(&key).copied().unwrap_or(NOTHING),
                }
            }
            cogzilla_kernel::CustomColormap::Intervals { intervals: spans, .. } => {
                let x = scaled.map_or(v, |(min, max)| f32::from(rescale_byte(v, min, max)));
                // Where intervals overlap, the one given last is the one that shows.
                spans.iter().rev().find(|(low, high, _)| x >= *low && x < *high).map_or(NOTHING, |span| span.2)
            }
            // A ramp between stops is made only for `colormap_type=linear`, which these
            // routes do not read: `parse` above never makes one.
            cogzilla_kernel::CustomColormap::Linear { .. } => NOTHING,
        }
    }
}

/// A colour formula on one band under a colour map, for a picture the renderer draws: the
/// band is drawn in grey, the formula works on that byte, and the colour map is looked up
/// with what it leaves, which is rio-tiler's order.
struct MapAfterFormula {
    levels: [u8; 256],
    colormap: ColormapName,
}

impl MapAfterFormula {
    /// Takes the colour map out of `bands`, when they are one band under a map that is
    /// not grey. Otherwise nothing, and the formula is for the finished picture.
    fn take(formula: &cogzilla_kernel::ColorFormula, bands: &mut BandRequest) -> Option<Self> {
        let colormap = match bands {
            BandRequest::Single { colormap, .. }
            | BandRequest::Dem { colormap, .. }
            | BandRequest::Spectral1 { colormap, .. }
            | BandRequest::Spectral2 { colormap, .. }
            | BandRequest::Spectral3 { colormap, .. }
            | BandRequest::CustomExpr { colormap, .. } => colormap,
            _ => return None,
        };
        if *colormap == ColormapName::Grayscale {
            return None;
        }
        let taken = std::mem::replace(colormap, ColormapName::Grayscale);
        Some(Self { levels: formula_levels(formula), colormap: taken })
    }

    /// The grey picture as the coloured one.
    fn colour(&self, rgba: &mut [u8]) {
        let lut = self.colormap.lut();
        for px in rgba.chunks_exact_mut(4) {
            if px[3] != 0 {
                let level = self.levels[usize::from(px[0])];
                px[..3].copy_from_slice(&lut.rgb[usize::from(level)]);
            }
        }
    }
}

/// Whether the bands are one band's values shown in grey: what a colour formula works on
/// as a single byte. The renderer makes that byte in single precision, where rio-tiler's
/// is `linear_rescale`'s, so such a picture is painted here from the values.
fn one_grey_band(bands: &BandRequest) -> bool {
    matches!(
        bands,
        BandRequest::Single { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Dem { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Spectral1 { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Spectral2 { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Spectral3 { colormap: ColormapName::Grayscale, .. }
            | BandRequest::CustomExpr { colormap: ColormapName::Grayscale, .. }
    )
}

/// Colours for the bands a mosaic of planes leaves, as TiTiler encodes them: one band
/// through the colour map, three as red, green and blue, and two as red and green with
/// the mask for blue (what GDAL makes of two bands and a mask). `Err` is a number of bands
/// no image format holds.
fn paint_bands(
    selected: &Selected,
    pixels: usize,
    rescale: &[(f32, f32)],
    default: (f32, f32),
    colormap: ColormapName,
    custom: Option<&MosaicColormap>,
    before_map: Option<&cogzilla_kernel::ColorFormula>,
) -> Result<Vec<u8>, usize> {
    // One range for each band when there are as many of them; the first for all
    // otherwise, as TiTiler has it.
    let range = |band: usize| match rescale {
        [] => default,
        several if several.len() == selected.bands => several[band],
        several => several[0],
    };
    match selected.bands {
        1 => {
            let (min, max) = range(0);
            // A colour formula works on the band's byte, and the colour map is looked up
            // with what it leaves: rio-tiler's order.
            let levels = before_map.map(formula_levels);
            let Some(custom) = custom else {
                let Some(levels) = levels else {
                    return Ok(Paint::Gray { min, max, colormap }.rgba(selected, pixels));
                };
                let lut = colormap.lut();
                let mut out = vec![0u8; pixels * 4];
                for (px, v) in out.chunks_exact_mut(4).zip(&selected.values) {
                    if !v.is_nan() {
                        px[..3].copy_from_slice(&lut.rgb[usize::from(levels[usize::from(rescale_byte(*v, min, max))])]);
                        px[3] = 255;
                    }
                }
                return Ok(out);
            };
            // A colour map of the request's own: each value's class or interval, with
            // the transparency it names, and nothing where there is no value. With
            // `rescale` the map is over the byte the value becomes, the same byte a grey
            // picture would show; without, over the value itself.
            let scaled = (!rescale.is_empty()).then_some((min, max));
            let mut out = vec![0u8; pixels * 4];
            for (px, v) in out.chunks_exact_mut(4).zip(&selected.values) {
                if !v.is_nan() {
                    px.copy_from_slice(&match &levels {
                        // The map is over the byte, and the byte is the formula's: of the
                        // value over `rescale`, or over its type's range without one.
                        Some(levels) => custom.colour(f32::from(levels[usize::from(rescale_byte(*v, min, max))]), Some((0.0, 255.0))),
                        None => custom.colour(*v, scaled),
                    });
                }
            }
            Ok(out)
        }
        3 => {
            let (r, g, b) = (range(0), range(1), range(2));
            Ok(Paint::Rgb { min: [r.0, g.0, b.0], max: [r.1, g.1, b.1] }.rgba(selected, pixels))
        }
        2 => {
            let byte = |v: f32, (min, max): (f32, f32)| {
                if v.is_nan() { 0 } else { rescale_byte(v, min, max) }
            };
            let mut out = vec![255u8; pixels * 4];
            for (i, px) in out.chunks_exact_mut(4).enumerate() {
                let (a, b) = (selected.values[i], selected.values[pixels + i]);
                px[0] = byte(a, range(0));
                px[1] = byte(b, range(1));
                px[2] = if a.is_nan() && b.is_nan() { 0 } else { 255 };
            }
            Ok(out)
        }
        other => Err(other),
    }
}

/// Bring the assets and their options to what the tile and image renderer takes, and
/// give back the asset names. The renderer reads either the bands of one raster or the
/// first band of several, so it can honour `bidx` and band math on a single asset, and
/// `bidx=1` on each of several. Anything else is refused rather than drawn differently,
/// and does not come here: it is made from planes ([`wants_planes`]).
fn renderer_assets(pq: &mut ParsedQuery, query: &MosaicQuery) -> Result<Vec<String>, Resp> {
    let unsupported = |what: &str| {
        detail(StatusCode::BAD_REQUEST, format!("{what} is not supported by the renderer"))
    };
    // TiTiler 2 takes band indexes per asset only; a bare `bidx` is passed over.
    pq.bidx.clear();
    let names = expression_names(pq, query)?;
    pq.assets = names.clone();
    if names.len() > MAX_ASSETS_PER_SCENE {
        return Err(unsupported(&format!("more than {MAX_ASSETS_PER_SCENE} assets per item")));
    }
    if query.selection.is_some_and(Selection::by_last_band) {
        return Err(unsupported("pixel_selection by the last band"));
    }
    match query.assets.as_slice() {
        [one] => {
            if let Some(expression) = &one.expression {
                if pq.expr_raw.is_some() {
                    return Err(unsupported("an `expression` on top of an asset's own expression"));
                }
                if expression.contains(';') {
                    return Err(unsupported("an asset expression with several results"));
                }
                pq.expr_raw = Some(expression.clone());
            } else if let Some(indexes) = &one.indexes {
                if pq.expr_raw.is_some() {
                    return Err(unsupported("an `expression` over bands chosen with `bidx`"));
                }
                if !matches!(indexes.len(), 1 | 3) {
                    return Err(unsupported("`bidx` with a number of bands other than one or three"));
                }
                pq.bidx = indexes.clone();
            }
        }
        several => {
            let plain = |a: &AssetSpec| a.expression.is_none() && a.indexes.as_ref().is_none_or(|i| i == &[1]);
            if !several.iter().all(plain) {
                return Err(unsupported("options other than `bidx=1` when several assets are given"));
            }
        }
    }
    Ok(names)
}


/// `Ok(None)` is a tile past the edge of the grid. That is not an error in
/// titiler-pgstac: no item covers it, so the answer is empty.
fn parse_zxy(zs: &str, xs: &str, y: u32) -> Result<Option<TileCoord>, Resp> {
    let bad = |name: &str| invalid_param("path", name, "Input should be a valid integer");
    let z = zs.parse::<u32>().map_err(|_| bad("z"))?;
    let x = xs.parse::<u32>().map_err(|_| bad("x"))?;
    if z > 30 || (x as u64) >= (1u64 << z) || (y as u64) >= (1u64 << z) {
        return Ok(None);
    }
    Ok(Some(TileCoord { z: z as u8, x, y }))
}

#[derive(Clone, Copy)]
struct TmsTileAddr {
    matrix: &'static crate::tms::TileMatrixDef,
    x: u32,
    y: u32,
}

fn parse_tms_addr(
    tms_def: &'static crate::tms::TmsDef,
    zs: &str,
    xs: &str,
    y: u32,
) -> Result<Option<TmsTileAddr>, Resp> {
    let bad = |name: &str| invalid_param("path", name, "Input should be a valid integer");
    let _z = zs.parse::<i32>().map_err(|_| bad("z"))?;
    let x = xs.parse::<u32>().map_err(|_| bad("x"))?;
    let Some(m) = tms_def.matrix_by_id(zs) else {
        return Ok(None);
    };
    if (x as u64) >= m.matrix_width || (y as u64) >= m.matrix_height {
        return Ok(None);
    }
    Ok(Some(TmsTileAddr { matrix: m, x, y }))
}

/// WGS84 GeoJSON polygon covering a tile in any built-in TileMatrixSet.
fn tms_tile_polygon(tms_def: &crate::tms::TmsDef, addr: TmsTileAddr) -> Option<Value> {
    let [min_x, min_y, max_x, max_y] =
        tms_def.tile_bounds_for_matrix(addr.matrix, addr.x, addr.y, 0.0, 256)?;
    if tms_def.epsg == 4326 {
        return Some(bbox_polygon([
            min_x.clamp(-180.0, 180.0),
            min_y.clamp(-90.0, 90.0),
            max_x.clamp(-180.0, 180.0),
            max_y.clamp(-90.0, 90.0),
        ]));
    }
    let dst_crs = tms_def.crs_model().ok()?;
    let (mut w, mut s, mut e, mut n) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    const STEPS: usize = 8;
    for i in 0..=STEPS {
        let f = i as f64 / STEPS as f64;
        let x = min_x + f * (max_x - min_x);
        let y = min_y + f * (max_y - min_y);
        for (px, py) in [(x, min_y), (x, max_y), (min_x, y), (max_x, y)] {
            let (lon, lat) = dst_crs.to_lonlat_deg(px, py);
            if lon.is_finite() && lat.is_finite() {
                w = w.min(lon);
                e = e.max(lon);
                s = s.min(lat);
                n = n.max(lat);
            }
        }
    }
    // Polar stereographic / azimuthal tiles containing a pole in their interior must expand to
    // the full polar cap so pgSTAC's WGS84 spatial query does not miss polar items.
    for &pole_lat in &[90.0_f64, -90.0_f64] {
        let (px, py) = dst_crs.from_lonlat_deg(0.0, pole_lat);
        if px.is_finite() && py.is_finite() && px >= min_x && px <= max_x && py >= min_y && py <= max_y {
            w = -180.0;
            e = 180.0;
            if pole_lat > 0.0 {
                n = 90.0;
            } else {
                s = -90.0;
            }
        }
    }
    if !w.is_finite() || !s.is_finite() || !e.is_finite() || !n.is_finite() {
        return None;
    }
    Some(bbox_polygon([
        w.clamp(-180.0, 180.0),
        s.clamp(-90.0, 90.0),
        e.clamp(-180.0, 180.0),
        n.clamp(-90.0, 90.0),
    ]))
}

async fn tile_assets_geom(
    pg: &PgStac,
    id: &str,
    geom: &Value,
    mercator_rows: bool,
    pairs: &[(String, String)],
) -> Resp {
    let limits = match parse_limits(pairs) {
        Ok(l) => l,
        Err(r) => return r,
    };
    // `tilesize` matters only to the lean lookup, which samples the tile as its picture would.
    let side = query_value(pairs, "tilesize").and_then(|v| v.parse::<u32>().ok()).unwrap_or(256);
    let grid = SampleGrid::for_image(side, side, mercator_rows);
    match listed_items(pg, id, geom, limits, pairs, grid).await {
        Ok(items) => json_resp(
            StatusCode::OK,
            &Value::Array(items.iter().map(MosaicItem::to_json).collect()),
        ),
        Err(r) => r,
    }
}

/// TiTiler's answer for an asset the item does not have: its assets, in jsonb key order.
/// Where an item's assets are read from, in the order they are named; a name the item
/// has not got fails the request before anything is read.
fn asset_hrefs<'a>(item: &MosaicItem, names: impl IntoIterator<Item = &'a str>) -> Result<Vec<String>, Resp> {
    names
        .into_iter()
        .map(|name| item.asset_href(name).map(|href| href.to_string()).ok_or_else(|| unknown_asset(name, item)))
        .collect()
}

/// An item as titiler-pgstac names it: `{collection}/{id}`.
fn item_name(item: &MosaicItem) -> String {
    match &item.collection {
        Some(c) => format!("{c}/{}", item.id),
        None => item.id.clone(),
    }
}

fn unknown_asset(asset: &str, item: &MosaicItem) -> Resp {
    let mut names: Vec<String> = item.assets.keys().cloned().collect();
    jsonb_key_order(&mut names);
    let known: Vec<String> = names.iter().map(|k| format!("'{k}'")).collect();
    detail(
        StatusCode::NOT_FOUND,
        format!("'{asset}' is not valid, should be one of [{}]", known.join(", ")),
    )
}

/// The largest side of a `bbox` or `feature` image.
const MAX_PART_SIDE: u32 = 4096;

/// The `{lon},{lat}` of a path. Without a comma it is no such path; with one, what is
/// not two numbers (spaces around them allowed) is a 422, as in TiTiler.
fn point_coords(seg: &str) -> Result<(f64, f64), Resp> {
    let seg = super::percent_decode(seg);
    let (x, y) = seg.split_once(',').ok_or_else(|| detail(StatusCode::NOT_FOUND, "Not Found"))?;
    // Python's `float` reads a number with spaces around it.
    match (x.trim().parse::<f64>(), y.trim().parse::<f64>()) {
        (Ok(x), Ok(y)) if x.is_finite() && y.is_finite() => Ok((x, y)),
        _ => Err(invalid_param("path", "lon,lat", "Input should be two numbers, as lon,lat")),
    }
}

/// The `{width}x{height}[.{format}]` of a path. A zero is a side left to follow the other,
/// or the data's own resolution when both are, as TiTiler has it; `max_size` in the query
/// then bounds it, and only then, as in rio-tiler.
fn path_size<'a>(dims: &'a str, pairs: &[(String, String)]) -> Result<(PartSize, Option<&'a str>), Resp> {
    let not_found = || detail(StatusCode::NOT_FOUND, "Not Found");
    let (sides, ext) = match dims.split_once('.') {
        Some((sides, ext)) => (sides, Some(ext)),
        None => (dims, None),
    };
    let (w, h) = sides.split_once('x').ok_or_else(not_found)?;
    let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) else {
        return Err(invalid_param("path", "width", "Input should be a valid integer"));
    };
    if w > 4096 || h > 4096 {
        return Err(not_found());
    }
    let size = match (w, h) {
        (0, 0) => {
            let max_size = query_value(pairs, "max_size").and_then(|v| v.parse::<u32>().ok()).filter(|n| *n > 0);
            PartSize::Query { max_size, width: None, height: None }
        }
        (w, 0) => PartSize::Query { max_size: None, width: Some(w), height: None },
        (0, h) => PartSize::Query { max_size: None, width: None, height: Some(h) },
        (w, h) => PartSize::Exact(w, h),
    };
    Ok((size, ext))
}

/// The answer to a format these routes have no name for: a 422, as FastAPI gives for a
/// path that does not fit.
fn format_error(ext: Option<&str>) -> Option<Resp> {
    let known = |e: &str| matches!(e, "png" | "npy" | "tif" | "tiff" | "jpeg" | "jpg" | "jp2" | "webp" | "pngraw" | "bin" | "raw" | "f32");
    ext.filter(|e| !e.is_empty() && !known(e))
        .map(|_| invalid_param("path", "format", "Input should be 'png', 'npy', 'tif', 'tiff', 'jpeg', 'jpg', 'jp2', 'webp' or 'pngraw'"))
}

/// The size asked for a `bbox` or `feature` image.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PartSize {
    /// `{width}x{height}` in the path.
    Exact(u32, u32),
    /// From the query string; whatever is missing follows the data's own resolution.
    Query { max_size: Option<u32>, width: Option<u32>, height: Option<u32> },
}

fn part_size_from_query(pairs: &[(String, String)]) -> Result<PartSize, Resp> {
    let int = |name: &str| -> Result<Option<u32>, Resp> {
        match query_value(pairs, name) {
            None => Ok(None),
            Some(v) => v
                .parse::<u32>()
                .ok()
                .filter(|n| *n > 0)
                .map(Some)
                .ok_or_else(|| invalid_param("query", name, "Input should be a valid integer")),
        }
    };
    let (width, height) = (int("width")?, int("height")?);
    // As in rio-tiler, `max_size` is ignored once a width or a height is given.
    let max_size = if width.is_some() || height.is_some() { None } else { int("max_size")? };
    Ok(PartSize::Query { max_size, width, height })
}

/// Output size of a part of a raster, by rio-tiler's rules. `res` is the raster's
/// resolution in the output coordinate system.
fn part_dims(size: PartSize, bbox: [f64; 4], res: f64) -> (u64, u64) {
    let native_w = ((bbox[2] - bbox[0]) / res).round().max(1.0);
    let native_h = ((bbox[3] - bbox[1]) / res).round().max(1.0);
    let ratio = native_h / native_w;
    let (w, h) = match size {
        PartSize::Exact(w, h) => (f64::from(w), f64::from(h)),
        PartSize::Query { width: Some(w), height: Some(h), .. } => (f64::from(w), f64::from(h)),
        PartSize::Query { width: Some(w), .. } => (f64::from(w), (f64::from(w) * ratio).ceil()),
        PartSize::Query { height: Some(h), .. } => ((f64::from(h) / ratio).ceil(), f64::from(h)),
        PartSize::Query { max_size: Some(m), .. } if native_w.max(native_h) >= f64::from(m) => {
            if ratio > 1.0 {
                ((f64::from(m) / ratio).ceil(), f64::from(m))
            } else {
                (f64::from(m), (f64::from(m) * ratio).ceil())
            }
        }
        PartSize::Query { .. } => (native_w, native_h),
    };
    (w.max(1.0) as u64, h.max(1.0) as u64)
}

/// A raster's resolution on a grid in degrees, or in Web Mercator metres, as GDAL's warp
/// works it out: the raster's diagonal there over its diagonal in pixels.
fn warped_resolution(crs: &CrsModel, cog: &cogzilla_io::OpenedCog, mercator: bool) -> Option<f64> {
    let full = cog.index.full_res();
    let (w, h) = (f64::from(full.width), f64::from(full.height));
    let corner = |col: f64, row: f64| {
        let (x, y) = full.pixel_to_crs.forward(col, row);
        let (lon, lat) = crs.to_lonlat_deg(x, y);
        if mercator { lonlat_deg_to_mercator(lon, lat) } else { (lon, lat) }
    };
    let (ul, lr) = (corner(0.0, 0.0), corner(w, h));
    let res = (lr.0 - ul.0).hypot(lr.1 - ul.1) / w.hypot(h);
    (res.is_finite() && res > 0.0).then_some(res)
}

/// `GET {base}/bbox/...` and `POST {base}/feature...`: one image of the mosaic over a box
/// in WGS84, optionally clipped to a polygon. Items are drawn in the search's order, each
/// filling the pixels the ones before it left empty.
#[allow(clippy::too_many_arguments)]
async fn part_image(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    hash: &str,
    geometry: &Value,
    bbox: [f64; 4],
    mask: Option<&PolygonMask>,
    size: PartSize,
    ext: Option<&str>,
    pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    part_image_of(ctx, pg, hash, geometry, None, bbox, mask, size, ext, pq, pairs).await
}

/// [`part_image`], with the items looked up under `searched` when it is given and not
/// under the box: a buffered tile is drawn past its bounds from the items of the tile.
#[allow(clippy::too_many_arguments)]
async fn part_image_of(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    hash: &str,
    geometry: &Value,
    searched: Option<&Value>,
    bbox: [f64; 4],
    mask: Option<&PolygonMask>,
    size: PartSize,
    ext: Option<&str>,
    mut pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    if let Some(resp) = format_error(ext) {
        return resp;
    }
    let index = normalized_index_as_band_math(&mut pq, ext);
    // A GeoTIFF is the array with a place on the ground: made as the array, wrapped at the end.
    // (With `algorithm` no array is written on these routes yet: there it stays the picture.)
    let tiff = matches!(ext.or(pq.format_param.as_deref()), Some("tif" | "tiff")) && pq.algorithm.is_none();
    let ext = if tiff { Some("npy") } else { ext };
    // `width` and `height` are read here; the shared parser does not know them.
    pq.errors.retain(|e| !own_size_error(e));
    let query = match check_mosaic_query(&pq, pairs) {
        Ok(q) => q,
        Err(r) => return r,
    };
    // An image is made from planes, each raster read the way GDAL's nearest-neighbour warp
    // reads it, which is what TiTiler's pixels are. The shared renderer draws the rest:
    // the other resampling methods and the algorithms, where it can.
    let nearest = query.resampling == ResampleMode::Nearest && query.reproject == ResampleMode::Nearest;
    // The renderer's band math knows names and functions that band math over planes does
    // not (`ndvi`, `^`, `max(a, b)`); what only it can read, it draws.
    let planes_can = planes_read_all(&pq, &query);
    // A colour map given as JSON is applied to planes; the renderer on this road has none.
    let custom = match pq.custom_colormap_json.as_deref().map(MosaicColormap::parse).transpose() {
        Ok(map) => map.flatten(),
        Err(e) => return detail(StatusCode::BAD_REQUEST, e),
    };
    // So is a `nodata` of the request's own: the renderer on this road reads the raster's.
    // And a NumPy array, which is written from planes in the data's own type.
    let array = matches!(ext.or(pq.format_param.as_deref()), Some("npy"));
    let planes_needed = wants_planes(&pq, &query) || custom.is_some() || pq.nodata.is_some() || (array && pq.algorithm.is_none());
    let planes_preferred = nearest && pq.algorithm.is_none() && planes_can;
    // The request's own band math, before an asset's is put in its place for the renderer.
    let expression = pq.expr_raw.clone();
    let names = if planes_needed { expression_names(&pq, &query) } else { renderer_assets(&mut pq, &query) };
    let names = match names {
        Ok(names) => names,
        Err(r) => return r,
    };
    // The box is given in `coord_crs` and drawn in `dst_crs`, which is `coord_crs` unless
    // given: degrees or Web Mercator metres, either way round. A posted shape is in degrees.
    let given_in_metres = pq.coord_crs_3857;
    if !given_in_metres && let Err(r) = check_coord_crs(&pq, pairs) {
        return r;
    }
    // `dst_crs` is read as `coord_crs` is, in any of the spellings TiTiler takes.
    let mercator = match drawn_in_mercator(pairs, given_in_metres) {
        Ok(mercator) => mercator,
        Err(r) => return r,
    };
    // Web Mercator ends at 85.05 degrees: a box in degrees that goes past it has no
    // corner there, and is refused sooner than drawn cut short under its own bounds.
    const MERCATOR_EDGE: f64 = 85.051_128_78;
    if mercator && !given_in_metres && (bbox[1] < -MERCATOR_EDGE || bbox[3] > MERCATOR_EDGE) {
        return detail(StatusCode::BAD_REQUEST, "the box reaches past the edge of Web Mercator, 85.05 degrees north or south");
    }
    if given_in_metres && mask.is_some() {
        return detail(StatusCode::BAD_REQUEST, "coord_crs 'epsg:3857' is not supported for a posted shape (use epsg:4326)");
    }
    let corners = |bbox: [f64; 4], to: fn(f64, f64) -> (f64, f64)| {
        let ((w, s), (e, n)) = (to(bbox[0], bbox[1]), to(bbox[2], bbox[3]));
        [w, s, e, n]
    };
    let in_degrees = if given_in_metres { corners(bbox, mercator_to_lonlat_deg) } else { bbox };
    // The box as the picture's grid has it, and as the search is asked about it.
    let bbox = match (mercator, given_in_metres) {
        (true, true) => bbox,
        (true, false) => corners(bbox, lonlat_deg_to_mercator),
        (false, _) => in_degrees,
    };
    if bbox.iter().any(|c| !c.is_finite()) || bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
        // A posted point covers no pixel: nothing to draw, as titiler-pgstac answers it.
        // So does a line along a row or a column of them.
        if mask.is_some() && bbox.iter().all(|c| c.is_finite()) && (bbox[0] == bbox[2] || bbox[1] == bbox[3]) {
            return no_content();
        }
        return detail(StatusCode::BAD_REQUEST, "the box has no area in the coordinate system asked for");
    }
    let own_geometry = given_in_metres.then(|| bbox_polygon(in_degrees));
    let geometry = searched.or(own_geometry.as_ref()).unwrap_or(geometry);
    // The renderer draws a box in Web Mercator as well as in degrees, but cuts a posted
    // shape out of degrees only.
    if mercator && mask.is_some() && !(planes_can && pq.algorithm.is_none()) {
        return detail(
            StatusCode::BAD_REQUEST,
            "dst_crs 'epsg:3857' is not drawn for a posted shape with `algorithm`, or with band math only the renderer reads",
        );
    }
    // The extension in the path, then `format=`; with neither, the pixels decide.
    let explicit = ext.map(str::to_string).or_else(|| pq.format_param.clone());
    let format = match explicit.as_deref() {
        None => None,
        Some(e) => match parse_ext_format(e, pq.webp_quality, pq.jpeg_quality) {
            Ok(f) => Some(f),
            Err(e) => return engine_error(&e),
        },
    };
    // A NumPy array is written from planes, in the data's own type and with its mask; the
    // renderer's is float32 without one, and bare bytes say nothing of either.
    if matches!(format, Some((OutputFormat::RawF32, _))) {
        return detail(StatusCode::BAD_REQUEST, "bare arrays are not written on pgSTAC bbox and feature routes; ask for `.npy`");
    }
    // A JPEG has no mask to carry the polygon, and TiTiler then writes the whole box as it
    // was read. With `rescale` it does not: rescaling leaves zero wherever the mask is.
    let whole_box = matches!(format, Some((OutputFormat::Jpeg { .. }, _))) && pq.rescale.is_empty();
    let mask = if whole_box { None } else { mask };

    // The lean lookup samples the box as a grid in degrees. A grid in metres is another
    // set of pixel centres, so there pgSTAC's own search lists the items.
    // Nor does it list for a `nodata` of the request's own: it passes over an item whose
    // footprint adds nothing, and such an item's raster may still hold values there.
    let grid = if mercator || pq.nodata.is_some() { None } else { part_grid(size, bbox) };
    let items = match pixel_items(ctx, pg, geometry, hash, query.limits, &names, grid, query.lookup).await {
        Ok(items) => items,
        Err(e) => return db_error(&e),
    };
    if items.is_empty() {
        return no_content();
    }
    let scenes: Result<Vec<Vec<String>>, Resp> =
        items.iter().take(MAX_SCENES_PER_TILE).map(|item| asset_hrefs(item, names.iter().map(String::as_str))).collect();
    let scenes = match scenes {
        Ok(scenes) => scenes,
        Err(resp) => return resp,
    };

    let engine = ctx.engine;
    let rendered: Result<(Bytes, &'static str), EngineError> = async {
        let formula = pq
            .color_formula
            .as_deref()
            .map(cogzilla_kernel::ColorFormula::parse)
            .transpose()
            .map_err(EngineError::BadRequest)?;
        // Assets that do not lie on one grid cannot be read as the bands of one image.
        let mut first = Vec::with_capacity(scenes[0].len());
        for href in &scenes[0] {
            first.push(engine.store.open_uri(href).await.map(|c| engine.store.with_nodata(&c, pq.nodata))?);
        }
        // A raster with an alpha band is the renderer's too: it reads the alpha as the mask.
        // So is one of two bands, or four and more, asked for plainly: it draws the first,
        // or the first three, where planes would stack them all and TiTiler refuse most.
        let alpha = first.iter().any(|cog| !cog.index.full_res().extra_samples.is_empty())
            || (first.len() == 1 && !matches!(first[0].index.full_res().samples_per_pixel, 1 | 3));
        // Those the renderer draws; in Web Mercator, only where no posted shape cuts the box.
        if mercator && alpha && !planes_needed && mask.is_some() {
            return Err(EngineError::BadRequest(
                "dst_crs 'epsg:3857' is not supported for a posted shape over a raster with an alpha band, or of two bands or more than three asked for whole".into(),
            ));
        }
        let renderer_in_mercator = mercator && mask.is_none() && !planes_needed && (pq.algorithm.is_some() || !planes_can || alpha);
        if (mercator && !renderer_in_mercator) || planes_needed || !TileEngine::on_one_grid(&first) || (planes_preferred && !alpha) {
            if pq.algorithm.is_some() {
                return Err(EngineError::BadRequest(
                    "`algorithm` is not supported with these assets and options yet".into(),
                ));
            }
            // The first item sets the size, by the finest of its rasters: rio-tiler reads
            // each asset at its own and brings them all to the largest.
            let mut res = f64::INFINITY;
            for cog in &first {
                let crs = engine.get_or_build_crs(cog)?;
                let own = warped_resolution(&crs, cog, mercator);
                res = res.min(own.ok_or_else(|| EngineError::BadRequest("cannot place the first item in WGS84".into()))?);
            }
            let (w, h) = part_dims(size, bbox, res);
            if w > u64::from(MAX_PART_SIDE) || h > u64::from(MAX_PART_SIDE) {
                return Err(EngineError::BadRequest(format!(
                    "the image would be {w}x{h} pixels; the limit is {MAX_PART_SIDE} per side (pass max_size, or width and height)"
                )));
            }
            let job = PlanesTile {
                assets: query.assets.clone(),
                as_band: query.asset_as_band,
                expression: expression.clone(),
                selection: query.selection.unwrap_or(Selection::First),
                resampling: query.resampling,
                reproject: query.reproject,
                nodata: pq.nodata,
                array,
                window: Window::Part { bbox, mercator, width: w as u32, height: h as u32, size, mask: mask.map(|m| (Arc::new(m.clone()), tokio::sync::OnceCell::new())), unscale: pq.unscale },
                pool: true,
                index,
            };
            let look = Look {
                rescale: &pq.rescale,
                colormap: pq.colormap,
                mapped: pq.colormap_explicit,
                custom: custom.as_ref(),
                formula: formula.as_ref(),
                asked: format.map(|(f, ct)| (f, if explicit.as_deref() == Some("jpg") { "image/jpg" } else { ct })),
                jpeg_quality: pq.jpeg_quality,
                return_mask: pq.return_mask,
            };
            let made = job.picture(engine, &scenes, w as usize, h as usize, &look).await?;
            return Ok(made);
        }
        let mut canvas: Vec<u8> = Vec::new();
        let (mut width, mut height) = (0usize, 0usize);
        // With a pixel selection other than `first`, or raw `.npy` under `algorithm`: every scene's values,
        // optional painter (not needed for raw `.npy`), and expected band count.
        let mut selecting: Option<(Fold, Option<Paint>, usize)> = None;
        let visual_array = pq.colormap_explicit || formula.is_some() || !pq.rescale.is_empty();
        // Degrees, or Web Mercator metres for a box drawn there.
        let drawn_in = if mercator { CrsModel::Epsg3857 } else { CrsModel::Geographic };
        // One band under a colour map, with a colour formula: drawn grey, coloured at the end.
        let mut map_after: Option<MapAfterFormula> = None;
        let mut one_grey = false;
        for (n, scene) in scenes.iter().enumerate() {
            let mut cogs = Vec::with_capacity(scene.len());
            for href in scene {
                cogs.push(engine.store.open_uri(href).await.map(|c| engine.store.with_nodata(&c, pq.nodata))?);
            }
            let crs = engine.get_or_build_crs(&cogs[0])?;
            let full = cogs[0].index.full_res();
            if query.asset_as_band && full.samples_per_pixel > 1 {
                return Err(EngineError::BadRequest("Can't use `asset_as_band` for multibands asset".into()));
            }
            if n == 0 {
                // The first item sets the size; the others are drawn onto its grid.
                let res = warped_resolution(&crs, &cogs[0], mercator).ok_or_else(|| {
                    EngineError::BadRequest("cannot place the first item in WGS84".into())
                })?;
                let (w, h) = part_dims(size, bbox, res);
                if w > u64::from(MAX_PART_SIDE) || h > u64::from(MAX_PART_SIDE) {
                    return Err(EngineError::BadRequest(format!(
                        "the image would be {w}x{h} pixels; the limit is {MAX_PART_SIDE} per side (pass max_size, or width and height)"
                    )));
                }
                (width, height) = (w as usize, h as usize);
                canvas = vec![0u8; width * height * 4];
            }
            let default_bands =
                if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
            // The renderer's bands are one or three: ranges that are not one each are the
            // first for all, as TiTiler has it (the shared routes refuse two for one band).
            if pq.rescale.len() > 1 && pq.rescale.len() != 3 {
                pq.rescale.truncate(1);
            }
            let mut bands = build_band_request(&pq, default_bands, full.sample_type)?;
            // Every scene is drawn the same way, and the first says how the picture is finished.
            let taken = match &formula {
                Some(formula) if pq.colormap_explicit => MapAfterFormula::take(formula, &mut bands),
                _ => None,
            };
            if n == 0 {
                map_after = taken;
                one_grey = !pq.colormap_explicit
                    && (one_grey_band(&bands) || query.selection == Some(Selection::Count))
                    && !super::has_embedded_palette(&pq, &cogs[0], &bands);
            }
            // A formula on one band works on the band's byte, which is made here from the
            // values: the first there is, as a mosaic is.
            let exact = formula.is_some() && one_grey_band(&bands) && (map_after.is_some() || one_grey);
            // As on tiles: `resampling` for a source already in the output's projection,
            // `reproject` for one that has to be warped into it.
            let resample = match (crs, mercator) {
                (CrsModel::Geographic, false) | (CrsModel::Epsg3857, true) => query.resampling,
                _ => query.reproject,
            };
            let raw_npy = array && !visual_array && !matches!(bands, BandRequest::Terrain { .. });
            if let Some(method) = query.selection.or((exact || raw_npy).then_some(Selection::First)) {
                // The scene's values rather than its colours; only `first` stops early.
                let count = TileEngine::value_bands(&bands);
                if n == 0 {
                    let paint = if raw_npy {
                        None
                    } else {
                        Some(Paint::of(&bands).ok_or_else(|| {
                            EngineError::BadRequest("pixel_selection other than 'first' cannot be used with a terrain encoding".into())
                        })?)
                    };
                    let integer = super::reads_whole_numbers(pq.unscale, &bands, full.sample_type);
                    selecting = Some((fold_within_limits(method, scenes.len(), count, width * height, integer, false)?, paint, count));
                }
                let values = render_crs_bbox(
                    engine,
                    &cogs,
                    bbox,
                    &drawn_in,
                    width as u32,
                    height as u32,
                    mask,
                    &bands,
                    resample,
                    OutputFormat::RawF32,
                    None,
                    None,
                    None,
                    true,
                    pq.unscale,
                )
                .await?;
                if let Some((fold, _, _)) = &mut selecting
                    && values.len() == count * width * height * 4
                {
                    fold.add_le_bytes(&values);
                    if fold.is_done() {
                        break;
                    }
                }
                continue;
            }
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let rgba = render_crs_bbox(
                engine,
                &cogs,
                bbox,
                &drawn_in,
                width as u32,
                height as u32,
                mask,
                &bands,
                resample,
                OutputFormat::RawRgba,
                None,
                None,
                None,
                true,
                pq.unscale,
            )
            .await?;
            if rgba.len() != canvas.len() {
                return Err(EngineError::BadRequest("unexpected image size from the renderer".into()));
            }
            let mut empty = 0usize;
            for (dst, src) in canvas.chunks_exact_mut(4).zip(rgba.chunks_exact(4)) {
                if dst[3] == 0 {
                    if src[3] != 0 {
                        dst.copy_from_slice(src);
                    } else {
                        empty += 1;
                    }
                }
            }
            if empty == 0 {
                break;
            }
        }
        if let Some((fold, paint, count)) = selecting {
            if let Some(selected) = fold.finish() {
                if array && !visual_array {
                    let npy = npy_array(
                        &selected.values,
                        selected.bands,
                        width,
                        height,
                        SampleType::Float32,
                        pq.return_mask,
                        None,
                    );
                    return Ok((npy, "application/x-binary"));
                }
                if let Some(paint) = paint {
                    canvas = paint.rgba(&selected, width * height);
                }
            } else if array && !visual_array {
                let empty = vec![f32::NAN; count.saturating_mul(width).saturating_mul(height)];
                return Ok((
                    npy_array(&empty, count, width, height, SampleType::Float32, pq.return_mask, None),
                    "application/x-binary",
                ));
            }
        }
        match (&map_after, formula) {
            (Some(map_after), _) => map_after.colour(&mut canvas),
            (None, Some(formula)) => {
                // One band is the formula's red, and the picture is of that band alone.
                let grey = one_grey && canvas.chunks_exact(4).all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
                formula.apply_buffer_in_place(&mut canvas, 4);
                if grey {
                    canvas.chunks_exact_mut(4).for_each(|p| (p[1], p[2]) = (p[0], p[0]));
                }
            }
            (None, None) => {}
        }
        // One band, or a count, is a grey picture as on tiles: one channel, or two with the mask.
        let grey = one_grey && canvas.chunks_exact(4).all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
        if array {
            return Ok((rgba_to_npy_u8(&canvas, width, height, grey, pq.return_mask), "application/x-binary"));
        }
        let (format, content_type) = match format {
            Some((f, ct)) => (f, if explicit.as_deref() == Some("jpg") { "image/jpg" } else { ct }),
            None => {
                let opaque = canvas.chunks_exact(4).all(|p| p[3] == 255);
                parse_ext_format(if opaque { "jpg" } else { "png" }, None, pq.jpeg_quality)?
            }
        };
        let made = if grey {
            encode_picture(canvas, width, height, None, Some((format, content_type)), pq.jpeg_quality, pq.return_mask, true)?
        } else {
            (encode_rgba_output_colour(width, height, &canvas, format, pq.return_mask), content_type)
        };
        // The renderer's picture under `algorithm`, asked for as a GeoTIFF: its RGBA, placed.
        if made.1 == super::GEOTIFF_CONTENT_TYPE && made.0.len() == width * height * 4 {
            let epsg = if mercator { 3857 } else { 4326 };
            let body = crate::geotiff::encode_geotiff_rgba(width as u32, height as u32, &made.0, bbox, epsg, pq.return_mask);
            return Ok((body, made.1));
        }
        Ok(made)
    }
    .await;
    match rendered {
        Ok(made) => {
            let epsg = if mercator { 3857 } else { 4326 };
            let (body, content_type) = as_geotiff(made, tiff, bbox, epsg, pq.return_mask);
            let mut resp = Response::new(Full::new(body));
            resp.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
            resp.headers_mut().insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
            // The system drawn in is `dst_crs`, or `coord_crs` without one.
            let spelled = query_value(pairs, "dst_crs").or_else(|| query_value(pairs, "coord_crs"));
            placed_as(resp, bbox, if mercator { 3857 } else { 4326 }, spelled)
        }
        Err(e) => engine_error(&e),
    }
}

/// Whether `dst_crs` asks for a grid in Web Mercator metres and not degrees, in any of the
/// spellings the shared parser takes; `unset` when it is not given.
fn drawn_in_mercator(pairs: &[(String, String)], unset: bool) -> Result<bool, Resp> {
    match query_value(pairs, "dst_crs").map(|dst| (dst, super::parse_coord_crs_with_epsg(dst))) {
        None => Ok(unset),
        Some((_, Ok((_, 4326)))) => Ok(false),
        Some((_, Ok((_, 3857)))) => Ok(true),
        Some((dst, _)) => Err(detail(
            StatusCode::BAD_REQUEST,
            format!("dst_crs '{dst}' is not supported on pgSTAC mosaic routes (use epsg:4326 or epsg:3857)"),
        )),
    }
}

/// A feature whose geometry is a GeometryCollection of polygons alone, as the one
/// MultiPolygon the mask reads: GDAL burns every member, which is their union. Read this
/// way a collection is bounded by the mask's limits on rings and vertices, as any shape is,
/// and not by its limits on members (64) and nesting (8), which titiler-pgstac has not got.
/// Any other feature, a collection with a line or a point in it among them, is returned
/// as it is for the mask's own parser.
fn polygons_of_collection(feature: &Value) -> std::borrow::Cow<'_, Value> {
    fn gather<'a>(geometry: &'a Value, into: &mut Vec<&'a Value>, depth: usize) -> bool {
        match geometry.get("type").and_then(Value::as_str) {
            Some("Polygon") => {
                into.push(&geometry["coordinates"]);
                true
            }
            Some("MultiPolygon") => geometry["coordinates"].as_array().is_some_and(|polygons| {
                into.extend(polygons);
                true
            }),
            Some("GeometryCollection") if depth < 16 => geometry["geometries"]
                .as_array()
                .is_some_and(|members| members.iter().all(|member| gather(member, into, depth + 1))),
            _ => false,
        }
    }
    let geometry = &feature["geometry"];
    let mut polygons = Vec::new();
    if geometry.get("type").and_then(Value::as_str) == Some("GeometryCollection")
        && gather(geometry, &mut polygons, 0)
        && !polygons.is_empty()
    {
        return std::borrow::Cow::Owned(json!({ "type": "MultiPolygon", "coordinates": polygons }));
    }
    std::borrow::Cow::Borrowed(feature)
}

fn bbox_polygon([w, s, e, n]: [f64; 4]) -> Value {
    json!({ "type": "Polygon", "coordinates": [[[w, s], [w, n], [e, n], [e, s], [w, s]]] })
}

/// `POST {base}/feature[.{format}]` and `POST {base}/feature/{width}x{height}.{format}`.
#[allow(clippy::too_many_arguments)]
async fn feature_image(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    hash: &str,
    rest: &[&str],
    body: Option<&Bytes>,
    pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    let (size, ext) = match rest {
        ["feature"] => (part_size_from_query(pairs), None),
        [one] => (part_size_from_query(pairs), one.strip_prefix("feature.")),
        ["feature", dims] => match path_size(dims, pairs) {
            Ok((size, ext)) => (Ok(size), ext),
            Err(resp) => return resp,
        },
        _ => return detail(StatusCode::NOT_FOUND, "Not Found"),
    };
    let size = match size {
        Ok(s) => s,
        Err(r) => return r,
    };
    let not_a_feature = || invalid_param("body", "body", "Input should be a valid GeoJSON Feature");
    let Some(geojson) = body.and_then(|b| serde_json::from_slice::<Value>(b).ok()) else {
        return not_a_feature();
    };
    // As in TiTiler the body is a Feature; a bare geometry is not taken.
    if geojson.get("type").and_then(Value::as_str) != Some("Feature") || !geojson["geometry"].is_object() {
        return not_a_feature();
    }
    let (mask, bbox) = match PolygonMask::from_geojson(&polygons_of_collection(&geojson)) {
        Ok(m) => m,
        Err(e) => return engine_error(&e),
    };
    // As in titiler-pgstac, the items are those of the feature's bounding box.
    part_image(ctx, pg, hash, &bbox_polygon(bbox), bbox, Some(&mask), size, ext, pq, pairs).await
}

/// GDAL's default grid for a raster warped to WGS84: where it starts and its pixel size.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Wgs84Grid {
    west: f64,
    north: f64,
    res: f64,
}

/// The corner of the raster's extent in WGS84, from points along each edge as GDAL samples
/// them (one every fifty pixels, between twenty and a hundred steps), and the resolution
/// of [`wgs84_resolution`]. With `mercator` the same for a raster warped to Web Mercator:
/// the corner and the pixel size are then in metres.
fn warped_grid(crs: &CrsModel, cog: &cogzilla_io::OpenedCog, mercator: bool) -> Option<Wgs84Grid> {
    let res = warped_resolution(crs, cog, mercator)?;
    let full = cog.index.full_res();
    let (w, h) = (f64::from(full.width), f64::from(full.height));
    let (mut west, mut north) = (f64::INFINITY, f64::NEG_INFINITY);
    let steps = ((w.min(h) / 50.0 + 0.5) as u32).clamp(20, 100);
    for i in 0..=steps {
        let f = f64::from(i) / f64::from(steps);
        for (col, row) in [(f * w, 0.0), (f * w, h), (0.0, f * h), (w, f * h)] {
            let (x, y) = full.pixel_to_crs.forward(col, row);
            let (lon, lat) = crs.to_lonlat_deg(x, y);
            let (across, up) = if mercator { lonlat_deg_to_mercator(lon, lat) } else { (lon, lat) };
            if across.is_finite() && up.is_finite() {
                west = west.min(across);
                north = north.max(up);
            }
        }
    }
    (west.is_finite() && north.is_finite()).then_some(Wgs84Grid { west, north, res })
}

impl Wgs84Grid {
    /// rio-tiler's `align_bounds_with_dataset`: the box grown outwards to whole pixels.
    fn align(&self, [w, s, e, n]: [f64; 4]) -> [f64; 4] {
        let col0 = ((w - self.west) / self.res).floor();
        let col1 = ((e - self.west) / self.res).ceil();
        let row0 = ((self.north - n) / self.res).floor();
        let row1 = ((self.north - s) / self.res).ceil();
        [
            self.west + col0 * self.res,
            self.north - row1 * self.res,
            self.west + col1 * self.res,
            self.north - row0 * self.res,
        ]
    }
}

/// The overview GDAL reads when a window of `native` pixels is asked for at `out` pixels:
/// the most reduced one whose reduction stays under 1.2 times that of the less reduced
/// axis. `sizes` are the raster's levels, the full resolution first.
fn gdal_overview_level(sizes: &[(u32, u32)], native: (u64, u64), out: (usize, usize)) -> usize {
    let Some(&(full_w, full_h)) = sizes.first() else {
        return 0;
    };
    let wanted = (native.0 as f64 / out.0.max(1) as f64).min(native.1 as f64 / out.1.max(1) as f64);
    let (mut best, mut best_factor) = (0usize, 0f64);
    for (level, &(w, h)) in sizes.iter().enumerate().skip(1) {
        let factor = (f64::from(full_w) / f64::from(w.max(1))).min(f64::from(full_h) / f64::from(h.max(1)));
        if factor < wanted * 1.2 && factor > best_factor {
            (best, best_factor) = (level, factor);
        }
    }
    best
}

/// GDAL's approximate transformer along one output row: the exact source position at the
/// two ends, a straight line between them when the middle is within an eighth of a pixel
/// of it, otherwise the same on each half. `exact` gives the position for a column.
fn approx_scanline(exact: &impl Fn(usize) -> (f64, f64), out: &mut [(f64, f64)], tolerance: f64) {
    let mut todo = vec![(0usize, out.len())];
    while let Some((lo, hi)) = todo.pop() {
        let points = hi - lo;
        if points <= 5 {
            for (k, slot) in out[lo..hi].iter_mut().enumerate() {
                *slot = exact(lo + k);
            }
            continue;
        }
        let middle = (points - 1) / 2;
        let ((u0, v0), (um, vm), (u1, v1)) = (exact(lo), exact(lo + middle), exact(hi - 1));
        let span = (points - 1) as f64;
        let (du, dv) = ((u1 - u0) / span, (v1 - v0) / span);
        let error = (u0 + du * middle as f64 - um).abs() + (v0 + dv * middle as f64 - vm).abs();
        if error.is_finite() && error <= tolerance {
            for (k, slot) in out[lo..hi].iter_mut().enumerate() {
                *slot = (u0 + du * k as f64, v0 + dv * k as f64);
            }
        } else if error.is_finite() {
            todo.push((lo, lo + middle));
            todo.push((lo + middle, hi));
        } else {
            // Part of the row is outside the projection: no line to draw.
            for (k, slot) in out[lo..hi].iter_mut().enumerate() {
                *slot = exact(lo + k);
            }
        }
    }
}

/// A row of a picture: the exact source position at the two ends and the middle of a
/// stretch, and a curve through the three for the columns between. The curve is checked
/// against the exact position a quarter and three quarters of the way along, and the
/// stretch is halved until it passes. A straight line through the ends, which is what GDAL
/// draws, is off by the row's whole bend; the curve is off only by how that bend changes
/// along the row, which for a map projection is next to nothing, so a row costs a handful
/// of projections and still lands within a thousandth of a pixel. The two halves of a
/// stretch share its middle, and the columns it was checked at are their middles, so
/// halving costs each half only its own two checks.
fn curved_scanline(exact: &impl Fn(usize) -> (f64, f64), out: &mut [(f64, f64)], tolerance: f64) {
    // Nine columns or fewer are worked out one by one.
    if out.len() <= 9 {
        for (k, slot) in out.iter_mut().enumerate() {
            *slot = exact(k);
        }
        return;
    }
    let end = out.len() - 1;
    // A stretch is its first and last column and the exact positions at those two and at
    // the column midway.
    let mut todo = vec![(0usize, end, exact(0), exact(end / 2), exact(end))];
    while let Some((lo, hi, p0, pm, p1)) = todo.pop() {
        let last = hi - lo;
        let middle = last / 2;
        if last < 9 {
            for k in 0..=last {
                out[lo + k] = if k == 0 {
                    p0
                } else if k == middle {
                    pm
                } else if k == last {
                    p1
                } else {
                    exact(lo + k)
                };
            }
            continue;
        }
        // The curve through the three, as `a + b x + c x^2` at column `x` of the stretch,
        // worked out once so that a column costs two multiplications a coordinate.
        let (m, n) = (middle as f64, last as f64);
        let through = |p0: f64, pm: f64, p1: f64| {
            let c = ((p1 - p0) / n - (pm - p0) / m) / (n - m);
            (p0, (pm - p0) / m - c * m, c)
        };
        let (u, v) = (through(p0.0, pm.0, p1.0), through(p0.1, pm.1, p1.1));
        let at = |k: usize| {
            let x = k as f64;
            (u.0 + x * (u.1 + x * u.2), v.0 + x * (v.1 + x * v.2))
        };
        // How far the curve is from the exact position at the two columns it is checked at.
        let (quarter, three_quarters) = (middle / 2, middle + (last - middle) / 2);
        let (q1, q3) = (exact(lo + quarter), exact(lo + three_quarters));
        let (first, second) = ((at(quarter).0 - q1.0, at(quarter).1 - q1.1), (at(three_quarters).0 - q3.0, at(three_quarters).1 - q3.1));
        // Any of the five positions outside the projection leaves one of these not a number.
        let finite = (first.0 + first.1 + second.0 + second.1).is_finite();
        let error = first.0.abs().max(first.1.abs()).max(second.0.abs()).max(second.1.abs());
        // What a smooth row leaves at the two columns is equal and opposite, bar a sliver.
        // A row that jumps, as a longitude does where it wraps, leaves them lopsided by at
        // least a quarter of the jump, even when the row's bend hides the jump from each
        // column on its own. Held to an eighth of the tolerance, no jump of more than half
        // the tolerance is drawn across, and a row that is smooth but for one jump is
        // nowhere more than twice the tolerance out: the margin the check at a source
        // pixel's edge is built on. That is not a bound on any row whatever: one that
        // wiggles between the five columns looked at is not seen, and no projection does.
        let lopsided = (first.0 + second.0).abs().max((first.1 + second.1).abs());
        if finite && error <= tolerance && lopsided <= tolerance / 8.0 {
            for (k, slot) in out[lo..=hi].iter_mut().enumerate() {
                *slot = at(k);
            }
        } else if finite {
            todo.push((lo, lo + middle, p0, q1, pm));
            todo.push((lo + middle, hi, pm, q3, p1));
        } else {
            // Part of the row is outside the projection: no curve to draw.
            for (k, slot) in out[lo..=hi].iter_mut().enumerate() {
                *slot = exact(lo + k);
            }
        }
    }
}

/// How far, in source pixels, a position drawn along a row of a picture may be from the
/// exact one. Positions within twice this of a source pixel's edge are worked out exactly.
const PICTURE_TOLERANCE: f64 = 1.0 / 1024.0;

/// Whether a position drawn along a picture's row is close enough to the edge of a source
/// pixel that the exact position could be in the pixel next to it.
fn near_pixel_edge(p: f64) -> bool {
    let part = p - p.floor();
    !(2.0 * PICTURE_TOLERANCE..=1.0 - 2.0 * PICTURE_TOLERANCE).contains(&part)
}

/// Where every column of one row of the grid reads from, a stretch of `block` columns at
/// a time: for a picture the checked curve, otherwise GDAL's straight lines.
fn row_source_positions(exact: &impl Fn(usize) -> (f64, f64), block: usize, every_pixel: bool, out: &mut [(f64, f64)]) {
    let block = block.max(1);
    for (n, stretch) in out.chunks_mut(block).enumerate() {
        if every_pixel {
            curved_scanline(&|i| exact(n * block + i), stretch, PICTURE_TOLERANCE);
        } else {
            approx_scanline(&|i| exact(n * block + i), stretch, 0.125);
        }
    }
}

/// The source pixel, as (column, row), that a position drawn along a row names in a level
/// `width` by `height`, or none when it is outside the level. `exact` works the position
/// out again where the drawn one cannot be trusted to name the right pixel.
#[inline]
fn source_cell(
    drawn: (f64, f64),
    (width, height): (u32, u32),
    every_pixel: bool,
    exact: impl FnOnce() -> (f64, f64),
) -> Option<(i32, i32)> {
    let (w, h) = (f64::from(width), f64::from(height));
    let inside = |u: f64, v: f64| u >= 0.0 && v >= 0.0 && u + 1e-10 <= w && v + 1e-10 <= h;
    let (mut u, mut v) = drawn;
    // A position less than a pixel outside the raster is worked out again exactly.
    if !inside(u, v) && u > -1.0 && v > -1.0 && u < w + 1.0 && v < h + 1.0 {
        (u, v) = exact();
    } else if every_pixel && u.is_finite() && v.is_finite() {
        // A position drawn between two exact ones is off by no more than the
        // tolerance, which only matters next to the edge of a source pixel: there
        // it could name the neighbour. Those are worked out exactly, so the pixel
        // taken is always the one the projection itself gives.
        if near_pixel_edge(u) || near_pixel_edge(v) {
            (u, v) = exact();
        }
    }
    (u.is_finite() && v.is_finite() && inside(u, v)).then(|| {
        (((u + 1e-10).floor() as i32).min(width as i32 - 1), ((v + 1e-10).floor() as i32).min(height as i32 - 1))
    })
}

/// How GDAL goes about a warped read, which decides the pixels it takes.
#[derive(Debug, Clone, Copy, PartialEq)]
enum WarpPath {
    /// Every band of the raster is asked for: one warp straight onto the output grid,
    /// from the source overview that suits it.
    Direct,
    /// Only some bands are: the warped raster is read block by block, 512 pixels wide,
    /// through its own overview when the output is smaller, and that is then resized.
    Blocks,
}

/// Every band of one raster on a WGS84 grid, or with `mercator` on a Web Mercator one, read the way GDAL's nearest-neighbour warp
/// reads it: from the overview GDAL would pick, each output pixel taking the source pixel
/// its centre falls in. A pixel outside the raster, or without data, is NaN. `native` is
/// the size of the grid at the raster's own resolution. With `only`, bands counted from
/// one, the other bands are neither fetched nor held: their planes come back empty.
#[allow(clippy::too_many_arguments)]
async fn warp_nearest(
    engine: &TileEngine,
    cog: &Arc<cogzilla_io::OpenedCog>,
    crs: &CrsModel,
    bounds: [f64; 4],
    mercator: bool,
    (width, height): (usize, usize),
    native: (u64, u64),
    path: WarpPath,
    unscale: bool,
    pool: bool,
    only: &[u16],
    every_pixel: bool,
    nodata: Option<f32>,
) -> Result<Vec<Vec<f32>>, EngineError> {
    if let (Some(named), Some(ifd)) = (nodata, cog.index.ifds.first())
        && !nodata_fits(named, ifd.sample_type)
    {
        return Err(EngineError::BadRequest(format!("`nodata` of {named} is not a value the raster's samples can hold")));
    }
    let sizes: Vec<(u32, u32)> = cog.index.ifds.iter().map(|i| (i.width, i.height)).collect();
    // The grid the warp is worked out on, and how wide a stretch of a row GDAL
    // approximates in one go.
    let (level, (grid_w, grid_h), block) = match path {
        WarpPath::Direct => (gdal_overview_level(&sizes, native, (width, height)), (width, height), width),
        WarpPath::Blocks => {
            // The warped raster has one overview per source overview, reduced as much.
            let (full_w, full_h) = sizes.first().copied().unwrap_or((1, 1));
            let reduced: Vec<(u32, u32)> = sizes
                .iter()
                .map(|&(w, h)| {
                    let across = native.0 as f64 / (f64::from(full_w) / f64::from(w.max(1)));
                    let down = native.1 as f64 / (f64::from(full_h) / f64::from(h.max(1)));
                    (((across + 0.5) as u32).max(1), ((down + 0.5) as u32).max(1))
                })
                .collect();
            let level = gdal_overview_level(&reduced, native, (width, height));
            let grid = if level == 0 { (native.0 as usize, native.1 as usize) } else { (reduced[level].0 as usize, reduced[level].1 as usize) };
            // Each output row is worked out along a whole row of that grid. A box far
            // larger than what is asked of it makes the grid enormous; past what a
            // picture's own rows cost many times over, the direct road is taken.
            if grid.0 > 16 * width.max(MAX_PART_SIDE as usize) {
                (gdal_overview_level(&sizes, native, (width, height)), (width, height), width)
            } else {
                (level, grid, grid.0.min(512))
            }
        }
    };
    let ifd = &cog.index.ifds[level];
    let bands = ifd.samples_per_pixel as usize;
    let (dx, dy) = ((bounds[2] - bounds[0]) / grid_w as f64, (bounds[3] - bounds[1]) / grid_h as f64);

    // Where each output pixel reads from, as (column, row) of the level. With `pool`
    // this and the sampling below are worked out on the pool for heavy work, so that the
    // rasters of a mosaic are worked out side by side and not on the workers that serve
    // requests; one raster of a tile is quicker done here than handed over.
    let (cog_for_cells, crs_for_cells) = (cog.clone(), crs.clone());
    let work = move || {
    let (cog, crs) = (cog_for_cells, crs_for_cells);
    let ifd = &cog.index.ifds[level];
    let mut cells: Vec<(i32, i32)> = Vec::with_capacity(width * height);
    let mut row_positions = vec![(0f64, 0f64); grid_w];
    let (mut c0, mut c1, mut r0, mut r1) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    let mut worked_row = usize::MAX;
    for j in 0..height {
        // The row and columns of the grid an output pixel takes: its own, or when the
        // grid is being resized the one its centre falls in.
        let grid_row = if grid_h == height { j } else { ((2 * j + 1) * grid_h / (2 * height)).min(grid_h - 1) };
        let lat = bounds[3] - (grid_row as f64 + 0.5) * dy;
        let exact = |i: usize| {
            let across = bounds[0] + (i as f64 + 0.5) * dx;
            // The grid is in degrees, or in Web Mercator metres for a map tile. A raster
            // in Web Mercator is then on the grid's own terms: going through degrees and
            // back would leave noise where pixel centres fall on source pixel edges.
            if mercator && matches!(crs, CrsModel::Epsg3857) {
                return ifd.crs_to_pixel.forward(across, lat);
            }
            let (lon, lat) = if mercator { mercator_to_lonlat_deg(across, lat) } else { (across, lat) };
            let (x, y) = crs.from_lonlat_deg(lon, lat);
            ifd.crs_to_pixel.forward(x, y)
        };
        if grid_row != worked_row {
            row_source_positions(&exact, block, every_pixel, &mut row_positions);
            worked_row = grid_row;
        }
        for i in 0..width {
            let grid_col = if grid_w == width { i } else { ((2 * i + 1) * grid_w / (2 * width)).min(grid_w - 1) };
            match source_cell(row_positions[grid_col], (ifd.width, ifd.height), every_pixel, || exact(grid_col)) {
                Some((col, row)) => {
                    (c0, c1, r0, r1) = (c0.min(col), c1.max(col), r0.min(row), r1.max(row));
                    cells.push((col, row));
                }
                None => cells.push((-1, -1)),
            }
        }
    }
    (cells, c0, c1, r0, r1)
    };
    let (cells, c0, c1, r0, r1) = if pool { engine.store.run_heavy_cpu(work).await? } else { work() };
    let read = |band: usize| only.is_empty() || only.contains(&(band as u16 + 1));
    if c0 > c1 {
        return Ok((0..bands).map(|b| if read(b) { vec![f32::NAN; width * height] } else { Vec::new() }).collect());
    }

    let (tx0, tx1) = (c0 as u32 / ifd.tile_width, c1 as u32 / ifd.tile_width);
    let (ty0, ty1) = (r0 as u32 / ifd.tile_height, r1 as u32 / ifd.tile_height);
    let (grid_w, grid_h) = (tx1 - tx0 + 1, ty1 - ty0 + 1);
    let planar = ifd.planar_config == cogzilla_tiff::PlanarConfig::Planar;
    let fetches = if planar { bands } else { 1 };
    let mut tiles = Vec::with_capacity(fetches);
    // A raster stored band by band is fetched band by band, and every band's blocks are
    // held until the planes are made: what that comes to is bounded before any is read.
    let held = (0..fetches).filter(|b| !planar || read(*b)).count() as u64
        * u64::from(grid_w) * u64::from(grid_h) * u64::from(ifd.tile_width) * u64::from(ifd.tile_height) * 4;
    if planar && held > MAX_WORKING_BYTES as u64 {
        return Err(EngineError::BadRequest(
            "the bands asked for are too many to read at this size; choose fewer with `bidx`".into(),
        ));
    }
    for plane in 0..fetches {
        if planar && !read(plane) {
            tiles.push(Vec::new());
            continue;
        }
        let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                indices.push(Some(ifd.tile_index(tx, ty, plane as u16) as u32));
            }
        }
        tiles.push(engine.store.get_blocks(cog, level, &indices).await?);
    }

    let cog_for_planes = cog.clone();
    let skip: Vec<bool> = if only.is_empty() { Vec::new() } else { (0..bands).map(|b| !read(b)).collect() };
    let work = move || {
    let cog = cog_for_planes;
    let ifd = &cog.index.ifds[level];
    let mut planes = Vec::with_capacity(bands);
    for band in 0..bands {
        if !skip.is_empty() && skip[band] {
            planes.push(Vec::new());
            continue;
        }
        let blocks: Vec<Option<&cogzilla_codec::DecodedBand>> = if planar {
            tiles[band].iter().map(|t| t.as_ref().and_then(|t| t.bands.first())).collect()
        } else {
            tiles[0].iter().map(|t| t.as_ref().and_then(|t| t.bands.get(band))).collect()
        };
        let mut source = cogzilla_kernel::BandSource::new(
            ifd.width,
            ifd.height,
            ifd.tile_width,
            ifd.tile_height,
            tx0,
            ty0,
            grid_w,
            grid_h,
            &blocks,
        );
        // A `nodata` of the request's own is told from the raster's values as stored.
        if unscale && nodata.is_none() {
            let scale = cog.index.scales.get(band).copied().unwrap_or(1.0);
            let offset = cog.index.offsets.get(band).copied().unwrap_or(0.0);
            source = source.with_scale_offset(scale, offset);
        }
        planes.push(cells.iter().map(|&(col, row)| source.sample_checked(col, row)).collect::<Vec<f32>>());
    }
    if let Some(named) = nodata {
        // The request's value stands in the place of the raster's own, as in GDAL: where
        // the raster had none, its own value is a number again, and each band is without
        // data where it holds the value named. A pixel some bands of which are left is
        // still drawn, and a mosaic fills the others from the items below.
        let own = cog.index.nodata.filter(|n| !n.is_nan());
        for plane in planes.iter_mut().filter(|p| !p.is_empty()) {
            for (v, &(col, _)) in plane.iter_mut().zip(&cells) {
                if col < 0 {
                    continue;
                }
                if let (true, Some(own)) = (v.is_nan(), own) {
                    *v = own;
                }
                if *v == named {
                    *v = f32::NAN;
                }
            }
        }
        if unscale {
            for (band, plane) in planes.iter_mut().enumerate() {
                let scale = cog.index.scales.get(band).copied().unwrap_or(1.0);
                let offset = cog.index.offsets.get(band).copied().unwrap_or(0.0);
                if scale != 1.0 || offset != 0.0 {
                    plane.iter_mut().for_each(|v| *v = *v * scale + offset);
                }
            }
        }
    }
    planes
    };
    Ok(if pool { engine.store.run_heavy_cpu(work).await? } else { work() })
}

/// A shape's rings, lines and points in the pixels of a grid over `bounds`, which is in
/// degrees or Web Mercator metres; the shape is in degrees whatever the grid is in.
fn rings_on_grid(mask: &PolygonMask, bounds: [f64; 4], (width, height): (usize, usize), mercator: bool) -> zonal::GridShape {
    let (dx, dy) = ((bounds[2] - bounds[0]) / width as f64, (bounds[3] - bounds[1]) / height as f64);
    let on_grid = |&(lon, lat): &(f64, f64)| {
        let (across, up) = if mercator { lonlat_deg_to_mercator(lon, lat) } else { (lon, lat) };
        ((across - bounds[0]) / dx, (bounds[3] - up) / dy)
    };
    zonal::GridShape {
        polygons: mask.polygons.iter().map(|polygon| polygon.iter().map(|ring| ring.iter().map(on_grid).collect()).collect()).collect(),
        lines: mask.lines.iter().map(|line| line.iter().map(on_grid).collect()).collect(),
        points: mask.points.iter().map(on_grid).collect(),
    }
}

/// Little-endian `f32` values as the renderer hands them over.
fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect()
}

/// Bands of one raster on a grid in degrees or Web Mercator metres through the shared
/// renderer, for the resampling methods other than nearest neighbour. A shape is burned
/// by the renderer on a grid in degrees; on one in metres the caller burns it.
#[allow(clippy::too_many_arguments)]
async fn resampled_planes(
    engine: &TileEngine,
    cog: &Arc<cogzilla_io::OpenedCog>,
    bounds: [f64; 4],
    mercator: bool,
    (width, height): (usize, usize),
    bands: &[u16],
    shape: Option<&PolygonMask>,
    resample: ResampleMode,
    unscale: bool,
    nodata: Option<f32>,
) -> Result<Vec<Vec<f32>>, EngineError> {
    let mut planes = Vec::with_capacity(bands.len());
    let cogs = std::slice::from_ref(cog);
    let (w, h) = (width as u32, height as u32);
    for band in bands {
        let one = BandRequest::Single {
            band_1based: *band,
            rescale: (0.0, 1.0),
            colormap: ColormapName::Grayscale,
        };
        let raw = if mercator {
            crate::stats::render_mercator_bbox(engine, cogs, bounds, w, h, &one, resample, OutputFormat::RawF32, nodata, None, None, false, unscale).await?
        } else {
            render_wgs84_bbox(engine, cogs, bounds, w, h, shape, &one, resample, OutputFormat::RawF32, nodata, None, None, shape.is_some(), unscale).await?
        };
        if raw.len() != width * height * 4 {
            return Err(EngineError::BadRequest("unexpected array size from the renderer".into()));
        }
        planes.push(f32s(&raw));
    }
    Ok(planes)
}

/// Planes read with a resampling that makes values between the raster's own, as GDAL's
/// warp leaves them: in the raster's type, so whole for a raster of whole numbers. What is
/// done with them after (statistics, band math, a picture) starts from there.
fn in_raster_type(mut planes: Vec<Vec<f32>>, sample_type: SampleType, unscale: bool) -> Vec<Vec<f32>> {
    // A cubic kernel overshoots: the type's own range holds what it can.
    let range = sample_type.value_range();
    if let (false, Some((low, high))) = (unscale, range) {
        for plane in &mut planes {
            for v in plane.iter_mut() {
                // NaN, a pixel without data, stays NaN through both.
                *v = v.round().clamp(low, high);
            }
        }
    }
    planes
}

/// A plane brought to another size by taking, for each pixel, the one its centre falls in.
fn resize_nearest(plane: Vec<f32>, (w, h): (usize, usize), (out_w, out_h): (usize, usize)) -> Vec<f32> {
    if (w, h) == (out_w, out_h) {
        return plane;
    }
    let mut out = Vec::with_capacity(out_w * out_h);
    for j in 0..out_h {
        let row = ((j * 2 + 1) * h / (out_h * 2)).min(h - 1) * w;
        out.extend((0..out_w).map(|i| plane[row + ((i * 2 + 1) * w / (out_w * 2)).min(w - 1)]));
    }
    out
}

/// The most pixels one feature's statistics may read.
const MAX_STATS_PIXELS: u64 = 8_388_608;
/// The most values, bands times pixels, the statistics of one feature are worked out over.
const MAX_STATS_VALUES: u64 = 4 * MAX_STATS_PIXELS;
/// The finest grid the coverage of a feature is worked out on, per side.
const MAX_COVER_SIDE: u64 = 65_536;

/// TiTiler's statistics, histogram and `cover_scale` parameters.
fn stats_options(pairs: &[(String, String)]) -> Result<(StatsOptions, usize), Resp> {
    let mut opts = StatsOptions::default();
    if let Some(v) = query_value(pairs, "categorical") {
        opts.categorical = parse_bool(v).ok_or_else(|| {
            invalid_param("query", "categorical", "Input should be a valid boolean")
        })?;
    }
    let all = |name: &'static str| pairs.iter().filter(move |(k, _)| k == name).map(|(_, v)| v.as_str());
    for v in all("c") {
        let c = v.parse::<f64>().ok().filter(|c| c.is_finite());
        opts.categories
            .push(c.ok_or_else(|| invalid_param("query", "c", "Input should be a valid number"))?);
    }
    let percentiles: Result<Vec<i64>, Resp> = all("p")
        .map(|v| {
            v.parse::<i64>()
                .ok()
                .filter(|p| (0..=100).contains(p))
                .ok_or_else(|| invalid_param("query", "p", "Input should be a valid integer"))
        })
        .collect();
    let percentiles = percentiles?;
    if !percentiles.is_empty() {
        opts.percentiles = percentiles;
    }
    if let Some(v) = query_value(pairs, "histogram_bins") {
        let bad = || invalid_param("query", "histogram_bins", r"String should match pattern '^\d+(,\d+)*$'");
        // FastAPI's regex requires unsigned integer tokens; delegate range & monotonicity checks to query.rs.
        if !v.split(',').all(|p| p.parse::<u32>().is_ok()) {
            return Err(bad());
        }
        opts.bins = match super::query::parse_histogram_bins_spec(v, 65_536, usize::MAX, false) {
            Ok(super::query::HistogramBinsSpec::Count(n)) => Bins::Count(n),
            Ok(super::query::HistogramBinsSpec::Edges(edges)) => Bins::Edges(edges),
            Err(_) => return Err(bad()),
        };
    }
    if let Some(v) = query_value(pairs, "histogram_range") {
        let range = super::query::parse_histogram_range_strict(v, true).map_err(|_| {
            invalid_param("query", "histogram_range", "Input should be two numbers, the lower one first")
        })?;
        opts.range = Some(range);
    }
    let cover_scale = match query_value(pairs, "cover_scale") {
        None => 10,
        Some(v) => v.parse::<usize>().ok().filter(|n| (1..=1000).contains(n)).ok_or_else(|| {
            invalid_param("query", "cover_scale", "Input should be an integer from 1 to 1000")
        })?,
    };
    Ok((opts, cover_scale))
}

/// What one feature's statistics need besides the feature.
/// One of rio-tiler's algorithms (titiler's `titiler.core.algorithm`) as it is applied to a
/// mosaic's bands for statistics, a pixel without a value being NaN in every band. Across the
/// bands, as numpy's masked arrays work them: `min`, `max`, `sum`, `mean`, `median`, and
/// `std` and `var` with one degree of freedom less, over the values a pixel has.
#[derive(Clone, Debug, PartialEq)]
enum BandAlgorithm {
    Min,
    Max,
    Sum,
    Mean,
    Median,
    Std,
    Var,
    /// ITU-R 601-2 luma of the first three bands, in the bands' own type.
    Grayscale,
    /// 255 where the luma (or the one band) is above 127, else 0; titiler takes no threshold.
    Bitonal,
    /// To bytes, the way numpy's `astype("uint8")` wraps: after `ceil` or `floor`, or as they are.
    Cast,
    Ceil,
    Floor,
    /// Mapzen's Terrarium encoding of the first band.
    Terrarium,
    /// Mapbox's Terrain-RGB encoding of the first band.
    TerrainRgb { interval: f64, baseval: f64 },
    /// titiler's hillshade of the first band: numpy's gradient in the grid's own pixels, as
    /// bytes. It looks at a pixel's neighbours, so the grid is read `buffer` pixels larger
    /// on every side and that border taken off after.
    Hillshade { azimuth: f64, altitude: f64, buffer: usize, z: f64 },
    /// Degrees of slope of the first band, the gradient over the grid's pixel size, as float32.
    Slope { buffer: usize, z: f64 },
    /// Compass aspect in degrees `[0, 360)` of the first band, as float32 (NaN where flat).
    Aspect { buffer: usize, z: f64 },
    /// The first band rescaled from `minz..maxz` to 1..255 and coloured with `terrain`, black
    /// where `height % increment < thickness`.
    Contours { increment: f64, thickness: f64, minz: f64, maxz: f64 },
}

impl BandAlgorithm {
    /// The algorithm of that name with its `algorithm_params`, derived exhaustively from
    /// [`crate::algorithms::ALGORITHMS`]. Returns `Ok(None)` only for an unknown name or
    /// `normalizedIndex` (handled as band math `(b2 - b1) / (b2 + b1)` by the caller), and an
    /// error when a parameter is outside TiTiler's pydantic bounds.
    fn parse(name: &str, params: Option<&str>) -> Result<Option<Self>, String> {
        use crate::algorithms::AlgorithmId;
        let Some(def) = crate::algorithms::lookup_algorithm(name) else {
            return Ok(None);
        };
        let params: Value = params.and_then(|p| serde_json::from_str(p).ok()).unwrap_or(Value::Null);
        let empty = serde_json::Map::new();
        let params = params.as_object().unwrap_or(&empty);
        // TiTiler's models' checks, shared with every route that applies an algorithm; what
        // they take is then read as pydantic's lax mode reads it.
        super::algorithm_params::check(def.name, params)?;
        let number = |key: &str, default: f64| super::algorithm_params::number(params, key).unwrap_or(default);
        Ok(Some(match def.id {
            AlgorithmId::Min => Self::Min,
            AlgorithmId::Max => Self::Max,
            AlgorithmId::Sum => Self::Sum,
            AlgorithmId::Mean => Self::Mean,
            AlgorithmId::Median => Self::Median,
            AlgorithmId::Std => Self::Std,
            AlgorithmId::Var => Self::Var,
            AlgorithmId::Grayscale => Self::Grayscale,
            AlgorithmId::Bitonal => Self::Bitonal,
            AlgorithmId::Cast => Self::Cast,
            AlgorithmId::Ceil => Self::Ceil,
            AlgorithmId::Floor => Self::Floor,
            AlgorithmId::Terrarium => Self::Terrarium,
            AlgorithmId::TerrainRgb => {
                Self::TerrainRgb { interval: number("interval", 0.1), baseval: number("baseval", -10000.0) }
            }
            AlgorithmId::Hillshade => Self::Hillshade {
                azimuth: number("azimuth", 45.0),
                altitude: number("angle_altitude", 45.0),
                buffer: number("buffer", 3.0) as usize,
                z: number("z_exaggeration", 1.0),
            },
            AlgorithmId::Slope => Self::Slope {
                buffer: number("buffer", 3.0) as usize,
                z: number("z_exaggeration", 1.0),
            },
            AlgorithmId::Aspect => Self::Aspect {
                buffer: number("buffer", 0.0) as usize,
                z: number("z_exaggeration", 1.0),
            },
            AlgorithmId::Contours => Self::Contours {
                increment: number("increment", 35.0),
                thickness: number("thickness", 1.0),
                minz: number("minz", -12000.0),
                maxz: number("maxz", 8000.0),
            },
            // Handled out-of-band as `(b2 - b1) / (b2 + b1)` band math when `expression` is absent.
            AlgorithmId::NormalizedIndex => return Ok(None),
        }))
    }

    /// The pixels read around the grid on every side, and taken off after.
    fn buffer(&self) -> usize {
        match self {
            Self::Hillshade { buffer, .. } | Self::Slope { buffer, .. } | Self::Aspect { buffer, .. } => *buffer,
            _ => 0,
        }
    }

    /// Whether a pixel's value comes from its neighbours too (`hillshade`, `slope`, `aspect`).
    fn looks_around(&self) -> bool {
        matches!(self, Self::Hillshade { .. } | Self::Slope { .. } | Self::Aspect { .. })
    }

    /// Whether the values are float32, as rio-tiler's statistics then count them.
    fn float32(&self) -> bool {
        matches!(self, Self::Slope { .. } | Self::Aspect { .. })
    }

    /// The bands after the algorithm, and their descriptions. `whole` says the bands hold
    /// their rasters' whole numbers, which `grayscale` brings its result back to.
    /// `width` is the grid's, and `pixel` the size of its pixels across and down.
    fn apply(&self, bands: Vec<Vec<f64>>, whole: bool, width: usize, pixel: (f64, f64)) -> Result<(Vec<Vec<f64>>, Vec<String>), String> {
        let n = bands.len();
        let len = bands.first().map_or(0, Vec::len);
        let height = len.checked_div(width).unwrap_or(0);
        let across = |f: &dyn Fn(&mut Vec<f64>) -> f64| -> Vec<f64> {
            let mut values = Vec::with_capacity(n);
            (0..len)
                .map(|i| {
                    values.clear();
                    values.extend(bands.iter().map(|b| b[i]).filter(|v| !v.is_nan()));
                    if values.is_empty() { f64::NAN } else { f(&mut values) }
                })
                .collect()
        };
        let one = |plane: Vec<f64>, name: &str| Ok((vec![plane], vec![name.to_string()]));
        let byte = |v: f64| if v.is_nan() { v } else { (v.trunc() as i64).rem_euclid(256) as f64 };
        // numpy: `a[0] * 299 / 1000 + a[1] * 587 / 1000 + a[2] * 114 / 1000`.
        let luma = |i: usize| bands[0][i] * 299.0 / 1000.0 + bands[1][i] * 587.0 / 1000.0 + bands[2][i] * 114.0 / 1000.0;
        match self {
            Self::Min => one(across(&|v| v.iter().copied().fold(f64::INFINITY, f64::min)), "min"),
            Self::Max => one(across(&|v| v.iter().copied().fold(f64::NEG_INFINITY, f64::max)), "max"),
            Self::Sum => one(across(&|v| v.iter().sum()), "sum"),
            Self::Mean => one(across(&|v| v.iter().sum::<f64>() / v.len() as f64), "mean"),
            Self::Median => one(
                across(&|v| {
                    v.sort_by(f64::total_cmp);
                    let m = v.len() / 2;
                    if v.len() % 2 == 1 { v[m] } else { (v[m - 1] + v[m]) / 2.0 }
                }),
                "median",
            ),
            Self::Std | Self::Var => {
                let var = across(&|v| {
                    if v.len() < 2 {
                        return f64::NAN;
                    }
                    let mean = v.iter().sum::<f64>() / v.len() as f64;
                    v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (v.len() - 1) as f64
                });
                match self {
                    Self::Std => one(var.into_iter().map(f64::sqrt).collect(), "std"),
                    _ => one(var, "var"),
                }
            }
            Self::Grayscale => {
                if n < 3 {
                    return Err(format!("Cannot apply `grayscale` algorithm on image with {n} bands."));
                }
                one((0..len).map(|i| if whole { luma(i).trunc() } else { luma(i) }).collect(), "grayscale")
            }
            Self::Bitonal => {
                let source: Vec<f64> = match n {
                    3 => (0..len).map(luma).collect(),
                    1 => bands[0].clone(),
                    _ => return Err(format!("Cannot apply `bitonal` algorithm on image with {n} bands.")),
                };
                one(source.into_iter().map(|v| if v.is_nan() { v } else if v > 127.0 { 255.0 } else { 0.0 }).collect(), "bitonal")
            }
            Self::Cast | Self::Ceil | Self::Floor => {
                let round = |v: f64| match self {
                    Self::Ceil => v.ceil(),
                    Self::Floor => v.floor(),
                    _ => v,
                };
                let out = bands.into_iter().map(|b| b.into_iter().map(|v| byte(round(v))).collect()).collect();
                Ok((out, Vec::new()))
            }
            Self::Terrarium => {
                let data: Vec<f64> = bands[0].iter().map(|v| (v + 32768.0).clamp(0.0, 65535.0)).collect();
                let r = data.iter().map(|d| byte(d / 256.0)).collect();
                let g = data.iter().map(|d| byte(d % 256.0)).collect();
                let b = data.iter().map(|d| byte((d * 256.0) % 256.0)).collect();
                Ok((vec![r, g, b], ["terranium_r", "terranium_g", "terranium_b"].map(String::from).to_vec()))
            }
            Self::TerrainRgb { interval, baseval } => {
                // numpy's `around` rounds half to even.
                let data: Vec<f64> = bands[0].iter().map(|v| ((v - baseval) / interval).round_ties_even()).collect();
                let (lo, hi) = data.iter().filter(|v| !v.is_nan()).fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| (l.min(*v), h.max(*v)));
                if hi - lo > 256f64.powi(3) {
                    return Err(format!("Data of {} larger than 256 ** 3", hi - lo));
                }
                let part = |shift: u32| data.iter().map(|d| if d.is_nan() { *d } else { f64::from(((*d as i32) >> shift) & 0xFF) }).collect();
                Ok((vec![part(16), part(8), part(0)], ["terrain_r", "terrain_g", "terrain_b"].map(String::from).to_vec()))
            }
            Self::Hillshade { azimuth, altitude, buffer, z } => {
                let (down, across) = gradient(&bands[0], width, height)?;
                let (azimuth, altitude) = ((360.0 - azimuth).to_radians(), altitude.to_radians());
                let shaded: Vec<f64> = down
                    .iter()
                    .zip(&across)
                    .map(|(x, y)| {
                        let (x, y) = (x * z, y * z);
                        let slope = std::f64::consts::FRAC_PI_2 - (x * x + y * y).sqrt().atan();
                        let aspect = (-x).atan2(y);
                        let shade = altitude.sin() * slope.sin() + altitude.cos() * slope.cos() * (azimuth - aspect).cos();
                        let v = 255.0 * (shade + 1.0) / 2.0;
                        if v < 0.0 { 0.0 } else { v.trunc() }
                    })
                    .collect();
                one(crop(shaded, width, height, *buffer), "hillshade")
            }
            Self::Slope { buffer, z } => {
                let (down, across) = gradient(&bands[0], width, height)?;
                let slope: Vec<f64> = down
                    .iter()
                    .zip(&across)
                    .map(|(x, y)| {
                        let (dx, dy) = (x * z / pixel.0, y * z / pixel.1);
                        f64::from((dx * dx + dy * dy).sqrt().atan().to_degrees() as f32)
                    })
                    .collect();
                one(crop(slope, width, height, *buffer), "slope")
            }
            Self::Aspect { buffer, z } => {
                let (down, across) = gradient(&bands[0], width, height)?;
                let aspect: Vec<f64> = down
                    .iter()
                    .zip(&across)
                    .map(|(y, x)| {
                        let (dx, dy) = (x * z / pixel.0, y * z / pixel.1);
                        if !dx.is_finite() || !dy.is_finite() || (dx.abs() < 1e-12 && dy.abs() < 1e-12) {
                            f64::NAN
                        } else {
                            f64::from((-dx).atan2(dy).to_degrees().rem_euclid(360.0) as f32)
                        }
                    })
                    .collect();
                one(crop(aspect, width, height, *buffer), "aspect")
            }
            Self::Contours { increment, thickness, minz, maxz } => {
                if n != 1 {
                    // rio-tiler's `apply_cmap`.
                    return Err("Source data must be 1 band".into());
                }
                let lut = &ColormapName::from_str_opt("terrain").ok_or("no terrain colour map")?.lut().rgb;
                let mut out: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(len)).collect();
                for v in &bands[0] {
                    let rgb = if v.is_nan() {
                        [f64::NAN; 3]
                    } else if v.rem_euclid(*increment) < *thickness {
                        [0.0; 3]
                    } else {
                        // rio-tiler's linear_rescale, then `astype("uint8")`.
                        let at = ((v.clamp(*minz, *maxz) - minz) / (maxz - minz) * 254.0 + 1.0).trunc();
                        let c = lut[(at as i64).rem_euclid(256) as usize];
                        c.map(f64::from)
                    };
                    for (band, value) in out.iter_mut().zip(rgb) {
                        band.push(value);
                    }
                }
                Ok((out, ["contours_r", "contours_g", "contours_b"].map(String::from).to_vec()))
            }
        }
    }
}

/// numpy's `gradient` of a `width` x `height` plane, down the rows and then across the
/// columns: half the difference of the two neighbours inside, the difference with the one
/// neighbour on an edge. A neighbour without a value leaves none, as numpy's masked arrays do.
fn gradient(plane: &[f64], width: usize, height: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if width < 2 || height < 2 {
        return Err("Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.".into());
    }
    let at = |r: usize, c: usize| plane[r * width + c];
    let mut down = vec![0.0; width * height];
    let mut across = vec![0.0; width * height];
    for r in 0..height {
        for c in 0..width {
            down[r * width + c] = match r {
                0 => at(1, c) - at(0, c),
                _ if r == height - 1 => at(r, c) - at(r - 1, c),
                _ => (at(r + 1, c) - at(r - 1, c)) / 2.0,
            };
            across[r * width + c] = match c {
                0 => at(r, 1) - at(r, 0),
                _ if c == width - 1 => at(r, c) - at(r, c - 1),
                _ => (at(r, c + 1) - at(r, c - 1)) / 2.0,
            };
        }
    }
    Ok((down, across))
}

/// A plane with `buffer` pixels taken off every side.
fn crop(plane: Vec<f64>, width: usize, height: usize, buffer: usize) -> Vec<f64> {
    if buffer == 0 {
        return plane;
    }
    let (w, h) = (width.saturating_sub(2 * buffer), height.saturating_sub(2 * buffer));
    (0..h).flat_map(|r| plane[(r + buffer) * width + buffer..][..w].to_vec()).collect()
}

struct StatsRequest<'a> {
    query: MosaicQuery,
    size: PartSize,
    opts: StatsOptions,
    cover_scale: usize,
    expression: Option<&'a str>,
    unscale: bool,
    /// A no-data value of the request's own, in the place of each raster's.
    nodata: Option<f32>,
    /// Whether the pixels counted are those of a grid in Web Mercator metres (`dst_crs`).
    mercator: bool,
    /// Whether the band math is `normalizedIndex`'s, whose values rio-tiler gives as float32.
    single: bool,
    /// An algorithm applied to the mosaic's bands before they are counted.
    algorithm: Option<BandAlgorithm>,
}

/// The statistics of one feature and the items they were read from, or `None` when the
/// search has nothing under the feature.
async fn feature_statistics(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    hash: &str,
    feature: &Value,
    req: &StatsRequest<'_>,
) -> Result<Option<(Value, Vec<String>)>, Resp> {
    let (mask, bbox) = PolygonMask::from_geojson(&polygons_of_collection(feature)).map_err(|e| engine_error(&e))?;
    // As in titiler-pgstac, the items are those of the feature's bounding box. Each raster
    // is read on a grid of its own, which no one set of sample points stands for, so the
    // numbers come from pgSTAC's list unless the request asks for the lean one.
    let names: Vec<String> = req.query.assets.iter().map(|a| a.name.clone()).collect();
    let lookup = if req.query.lookup == Lookup::Lean { Lookup::Lean } else { Lookup::Reference };
    // The lean lookup samples a grid in degrees; one in metres is other pixel centres.
    let sampled = if req.mercator { None } else { part_grid(req.size, bbox) };
    let search_box = bbox_polygon(bbox);
    // From here the box and the shape are in the system the pixels are counted in.
    let on_grid = |lon: f64, lat: f64| if req.mercator { lonlat_deg_to_mercator(lon, lat) } else { (lon, lat) };
    let bbox = {
        let ((w, s), (e, n)) = (on_grid(bbox[0], bbox[1]), on_grid(bbox[2], bbox[3]));
        [w, s, e, n]
    };
    if bbox.iter().any(|c| !c.is_finite()) {
        return Err(detail(StatusCode::BAD_REQUEST, "the feature reaches past the edge of Web Mercator, 85.05 degrees north or south"));
    }
    let mercator = req.mercator;
    let items = pixel_items(ctx, pg, &search_box, hash, req.query.limits, &names, sampled, lookup)
        .await
        .map_err(|e| db_error(&e))?;
    if items.is_empty() {
        return Ok(None);
    }
    let mut scenes: Vec<(String, Vec<String>)> = Vec::with_capacity(items.len().min(MAX_SCENES_PER_TILE));
    let assets = &req.query.assets;
    for item in items.iter().take(MAX_SCENES_PER_TILE) {
        let hrefs = asset_hrefs(item, assets.iter().map(|asset| asset.name.as_str()))?;
        scenes.push((item_name(item), hrefs));
    }

    let engine = ctx.engine;
    let computed: Result<Option<(Value, Vec<String>)>, EngineError> = async {
        // The mosaic: one plane per band, a pixel nothing has filled yet being NaN.
        let mut canvas: Vec<Vec<f64>> = Vec::new();
        let mut descriptions: Vec<String> = Vec::new();
        let mut program: Option<Program> = None;
        let mut program_bands = 0;
        // Whether the mosaic holds its rasters' own whole numbers, which an algorithm keeps.
        let mut whole = false;
        let mut used: Vec<String> = Vec::new();
        let (mut width, mut height, mut bounds) = (0usize, 0usize, bbox);
        // With a pixel selection other than `first`, every scene's bands go through this
        // and nothing stops the reading early.
        let mut fold: Option<Fold<f64>> = None;
        let scene_count = scenes.len();
        let pad = req.algorithm.as_ref().map_or(0, BandAlgorithm::buffer);
        for (name, hrefs) in scenes {
            let mut cogs = Vec::with_capacity(hrefs.len());
            for href in &hrefs {
                cogs.push(engine.store.open_uri(href).await.map(|c| engine.store.with_nodata(&c, req.nodata))?);
            }
            let mut planes: Vec<Vec<f64>> = Vec::new();
            let mut names: Vec<String> = Vec::new();
            let mut missing = false;
            'assets: for (asset, cog) in assets.iter().zip(&cogs) {
                // As rio-tiler does, every raster is read on its own grid: the feature's box
                // grown to whole pixels of the raster as GDAL would warp it to WGS84. The
                // first one read sets the size of the result.
                let crs = engine.get_or_build_crs(cog)?;
                let grid = warped_grid(&crs, cog, mercator).ok_or_else(|| {
                    EngineError::BadRequest(format!("cannot place asset '{}' in WGS84", asset.name))
                })?;
                let own_bounds = grid.align(bbox);
                let own = PartSize::Query { max_size: None, width: None, height: None };
                let native = part_dims(own, own_bounds, grid.res);
                let (w, h) = part_dims(req.size, own_bounds, grid.res);
                // An algorithm that looks at a pixel's neighbours is given `pad` pixels more on
                // every side, and takes them off after.
                let (own_bounds, native, w, h) = if pad > 0 {
                    let (span_x, span_y) = (own_bounds[2] - own_bounds[0], own_bounds[3] - own_bounds[1]);
                    let (grow_x, grow_y) = (pad as f64 * span_x / w as f64, pad as f64 * span_y / h as f64);
                    let grown = [own_bounds[0] - grow_x, own_bounds[1] - grow_y, own_bounds[2] + grow_x, own_bounds[3] + grow_y];
                    let more = |n: u64, grow: f64, span: f64| n + 2 * (grow / (span / n as f64)).round() as u64;
                    (grown, (more(native.0, grow_x, span_x), more(native.1, grow_y, span_y)), w + 2 * pad as u64, h + 2 * pad as u64)
                } else {
                    (own_bounds, native, w, h)
                };
                if w * h > MAX_STATS_PIXELS || w.max(h) * req.cover_scale as u64 > MAX_COVER_SIDE {
                    return Err(EngineError::BadRequest(format!(
                        "the feature would be read at {w}x{h} pixels with cover_scale {}; pass max_size, or width and height",
                        req.cover_scale
                    )));
                }
                let size = (w as usize, h as usize);
                if width == 0 {
                    (width, height, bounds) = (size.0, size.1, own_bounds);
                }
                // As on tiles: `resampling` for a source already in the output's projection,
                // `reproject` for one that has to be warped into it.
                let resample = match crs {
                    CrsModel::Geographic if !mercator => req.query.resampling,
                    CrsModel::Epsg3857 if mercator => req.query.resampling,
                    _ => req.query.reproject,
                };
                // GDAL takes another road when only some of a raster's bands are read.
                let all = usize::from(cog.index.full_res().samples_per_pixel);
                let path = match &asset.indexes {
                    Some(indexes) if asset.expression.is_none() && indexes.len() != all => WarpPath::Blocks,
                    _ => WarpPath::Direct,
                };
                let read = if resample == ResampleMode::Nearest {
                    warp_nearest(engine, cog, &crs, own_bounds, mercator, size, native, path, req.unscale, true, &[], false, req.nodata).await
                } else {
                    let every: Vec<u16> = (1..=cog.index.full_res().samples_per_pixel).collect();
                    resampled_planes(engine, cog, own_bounds, mercator, size, &every, None, resample, req.unscale, req.nodata)
                        .await
                        .map(|planes| in_raster_type(planes, cog.index.full_res().sample_type, req.unscale))
                };
                match read {
                    Ok(read) => {
                        let (plan, described) = asset_bands(asset, read.len(), req.query.asset_as_band)
                            .map_err(EngineError::BadRequest)?;
                        let read: Vec<Vec<f32>> =
                            read.into_iter().map(|p| resize_nearest(p, size, (width, height))).collect();
                        planes.extend(plan.planes(&read));
                        names.extend(described);
                    }
                    // The item's footprint meets the box but this raster does not.
                    Err(EngineError::NotFound(_)) => {
                        missing = true;
                        break 'assets;
                    }
                    Err(e) => return Err(e),
                }
            }
            if missing {
                continue;
            }
            if canvas.is_empty() {
                if let Some(expression) = req.expression {
                    let parsed = Program::parse(expression, planes.len()).map_err(EngineError::BadRequest)?;
                    program_bands = planes.len();
                    descriptions = parsed.describe(&names);
                    program = Some(parsed);
                } else {
                    descriptions = names;
                }
                // The pixel limit is each raster's; with many bands it is their sum that is held.
                let bands = planes.len().max(descriptions.len()) as u64;
                if bands * (width * height) as u64 > MAX_STATS_VALUES {
                    return Err(EngineError::BadRequest(format!(
                        "{bands} bands at {width}x{height} pixels are too many to hold; pass max_size, or width and height"
                    )));
                }
                canvas = vec![vec![f64::NAN; width * height]; descriptions.len()];
                whole = !req.unscale
                    && assets.iter().all(|a| a.expression.is_none())
                    && !matches!(cogs[0].index.full_res().sample_type, SampleType::Float32 | SampleType::Float64);
                if let Some(method) = req.query.selection {
                    // The last band decides and is left out: alone, it would leave nothing.
                    if method.by_last_band() && canvas.len() < 2 {
                        return Err(EngineError::BadRequest(
                            "pixel_selection by the last band needs at least two bands".into(),
                        ));
                    }
                    // Whole numbers when the values are the first raster's own, or band math
                    // over them that numexpr keeps whole (`b1*2`, not `b1/2`).
                    let integer = program.as_ref().is_none_or(|p| p.kind() == Kind::Whole)
                        && !req.unscale
                        && assets.iter().all(|a| a.expression.is_none())
                        && !matches!(cogs[0].index.full_res().sample_type, SampleType::Float32 | SampleType::Float64);
                    let new = fold_within_limits::<f64>(method, scene_count, canvas.len(), width * height, integer, false)?;
                    fold = Some(new.absent_as(req.nodata.map_or(0.0, f64::from)));
                }
            }
            // The expression was read for the first item's bands; an item with another number
            // of them has no band for it to name.
            if program.is_some() && planes.len() != program_bands {
                return Err(EngineError::BadRequest(
                    "the items under the feature do not have the same number of bands".into(),
                ));
            }
            // The request's band math over this item's bands, or the bands as they are.
            let bands: Vec<Vec<f64>> = match &program {
                Some(program) => {
                    // With `first`, only what the mosaic still lacks is worked out.
                    let mut bands = if fold.is_some() { vec![vec![f64::NAN; width * height]; canvas.len()] } else { canvas.clone() };
                    let mut pixel = vec![0f64; planes.len()];
                    for i in 0..width * height {
                        if bands.iter().all(|band| !band[i].is_nan()) {
                            continue;
                        }
                        for (value, plane) in pixel.iter_mut().zip(&planes) {
                            *value = plane[i];
                        }
                        for (part, band) in bands.iter_mut().enumerate() {
                            if band[i].is_nan() {
                                let v = program.eval(part, &pixel);
                                let v = if req.single { f64::from(v as f32) } else { v };
                                if v.is_finite() {
                                    band[i] = v;
                                }
                            }
                        }
                    }
                    bands
                }
                None => {
                    if planes.len() != canvas.len() {
                        return Err(EngineError::BadRequest(
                            "the items under the feature do not have the same number of bands".into(),
                        ));
                    }
                    planes
                }
            };
            if let Some(fold) = &mut fold {
                fold.add(bands.iter().flatten().map(|v| if v.is_finite() { *v } else { f64::NAN }));
                used.push(name);
                continue;
            }
            for (band, plane) in canvas.iter_mut().zip(&bands) {
                for (dst, src) in band.iter_mut().zip(plane) {
                    if dst.is_nan() && src.is_finite() {
                        *dst = *src;
                    }
                }
            }
            used.push(name);
            if canvas.iter().all(|band| band.iter().all(|v| !v.is_nan())) {
                break;
            }
        }
        if canvas.is_empty() {
            return Ok(None);
        }
        if let Some(selected) = fold.and_then(Fold::finish) {
            // Comparisons are bools in numpy, which a mean or a median is cast back to: true
            // where it is not 0. A whole number cast back has no sign at 0 (-0.5 is 0).
            let bools = program.as_ref().is_some_and(|p| p.kind() == Kind::Bool);
            let cast = |v: f64| if bools && v.is_finite() { f64::from(u8::from(v != 0.0)) } else { v + 0.0 };
            // `count` leaves one band, described as the first.
            canvas = selected.values.chunks(width * height).map(|band| band.iter().map(|&v| cast(v)).collect()).collect();
            descriptions.truncate(selected.bands);
        }
        if let Some(algorithm) = &req.algorithm {
            // Band math over whole numbers that leaves whole numbers (numexpr's integer
            // arithmetic) keeps the raster's type, which `grayscale` comes back to.
            let whole = whole && (program.is_none() || canvas.iter().flatten().all(|v| v.is_nan() || v.fract() == 0.0));
            let pixel = ((bounds[2] - bounds[0]) / width as f64, (bounds[3] - bounds[1]) / height as f64);
            // rio-tiler's feature image is masked outside the shape, and a gradient beside the
            // mask is masked too: the shape is burned on the grid before them. A `buffer` is
            // there to give the pixels on the edge real neighbours (titiler-pgstac fails with
            // one, its coverage the size of the uncropped image), so with one the gradient is
            // taken over everything read and only the statistics keep to the shape.
            if algorithm.looks_around() && pad == 0 {
                let shape = rings_on_grid(&mask, bounds, (width, height), mercator);
                let inside = zonal::shape_cover_counts(&shape, width, height, 1).map_err(EngineError::BadRequest)?;
                for band in &mut canvas {
                    for (v, hit) in band.iter_mut().zip(&inside) {
                        if *hit == 0 {
                            *v = f64::NAN;
                        }
                    }
                }
            }
            let (bands, described) = algorithm.apply(canvas, whole, width, pixel).map_err(EngineError::BadRequest)?;
            canvas = bands;
            if pad > 0 {
                (width, height) = (width - 2 * pad, height - 2 * pad);
                bounds = [bounds[0] + pad as f64 * pixel.0, bounds[1] + pad as f64 * pixel.1, bounds[2] - pad as f64 * pixel.0, bounds[3] - pad as f64 * pixel.1];
            }
            // `cast`, `ceil` and `floor` keep the bands' own descriptions.
            if !described.is_empty() {
                descriptions = described;
            }
        }

        // The polygon on the image's grid: once for the mask, once finer for the coverage.
        let rings = rings_on_grid(&mask, bounds, (width, height), mercator);
        let (opts, cover_scale, single) = (req.opts.clone(), req.cover_scale, req.single);
        let stats = engine
            .store
            .run_heavy_cpu(move || {
                let inside = zonal::shape_cover_counts(&rings, width, height, 1).map_err(EngineError::BadRequest)?;
                let cells = (cover_scale * cover_scale) as f32;
                let coverage: Vec<f32> = zonal::shape_cover_counts(&rings, width, height, cover_scale)
                    .map_err(EngineError::BadRequest)?
                    .iter()
                    .map(|c| *c as f32 / cells)
                    .collect();
                let mut out = serde_json::Map::new();
                for (n, (band, description)) in canvas.iter_mut().zip(&descriptions).enumerate() {
                    for (v, hit) in band.iter_mut().zip(&inside) {
                        if *hit == 0 {
                            *v = f64::NAN;
                        }
                    }
                    let stats = if single {
                        zonal::band_statistics(band, &coverage, description, &float32_bins(&opts, band))
                    } else {
                        zonal::band_statistics(band, &coverage, description, &opts)
                    };
                    out.insert(format!("b{}", n + 1), stats);
                }
                Ok::<_, EngineError>(Value::Object(out))
            })
            .await??;
        Ok(Some((stats, used)))
    }
    .await;
    computed.map_err(|e| engine_error(&e))
}

/// The histogram options for a float32 band, with its bin edges as numpy's `histogram`
/// makes them and counts against them. Between the band's own least and greatest values
/// (float32 both), `linspace` works in float32: the step is their difference over the count,
/// each edge the step times its index plus the first, the last set to the greatest. Between a
/// range given in the request (Python floats) it works in double precision and casts each
/// edge to float32. Bins given as edges, and a band of one value, are left as they are.
fn float32_bins(opts: &StatsOptions, band: &[f64]) -> StatsOptions {
    let mut out = opts.clone();
    let zonal::Bins::Count(count) = opts.bins else { return out };
    if count == 0 {
        return out;
    }
    let edges: Vec<f64> = match opts.range {
        Some((lo, hi)) => (0..=count).map(|i| f64::from((lo + (hi - lo) * i as f64 / count as f64) as f32)).collect(),
        None => {
            let (lo, hi) = band
                .iter()
                .filter(|v| v.is_finite())
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v as f32), hi.max(v as f32)));
            if lo >= hi || lo.is_nan() || hi.is_nan() {
                return out;
            }
            let step = (hi - lo) / count as f32;
            (0..=count).map(|i| f64::from(if i == count { hi } else { i as f32 * step + lo })).collect()
        }
    };
    out.bins = zonal::Bins::Edges(edges);
    out
}

/// `POST {base}/statistics`: statistics of the mosaic under each feature of a GeoJSON
/// Feature or FeatureCollection, added to the feature's properties with the items used.
async fn mosaic_statistics(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    hash: &str,
    body: Option<&Bytes>,
    as_posted: Option<&Bytes>,
    mut pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    // Read here; the shared parser does not know them on this route.
    const OWN: [&str; 10] = [
        "width", "height", "categorical", "c", "p", "histogram_bins", "histogram_range", "cover_scale",
        // Statistics have no colours: titiler-pgstac passes these over, whatever they hold.
        "colormap", "colormap_name",
    ];
    pq.errors.retain(|e| !error_parameter(e).is_some_and(|name| OWN.contains(&name)));
    let uncoloured: Vec<(String, String)> =
        pairs.iter().filter(|(k, _)| !matches!(k.as_str(), "colormap" | "colormap_name")).cloned().collect();
    let query = match check_mosaic_query(&pq, &uncoloured) {
        Ok(q) => q,
        Err(r) => return r,
    };
    if let Err(r) = check_coord_crs(&pq, pairs) {
        return r;
    }
    // The pixels counted are those of a grid in `dst_crs`: degrees, or Web Mercator metres.
    let mercator = match drawn_in_mercator(pairs, false) {
        Ok(mercator) => mercator,
        Err(r) => return r,
    };
    // `normalizedIndex` is rio-tiler's band math of the first two bands, written as it
    // describes it; the other algorithms are applied to the mosaic's bands
    // (`BandAlgorithm`), as titiler-pgstac's `post_process` applies them.
    let mut algorithm = None;
    let index_math = match query_value(pairs, "algorithm").map(str::to_ascii_lowercase).as_deref() {
        None => None,
        Some("normalizedindex") if query_value(pairs, "expression").is_none() => Some("(b2 - b1) / (b2 + b1)"),
        Some(name) => match BandAlgorithm::parse(name, query_value(pairs, "algorithm_params")) {
            Ok(Some(a)) => {
                algorithm = Some(a);
                None
            }
            Ok(None) => {
                return detail(
                    StatusCode::BAD_REQUEST,
                    format!("algorithm '{name}' is not supported on statistics routes (normalizedIndex is not with `expression`)"),
                );
            }
            Err(msg) => return detail(StatusCode::BAD_REQUEST, msg),
        },
    };
    let size = match part_size_from_query(pairs) {
        Ok(s) => s,
        Err(r) => return r,
    };
    let (opts, cover_scale) = match stats_options(pairs) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let bad_body = || invalid_param("body", "body", "Input should be a GeoJSON Feature or FeatureCollection");
    let Some(geojson) = body.and_then(|b| serde_json::from_slice::<Value>(b).ok()) else {
        return bad_body();
    };
    let is_feature = |v: &Value| v.get("type").and_then(Value::as_str) == Some("Feature") && v.get("geometry").is_some();
    let (features, single): (Vec<Value>, bool) = match geojson.get("type").and_then(Value::as_str) {
        Some("Feature") if is_feature(&geojson) => (vec![geojson], true),
        Some("FeatureCollection") => match geojson.get("features").and_then(Value::as_array) {
            Some(list) if list.iter().all(is_feature) => (list.clone(), false),
            _ => return bad_body(),
        },
        _ => return bad_body(),
    };
    let req = StatsRequest {
        query,
        size,
        opts,
        cover_scale,
        expression: index_math.or_else(|| query_value(pairs, "expression")),
        single: index_math.is_some() || algorithm.as_ref().is_some_and(BandAlgorithm::float32),
        algorithm,
        unscale: pq.unscale,
        nodata: pq.nodata,
        mercator,
    };
    // The geometries as they were posted, where the body was in another system than degrees.
    let posted: Vec<Value> = match as_posted.and_then(|b| serde_json::from_slice::<Value>(b).ok()) {
        Some(mut doc) => match doc.get_mut("features").and_then(Value::as_array_mut) {
            Some(list) => list.iter_mut().map(|f| f["geometry"].take()).collect(),
            None => vec![doc["geometry"].take()],
        },
        None => Vec::new(),
    };
    let mut out = Vec::with_capacity(features.len());
    for (nth, feature) in features.iter().enumerate() {
        let (stats, used) = match feature_statistics(ctx, pg, hash, feature, &req).await {
            Ok(Some(found)) => found,
            Ok(None) => return no_content(),
            Err(r) => return r,
        };
        // TiTiler's order: the statistics, the feature's own properties, the items used.
        let mut properties = serde_json::Map::new();
        properties.insert("statistics".into(), stats);
        for (k, v) in feature.get("properties").and_then(Value::as_object).into_iter().flatten() {
            if k != "statistics" && k != "used_assets" {
                properties.insert(k.clone(), v.clone());
            }
        }
        // An item's statistics are of the item: titiler-pgstac lists no items there.
        if ctx.only.is_none() {
            properties.insert("used_assets".into(), json!(used));
        }
        let mut one = serde_json::Map::new();
        one.insert("type".into(), json!("Feature"));
        one.insert("geometry".into(), posted.get(nth).unwrap_or(&feature["geometry"]).clone());
        one.insert("properties".into(), Value::Object(properties));
        for key in ["id", "bbox"] {
            if let Some(v) = feature.get(key).filter(|v| !v.is_null()) {
                one.insert(key.into(), v.clone());
            }
        }
        out.push(Value::Object(one));
    }
    let doc = if single {
        out.pop().unwrap_or(Value::Null)
    } else {
        json!({ "type": "FeatureCollection", "features": out })
    };
    let mut resp = json_resp(StatusCode::OK, &doc);
    resp.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static("application/geo+json"));
    resp
}

/// Turn the stacked band values of a point into the results of `expression`, with the
/// names and descriptions TiTiler gives them.
#[allow(clippy::type_complexity)]
fn point_expression(
    expression: &str,
    values: &[Value],
    descriptions: &[String],
) -> Result<(Vec<Value>, Vec<String>, Vec<String>), String> {
    let numbers: Vec<Option<f64>> = values.iter().map(Value::as_f64).collect();
    let (results, described) = super::point_expr::apply(expression, &numbers, descriptions)?;
    let names = (1..=results.len()).map(|i| format!("b{i}")).collect();
    let values = results.into_iter().map(|v| v.map_or(Value::Null, |v| json!(v))).collect();
    Ok((values, names, described))
}

/// Each asset's own band math (`assets=red|expression=b1*2`) over an item's point, which the
/// `/stac` handler answers with every band of such an asset: the bands of each asset are
/// the run of values its description names (`{asset}_b{k}`, or the asset under
/// `asset_as_band`), worked out and described as the search routes do (`asset_bands`).
fn point_asset_math(doc: &mut Value, pairs: &[(String, String)]) -> Result<(), String> {
    if !asset_math_asked(pairs) {
        return Ok(());
    }
    let Ok(specs) = parse_assets(pairs) else { return Ok(()) };
    if specs.iter().all(|s| s.expression.is_none()) {
        return Ok(());
    }
    let chosen = entry_selections(pairs);
    let as_band = query_value(pairs, "asset_as_band").and_then(parse_bool).unwrap_or(false);
    let values: Vec<f64> =
        doc["values"].as_array().into_iter().flatten().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect();
    let descriptions: Vec<String> =
        doc["band_descriptions"].as_array().into_iter().flatten().map(|d| d.as_str().unwrap_or_default().to_string()).collect();
    if values.len() != descriptions.len() || chosen.len() != specs.len() {
        return Ok(());
    }
    // `{asset}_b{k}`: the band of the asset a description names.
    let band_of = |name: &str, d: &str| -> Option<usize> {
        let k = d.strip_prefix(name)?.strip_prefix("_b")?;
        (!k.is_empty() && k.bytes().all(|b| b.is_ascii_digit())).then(|| k.parse().ok()).flatten()
    };
    let (mut out, mut described) = (Vec::new(), Vec::new());
    let mut at = 0;
    for (spec, chosen) in specs.iter().zip(chosen) {
        if spec.name == ":all:" {
            return Err("':all:' cannot be used with per-asset expressions".into());
        }
        // Every asset is read whole (`point_asset_math_reads`): its run is its bands from the
        // first, so an asset asked for twice is read twice.
        let n = descriptions[at..].iter().enumerate().take_while(|(i, d)| band_of(&spec.name, d) == Some(i + 1)).count();
        if n == 0 {
            return Err(format!("could not match bands for asset '{}' in point response", spec.name));
        }
        let bands = &values[at..at + n];
        at += n;
        // An asset's band math is over all of its bands; a plain asset's choice of bands is
        // taken from what was read.
        let spec = AssetSpec { indexes: if spec.expression.is_some() { None } else { chosen }, ..spec.clone() };
        let (plan, names) = asset_bands(&spec, n, as_band)?;
        let mut results = Vec::new();
        plan.pixel(bands, &mut results);
        out.extend(results);
        described.extend(names);
    }
    if at != values.len() {
        return Err("point response bands did not match requested assets".into());
    }
    doc["values"] = json!(out.iter().map(|v| if v.is_finite() { json!(v) } else { Value::Null }).collect::<Vec<_>>());
    doc["band_names"] = json!((1..=out.len()).map(|i| format!("b{i}")).collect::<Vec<_>>());
    doc["band_descriptions"] = json!(described);
    Ok(())
}

/// What an item's point with an asset's own band math gives the `/stac` handler, which reads
/// the bands for [`point_asset_math`] to work out: every band of every asset, without the
/// request's `bidx`; [`point_asset_math`] applies each entry's own choice of bands (`|bidx=`,
/// `|indexes=`, `asset_bidx=`), and none to an asset with band math, as titiler-pgstac reads
/// it. Only the assets of `assets=` are read, as titiler-pgstac has no `asset_bidx` to add
/// others with.
pub(super) fn point_asset_math_reads(pq: &mut ParsedQuery, raw_query: &str) {
    let Ok(specs) = parse_assets(&query_pairs(raw_query)) else { return };
    // Every asset is read whole; each entry's choice of bands is applied afterwards, so that
    // an asset asked for twice is read for each time as asked.
    pq.asset_bidx.clear();
    pq.assets = specs.into_iter().map(|s| s.name).collect();
    pq.bidx.clear();
    pq.asset_exprs.clear();
    pq.forget_asset_entries();
    pq.asset_as_band = false;
}

/// How many items of a point query are read at once.
const POINT_ITEMS_IN_FLIGHT: usize = 8;

/// `GET {base}/point/{x},{y}`: the pixel values of every item under a point, in the
/// search's order, each as `{name, values, band_names, band_descriptions}`.
async fn point_values(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    hash: &str,
    coord: &str,
    pq: &ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    let (x, y) = match point_coords(coord) {
        Ok(xy) => xy,
        Err(resp) => return resp,
    };
    if pq.assets.is_empty() {
        return missing_query_param("assets");
    }
    if let Some(resp) = point_query_error(pairs) {
        return resp;
    }
    if let Some(e) = pq.errors.iter().find(|e| !own_error(e)) {
        return parse_error(e);
    }
    // A point is one pixel whatever the method; one that is not a method is still refused.
    if let Err(resp) = parse_resample(pairs, "reproject") {
        return resp;
    }
    let specs = match parse_assets(pairs) {
        Ok(specs) if specs.is_empty() => return missing_query_param("assets"),
        Ok(specs) => specs,
        Err(r) => return r,
    };
    let as_band = match query_value(pairs, "asset_as_band").map(parse_bool) {
        None => false,
        Some(Some(flag)) => flag,
        Some(None) => return invalid_param("query", "asset_as_band", "Input should be a valid boolean"),
    };
    let expression = query_value(pairs, "expression").map(str::to_string);
    let (lon, lat) = if pq.coord_crs_3857 {
        mercator_to_lonlat_deg(x, y)
    } else {
        if let Err(r) = check_coord_crs(pq, pairs) {
            return r;
        }
        (x, y)
    };
    let mut limits = match parse_limits(pairs) {
        Ok(l) => l,
        Err(r) => return r,
    };
    // A point is never "covered", so every item under it is wanted.
    limits.exitwhenfull = false;
    limits.skipcovered = false;
    let point = json!({ "type": "Point", "coordinates": [lon, lat] });
    let lookup = match parse_lookup(pairs) {
        Ok(l) => l,
        Err(r) => return r,
    };
    let names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
    let items = match pg.items_for_pixels(&point, hash, limits, &names, None, lookup).await {
        Ok(items) => items,
        Err(e) => return db_error(&e),
    };
    if items.is_empty() {
        return no_content();
    }

    // Resolve every asset before reading anything, so a bad name fails the request at once.
    let mut jobs: Vec<(String, Vec<(AssetSpec, String)>)> = Vec::with_capacity(items.len());
    for item in items.iter() {
        let hrefs = match asset_hrefs(item, specs.iter().map(|spec| spec.name.as_str())) {
            Ok(hrefs) => specs.iter().cloned().zip(hrefs).collect(),
            Err(resp) => return resp,
        };
        jobs.push((item_name(item), hrefs));
    }

    let gate = Arc::new(tokio::sync::Semaphore::new(POINT_ITEMS_IN_FLIGHT));
    let mut set = tokio::task::JoinSet::new();
    let (unscale, nodata) = (pq.unscale, pq.nodata);
    for (idx, (name, hrefs)) in jobs.into_iter().enumerate() {
        let engine = ctx.engine.clone();
        let gate = gate.clone();
        let expression = expression.clone();
        set.spawn(async move {
            let _permit = gate.acquire_owned().await;
            let mut values = Vec::new();
            let mut descriptions = Vec::new();
            for (spec, href) in &hrefs {
                let cog = engine.store.open_uri(href).await.map(|c| engine.store.with_nodata(&c, nodata))?;
                let pt = match query_point(&engine, &cog, lon, lat, unscale, nodata).await {
                    Ok(pt) => pt,
                    // The item's footprint holds the point but this raster does not.
                    Err(EngineError::NotFound(_)) => return Ok((idx, None)),
                    Err(e) => return Err(e),
                };
                // Every band of the asset under the point, then what the asset's options
                // make of them.
                let bands: Vec<f64> = pt["values"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
                    .collect();
                let (plan, described) =
                    asset_bands(spec, bands.len(), as_band).map_err(EngineError::BadRequest)?;
                let mut out = Vec::new();
                plan.pixel(&bands, &mut out);
                values.extend(out.into_iter().map(|v| if v.is_nan() { Value::Null } else { json!(v) }));
                descriptions.extend(described);
            }
            let mut band_names: Vec<String> = (1..=values.len()).map(|i| format!("b{i}")).collect();
            if let Some(expression) = &expression {
                (values, band_names, descriptions) =
                    point_expression(expression, &values, &descriptions)
                        .map_err(EngineError::BadRequest)?;
            }
            let entry = json!({
                "name": name,
                "values": values,
                "band_names": band_names,
                "band_descriptions": descriptions,
            });
            Ok::<_, EngineError>((idx, Some(entry)))
        });
    }
    let mut ordered: Vec<Option<Value>> = vec![None; items.len()];
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok((idx, entry))) => ordered[idx] = entry,
            Ok(Err(e)) => return engine_error(&e),
            Err(e) => {
                tracing::warn!(error = %e, "pgSTAC point task failed");
                return detail(StatusCode::INTERNAL_SERVER_ERROR, "point query failed");
            }
        }
    }
    let assets: Vec<Value> = ordered.into_iter().flatten().collect();
    if assets.is_empty() {
        return no_content();
    }
    json_resp(StatusCode::OK, &json!({ "coordinates": [x, y], "assets": assets }))
}

#[allow(clippy::too_many_arguments)]
/// `normalizedIndex` over several assets is rio-tiler's `(b2 - b1) / (b2 + b1)` of the first
/// two bands: band math, which planes read as TiTiler's pixels, where the renderer reads one
/// raster's bands only. A picture is drawn from -1 to 1 unless `rescale` says otherwise, as the
/// renderer draws the algorithm. (TiTiler, without `rescale`, paints every pixel 127.) An array
/// is the algorithm's values as they are, which TiTiler writes as float32: `true` says so, for
/// band math that would otherwise be doubles. Bare arrays are left to the algorithm.
fn normalized_index_as_band_math(pq: &mut ParsedQuery, ext: Option<&str>) -> bool {
    let asked = ext.or(pq.format_param.as_deref());
    if pq.algorithm.as_deref() != Some("normalizedindex")
        || pq.expr_raw.is_some()
        || !pq.asset_exprs.is_empty()
        || pq.assets.len() < 2
        || matches!(asked, Some("bin" | "raw" | "f32"))
    {
        return false;
    }
    pq.algorithm = None;
    pq.expr_raw = Some("(b2-b1)/(b2+b1)".into());
    if pq.rescale.is_empty() && !matches!(asked, Some("npy" | "tif" | "tiff")) {
        pq.rescale.push((-1.0, 1.0));
    }
    true
}

/// Whether samples of this kind can hold `named`. GDAL refuses a no-data value they
/// cannot; a fraction among whole numbers it takes, and no sample is ever equal to it.
fn nodata_fits(named: f32, kind: cogzilla_raster::SampleType) -> bool {
    use cogzilla_raster::SampleType as T;
    let named = f64::from(named);
    let within = |lo: f64, hi: f64| named >= lo && named <= hi;
    match kind {
        T::Uint8 => within(0.0, f64::from(u8::MAX)),
        T::Int8 => within(f64::from(i8::MIN), f64::from(i8::MAX)),
        T::Uint16 => within(0.0, f64::from(u16::MAX)),
        T::Int16 => within(f64::from(i16::MIN), f64::from(i16::MAX)),
        T::Uint32 => within(0.0, f64::from(u32::MAX)),
        T::Int32 => within(f64::from(i32::MIN), f64::from(i32::MAX)),
        _ => true,
    }
}

/// A number as Python writes it, which is how the reference's headers carry bounds: the
/// shortest digits that read back the same, in powers of ten below 1e-4 and from 1e16.
fn python_float(c: f64) -> String {
    let c = c + 0.0;
    let short = format!("{c:e}");
    let Some((digits, power)) = short.split_once('e') else { return short };
    match power.parse::<i32>() {
        Ok(power) if c.is_finite() && !(-4..16).contains(&power) => {
            format!("{digits}e{}{:02}", if power < 0 { '-' } else { '+' }, power.abs())
        }
        _ => format!("{c:?}"),
    }
}

/// A picture's answer with where the picture lies, as TiTiler says it: `Content-Bbox`,
/// the bounds as Python writes floats, and `Content-Crs`, the system they are in.
fn placed(resp: Resp, bounds: [f64; 4], epsg: u16) -> Resp {
    placed_as(resp, bounds, epsg, None)
}

/// [`placed`], with the system named as the request spelled it: degrees asked for as
/// `CRS84` are TiTiler's `OGC/0/CRS84`, not `EPSG/0/4326`, though the picture is the same.
fn placed_as(mut resp: Resp, bounds: [f64; 4], epsg: u16, spelled: Option<&str>) -> Resp {
    if resp.status() == StatusCode::OK {
        let bbox = bounds.map(python_float).join(",");
        let crs84 = epsg == 4326 && spelled.is_some_and(|s| s.to_ascii_lowercase().contains("crs84"));
        let crs = if crs84 {
            "<http://www.opengis.net/def/crs/OGC/0/CRS84>".to_string()
        } else {
            format!("<http://www.opengis.net/def/crs/EPSG/0/{epsg}>")
        };
        if let (Ok(bbox), Ok(crs)) = (HeaderValue::from_str(&bbox), HeaderValue::from_str(&crs)) {
            resp.headers_mut().insert("content-bbox", bbox);
            resp.headers_mut().insert("content-crs", crs);
        }
    }
    resp
}

/// The NumPy array these routes write, as a GeoTIFF of the same values in the same type:
/// what TiTiler's `.tif` is, the array with a place on the ground. One strip to a band,
/// which is the order the array is already in; with `mask` the last band is the alpha.
/// `None` when the bytes are not such an array.
fn npy_as_geotiff(npy: &[u8], bounds: [f64; 4], epsg: u16, mask: bool) -> Option<Bytes> {
    let head_len = usize::from(u16::from_le_bytes([*npy.get(8)?, *npy.get(9)?]));
    let head = std::str::from_utf8(npy.get(10..10 + head_len)?).ok()?;
    let data = npy.get(10 + head_len..)?;
    let descr = head.split("'descr': '").nth(1)?.split('\'').next()?;
    use crate::geotiff::{GeoTiffLayout, GeoTiffPhotometric, GeoTiffSample, GeoTiffSpec, write_geotiff};
    let sample = match descr {
        "|u1" => GeoTiffSample::U8,
        "|i1" => GeoTiffSample::I8,
        "<u2" => GeoTiffSample::U16,
        "<i2" => GeoTiffSample::I16,
        "<u4" => GeoTiffSample::U32,
        "<i4" => GeoTiffSample::I32,
        "<f4" => GeoTiffSample::F32,
        "<f8" => GeoTiffSample::F64,
        _ => return None,
    };
    let mut shape = head.split("'shape': (").nth(1)?.split(')').next()?.split(',').filter_map(|n| n.trim().parse::<usize>().ok());
    let (bands, height, width) = (shape.next()?, shape.next()?, shape.next()?);
    let strip = width.checked_mul(height)?.checked_mul(usize::from(sample.bits() / 8))?;
    if bands == 0 || bands > 64 || strip == 0 || data.len() != strip.checked_mul(bands)? || u32::try_from(data.len()).is_err() {
        return None;
    }
    let shown = bands - usize::from(mask && bands > 1);
    // Three bands of bytes and no more are a colour picture, as GDAL writes them; otherwise
    // the first is grey and the rest are extra, the mask an alpha that is not multiplied in.
    let colour = shown == 3 && descr == "|u1";
    let extra: Vec<u16> = (if colour { 3 } else { 1 }..bands).map(|band| if mask && band + 1 == bands { 2 } else { 0 }).collect();
    // The array is band after band already: one strip a band, written as it is.
    let spec = GeoTiffSpec {
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        bands: bands as u16,
        sample,
        sample_format_tag: true,
        layout: GeoTiffLayout::Planar,
        photometric: if colour { GeoTiffPhotometric::Rgb } else { GeoTiffPhotometric::MinIsBlack },
        extra_samples: &extra,
        nodata: None,
        bbox: bounds,
        epsg,
    };
    Some(write_geotiff(&spec, |out| out.extend_from_slice(data)))
}

/// [`npy_as_geotiff`] of a route's answer when a GeoTIFF was asked for: the one place a
/// tile, a `bbox` and a `feature` become one.
fn as_geotiff(made: (Bytes, &'static str), tiff: bool, bounds: [f64; 4], epsg: u16, mask: bool) -> (Bytes, &'static str) {
    if tiff
        && made.1 != super::GEOTIFF_CONTENT_TYPE
        && let Some(body) = npy_as_geotiff(&made.0, bounds, epsg, mask)
    {
        return (body, super::GEOTIFF_CONTENT_TYPE);
    }
    made
}

/// An RGBA picture as the response's bytes and media type: the colour formula, then
/// the format asked for or, with none, TiTiler's choice of JPEG when every pixel is there
/// and PNG when not.
fn encode_picture(
    mut rgba: Vec<u8>,
    width: usize,
    height: usize,
    formula: Option<&cogzilla_kernel::ColorFormula>,
    asked: Option<(OutputFormat, &'static str)>,
    jpeg_quality: u8,
    return_mask: bool,
    grey: bool,
) -> Result<(Bytes, &'static str), EngineError> {
    if let Some(formula) = formula {
        formula.apply_buffer_in_place(&mut rgba, 4);
        // One band is the formula's red, and the picture is of that band alone.
        if grey {
            rgba.chunks_exact_mut(4).for_each(|p| (p[1], p[2]) = (p[0], p[0]));
        }
    }
    let (format, content_type) = match asked {
        Some(asked) => asked,
        None => {
            let opaque = rgba.len() == width * height * 4 && rgba.chunks_exact(4).all(|p| p[3] == 255);
            parse_ext_format(if opaque { "jpg" } else { "png" }, None, jpeg_quality)?
        }
    };
    // A PNG whose every pixel is there needs no alpha: the same picture, and smaller and
    // quicker to write, which is what the renderer does for its own tiles.
    let alpha = return_mask
        && !(matches!(format, OutputFormat::Png) && rgba.chunks_exact(4).all(|p| p[3] == 255));
    // One band is a grey picture, as TiTiler writes it: a PNG of one channel, or two
    // with the mask, and a JPEG of one component. WebP has no such picture.
    if grey && rgba.len() == width * height * 4 {
        match format {
            OutputFormat::Png => return Ok((grey_png(&rgba, width, height, alpha), content_type)),
            OutputFormat::Jpeg { quality } => {
                let luma: Vec<u8> = rgba.chunks_exact(4).map(|p| if p[3] == 0 { 0 } else { p[0] }).collect();
                return Ok((cogzilla_kernel::jpeg_fast::encode_jpeg_gray(width as u32, height as u32, &luma, quality), content_type));
            }
            _ => {}
        }
    }
    Ok((encode_rgba_output_colour(width, height, &rgba, format, alpha), content_type))
}

/// A grey PNG of an RGBA picture whose red is the grey: one channel, or two with `alpha`.
/// Without it a pixel that is not there is black, as it is in the RGB the renderer writes.
/// Bands of 16-bit whole numbers as a PNG of 16 bits a sample, with `alpha` a last sample
/// that is the largest number where any band has a value and zero elsewhere; four samples
/// at most. What the samples are taken for goes by how many there are, as in GDAL's PNG
/// driver: one is grey, two grey and alpha, three colour, four colour and alpha, so that
/// two bands and their mask make a colour picture. A value that is not there is zero.
pub(super) fn png_of_16_bits(values: &[f32], bands: usize, width: usize, height: usize, alpha: bool) -> Bytes {
    let pixels = width * height;
    let samples = bands + usize::from(alpha);
    let mut rows = Vec::with_capacity(height * (1 + width * samples * 2));
    for row in 0..height {
        // No filter: the samples as they are. (PNG's Sub filter was tried: no smaller on
        // Sentinel-2 and 5% slower.)
        rows.push(0);
        for i in row * width..(row + 1) * width {
            let mut there = false;
            for band in 0..bands {
                let v = values.get(band * pixels + i).copied().unwrap_or(f32::NAN);
                there |= !v.is_nan();
                let sample = if v.is_nan() { 0u16 } else { v.round().clamp(0.0, 65535.0) as u16 };
                rows.extend_from_slice(&sample.to_be_bytes());
            }
            if alpha {
                rows.extend_from_slice(&(if there { u16::MAX } else { 0 }).to_be_bytes());
            }
        }
    }
    let colour_type = match samples {
        1 => 0,
        2 => 4,
        3 => 2,
        _ => 6,
    };
    cogzilla_kernel::png_fast::encode_filtered_png_depth(width as u32, height as u32, colour_type, 16, &rows)
}

fn grey_png(rgba: &[u8], width: usize, height: usize, alpha: bool) -> Bytes {
    let channels = if alpha { 2 } else { 1 };
    let stride = 1 + width * channels;
    let mut filtered = vec![0u8; height * stride];
    for (row, out) in rgba.chunks_exact(width * 4).zip(filtered.chunks_exact_mut(stride)) {
        // PNG's Sub filter: each byte less the same channel of the pixel before.
        out[0] = 1;
        let mut before = [0u8; 2];
        for (p, o) in row.chunks_exact(4).zip(out[1..].chunks_exact_mut(channels)) {
            let now = [if p[3] == 0 && !alpha { 0 } else { p[0] }, p[3]];
            for c in 0..channels {
                o[c] = now[c].wrapping_sub(before[c]);
            }
            before = now;
        }
    }
    cogzilla_kernel::png_fast::encode_filtered_png(width as u32, height as u32, if alpha { 4 } else { 0 }, &filtered)
}

/// What the router does with `/collections/{c}/items/{i}/...`.
pub(super) enum ItemRoute {
    /// Run the `/stac` handlers over this item, then pass the answer through [`item_fixup`].
    Stac(Arc<StacItem>, Option<ItemExtent>),
    /// Already answered.
    Done(Resp),
}

/// Where an item lies and the zooms it is worth tiling at, as rio-tiler's STAC reader
/// derives them: from the projection extension when the item has it, else from its bbox.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ItemExtent {
    wgs84: [f64; 4],
    /// The same extent in Web Mercator, when every edge can be placed there.
    mercator: Option<[f64; 4]>,
    minzoom: i64,
    maxzoom: i64,
}

/// A raster grid in its own coordinate system: `[west, south, east, north]` and its size.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ProjGrid {
    bounds: [f64; 4],
    width: f64,
    height: f64,
}

/// `proj:code`/`proj:epsg` of an item's properties or of one asset, as one comparable string.
fn proj_crs_string(obj: &Value) -> Option<String> {
    if let Some(code) = obj.get("proj:code").and_then(Value::as_str) {
        return Some(code.to_string());
    }
    obj.get("proj:epsg").and_then(Value::as_i64).map(|e| format!("EPSG:{e}"))
}

fn proj_grid_of(obj: &Value) -> Option<ProjGrid> {
    let t: Vec<f64> = obj
        .get("proj:transform")?
        .as_array()?
        .iter()
        .map(Value::as_f64)
        .collect::<Option<_>>()?;
    let shape: Vec<f64> =
        obj.get("proj:shape")?.as_array()?.iter().map(Value::as_f64).collect::<Option<_>>()?;
    if t.len() < 6 || shape.len() != 2 || t[0] == 0.0 || t[4] == 0.0 {
        return None;
    }
    let (height, width) = (shape[0], shape[1]);
    let (x0, y0) = (t[2], t[5]);
    let (x1, y1) = (x0 + t[0] * width, y0 + t[4] * height);
    Some(ProjGrid { bounds: [x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)], width, height })
}

/// The grid rio-tiler gives an item: the union of its assets' grids at the finest asset's
/// resolution, or the item-level projection fields. `None` when the item does not say.
fn item_proj_grid(item: &Value, readable: &[String]) -> Option<ProjGrid> {
    let props = item.get("properties")?;
    let item_crs = proj_crs_string(props);
    let mut grids: Vec<(ProjGrid, f64, f64)> = Vec::new();
    let mut crs_strings: Vec<String> = item_crs.iter().cloned().collect();
    for name in readable {
        let asset = &item["assets"][name];
        if let (Some(grid), Some(t)) = (proj_grid_of(asset), asset.get("proj:transform")) {
            crs_strings.push(proj_crs_string(asset).or_else(|| item_crs.clone())?);
            let xres = t[0].as_f64().unwrap_or(0.0);
            let yres = t[4].as_f64().unwrap_or(0.0);
            grids.push((grid, xres, yres));
        }
    }
    if !grids.is_empty() && crs_strings.iter().all(|c| c == &crs_strings[0]) {
        let fold = |i: usize, f: fn(f64, f64) -> f64, init: f64| {
            grids.iter().fold(init, |acc, (g, _, _)| f(acc, g.bounds[i]))
        };
        let bounds = [
            fold(0, f64::min, f64::INFINITY),
            fold(1, f64::min, f64::INFINITY),
            fold(2, f64::max, f64::NEG_INFINITY),
            fold(3, f64::max, f64::NEG_INFINITY),
        ];
        let (_, xres, yres) = grids.iter().copied().min_by(|a, b| {
            (a.1.abs(), a.2.abs()).partial_cmp(&(b.1.abs(), b.2.abs())).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        let height = ((bounds[3] - bounds[1]) / yres).floor().abs();
        let width = ((bounds[2] - bounds[0]) / xres).floor().abs();
        return Some(ProjGrid { bounds, width, height });
    }
    item_crs.as_ref()?;
    proj_grid_of(props)
}

/// Twenty-one points along each edge of a box.
fn edge_points(b: [f64; 4]) -> Vec<(f64, f64)> {
    let mut pts = Vec::with_capacity(84);
    for i in 0..=20 {
        let f = i as f64 / 20.0;
        let (x, y) = (b[0] + f * (b[2] - b[0]), b[1] + f * (b[3] - b[1]));
        pts.extend([(x, b[3]), (x, b[1]), (b[0], y), (b[2], y)]);
    }
    pts
}

/// Extent and zoom range of a grid, following rio-tiler: bounds from twenty-one points per
/// edge, zooms from the size the grid would have once warped to Web Mercator.
fn grid_extent(crs: &CrsModel, grid: ProjGrid) -> ItemExtent {
    let lonlat: Vec<(f64, f64)> =
        edge_points(grid.bounds).iter().map(|&(x, y)| crs.to_lonlat_deg(x, y)).collect();
    let span = |pts: &[(f64, f64)]| -> Option<[f64; 4]> {
        if pts.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
            return None;
        }
        Some(pts.iter().fold(
            [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
            |a, p| [a[0].min(p.0), a[1].min(p.1), a[2].max(p.0), a[3].max(p.1)],
        ))
    };
    let wgs84 = span(&lonlat).unwrap_or([-180.0, -90.0, 180.0, 90.0]);
    let merc_pts: Vec<(f64, f64)> = if matches!(crs, CrsModel::Epsg3857) {
        edge_points(grid.bounds)
    } else {
        lonlat.iter().map(|&(lon, lat)| lonlat_deg_to_mercator(lon, lat)).collect()
    };
    let mercator = span(&merc_pts);
    // The same rule `/cog` and `/stac` TileJSON use, on the item's grid.
    let pixel_to_crs = cogzilla_raster::Affine2D {
        a: (grid.bounds[2] - grid.bounds[0]) / grid.width,
        b: 0.0,
        c: grid.bounds[0],
        d: 0.0,
        e: -(grid.bounds[3] - grid.bounds[1]) / grid.height,
        f: grid.bounds[3],
    };
    let (minzoom, maxzoom) =
        crs.web_mercator_zoom_range(&pixel_to_crs, grid.width as u32, grid.height as u32, 1.0);
    ItemExtent { wgs84, mercator, minzoom: i64::from(minzoom), maxzoom: i64::from(maxzoom) }
}

/// The extent of an item. The coordinate system comes from one of its rasters: the first
/// asset asked for, or the first that can be read.
async fn item_extent(
    ctx: &Ctx<'_>,
    doc: &Value,
    item: &StacItem,
    readable: &[String],
    pq: &ParsedQuery,
) -> Result<ItemExtent, Resp> {
    let Some(grid) = item_proj_grid(doc, readable) else {
        // No projection metadata: rio-tiler falls back to the item's bbox and every zoom.
        let b = item.bbox.unwrap_or([-180.0, -90.0, 180.0, 90.0]);
        let (x0, y0) = lonlat_deg_to_mercator(b[0], b[1]);
        let (x1, y1) = lonlat_deg_to_mercator(b[2], b[3]);
        let mercator = [x0, y0, x1, y1].iter().all(|v| v.is_finite()).then_some([x0, y0, x1, y1]);
        return Ok(ItemExtent { wgs84: b, mercator, minzoom: 0, maxzoom: 24 });
    };
    let wanted = pq.assets.first().map(|a| a.split('|').next().unwrap_or(a).to_string());
    let href = match &wanted {
        Some(name) => item.assets.get(name),
        None => readable.iter().find_map(|name| item.assets.get(name)),
    };
    let Some(href) = href else {
        return Err(detail(StatusCode::NOT_FOUND, "the item has no readable raster asset"));
    };
    let crs = async {
        let cog = ctx.engine.store.open_uri(href).await?;
        ctx.engine.get_or_build_crs(&cog)
    }
    .await
    .map_err(|e| engine_error(&e))?;
    Ok(grid_extent(&crs, grid))
}

/// An item tile asked for without an extension: a JPEG when every pixel is valid and a
/// PNG when some are not, as TiTiler chooses.
pub(super) async fn item_tile_auto_format(
    engine: &TileEngine,
    cogs: &[Arc<cogzilla_io::OpenedCog>],
    mut treq: crate::engine::TileRequest,
    pq: &ParsedQuery,
) -> Result<(Bytes, &'static str), EngineError> {
    let formula = pq
        .color_formula
        .as_deref()
        .map(cogzilla_kernel::ColorFormula::parse)
        .transpose()
        .map_err(EngineError::BadRequest)?;
    treq.format = OutputFormat::RawRgba;
    let mut rgba = engine.render_multi_cog_tile(cogs, &treq).await?.to_vec();
    if let Some(formula) = formula {
        formula.apply_buffer_in_place(&mut rgba, 4);
    }
    let dim = treq.tile_size as usize;
    let opaque = rgba.len() == dim * dim * 4 && rgba.chunks_exact(4).all(|p| p[3] == 255);
    let (format, content_type) =
        parse_ext_format(if opaque { "jpg" } else { "png" }, None, pq.jpeg_quality)?;
    Ok((encode_rgba_output_colour(dim, dim, &rgba, format, pq.return_mask), content_type))
}

/// The place a tile already says it lies, written as titiler-pgstac writes it: degrees
/// are `EPSG/0/4326` (the shared routes say `OGC/0/CRS84`), and a number is a Python float.
fn respelled(resp: Resp) -> Resp {
    let read = |name: &str| resp.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let (Some(bbox), Some(crs)) = (read("content-bbox"), read("content-crs")) else {
        return resp;
    };
    let epsg = if crs.contains("OGC/0/CRS84") || crs.contains("EPSG/0/4326") {
        4326
    } else if let Some(rest) = crs.split("EPSG/0/").nth(1) {
        let digits = rest.trim_end_matches(['>', ' ']);
        match digits.parse::<u16>() {
            Ok(code) => code,
            Err(_) => return resp,
        }
    } else {
        return resp;
    };
    let numbers: Vec<f64> = bbox.split(',').filter_map(|n| n.trim().parse().ok()).collect();
    match numbers[..] {
        [west, south, east, north] => placed(resp, [west, south, east, north], epsg),
        _ => resp,
    }
}

/// An item's tile or `bbox` picture with where it lies, as titiler-pgstac's item routes
/// say it: the tile's bounds in Web Mercator, or the box in the system it was drawn in
/// (for any supported CRS). A preview or a posted shape is placed by [`item_drawn_placed`].
pub(super) fn item_placed(resp: Resp, sub: &[&str], raw_query: &str) -> Resp {
    if resp.status() != StatusCode::OK {
        return resp;
    }
    // A parameter as the handlers that drew the picture read it: by its key, percent-decoded
    // as Starlette decodes it, and the last one given.
    let asked = |key: &str| {
        raw_query
            .split('&')
            .rev()
            .filter_map(|pair| pair.split_once('='))
            .find(|(k, _)| super::query_key(k) == key)
            .map(|(_, v)| super::percent_decode(v))
    };
    match sub {
        ["tiles", "WebMercatorQuad", z, x, y] => {
            let (Ok(z), Ok(x), Some((y, _, _))) = (z.parse::<u8>(), x.parse::<u32>(), super::parse_y_scale_ext(y)) else {
                return resp;
            };
            if z > 30 || u64::from(x.max(y)) >= 1u64 << z {
                return resp;
            }
            // A buffered tile reaches past its bounds by an amount that depends on its
            // size: it goes without the headers sooner than with the wrong ones.
            if asked("buffer").is_some_and(|b| b.parse::<f64>() != Ok(0.0)) {
                return respelled(resp);
            }
            let (west, south, east, north) = tile_bounds_3857(TileCoord { z, x, y });
            placed(resp, [west, south, east, north], 3857)
        }
        ["tiles", ..] => respelled(resp),
        ["bbox", seg] | ["bbox", seg, _] => {
            let Some((bbox, _)) = super::parse_bbox4(seg) else {
                return resp;
            };
            let parse_sys = |key: &str, default: (CrsModel, u16)| match asked(key)
                .as_deref()
                .map(super::parse_coord_crs_with_epsg)
            {
                None => Some(default),
                Some(Ok(pair)) => Some(pair),
                Some(Err(_)) => None,
            };
            let Some((given_crs, given_epsg)) = parse_sys("coord_crs", (CrsModel::Geographic, 4326)) else {
                return resp;
            };
            let Some((drawn_crs, drawn_epsg)) = parse_sys("dst_crs", (given_crs.clone(), given_epsg)) else {
                return resp;
            };
            let bounds = if given_epsg == drawn_epsg {
                bbox
            } else {
                let Ok(wgs_bbox) = super::bbox_to_wgs84(bbox, Some(&given_crs)) else {
                    return resp;
                };
                if drawn_epsg == 4326 {
                    wgs_bbox
                } else {
                    let Ok(reproj) = super::reproject_wgs84_bounds(wgs_bbox, &drawn_crs) else {
                        return resp;
                    };
                    reproj
                }
            };
            if bounds.iter().any(|c| !c.is_finite()) || bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
                return resp;
            }
            let spelled = asked("dst_crs").or_else(|| asked("coord_crs"));
            placed_as(resp, bounds, drawn_epsg, spelled.as_deref())
        }
        _ => resp,
    }
}

/// An item's preview or posted `feature` picture with where it lies, from what the `/stac`
/// handler that drew it worked out (the raster's own system for a preview): the place
/// [`item_placed`] cannot tell from the path, in titiler-pgstac's spelling.
pub(super) fn item_drawn_placed(resp: Resp, sub: &[&str], drawn: Option<([f64; 4], Option<u16>, bool)>) -> Resp {
    let picture = match sub {
        ["preview", ..] | ["feature", ..] => true,
        [one] => one.starts_with("preview.") || one.starts_with("feature."),
        _ => false,
    };
    match drawn {
        Some((bounds, Some(epsg), crs84)) if picture && bounds.iter().all(|c| c.is_finite()) => {
            placed_as(resp, bounds, epsg, crs84.then_some("CRS84"))
        }
        _ => resp,
    }
}

/// Band math on an item's statistics, as titiler-pgstac reads it: `b1..bn` over the bands
/// of the assets stacked in order, and never an asset's name (`Invalid band/asset name`).
/// Which bands there are is the stack's to say (`expand_stac_asset_bidx`).
pub(super) fn check_item_band_math_names(expression: &str) -> Result<(), EngineError> {
    Program::parse(expression, MAX_BANDS_PER_ITEM).map(|_| ()).map_err(EngineError::BadRequest)
}

/// The names titiler-pgstac gives an item's stacked bands, asset by asset: `{asset}_{part}`
/// for each result of an asset's own band math, `{asset}_b{k}` for each band read of a plain
/// asset (those its `|bidx=` picks, else all `bands` of it), and the asset itself under
/// `asset_as_band`. `options` are each asset's own band math and bands (`stac_entry_options`).
pub(super) fn item_stack_names(
    names: &[String],
    bands: &[u16],
    options: &[(Option<String>, Option<Vec<u16>>)],
    as_band: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    for ((name, &count), (own, chosen)) in names.iter().zip(bands).zip(options) {
        if as_band {
            out.push(name.clone());
            continue;
        }
        match own {
            Some(own) => out.extend(super::expression_parts(own).map(|part| format!("{name}_{part}"))),
            None => {
                let read: Vec<u16> = chosen.clone().unwrap_or_else(|| (1..=count).collect());
                out.extend(read.iter().map(|k| format!("{name}_b{k}")));
            }
        }
    }
    out
}

/// Make a `/stac` answer read the way titiler-pgstac's item routes do.
pub(super) fn item_fixup(
    sub: &[&str],
    extent: Option<&ItemExtent>,
    raw_query: &str,
    body: Bytes,
    content_type: &'static str,
) -> Result<(Bytes, &'static str), Resp> {
    // TiTiler names the media type after the extension that was asked for.
    if let ["tiles", .., last] = sub
        && last.ends_with(".jpg")
        && content_type == "image/jpeg"
    {
        return Ok((body, "image/jpg"));
    }
    let wants_json = matches!(sub, ["info" | "info.geojson"] | ["point", _] | [_, "tilejson.json"]);
    if !wants_json {
        return Ok((body, content_type));
    }
    let Ok(mut doc) = serde_json::from_slice::<Value>(&body) else {
        return Ok((body, content_type));
    };
    // Per-asset fields the `/stac` routes add and titiler-pgstac does not have.
    let strip = |assets: Option<&mut serde_json::Map<String, Value>>| {
        for info in assets.into_iter().flat_map(|m| m.values_mut()) {
            if let Some(info) = info.as_object_mut() {
                for key in ["minzoom", "maxzoom", "wgs84_bounds"] {
                    info.remove(key);
                }
            }
        }
    };
    match sub {
        ["info"] => strip(doc.as_object_mut()),
        ["info.geojson"] => {
            strip(doc.get_mut("properties").and_then(Value::as_object_mut));
            // The feature is the item's extent, not its footprint.
            if let Some(e) = extent {
                let [w, s, e_, n] = e.wgs84;
                doc["bbox"] = json!([w, s, e_, n]);
                doc["geometry"] = json!({
                    "type": "Polygon",
                    "coordinates": [[[w, s], [e_, s], [e_, n], [w, n], [w, s]]]
                });
            }
        }
        ["point", _] => {
            if let Some(names) = doc.get_mut("band_names").and_then(Value::as_array_mut) {
                for (i, name) in names.iter_mut().enumerate() {
                    *name = json!(format!("b{}", i + 1));
                }
            }
            // The `/stac` handler returns the bands as they are; band math is applied here,
            // each asset's own first.
            let pairs = query_pairs(raw_query);
            if let Err(message) = point_asset_math(&mut doc, &pairs) {
                return Err(detail(StatusCode::BAD_REQUEST, message));
            }
            if let Some(expression) = request_expression(&pairs) {
                let values = doc["values"].as_array().cloned().unwrap_or_default();
                let descriptions: Vec<String> = doc["band_descriptions"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|d| d.as_str().map(str::to_string))
                    .collect();
                match point_expression(expression, &values, &descriptions) {
                    Ok((values, names, described)) => {
                        doc["values"] = json!(values);
                        doc["band_names"] = json!(names);
                        doc["band_descriptions"] = json!(described);
                    }
                    Err(message) => return Err(detail(StatusCode::BAD_REQUEST, message)),
                }
            }
        }
        [_, "tilejson.json"] => {
            if let Some(e) = extent {
                let pairs = query_pairs(raw_query);
                doc["bounds"] = json!(e.wgs84);
                if query_value(&pairs, "minzoom").is_none() {
                    doc["minzoom"] = json!(e.minzoom);
                }
                if query_value(&pairs, "maxzoom").is_none() {
                    doc["maxzoom"] = json!(e.maxzoom);
                }
                let minzoom = doc["minzoom"].clone();
                doc["center"] =
                    json!([(e.wgs84[0] + e.wgs84[2]) / 2.0, (e.wgs84[1] + e.wgs84[3]) / 2.0, minzoom]);
            }
        }
        _ => {}
    }
    Ok((Bytes::from(doc.to_string()), content_type))
}

/// The asset media types rio-tiler reads by default. An asset with no type is taken too.
const READABLE_ASSET_TYPES: [&str; 12] = [
    "image/tiff; application=geotiff",
    "image/tiff; application=geotiff; profile=cloud-optimized",
    "image/tiff; profile=cloud-optimized; application=geotiff",
    "image/vnd.stac.geotiff; cloud-optimized=true",
    "image/tiff",
    "image/x.geotiff",
    "image/jp2",
    "application/x-hdf5",
    "application/x-hdf",
    "application/vnd+zarr",
    "application/x-netcdf",
    "application/netcdf",
];

/// The asset names titiler-pgstac offers for an item, in the order it lists them.
fn readable_assets(item: &Value) -> Vec<String> {
    let Some(assets) = item.get("assets").and_then(Value::as_object) else {
        return Vec::new();
    };
    let mut names: Vec<String> = assets
        .iter()
        .filter(|(_, a)| match a.get("type").and_then(Value::as_str) {
            Some(t) if !t.is_empty() => READABLE_ASSET_TYPES.contains(&t),
            _ => true,
        })
        .map(|(k, _)| k.clone())
        .collect();
    jsonb_key_order(&mut names);
    names
}

/// Python's `urlencode(obj, doseq=True)` over a JSON object, keys in jsonb order.
fn urlencode_doseq(obj: &serde_json::Map<String, Value>) -> String {
    let mut keys: Vec<String> = obj.keys().cloned().collect();
    jsonb_key_order(&mut keys);
    let mut out = Vec::new();
    for k in keys {
        match &obj[&k] {
            Value::Array(items) => {
                out.extend(items.iter().map(|i| format!("{}={}", quote_plus(&k), quote_plus(&py_str(i)))));
            }
            other => out.push(format!("{}={}", quote_plus(&k), quote_plus(&py_str(other)))),
        }
    }
    out.join("&")
}

/// One of an item's STAC renders, the way TiTiler presents it.
struct ItemRender {
    name: String,
    /// The render with `asset_bidx` and `asset_expression` folded into the asset names.
    params: serde_json::Map<String, Value>,
    /// Whether it can make a tile request: it has to name assets.
    valid: bool,
}

/// The renders in an item's properties, in jsonb key order.
fn item_renders(item: &Value) -> Vec<ItemRender> {
    let Some(renders) = item.pointer("/properties/renders").and_then(Value::as_object) else {
        return Vec::new();
    };
    let as_list = |v: &Value| match v {
        Value::Array(a) => a.clone(),
        other => vec![other.clone()],
    };
    let mut names: Vec<String> = renders.keys().cloned().collect();
    jsonb_key_order(&mut names);
    let mut out = Vec::new();
    for name in names {
        let Some(render) = renders[&name].as_object() else {
            continue;
        };
        let mut params = render.clone();
        let assets: Vec<String> = match params.get("assets") {
            Some(v) if py_truthy(v) => as_list(v).iter().map(py_str).collect(),
            _ => Vec::new(),
        };
        if !assets.is_empty() {
            let mut options: Vec<Vec<String>> = vec![Vec::new(); assets.len()];
            for (key, label) in [("asset_bidx", "bidx"), ("asset_expression", "expression")] {
                let Some(value) = params.remove(key) else {
                    continue;
                };
                for entry in as_list(&value) {
                    let entry = py_str(&entry);
                    if let Some((asset, opt)) = entry.split_once('|')
                        && let Some(i) = assets.iter().position(|a| a == asset)
                    {
                        options[i].push(format!("{label}={opt}"));
                    }
                }
            }
            let merged: Vec<Value> = assets
                .iter()
                .zip(&options)
                .map(|(a, o)| if o.is_empty() { json!(a) } else { json!(format!("{a}|{}", o.join("|"))) })
                .collect();
            params.insert("assets".into(), Value::Array(merged));
        }
        out.push(ItemRender { name, params, valid: !assets.is_empty() });
    }
    out
}

/// A render as `GET {base}/renders/{id}` returns it. TiTiler puts the item into every link
/// as `url=`, and for an item read from pgSTAC that is its Python name, `<Item id=...>`.
fn item_render_json(host: &str, base: &str, item_id: &str, render: &ItemRender) -> Value {
    let url = format!("url={}", quote_plus(&format!("<Item id={item_id}>")));
    let mut links = vec![json!({
        "href": format!("{host}{base}/renders/{}?{url}", render.name),
        "rel": "self",
        "type": "application/json",
        "title": format!("STAC Renders metadata for {}", render.name)
    })];
    if render.valid {
        let qs = urlencode_doseq(&render.params);
        links.push(json!({
            "href": format!("{host}{base}/%7BtileMatrixSetId%7D/tilejson.json?{url}&{qs}"),
            "rel": "tilesets-map",
            "templated": true,
            "title": format!("tilejson file for {}", render.name)
        }));
        links.push(json!({
            "href": format!("{host}{base}/WMTSCapabilities.xml?{url}&{qs}"),
            "rel": "tilesets-map",
            "templated": true,
            "title": format!("WMTS service for {}", render.name)
        }));
    }
    json!({ "valid": render.valid, "params": render.params, "links": links })
}

/// Routes of the item family that titiler-pgstac refuses without `assets`.
fn item_route_needs_assets(rest: &[&str]) -> bool {
    match rest {
        ["info" | "info.geojson" | "statistics" | "asset_statistics"] => true,
        ["point", _] | ["bbox", ..] | ["tiles", _, _, _, _] => true,
        [_, "tilejson.json" | "map.html"] => true,
        [first, ..] => {
            *first == "preview"
                || first.starts_with("preview.")
                || *first == "feature"
                || first.starts_with("feature.")
        }
        [] => false,
    }
}

/// An item's `preview/{width}x{height}.{format}` with a side of zero, as the
/// `preview.{format}` it stands for: a zero is a side left to follow the other, or
/// `max_size` when both are, as TiTiler has it. The side that is given goes into the query,
/// where it outranks `max_size` as it does in rio-tiler. Any other path is left alone.
pub(super) fn item_preview_by_query(rest: &[&str], pq: &mut ParsedQuery) -> Option<String> {
    let ["preview", dims] = rest else { return None };
    let (sides, ext) = dims.split_once('.')?;
    let (w, h) = sides.split_once('x')?;
    let (w, h) = (w.parse::<u32>().ok()?, h.parse::<u32>().ok()?);
    if (w != 0 && h != 0) || w > MAX_PART_SIDE || h > MAX_PART_SIDE {
        return None;
    }
    pq.width = (w != 0).then_some(w);
    pq.height = (h != 0).then_some(h);
    Some(format!("preview.{ext}"))
}

/// Resolve the item behind `/collections/{collection}/items/{item_id}/...` and answer what
/// is specific to titiler-pgstac. Everything else goes to the `/stac` handlers.
pub(super) async fn item_route(
    ctx: &Ctx<'_>,
    method: &Method,
    collection: &str,
    item_id: &str,
    rest: &[&str],
    pq: &ParsedQuery,
    body: Option<Bytes>,
) -> ItemRoute {
    use ItemRoute::Done;
    let Some(pg) = ctx.engine.pgstac() else {
        return Done(detail(
            StatusCode::SERVICE_UNAVAILABLE,
            "pgSTAC is not configured: set DATABASE_URL, or PGHOST and the other PG* variables",
        ));
    };
    // What FastAPI answers before the item is looked for: the method, then the path.
    let pairs = query_pairs(ctx.raw_query);
    if let Some(resp) = item_path_error(method, rest, pq, &pairs) {
        return Done(resp);
    }
    let base = format!("/collections/{collection}/items/{item_id}");
    let (collection, item_id) = (percent_decode(collection), percent_decode(item_id));
    let doc = match pg.get_item_cached(&collection, &item_id).await {
        Ok(Some(doc)) => doc,
        Ok(None) => {
            return Done(detail(
                StatusCode::NOT_FOUND,
                format!("No item '{item_id}' found in '{collection}' collection"),
            ));
        }
        Err(e) => return Done(db_error(&e)),
    };
    let readable = readable_assets(&doc);
    let self_href = doc
        .get("links")
        .and_then(Value::as_array)
        .and_then(|links| links.iter().find(|l| l["rel"] == "self"))
        .and_then(|l| l["href"].as_str())
        .unwrap_or("");
    let mut item = stac_item_from_json(&doc, self_href);
    // `/stac` goes by an href's extension where an asset has no media type. Here every
    // asset `/assets` lists can be read, whatever its href looks like.
    for name in &readable {
        if !item.assets.contains_key(name)
            && let Some(href) = doc["assets"][name]["href"].as_str().filter(|h| !h.is_empty())
        {
            item.assets.insert(name.clone(), crate::stac::resolve_href(self_href, href));
        }
    }
    // With no `self` link a relative href is relative to nothing: such an asset is not
    // read from wherever the server happens to run.
    if self_href.is_empty() {
        item.assets.retain(|name, _| {
            doc["assets"][name]["href"].as_str().is_some_and(|h| h.contains("://") || h.starts_with('/'))
        });
    }
    let item = Arc::new(item);
    let get = *method == Method::GET;

    match rest {
        ["assets"] if get => return Done(json_resp(StatusCode::OK, &json!(readable))),
        ["renders"] if get => {
            let renders: serde_json::Map<String, Value> = item_renders(&doc)
                .iter()
                .map(|r| (r.name.clone(), item_render_json(ctx.host, &base, &item_id, r)))
                .collect();
            return Done(json_resp(
                StatusCode::OK,
                &json!({ "renders": renders, "links": [{
                    "href": format!("{}{base}/renders", ctx.host),
                    "rel": "self",
                    "type": "application/json",
                    "title": "List STAC Renders metadata"
                }]}),
            ));
        }
        ["renders", render_id] if get => {
            let wanted = percent_decode(render_id);
            return Done(match item_renders(&doc).iter().find(|r| r.name == wanted) {
                Some(r) => json_resp(StatusCode::OK, &item_render_json(ctx.host, &base, &item_id, r)),
                None => detail(StatusCode::NOT_FOUND, "Render not found"),
            });
        }
        ["WMTSCapabilities.xml"] if get => {
            let extent = match item_extent(ctx, &doc, &item, &readable, pq).await {
                Ok(e) => e,
                Err(r) => return Done(r),
            };
            let source = WmtsSource {
                title: "TiTiler",
                bounds: extent.wgs84,
                minzoom: extent.minzoom,
                maxzoom: extent.maxzoom,
                // Each render that can make a tile is a layer.
                default_layers: item_renders(&doc)
                    .into_iter()
                    .filter(|r| r.valid)
                    .map(|r| (r.name, urlencode_doseq(&r.params)))
                    .collect(),
            };
            return Done(wmts(ctx, &base, source, &pairs));
        }
        ["tiles"] | ["tiles", _] if get => {
            if let ["tiles", tms] = rest
                && let Err(r) = check_tms(tms)
            {
                return Done(r);
            }
            if let Some(r) = tileset_format_error(&pairs) {
                return Done(r);
            }
            let extent = match item_extent(ctx, &doc, &item, &readable, pq).await {
                Ok(e) => e,
                Err(r) => return Done(r),
            };
            let metadata = json!({
                "bounds": extent.wgs84, "minzoom": extent.minzoom, "maxzoom": extent.maxzoom
            });
            return Done(match rest {
                ["tiles", tms] => tileset(ctx, &base, &metadata, tms, &pairs),
                _ => tilesets(ctx, &base, &metadata, &pairs),
            });
        }
        _ => {}
    }

    if let (["tiles", _, _, _, _] | [_, "tilejson.json" | "map.html"], Some(resp)) = (rest, tile_bounds_error(&pairs)) {
        return Done(resp);
    }
    if let (["point", _], Some(resp)) = (rest, point_query_error(&pairs)) {
        return Done(resp);
    }
    if item_route_needs_assets(rest) {
        if pq.assets.is_empty() {
            return Done(missing_query_param("assets"));
        }
        for asset in &pq.assets {
            let name = asset.split('|').next().unwrap_or(asset);
            if !readable.iter().any(|r| r == name) {
                let known: Vec<String> = readable.iter().map(|r| format!("'{r}'")).collect();
                return Done(detail(
                    StatusCode::NOT_FOUND,
                    format!("'{name}' is not valid, should be one of [{}]", known.join(", ")),
                ));
            }
        }
    }
    // The routes whose answer depends on where the item lies.
    let needs_extent = matches!(rest, ["info.geojson"] | ["tiles", _, _, _, _] | [_, "tilejson.json"]);
    let extent = if needs_extent {
        match item_extent(ctx, &doc, &item, &readable, pq).await {
            Ok(e) => Some(e),
            Err(r) => return Done(r),
        }
    } else {
        None
    };
    // A tile that does not touch the item is a 404, as in rio-tiler's `tile_exists`.
    if let (["tiles", "WebMercatorQuad", zs, xs, ys], Some(m)) =
        (rest, extent.as_ref().and_then(|e| e.mercator))
        && let Some((y, _, _)) = parse_y_scale_ext(ys)
        && let Ok(Some(tile)) = parse_zxy(zs, xs, y)
    {
        let TileCoord { z, x, y } = tile;
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
        let touches = min_x < m[2] && max_x > m[0] && max_y > m[1] && min_y < m[3];
        if !touches {
            return Done(detail(
                StatusCode::NOT_FOUND,
                format!("Tile(x={x}, y={y}, z={z}) is outside bounds"),
            ));
        }
    }
    // Band math on one asset, and bands chosen on each of several, are things the search
    // routes draw and the `/stac` ones do not: a tile or a box of this item with either
    // is drawn by the search routes, from this item alone. So are the statistics of a
    // posted feature, which titiler-pgstac works out for an item as for a mosaic of it.
    let posted_statistics = *method == Method::POST && rest == ["statistics"];
    if (get && item_picture_by_asset(rest, &pairs)) || posted_statistics {
        let assets: std::collections::BTreeMap<String, Value> = doc["assets"]
            .as_object()
            .map(|a| a.iter().filter(|(name, _)| readable.contains(name)).map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        let bbox = doc["bbox"].as_array().and_then(|b| match b.len() {
            4 => Some([b[0].as_f64()?, b[1].as_f64()?, b[2].as_f64()?, b[3].as_f64()?]),
            6 => Some([b[0].as_f64()?, b[1].as_f64()?, b[3].as_f64()?, b[4].as_f64()?]),
            _ => None,
        });
        // What is drawn is cached under the item and where its assets are read from.
        let identity = source_identity(&(&collection, &item_id, &item.assets));
        let only = Arc::new(vec![MosaicItem {
            id: item_id.clone(),
            collection: Some(collection.clone()),
            bbox,
            assets,
            self_href: (!self_href.is_empty()).then(|| self_href.to_string()),
        }]);
        let rec = SearchRecord {
            hash: format!("item-{identity:016x}"),
            search: Value::Null,
            where_clause: None,
            orderby: None,
            lastused: None,
            usecount: None,
            metadata: Value::Null,
        };
        let ctx = Ctx {
            engine: ctx.engine,
            host: ctx.host,
            forwarded_prefix: ctx.forwarded_prefix,
            path: ctx.path,
            raw_query: ctx.raw_query,
            if_none_match: ctx.if_none_match,
            only: Some(&only),
        };
        // A tile the grid does not have is outside the item like any other.
        if let ["tiles", _, zs, xs, ys] = rest
            && let (Ok(z), Ok(x), Some((y, _, _))) = (zs.parse::<u32>(), xs.parse::<u64>(), super::parse_y_scale_ext(ys))
            && (z > 30 || x >= (1u64 << z) || u64::from(y) >= (1u64 << z))
        {
            return Done(detail(StatusCode::NOT_FOUND, format!("Tile(x={x}, y={y}, z={z}) is outside bounds")));
        }
        // What only a search reads means nothing to an item, and titiler-pgstac passes it
        // over there: the search routes are not shown it.
        let query = without_search_options(ctx.raw_query);
        // Nor what the item's route passes over, which the search routes would check.
        let query: String = query
            .split('&')
            .filter(|part| {
                let name = super::query_key(part.split('=').next().unwrap_or(part));
                let name = name.as_ref();
                !CHECKED.contains(&name) || item_route_reads(method, rest, name)
            })
            .collect::<Vec<_>>()
            .join("&");
        let (own, pairs) = (super::parse_query((!query.is_empty()).then_some(query.as_str())), query_pairs(&query));
        return Done(mosaic_route(&ctx, pg, &base, &rec, rest, body.as_ref(), own, &pairs).await);
    }
    ItemRoute::Stac(item, extent)
}

/// A query string without the parameters only a search's routes read: how its items are
/// chosen among and how many are looked for.
fn without_search_options(raw_query: &str) -> String {
    let algorithm = raw_query.split('&').any(|part| part.split_once('=').is_some_and(|(k, _)| super::query_key(k) == "algorithm"));
    let kept: Vec<&str> = raw_query
        .split('&')
        .filter(|part| {
            let name = super::query_key(part.split('=').next().unwrap_or(part));
            let name = name.as_ref();
            !matches!(name, "pixel_selection" | "items_limit" | "scan_limit" | "time_limit" | "exitwhenfull" | "skipcovered")
                && (algorithm || name != "algorithm_params")
        })
        .collect();
    kept.join("&")
}

/// Whether an item request is one the search routes draw: a tile or a box whose assets
/// carry band math, or bands of their own on more than one of them, or one asked for as an
/// array. A box is one with a format in its path, as titiler-pgstac's item routes have it.
fn item_picture_by_asset(rest: &[&str], pairs: &[(String, String)]) -> bool {
    let picture = match rest {
        ["tiles", "WebMercatorQuad", _, _, _] => true,
        ["bbox", seg] => matches!(super::parse_bbox4(seg), Some((_, Some(_)))),
        // A side of zero follows the other, or the data's own resolution: a size the
        // search routes work out and the `/stac` ones refuse.
        ["bbox", _, dims] if matches!(path_size(dims, pairs), Ok((PartSize::Query { .. }, Some(_)))) => return true,
        ["bbox", _, dims] => matches!(super::parse_wxh_ext(dims), Ok((_, _, Some(_)))),
        _ => false,
    };
    if !picture {
        return false;
    }
    // An array, as `.npy` or as a GeoTIFF, is written by the search routes in the data's
    // own type, as titiler-pgstac writes it; the `/stac` routes write float32.
    let format = rest
        .last()
        .and_then(|last| last.rsplit_once('.'))
        .map(|(_, ext)| ext)
        .filter(|ext| !ext.contains(','))
        .or_else(|| query_value(pairs, "format").or_else(|| query_value(pairs, "f")));
    if matches!(format, Some("npy" | "tif" | "tiff")) {
        return true;
    }
    let Ok(specs) = parse_assets(pairs) else { return false };
    specs.iter().any(|s| s.expression.is_some()) || (specs.len() > 1 && specs.iter().any(|s| s.indexes.is_some()))
}

/// Serve a request whose path starts with `/searches`.
pub(super) async fn handle<B>(
    ctx: Ctx<'_>,
    method: &Method,
    segs: &[&str],
    pq: ParsedQuery,
    req: Request<B>,
) -> Resp
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let Some(pg) = ctx.engine.pgstac() else {
        return detail(
            StatusCode::SERVICE_UNAVAILABLE,
            "pgSTAC is not configured: set DATABASE_URL, or PGHOST and the other PG* variables",
        );
    };
    let pg: &PgStac = pg;
    let pairs = query_pairs(ctx.raw_query);
    let get = *method == Method::GET;
    let not_found = || detail(StatusCode::NOT_FOUND, "Not Found");

    match segs {
        ["searches", "register"] if *method == Method::POST => register(&ctx, pg, req).await,
        ["searches"] | ["searches", ""] | ["searches", "list"] if get => list(&ctx, pg, &pairs).await,
        // Item routes with a path are answered before this is reached.
        ["collections", _, "items", ..] => not_found(),
        // A route asked with a method it does not take is a 405 before anything is looked up.
        ["searches" | "collections", _, rest @ ..] if route_method(rest).is_some_and(|takes| takes != *method) => {
            method_not_allowed(route_method(rest))
        }
        [family @ ("searches" | "collections"), id, rest @ ..]
            if (get || *method == Method::POST) && !rest.is_empty() =>
        {
            // A tile matrix set in the path is validated before anything is looked up.
            let tms = match rest {
                [tms, "tilejson.json" | "map.html"] => Some(*tms),
                ["tiles", tms, ..] => Some(*tms),
                _ => None,
            };
            if let Some(Err(r)) = tms.map(check_tms) {
                return r;
            }
            let found = if *family == "searches" {
                find_search(pg, id).await
            } else {
                collection_mosaic(pg, id, &pairs).await
            };
            let rec = match found {
                Ok(rec) => rec,
                Err(r) => return r,
            };
            let base = format!("/{family}/{id}");
            let body = if get {
                None
            } else {
                match read_limited_body(req.into_body()).await {
                    Ok(bytes) => Some(bytes),
                    Err(e) => return engine_error(&e),
                }
            };
            mosaic_route(&ctx, pg, &base, &rec, rest, body.as_ref(), pq, &pairs).await
        }
        _ => not_found(),
    }
}

/// The methods an item's routes take, by the shape of the path: `(get, post)`; `None` for
/// a path that is no route of an item.
fn item_route_methods(rest: &[&str]) -> Option<(bool, bool)> {
    match rest {
        ["statistics"] => Some((true, true)),
        [first, ..] if *first == "feature" || first.starts_with("feature.") => Some((false, true)),
        ["info" | "info.geojson" | "assets" | "asset_statistics" | "renders" | "WMTSCapabilities.xml" | "preview" | "tiles"] => Some((true, false)),
        [one] if one.starts_with("preview.") => Some((true, false)),
        [_, "tilejson.json" | "map.html"] | ["preview" | "renders" | "tiles" | "point" | "bbox", _] => Some((true, false)),
        ["tiles", _, _, _, _] | ["bbox", _, _] => Some((true, false)),
        _ => None,
    }
}

/// What is wrong with an item's request that FastAPI says before anything is read: a
/// method the route does not take (405), then a piece of the path, or a size, that
/// cannot be read (422). The parsers are the ones a search's routes read their paths with.
fn item_path_error(method: &Method, rest: &[&str], pq: &ParsedQuery, pairs: &[(String, String)]) -> Option<Resp> {
    let (get, post) = item_route_methods(rest)?;
    if !(*method == Method::GET && get || *method == Method::POST && post) {
        return Some(method_not_allowed(Some(if get { Method::GET } else { Method::POST })));
    }
    let unknown_tms = |tms: &str| {
        crate::tms::lookup_tms(tms).is_none()
            .then(|| invalid_param("path", "tileMatrixSetId", "Input should be a supported TileMatrixSet"))
    };
    let of_path = match rest {
        ["tiles", tms, zs, xs, ys] => unknown_tms(tms).or_else(|| match parse_y_scale_ext(ys) {
            None => Some(invalid_param("path", "y", "Input should be a valid integer")),
            Some((y, _, ext)) => parse_zxy(zs, xs, y).err().or_else(|| format_error(ext.filter(|_| ys.contains('.')))),
        }),
        ["tiles", tms] | [tms, "tilejson.json" | "map.html"] => unknown_tms(tms),
        ["point", at] => point_coords(at).err(),
        ["bbox", _, dims] => match path_size(dims, pairs) {
            Err(resp) if resp.status() == StatusCode::UNPROCESSABLE_ENTITY => Some(resp),
            _ => format_error(dims.split_once('.').map(|(_, ext)| ext)),
        },
        ["preview", dims] => format_error(dims.split_once('.').map(|(_, ext)| ext)),
        [one] if one.starts_with("preview.") => format_error(one.strip_prefix("preview.")),
        _ => None,
    };
    // An algorithm's parameters, before the item is looked for, where the route applies one.
    let of_path = of_path.or_else(|| {
        if !item_applies_algorithm(rest) {
            return None;
        }
        if let Some(input) = &pq.algorithm_params_unread {
            if rest == ["WMTSCapabilities.xml"] {
                return Some(detail(StatusCode::BAD_REQUEST, "invalid JSON content in algorithm_params"));
            }
            return Some(json_resp(
                StatusCode::UNPROCESSABLE_ENTITY,
                &json!({ "detail": [{ "type": "value_error", "loc": ["query", "algorithm_params"], "msg": "Value error, invalid JSON content", "input": input }] }),
            ));
        }
        pq.algorithm_error.as_ref().map(|msg| detail(StatusCode::BAD_REQUEST, msg.clone()))
    });
    // A size that is not a number. One that is, and is out of the range drawn here, stays
    // the 400 of a limit; `padding` and `buffer` are answered for where tiles are checked.
    of_path.or_else(|| {
        pq.errors.iter().find_map(|e| {
            let name = error_parameter(e).filter(|name| item_route_reads(method, rest, name))?;
            if matches!(name, "max_size" | "width" | "height") {
                let unread = pairs.iter().rev().find(|(k, _)| k == name).is_some_and(|(_, v)| !py_whole(v));
                return unread.then(|| invalid_param("query", name, "Input should be a valid value"));
            }
            e.starts_with("invalid ").then(|| parse_error(e))
        })
    })
}

/// Whether a route of a mosaic applies `algorithm`: its tiles and what points at them, its
/// pictures of a box or a feature, and its statistics; not the items under a tile, a box or
/// a point, nor `info`.
fn applies_algorithm(rest: &[&str]) -> bool {
    match rest {
        [.., "assets"] => false,
        ["tiles", ..] | [_, "tilejson.json" | "map.html"] | ["WMTSCapabilities.xml"] | ["bbox", ..] | ["statistics"] => true,
        [one, ..] => *one == "feature" || one.starts_with("feature."),
        _ => false,
    }
}

/// The method each route of a mosaic takes, by the shape of its path; `None` for a path
/// that is no route. The one table the dispatcher and its 405s read.
fn route_method(rest: &[&str]) -> Option<Method> {
    match rest {
        ["statistics"] => Some(Method::POST),
        [first, ..] if *first == "feature" || first.starts_with("feature.") => Some(Method::POST),
        ["info"] | ["WMTSCapabilities.xml"] | [_, "tilejson.json" | "map.html"] => Some(Method::GET),
        ["tiles"] | ["tiles", _] | ["tiles", _, _, _, _] | ["tiles", _, _, _, _, "assets"] => Some(Method::GET),
        ["point", _] | ["point", _, "assets"] | ["bbox", _] | ["bbox", _, _] => Some(Method::GET),
        _ => None,
    }
}

/// A posted Feature or FeatureCollection whose coordinates are Web Mercator metres, with
/// them in degrees. `None` when the body is not JSON, which its route then says.
fn posted_in_degrees(body: &Bytes) -> Option<Bytes> {
    fn turn(coordinates: &mut Value) {
        let Some(list) = coordinates.as_array_mut() else { return };
        match (list.first().and_then(Value::as_f64), list.get(1).and_then(Value::as_f64)) {
            (Some(x), Some(y)) => {
                let (lon, lat) = mercator_to_lonlat_deg(x, y);
                (list[0], list[1]) = (json!(lon), json!(lat));
            }
            _ => list.iter_mut().for_each(turn),
        }
    }
    fn geometry(of: &mut Value) {
        if let Some(members) = of.get_mut("geometries").and_then(Value::as_array_mut) {
            members.iter_mut().for_each(geometry);
        }
        if let Some(coordinates) = of.get_mut("coordinates") {
            turn(coordinates);
        }
    }
    let mut doc = serde_json::from_slice::<Value>(body).ok()?;
    match doc.get_mut("features").and_then(Value::as_array_mut) {
        Some(features) => features.iter_mut().filter_map(|f| f.get_mut("geometry")).for_each(geometry),
        None => {
            if let Some(g) = doc.get_mut("geometry") {
                geometry(g);
            }
        }
    }
    serde_json::to_vec(&doc).ok().map(Bytes::from)
}

/// The routes shared by a registered search and a collection's mosaic.
#[allow(clippy::too_many_arguments)]
async fn mosaic_route(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    base: &str,
    rec: &SearchRecord,
    rest: &[&str],
    body: Option<&Bytes>,
    pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    let not_found = || detail(StatusCode::NOT_FOUND, "Not Found");
    // A body where a route takes none, or none where it takes one. (The dispatcher has
    // answered these for a search and a collection; an item's routes come straight here.)
    let takes_body = route_method(rest) == Some(Method::POST);
    if body.is_some() != takes_body {
        return if body.is_some() && route_method(rest).is_none() { not_found() } else { method_not_allowed(route_method(rest)) };
    }
    // An algorithm's parameters, checked as TiTiler's models check them, where it is applied:
    // not JSON is FastAPI's 422 (WMTS answers 400), what the model refuses a 400.
    if applies_algorithm(rest) {
        if let Some(input) = &pq.algorithm_params_unread {
            if rest == ["WMTSCapabilities.xml"] {
                return detail(StatusCode::BAD_REQUEST, "invalid JSON content in algorithm_params");
            }
            return json_resp(
                StatusCode::UNPROCESSABLE_ENTITY,
                &json!({ "detail": [{ "type": "value_error", "loc": ["query", "algorithm_params"], "msg": "Value error, invalid JSON content", "input": input }] }),
            );
        }
        if let Some(msg) = &pq.algorithm_error {
            return detail(StatusCode::BAD_REQUEST, msg.clone());
        }
    }
    // What is posted in Web Mercator metres is read in degrees from here on, the one place
    // a posted geometry changes system. As in TiTiler, a picture of it is then drawn in
    // the system it came in unless `dst_crs` says another.
    let mut pq = pq;
    let in_metres = pq.coord_crs_3857.then(|| body.and_then(posted_in_degrees)).flatten();
    let own_pairs: Vec<(String, String)>;
    // What was posted, when it was turned: statistics hand each feature back as it came.
    let as_posted = body.filter(|_| in_metres.is_some());
    let (body, pairs) = match &in_metres {
        Some(turned) => {
            (pq.coord_crs_3857, pq.coord_crs, pq.coord_crs_epsg) = (false, None, None);
            let mut kept: Vec<(String, String)> = pairs.iter().filter(|(k, _)| k != "coord_crs").cloned().collect();
            if query_value(pairs, "dst_crs").is_none() {
                kept.push(("dst_crs".into(), "epsg:3857".into()));
            }
            own_pairs = kept;
            (Some(turned), own_pairs.as_slice())
        }
        None => (body, pairs),
    };
    match rest {
        ["info"] => {
            let mut search = search_json(rec);
            // titiler-pgstac leaves the compiled SQL out of a collection's description.
            if let (true, Some(obj)) = (base.starts_with("/collections/"), search.as_object_mut()) {
                obj.remove("_where");
                obj.remove("orderby");
            }
            json_resp(
                StatusCode::OK,
                &json!({ "search": search, "links": info_links(ctx.host, base, rec) }),
            )
        }
        [_, "tilejson.json" | "map.html"] if render_query_error(pairs).is_some() => {
            render_query_error(pairs).unwrap_or_else(not_found)
        }
        [_, "tilejson.json" | "map.html"]
            if query_value(pairs, "tile_format")
                .is_some_and(|f| !matches!(f, "png" | "npy" | "tif" | "tiff" | "jpeg" | "jpg" | "jp2" | "webp" | "pngraw")) =>
        {
            invalid_param("query", "tile_format", "Input should be an image format")
        }
        // A map with nothing to show is refused as its tiles would be.
        [_, "map.html"] if query_value(pairs, "assets").is_none() => missing_query_param("assets"),
        [_, "map.html"] if ["tilesize", "minzoom", "maxzoom"].iter().any(|k| query_value(pairs, k).is_some_and(|v| !py_whole(v))) => {
            invalid_param("query", "tilesize", "Input should be a valid integer")
        }
        // The WMTS document says the same with a 400.
        ["WMTSCapabilities.xml"]
            if render_query_error(&pairs.iter().filter(|(k, _)| k != "tilesize").cloned().collect::<Vec<_>>()).is_some() =>
        {
            detail(StatusCode::BAD_REQUEST, "a parameter of the tiles this document describes is not valid")
        }
        [tms, "tilejson.json"] => tilejson(ctx, base, rec, tms, pairs),
        [tms, "map.html"] => {
            let tilejson_url = format!(
                "{}{base}/{tms}/tilejson.json?{}",
                ctx.forwarded_prefix, ctx.raw_query
            );
            let mut resp = Response::new(Full::new(Bytes::from(build_map_html(&tilejson_url))));
            resp.headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
            resp.headers_mut()
                .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
            resp
        }
        ["WMTSCapabilities.xml"] => {
            let (minzoom, maxzoom, bounds) = match mosaic_extent(&rec.metadata, &[]) {
                Ok(e) => e,
                Err(r) => return r,
            };
            let source = WmtsSource {
                title: &rec.hash,
                bounds,
                minzoom,
                maxzoom,
                default_layers: layer_queries(&rec.metadata, true),
            };
            wmts(ctx, base, source, pairs)
        }
        ["tiles"] | ["tiles", _] if tileset_format_error(pairs).is_some() => tileset_format_error(pairs).unwrap_or_else(not_found),
        ["tiles"] => tilesets(ctx, base, &rec.metadata, pairs),
        ["tiles", tms] => tileset(ctx, base, &rec.metadata, tms, pairs),
        ["tiles", tms, zs, xs, ys, "assets"] => {
            let Some(tms_def) = crate::tms::lookup_tms(tms) else {
                return invalid_param("path", "tileMatrixSetId", "Input should be a supported TileMatrixSet");
            };
            let Ok(y) = ys.parse::<u32>() else {
                return detail(StatusCode::BAD_REQUEST, "invalid z/x/y tile coordinate");
            };
            if tms_def.id == "WebMercatorQuad" {
                match parse_zxy(zs, xs, y) {
                    Err(r) => r,
                    Ok(None) => json_resp(StatusCode::OK, &json!([])),
                    Ok(Some(t)) => tile_assets_geom(pg, &rec.hash, &tile_polygon(t), true, pairs).await,
                }
            } else {
                match parse_tms_addr(tms_def, zs, xs, y) {
                    Err(r) => r,
                    Ok(None) => json_resp(StatusCode::OK, &json!([])),
                    Ok(Some(addr)) => match tms_tile_polygon(tms_def, addr) {
                        Some(geom) => tile_assets_geom(pg, &rec.hash, &geom, false, pairs).await,
                        None => json_resp(StatusCode::OK, &json!([])),
                    },
                }
            }
        }
        ["tiles", _, _, _, _] if tile_bounds_error(pairs).is_some() => tile_bounds_error(pairs).unwrap_or_else(not_found),
        ["tiles", tms, zs, xs, ys] => {
            let Some(tms_def) = crate::tms::lookup_tms(tms) else {
                return invalid_param("path", "tileMatrixSetId", "Input should be a supported TileMatrixSet");
            };
            let Some((y, scale, ext)) = parse_y_scale_ext(ys) else {
                return invalid_param("path", "y", "Input should be a valid integer");
            };
            // Without an extension the format is chosen from the rendered pixels.
            let ext = if ys.contains('.') { ext.unwrap_or("") } else { "" };
            if tms_def.id == "WebMercatorQuad" {
                match parse_zxy(zs, xs, y) {
                    Err(r) => r,
                    Ok(_) if format_error(Some(ext)).is_some() => format_error(Some(ext)).unwrap_or_else(not_found),
                    Ok(None) if pq.assets.is_empty() => missing_query_param("assets"),
                    Ok(None) => no_content(),
                    Ok(Some(coord)) => tile(ctx, pg, &rec.hash, coord, scale, ext, pq, pairs).await,
                }
            } else {
                match parse_tms_addr(tms_def, zs, xs, y) {
                    Err(r) => r,
                    Ok(_) if format_error(Some(ext)).is_some() => format_error(Some(ext)).unwrap_or_else(not_found),
                    Ok(None) if pq.assets.is_empty() => missing_query_param("assets"),
                    Ok(None) => no_content(),
                    Ok(Some(addr)) => {
                        render_non_mercator_tms_tile(ctx, pg, &rec.hash, tms_def, addr, scale, ext, pq, pairs).await
                    }
                }
            }
        }
        [first, ..] if *first == "feature" || first.starts_with("feature.") => {
            feature_image(ctx, pg, &rec.hash, rest, body, pq, pairs).await
        }
        ["bbox", seg] => {
            // The format is part of the path here: `{minx},{miny},{maxx},{maxy}.{format}`.
            let Some((bbox, ext)) = parse_bbox4(seg) else {
                // Four numbers and a format nobody has is the format's fault.
                return match seg.rsplit_once('.') {
                    Some((numbers, ext)) if parse_bbox4(numbers).is_some() => format_error(Some(ext)).unwrap_or_else(not_found),
                    _ => not_found(),
                };
            };
            let Some(ext) = ext else {
                return invalid_param(
                    "path",
                    "format",
                    "Input should be 'png', 'npy', 'tif', 'tiff', 'jpeg', 'jpg', 'jp2', 'webp' or 'pngraw'",
                );
            };
            let size = match part_size_from_query(pairs) {
                Ok(s) => s,
                Err(r) => return r,
            };
            let polygon = bbox_polygon(bbox);
            part_image(ctx, pg, &rec.hash, &polygon, bbox, None, size, Some(ext), pq, pairs).await
        }
        ["bbox", bbox, "assets"] => {
            let Some(([w, s, e, n], None)) = parse_bbox4(bbox) else {
                return not_found();
            };
            let corners = in_degrees(&pq, pairs, w, s).and_then(|low| Ok((low, in_degrees(&pq, pairs, e, n)?)));
            let (((w, s), (e, n)), limits) = match (corners, parse_limits(pairs)) {
                (Err(r), _) | (_, Err(r)) => return r,
                (Ok(corners), Ok(l)) => (corners, l),
            };
            let polygon = bbox_polygon([w, s, e, n]);
            // A size matters only to the lean lookup, and this route has no size of its own:
            // without a usable one the box is sampled as a 1024-pixel picture would be.
            let size = match part_size_from_query(pairs) {
                Ok(PartSize::Query { max_size: None, width: None, height: None }) | Err(_) => {
                    PartSize::Query { max_size: Some(1024), width: None, height: None }
                }
                Ok(size) => size,
            };
            geometry_assets(pg, &rec.hash, polygon, limits, pairs, part_grid(size, [w, s, e, n])).await
        }
        ["bbox", seg, dims] => {
            let Some((bbox, None)) = parse_bbox4(seg) else {
                return not_found();
            };
            let (size, ext) = match path_size(dims, pairs) {
                Ok(parsed) => parsed,
                Err(resp) => return resp,
            };
            let polygon = bbox_polygon(bbox);
            part_image(ctx, pg, &rec.hash, &polygon, bbox, None, size, ext, pq, pairs)
                .await
        }
        ["statistics"] => mosaic_statistics(ctx, pg, &rec.hash, body, as_posted, pq, pairs).await,
        ["point", lonlat] => point_values(ctx, pg, &rec.hash, lonlat, &pq, pairs).await,
        ["point", lonlat, "assets"] => {
            let (lon, lat) = match point_coords(lonlat) {
                Ok(at) => at,
                Err(resp) => return resp,
            };
            let ((lon, lat), mut limits) = match (in_degrees(&pq, pairs, lon, lat), parse_limits(pairs)) {
                (Err(r), _) | (_, Err(r)) => return r,
                (Ok(at), Ok(l)) => (at, l),
            };
            // A point is never "covered", so every item under it is wanted.
            limits.exitwhenfull = false;
            limits.skipcovered = false;
            let point = json!({ "type": "Point", "coordinates": [lon, lat] });
            geometry_assets(pg, &rec.hash, point, limits, pairs, None).await
        }
        _ => not_found(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An item's stacked bands are named as titiler-pgstac names them: each result of an
    /// asset's own band math after the asset, each band read of a plain asset as `{asset}_bK`
    /// (those its `|bidx=` picks), and the asset itself under `asset_as_band`.
    #[test]
    fn stacked_bands_have_titiler_pgstacs_names() {
        let names = |assets: &[&str]| assets.iter().map(|a| a.to_string()).collect::<Vec<_>>();
        let own = |e: &str| (Some(e.to_string()), None);
        let plain = (None, None);
        assert_eq!(
            item_stack_names(&names(&["visual", "red"]), &[3, 1], &[own("b1+b2;b3"), own("b1*2")], false),
            ["visual_b1+b2", "visual_b3", "red_b1*2"]
        );
        assert_eq!(
            item_stack_names(&names(&["visual", "red"]), &[3, 1], &[plain.clone(), plain.clone()], false),
            ["visual_b1", "visual_b2", "visual_b3", "red_b1"]
        );
        assert_eq!(
            item_stack_names(&names(&["visual", "red"]), &[3, 1], &[(None, Some(vec![3, 1])), plain.clone()], false),
            ["visual_b3", "visual_b1", "red_b1"]
        );
        assert_eq!(item_stack_names(&names(&["nir", "red"]), &[1, 1], &[plain.clone(), plain], true), ["nir", "red"]);
    }


    /// An asset's band math is over all of its bands: the bands it chooses are dropped, and the
    /// options given back so that the shared parser's choice of them is not read either.
    #[test]
    fn asset_math_drops_the_assets_own_band_choice() {
        let options = |q: &str| item_asset_options(&["point", "1,2"], q).ok().flatten();
        assert_eq!(options("assets=visual%7Cexpression%3Db3%7Cbidx%3D1"), Some((vec![], vec![])));
        assert_eq!(
            options("assets=visual%7Cexpression%3Db3%7Cbidx%3D1&assets=red%7Cbidx%3D1"),
            Some((vec![("red".to_string(), vec![1])], vec![]))
        );
        assert_eq!(options("assets=visual%7Cexpression%3Db3"), None);
    }


    /// Band math is an `|expression=` option of an asset, not an option whose name ends so.
    #[test]
    fn asset_math_is_an_expression_option() {
        assert!(asset_math_asked(&query_pairs("assets=red%7Cexpression%3Db1*2")));
        assert!(asset_math_asked(&query_pairs("assets=visual%7Cbidx%3D1%7Cexpression%3Db1")));
        assert!(!asset_math_asked(&query_pairs("assets=visual%7Ccustom_expression%3Db1*2%7Cbidx%3D3")));
        assert!(!asset_math_asked(&query_pairs("assets=red&expression=b1*2")));
    }

    /// An item's point is read band by band and each asset's band math applied afterwards.
    #[test]
    fn point_asset_math_applies_each_assets_band_math() {
        let mut doc = json!({
            "values": [77.0, 81.0, 83.0, 750.0, 1234.0],
            "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "red_b1", "nir_b1"],
        });
        let q = "assets=visual%7Cexpression%3Db1%2Bb2%3Bb3&assets=red%7Cexpression%3Db1*2&assets=nir";
        point_asset_math(&mut doc, &query_pairs(q)).unwrap();
        assert_eq!(doc["values"], json!([158.0, 83.0, 1500.0, 1234.0]));
        assert_eq!(doc["band_names"], json!(["b1", "b2", "b3", "b4"]));
        assert_eq!(doc["band_descriptions"], json!(["visual_b1+b2", "visual_b3", "red_b1*2", "nir_b1"]));

        let mut doc = json!({ "values": [77.0, 81.0, 83.0, 750.0], "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "red_b1"] });
        let as_band = "assets=visual%7Cexpression%3Db3&assets=red&asset_as_band=true";
        point_asset_math(&mut doc, &query_pairs(as_band)).unwrap();
        assert_eq!((&doc["values"], &doc["band_descriptions"]), (&json!([83.0, 750.0]), &json!(["visual", "red"])));

        let mut doc = json!({ "values": [77.0, 81.0, 83.0, 750.0], "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "red_b1"] });
        let plain_many = "assets=visual&assets=red%7Cexpression%3Db1&asset_as_band=true";
        assert!(point_asset_math(&mut doc, &query_pairs(plain_many)).is_err(), "a plain asset of three bands is not one band");

        // An asset asked for twice is read twice; a name that ends like a band is its own asset.
        let mut doc = json!({ "values": [750.0, 750.0], "band_descriptions": ["red_b1", "red_b1"] });
        point_asset_math(&mut doc, &query_pairs("assets=red%7Cexpression%3Db1*2&assets=red%7Cexpression%3Db1*3")).unwrap();
        assert_eq!((&doc["values"], &doc["band_descriptions"]), (&json!([1500.0, 2250.0]), &json!(["red_b1*2", "red_b1*3"])));
        let mut doc = json!({ "values": [10.0, 750.0], "band_descriptions": ["red_b1_b1", "red_b1"] });
        point_asset_math(&mut doc, &query_pairs("assets=red_b1%7Cexpression%3Db1*2&assets=red")).unwrap();
        assert_eq!((&doc["values"], &doc["band_descriptions"]), (&json!([20.0, 750.0]), &json!(["red_b1_b1*2", "red_b1"])));
        // Every asset is read whole and each entry's choice of bands taken from it, in the
        // order chosen, whichever way it is spelled; an asset with band math takes none.
        let whole = || json!({ "values": [77.0, 81.0, 83.0, 750.0], "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "red_b1"] });
        for (q, want) in [
            ("assets=visual%7Cbidx%3D3,1&assets=red%7Cexpression%3Db1*2", json!([83.0, 77.0, 1500.0])),
            ("assets=visual%7Cindexes%3D3&assets=red%7Cexpression%3Db1*2", json!([83.0, 1500.0])),
            ("assets=visual&assets=red%7Cexpression%3Db1*2&asset_bidx=visual%7C3", json!([83.0, 1500.0])),
            ("assets=visual&assets=red%7Cexpression%3Db1*2&asset_bidx=red%7C1", json!([77.0, 81.0, 83.0, 1500.0])),
        ] {
            let mut doc = whole();
            point_asset_math(&mut doc, &query_pairs(q)).unwrap();
            assert_eq!(doc["values"], want, "{q}");
        }
        // The same asset with band math, then plain with a band chosen.
        let mut doc = json!({ "values": [77.0, 81.0, 83.0, 77.0, 81.0, 83.0], "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "visual_b1", "visual_b2", "visual_b3"] });
        point_asset_math(&mut doc, &query_pairs("assets=visual%7Cexpression%3Db3&assets=visual%7Cbidx%3D1")).unwrap();
        assert_eq!((&doc["values"], &doc["band_descriptions"]), (&json!([83.0, 77.0]), &json!(["visual_b3", "visual_b1"])));
        // Null (nodata) propagates as null; 0/0 maps via nan_to_num to 0.0; empty expression errors.
        let mut doc = json!({ "values": [null, 0.0], "band_descriptions": ["red_b1", "nir_b1"] });
        point_asset_math(&mut doc, &query_pairs("assets=red%7Cexpression%3Db1*2&assets=nir%7Cexpression%3Db1/b1")).unwrap();
        assert_eq!(doc["values"], json!([null, 0.0]));
        let mut doc = json!({ "values": [10.0], "band_descriptions": ["red_b1"] });
        assert!(point_asset_math(&mut doc, &query_pairs("assets=red%7Cexpression%3D")).is_err());
        assert!(point_asset_math(&mut doc, &query_pairs("assets=:all:%7Cexpression%3Db1*2")).is_err());
        // point_asset_math_reads clears bidx/asset_bidx/asset_exprs/asset_as_band and preserves asset order/duplicates.
        let raw = "assets=red%7Cexpression%3Db1*2&assets=red&bidx=1&asset_bidx=red%7C1&asset_as_band=true";
        let mut pq = super::super::parse_query(Some(raw));
        point_asset_math_reads(&mut pq, raw);
        assert_eq!(pq.assets, vec!["red".to_string(), "red".to_string()]);
        assert!(pq.bidx.is_empty() && pq.asset_bidx.is_empty() && pq.asset_exprs.is_empty() && !pq.asset_as_band);
    }

    #[test]
    fn every_resampling_method_is_its_own_kernel() {
        // `cubic_spline` and `lanczos` are drawn with their own kernels, as rasterio's are.
        for parameter in ["resampling", "reproject"] {
            assert_eq!(resample_method(parameter, "cubic_spline"), Some(ResampleMode::CubicSpline));
            assert_eq!(resample_method(parameter, "lanczos"), Some(ResampleMode::Lanczos));
            assert_eq!(resample_method(parameter, "cubic"), Some(ResampleMode::Cubic));
            assert_eq!(resample_method(parameter, "cubicspline"), None, "not rasterio's spelling");
        }
        assert_eq!(resample_method("reproject", "gauss"), None);
        assert_eq!(resample_method("resampling", "sum"), None);
    }

    #[test]
    fn complaints_are_matched_by_the_parameter_they_name() {
        assert_eq!(error_parameter("invalid width=wide: expected width in 1..=4096"), Some("width"));
        assert_eq!(error_parameter("parameter 'cover_scale' is not supported by cogzilla"), Some("cover_scale"));
        assert!(own_error("invalid asset_as_band=perhaps: expected a boolean"));
        assert!(own_error("invalid assets=visual|bidx=x: expected positive integer band indices in |bidx="));
        // A value that spells a parameter's name is not that parameter.
        assert!(!own_error("invalid nodata=histogram_bins: expected a number"));
        assert!(!own_size_error("invalid nodata=width=3: expected a number"));
        let pairs = |q: &str| query_pairs(q);
        assert_eq!(request_expression(&pairs("assets=red&expr=b1*2")), Some("b1*2"));
        assert_eq!(request_expression(&pairs("expression=b1&index=b2")), Some("b2"));
        assert_eq!(request_expression(&pairs("assets=red")), None);
    }

    #[test]
    fn register_body_is_split_like_titiler_pgstac() {
        let (search, metadata) = split_register_body(json!({
            "collections": ["sentinel-2-l2a"],
            "datetime": null,
            "metadata": { "name": "alps", "minzoom": null }
        }))
        .unwrap();
        assert_eq!(search, json!({ "collections": ["sentinel-2-l2a"], "filter-lang": "cql2-json" }));
        assert_eq!(metadata, json!({ "name": "alps", "type": "mosaic" }));

        let (_, metadata) = split_register_body(json!({ "metadata": { "type": "search" } })).unwrap();
        assert_eq!(metadata, json!({ "type": "search" }));
    }

    #[test]
    fn register_body_rejects_what_titiler_pgstac_rejects() {
        let both = json!({ "bbox": [0, 0, 1, 1], "intersects": { "type": "Point", "coordinates": [0, 0] } });
        assert_eq!(split_register_body(both).unwrap_err().status(), StatusCode::UNPROCESSABLE_ENTITY);
        let lang = json!({ "filter-lang": "cql2-text" });
        assert_eq!(split_register_body(lang).unwrap_err().status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(split_register_body(json!([1, 2])).is_err());
    }

    #[test]
    fn limits_default_and_parse() {
        assert_eq!(parse_limits(&[]).unwrap(), SearchLimits::default());
        let pairs = query_pairs("scan_limit=50&items_limit=3&time_limit=1&exitwhenfull=false&skipcovered=0");
        let l = parse_limits(&pairs).unwrap();
        assert_eq!((l.scan_limit, l.items_limit, l.time_limit_secs), (50, 3, 1));
        assert!(!l.exitwhenfull && !l.skipcovered);
        assert!(parse_limits(&query_pairs("items_limit=many")).is_err());
        assert!(parse_limits(&query_pairs("scan_limit=-1")).is_err());
    }

    #[test]
    fn tile_polygon_is_a_closed_wgs84_ring() {
        let poly = tile_polygon(TileCoord { z: 1, x: 0, y: 0 });
        let ring = poly["coordinates"][0].as_array().unwrap();
        assert_eq!(ring.len(), 5);
        assert_eq!(ring[0], ring[4]);
        let west = ring[0][0].as_f64().unwrap();
        let north = ring[1][1].as_f64().unwrap();
        assert!((west + 180.0).abs() < 1e-9);
        assert!((north - 85.0511287798066).abs() < 1e-6);
        assert!(ring[2][0].as_f64().unwrap().abs() < 1e-9);
    }

    #[test]
    fn collection_search_is_built_like_titiler_pgstac() {
        let collection = json!({
            "extent": { "spatial": { "bbox": [[-180.0, -90.0, 180.0, 90.0], [1.0, 2.0, 3.0, 4.0]] } },
            "item_assets": { "visual": {}, "aot": {}, "red-jp2": {}, "nir": {}, "blue": {} }
        });
        let (search, metadata) = collection_search("c", &collection, &[]).unwrap();
        assert_eq!(search, json!({ "collections": ["c"], "filter-lang": "cql2-json" }));
        assert_eq!(
            metadata,
            json!({
                "type": "mosaic",
                "name": "Mosaic for 'c' Collection",
                "bounds": [-180.0, -90.0, 180.0, 90.0],
                "assets": ["aot", "nir", "blue", "visual", "red-jp2"]
            })
        );

        let pairs = query_pairs(
            "ids=a,b&bbox=6,44,8,46&datetime=2024-01-01T00:00:00Z/..&sortby=%2Bproperties.eo:cloud_cover,-datetime,id\
             &query=%7B%22eo%3Acloud_cover%22%3A%7B%22lt%22%3A20%7D%7D\
             &filter=%7B%22op%22%3A%22%3D%22%2C%22args%22%3A%5B1%2C1%5D%7D&filter-lang=cql2-json",
        );
        let (search, metadata) = collection_search("c", &json!({}), &pairs).unwrap();
        assert_eq!(
            search,
            json!({
                "collections": ["c"],
                "filter-lang": "cql2-json",
                "ids": ["a", "b"],
                "bbox": [6.0, 44.0, 8.0, 46.0],
                "datetime": "2024-01-01T00:00:00Z/..",
                "sortby": [
                    { "field": "properties.eo:cloud_cover", "direction": "asc" },
                    { "field": "datetime", "direction": "desc" },
                    { "field": "id", "direction": "asc" }
                ],
                "query": { "eo:cloud_cover": { "lt": 20 } },
                "filter": { "op": "=", "args": [1, 1] }
            })
        );
        // The bbox parameter, not the collection's extent, bounds the mosaic.
        assert_eq!(metadata["bounds"], json!([6.0, 44.0, 8.0, 46.0]));
        assert!(metadata.get("assets").is_none());

        // Renders become defaults, without the keys titiler-pgstac drops.
        let with_renders = json!({ "renders": {
            "rgb": { "title": "True colour", "assets": ["visual"], "minmax_zoom": [8, 14], "tilematrixsets": {} },
            "ndvi": { "expression": "(nir-red)/(nir+red)", "rescale": [[-1, 1]], "minmax_zoom": [8] },
            "broken": "not an object"
        }});
        let (_, metadata) = collection_search("c", &with_renders, &[]).unwrap();
        assert_eq!(
            metadata["defaults"],
            json!({
                "rgb": { "assets": ["visual"], "minzoom": 8, "maxzoom": 14 },
                "ndvi": { "expression": "(nir-red)/(nir+red)", "rescale": [[-1, 1]] }
            })
        );

        let status = |q: &str| collection_search("c", &json!({}), &query_pairs(q)).unwrap_err().status();
        assert_eq!(status("bbox=1,2,3"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status("bbox=1,2,x,4"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status("query=nope"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status("filter=eo:cloud_cover%3C"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status("filter=eo:cloud_cover"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status("filter=a%3D1&filter-lang=cql2-nope"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status("filter=a%3D1&filter-lang=cql2-json"), StatusCode::UNPROCESSABLE_ENTITY);

        // A filter is CQL2 text unless `filter-lang` says otherwise, and is stored as the
        // JSON titiler-pgstac's converter gives: numbers as floats, `filter-lang` JSON.
        for query in [
            "filter=eo:cloud_cover%20%3C%2020%20AND%20platform%20%3D%20%27sentinel-2b%27",
            "filter=eo:cloud_cover%3C20+and+platform%3D%27sentinel-2b%27&filter-lang=cql2-text",
        ] {
            let (search, _) = collection_search("c", &json!({}), &query_pairs(query)).unwrap();
            assert_eq!(
                search,
                json!({
                    "collections": ["c"],
                    "filter-lang": "cql2-json",
                    "filter": { "op": "and", "args": [
                        { "op": "<", "args": [{ "property": "eo:cloud_cover" }, 20.0] },
                        { "op": "=", "args": [{ "property": "platform" }, "sentinel-2b"] }
                    ]}
                }),
                "{query}"
            );
            assert!(search.to_string().contains("20.0"), "{query}");
        }
    }

    #[test]
    fn default_layers_become_link_query_strings() {
        let metadata = json!({ "defaults": {
            "rgb": { "assets": ["visual"], "minzoom": 8, "maxzoom": 14 },
            "ndvi": { "expression": "(nir-red)/(nir+red)", "rescale": [[-1, 1]], "colormap_name": "viridis" },
            "bands": {
                "assets": "red", "asset_bidx": ["red|1"], "rescale": [[0, 3000], "0,1"],
                "colormap": { "1": [0, 0, 0, 255] }, "unscale": true
            }
        }});
        // jsonb order of the names; `ndvi` has no assets, so it cannot be a tile request.
        assert_eq!(
            layer_queries(&metadata, true),
            vec![
                ("rgb".to_string(), "maxzoom=14&minzoom=8&assets=visual".to_string()),
                (
                    "bands".to_string(),
                    "unscale=True&rescale=0%2C3000&rescale=0%2C1&colormap=%7B%221%22%3A+%5B0%2C+0%2C+0%2C+255%5D%7D&assets=red%7Cbidx%3D1".to_string()
                ),
            ]
        );
        // Registration links keep asset_bidx as its own parameter.
        assert!(layer_queries(&metadata, false)[1].1.ends_with("&assets=red&asset_bidx=red%7C1"));
        assert!(layer_queries(&json!({ "name": "x" }), true).is_empty());
    }

    fn utm(epsg: u16) -> CrsModel {
        let keys = cogzilla_tiff::GeoKeys {
            model_type: Some(1),
            projected_cs_type: Some(epsg),
            ..Default::default()
        };
        CrsModel::from_geo_keys(&keys).unwrap()
    }

    #[test]
    fn item_zooms_follow_rio_tiler() {
        // Expected values are rio-tiler 9.4's for the same grids (rasterio's
        // calculate_default_transform, then zoom_for_res).
        let zooms = |crs: &CrsModel, width: f64, height: f64, bounds: [f64; 4]| {
            let e = grid_extent(crs, ProjGrid { bounds, width, height });
            (e.minzoom, e.maxzoom)
        };
        let s2 = [699_960.0, 4_790_220.0, 809_760.0, 4_900_020.0];
        assert_eq!(zooms(&utm(32631), 10980.0, 10980.0, s2), (7, 13));
        assert_eq!(zooms(&utm(32631), 5490.0, 5490.0, s2), (7, 12));
        assert_eq!(zooms(&utm(32631), 1830.0, 1830.0, s2), (8, 11));
        assert_eq!(
            zooms(&utm(32632), 10980.0, 10980.0, [399_960.0, 5_090_220.0, 509_760.0, 5_200_020.0]),
            (7, 13)
        );
        assert_eq!(
            zooms(&utm(32633), 10980.0, 10980.0, [499_980.0, 6_790_200.0, 609_780.0, 6_900_000.0]),
            (7, 13)
        );
        assert_eq!(
            zooms(&utm(32619), 7621.0, 7791.0, [300_000.0, 100_000.0, 528_630.0, 333_730.0]),
            (7, 12)
        );
        assert_eq!(
            zooms(&utm(32755), 10980.0, 10980.0, [300_000.0, 5_690_220.0, 409_800.0, 5_800_020.0]),
            (8, 14)
        );
        assert_eq!(zooms(&CrsModel::Geographic, 3600.0, 1800.0, [-10.0, 30.0, 26.0, 48.0]), (3, 7));
        assert_eq!(
            zooms(&CrsModel::Epsg3857, 4096.0, 2048.0, [500_000.0, 5_300_000.0, 622_304.0, 5_361_152.0]),
            (9, 12)
        );

        // The extent of the Sentinel-2 tile in WGS84, as rasterio's transform_bounds gives it.
        let e = grid_extent(&utm(32631), ProjGrid { bounds: s2, width: 10980.0, height: 10980.0 });
        let want = [5.462735477681616, 43.201211354944526, 6.875738946301351, 44.22597512756446];
        for (got, want) in e.wgs84.iter().zip(want) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        assert!(e.mercator.is_some());
    }

    #[test]
    fn item_grid_comes_from_the_projection_extension() {
        let tiff = "image/tiff; application=geotiff; profile=cloud-optimized";
        let item = json!({
            "properties": { "proj:epsg": 32631 },
            "assets": {
                "visual": { "href": "v.tif", "type": tiff, "proj:shape": [10980, 10980],
                            "proj:transform": [10, 0, 699960, 0, -10, 4900020] },
                "aot": { "href": "a.tif", "type": tiff, "proj:shape": [5490, 5490],
                         "proj:transform": [20, 0, 699960, 0, -20, 4900020] },
                "aot-jp2": { "href": "a.jp2", "type": "image/jp2" },
                "untyped": { "href": "u.bin" },
                "thumbnail": { "href": "t.jpg", "type": "image/jpeg" },
                "granule_metadata": { "href": "m.xml", "type": "application/xml" }
            }
        });
        // jsonb key order, with the media types rio-tiler reads and untyped assets.
        let readable = readable_assets(&item);
        assert_eq!(readable, ["aot", "visual", "aot-jp2", "untyped"]);
        // The union of the assets' grids at the finest asset's resolution.
        let grid = item_proj_grid(&item, &readable).unwrap();
        assert_eq!(grid.bounds, [699_960.0, 4_790_220.0, 809_760.0, 4_900_020.0]);
        assert_eq!((grid.width, grid.height), (10980.0, 10980.0));

        // Item-level fields are used when no asset has its own.
        let item_level = json!({
            "properties": { "proj:code": "EPSG:32631", "proj:shape": [100, 200],
                            "proj:transform": [30, 0, 500000, 0, -30, 4000000] },
            "assets": { "data": { "href": "d.tif" } }
        });
        let grid = item_proj_grid(&item_level, &["data".to_string()]).unwrap();
        assert_eq!(grid.bounds, [500_000.0, 3_997_000.0, 506_000.0, 4_000_000.0]);
        assert_eq!((grid.width, grid.height), (200.0, 100.0));

        // Assets in different coordinate systems, or no projection fields: nothing to go on.
        let mixed = json!({
            "properties": { "proj:epsg": 32631 },
            "assets": { "a": { "href": "a.tif", "proj:epsg": 32632, "proj:shape": [10, 10],
                               "proj:transform": [10, 0, 0, 0, -10, 0] } }
        });
        assert!(item_proj_grid(&mixed, &["a".to_string()]).is_none());
        let bare = json!({ "properties": {}, "assets": { "a": { "href": "a.tif" } } });
        assert!(item_proj_grid(&bare, &["a".to_string()]).is_none());
    }

    #[test]
    fn item_answers_are_reshaped_like_titiler_pgstac() {
        let extent = ItemExtent { wgs84: [5.0, 43.0, 7.0, 44.0], mercator: None, minzoom: 7, maxzoom: 13 };
        let fix = |sub: &[&str], query: &str, body: Value| {
            let (bytes, _) =
                item_fixup(sub, Some(&extent), query, Bytes::from(body.to_string()), "application/json").unwrap();
            serde_json::from_slice::<Value>(&bytes).unwrap()
        };
        let info = json!({ "visual": { "dtype": "uint8", "minzoom": 9, "maxzoom": 14, "wgs84_bounds": [1, 2, 3, 4] } });
        assert_eq!(fix(&["info"], "", info.clone()), json!({ "visual": { "dtype": "uint8" } }));

        let feature = fix(&["info.geojson"], "", json!({ "type": "Feature", "geometry": null, "properties": info }));
        assert_eq!(feature["bbox"], json!([5.0, 43.0, 7.0, 44.0]));
        assert_eq!(feature["geometry"]["coordinates"][0][2], json!([7.0, 44.0]));
        assert_eq!(feature["properties"], json!({ "visual": { "dtype": "uint8" } }));

        let point = fix(&["point", "6,43"], "", json!({ "band_names": ["red_b1", "nir_b1"], "values": [1.0, 2.0] }));
        assert_eq!(point["band_names"], json!(["b1", "b2"]));

        let tj = json!({ "minzoom": 9, "maxzoom": 14, "bounds": [0, 0, 1, 1], "center": [0.5, 0.5, 9] });
        let fixed = fix(&["WebMercatorQuad", "tilejson.json"], "assets=visual", tj.clone());
        assert_eq!((&fixed["minzoom"], &fixed["maxzoom"]), (&json!(7), &json!(13)));
        assert_eq!(fixed["center"], json!([6.0, 43.5, 7]));
        // Zooms given in the query string stand.
        let fixed = fix(&["WebMercatorQuad", "tilejson.json"], "assets=visual&minzoom=10", tj);
        assert_eq!((&fixed["minzoom"], &fixed["maxzoom"]), (&json!(9), &json!(13)));

        // Tiles are passed through; only the media type of `.jpg` changes.
        let jpg = item_fixup(&["tiles", "WebMercatorQuad", "1", "2", "3.jpg"], None, "", Bytes::from_static(b"x"), "image/jpeg").unwrap();
        assert_eq!(jpg.1, "image/jpg");
        let png = item_fixup(&["tiles", "WebMercatorQuad", "1", "2", "3.png"], None, "", Bytes::from_static(b"x"), "image/png").unwrap();
        assert_eq!((png.0.as_ref(), png.1), (b"x".as_ref(), "image/png"));

        assert!(item_route_needs_assets(&["info"]));
        assert!(item_route_needs_assets(&["preview.png"]));
        assert!(item_route_needs_assets(&["tiles", "WebMercatorQuad", "1", "2", "3"]));
        assert!(!item_route_needs_assets(&["assets"]));
        assert!(!item_route_needs_assets(&["tiles", "WebMercatorQuad"]));
        assert!(!item_route_needs_assets(&["nope"]));
    }

    #[test]
    fn wmts_lists_default_layers_and_the_query_layer() {
        let engine = Arc::new(TileEngine::new(1 << 20));
        let raw_query = "assets=visual&rescale=0,255&tilesize=512&tile_format=jpg&zooms=WebMercatorQuad::8,9";
        let ctx = Ctx {
            engine: &engine,
            host: "http://example.test",
            forwarded_prefix: "",
            path: "/searches/abc/WMTSCapabilities.xml",
            raw_query,
            if_none_match: None,
            only: None,
        };
        let source = || WmtsSource {
            title: "abc",
            bounds: [5.0, 43.5, 12.0, 48.0],
            minzoom: 7,
            maxzoom: 8,
            default_layers: vec![("rgb".into(), "tilesize=256&assets=visual".into())],
        };
        // The body is already in memory, so collecting it is ready on the first poll.
        fn ready<F: std::future::Future>(f: F) -> F::Output {
            let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
            match std::pin::pin!(f).as_mut().poll(&mut cx) {
                std::task::Poll::Ready(v) => v,
                std::task::Poll::Pending => panic!("body was not ready"),
            }
        }
        let body = |resp: Resp| {
            use http_body_util::BodyExt;
            let bytes = ready(resp.into_body().collect()).unwrap().to_bytes();
            String::from_utf8(bytes.to_vec()).unwrap()
        };

        let resp = wmts(&ctx, "/searches/abc", source(), &query_pairs(raw_query));
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()[CONTENT_TYPE], "application/xml");
        let xml = body(resp);
        // The layer from `defaults` keeps the mosaic's zooms and loses its tile size.
        assert!(xml.contains("<ows:Identifier>abc_WebMercatorQuad_rgb</ows:Identifier>"));
        assert!(xml.contains(r#"template="http://example.test/searches/abc/tiles/WebMercatorQuad/{TileMatrix}/{TileCol}/{TileRow}.jpg?assets=visual" />"#));
        assert!(xml.contains("<TileMatrix>7</TileMatrix>"));
        // The query-string layer drops the WMTS-only parameters and takes the zoom override.
        assert!(xml.contains("<ows:Identifier>abc_WebMercatorQuad_default</ows:Identifier>"));
        assert!(xml.contains(r#"{TileRow}.jpg?assets=visual&amp;rescale=0%2C255" />"#));
        assert!(xml.contains("<TileMatrix>9</TileMatrix>"));
        assert!(xml.contains("<MinTileRow>177</MinTileRow>"));
        assert_eq!(xml.matches("<Layer>").count(), 2);
        assert_eq!(xml.matches("<Format>image/jpg</Format>").count(), 2);
        assert!(xml.contains("<ows:LowerCorner>5.0 43.5</ows:LowerCorner>"));
        assert!(xml.contains("<ows:SupportedCRS>http://www.opengis.net/def/crs/EPSG/0/3857</ows:SupportedCRS>"));
        assert!(xml.contains("<ScaleDenominator>559082264.0287178</ScaleDenominator>"));
        assert!(xml.contains("<ScaleDenominator>33.323899747652874</ScaleDenominator>"));
        assert!(xml.contains("WMTSCapabilities.xml?assets=visual&amp;rescale=0,255&amp;tilesize=512"));

        // Nothing that can make a tile, an unknown format, and a format these routes cannot write.
        let bare = WmtsSource { default_layers: Vec::new(), ..source() };
        assert_eq!(wmts(&ctx, "/searches/abc", bare, &[]).status(), StatusCode::BAD_REQUEST);
        let status = |q: &str| wmts(&ctx, "/searches/abc", source(), &query_pairs(q)).status();
        assert_eq!(status("tile_format=nope"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status("tile_format=jp2"), StatusCode::BAD_REQUEST);
        assert_eq!(status("tile_format=tif"), StatusCode::OK);
        assert_eq!(status("use_epsg=maybe"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status("zooms=WebMercatorQuad"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status(""), StatusCode::OK);
    }

    #[test]
    fn an_item_preview_with_a_side_of_zero_is_sized_from_the_query() {
        let sized = |dims: &str| {
            let mut pq = crate::http::parse_query(Some("max_size=32"));
            let path = item_preview_by_query(&["preview", dims], &mut pq);
            (path, pq.width, pq.height, pq.max_size)
        };
        assert_eq!(sized("0x64.png"), (Some("preview.png".into()), None, Some(64), 32));
        assert_eq!(sized("100x0.jpg"), (Some("preview.jpg".into()), Some(100), None, 32));
        assert_eq!(sized("0x0.tif"), (Some("preview.tif".into()), None, None, 32));
        // Both sides given, no format, a side too large or not a number: as it was.
        for dims in ["64x64.png", "0x64", "0x5000.png", "0x-1.png", "ax0.png", "64.png"] {
            assert_eq!(sized(dims), (None, None, None, 32), "{dims}");
        }
        let mut pq = crate::http::parse_query(None);
        assert_eq!(item_preview_by_query(&["bbox", "0x64.png"], &mut pq), None);
        assert_eq!(item_preview_by_query(&["preview", "0x64.png", "more"], &mut pq), None);
    }

    #[test]
    fn selected_values_are_painted_with_the_requests_own_rescale() {
        let nan = f32::NAN;
        // One band through a colour map: 0 to 100 onto 0 to 255, the fraction dropped.
        let gray = Paint::of(&BandRequest::Single { band_1based: 1, rescale: (0.0, 100.0), colormap: ColormapName::Grayscale }).unwrap();
        let rgba = gray.rgba(&Selected { bands: 1, values: vec![0.0, 50.0, 100.0, 250.0, -5.0, nan] }, 6);
        assert_eq!(rgba, [0, 0, 0, 255, 127, 127, 127, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0]);
        let viridis = Paint::of(&BandRequest::Single { band_1based: 1, rescale: (0.0, 255.0), colormap: ColormapName::Viridis }).unwrap();
        let rgba = viridis.rgba(&Selected { bands: 1, values: vec![0.0, 255.0] }, 2);
        assert_eq!(rgba[..4], [ColormapName::Viridis.lut().rgb[0][0], ColormapName::Viridis.lut().rgb[0][1], ColormapName::Viridis.lut().rgb[0][2], 255]);
        assert_eq!(rgba[4..7], ColormapName::Viridis.lut().rgb[255]);

        // Three bands, each with its own range. A pixel is there when any band of it is.
        let rgb = Paint::of(&BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0, 0.0, 0.0], rescale_max: [255.0, 100.0, 10.0] }).unwrap();
        let selected = Selected { bands: 3, values: vec![255.0, nan, nan, 50.0, 50.0, nan, 5.0, nan, nan] };
        assert_eq!(rgb.rgba(&selected, 3), [255, 127, 127, 255, 0, 127, 0, 255, 0, 0, 0, 0]);
        // A count is one band whatever was asked for: grey on the first band's range.
        let count = Selected { bands: 1, values: vec![0.0, 255.0] };
        assert_eq!(rgb.rgba(&count, 2), [0, 0, 0, 255, 255, 255, 255, 255]);

        let terrain = BandRequest::Terrain { band_1based: 1, algorithm: cogzilla_kernel::DemAlgorithm::Terrarium };
        assert!(Paint::of(&terrain).is_none());
    }

    #[test]
    fn bands_are_painted_as_titiler_encodes_them() {
        let nan = f32::NAN;
        let ranges = [(0.0, 100.0), (0.0, 10.0), (0.0, 1.0)];
        // One range is for every band, several are one each, none is the default.
        let one = Selected { bands: 1, values: vec![50.0, nan] };
        assert_eq!(paint_bands(&one, 2, &ranges[..1], (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap(), [127, 127, 127, 255, 0, 0, 0, 0]);
        assert_eq!(paint_bands(&one, 2, &[], (0.0, 50.0), ColormapName::Grayscale, None, None).unwrap()[..4], [255, 255, 255, 255]);
        // A colour map of the request's own: classes over the values themselves, or over
        // the byte `rescale` makes of them, or intervals; nothing where there is no value
        // or no class for it.
        let map = |json: &str| MosaicColormap::parse(json).unwrap().unwrap();
        let classes = map(r#"{"50": [1, 2, 3, 200], "127": [9, 9, 9, 255], "700": [7, 7, 7, 255]}"#);
        assert_eq!(paint_bands(&one, 2, &[], (0.0, 1.0), ColormapName::Grayscale, Some(&classes), None).unwrap(), [1, 2, 3, 200, 0, 0, 0, 0]);
        assert_eq!(paint_bands(&one, 2, &ranges[..1], (0.0, 1.0), ColormapName::Grayscale, Some(&classes), None).unwrap()[..4], [9, 9, 9, 255]);
        // A class is for the value that equals it: none for one beside it, or with no class.
        let near = Selected { bands: 1, values: vec![50.4, 70.0, 700.0] };
        assert_eq!(paint_bands(&near, 3, &[], (0.0, 1.0), ColormapName::Grayscale, Some(&classes), None).unwrap(), [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 255]);
        // Intervals: by the value, or under `rescale` by the byte, and the last given wins.
        let spans = map("[[[0, 40], [1, 1, 1, 255]], [[40, 60], [2, 2, 2, 255]], [[45, 55], [3, 3, 3, 255]], [[120, 130], [4, 4, 4, 255]]]");
        assert_eq!(paint_bands(&one, 2, &[], (0.0, 1.0), ColormapName::Grayscale, Some(&spans), None).unwrap()[..4], [3, 3, 3, 255]);
        assert_eq!(paint_bands(&one, 2, &ranges[..1], (0.0, 1.0), ColormapName::Grayscale, Some(&spans), None).unwrap()[..4], [4, 4, 4, 255]);
        // Classes for every byte take the value as a byte: cut towards zero, round the clock.
        let bytes: Vec<String> = (0..256).map(|k| format!(r#""{k}": [{k}, 0, 0, 255]"#)).collect();
        let bytes = map(&format!("{{{}}}", bytes.join(",")));
        let wide = Selected { bands: 1, values: vec![300.9, -1.0, 7.9] };
        assert_eq!(paint_bands(&wide, 3, &[], (0.0, 1.0), ColormapName::Grayscale, Some(&bytes), None).unwrap(), [44, 0, 0, 255, 255, 0, 0, 255, 7, 0, 0, 255]);
        // A map with nothing in it is no map, and one with too many intervals is refused.
        assert!(MosaicColormap::parse("{}").unwrap().is_none() && MosaicColormap::parse("[]").unwrap().is_none());
        let many: Vec<String> = (0..257).map(|k| format!("[[{k}, {}], [1, 1, 1, 255]]", k + 1)).collect();
        assert!(MosaicColormap::parse(&format!("[{}]", many.join(","))).is_err() && MosaicColormap::parse("{nope").is_err());
        let three = Selected { bands: 3, values: vec![50.0, 5.0, 0.5] };
        assert_eq!(paint_bands(&three, 1, &ranges, (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap(), [127, 127, 127, 255]);
        assert_eq!(paint_bands(&three, 1, &ranges[..1], (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap(), [127, 12, 1, 255]);
        // Two bands: red and green, and the mask for blue; never transparent.
        let two = Selected { bands: 2, values: vec![50.0, nan, nan, 5.0, 10.0, nan] };
        assert_eq!(
            paint_bands(&two, 3, &ranges[..2], (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap(),
            [127, 127, 255, 255, 0, 255, 255, 255, 0, 0, 0, 255]
        );
        // Four bands and more are no picture.
        let four = Selected { bands: 4, values: vec![1.0; 4] };
        assert_eq!(paint_bands(&four, 1, &[], (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap_err(), 4);
    }

    #[test]
    fn band_math_is_held_to_what_a_tile_needs() {
        let bands = [vec![1.0f32; 4]];
        // Results, length and nesting all come from a URL.
        let many = vec!["b1"; super::super::point_expr::MAX_RESULTS + 1].join(";");
        assert!(band_math(&many, &bands).unwrap_err().to_string().contains("results"));
        let long = format!("b1{}", "+b1".repeat(super::super::point_expr::MAX_EXPRESSION_BYTES / 3));
        assert!(band_math(&long, &bands).unwrap_err().to_string().contains("longer"));
        let deep = format!("{}b1{}", "(".repeat(1000), ")".repeat(1000));
        assert!(band_math(&deep, &bands).unwrap_err().to_string().contains("nests"));
        let minus = format!("{}b1", "-".repeat(1000));
        assert!(band_math(&minus, &bands).unwrap_err().to_string().contains("nests"));
        // What band math is in practice is far inside the limits.
        let chain = format!("b1{}", "+b1".repeat(500));
        assert_eq!(band_math(&chain, &bands).unwrap()[0][0], 501.0);
        assert_eq!(band_math("((((b1+1)*2)-3)/(b1+1))**2", &bands).unwrap()[0][0], 0.25);
    }

    #[test]
    fn bytes_are_rio_tilers_to_the_level() {
        // Values on a boundary are where another order of arithmetic lands a level off:
        // 1600 of 4000 is 102 exactly, 51 of 255 is 51.
        assert_eq!(rescale_byte(1600.0, 0.0, 4000.0), 102);
        assert_eq!(rescale_byte(51.0, 0.0, 255.0), 51);
        assert_eq!(rescale_byte(1168.0, 0.0, 4000.0), 74);
        assert_eq!(rescale_byte(-5.0, 0.0, 100.0), 0);
        assert_eq!(rescale_byte(500.0, 0.0, 100.0), 255);
        // A range of nothing is not a panic; one that runs backwards is the ramp inverted.
        assert_eq!(rescale_byte(3.0, 3.0, 3.0), 0);
        assert_eq!(rescale_byte(3.0, 10.0, 0.0), 178);
        // Grey is the byte itself, for every level.
        let grey = Paint::Gray { min: 0.0, max: 255.0, colormap: ColormapName::Grayscale };
        let all: Vec<f32> = (0..=255).map(|v| v as f32).collect();
        let rgba = grey.rgba(&Selected { bands: 1, values: all }, 256);
        for (level, px) in rgba.chunks_exact(4).enumerate() {
            assert_eq!(px, [level as u8, level as u8, level as u8, 255]);
        }
    }

    #[test]
    fn band_math_is_over_whole_planes() {
        let bands = [vec![1.0f32, 2.0, f32::NAN, 4.0], vec![3.0, 0.0, 1.0, 4.0]];
        let out = band_math("(b2-b1)/(b2+b1);b1/b2", &bands).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0][0], 0.5);
        assert_eq!(out[0][1], -1.0);
        assert!(out[0][2].is_nan(), "no data in, no data out");
        assert_eq!(out[0][3], 0.0);
        assert_eq!(out[1][1], f32::INFINITY, "a division by zero is the largest number, as numpy's nan_to_num makes it");
        assert!(band_math("b3", &bands).is_err());
    }

    #[test]
    fn part_sizes_follow_rio_tiler() {
        // A 0.2 by 0.1 degree box over data whose resolution makes it 1802 by 901 pixels.
        let bbox = [8.4, 45.7, 8.6, 45.8];
        let res = 0.2 / 1802.0;
        let query = |max_size, width, height| PartSize::Query { max_size, width, height };
        assert_eq!(part_dims(query(None, None, None), bbox, res), (1802, 901));
        assert_eq!(part_dims(query(Some(256), None, None), bbox, res), (256, 128));
        assert_eq!(part_dims(query(None, Some(300), None), bbox, res), (300, 150));
        assert_eq!(part_dims(query(None, None, Some(90)), bbox, res), (180, 90));
        assert_eq!(part_dims(query(None, Some(300), Some(200)), bbox, res), (300, 200));
        assert_eq!(part_dims(PartSize::Exact(64, 48), bbox, res), (64, 48));
        // `max_size` only shrinks: a box smaller than it keeps the data's resolution.
        assert_eq!(part_dims(query(Some(4096), None, None), bbox, res), (1802, 901));
        // A tall box is limited by its height.
        assert_eq!(part_dims(query(Some(100), None, None), [0.0, 0.0, 0.1, 0.2], res), (50, 100));

        // The lean lookup samples the box as the picture will: the same sizes, wherever
        // they do not depend on the data.
        let grid = |w, h| SampleGrid::for_image(w, h, false);
        assert_eq!(part_grid(query(Some(256), None, None), bbox), grid(256, 128));
        assert_eq!(part_grid(query(None, Some(300), None), bbox), grid(300, 150));
        assert_eq!(part_grid(query(None, None, Some(90)), bbox), grid(180, 90));
        assert_eq!(part_grid(query(None, Some(300), Some(200)), bbox), grid(300, 200));
        assert_eq!(part_grid(PartSize::Exact(64, 48), bbox), grid(64, 48));
        assert_eq!(part_grid(query(Some(100), None, None), [0.0, 0.0, 0.1, 0.2]), grid(50, 100));
        assert!(grid(64, 48).is_some());
        // The data's own resolution is not known yet, and a picture this large is not sampled.
        assert_eq!(part_grid(query(None, None, None), bbox), None);
        assert_eq!(part_grid(PartSize::Exact(5000, 10), bbox), None);
        assert_eq!(part_grid(PartSize::Exact(64, 48), [0.0, 0.0, 0.0, 1.0]), None);

        // `max_size` is dropped once a width or a height is given, as in rio-tiler.
        let pairs = query_pairs("max_size=64&width=300");
        assert_eq!(part_size_from_query(&pairs).unwrap(), query(None, Some(300), None));
        assert_eq!(part_size_from_query(&query_pairs("max_size=64")).unwrap(), query(Some(64), None, None));
        assert!(part_size_from_query(&query_pairs("width=wide")).is_err());
        assert!(part_size_from_query(&query_pairs("height=0")).is_err());
    }

    #[test]
    fn item_renders_are_presented_like_titiler() {
        let item = json!({ "properties": { "renders": {
            "ndvi": { "title": "NDVI", "assets": ["red", "nir"], "expression": "(nir-red)/(nir+red)",
                      "rescale": [[-1, 1]], "colormap_name": "viridis" },
            "one_band": { "assets": ["red"], "asset_bidx": ["red|1"], "minmax_zoom": [8, 12] },
            "no_assets": { "expression": "b1*2" },
            "rgb": { "assets": "visual" }
        }}});
        let renders = item_renders(&item);
        // jsonb key order; a single asset name is a list of one.
        let names: Vec<&str> = renders.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["rgb", "ndvi", "one_band", "no_assets"]);
        assert_eq!(renders[0].params["assets"], json!(["visual"]));
        assert_eq!(renders[2].params["assets"], json!(["red|bidx=1"]));
        assert!(renders[2].params.get("asset_bidx").is_none());
        assert!(!renders[3].valid && renders[1].valid);

        // The query string is Python's urlencode of the render, lists repeated.
        assert_eq!(
            urlencode_doseq(&renders[1].params),
            "title=NDVI&assets=red&assets=nir&rescale=%5B-1%2C+1%5D&expression=%28nir-red%29%2F%28nir%2Bred%29&colormap_name=viridis"
        );
        assert_eq!(urlencode_doseq(&renders[2].params), "assets=red%7Cbidx%3D1&minmax_zoom=8&minmax_zoom=12");

        let doc = item_render_json("http://h", "/collections/c/items/i", "i", &renders[1]);
        assert_eq!(doc["valid"], true);
        assert_eq!(doc["links"][0]["href"], "http://h/collections/c/items/i/renders/ndvi?url=%3CItem+id%3Di%3E");
        assert_eq!(doc["links"][1]["rel"], "tilesets-map");
        assert!(doc["links"][1]["href"].as_str().unwrap().starts_with(
            "http://h/collections/c/items/i/%7BtileMatrixSetId%7D/tilejson.json?url=%3CItem+id%3Di%3E&title=NDVI&"
        ));
        assert!(doc["links"][2]["href"].as_str().unwrap().contains("/WMTSCapabilities.xml?url="));
        // A render that cannot make a tile has only its own link.
        let doc = item_render_json("http://h", "/collections/c/items/i", "i", &renders[3]);
        assert_eq!((doc["valid"].clone(), doc["links"].as_array().unwrap().len()), (json!(false), 1));
        assert!(item_renders(&json!({ "properties": {} })).is_empty());
    }

    #[test]
    fn point_values_take_band_math() {
        let values = [json!(1060.0), json!(2386.0)];
        let names = ["red_b1".to_string(), "nir_b1".to_string()];
        let (v, band_names, described) = point_expression("(b2-b1)/(b2+b1);b1*2", &values, &names).unwrap();
        assert_eq!(v, [json!(0.3847939640162507), json!(2120.0)]);
        assert_eq!(band_names, ["b1", "b2"]);
        assert_eq!(described, ["(nir_b1-red_b1)/(nir_b1+red_b1)", "red_b1*2"]);
        // A band without a value gives an empty result, and asset names are not band names.
        let (v, _, _) = point_expression("b1+b2", &[Value::Null, json!(1.0)], &names).unwrap();
        assert_eq!(v, [Value::Null]);
        assert_eq!(point_expression("nir-red", &values, &names).unwrap_err(), "Invalid band/asset name 'nir'");

        // On item routes the expression is applied to the `/stac` handler's answer.
        let body = json!({ "coordinates": [1.0, 2.0], "values": [1060.0, 2386.0],
                           "band_names": ["red_b1", "nir_b1"], "band_descriptions": ["red_b1", "nir_b1"] });
        let fix = |query: &str| item_fixup(&["point", "1,2"], None, query, Bytes::from(body.to_string()), "application/json");
        let (bytes, _) = fix("assets=red&assets=nir&expression=b2-b1").unwrap();
        let doc: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(doc["values"], json!([1326.0]));
        assert_eq!(doc["band_names"], json!(["b1"]));
        assert_eq!(doc["band_descriptions"], json!(["nir_b1-red_b1"]));
        assert_eq!(fix("assets=red&expression=nope").unwrap_err().status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn statistics_grid_follows_gdal() {
        // A Sentinel-2 10 m raster: full resolution and four overviews.
        let levels = [(10980, 10980), (5490, 5490), (2745, 2745), (1373, 1373), (687, 687)];
        // Read at its own size, or a little reduced: the full resolution.
        assert_eq!(gdal_overview_level(&levels, (541, 542), (541, 542)), 0);
        assert_eq!(gdal_overview_level(&levels, (541, 542), (512, 512)), 0);
        // Reduced 1.8 times: the half-size level is within 1.2 times of that.
        assert_eq!(gdal_overview_level(&levels, (541, 542), (300, 300)), 1);
        // Reduced 8.45 times: the eighth, not the sixteenth.
        assert_eq!(gdal_overview_level(&levels, (541, 542), (64, 64)), 3);
        // The less reduced axis decides.
        assert_eq!(gdal_overview_level(&levels, (1000, 4000), (500, 500)), 1);
        assert_eq!(gdal_overview_level(&levels, (10000, 10000), (100, 100)), 4);
        assert_eq!(gdal_overview_level(&levels[..1], (1000, 1000), (10, 10)), 0);

        // The box grows outwards to whole pixels of the grid.
        let grid = Wgs84Grid { west: 0.0, north: 10.0, res: 0.5 };
        assert_eq!(grid.align([0.6, 8.1, 1.9, 9.4]), [0.5, 8.0, 2.0, 9.5]);
        assert_eq!(grid.align([0.5, 8.0, 2.0, 9.5]), [0.5, 8.0, 2.0, 9.5]);

        // Nearest-neighbour resize, both ways.
        let plane = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(resize_nearest(plane.clone(), (4, 2), (2, 1)), [6.0, 8.0]);
        assert_eq!(resize_nearest(plane.clone(), (4, 2), (2, 2)), [2.0, 4.0, 6.0, 8.0]);
        assert_eq!(resize_nearest(vec![1.0, 2.0], (2, 1), (4, 1)), [1.0, 1.0, 2.0, 2.0]);
        assert_eq!(resize_nearest(plane.clone(), (4, 2), (4, 2)), plane);
    }

    #[test]
    fn scanlines_are_approximated_like_gdal() {
        // A straight row is its two ends joined, whatever lies between.
        let mut out = vec![(0.0, 0.0); 100];
        approx_scanline(&|i| (i as f64 * 2.0 + 1.0, 7.0 - i as f64 * 0.01), &mut out, 0.125);
        assert!((out[50].0 - 101.0).abs() < 1e-9 && (out[50].1 - 6.5).abs() < 1e-9);

        // A gently curved row is still one straight line: the middle is 0.1 off, under
        // the eighth of a pixel GDAL allows.
        let bend = |i: usize| (i as f64, 0.4 * (i as f64 / 99.0) * (1.0 - i as f64 / 99.0));
        approx_scanline(&bend, &mut out, 0.125);
        assert_eq!(out[49].1, 0.0);
        assert!((bend(49).1 - 0.1).abs() < 0.001);

        // A sharper curve is split until every piece is within the allowance.
        let curve = |i: usize| (i as f64, 40.0 * (i as f64 / 99.0) * (1.0 - i as f64 / 99.0));
        approx_scanline(&curve, &mut out, 0.125);
        for (i, got) in out.iter().enumerate() {
            assert!((got.1 - curve(i).1).abs() <= 0.125, "column {i}: {} vs {}", got.1, curve(i).1);
        }
        // Five points or fewer are worked out one by one, and so is a row that leaves
        // the projection.
        let mut short = vec![(0.0, 0.0); 5];
        approx_scanline(&curve, &mut short, 0.125);
        assert_eq!(short[2], curve(2));
        let broken = |i: usize| if i < 30 { (f64::NAN, f64::NAN) } else { (i as f64, 1.0) };
        approx_scanline(&broken, &mut out, 0.125);
        assert!(out[10].0.is_nan());
        assert_eq!(out[60], (60.0, 1.0));

        // A picture's rows are held to a tolerance at which, with the positions near a
        // source pixel's edge worked out again, every pixel is the projection's own.
        let wave = |i: usize| (i as f64 * 0.37, 300.0 * ((i as f64) / 255.0).powi(2) + (i as f64 * 0.05).sin());
        let mut row = vec![(0.0, 0.0); 256];
        curved_scanline(&wave, &mut row, PICTURE_TOLERANCE);
        for (i, drawn) in row.iter().enumerate() {
            let exact = wave(i);
            let off = (drawn.0 - exact.0).abs().max((drawn.1 - exact.1).abs());
            assert!(off <= 2.0 * PICTURE_TOLERANCE, "column {i} is {off} off");
            // So a drawn position names another pixel than the exact one only near an edge.
            if (drawn.0.floor(), drawn.1.floor()) != (exact.0.floor(), exact.1.floor()) {
                assert!(near_pixel_edge(drawn.0) || near_pixel_edge(drawn.1), "column {i}");
            }
        }
        assert!(near_pixel_edge(7.001) && near_pixel_edge(7.999) && !near_pixel_edge(7.5) && !near_pixel_edge(7.01));

        // A row that bends evenly is one curve: three positions and the two it is checked at.
        let asked = std::cell::Cell::new(0);
        let bow = |i: usize| {
            asked.set(asked.get() + 1);
            (i as f64 * 0.9, 50.0 * (i as f64 / 255.0).powi(2))
        };
        curved_scanline(&bow, &mut row, PICTURE_TOLERANCE);
        assert_eq!(asked.get(), 5);
        assert!((row[100].1 - 50.0 * (100.0f64 / 255.0).powi(2)).abs() < 1e-9);
        // One that leaves the projection is worked out column by column.
        let broken = |i: usize| if i < 30 { (f64::NAN, f64::NAN) } else { (i as f64, 1.0) };
        curved_scanline(&broken, &mut row, PICTURE_TOLERANCE);
        assert!(row[10].0.is_nan());
        assert_eq!(row[60], (60.0, 1.0));
    }

    /// Every way a picture's row can be asked of a projection that could trip the curve:
    /// each family of projection, at home and far outside its area, under tiles that span
    /// the world, sit on the antimeridian or reach the poles, with rasters turned, tiny,
    /// coarse and far finer than the picture. The pixel taken must be the projection's own.
    #[test]
    fn pictures_take_the_projections_pixel_in_hostile_cases() {
        use cogzilla_proj::{Ellipsoid, Sinusoidal};
        use cogzilla_raster::Affine2D;

        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut pick = |n: usize| (next() % n as u64) as usize;

        let mut systems: Vec<(String, CrsModel, (f64, f64))> = Vec::new();
        for (code, home) in [
            (3857u16, (10.0, 45.0)), (3395, (10.0, 45.0)), (4326, (10.0, 45.0)),
            (32601, (-177.0, 10.0)), (32631, (3.0, 50.0)), (32660, (177.0, 70.0)), (32733, (15.0, -30.0)), (32760, (177.0, -79.0)),
            (3031, (0.0, -80.0)), (3413, (-45.0, 75.0)), (3995, (0.0, 85.0)), (3035, (10.0, 52.0)), (3575, (10.0, 80.0)),
            (5070, (-96.0, 37.0)), (3338, (-154.0, 60.0)), (2154, (3.0, 46.5)), (3978, (-95.0, 60.0)), (2193, (173.0, -41.0)), (3006, (15.0, 62.0)),
        ] {
            match CrsModel::from_epsg(code) {
                Ok(crs) => systems.push((format!("EPSG:{code}"), crs, home)),
                Err(e) => panic!("EPSG:{code}: {e:?}"),
            }
        }
        systems.push(("sinusoidal".into(), CrsModel::Sinusoidal(Sinusoidal::new(Ellipsoid::sphere(6_371_007.181), 0.0, 0.0, 0.0)), (20.0, 40.0)));

        let world = cogzilla_proj::ORIGIN_SHIFT;
        // `HOSTILE_CASES` runs more of them: twenty thousand is half a billion pixels.
        let cases: usize = std::env::var("HOSTILE_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
        let (mut pixels, mut worst_off) = (0u64, 0f64);
        for case in 0..cases {
            let (name, crs, natural) = &systems[pick(systems.len())];
            // Where the raster is: at home, or somewhere the projection is strained.
            let place = pick(6);
            let home = match place {
                0 | 1 => *natural,
                2 => (natural.0 + 60.0, natural.1),
                3 => (179.9, natural.1),
                4 => (natural.0, 84.0f64.copysign(natural.1)),
                _ => (natural.0 - 90.0, -natural.1),
            };
            // What is asked: a map tile or a box in degrees.
            let mercator = pick(3) != 0;
            let view = pick(7);
            let (bounds, view_name): ([f64; 4], String) = if mercator {
                let z = [0u8, 1, 2, 3, 5, 8, 12, 16, 20][pick(9)];
                let n = 1u32 << z;
                let (mx, my) = lonlat_deg_to_mercator(home.0.clamp(-179.99, 179.99), home.1);
                let hx = (((mx + world) / (2.0 * world) * f64::from(n)) as u32).min(n - 1);
                let hy = (((world - my) / (2.0 * world) * f64::from(n)) as u32).min(n - 1);
                let (x, y, what) = match view {
                    0..=2 => (hx, hy, "home"),
                    3 => ((hx + 1) % n, hy, "beside"),
                    4 => (0, hy, "west edge"),
                    5 => (n - 1, hy, "east edge"),
                    _ => (hx, if home.1 > 0.0 { 0 } else { n - 1 }, "pole row"),
                };
                let (a, b, c, d) = tile_bounds_3857(TileCoord { z, x, y });
                ([a, b, c, d], format!("tile z{z} {what}"))
            } else {
                match view {
                    0 | 1 => ([home.0 - 0.5, home.1 - 0.5, home.0 + 0.5, home.1 + 0.5], "box at home".into()),
                    2 => ([home.0 - 0.001, home.1 - 0.001, home.0 + 0.001, home.1 + 0.001], "tiny box".into()),
                    3 => ([-180.0, -90.0, 180.0, 90.0], "whole world".into()),
                    4 => ([-200.0, -95.0, 200.0, 95.0], "past the world".into()),
                    5 => ([170.0, home.1 - 10.0, 190.0, home.1 + 10.0], "over the antimeridian".into()),
                    _ => ([-180.0, 80.0f64.copysign(home.1).min(90.0f64.copysign(home.1)), 180.0, 80.0f64.copysign(home.1).max(90.0f64.copysign(home.1))], "polar cap".into()),
                }
            };
            let (grid_w, grid_h) = [(256, 256), (512, 512), (1024, 1024), (4096, 64), (37, 1000), (10, 10), (2048, 2048), (1531, 311)][pick(8)];
            let block = [grid_w, 512.min(grid_w), 100.min(grid_w)][pick(3)];
            let (dx, dy) = ((bounds[2] - bounds[0]) / grid_w as f64, (bounds[3] - bounds[1]) / grid_h as f64);
            let project = |across: f64, up: f64| {
                if mercator && matches!(crs, CrsModel::Epsg3857) {
                    return (across, up);
                }
                let (lon, lat) = if mercator { mercator_to_lonlat_deg(across, up) } else { (across, up) };
                crs.from_lonlat_deg(lon, lat)
            };
            // The raster: its pixels a chosen number of times the picture's, turned, and
            // either endless or small enough that its edges cross the picture.
            let centre = {
                let h = if mercator && !matches!(crs, CrsModel::Epsg3857) { crs.from_lonlat_deg(home.0, home.1) } else if mercator { lonlat_deg_to_mercator(home.0, home.1) } else { crs.from_lonlat_deg(home.0, home.1) };
                if h.0.is_finite() && h.1.is_finite() { h } else { (0.0, 0.0) }
            };
            let (mid_x, mid_y) = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0);
            let (p, q) = (project(mid_x, mid_y), project(mid_x + dx, mid_y));
            let step = (q.0 - p.0).hypot(q.1 - p.1);
            let step = if step.is_finite() && step > 0.0 { step } else { 10.0 };
            let ratio = [0.05, 0.7, 1.0, 1.7, 30.0][pick(5)];
            let res = step * ratio;
            let turn = [0.0f64, 0.0, 30.0, 90.0, -137.0][pick(5)].to_radians();
            let side = [1u32 << 20, 1000, 257][pick(3)];
            let half = f64::from(side) / 2.0;
            let (cos, sin) = (turn.cos() * res, turn.sin() * res);
            // pixel -> crs: turned, north up when not turned, the home at the middle.
            let to_crs = Affine2D { a: cos, b: sin, c: centre.0 - cos * half - sin * half, d: sin, e: -cos, f: centre.1 - sin * half + cos * half };
            let Some(to_pixel) = to_crs.invert() else { continue };

            let rows: Vec<usize> = if grid_h <= 24 { (0..grid_h).collect() } else { (0..24).map(|k| k * (grid_h - 1) / 23).collect() };
            let mut row_positions = vec![(0f64, 0f64); grid_w];
            for &j in &rows {
                let up = bounds[3] - (j as f64 + 0.5) * dy;
                let exact = |i: usize| {
                    let (x, y) = project(bounds[0] + (i as f64 + 0.5) * dx, up);
                    to_pixel.forward(x, y)
                };
                row_source_positions(&exact, block, true, &mut row_positions);
                for (i, &drawn) in row_positions.iter().enumerate() {
                    let real = exact(i);
                    let got = source_cell(drawn, (side, side), true, || exact(i));
                    let want = source_cell(real, (side, side), true, || real);
                    assert_eq!(
                        got, want,
                        "case {case}: {name} raster at {home:?}, {view_name}, {grid_w}x{grid_h} in stretches of {block}, \
                         pixels {ratio} times the picture's, turned {:.0}, {side} a side: row {j} column {i} drawn {drawn:?}, exactly {real:?}",
                        turn.to_degrees()
                    );
                    pixels += 1;
                    // How far a drawn position strays matters where it could name a pixel.
                    let reach = f64::from(side) + 2.0;
                    if drawn.0.is_finite() && drawn.1.is_finite() && real.0 > -2.0 && real.1 > -2.0 && real.0 < reach && real.1 < reach {
                        worst_off = worst_off.max((drawn.0 - real.0).abs().max((drawn.1 - real.1).abs()));
                    }
                }
            }
        }
        assert!(pixels > 1_000_000, "{pixels} pixels");
        // The check at a source pixel's edge allows for twice the tolerance.
        assert!(worst_off <= 2.0 * PICTURE_TOLERANCE, "a position was drawn {} tolerances out", worst_off / PICTURE_TOLERANCE);
    }

    /// What a row may do that no map projection does smoothly: jump (a wrapped longitude),
    /// turn a corner (a clamped latitude), or leave the projection at a single column.
    /// Wherever the curve is drawn all the same, it must be within the allowance the edge
    /// check is built on, or the pixel next door could be taken.
    #[test]
    fn rows_that_jump_turn_or_break_are_not_drawn_across() {
        let allowed = 2.0 * PICTURE_TOLERANCE;
        let (mut worst_jump, mut worst_kink) = (0f64, 0f64);
        for &points in &[10usize, 11, 64, 255, 256, 257, 512, 1000] {
            let n = (points - 1) as f64;
            let mut row = vec![(0.0, 0.0); points];
            // A jump of every size around the tolerance, after every column, on a row that
            // is otherwise straight, bowed, or bent so as to hide the jump from the check.
            for at in 0..points - 1 {
                for &jump in &[0.2, 0.5, 0.9, 1.0, 1.5, 2.0, 2.6, 3.0, 5.0, 7.9, 12.0, 100.0] {
                    for &bend in &[0.0, 1.0, -2.0 / 3.0, 2.0 / 3.0, -2.0, 6.0] {
                        let jump = jump * PICTURE_TOLERANCE;
                        let real = |i: usize| {
                            let x = i as f64 / n;
                            // A cubic through the three fitted columns, sized against the jump.
                            let hidden = bend * jump * 8.0 * x * (x - 0.5) * (x - 1.0);
                            let step = if i > at { jump } else { 0.0 };
                            (i as f64 * 0.83 + 0.4 * x * x + step + hidden, 100.0 + 3.0 * x - step)
                        };
                        curved_scanline(&real, &mut row, PICTURE_TOLERANCE);
                        for (i, drawn) in row.iter().enumerate() {
                            let want = real(i);
                            worst_jump = worst_jump.max((drawn.0 - want.0).abs().max((drawn.1 - want.1).abs()));
                        }
                    }
                }
                // A corner at every column.
                for &turn in &[0.5, 1.0, 3.0, 10.0, 40.0, 1000.0] {
                    let turn = turn * PICTURE_TOLERANCE / n;
                    let real = |i: usize| (i as f64 * 1.3, 7.0 + turn * (i as f64 - at as f64 - 0.5).abs() * 4.0);
                    curved_scanline(&real, &mut row, PICTURE_TOLERANCE);
                    for (i, drawn) in row.iter().enumerate() {
                        worst_kink = worst_kink.max((drawn.1 - real(i).1).abs());
                    }
                }
                // A stretch of columns outside the projection, long enough that it cannot
                // lie between two of the columns a curve is fitted and checked at, in one
                // coordinate or both: nothing may be drawn across it.
                for hole in [(f64::NAN, 1.0), (1.0, f64::NAN), (f64::INFINITY, 1.0), (f64::NAN, f64::NAN)] {
                    let end = (at + points / 4 + 2).min(points);
                    let real = |i: usize| if (at..end).contains(&i) { hole } else { (i as f64 * 0.5, 9.0) };
                    curved_scanline(&real, &mut row, PICTURE_TOLERANCE);
                    let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
                    for (i, drawn) in row.iter().enumerate() {
                        let want = real(i);
                        assert!(same(drawn.0, want.0) && same(drawn.1, want.1), "{points} points, hole {hole:?} from {at}: column {i} drew {drawn:?}");
                    }
                }
            }
        }
        assert!(
            worst_jump <= allowed && worst_kink <= allowed,
            "drawn {} tolerances out after a jump, {} after a corner",
            worst_jump / PICTURE_TOLERANCE,
            worst_kink / PICTURE_TOLERANCE
        );
    }

    #[test]
    fn item_pictures_with_options_on_assets_go_to_the_search_routes() {
        let by_asset = |rest: &[&str], q: &str| item_picture_by_asset(rest, &query_pairs(q));
        let tile = ["tiles", "WebMercatorQuad", "11", "1059", "746.png"];
        // An array, as `.npy` or a GeoTIFF, whatever its assets: of a tile or a box.
        for array in ["746.npy", "746.tif", "746.tiff"] {
            assert!(by_asset(&["tiles", "WebMercatorQuad", "11", "1059", array], "assets=red"), "{array}");
        }
        assert!(by_asset(&["tiles", "WebMercatorQuad", "11", "1059", "746"], "assets=red&format=npy"));
        assert!(by_asset(&["bbox", "6.3,43.6,6.5,43.8.npy"], "assets=red"));
        assert!(by_asset(&["bbox", "6.3,43.6,6.5,43.8", "64x64.tif"], "assets=red"));
        assert!(!by_asset(&["bbox", "6.3,43.6,6.5,43.8", "64x64.png"], "assets=red"));
        assert!(!by_asset(&["bbox", "6.3,43.6,6.5,43.8", "64x64"], "assets=red&format=npy"));
        assert!(!by_asset(&["preview.npy"], "assets=red"));
        assert!(!by_asset(&tile, "assets=red"));
        // Band math on an asset, and bands chosen on more than one.
        assert!(by_asset(&tile, "assets=red%7Cexpression=b1*2"));
        assert!(by_asset(&tile, "assets=visual%7Cbidx=1&assets=red%7Cbidx=1"));
        assert!(by_asset(&tile, "assets=visual%7Cbidx=3,1&assets=red"));
        assert!(by_asset(&["bbox", "6,43,7,44.png"], "assets=red%7Cexpression=b1*2"));
        assert!(by_asset(&["bbox", "6,43,7,44", "256x256.png"], "assets=red%7Cexpression=b1*2"));
        // What `/stac` draws stays there: plain assets, and bands of a single one.
        assert!(!by_asset(&tile, "assets=visual"));
        assert!(!by_asset(&tile, "assets=red&assets=green&expression=b1/b2"));
        assert!(!by_asset(&tile, "assets=visual%7Cbidx=1"));
        // So does every other route, and a tile in another tile matrix set.
        for rest in [&["preview.png"][..], &["info"], &["statistics"], &["point", "6,43"], &["bbox", "6,43,7,44", "assets"]] {
            assert!(!by_asset(rest, "assets=red%7Cexpression=b1*2"), "{rest:?}");
        }
        assert!(!by_asset(&["tiles", "WGS1984Quad", "11", "1059", "746.png"], "assets=red%7Cexpression=b1*2"));
        // Options that cannot be read are answered for by the route they were sent to.
        assert!(!by_asset(&tile, "assets=red%7Cbidx=x&assets=green"));
        // A box without a format in its path is not one of an item's routes.
        assert!(!by_asset(&["bbox", "6,43,7,44"], "assets=red%7Cexpression=b1*2"));
        assert!(!by_asset(&["bbox", "6,43,7,44", "64x64"], "assets=red%7Cexpression=b1*2"));
        // What only a search reads is not shown to the search routes.
        assert_eq!(
            without_search_options("assets=red%7Cexpression=b1*2&pixel_selection=count&items_limit=many&rescale=0,9&scan_limit=-1&algorithm_params={}"),
            "assets=red%7Cexpression=b1*2&rescale=0,9"
        );
        assert_eq!(without_search_options("algorithm=hillshade&algorithm_params={}"), "algorithm=hillshade&algorithm_params={}");
    }

    #[test]
    fn an_array_asked_for_as_a_geotiff_is_one() {
        let npy = npy_array(&[1.0, f32::NAN, 3.0, 4.0], 2, 2, 1, SampleType::Uint16, true, None);
        let (tiff, kind) = as_geotiff((npy.clone(), "application/x-binary"), true, [8.0, 45.0, 9.0, 46.0], 4326, true);
        assert_eq!(kind, super::super::GEOTIFF_CONTENT_TYPE);
        assert_eq!(&tiff[..4], b"II*\0");
        // Two bands and the mask, in strips of the array's own bytes.
        assert_eq!(&tiff[tiff.len() - 12..], &npy[npy.len() - 12..]);
        let entries = u16::from_le_bytes([tiff[8], tiff[9]]);
        assert_eq!(entries, 15);
        // Anything else is left as it was made.
        let (same, kind) = as_geotiff((npy.clone(), "application/x-binary"), false, [8.0, 45.0, 9.0, 46.0], 4326, true);
        assert_eq!((same, kind), (npy, "application/x-binary"));
        let png = Bytes::from_static(b"\x89PNG....");
        assert_eq!(as_geotiff((png.clone(), "image/png"), true, [0.0, 0.0, 1.0, 1.0], 3857, true).0, png);
    }

    #[test]
    fn assets_take_titilers_options() {
        let parse = |q: &str| parse_assets(&query_pairs(q));
        let spec = |name: &str, indexes: Option<Vec<u16>>, expression: Option<&str>| AssetSpec {
            name: name.into(),
            indexes,
            expression: expression.map(str::to_string),
        };
        assert_eq!(parse("assets=red&assets=nir").unwrap(), [spec("red", None, None), spec("nir", None, None)]);
        // Options keep their commas; a plain entry may still list names with them.
        assert_eq!(
            parse("assets=visual%7Cbidx%3D3,2,1&assets=red,nir").unwrap(),
            [spec("visual", Some(vec![3, 2, 1]), None), spec("red", None, None), spec("nir", None, None)]
        );
        assert_eq!(
            parse("assets=red%7Cexpression%3Db1*2%7Cnope%3D1%7Cbands%3Dx").unwrap(),
            [spec("red", None, Some("b1*2"))]
        );
        for bad in ["assets=visual%7Cbidx", "assets=visual%7Cbidx%3Dx", "assets=visual%7Cbidx%3D1,"] {
            assert_eq!(parse(bad).unwrap_err().status(), StatusCode::UNPROCESSABLE_ENTITY, "{bad}");
        }
        assert_eq!(parse("assets=%7Cbidx%3D1").unwrap_err().status(), StatusCode::BAD_REQUEST);

        // What an asset gives, and how it is described.
        let pixel = |spec: &AssetSpec, as_band: bool| {
            let (plan, described) = asset_bands(spec, 3, as_band)?;
            let mut out = Vec::new();
            plan.pixel(&[109.0, 79.0, 46.0], &mut out);
            Ok::<_, String>((out, described))
        };
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(pixel(&spec("visual", None, None), false).unwrap(), (vec![109.0, 79.0, 46.0], names(&["visual_b1", "visual_b2", "visual_b3"])));
        assert_eq!(pixel(&spec("visual", Some(vec![3, 1]), None), false).unwrap(), (vec![46.0, 109.0], names(&["visual_b3", "visual_b1"])));
        assert_eq!(pixel(&spec("visual", None, Some("b1+b2;b3")), false).unwrap(), (vec![188.0, 46.0], names(&["visual_b1+b2", "visual_b3"])));
        // Band math wins over band indexes, and sees the asset's own bands.
        assert_eq!(pixel(&spec("visual", Some(vec![3]), Some("b1*2")), false).unwrap().0, [218.0]);
        assert_eq!(pixel(&spec("visual", Some(vec![2]), None), true).unwrap(), (vec![79.0], names(&["visual"])));
        assert_eq!(pixel(&spec("visual", None, None), true).unwrap_err(), "Can't use `asset_as_band` for multibands asset");
        assert_eq!(pixel(&spec("visual", Some(vec![9]), None), false).unwrap_err(), "band index 9 out of range (not in (1, 2, 3))");
        assert_eq!(asset_bands(&spec("red", Some(vec![0]), None), 1, false).err().unwrap(), "band index 0 out of range (not in (1,))");
        // Whole planes go the same way.
        let (plan, _) = asset_bands(&spec("v", None, Some("b2/b1")), 2, false).unwrap();
        let planes = plan.planes(&[vec![2.0, 0.0], vec![1.0, 1.0]]);
        assert_eq!(planes[0][0], 0.5);
        assert_eq!(planes[0][1], f64::MAX);
    }

    #[test]
    fn the_renderer_gets_what_it_can_draw() {
        let run = |q: &str| {
            let mut pq = super::super::parse_query(Some(q));
            let pairs = query_pairs(q);
            let query = check_mosaic_query(&pq, &pairs).map_err(|r| r.status())?;
            let names = renderer_assets(&mut pq, &query).map_err(|r| r.status())?;
            Ok::<_, StatusCode>((names, pq.bidx, pq.expr_raw, query.asset_as_band))
        };
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(run("assets=visual").unwrap(), (names(&["visual"]), vec![], None, false));
        // A bare `bidx` is passed over, as in TiTiler 2; an asset's own is used.
        assert_eq!(run("assets=visual&bidx=2").unwrap().1, Vec::<u16>::new());
        assert_eq!(run("assets=visual%7Cbidx%3D3,2,1").unwrap(), (names(&["visual"]), vec![3, 2, 1], None, false));
        assert_eq!(run("assets=red%7Cexpression%3Db1*2").unwrap().2.as_deref(), Some("b1*2"));
        assert_eq!(
            run("assets=red%7Cbidx%3D1&assets=nir&asset_as_band=true&expression=b2-b1").unwrap(),
            (names(&["red", "nir"]), vec![], Some("b2-b1".to_string()), true)
        );
        // Band math is over b1..bn; asset names are refused as TiTiler refuses them.
        assert_eq!(run("assets=red&assets=nir&expression=nir-red").unwrap_err(), StatusCode::BAD_REQUEST);
        // What the renderer cannot draw is refused, not drawn differently. On tiles all of
        // it is made from planes instead, and nothing else is.
        let planes = |q: &str| {
            let pq = super::super::parse_query(Some(q));
            wants_planes(&pq, &check_mosaic_query(&pq, &query_pairs(q)).unwrap())
        };
        for q in [
            "assets=visual",
            "assets=visual%7Cbidx%3D3,2,1",
            "assets=red%7Cexpression%3Db1*2",
            "assets=red&assets=green&assets=blue",
            "assets=red%7Cbidx%3D1&assets=nir&asset_as_band=true&expression=b2-b1",
            "assets=visual&pixel_selection=median",
        ] {
            assert!(run(q).is_ok() && !planes(q), "{q}");
        }
        for q in [
            "assets=red&assets=nir",
            "assets=red&assets=green&assets=blue&assets=nir",
            "assets=visual&pixel_selection=lastbandlow",
            "assets=red&assets=nir&expression=b1;b2",
        ] {
            assert!(planes(q), "{q}");
        }
        for q in [
            "assets=visual%7Cbidx%3D2&assets=nir",
            "assets=visual%7Cbidx%3D1,2",
            "assets=visual%7Cbidx%3D1&expression=b1*2",
            "assets=visual%7Cexpression%3Db1;b2",
            "assets=red%7Cexpression%3Db1*2&assets=nir",
            "assets=red%7Cexpression%3Db1*2&expression=b1%2B1",
        ] {
            assert_eq!(run(q).unwrap_err(), StatusCode::BAD_REQUEST, "{q}");
            assert!(planes(q), "{q}");
        }
        assert_eq!(run("assets=red&asset_as_band=perhaps").unwrap_err(), StatusCode::UNPROCESSABLE_ENTITY);
        // An `assets` that names nothing is a missing parameter.
        assert_eq!(run("assets=,").unwrap_err(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn statistics_options_are_titilers() {
        let pairs = |q: &str| query_pairs(q);
        let (opts, cover) = stats_options(&pairs("assets=red")).unwrap();
        assert_eq!((opts.categorical, opts.percentiles.clone(), opts.bins.clone(), opts.range, cover),
                   (false, vec![2, 98], Bins::Count(10), None, 10));
        let (opts, cover) = stats_options(&pairs(
            "categorical=true&c=1&c=2.5&p=5&p=95&histogram_bins=4&histogram_range=0,2000.5&cover_scale=4",
        ))
        .unwrap();
        assert!(opts.categorical);
        assert_eq!(opts.categories, [1.0, 2.5]);
        assert_eq!(opts.percentiles, [5, 95]);
        assert_eq!(opts.bins, Bins::Count(4));
        assert_eq!(opts.range, Some((0.0, 2000.5)));
        assert_eq!(cover, 4);
        let (opts, _) = stats_options(&pairs("histogram_bins=0,100,255")).unwrap();
        assert_eq!(opts.bins, Bins::Edges(vec![0.0, 100.0, 255.0]));
        for bad in [
            "categorical=maybe", "c=x", "p=101", "p=1.5", "histogram_bins=ten", "histogram_bins=0",
            "histogram_bins=3,2,1", "histogram_range=5", "histogram_range=9,1", "cover_scale=0",
            "cover_scale=1001",
        ] {
            let resp = stats_options(&pairs(bad)).unwrap_err();
            assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY, "{bad}");
        }
    }

    #[test]
    fn tile_limits_cover_the_bounds() {
        let limits = |z, b| tile_limits(z, b);
        assert_eq!(
            limits(8, [5.0, 43.5, 12.0, 48.0]),
            json!({ "tileMatrix": "8", "minTileRow": 88, "maxTileRow": 93, "minTileCol": 131, "maxTileCol": 136 })
        );
        assert_eq!(
            limits(7, [6.0, 44.0, 8.0, 46.0]),
            json!({ "tileMatrix": "7", "minTileRow": 45, "maxTileRow": 46, "minTileCol": 66, "maxTileCol": 66 })
        );
        // The whole world, including latitudes past the Web Mercator limit, stays on the grid.
        assert_eq!(
            limits(0, [-180.0, -90.0, 180.0, 90.0]),
            json!({ "tileMatrix": "0", "minTileRow": 0, "maxTileRow": 0, "minTileCol": 0, "maxTileCol": 0 })
        );
        assert_eq!(
            limits(3, [-180.0, -90.0, 180.0, 90.0]),
            json!({ "tileMatrix": "3", "minTileRow": 0, "maxTileRow": 7, "minTileCol": 0, "maxTileCol": 7 })
        );
    }

    #[test]
    fn query_strings_are_written_like_python() {
        assert_eq!(quote_plus("(b1-b2)/(b1+b2)"), "%28b1-b2%29%2F%28b1%2Bb2%29");
        assert_eq!(quote_plus("a b~c_d.e"), "a+b~c_d.e");
    }

    #[test]
    fn tile_coordinates_are_parsed() {
        assert!(parse_zxy("3", "7", 7).unwrap().is_some());
        assert!(parse_zxy("3", "8", 0).unwrap().is_none());
        assert!(parse_zxy("31", "0", 0).unwrap().is_none());
        assert_eq!(parse_zxy("x", "0", 0).unwrap_err().status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
    #[test]
    fn bounds_are_written_as_python_writes_floats() {
        for (given, written) in [
            (6.0, "6.0"), (43.580390855607845, "43.580390855607845"), (-0.0, "0.0"), (0.00001, "1e-05"),
            (-0.00001234, "-1.234e-05"), (0.0001, "0.0001"), (1e16, "1e+16"), (1234567890123456.0, "1234567890123456.0"),
            (-20037508.342789244, "-20037508.342789244"),
        ] {
            assert_eq!(python_float(given), written);
        }
    }

    #[test]
    fn a_nodata_the_samples_cannot_hold_is_refused() {
        use cogzilla_raster::SampleType as T;
        assert!(nodata_fits(1000.0, T::Uint16) && nodata_fits(1000.5, T::Uint16) && nodata_fits(-1.0, T::Int16));
        for bad in [f32::NAN, f32::INFINITY, -1.0, 70000.0] {
            assert!(!nodata_fits(bad, T::Uint16));
        }
        assert!(!nodata_fits(256.0, T::Uint8) && nodata_fits(f32::NAN, T::Float32));
    }

    #[test]
    fn item_pictures_say_where_they_lie() {
        let ok = || Response::builder().status(StatusCode::OK).body(Full::new(Bytes::new())).unwrap();
        let said = |sub: &[&str], query: &str| {
            let resp = item_placed(ok(), sub, query);
            let header = |name: &str| resp.headers().get(name).map(|v| v.to_str().unwrap().to_string());
            header("content-bbox").zip(header("content-crs"))
        };
        let crs = |epsg: u16| format!("<http://www.opengis.net/def/crs/EPSG/0/{epsg}>");
        let (bbox, system) = said(&["tiles", "WebMercatorQuad", "0", "0", "0.png"], "assets=visual").unwrap();
        assert_eq!(system, crs(3857));
        assert!(bbox.starts_with("-20037508.34278") && bbox.split(',').count() == 4);
        assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "assets=visual").unwrap(), ("6.3,43.6,6.5,43.8".into(), crs(4326)));
        assert_eq!(
            said(&["bbox", "701000,5404000,720000,5420000", "64x64.png"], "coord_crs=epsg:3857").unwrap(),
            ("701000.0,5404000.0,720000.0,5420000.0".into(), crs(3857))
        );
        assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst_crs=EPSG%3A3857").unwrap().1, crs(3857));
        assert_eq!(said(&["bbox", "701000,5404000,720000,5420000.png"], "coord_crs=epsg:3857&dst_crs=epsg:4326").unwrap().1, crs(4326));
        assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst_crs=epsg:32631").unwrap().1, crs(32631));
        assert_eq!(said(&["bbox", "765000,4832000,782000,4855000.png"], "coord_crs=epsg:32631").unwrap().1, crs(32631));
        // Unsupported systems, tiles that are not tiles, and other routes go without.
        for (sub, query) in [
            (&["bbox", "6.3,43.6,6.5,43.8.png"][..], "dst_crs=epsg:4275"),
            (&["tiles", "WebMercatorQuad", "1", "2", "0.png"][..], ""),
            (&["tiles", "WGS1984Quad", "0", "0", "0.png"][..], ""),
            (&["preview.png"][..], ""),
            (&["info"][..], ""),
        ] {
            assert_eq!(said(sub, query), None, "{sub:?} {query}");
        }
        // A percent-encoded key is the key it spells, as the handlers read it.
        assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst%5Fcrs=EPSG:3857").unwrap().1, crs(3857));
        assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst_crs=epsg:3857&dst_crs=epsg:4326").unwrap().1, crs(4326));
        // Degrees asked for as CRS84 are named so; a buffered tile is not placed.
        let ogc = "<http://www.opengis.net/def/crs/OGC/0/CRS84>";
        assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "coord_crs=OGC:CRS84").unwrap().1, ogc);
        assert_eq!(said(&["bbox", "701000,5404000,720000,5420000.png"], "coord_crs=epsg:3857&dst_crs=OGC%3ACRS84").unwrap().1, ogc);
        assert_eq!(said(&["tiles", "WebMercatorQuad", "0", "0", "0.png"], "buffer=10"), None);
        assert!(said(&["tiles", "WebMercatorQuad", "0", "0", "0.png"], "buffer=0").is_some());
        let missing = Response::builder().status(StatusCode::NOT_FOUND).body(Full::new(Bytes::new())).unwrap();
        assert!(item_placed(missing, &["bbox", "6.3,43.6,6.5,43.8.png"], "").headers().get("content-bbox").is_none());
        // `respelled` preserves arbitrary EPSG codes from upstream Content-Crs headers.
        let utm_tile = Response::builder()
            .status(StatusCode::OK)
            .header("content-bbox", "500000,4800000,510000,4810000")
            .header("content-crs", "<http://www.opengis.net/def/crs/EPSG/0/32631>")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let respelled_utm = item_placed(utm_tile, &["tiles", "UTM31N", "0", "0", "0.png"], "");
        assert_eq!(
            respelled_utm.headers().get("content-crs").unwrap().to_str().unwrap(),
            crs(32631)
        );
        assert_eq!(
            respelled_utm.headers().get("content-bbox").unwrap().to_str().unwrap(),
            "500000.0,4800000.0,510000.0,4810000.0"
        );
    }

    #[test]
    fn rgba_to_npy_u8_writes_chw_planes_and_optional_mask() {
        // 2x1 RGBA image: pixel 0 opaque (10,20,30,255), pixel 1 transparent (40,50,60,0).
        let rgba = [10u8, 20, 30, 255, 40, 50, 60, 0];
        let with_mask = rgba_to_npy_u8(&rgba, 2, 1, false, true);
        assert!(with_mask.starts_with(b"\x93NUMPY"));
        let hdr_len = 10 + u16::from_le_bytes([with_mask[8], with_mask[9]]) as usize;
        let hdr = std::str::from_utf8(&with_mask[10..hdr_len]).unwrap();
        assert!(hdr.contains("'|u1'"));
        assert!(hdr.contains("(4, 1, 2)"));
        assert_eq!(&with_mask[hdr_len..], &[10, 40, 20, 50, 30, 60, 255, 0]);

        let no_mask = rgba_to_npy_u8(&rgba, 2, 1, false, false);
        let hdr_len2 = 10 + u16::from_le_bytes([no_mask[8], no_mask[9]]) as usize;
        let hdr2 = std::str::from_utf8(&no_mask[10..hdr_len2]).unwrap();
        assert!(hdr2.contains("(3, 1, 2)"));
        assert_eq!(&no_mask[hdr_len2..], &[10, 40, 20, 50, 30, 60]);

        // Grey 1-band + mask and short-buffer zero-padding safety.
        let grey_short = rgba_to_npy_u8(&[77, 77, 77, 255], 2, 1, true, true);
        let hdr_len3 = 10 + u16::from_le_bytes([grey_short[8], grey_short[9]]) as usize;
        let hdr3 = std::str::from_utf8(&grey_short[10..hdr_len3]).unwrap();
        assert!(hdr3.contains("(2, 1, 2)"));
        assert_eq!(&grey_short[hdr_len3..], &[77, 0, 255, 0]);
    }

    #[test]
    fn a_normalized_index_over_assets_is_band_math() {
        let asked = |query: &str, ext: Option<&str>| {
            let mut pq = super::super::parse_query(Some(query));
            let index = normalized_index_as_band_math(&mut pq, ext);
            assert_eq!(index, pq.algorithm.is_none(), "{query}");
            (pq.algorithm, pq.expr_raw, pq.rescale)
        };
        let math = (None, Some("(b2-b1)/(b2+b1)".to_string()));
        let (algorithm, expression, rescale) = asked("assets=red&assets=nir&algorithm=normalizedIndex", Some("png"));
        assert_eq!((algorithm, expression), math.clone());
        assert_eq!(rescale, vec![(-1.0, 1.0)]);
        // A range asked for is kept; without an extension the picture is still a picture.
        let (algorithm, expression, rescale) = asked("assets=red&assets=nir&algorithm=normalizedIndex&rescale=0,1", None);
        assert_eq!((algorithm, expression), math.clone());
        assert_eq!(rescale, vec![(0.0, 1.0)]);
        // An array is the values as they are, with no range put in.
        for (query, ext) in [
            ("assets=red&assets=nir&algorithm=normalizedIndex", Some("npy")),
            ("assets=red&assets=nir&algorithm=normalizedIndex&format=tif", None),
        ] {
            let (algorithm, expression, rescale) = asked(query, ext);
            assert_eq!((algorithm, expression), math.clone(), "{query}");
            assert!(rescale.is_empty(), "{query}");
        }
        // One asset is the renderer's; a bare array is the algorithm's; band math asked for wins.
        for (query, ext) in [
            ("assets=visual&algorithm=normalizedIndex", Some("png")),
            ("assets=red&assets=nir&algorithm=normalizedIndex", Some("bin")),
            ("assets=red&assets=nir&algorithm=normalizedIndex&expression=b1", Some("png")),
            ("assets=red&assets=nir&algorithm=hillshade", Some("png")),
        ] {
            let (algorithm, _, _) = asked(query, ext);
            assert!(algorithm.is_some(), "{query}");
        }
    }

    #[test]
    fn algorithms_on_a_mosaic_s_bands() {
        let nan = f64::NAN;
        let bands = vec![vec![1.0, 10.0, nan], vec![2.0, nan, nan], vec![4.0, 20.0, nan]];
        let run = |a: BandAlgorithm, whole| a.apply(bands.clone(), whole, 3, (1.0, 1.0)).unwrap();
        let first = |a: BandAlgorithm| run(a, true).0.remove(0);
        assert_eq!(first(BandAlgorithm::Min)[..2], [1.0, 10.0]);
        assert_eq!(first(BandAlgorithm::Sum)[..2], [7.0, 30.0]);
        assert_eq!(first(BandAlgorithm::Mean)[..2], [7.0 / 3.0, 15.0]);
        // numpy's median of an even count is the mean of the two in the middle.
        assert_eq!(first(BandAlgorithm::Median)[..2], [2.0, 15.0]);
        // One degree of freedom less: 1, 2, 4 about 7/3.
        let var = first(BandAlgorithm::Var);
        assert!((var[0] - 7.0 / 3.0).abs() < 1e-12 && (var[1] - 50.0).abs() < 1e-12);
        assert!(first(BandAlgorithm::Std)[2].is_nan());
        // Luma, in the bands' own type: 0.299 + 1.174 + 0.456.
        assert_eq!(first(BandAlgorithm::Grayscale)[0], 1.0);
        assert!((run(BandAlgorithm::Grayscale, false).0[0][0] - 1.929).abs() < 1e-12);
        assert!(BandAlgorithm::Grayscale.apply(vec![vec![1.0]], true, 1, (1.0, 1.0)).is_err());
        assert_eq!(BandAlgorithm::Bitonal.apply(vec![vec![127.0, 128.0, nan]], true, 1, (1.0, 1.0)).unwrap().0[0][..2], [0.0, 255.0]);
        // To bytes as `astype("uint8")` wraps, keeping the bands' descriptions.
        let (cast, described) = BandAlgorithm::Cast.apply(vec![vec![300.0, -1.0, 255.9]], true, 1, (1.0, 1.0)).unwrap();
        assert_eq!((cast[0].as_slice(), described.len()), (&[44.0, 255.0, 255.0][..], 0));
        assert_eq!(BandAlgorithm::Ceil.apply(vec![vec![255.5]], false, 1, (1.0, 1.0)).unwrap().0[0], [0.0]);
        // Terrain-RGB of 100 m: (100 + 10000) / 0.1 = 101000 = 0x018A88.
        let (rgb, names) = BandAlgorithm::TerrainRgb { interval: 0.1, baseval: -10000.0 }.apply(vec![vec![100.0]], true, 1, (1.0, 1.0)).unwrap();
        assert_eq!(rgb, [vec![1.0], vec![138.0], vec![136.0]]);
        assert_eq!(names, ["terrain_r", "terrain_g", "terrain_b"]);
        // Terrarium of 100 m: 32868 = 128 * 256 + 100.
        assert_eq!(BandAlgorithm::Terrarium.apply(vec![vec![100.0]], true, 1, (1.0, 1.0)).unwrap().0, [vec![128.0], vec![100.0], vec![0.0]]);
        // `aspect` is parsed and applied (matching `cogzilla-kernel::DemAlgorithm::Aspect`).
        assert_eq!(BandAlgorithm::parse("aspect", None), Ok(Some(BandAlgorithm::Aspect { buffer: 0, z: 1.0 })));
        assert!(BandAlgorithm::parse("aspect", Some(r#"{"buffer":100}"#)).is_err());
        // Elevation decreasing Eastward (left=20, mid=10, right=0) -> aspect = 90° (East).
        let (east, east_names) = BandAlgorithm::Aspect { buffer: 0, z: 1.0 }
            .apply(vec![vec![20.0, 10.0, 0.0, 20.0, 10.0, 0.0, 20.0, 10.0, 0.0]], false, 3, (1.0, 1.0))
            .unwrap();
        assert_eq!(east_names, ["aspect"]);
        assert!((east[0][4] - 90.0).abs() < 1e-4, "expected 90° East, got {}", east[0][4]);
        // Elevation increasing Southward (row 0 = 0, row 1 = 10, row 2 = 20) -> downhill is North (0°).
        let (north, _) = BandAlgorithm::Aspect { buffer: 0, z: 1.0 }
            .apply(vec![vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0]], false, 3, (1.0, 1.0))
            .unwrap();
        assert!(north[0][4].abs() < 1e-4, "expected 0° North, got {}", north[0][4]);
        // Flat plane -> NaN aspect.
        let (flat, _) = BandAlgorithm::Aspect { buffer: 0, z: 1.0 }
            .apply(vec![vec![5.0; 9]], false, 3, (1.0, 1.0))
            .unwrap();
        assert!(flat[0][4].is_nan());
        assert!(BandAlgorithm::parse("hillshade", Some(r#"{"azimuth":400}"#)).is_err());
        // A fractional increment is taken, as on the shared routes; one past TiTiler's bound is not.
        assert!(matches!(BandAlgorithm::parse("contours", Some(r#"{"increment":2.5}"#)), Ok(Some(BandAlgorithm::Contours { increment, .. })) if increment == 2.5));
        assert!(BandAlgorithm::parse("contours", Some(r#"{"increment":1000}"#)).is_err());
        assert_eq!(BandAlgorithm::parse("terrainrgb", Some(r#"{"interval":1}"#)), Ok(Some(BandAlgorithm::TerrainRgb { interval: 1.0, baseval: -10000.0 })));
        assert_eq!(BandAlgorithm::parse("terrain-rgb", None), Ok(Some(BandAlgorithm::TerrainRgb { interval: 0.1, baseval: -10000.0 })));
        assert_eq!(BandAlgorithm::parse("mapbox", None), Ok(Some(BandAlgorithm::TerrainRgb { interval: 0.1, baseval: -10000.0 })));
        // numpy's gradient: one-sided on the edges, centred inside.
        let (down, across) = gradient(&[0.0, 1.0, 4.0, 2.0, 4.0, 8.0], 3, 2).unwrap();
        assert_eq!(down, [2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);
        assert_eq!(across, [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
        assert_eq!(crop((0..16).map(f64::from).collect(), 4, 4, 1), [5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn every_advertised_algorithm_is_applied_on_both_route_stacks() {
        use crate::algorithms::{ALGORITHMS, AlgorithmId};
        use crate::engine::BandRequest;
        use cogzilla_raster::SampleType;

        // 8x8 sloped plane so buffered DEM algorithms (default buffer=3 -> 2x2 inner crop) succeed.
        let plane = |offset: f64| -> Vec<f64> {
            (0..64).map(|i| offset + (i % 8) as f64 * 10.0 + (i / 8) as f64 * 5.0).collect()
        };
        let three_bands = vec![plane(10.0), plane(50.0), plane(100.0)];
        let one_band = vec![plane(10.0)];

        for def in ALGORITHMS {
            // 1. COG/STAC route stack (`http.rs::build_band_request`): never silently falls back to raw bands.
            let pq = super::super::parse_query(Some(&format!("algorithm={}", def.name)));
            let req = super::super::build_band_request(&pq, 3, SampleType::Float32)
                .unwrap_or_else(|e| panic!("build_band_request failed for '{}': {e}", def.name));
            assert!(
                matches!(
                    req,
                    BandRequest::Dem { .. }
                        | BandRequest::Terrain { .. }
                        | BandRequest::CustomExpr { .. }
                        | BandRequest::CustomExprRgb { .. }
                ),
                "algorithm '{}' fell through to raw BandRequest: {req:?}",
                def.name
            );

            // 2. pgSTAC route stack (`BandAlgorithm::parse` + `apply`, or `normalizedIndex` band-math path).
            if def.id == AlgorithmId::NormalizedIndex {
                assert_eq!(BandAlgorithm::parse(def.name, None), Ok(None));
                let mut pq_ni = super::super::parse_query(Some("assets=red&assets=nir&algorithm=normalizedIndex"));
                assert!(normalized_index_as_band_math(&mut pq_ni, Some("png")));
                assert_eq!(pq_ni.expr_raw.as_deref(), Some("(b2-b1)/(b2+b1)"));
                continue;
            }
            let parsed = BandAlgorithm::parse(def.name, None)
                .unwrap_or_else(|e| panic!("BandAlgorithm::parse failed for '{}': {e}", def.name))
                .unwrap_or_else(|| panic!("BandAlgorithm::parse silently returned Ok(None) for '{}'", def.name));
            let input = if def.input_bands == Some(1) { one_band.clone() } else { three_bands.clone() };
            let (out_bands, _) = parsed
                .apply(input, false, 8, (1.0, 1.0))
                .unwrap_or_else(|e| panic!("BandAlgorithm::apply failed for '{}': {e}", def.name));
            assert!(!out_bands.is_empty(), "algorithm '{}' produced 0 bands", def.name);
            if let Some(expected_nb) = def.out_bands {
                assert_eq!(
                    out_bands.len(),
                    usize::from(expected_nb),
                    "algorithm '{}' output band count mismatch",
                    def.name
                );
            }
        }
    }

    #[test]
    fn a_point_s_numbers_and_a_tileset_s_format() {
        assert_eq!(point_coords("6.73%20,%2043.402").ok(), Some((6.73, 43.402)));
        assert_eq!(point_coords(" 1 , 2 ").ok(), Some((1.0, 2.0)));
        assert!(point_coords("1,abc").is_err());
        let pairs = |f: &str| vec![("f".to_string(), f.to_string())];
        assert!(tileset_format_error(&pairs("html")).is_none());
        assert!(tileset_format_error(&pairs("json")).is_none());
        assert!(tileset_format_error(&[]).is_none());
        for f in ["png", "HTML", ""] {
            assert_eq!(tileset_format_error(&pairs(f)).map(|r| r.status()), Some(StatusCode::UNPROCESSABLE_ENTITY), "{f:?}");
        }
    }

    #[test]
    fn a_float32_band_s_bins_are_numpy_s() {
        let edges = |opts: &StatsOptions, band: &[f64]| match float32_bins(opts, band).bins {
            zonal::Bins::Edges(e) => e,
            zonal::Bins::Count(_) => vec![],
        };
        let band = [f64::from(-0.759_312_33_f32), 0.1, f64::NAN, f64::from(0.759_312_33_f32)];
        // From the band's own ends: float32 arithmetic, the last edge the greatest value.
        let opts = StatsOptions { bins: zonal::Bins::Count(10), ..StatsOptions::default() };
        let from_values = edges(&opts, &band);
        assert_eq!(from_values.len(), 11);
        let (lo, hi) = (-0.759_312_33_f32, 0.759_312_33_f32);
        let step = (hi - lo) / 10.0;
        assert_eq!(from_values[3], f64::from(3.0 * step + lo));
        assert_eq!(from_values[10], f64::from(hi));
        // From a range given: double precision, each edge cast to float32 (-5/7 is one).
        let opts = StatsOptions { bins: zonal::Bins::Count(7), range: Some((-1.0, 1.0)), ..StatsOptions::default() };
        assert_eq!(edges(&opts, &band)[1], f64::from((-1.0 + 2.0 / 7.0) as f32));
        // Edges given, or one value only: left as they are.
        let given = StatsOptions { bins: zonal::Bins::Edges(vec![0.0, 1.0]), ..StatsOptions::default() };
        assert!(matches!(float32_bins(&given, &band).bins, zonal::Bins::Edges(ref e) if e == &vec![0.0, 1.0]));
        let flat = StatsOptions { bins: zonal::Bins::Count(10), ..StatsOptions::default() };
        assert!(matches!(float32_bins(&flat, &[2.0, 2.0]).bins, zonal::Bins::Count(10)));
    }

    #[test]
    fn band_math_on_an_item_s_statistics() {
        // b1..bn over the stacked bands; an asset's name is not one, as in titiler-pgstac.
        assert!(check_item_band_math_names("(b1-b2)/(b1+b2)").is_ok());
        assert!(check_item_band_math_names("b1-b3;b4*2").is_ok());
        let named = check_item_band_math_names("(nir-red)/(nir+red)").unwrap_err();
        assert!(named.to_string().contains("Invalid band/asset name 'nir'"));
    }

    #[test]
    fn an_item_preview_and_feature_say_where_they_were_drawn() {
        let ok = || Response::builder().status(StatusCode::OK).body(Full::new(Bytes::new())).unwrap();
        let header = |resp: &Resp, name: &str| resp.headers().get(name).map(|v| v.to_str().unwrap().to_string());
        let drawn = Some(([699960.0, 4790220.0, 809760.0, 4900020.0], Some(32631), false));
        for sub in [&["preview.png"][..], &["preview"], &["preview", "100x80.png"], &["feature", "64x64.png"], &["feature.png"]] {
            let resp = item_drawn_placed(ok(), sub, drawn);
            assert_eq!(header(&resp, "content-bbox").as_deref(), Some("699960.0,4790220.0,809760.0,4900020.0"), "{sub:?}");
            assert_eq!(header(&resp, "content-crs").as_deref(), Some("<http://www.opengis.net/def/crs/EPSG/0/32631>"));
        }
        // Degrees asked for as CRS84 are named so.
        let resp = item_drawn_placed(ok(), &["feature.png"], Some(([6.5, 43.3, 6.6, 43.4], Some(4326), true)));
        assert_eq!(header(&resp, "content-crs").as_deref(), Some("<http://www.opengis.net/def/crs/OGC/0/CRS84>"));
        // Not another route's, nor without a system, nor on an error.
        assert!(item_drawn_placed(ok(), &["info"], drawn).headers().get("content-bbox").is_none());
        assert!(item_drawn_placed(ok(), &["preview.png"], Some(([0.0, 0.0, 1.0, 1.0], None, false))).headers().get("content-crs").is_none());
        let missing = Response::builder().status(StatusCode::NOT_FOUND).body(Full::new(Bytes::new())).unwrap();
        assert!(item_drawn_placed(missing, &["preview.png"], drawn).headers().get("content-bbox").is_none());
    }

    #[test]
    fn arrays_are_written_in_the_datas_type_with_a_mask() {
        let values = [1.0f32, f32::NAN, 300.0, 4.0];
        let made = npy_array(&values, 2, 2, 1, SampleType::Uint16, true, None);
        assert_eq!(&made[..8], b"\x93NUMPY\x01\x00");
        let header = usize::from(u16::from_le_bytes([made[8], made[9]]));
        assert_eq!((10 + header) % 64, 0);
        let dict = std::str::from_utf8(&made[10..10 + header]).unwrap();
        assert!(dict.starts_with("{'descr': '<u2', 'fortran_order': False, 'shape': (3, 1, 2), }") && dict.ends_with('\n'));
        let words: Vec<u16> = made[10 + header..].chunks_exact(2).map(|w| u16::from_le_bytes([w[0], w[1]])).collect();
        // Two bands, then the mask: a pixel is there when any band of it is.
        assert_eq!(words, [1, 0, 300, 4, u16::MAX, u16::MAX]);
        // Without the mask, and a missing value as the number the request named.
        let bare = npy_array(&values, 2, 2, 1, SampleType::Uint8, false, Some(9.0));
        assert_eq!(&bare[bare.len() - 4..], [1, 9, 255, 4]);
        // Where there is nothing, the mask holds the type's smallest number, as rio-tiler's.
        let doubles = npy_array(&[f32::NAN, 0.5], 1, 2, 1, SampleType::Float64, true, None);
        assert_eq!(&doubles[doubles.len() - 16..doubles.len() - 8], f64::MIN.to_le_bytes());
        assert_eq!(&doubles[doubles.len() - 8..], f64::MAX.to_le_bytes());
    }

    #[test]
    fn a_colour_formula_comes_before_the_colour_map() {
        let one = Selected { bands: 1, values: vec![64.0, f32::NAN] };
        let formula = cogzilla_kernel::ColorFormula::parse("gamma r 2").unwrap();
        let mut level = [64u8, 64, 64, 255];
        formula.apply_buffer_in_place(&mut level, 4);
        assert_ne!(level[0], 64);
        let viridis = ColormapName::Viridis.lut();
        let got = paint_bands(&one, 2, &[(0.0, 255.0)], (0.0, 255.0), ColormapName::Viridis, None, Some(&formula)).unwrap();
        assert_eq!(got[..3], viridis.rgb[usize::from(level[0])]);
        assert_eq!(got[3..], [255, 0, 0, 0, 0]);
    }

    #[test]
    fn the_renderers_grey_takes_the_formula_and_then_the_map() {
        let formula = cogzilla_kernel::ColorFormula::parse("gamma r 2").unwrap();
        let mut bands = BandRequest::Single { band_1based: 1, rescale: (0.0, 255.0), colormap: ColormapName::Viridis };
        let after = MapAfterFormula::take(&formula, &mut bands).unwrap();
        assert!(matches!(bands, BandRequest::Single { colormap: ColormapName::Grayscale, .. }));
        let mut rgba = [64u8, 64, 64, 255, 9, 9, 9, 0];
        after.colour(&mut rgba);
        let level = formula_levels(&formula)[64];
        assert_ne!(level, 64);
        assert_eq!(rgba[..3], ColormapName::Viridis.lut().rgb[usize::from(level)]);
        // A pixel that is not there stays as it was.
        assert_eq!(rgba[3..], [255, 9, 9, 9, 0]);
        // Grey already, or three bands: the formula is for the finished picture.
        assert!(MapAfterFormula::take(&formula, &mut bands).is_none());
        let mut rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
        assert!(MapAfterFormula::take(&formula, &mut rgb).is_none());
        // DEM algorithms (hillshade / slope) also apply color_formula before colormap and emit grey when unmapped.
        let mut dem = BandRequest::Dem {
            band_1based: 1,
            algorithm: cogzilla_kernel::DemAlgorithm::Hillshade { azimuth_deg: 315.0, altitude_deg: 45.0, z_factor: 1.0 },
            rescale: (0.0, 255.0),
            colormap: ColormapName::Viridis,
        };
        assert!(!one_grey_band(&dem));
        let dem_after = MapAfterFormula::take(&formula, &mut dem).unwrap();
        assert!(one_grey_band(&dem));
        let mut dem_px = [64u8, 64, 64, 255];
        dem_after.colour(&mut dem_px);
        assert_eq!(dem_px[..3], ColormapName::Viridis.lut().rgb[usize::from(level)]);
    }

    #[test]
    fn encoded_query_strips_doc_negotiation_params() {
        let pairs = vec![
            ("assets".to_string(), "data".to_string()),
            ("f".to_string(), "html".to_string()),
            ("format".to_string(), "json".to_string()),
            ("rescale".to_string(), "0,255".to_string()),
        ];
        assert_eq!(encoded_query(&pairs), "assets=data&rescale=0%2C255");

        // Image tile format=png is preserved.
        let img = vec![
            ("assets".to_string(), "data".to_string()),
            ("format".to_string(), "png".to_string()),
        ];
        assert_eq!(encoded_query(&img), "assets=data&format=png");
    }

}
