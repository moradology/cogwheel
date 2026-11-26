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
use cogzilla_tiff::SampleType;
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CONTENT_TYPE, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{Value, json};

use super::{
    ParsedQuery, apply_unscale_cogs, build_band_request, build_tile_request, parse_bbox4,
    parse_ext_format, parse_wxh_ext, parse_y_scale_ext, percent_decode, read_limited_body,
    to_http_response_cached,
};
use super::pixel_selection::{Fold, Selected, Selection};
use super::point_expr::Program;
use super::zonal::{self, Bins, StatsOptions};
use crate::engine::{BandRequest, EngineError, MOSAIC_SCENES_AHEAD, TileEngine};
use crate::stac::{StacItem, stac_item_from_json};
use crate::titiler::{
    PolygonMask, build_map_html, encode_rgba_output_colour, query_point, render_wgs84_bbox,
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
    let mut resp = Response::new(Full::new(Bytes::from(body.to_string())));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    resp.headers_mut()
        .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    resp
}

fn detail(status: StatusCode, msg: impl Into<String>) -> Resp {
    json_resp(status, &json!({ "detail": msg.into() }))
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

fn no_content() -> Resp {
    let mut resp = Response::new(Full::new(Bytes::new()));
    *resp.status_mut() = StatusCode::NO_CONTENT;
    resp.headers_mut()
        .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    resp
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

pub(super) fn engine_error(err: &EngineError) -> Resp {
    match err {
        EngineError::Shared(inner) => engine_error(inner),
        EngineError::NotFound(msg) => detail(StatusCode::NOT_FOUND, msg.clone()),
        EngineError::BadRequest(msg) => detail(StatusCode::BAD_REQUEST, msg.clone()),
        // As Starlette answers it. An item's routes are each for one method.
        EngineError::MethodNotAllowed(allow) => {
            let mut resp = detail(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed");
            let allow = if allow.contains("GET") { "GET" } else { "POST" };
            resp.headers_mut().insert(hyper::header::ALLOW, HeaderValue::from_static(allow));
            resp
        }
        EngineError::Proj(e) => detail(StatusCode::BAD_REQUEST, e.to_string()),
        EngineError::Io(e @ cogzilla_io::IoError::InvalidUri(_)) => {
            detail(StatusCode::BAD_REQUEST, e.to_string())
        }
        EngineError::Io(e @ cogzilla_io::IoError::Forbidden(_)) => {
            detail(StatusCode::FORBIDDEN, e.to_string())
        }
        e => {
            tracing::warn!(error = %e, "pgSTAC tile failed with 500");
            detail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        }
    }
}

/// Decoded `key=value` pairs in request order.
fn query_pairs(raw_query: &str) -> Vec<(String, String)> {
    raw_query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (percent_decode(k), percent_decode(v))
        })
        .collect()
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

fn parse_bool(v: &str) -> Option<bool> {
    match v.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" | "t" | "y" => Some(true),
        "false" | "0" | "no" | "off" | "f" | "n" => Some(false),
        _ => None,
    }
}

/// Whether `s` is a date, or a date and a time, as RFC 3339 writes them.
fn looks_like_datetime(s: &str) -> bool {
    let b = s.as_bytes();
    let digits = |from: usize, n: usize| b.get(from..from + n).is_some_and(|d| d.iter().all(u8::is_ascii_digit));
    if !(digits(0, 4) && b.get(4) == Some(&b'-') && digits(5, 2) && b.get(7) == Some(&b'-') && digits(8, 2)) {
        return false;
    }
    if b.len() == 10 {
        return true;
    }
    if !(matches!(b[10], b'T' | b't' | b' ') && digits(11, 2) && b.get(13) == Some(&b':') && digits(14, 2)) {
        return false;
    }
    let mut at = 16;
    if b.get(at) == Some(&b':') && digits(at + 1, 2) {
        at += 3;
        if b.get(at) == Some(&b'.') {
            at += 1;
            let start = at;
            while b.get(at).is_some_and(u8::is_ascii_digit) {
                at += 1;
            }
            if at == start {
                return false;
            }
        }
    }
    match &b[at..] {
        [] | [b'Z' | b'z'] => true,
        [b'+' | b'-', rest @ ..] => matches!(rest, [h1, h2, b':', m1, m2] | [h1, h2, m1, m2] if [h1, h2, m1, m2].iter().all(|c| c.is_ascii_digit())),
        _ => false,
    }
}

/// What FastAPI refuses of a picture's parameters before anything is drawn. The routes
/// that only describe a mosaic (`tilejson.json`, `map.html`, the WMTS document) take the
/// same parameters as its tiles, and titiler-pgstac checks them there too.
fn render_query_error(pairs: &[(String, String)]) -> Option<Resp> {
    let bad = |name: &str, msg: &str| Some(invalid_param("query", name, msg));
    if let Some(map) = query_value(pairs, "colormap")
        && !map.is_empty()
        && (!map.trim_start().starts_with(['{', '[']) || serde_json::from_str::<Value>(map).is_err())
    {
        return Some(detail(StatusCode::BAD_REQUEST, "Could not parse the colormap value."));
    }
    if let Some(name) = query_value(pairs, "colormap_name")
        && (ColormapName::from_str_opt(name).is_none() || name.chars().any(char::is_uppercase))
    {
        return bad("colormap_name", "Input should be the name of a colour map (see /colorMaps)");
    }
    for (key, value) in pairs {
        let number = || value.parse::<f64>().is_ok();
        let whole = || value.parse::<i64>().is_ok();
        let wrong = match key.as_str() {
            "rescale" => {
                let parts: Vec<&str> = value.split(',').collect();
                (parts.len() != 2 || !parts.iter().all(|p| p.trim().parse::<f64>().is_ok())).then_some("Input should be a range, as min,max")
            }
            "resampling" | "reproject" => (!matches!(
                value.as_str(),
                "nearest" | "bilinear" | "cubic" | "cubic_spline" | "lanczos" | "average" | "mode" | "gauss" | "rms"
                    | "max" | "min" | "med" | "q1" | "q3" | "sum"
            ))
            .then_some("Input should be a resampling method"),
            "pixel_selection" => Selection::parse(value).is_none().then_some("Input should be a pixel selection method"),
            "color_formula" => cogzilla_kernel::ColorFormula::parse(value).is_err().then_some("Input should be a colour formula"),
            "nodata" => (!number() && !matches!(value.to_ascii_lowercase().as_str(), "nan" | "inf" | "-inf")).then_some("Input should be a valid number"),
            "unscale" | "return_mask" | "asset_as_band" | "exitwhenfull" | "skipcovered" => {
                parse_bool(value).is_none().then_some("Input should be a valid boolean")
            }
            "tilesize" | "items_limit" | "time_limit" | "scan_limit" | "padding" => (!whole()).then_some("Input should be a valid integer"),
            "buffer" => (!number()).then_some("Input should be a valid number"),
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
    let int = |name: &str, slot: &mut i32| -> Result<(), Resp> {
        if let Some(v) = query_value(pairs, name) {
            let given = v
                .parse::<i32>()
                .ok()
                .filter(|n| *n >= 0)
                .ok_or_else(|| invalid_param("query", name, "Input should be a valid integer"))?;
            // Zero is no limit given, as in titiler-pgstac: the default stands.
            if given > 0 {
                *slot = given;
            }
        }
        Ok(())
    };
    int("scan_limit", &mut limits.scan_limit)?;
    int("items_limit", &mut limits.items_limit)?;
    int("time_limit", &mut limits.time_limit_secs)?;
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
    json!({
        "type": "Polygon",
        "coordinates": [[[west, south], [west, north], [east, north], [east, south], [west, south]]]
    })
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

/// The request's query string, re-encoded the way TiTiler writes it into links.
fn encoded_query(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
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
        if !v.split('/').all(|part| matches!(part, "" | "..") || looks_like_datetime(part)) || v.split('/').count() > 2 {
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
            Some(v) => v
                .parse::<i64>()
                .map(Some)
                .map_err(|_| invalid_param("query", name, "Input should be a valid integer")),
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
    let (minzoom, maxzoom, bounds) = match mosaic_extent(metadata, pairs) {
        Ok(e) => e,
        Err(r) => return r,
    };
    let host = ctx.host;
    let qs = encoded_query(pairs);
    let with_qs = |url: String| if qs.is_empty() { url } else { format!("{url}?{qs}") };
    let limits: Vec<Value> = (minzoom.clamp(0, 30)..=maxzoom.clamp(0, 30))
        .map(|z| tile_limits(z, bounds))
        .collect();
    json_resp(
        StatusCode::OK,
        &json!({
            "title": format!("tileset tiled using {tms} TileMatrixSet"),
            "dataType": "map",
            "crs": "http://www.opengis.net/def/crs/EPSG/0/3857",
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

/// `GET {base}/tiles`: the tilesets on offer. TiTiler lists thirteen tile matrix sets;
/// these routes serve Web Mercator only, so that is the one listed.
fn tilesets(ctx: &Ctx<'_>, base: &str, metadata: &Value, pairs: &[(String, String)]) -> Resp {
    let (_, _, bounds) = match mosaic_extent(metadata, pairs) {
        Ok(e) => e,
        Err(r) => return r,
    };
    let host = ctx.host;
    let tms = "WebMercatorQuad";
    let qs = encoded_query(pairs);
    let with_qs = |url: String| if qs.is_empty() { url } else { format!("{url}?{qs}") };
    json_resp(
        StatusCode::OK,
        &json!({ "tilesets": [{
            "title": format!("tileset tiled using {tms} TileMatrixSet"),
            "dataType": "map",
            "crs": "http://www.opengis.net/def/crs/EPSG/0/3857",
            "links": [
                { "href": with_qs(format!("{host}{base}/tiles/{tms}")), "rel": "self", "type": "application/json", "title": format!("Tileset tiled using {tms} TileMatrixSet") },
                { "href": with_qs(format!("{host}{base}/tiles/{tms}/%7Bz%7D/%7Bx%7D/%7By%7D")), "rel": "tile", "title": "Templated link for retrieving Raster tiles" },
                { "href": format!("{host}/tileMatrixSets/{tms}"), "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme", "type": "application/json", "title": format!("Definition of '{tms}' tileMatrixSet") }
            ],
            "boundingBox": bounding_box(bounds),
            "accessConstraints": "unclassified"
        }]}),
    )
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// A float the way Python prints it (`-180.0`, not `-180`).
fn py_float(v: f64) -> String {
    json!(v).to_string()
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
    let tile_format = query_value(pairs, "tile_format").unwrap_or("png");
    let media_type = match tile_format {
        "png" => "image/png",
        "jpeg" => "image/jpeg",
        "jpg" => "image/jpg",
        "webp" => "image/webp",
        "npy" => "application/x-binary",
        "tif" | "tiff" | "jp2" | "pngraw" => {
            return detail(
                StatusCode::BAD_REQUEST,
                format!("tile_format={tile_format} is not supported on pgSTAC routes (use png, jpg, jpeg, webp or npy)"),
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
            let qs: Vec<&str> = qs.split('&').filter(|kv| !kv.starts_with("tilesize=")).collect();
            (name, qs.join("&"), (source.minzoom, source.maxzoom))
        })
        .collect();
    const NOT_TILE_PARAMS: [&str; 10] = [
        "tile_format", "use_epsg", "tilesize", "zooms", "service", "request", "acceptversions",
        "sections", "updatesequence", "acceptformats",
    ];
    if query_value(pairs, "assets").is_some_and(|a| !a.is_empty()) {
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
    let this_url = xml_escape(&if ctx.raw_query.is_empty() {
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
        let id = xml_escape(&format!("{}_{TMS}_{name}", source.title));
        let template = xml_escape(&if qs.is_empty() {
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
            py_float(w),
            py_float(s),
            py_float(e),
            py_float(n)
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
            py_float(559_082_264.028_717_8 / side as f64)
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
fn check_coord_crs(pairs: &[(String, String)]) -> Result<(), Resp> {
    match query_value(pairs, "coord_crs").map(str::to_ascii_lowercase).as_deref() {
        None | Some("epsg:4326" | "ogc:crs84") => Ok(()),
        Some(other) => Err(detail(
            StatusCode::BAD_REQUEST,
            format!("coord_crs '{other}' is not supported on pgSTAC mosaic routes (use epsg:4326)"),
        )),
    }
}

/// A position given in the request's `coord_crs`, in degrees: Web Mercator metres are
/// turned into them, and any other system is refused.
fn in_degrees(pq: &ParsedQuery, pairs: &[(String, String)], x: f64, y: f64) -> Result<(f64, f64), Resp> {
    if pq.coord_crs_3857 {
        return Ok(mercator_to_lonlat_deg(x, y));
    }
    check_coord_crs(pairs).map(|()| (x, y))
}

fn check_dst_crs(pairs: &[(String, String)]) -> Result<(), Resp> {
    match query_value(pairs, "dst_crs") {
        Some(dst) if !matches!(dst.to_ascii_lowercase().as_str(), "epsg:4326" | "ogc:crs84") => Err(detail(
            StatusCode::BAD_REQUEST,
            format!("dst_crs '{dst}' is not supported on pgSTAC mosaic routes (use epsg:4326)"),
        )),
        _ => Ok(()),
    }
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

/// The tile matrix sets TiTiler registers by default.
const TITILER_TMS: &[&str] = &[
    "CDB1GlobalGrid",
    "CanadianNAD83_LCC",
    "EuropeanETRS89_LAEAQuad",
    "GNOSISGlobalGrid",
    "LINZAntarticaMapTilegrid",
    "NZTM2000Quad",
    "UPSAntarcticWGS84Quad",
    "UPSArcticWGS84Quad",
    "UTM31WGS84Quad",
    "WGS1984Quad",
    "WebMercatorQuad",
    "WorldCRS84Quad",
    "WorldMercatorWGS84Quad",
];

/// Only Web Mercator tiles are served on the mosaic routes for now. A name TiTiler does
/// not know is a validation error there; one it knows but this server cannot serve is
/// refused with the reason.
fn check_tms(tms: &str) -> Result<(), Resp> {
    if tms == "WebMercatorQuad" {
        Ok(())
    } else if TITILER_TMS.contains(&tms) {
        Err(detail(
            StatusCode::BAD_REQUEST,
            format!("tile matrix set '{tms}' is not supported on pgSTAC mosaic routes (use WebMercatorQuad)"),
        ))
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
    let (minzoom, maxzoom, bounds) = match mosaic_extent(&rec.metadata, pairs) {
        Ok(e) => e,
        Err(r) => return r,
    };

    // TiTiler drops the parameters that shape the TileJSON itself and pins the tile size.
    let dropped = ["tilematrixsetid", "tile_format", "tile_scale", "minzoom", "maxzoom"];
    let mut qs: Vec<String> = pairs
        .iter()
        .filter(|(k, _)| !dropped.contains(&k.to_ascii_lowercase().as_str()))
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
fn check_working_bytes(bytes: usize) -> Result<(), EngineError> {
    if bytes > MAX_WORKING_BYTES {
        return Err(EngineError::BadRequest(
            "the image is too large for this pixel_selection; pass max_size, or width and height".into(),
        ));
    }
    Ok(())
}

/// Whether a request's values are the raster's own whole numbers, which `mean` and `median`
/// are cast back to; only then does the cast lose anything.
fn whole_numbers(unscale: bool, bands: &BandRequest, sample_type: SampleType) -> bool {
    !unscale
        && matches!(bands, BandRequest::Single { .. } | BandRequest::Rgb { .. })
        && !matches!(sample_type, SampleType::Float32 | SampleType::Float64)
}

/// A value as a byte, by rio-tiler's `linear_rescale` step for step and in its double
/// precision: clip to the range, take the share of it, scale to 255, drop the fraction.
/// Done in another order or in single precision, a value on a boundary lands a level off.
fn rescale_byte(v: f32, min: f32, max: f32) -> u8 {
    let (v, min, max) = (f64::from(v), f64::from(min), f64::from(max));
    let span = if (max - min).abs() < 1e-12 { 1e-12 } else { max - min };
    let share = (v.max(min).min(max) - min) / span;
    (share * 255.0).clamp(0.0, 255.0) as u8
}

/// How the values a pixel selection leaves become colours: the request's own rescaling
/// and colour map, applied as the renderer applies them to a single scene.
enum Paint {
    Gray { min: f32, max: f32, colormap: ColormapName },
    Rgb { min: [f32; 3], max: [f32; 3] },
}

impl Paint {
    /// `None` for a request whose pixels are not values put through a rescale (a terrain encoding).
    fn of(bands: &BandRequest) -> Option<Self> {
        Some(match bands {
            BandRequest::Single { rescale, colormap, .. }
            | BandRequest::Spectral1 { rescale, colormap, .. }
            | BandRequest::Spectral2 { rescale, colormap, .. }
            | BandRequest::Spectral3 { rescale, colormap, .. }
            | BandRequest::CustomExpr { rescale, colormap, .. }
            | BandRequest::Dem { rescale, colormap, .. } => Self::Gray { min: rescale.0, max: rescale.1, colormap: *colormap },
            BandRequest::Rgb { rescale_min, rescale_max, .. }
            | BandRequest::SarRgb { rescale_min, rescale_max, .. } => Self::Rgb { min: *rescale_min, max: *rescale_max },
            BandRequest::Terrain { .. } | BandRequest::CustomExprRgb { .. } | BandRequest::MultiBand { .. } => return None,
        })
    }

    /// RGBA for `pixels` pixels. A pixel is transparent when no band of it has a value.
    fn rgba(&self, selected: &Selected, pixels: usize) -> Vec<u8> {
        let byte = rescale_byte;
        let mut out = vec![0u8; pixels * 4];
        match (self, selected.bands) {
            (Self::Rgb { min, max }, 3) => {
                for (i, px) in out.chunks_exact_mut(4).enumerate() {
                    let v = [selected.values[i], selected.values[pixels + i], selected.values[2 * pixels + i]];
                    if v.iter().all(|v| v.is_nan()) {
                        continue;
                    }
                    for k in 0..3 {
                        px[k] = if v[k].is_nan() { 0 } else { byte(v[k], min[k], max[k]) };
                    }
                    px[3] = 255;
                }
            }
            _ => {
                // One band: the request's colour map, or grey for the count of an RGB request.
                let (min, max, colormap) = match self {
                    Self::Gray { min, max, colormap } => (*min, *max, *colormap),
                    Self::Rgb { min, max } => (min[0], max[0], ColormapName::Grayscale),
                };
                // Without a colour map the byte is the grey, as in TiTiler. The kernel's grey
                // table is not that: some of its entries are a level below their index.
                let lut = (colormap != ColormapName::Grayscale).then(|| colormap.lut());
                for (px, v) in out.chunks_exact_mut(4).zip(&selected.values) {
                    if !v.is_nan() {
                        let level = byte(*v, min, max);
                        match lut {
                            Some(lut) => px[..3].copy_from_slice(&lut.rgb[usize::from(level)]),
                            None => px[..3].fill(level),
                        }
                        px[3] = 255;
                    }
                }
            }
        }
        out
    }
}

/// TiTiler's two resampling parameters. Both default to nearest neighbour.
fn parse_resample(pairs: &[(String, String)], name: &str) -> Result<ResampleMode, Resp> {
    match query_value(pairs, name) {
        None | Some("nearest") => Ok(ResampleMode::Nearest),
        Some("bilinear") => Ok(ResampleMode::Bilinear),
        Some("cubic" | "cubic_spline" | "lanczos") => Ok(ResampleMode::Cubic),
        Some(other) => Err(detail(
            StatusCode::BAD_REQUEST,
            format!("{name} method '{other}' is not supported (use nearest, bilinear or cubic)"),
        )),
    }
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
    if let Some(name) = query_value(pairs, "colormap_name")
        && (ColormapName::from_str_opt(name).is_none() || name.chars().any(char::is_uppercase))
    {
        return Err(json_resp(
            StatusCode::UNPROCESSABLE_ENTITY,
            &json!({ "detail": [{
                "type": "literal_error",
                "loc": ["query", "colormap_name"],
                "msg": "Input should be the name of a colour map (see /colorMaps)",
                "input": name,
            }] }),
        ));
    }
    if let Some(e) = pq.errors.iter().find(|e| !own_error(e)) {
        return Err(detail(StatusCode::BAD_REQUEST, e.clone()));
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
    if query_value(pairs, "algorithm_params").is_some() {
        return Err(detail(StatusCode::BAD_REQUEST, "parameter 'algorithm_params' is not supported yet"));
    }
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
/// routes draw those (`item_picture_by_asset`). On what is left, a preview or an image
/// over a posted shape, it is refused, not left out of the picture.
///
/// Returns each asset's bands and the request's own `bidx`, or `None` when no asset has
/// options. Only the info, statistics and point handlers read bands asset by asset; on
/// the others several assets with bands of their own are refused too.
pub(super) fn item_asset_options(rest: &[&str], raw_query: &str) -> Result<Option<(AssetBidx, Vec<u16>)>, Resp> {
    let pairs = query_pairs(raw_query);
    let specs = parse_assets(&pairs)?;
    if let Some(spec) = specs.iter().find(|s| s.expression.is_some()) {
        return Err(detail(
            StatusCode::BAD_REQUEST,
            format!("assets={}|expression= is not supported on this item route yet; use `expression`", spec.name),
        ));
    }
    if specs.iter().all(|s| s.indexes.is_none()) {
        return Ok(None);
    }
    let by_asset = matches!(rest, ["info" | "info.geojson" | "statistics" | "asset_statistics"] | ["point", _]);
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
        Some("reproject" | "asset_as_band" | "histogram_bins") => true,
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
/// engine knows names and functions it does not (`ndvi`, `^`, `max(a, b)`).
fn planes_read_all(pq: &ParsedQuery, query: &MosaicQuery) -> bool {
    let reads = |e: &String| Program::parse(e, usize::from(u16::MAX)).is_ok();
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
        // Only the reader of nearest pixels takes a `nodata` of the request's own.
        let nearest_only = || EngineError::BadRequest("`nodata` is read with nearest-neighbour resampling only".into());
        let (bbox, mercator, width, height, size, mask, unscale) = match self {
            Window::Tile(request) => {
                let cog = engine.store.open_uri(href).await?;
                let crs = engine.get_or_build_crs(&cog)?;
                let resample = if matches!(crs, CrsModel::Epsg3857) { resampling } else { reproject };
                let Some(res) = mercator_resolution(&crs, &cog).filter(|_| resample == ResampleMode::Nearest) else {
                    if nodata.is_some() {
                        return Err(nearest_only());
                    }
                    if !matches!(request.tile_size, 256 | 512) {
                        return Err(EngineError::BadRequest(format!(
                            "tilesize={} is drawn with nearest-neighbour resampling only; use 256 or 512 otherwise",
                            request.tile_size
                        )));
                    }
                    return engine.raster_planes(href, bands_1based, request, resampling, reproject).await;
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
        let cog = engine.store.open_uri(href).await?;
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
        if mercator && resample != ResampleMode::Nearest {
            return Err(EngineError::BadRequest("a picture in Web Mercator is read with nearest-neighbour resampling only".into()));
        }
        if resample == ResampleMode::Nearest {
            // As rio-tiler does: the raster is read at the size its own resolution gives
            // for what was asked, the way GDAL's warp reads it, and brought to the
            // picture's size afterwards.
            let crs = engine.get_or_build_crs(&cog)?;
            let res = if mercator { mercator_resolution(&crs, &cog) } else { wgs84_resolution(&crs, &cog) }
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
            let out = (width as usize, height as usize);
            let mut planes: Vec<Vec<f32>> = Vec::with_capacity(wanted.len());
            for band in &wanted {
                let plane = read.get(usize::from(*band) - 1).cloned().ok_or_else(|| {
                    EngineError::BadRequest("unexpected number of planes from the reader".into())
                })?;
                planes.push(resize_nearest(plane, own, out));
            }
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
                                let (dx, dy) =
                                    ((bbox[2] - bbox[0]) / out.0 as f64, (bbox[3] - bbox[1]) / out.1 as f64);
                                let polygons: Vec<Vec<zonal::Ring>> = mask
                                    .polygons
                                    .iter()
                                    .map(|polygon| {
                                        polygon
                                            .iter()
                                            .map(|ring| {
                                                ring.iter()
                                                    .map(|&(lon, lat)| {
                                                        // The shape is in degrees whatever the grid is in.
                                                        let (across, up) = if mercator {
                                                            cogzilla_proj::mercator::lonlat_deg_to_mercator(lon, lat)
                                                        } else {
                                                            (lon, lat)
                                                        };
                                                        ((across - bbox[0]) / dx, (bbox[3] - up) / dy)
                                                    })
                                                    .collect()
                                            })
                                            .collect()
                                    })
                                    .collect();
                                let burned = zonal::cover_counts(&polygons, out.0, out.1, 1).map_err(EngineError::BadRequest)?;
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
        if nodata.is_some() {
            return Err(nearest_only());
        }
        let cogs = [cog];
        let mut planes = Vec::with_capacity(wanted.len());
        for band in wanted {
            let bands = BandRequest::Single { band_1based: band, rescale: (0.0, 255.0), colormap: ColormapName::Grayscale };
            let values = render_wgs84_bbox(
                engine,
                &cogs,
                bbox,
                width,
                height,
                mask.as_ref().map(|(mask, _)| mask.as_ref()),
                &bands,
                resample,
                OutputFormat::RawF32,
                None,
                None,
                None,
                true,
                unscale,
            )
            .await?;
            if values.len() != pixels * 4 {
                return Err(EngineError::BadRequest("unexpected plane size from the renderer".into()));
            }
            planes.push(values.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect());
        }
        Ok((planes, sample_type))
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
                    if Fold::<f32>::held_bytes(self.selection, scenes.len(), bands, pixels) > MAX_HELD_BYTES {
                        return Err(EngineError::BadRequest(format!(
                            "{} items are too many for a median at this tile size; narrow the search or lower items_limit",
                            scenes.len()
                        )));
                    }
                    check_working_bytes(Fold::<f32>::working_bytes(self.selection, bands, pixels))?;
                    fold.insert((Fold::new(self.selection, bands, pixels, item.integer).absent_as(self.nodata.unwrap_or(0.0)), item.sample_type, item.widest))
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
        let pixels = width * height;
        let mut formula = look.formula;
        let (rgba, bands, full) = match self.mosaic(engine, scenes).await? {
            None => (vec![0u8; pixels * 4], 1, false),
            Some((selected, sample_type, widest)) => {
                // Without `rescale`: the type's own range for a raster's numbers, and the
                // renderer's default for band math.
                let math = self.expression.is_some() || self.assets.iter().any(|a| a.expression.is_some());
                let default = if math {
                    (-1.0, 1.0)
                } else {
                    super::default_rescale_for_dtype(sample_type, selected.bands == 1)
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
                if let (Some((OutputFormat::Npy, content_type)), false) = (look.asked, shown) {
                    // Band math comes out in numexpr's types: whole numbers of up to 16
                    // bits stay whole, as 32-bit ones, through sums, differences and
                    // products, and 32-bit floats stay so; a division, a function or a fraction
                    // makes doubles.
                    let whole = |e: &String| {
                        let mut rest = e.as_str();
                        while let Some(at) = rest.find('b') {
                            let digits = rest[at + 1..].bytes().take_while(u8::is_ascii_digit).count();
                            if digits == 0 {
                                return false;
                            }
                            rest = &rest[at + 1 + digits..];
                        }
                        !e.bytes().any(|c| c == b'/' || c == b'.' || (c.is_ascii_alphabetic() && c != b'b'))
                    };
                    let math_of = || self.expression.iter().chain(self.assets.iter().filter_map(|a| a.expression.as_ref()));
                    let whole_math = math_of().all(whole);
                    // float32 stays float32 through operations and functions alike; a literal
                    // with a fraction or an exponent is a double and makes the result one.
                    let plain_math = math_of().all(|e| {
                        let e = e.as_bytes();
                        !e.contains(&b'.') && !e.windows(2).any(|w| w[0].is_ascii_digit() && matches!(w[1], b'e' | b'E'))
                    });
                    // What the bands are before any math: floats once unscaled.
                    let read = if self.window.unscale() { SampleType::Float32 } else { widest };
                    let kind = if matches!(self.selection, Selection::Stdev) {
                        SampleType::Float64
                    } else if matches!(self.selection, Selection::Count) {
                        SampleType::Uint8
                    } else if !math {
                        read
                    } else if read == SampleType::Float32 && plain_math {
                        SampleType::Float32
                    } else if !whole_math {
                        SampleType::Float64
                    } else {
                        match read {
                            SampleType::Uint8 | SampleType::Int8 | SampleType::Uint16 | SampleType::Int16 => SampleType::Int32,
                            _ => SampleType::Float64,
                        }
                    };
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

/// The type that holds the numbers of both, as numpy promotes them when it stacks bands.
fn wider_type(a: SampleType, b: SampleType) -> SampleType {
    use SampleType as T;
    if a == b {
        return a;
    }
    let float = |t| matches!(t, T::Float32 | T::Float64);
    let signed = |t| matches!(t, T::Int8 | T::Int16 | T::Int32);
    let bits = |t| match t {
        T::Uint8 | T::Int8 => 8u32,
        T::Uint16 | T::Int16 => 16,
        T::Uint32 | T::Int32 | T::Float32 => 32,
        T::Float64 => 64,
    };
    if float(a) || float(b) {
        // float32 holds whole numbers of 16 bits and no more.
        let narrow = |t| if float(t) { bits(t) == 32 } else { bits(t) <= 16 };
        return if narrow(a) && narrow(b) { T::Float32 } else { T::Float64 };
    }
    let sized = |kinds: [SampleType; 3], width: u32| kinds[match width { 8 => 0, 16 => 1, _ => 2 }];
    match (signed(a), signed(b)) {
        (false, false) => sized([T::Uint8, T::Uint16, T::Uint32], bits(a).max(bits(b))),
        (true, true) => sized([T::Int8, T::Int16, T::Int32], bits(a).max(bits(b))),
        // A signed type one size up from the unsigned one, or the signed one's own if larger.
        _ => {
            let (own, unsigned) = if signed(a) { (bits(a), bits(b)) } else { (bits(b), bits(a)) };
            match (unsigned * 2).max(own) {
                16 => T::Int16,
                32 => T::Int32,
                _ => T::Float64,
            }
        }
    }
}

/// A NumPy array of `bands` planes of `width` by `height`, as TiTiler writes one: in the
/// type `kind`, a value that is not there as `absent` (the no-data value the request
/// named) or zero, and after the bands, with `mask`, one
/// more that holds the type's largest number where any band has a value and zero elsewhere.
fn npy_array(
    values: &[f32],
    bands: usize,
    width: usize,
    height: usize,
    kind: SampleType,
    mask: bool,
    absent: Option<f32>,
) -> Bytes {
    let pixels = width * height;
    let (descr, size) = match kind {
        SampleType::Uint8 => ("|u1", 1),
        SampleType::Int8 => ("|i1", 1),
        SampleType::Uint16 => ("<u2", 2),
        SampleType::Int16 => ("<i2", 2),
        SampleType::Uint32 => ("<u4", 4),
        SampleType::Int32 => ("<i4", 4),
        SampleType::Float32 => ("<f4", 4),
        SampleType::Float64 => ("<f8", 8),
    };
    let planes = bands + usize::from(mask);
    let mut dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({planes}, {height}, {width}), }}");
    // The header ends on a multiple of 64 bytes, in spaces and a newline.
    let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
    dict.extend(std::iter::repeat_n(' ', pad));
    dict.push('\n');
    let mut out = Vec::with_capacity(10 + dict.len() + planes * pixels * size);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    let absent = f64::from(absent.unwrap_or(0.0));
    let data = &values[..(bands * pixels).min(values.len())];
    // The values in the array's type, then the mask: the type's largest number where any
    // band has a value. One loop for each type, so the type is not asked about per value.
    macro_rules! write {
        ($t:ty, $cast:expr) => {{
            let cast = $cast;
            for v in data {
                let x: $t = cast(if v.is_nan() { absent } else { f64::from(*v) });
                out.extend_from_slice(&x.to_le_bytes());
            }
            if mask {
                for i in 0..pixels {
                    let there = (0..bands).any(|band| data.get(band * pixels + i).is_some_and(|v| !v.is_nan()));
                    let x: $t = if there { <$t>::MAX } else { cast(0.0) };
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }};
    }
    match kind {
        SampleType::Uint8 => write!(u8, |v: f64| v as u8),
        SampleType::Int8 => write!(i8, |v: f64| v as i8),
        SampleType::Uint16 => write!(u16, |v: f64| v as u16),
        SampleType::Int16 => write!(i16, |v: f64| v as i16),
        SampleType::Uint32 => write!(u32, |v: f64| v as u32),
        SampleType::Int32 => write!(i32, |v: f64| v as i32),
        // Band math's largest number is an infinity once it has been through a plane of f32.
        SampleType::Float32 => write!(f32, |v: f64| (v as f32).clamp(f32::MIN, f32::MAX)),
        SampleType::Float64 => write!(f64, |v: f64| v.clamp(f64::MIN, f64::MAX)),
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
            cogzilla_kernel::CustomColormap::Intervals(spans) => {
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
/// as a single byte. The renderer's own grey is not that byte (some levels of its grey
/// table are one below their index), so such a picture is painted here from the values.
fn one_grey_band(bands: &BandRequest) -> bool {
    matches!(
        bands,
        BandRequest::Single { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Spectral1 { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Spectral2 { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Spectral3 { colormap: ColormapName::Grayscale, .. }
            | BandRequest::CustomExpr { colormap: ColormapName::Grayscale, .. }
    )
}

/// What a colour formula makes of each grey level.
fn formula_levels(formula: &cogzilla_kernel::ColorFormula) -> [u8; 256] {
    let mut levels = [0u8; 256];
    for (level, out) in levels.iter_mut().enumerate() {
        let mut px = [level as u8, level as u8, level as u8, 255];
        formula.apply_buffer_in_place(&mut px, 4);
        *out = px[0];
    }
    levels
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
    let bad = || detail(StatusCode::BAD_REQUEST, "invalid z/x/y tile coordinate");
    let z = zs.parse::<u32>().map_err(|_| bad())?;
    let x = xs.parse::<u32>().map_err(|_| bad())?;
    if z > 30 || (x as u64) >= (1u64 << z) || (y as u64) >= (1u64 << z) {
        return Ok(None);
    }
    Ok(Some(TileCoord { z: z as u8, x, y }))
}

async fn tile_assets(pg: &PgStac, id: &str, tile: TileCoord, pairs: &[(String, String)]) -> Resp {
    let limits = match parse_limits(pairs) {
        Ok(l) => l,
        Err(r) => return r,
    };
    // `tilesize` matters only to the lean lookup, which samples the tile as its picture would.
    let side = query_value(pairs, "tilesize").and_then(|v| v.parse::<u32>().ok()).unwrap_or(256);
    let grid = SampleGrid::for_image(side, side, true);
    match listed_items(pg, id, &tile_polygon(tile), limits, pairs, grid).await {
        Ok(items) => json_resp(
            StatusCode::OK,
            &Value::Array(items.iter().map(MosaicItem::to_json).collect()),
        ),
        Err(r) => r,
    }
}

/// TiTiler's answer for an asset the item does not have: its assets, in jsonb key order.
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

/// The resolution, in degrees, GDAL would give a raster warped to WGS84: the length of
/// its warped diagonal over the length of its diagonal in pixels.
fn wgs84_resolution(crs: &CrsModel, cog: &cogzilla_io::OpenedCog) -> Option<f64> {
    let full = cog.index.full_res();
    let (w, h) = (f64::from(full.width), f64::from(full.height));
    let corner = |col: f64, row: f64| {
        let (x, y) = full.pixel_to_crs.forward(col, row);
        crs.to_lonlat_deg(x, y)
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
    let planes_needed = wants_planes(&pq, &query) || custom.is_some() || pq.nodata.is_some() || array;
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
    if !given_in_metres && let Err(r) = check_coord_crs(pairs) {
        return r;
    }
    // `dst_crs` is read as `coord_crs` is, in any of the spellings TiTiler takes.
    let mercator = match query_value(pairs, "dst_crs").map(|dst| (dst, super::parse_coord_crs_with_epsg(dst))) {
        None => given_in_metres,
        Some((_, Ok((_, 4326)))) => false,
        Some((_, Ok((_, 3857)))) => true,
        Some((dst, _)) => {
            return detail(
                StatusCode::BAD_REQUEST,
                format!("dst_crs '{dst}' is not supported on pgSTAC mosaic routes (use epsg:4326 or epsg:3857)"),
            );
        }
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
    if mercator && !(nearest && planes_can && pq.algorithm.is_none()) {
        return detail(
            StatusCode::BAD_REQUEST,
            "dst_crs 'epsg:3857' is drawn with nearest-neighbour resampling only, and not with `algorithm` or band math only the renderer reads",
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
    let mut scenes: Vec<Vec<String>> = Vec::with_capacity(items.len().min(MAX_SCENES_PER_TILE));
    for item in items.iter().take(MAX_SCENES_PER_TILE) {
        let mut hrefs = Vec::with_capacity(names.len());
        for asset in &names {
            match item.asset_href(asset) {
                Some(href) => hrefs.push(href.to_string()),
                None => return unknown_asset(asset, item),
            }
        }
        scenes.push(hrefs);
    }

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
            first.push(engine.store.open_uri(href).await?);
        }
        // A raster with an alpha band is the renderer's too: it reads the alpha as the mask.
        // So is one of two bands, or four and more, asked for plainly: it draws the first,
        // or the first three, where planes would stack them all and TiTiler refuse most.
        let alpha = first.iter().any(|cog| !cog.index.full_res().extra_samples.is_empty())
            || (first.len() == 1 && !matches!(first[0].index.full_res().samples_per_pixel, 1 | 3));
        // Those the renderer draws, and it draws in degrees only.
        if mercator && alpha && !planes_needed {
            return Err(EngineError::BadRequest(
                "dst_crs 'epsg:3857' is not supported for a raster with an alpha band, or of two bands or more than three asked for whole".into(),
            ));
        }
        if mercator || planes_needed || !TileEngine::on_one_grid(&first) || (planes_preferred && !alpha) {
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
                let own = if mercator { mercator_resolution(&crs, cog) } else { wgs84_resolution(&crs, cog) };
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
            return Ok(as_geotiff(made, w as usize, h as usize, bbox, if mercator { 3857 } else { 4326 }, pq.return_mask));
        }
        let mut canvas: Vec<u8> = Vec::new();
        let (mut width, mut height) = (0usize, 0usize);
        // With a pixel selection other than `first`: every scene's values, and how to colour them.
        let mut selecting: Option<(Fold, Paint)> = None;
        // One band under a colour map, with a colour formula: drawn grey, coloured at the end.
        let mut map_after: Option<MapAfterFormula> = None;
        let mut one_grey = false;
        for (n, scene) in scenes.iter().enumerate() {
            let mut cogs = Vec::with_capacity(scene.len());
            for href in scene {
                cogs.push(engine.store.open_uri(href).await?);
            }
            let crs = engine.get_or_build_crs(&cogs[0])?;
            let full = cogs[0].index.full_res();
            if query.asset_as_band && full.samples_per_pixel > 1 {
                return Err(EngineError::BadRequest("Can't use `asset_as_band` for multibands asset".into()));
            }
            if n == 0 {
                // The first item sets the size; the others are drawn onto its grid.
                let res = wgs84_resolution(&crs, &cogs[0]).ok_or_else(|| {
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
            let resample = match crs {
                CrsModel::Geographic => query.resampling,
                _ => query.reproject,
            };
            if let Some(method) = query.selection.or(exact.then_some(Selection::First)) {
                // The scene's values rather than its colours; only `first` stops early.
                let count = TileEngine::value_bands(&bands);
                if n == 0 {
                    let paint = Paint::of(&bands).ok_or_else(|| {
                        EngineError::BadRequest("pixel_selection other than 'first' cannot be used with a terrain encoding".into())
                    })?;
                    if Fold::<f32>::held_bytes(method, scenes.len(), count, width * height) > MAX_HELD_BYTES {
                        return Err(EngineError::BadRequest(format!(
                            "{} items are too many for a median at this size; narrow the search, lower items_limit or pass max_size",
                            scenes.len()
                        )));
                    }
                    check_working_bytes(Fold::<f32>::working_bytes(method, count, width * height))?;
                    let integer = whole_numbers(pq.unscale, &bands, full.sample_type);
                    selecting = Some((Fold::new(method, count, width * height, integer), paint));
                }
                let values = render_wgs84_bbox(
                    engine,
                    &cogs,
                    bbox,
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
                if let Some((fold, _)) = &mut selecting
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
                apply_unscale_cogs(&mut bands, &cogs);
            }
            let rgba = render_wgs84_bbox(
                engine,
                &cogs,
                bbox,
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
        if let Some((fold, paint)) = selecting
            && let Some(selected) = fold.finish()
        {
            canvas = paint.rgba(&selected, width * height);
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
        let (format, content_type) = match format {
            Some((f, ct)) => (f, if explicit.as_deref() == Some("jpg") { "image/jpg" } else { ct }),
            None => {
                let opaque = canvas.chunks_exact(4).all(|p| p[3] == 255);
                parse_ext_format(if opaque { "jpg" } else { "png" }, None, pq.jpeg_quality)?
            }
        };
        // One band, or a count, is a grey picture as on tiles: one channel, or two with the mask.
        let grey = one_grey && canvas.chunks_exact(4).all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
        let made = if grey {
            encode_picture(canvas, width, height, None, Some((format, content_type)), pq.jpeg_quality, pq.return_mask, true)?
        } else {
            (encode_rgba_output_colour(width, height, &canvas, format, pq.return_mask), content_type)
        };
        Ok(as_geotiff(made, width, height, bbox, 4326, pq.return_mask))
    }
    .await;
    match rendered {
        Ok((body, content_type)) => {
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
        ["feature", dims] => match parse_wxh_ext(dims).ok() {
            Some((w, h, ext)) if w > 0 && h > 0 => (Ok(PartSize::Exact(w, h)), ext),
            _ => return detail(StatusCode::NOT_FOUND, "Not Found"),
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
    let (mask, bbox) = match polygons_of_collection(&geojson).and_then(|shape| PolygonMask::from_geojson(&shape)) {
        Ok(m) => m,
        Err(e) => return engine_error(&e),
    };
    // As in titiler-pgstac, the items are those of the feature's bounding box.
    part_image(ctx, pg, hash, &bbox_polygon(bbox), bbox, Some(&mask), size, ext, pq, pairs).await
}

/// A feature whose geometry is a GeometryCollection of polygons, as the one MultiPolygon
/// the mask reads: GDAL burns every member, which is their union. Any other feature is
/// returned as it is. A collection with a line or a point in it, or with nothing, is
/// refused: lines and points are not burned here.
fn polygons_of_collection(feature: &Value) -> Result<std::borrow::Cow<'_, Value>, EngineError> {
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
    if geometry.get("type").and_then(Value::as_str) != Some("GeometryCollection") {
        return Ok(std::borrow::Cow::Borrowed(feature));
    }
    let mut polygons = Vec::new();
    if !gather(geometry, &mut polygons, 0) || polygons.is_empty() {
        return Err(EngineError::BadRequest(
            "a GeometryCollection is taken when every member is a Polygon or a MultiPolygon, and it has one".into(),
        ));
    }
    // The geometry alone, which the mask reads as it reads a feature: one copy of the
    // coordinates, checked against the mask's own limits as any other shape is.
    Ok(std::borrow::Cow::Owned(json!({ "type": "MultiPolygon", "coordinates": polygons })))
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
/// of [`wgs84_resolution`].
fn wgs84_grid(crs: &CrsModel, cog: &cogzilla_io::OpenedCog) -> Option<Wgs84Grid> {
    let res = wgs84_resolution(crs, cog)?;
    let full = cog.index.full_res();
    let (w, h) = (f64::from(full.width), f64::from(full.height));
    let (mut west, mut north) = (f64::INFINITY, f64::NEG_INFINITY);
    let steps = ((w.min(h) / 50.0 + 0.5) as u32).clamp(20, 100);
    for i in 0..=steps {
        let f = f64::from(i) / f64::from(steps);
        for (col, row) in [(f * w, 0.0), (f * w, h), (0.0, f * h), (w, f * h)] {
            let (x, y) = full.pixel_to_crs.forward(col, row);
            let (lon, lat) = crs.to_lonlat_deg(x, y);
            if lon.is_finite() && lat.is_finite() {
                west = west.min(lon);
                north = north.max(lat);
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

/// The resolution GDAL gives a raster warped to Web Mercator, in metres: as
/// [`wgs84_resolution`], the diagonal there over the diagonal in pixels.
fn mercator_resolution(crs: &CrsModel, cog: &cogzilla_io::OpenedCog) -> Option<f64> {
    let full = cog.index.full_res();
    let (w, h) = (f64::from(full.width), f64::from(full.height));
    let corner = |col: f64, row: f64| {
        let (x, y) = full.pixel_to_crs.forward(col, row);
        let (lon, lat) = crs.to_lonlat_deg(x, y);
        lonlat_deg_to_mercator(lon, lat)
    };
    let (ul, lr) = (corner(0.0, 0.0), corner(w, h));
    let res = (lr.0 - ul.0).hypot(lr.1 - ul.1) / w.hypot(h);
    (res.is_finite() && res > 0.0).then_some(res)
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

/// Every band of one raster on a WGS84 grid through the shared renderer, for the
/// resampling methods other than nearest neighbour.
async fn resampled_planes(
    engine: &Arc<TileEngine>,
    cog: &Arc<cogzilla_io::OpenedCog>,
    bounds: [f64; 4],
    (width, height): (usize, usize),
    resample: ResampleMode,
    unscale: bool,
) -> Result<Vec<Vec<f32>>, EngineError> {
    let mut planes = Vec::new();
    for band in 1..=cog.index.full_res().samples_per_pixel {
        let one = BandRequest::Single {
            band_1based: band,
            rescale: (0.0, 1.0),
            colormap: ColormapName::Grayscale,
        };
        let raw = render_wgs84_bbox(
            engine,
            std::slice::from_ref(cog),
            bounds,
            width as u32,
            height as u32,
            None,
            &one,
            resample,
            OutputFormat::RawF32,
            None,
            None,
            None,
            false,
            unscale,
        )
        .await?;
        if raw.len() != width * height * 4 {
            return Err(EngineError::BadRequest("unexpected array size from the renderer".into()));
        }
        planes.push(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }
    Ok(planes)
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
        let parts: Option<Vec<u32>> = v.split(',').map(|p| p.parse::<u32>().ok()).collect();
        opts.bins = match parts.as_deref() {
            Some([n]) if (1..=65_536).contains(n) => Bins::Count(*n as usize),
            Some(edges) if edges.len() > 1 && edges.windows(2).all(|w| w[0] <= w[1]) => {
                Bins::Edges(edges.iter().map(|e| f64::from(*e)).collect())
            }
            _ => return Err(bad()),
        };
    }
    if let Some(v) = query_value(pairs, "histogram_range") {
        let range = v
            .split_once(',')
            .and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?)))
            .filter(|(a, b)| a.is_finite() && b.is_finite() && a <= b);
        opts.range = Some(range.ok_or_else(|| {
            invalid_param("query", "histogram_range", "Input should be two numbers, the lower one first")
        })?);
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
struct StatsRequest<'a> {
    query: MosaicQuery,
    size: PartSize,
    opts: StatsOptions,
    cover_scale: usize,
    expression: Option<&'a str>,
    unscale: bool,
    /// A no-data value of the request's own, in the place of each raster's.
    nodata: Option<f32>,
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
    let (mask, bbox) = polygons_of_collection(feature)
        .and_then(|shape| PolygonMask::from_geojson(&shape))
        .map_err(|e| engine_error(&e))?;
    // As in titiler-pgstac, the items are those of the feature's bounding box. Each raster
    // is read on a grid of its own, which no one set of sample points stands for, so the
    // numbers come from pgSTAC's list unless the request asks for the lean one.
    let names: Vec<String> = req.query.assets.iter().map(|a| a.name.clone()).collect();
    let lookup = if req.query.lookup == Lookup::Lean { Lookup::Lean } else { Lookup::Reference };
    let items = pg
        .items_for_pixels(&bbox_polygon(bbox), hash, req.query.limits, &names, part_grid(req.size, bbox), lookup)
        .await
        .map_err(|e| db_error(&e))?;
    if items.is_empty() {
        return Ok(None);
    }
    let mut scenes: Vec<(String, Vec<String>)> = Vec::with_capacity(items.len().min(MAX_SCENES_PER_TILE));
    let assets = &req.query.assets;
    for item in items.iter().take(MAX_SCENES_PER_TILE) {
        let mut hrefs = Vec::with_capacity(assets.len());
        for asset in assets {
            match item.asset_href(&asset.name) {
                Some(href) => hrefs.push(href.to_string()),
                None => return Err(unknown_asset(&asset.name, item)),
            }
        }
        let name = match &item.collection {
            Some(c) => format!("{c}/{}", item.id),
            None => item.id.clone(),
        };
        scenes.push((name, hrefs));
    }

    let engine = ctx.engine;
    let computed: Result<Option<(Value, Vec<String>)>, EngineError> = async {
        // The mosaic: one plane per band, a pixel nothing has filled yet being NaN.
        let mut canvas: Vec<Vec<f64>> = Vec::new();
        let mut descriptions: Vec<String> = Vec::new();
        let mut program: Option<Program> = None;
        let mut program_bands = 0;
        let mut used: Vec<String> = Vec::new();
        let (mut width, mut height, mut bounds) = (0usize, 0usize, bbox);
        // With a pixel selection other than `first`, every scene's bands go through this
        // and nothing stops the reading early.
        let mut fold: Option<Fold<f64>> = None;
        let scene_count = scenes.len();
        for (name, hrefs) in scenes {
            let mut cogs = Vec::with_capacity(hrefs.len());
            for href in &hrefs {
                cogs.push(engine.store.open_uri(href).await?);
            }
            let mut planes: Vec<Vec<f64>> = Vec::new();
            let mut names: Vec<String> = Vec::new();
            let mut missing = false;
            'assets: for (asset, cog) in assets.iter().zip(&cogs) {
                // As rio-tiler does, every raster is read on its own grid: the feature's box
                // grown to whole pixels of the raster as GDAL would warp it to WGS84. The
                // first one read sets the size of the result.
                let crs = engine.get_or_build_crs(cog)?;
                let grid = wgs84_grid(&crs, cog).ok_or_else(|| {
                    EngineError::BadRequest(format!("cannot place asset '{}' in WGS84", asset.name))
                })?;
                let own_bounds = grid.align(bbox);
                let own = PartSize::Query { max_size: None, width: None, height: None };
                let native = part_dims(own, own_bounds, grid.res);
                let (w, h) = part_dims(req.size, own_bounds, grid.res);
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
                    CrsModel::Geographic => req.query.resampling,
                    _ => req.query.reproject,
                };
                // GDAL takes another road when only some of a raster's bands are read.
                let all = usize::from(cog.index.full_res().samples_per_pixel);
                let path = match &asset.indexes {
                    Some(indexes) if asset.expression.is_none() && indexes.len() != all => WarpPath::Blocks,
                    _ => WarpPath::Direct,
                };
                let read = if resample == ResampleMode::Nearest {
                    warp_nearest(engine, cog, &crs, own_bounds, false, size, native, path, req.unscale, true, &[], false, req.nodata).await
                } else if req.nodata.is_some() {
                    Err(EngineError::BadRequest("`nodata` is read with nearest-neighbour resampling only".into()))
                } else {
                    resampled_planes(engine, cog, own_bounds, size, resample, req.unscale).await
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
                if let Some(method) = req.query.selection {
                    // The last band decides and is left out: alone, it would leave nothing.
                    if method.by_last_band() && canvas.len() < 2 {
                        return Err(EngineError::BadRequest(
                            "pixel_selection by the last band needs at least two bands".into(),
                        ));
                    }
                    if Fold::<f64>::held_bytes(method, scene_count, canvas.len(), width * height) > MAX_HELD_BYTES {
                        return Err(EngineError::BadRequest(format!(
                            "{scene_count} items are too many for a median at this size; narrow the search, lower items_limit or pass max_size"
                        )));
                    }
                    // Whole numbers only when the values are the first raster's own.
                    let integer = program.is_none()
                        && !req.unscale
                        && assets.iter().all(|a| a.expression.is_none())
                        && !matches!(cogs[0].index.full_res().sample_type, SampleType::Float32 | SampleType::Float64);
                    check_working_bytes(Fold::<f64>::working_bytes(method, canvas.len(), width * height))?;
                    fold = Some(Fold::new(method, canvas.len(), width * height, integer).absent_as(req.nodata.map_or(0.0, f64::from)));
                }
            }
            // The expression was read for the first item's bands; an item with another number
            // of them has no band for it to name.
            if program.is_some() && planes.len() != program_bands {
                return Err(EngineError::BadRequest(
                    "the items under the feature do not have the same number of bands".into(),
                ));
            }
            if let Some(fold) = &mut fold {
                let bands: Vec<Vec<f64>> = match &program {
                    Some(program) => {
                        let mut bands = vec![vec![f64::NAN; width * height]; canvas.len()];
                        let mut pixel = vec![0f64; planes.len()];
                        for i in 0..width * height {
                            for (value, plane) in pixel.iter_mut().zip(&planes) {
                                *value = plane[i];
                            }
                            for (part, band) in bands.iter_mut().enumerate() {
                                let v = program.eval(part, &pixel);
                                if v.is_finite() {
                                    band[i] = v;
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
                fold.add(bands.iter().flatten().map(|v| if v.is_finite() { *v } else { f64::NAN }));
                used.push(name);
                continue;
            }
            match &program {
                Some(program) => {
                    let mut pixel = vec![0f64; planes.len()];
                    for i in 0..width * height {
                        if canvas.iter().all(|band| !band[i].is_nan()) {
                            continue;
                        }
                        for (value, plane) in pixel.iter_mut().zip(&planes) {
                            *value = plane[i];
                        }
                        for (part, band) in canvas.iter_mut().enumerate() {
                            if band[i].is_nan() {
                                let v = program.eval(part, &pixel);
                                if v.is_finite() {
                                    band[i] = v;
                                }
                            }
                        }
                    }
                }
                None => {
                    if planes.len() != canvas.len() {
                        return Err(EngineError::BadRequest(
                            "the items under the feature do not have the same number of bands".into(),
                        ));
                    }
                    for (band, plane) in canvas.iter_mut().zip(&planes) {
                        for (dst, src) in band.iter_mut().zip(plane) {
                            if dst.is_nan() && src.is_finite() {
                                *dst = *src;
                            }
                        }
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
            // `count` leaves one band, described as the first.
            canvas = selected.values.chunks(width * height).map(<[f64]>::to_vec).collect();
            descriptions.truncate(selected.bands);
        }

        // The polygon on the image's grid: once for the mask, once finer for the coverage.
        let (sx, sy) = (
            width as f64 / (bounds[2] - bounds[0]),
            height as f64 / (bounds[3] - bounds[1]),
        );
        let rings: Vec<Vec<zonal::Ring>> = mask
            .polygons
            .iter()
            .map(|polygon| {
                polygon
                    .iter()
                    .map(|ring| ring.iter().map(|&(lon, lat)| ((lon - bounds[0]) * sx, (bounds[3] - lat) * sy)).collect())
                    .collect()
            })
            .collect();
        let (opts, cover_scale) = (req.opts.clone(), req.cover_scale);
        let stats = engine
            .store
            .run_heavy_cpu(move || {
                let inside = zonal::cover_counts(&rings, width, height, 1).map_err(EngineError::BadRequest)?;
                let cells = (cover_scale * cover_scale) as f32;
                let coverage: Vec<f32> = zonal::cover_counts(&rings, width, height, cover_scale)
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
                    out.insert(
                        format!("b{}", n + 1),
                        zonal::band_statistics(band, &coverage, description, &opts),
                    );
                }
                Ok::<_, EngineError>(Value::Object(out))
            })
            .await??;
        Ok(Some((stats, used)))
    }
    .await;
    computed.map_err(|e| engine_error(&e))
}

/// `POST {base}/statistics`: statistics of the mosaic under each feature of a GeoJSON
/// Feature or FeatureCollection, added to the feature's properties with the items used.
async fn mosaic_statistics(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    hash: &str,
    body: Option<&Bytes>,
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
    if let Err(r) = check_coord_crs(pairs).and_then(|()| check_dst_crs(pairs)) {
        return r;
    }
    if query_value(pairs, "algorithm").is_some() {
        return detail(StatusCode::BAD_REQUEST, "parameter 'algorithm' is not supported on statistics routes yet");
    }
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
        expression: query_value(pairs, "expression"),
        unscale: pq.unscale,
        nodata: pq.nodata,
    };
    let mut out = Vec::with_capacity(features.len());
    for feature in &features {
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
        properties.insert("used_assets".into(), json!(used));
        let mut one = serde_json::Map::new();
        one.insert("type".into(), json!("Feature"));
        one.insert("geometry".into(), feature["geometry"].clone());
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
    let xy = coord
        .split_once(',')
        .and_then(|(x, y)| Some((x.parse::<f64>().ok()?, y.parse::<f64>().ok()?)))
        .filter(|(x, y)| x.is_finite() && y.is_finite());
    let Some((x, y)) = xy else {
        return detail(StatusCode::NOT_FOUND, "Not Found");
    };
    if pq.assets.is_empty() {
        return missing_query_param("assets");
    }
    if let Some(e) = pq.errors.iter().find(|e| !own_error(e)) {
        return detail(StatusCode::BAD_REQUEST, e.clone());
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
        if let Err(r) = check_coord_crs(pairs) {
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
        let mut hrefs = Vec::with_capacity(specs.len());
        for spec in &specs {
            match item.asset_href(&spec.name) {
                Some(href) => hrefs.push((spec.clone(), href.to_string())),
                None => return unknown_asset(&spec.name, item),
            }
        }
        let name = match &item.collection {
            Some(c) => format!("{c}/{}", item.id),
            None => item.id.clone(),
        };
        jobs.push((name, hrefs));
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
                let cog = engine.store.open_uri(href).await?;
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
async fn tile(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    id: &str,
    tile: TileCoord,
    scale: Option<u32>,
    ext: &str,
    mut pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    // A tile with a `buffer` is the tile's box grown by that many pixels on every side,
    // at a size grown as much: to rio-tiler, and here, a box in Web Mercator like any other.
    if pq.buffer > 0.0 {
        let buffer_asked = pq.buffer;
        if scale.is_some_and(|s| !(1..=8).contains(&s)) {
            return detail(StatusCode::BAD_REQUEST, "a tile's scale is from @1x to @8x");
        }
        pq.buffer = 0.0;
        // What the renderer draws of a tile, it draws without a `buffer`: said here, in a
        // tile's words, sooner than further on in a box's.
        let raw = matches!(ext, "bin" | "raw" | "f32") || matches!(query_value(pairs, "format"), Some("bin" | "raw" | "f32"));
        match check_mosaic_query(&pq, pairs) {
            Ok(q)
                if q.resampling != ResampleMode::Nearest
                    || q.reproject != ResampleMode::Nearest
                    || pq.algorithm.is_some()
                    || raw
                    || !planes_read_all(&pq, &q) =>
            {
                return detail(
                    StatusCode::BAD_REQUEST,
                    "a tile with a `buffer` is drawn with nearest-neighbour resampling, as a picture or `.npy`, and without `algorithm` or band math only the renderer reads",
                );
            }
            Ok(_) => {}
            Err(r) => return r,
        }
        let buffer = buffer_asked;
        // As in rio-tiler, which grows the tile by whole pixels: halves, since both sides grow.
        if (buffer * 2.0).fract() != 0.0 {
            return detail(StatusCode::BAD_REQUEST, "`buffer` must be a multiple of `0.5`");
        }
        let side = scale.map_or(pq.tile_size, |s| s * 256);
        let grown = (side as f32 + 2.0 * buffer).round() as u32;
        let bbox = super::web_mercator_buffered_3857_bbox(tile.z, tile.x, tile.y, buffer, side);
        pq.coord_crs_3857 = true;
        // What a box reads and a tile passes over: the system is the tile's, and so is the size.
        const OF_A_BOX: [&str; 6] = ["buffer", "coord_crs", "dst_crs", "width", "height", "max_size"];
        let pairs: Vec<(String, String)> = pairs.iter().filter(|(k, _)| !OF_A_BOX.contains(&k.as_str())).cloned().collect();
        let ext = (!ext.is_empty()).then_some(ext);
        let size = PartSize::Exact(grown, grown);
        let under = tile_polygon(tile);
        return Box::pin(part_image_of(ctx, pg, id, &under, Some(&under), bbox, None, size, ext, pq, &pairs)).await;
    }
    let query = match check_mosaic_query(&pq, pairs) {
        Ok(q) => q,
        Err(r) => return r,
    };
    // What the renderer cannot draw in one pass is made from planes instead. So is a tile
    // with nearest-neighbour resampling, whose planes are read the way GDAL's warp reads
    // and are TiTiler's pixels; the renderer draws the other methods, the algorithms, and
    // band math only its own engine reads.
    // A colour map given as JSON is applied to planes; the renderer on this road has none.
    let custom = match pq.custom_colormap_json.as_deref().map(MosaicColormap::parse).transpose() {
        Ok(map) => map.flatten(),
        Err(e) => return detail(StatusCode::BAD_REQUEST, e),
    };
    // So is a `nodata` of the request's own: the renderer on this road reads the raster's.
    let nodata = pq.nodata;
    let planes_wanted = wants_planes(&pq, &query) || custom.is_some() || nodata.is_some();
    let planes_preferred = planes_read_all(&pq, &query)
        && query.resampling == ResampleMode::Nearest
        && query.reproject == ResampleMode::Nearest
        && pq.algorithm.is_none();
    // The request's own band math, before an asset's is put in its place for the renderer.
    let expression = pq.expr_raw.clone();
    let names = if planes_wanted { expression_names(&pq, &query) } else { renderer_assets(&mut pq, &query) };
    let names = match names {
        Ok(names) => names,
        Err(r) => return r,
    };
    // `@{n}x` is a tile of n times 256 pixels, within the sizes `tilesize` takes.
    if scale.is_some_and(|s| !(1..=8).contains(&s)) {
        return detail(StatusCode::BAD_REQUEST, "a tile's scale is from @1x to @8x");
    }
    // A `nodata` of the request's own leaves the listing to pgSTAC's search: the lean
    // lookup passes over an item whose footprint adds nothing, and its raster may not.
    let grid = if pq.nodata.is_some() { None } else { tile_grid(scale, &pq) };
    let items = match pixel_items(ctx, pg, &tile_polygon(tile), id, query.limits, &names, grid, query.lookup).await {
        Ok(items) => items,
        Err(e) => return db_error(&e),
    };
    if items.is_empty() {
        return no_content();
    }

    let mut scenes: Vec<Vec<String>> = Vec::with_capacity(items.len().min(MAX_SCENES_PER_TILE));
    for item in items.iter().take(MAX_SCENES_PER_TILE) {
        let mut hrefs = Vec::with_capacity(names.len());
        for asset in &names {
            match item.asset_href(asset) {
                Some(href) => hrefs.push(href.to_string()),
                None => return unknown_asset(asset, item),
            }
        }
        scenes.push(hrefs);
    }

    // An explicit extension, then `format=`, then TiTiler's rule: JPEG when every pixel is
    // valid and PNG when some are not.
    let explicit = if ext.is_empty() {
        pq.format_param.clone().unwrap_or_default()
    } else {
        ext.to_string()
    };
    let auto_format = explicit.is_empty();
    let (resampling, reproject) = (query.resampling, query.reproject);
    let asset_as_band = query.asset_as_band;
    let selection = query.selection;
    let return_mask = pq.return_mask;
    let jpeg_quality = pq.jpeg_quality;
    let engine = ctx.engine;
    let specs = query.assets;
    // The tile is of these rasters: a search that comes to list others, or an item whose
    // asset has moved, is another tile, not this one served again.
    let cache_key = format!("{}?{}#{:x}", ctx.path, ctx.raw_query, source_identity(&scenes));
    let rendered = engine
        .tile_cache
        .get_or_render(cache_key, || async {
            // A tile asked for as a GeoTIFF is one, placed where the tile lies.
            let (west, south, east, north) = tile_bounds_3857(tile);
            let lies = [west, south, east, north];
            // The first scene is opened here, to learn what the tile's bands are. The scenes
            // after it are put on their way meanwhile, so the mosaic does not start their
            // headers only once this one has come back. The list holds only scenes whose
            // footprints add to the tile, so these are opens the tile is going to need.
            for uri in scenes.iter().skip(1).take(MOSAIC_SCENES_AHEAD - 1).flatten() {
                let (store, uri) = (engine.store.clone(), uri.clone());
                tokio::spawn(async move {
                    let _ = store.open_uri(&uri).await;
                });
            }
            let first = engine.store.open_uri(&scenes[0][0]).await?;
            let full = first.index.full_res();
            let formula = pq
                .color_formula
                .as_deref()
                .map(cogzilla_kernel::ColorFormula::parse)
                .transpose()
                .map_err(EngineError::BadRequest)?;
            // Assets that do not lie on one grid cannot be read as the bands of one image.
            let mut one_grid = true;
            if !planes_wanted && scenes[0].len() > 1 {
                let mut cogs = vec![first.clone()];
                for uri in &scenes[0][1..] {
                    cogs.push(engine.store.open_uri(uri).await?);
                }
                one_grid = TileEngine::on_one_grid(&cogs);
            }
            // Left to the renderer: a raster with an alpha band, which it reads as the mask,
            // and raw arrays.
            let dim = scale.map_or(pq.tile_size, |s| s.saturating_mul(256));
            let raw = matches!(explicit.as_str(), "bin" | "raw" | "f32");
            // A NumPy array is written from planes, in the data's own type and with its
            // mask, whatever the number of bands.
            let array = explicit == "npy";
            // And a raster of two bands, or four and more: the renderer draws the first, or
            // the first three, where planes would stack them all and TiTiler refuse most.
            let preferred = planes_preferred
                && !raw
                && full.extra_samples.is_empty()
                && (array || scenes[0].len() > 1 || matches!(full.samples_per_pixel, 1 | 3));
            // A plain request that takes this road only because it is the better reader keeps
            // the bands of its first raster for every item, as the renderer does: a later
            // item with more bands gives the same ones, not a tile of another shape.
            let mut specs = specs.clone();
            if !planes_wanted && one_grid && preferred && specs.len() == 1 && scenes[0].len() == 1 {
                let spec = &mut specs[0];
                if spec.indexes.is_none() && spec.expression.is_none() {
                    spec.indexes = Some((1..=full.samples_per_pixel).collect());
                }
            }
            if planes_wanted || !one_grid || preferred {
                if pq.algorithm.is_some() {
                    return Err(EngineError::BadRequest(
                        "`algorithm` is not supported with these assets and options yet".into(),
                    ));
                }
                let asked = if auto_format {
                    None
                } else {
                    let (format, content_type) = parse_ext_format(&explicit, pq.webp_quality, jpeg_quality)?;
                    if matches!(format, OutputFormat::RawF32) {
                        return Err(EngineError::BadRequest(
                            "raw array formats are not supported with these assets and options yet".into(),
                        ));
                    }
                    Some((format, if explicit == "jpg" { "image/jpg" } else { content_type }))
                };
                let job = PlanesTile {
                    assets: specs.clone(),
                    as_band: asset_as_band,
                    expression: expression.clone(),
                    selection: selection.unwrap_or(Selection::First),
                    resampling,
                    reproject,
                    nodata,
                    array,
                    window: Window::Tile(crate::engine::TileRequest {
                        tile,
                        tile_size: dim,
                        bands: BandRequest::Single { band_1based: 1, rescale: (0.0, 255.0), colormap: ColormapName::Grayscale },
                        resample: reproject,
                        format: OutputFormat::RawF32,
                        unscale: pq.unscale,
                    }),
                    pool: scenes.len() * specs.len() > 1,
                };
                let look = Look {
                    rescale: &pq.rescale,
                    colormap: pq.colormap,
                    mapped: pq.colormap_explicit,
                    custom: custom.as_ref(),
                    formula: formula.as_ref(),
                    asked,
                    jpeg_quality,
                    return_mask,
                };
                let made = job.picture(engine, &scenes, dim as usize, dim as usize, &look).await?;
                return Ok(as_geotiff(made, dim as usize, dim as usize, lies, 3857, return_mask));
            }
            // On planes the bands are counted after an asset's own options; here they are the raster's.
            if asset_as_band && full.samples_per_pixel > 1 {
                return Err(EngineError::BadRequest("Can't use `asset_as_band` for multibands asset".into()));
            }
            let default_bands = if scenes[0].len() > 1 {
                scenes[0].len() as u16
            } else {
                full.samples_per_pixel
            };
            // The renderer's bands are one or three: ranges that are not one each are the
            // first for all, as TiTiler has it (the shared routes refuse two for one band).
            let pq = if pq.rescale.len() > 1 && pq.rescale.len() != 3 {
                let mut first = pq.clone();
                first.rescale.truncate(1);
                std::borrow::Cow::Owned(first)
            } else {
                std::borrow::Cow::Borrowed(&pq)
            };
            let pq: &ParsedQuery = &pq;
            let (mut treq, content_type) = build_tile_request(
                Some(&engine),
                tile,
                scale,
                Some(explicit.as_str()),
                &pq,
                default_bands,
                full.sample_type,
                false,
                Some(&first),
            )?;
            // As in TiTiler, `resampling` applies to a source already in the tile's projection
            // and `reproject` to one that has to be warped into it.
            treq.resample = match engine.get_or_build_crs(&first)? {
                CrsModel::Epsg3857 => resampling,
                _ => reproject,
            };
            // The renderer draws tiles of 256 and 512 pixels; any other size is read as
            // planes, which is where a request with nearest-neighbour pixels has gone.
            if treq.tile_size != 256 && treq.tile_size != 512 {
                return Err(EngineError::BadRequest(format!(
                    "tilesize={} is drawn with nearest-neighbour resampling, without `algorithm`, as a picture, and of rasters without an alpha band; use 256 or 512 otherwise",
                    treq.tile_size
                )));
            }
            let dim = treq.tile_size as usize;
            let raw_array = matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy);
            // TiTiler names the media type after the extension that was asked for.
            let content_type = if explicit == "jpg" { "image/jpg" } else { content_type };
            let asked = treq.format;
            // One band under a colour map, with a colour formula: drawn grey here, then
            // the formula, then the map.
            let map_after = match &formula {
                Some(formula) if pq.colormap_explicit && !raw_array => MapAfterFormula::take(formula, &mut treq.bands),
                _ => None,
            };
            // An array of a raster's own bands, read with another resampling than nearest: the
            // renderer's values, written in the raster's type with the mask as a band, as
            // an array from planes is. What makes a picture's bytes of them, band math and
            // `algorithm` are still the renderer's own float32.
            let own_bands = matches!(treq.bands, BandRequest::Single { .. } | BandRequest::Rgb { .. });
            if matches!(treq.format, OutputFormat::Npy)
                && own_bands
                && selection.is_none()
                && !pq.colormap_explicit
                && formula.is_none()
                && pq.expr_raw.is_none()
                && pq.algorithm.is_none()
                && full.extra_samples.is_empty()
                && !super::has_embedded_palette(&pq, &first, &treq.bands)
            {
                let bands = TileEngine::value_bands(&treq.bands);
                treq.format = OutputFormat::RawF32;
                let raw = engine.render_mosaic_scenes_ahead(&scenes, &treq, MOSAIC_SCENES_AHEAD).await?;
                if raw.len() == bands * dim * dim * 4 {
                    // Several assets read as bands: the type that holds them all.
                    let mut kind = full.sample_type;
                    for uri in scenes[0].iter().skip(1) {
                        kind = wider_type(kind, engine.store.open_uri(uri).await?.index.full_res().sample_type);
                    }
                    let kind = if pq.unscale { SampleType::Float32 } else { kind };
                    let whole = !matches!(kind, SampleType::Float32 | SampleType::Float64);
                    // A whole-number raster's values between pixels are rounded to its type.
                    let read = raw
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                        .map(|v| if whole { v.round() } else { v });
                    // With `rescale` the array is of the bytes a picture would show, band by
                    // band, in the ranges the picture's bands were given.
                    if !pq.rescale.is_empty() {
                        let range = |band: usize| match &treq.bands {
                            BandRequest::Rgb { rescale_min, rescale_max, .. } => (rescale_min[band.min(2)], rescale_max[band.min(2)]),
                            BandRequest::Single { rescale, .. } => *rescale,
                            _ => pq.rescale[0],
                        };
                        let values: Vec<f32> = read
                            .enumerate()
                            .map(|(i, v)| {
                                let (min, max) = range(i / (dim * dim));
                                if v.is_nan() { v } else { f32::from(rescale_byte(v, min, max)) }
                            })
                            .collect();
                        return Ok((npy_array(&values, bands, dim, dim, SampleType::Uint8, return_mask, None), content_type));
                    }
                    let values: Vec<f32> = read.collect();
                    return Ok((npy_array(&values, bands, dim, dim, kind, return_mask, None), content_type));
                }
                treq.format = OutputFormat::Npy;
            }
            // A formula on one band works on the band's byte, made here from the values.
            let exact = formula.is_some()
                && !raw_array
                && one_grey_band(&treq.bands)
                && !super::has_embedded_palette(&pq, &first, &treq.bands);
            let mut rgba = if let (None, true) = (selection, exact) {
                let paint = Paint::of(&treq.bands).ok_or_else(|| EngineError::BadRequest("not a band of values".into()))?;
                treq.format = OutputFormat::RawF32;
                let raw = engine.render_mosaic_scenes_ahead(&scenes, &treq, MOSAIC_SCENES_AHEAD).await?;
                let values: Vec<f32> = raw.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect();
                if values.len() == dim * dim {
                    paint.rgba(&Selected { bands: 1, values }, dim * dim)
                } else {
                    vec![0u8; dim * dim * 4]
                }
            } else if let Some(method) = selection {
                // Every scene's values, then rio-tiler's arithmetic over them, then colour.
                let paint = Paint::of(&treq.bands).ok_or_else(|| {
                    EngineError::BadRequest("pixel_selection other than 'first' cannot be used with a terrain encoding".into())
                })?;
                // An array of what the method leaves, when it is of the raster's own bands
                // and nothing makes a picture's bytes of them.
                let array = matches!(treq.format, OutputFormat::Npy)
                    && own_bands
                    && !pq.colormap_explicit
                    && formula.is_none()
                    && pq.expr_raw.is_none()
                    && pq.algorithm.is_none()
                    && full.extra_samples.is_empty()
                    && !super::has_embedded_palette(&pq, &first, &treq.bands);
                if raw_array && !array {
                    return Err(EngineError::BadRequest(
                        "raw array formats are not supported with pixel_selection other than 'first' yet".into(),
                    ));
                }
                // `mean` and `median` are cast back to the source's type, which only loses
                // anything when the values are the raster's own whole numbers.
                let integer = whole_numbers(pq.unscale, &treq.bands, full.sample_type);
                let bands = TileEngine::value_bands(&treq.bands);
                // The median has to hold every scene, so it is refused before anything is
                // read when that would be too much.
                if Fold::<f32>::held_bytes(method, scenes.len(), bands, dim * dim) > MAX_HELD_BYTES {
                    return Err(EngineError::BadRequest(format!(
                        "{} items are too many for a median at this tile size; narrow the search or lower items_limit",
                        scenes.len()
                    )));
                }
                let mut fold = Fold::new(method, bands, dim * dim, integer);
                // For an array each scene's values are brought to whole numbers first, where
                // the raster's are: rio-tiler's scenes are in the raster's type before the
                // method sees them, and a mean cut off after is otherwise half a level low.
                if array && integer {
                    let pixels = bands * dim * dim * 4;
                    engine
                        .mosaic_scene_values(&scenes, &treq, |scene| {
                            if scene.len() == pixels {
                                fold.add(scene.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]).round()));
                            }
                        })
                        .await?;
                } else {
                    engine.mosaic_scene_values(&scenes, &treq, |scene| fold.add_le_bytes(scene)).await?;
                }
                let selected = fold.finish();
                if array {
                    // A count is of bytes and a deviation of doubles; the rest are in the
                    // type that holds every asset read, rounded where that is of whole numbers.
                    let mut kind = full.sample_type;
                    for uri in scenes[0].iter().skip(1) {
                        kind = wider_type(kind, engine.store.open_uri(uri).await?.index.full_res().sample_type);
                    }
                    let kind = match method {
                        Selection::Count => SampleType::Uint8,
                        Selection::Stdev => SampleType::Float64,
                        _ if pq.unscale => SampleType::Float32,
                        _ => kind,
                    };
                    let whole = !matches!(kind, SampleType::Float32 | SampleType::Float64);
                    let values: Vec<f32> = match selected {
                        Some(selected) => selected.values.iter().map(|v| if whole { v.round() } else { *v }).collect(),
                        // No scene reached the tile: nothing, in the method's own shape.
                        None => vec![f32::NAN; if method == Selection::Count { 1 } else { bands } * dim * dim],
                    };
                    let shown = values.len() / (dim * dim);
                    // With `rescale` the array is of the bytes a picture would show, in the
                    // ranges the picture's bands were given.
                    if !pq.rescale.is_empty() {
                        let range = |band: usize| match &treq.bands {
                            BandRequest::Rgb { rescale_min, rescale_max, .. } => (rescale_min[band.min(2)], rescale_max[band.min(2)]),
                            BandRequest::Single { rescale, .. } => *rescale,
                            _ => pq.rescale[0],
                        };
                        let bytes: Vec<f32> = values
                            .iter()
                            .enumerate()
                            .map(|(i, v)| {
                                let (min, max) = range(i / (dim * dim));
                                if v.is_nan() { *v } else { f32::from(rescale_byte(*v, min, max)) }
                            })
                            .collect();
                        return Ok((npy_array(&bytes, shown, dim, dim, SampleType::Uint8, return_mask, None), content_type));
                    }
                    return Ok((npy_array(&values, shown, dim, dim, kind, return_mask, None), content_type));
                }
                match selected {
                    Some(selected) => paint.rgba(&selected, dim * dim),
                    None => vec![0u8; dim * dim * 4],
                }
            } else {
                // Nothing to do to the pixels after the mosaic: the renderer encodes them itself.
                if !auto_format && (raw_array || (formula.is_none() && return_mask)) {
                    let made = engine.render_mosaic_scenes_ahead(&scenes, &treq, MOSAIC_SCENES_AHEAD).await?;
                    // The renderer's bare pixels or values, written as the GeoTIFF asked for:
                    // floats when the tile is of numbers, not of a picture.
                    let made = super::wrap_geotiff_fmt(made, content_type, treq.format, dim as u32, dim as u32, lies, 3857, return_mask);
                    return Ok((made, content_type));
                }
                treq.format = OutputFormat::RawRgba;
                engine.render_mosaic_scenes_ahead(&scenes, &treq, MOSAIC_SCENES_AHEAD).await?.to_vec()
            };
            let asked = (!auto_format).then_some((asked, content_type));
            // A count is one band whatever was counted.
            // So is one band drawn in grey, whichever scene it came from: a raster with a
            // palette of its own is one band too, and is not grey.
            let one_grey = matches!(Paint::of(&treq.bands), Some(Paint::Gray { colormap: ColormapName::Grayscale, .. }))
                && !super::has_embedded_palette(&pq, &first, &treq.bands)
                && rgba.chunks_exact(4).all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
            let grey = (one_grey || selection.is_some_and(|method| method == Selection::Count || TileEngine::value_bands(&treq.bands) == 1))
                && !pq.colormap_explicit;
            let formula = match &map_after {
                Some(map_after) => {
                    map_after.colour(&mut rgba);
                    None
                }
                None => formula.as_ref(),
            };
            encode_picture(rgba, dim, dim, formula, asked, jpeg_quality, return_mask, grey)
                .map(|made| as_geotiff(made, dim, dim, lies, 3857, return_mask))
        })
        .await;
    match rendered {
        Ok(made) => {
            let (west, south, east, north) = tile_bounds_3857(tile);
            placed(to_http_response_cached(engine, Ok(made), ctx.if_none_match), [west, south, east, north], 3857)
        }
        Err(e) => engine_error(&e),
    }
}

/// Whether samples of this kind can hold `named`. GDAL refuses a no-data value they
/// cannot; a fraction among whole numbers it takes, and no sample is ever equal to it.
fn nodata_fits(named: f32, kind: cogzilla_tiff::SampleType) -> bool {
    use cogzilla_tiff::SampleType as T;
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

/// A picture asked for as a GeoTIFF, which until here is its bare RGBA pixels, written as
/// one: `bounds` and `epsg` say where it lies. Any other format is left as it is.
fn as_geotiff(
    (body, content_type): (Bytes, &'static str),
    width: usize,
    height: usize,
    bounds: [f64; 4],
    epsg: u16,
    return_mask: bool,
) -> (Bytes, &'static str) {
    if content_type != super::GEOTIFF_CONTENT_TYPE || width == 0 || body.len() != width * height * 4 {
        return (body, content_type);
    }
    (crate::titiler::encode_geotiff_rgba(width as u32, height as u32, &body, bounds, epsg, return_mask), content_type)
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
            match proj_crs_string(asset).or_else(|| item_crs.clone()) {
                Some(crs) => crs_strings.push(crs),
                None => return None,
            }
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
    let pixel_to_crs = cogzilla_tiff::Affine2D {
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
    } else if crs.contains("EPSG/0/3857") {
        3857
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
/// say it: the tile's bounds in Web Mercator, or the box in the system it was drawn in.
/// Only degrees and Web Mercator are worked out here; a picture in any other system, a
/// preview and a posted shape go without the headers.
pub(super) fn item_placed(resp: Resp, sub: &[&str], raw_query: &str) -> Resp {
    if resp.status() != StatusCode::OK {
        return resp;
    }
    // A parameter as the handlers that drew the picture read it: by its key as written,
    // not percent-decoded, and the last one given.
    let asked = |key: &str| {
        raw_query
            .split('&')
            .rev()
            .filter_map(|pair| pair.split_once('='))
            .find(|(k, _)| *k == key)
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
            if asked("buffer").is_some_and(|b| b.parse::<f64>().map_or(true, |b| b != 0.0)) {
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
            // 4326 or 3857, or nothing for a system not placed here.
            let system = |key: &str, default: u16| match asked(key).as_deref().map(super::parse_coord_crs_with_epsg) {
                None => Some(default),
                Some(Ok((_, epsg @ (4326 | 3857)))) => Some(epsg),
                Some(_) => None,
            };
            let Some(given) = system("coord_crs", 4326) else {
                return resp;
            };
            let Some(drawn) = system("dst_crs", given) else {
                return resp;
            };
            let corners = |to: fn(f64, f64) -> (f64, f64)| {
                let ((w, s), (e, n)) = (to(bbox[0], bbox[1]), to(bbox[2], bbox[3]));
                [w, s, e, n]
            };
            let bounds = match (given, drawn) {
                (4326, 3857) => corners(cogzilla_proj::mercator::lonlat_deg_to_mercator),
                (3857, 4326) => corners(mercator_to_lonlat_deg),
                _ => bbox,
            };
            if bounds.iter().any(|c| !c.is_finite()) {
                return resp;
            }
            let spelled = asked("dst_crs").or_else(|| asked("coord_crs"));
            placed_as(resp, bounds, drawn, spelled.as_deref())
        }
        _ => resp,
    }
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
            // The `/stac` handler returns the bands as they are; band math is applied here.
            let pairs = query_pairs(raw_query);
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

/// Resolve the item behind `/collections/{collection}/items/{item_id}/...` and answer what
/// is specific to titiler-pgstac. Everything else goes to the `/stac` handlers.
pub(super) async fn item_route(
    ctx: &Ctx<'_>,
    method: &Method,
    collection: &str,
    item_id: &str,
    rest: &[&str],
    pq: &ParsedQuery,
) -> ItemRoute {
    use ItemRoute::Done;
    let Some(pg) = ctx.engine.pgstac() else {
        return Done(detail(
            StatusCode::SERVICE_UNAVAILABLE,
            "pgSTAC is not configured: set DATABASE_URL, or PGHOST and the other PG* variables",
        ));
    };
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
    let pairs = query_pairs(ctx.raw_query);

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
            if query_value(&pairs, "f") == Some("html") {
                return Done(detail(
                    StatusCode::BAD_REQUEST,
                    "f=html is not supported on tileset routes (use f=json)",
                ));
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
        && let (Ok(z), Ok(x), Some((y, _, _))) =
            (zs.parse::<u8>(), xs.parse::<u32>(), parse_y_scale_ext(ys))
        && z <= 30
        && u64::from(x) < (1u64 << z)
        && u64::from(y) < (1u64 << z)
    {
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(TileCoord { z, x, y });
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
    // is drawn by the search routes, from this item alone.
    if get && item_picture_by_asset(rest, &pairs) {
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
        let (own, pairs) = (super::parse_query((!query.is_empty()).then_some(query.as_str())), query_pairs(&query));
        return Done(mosaic_route(&ctx, pg, &base, &rec, rest, None, own, &pairs).await);
    }
    ItemRoute::Stac(item, extent)
}

/// A query string without the parameters only a search's routes read: how its items are
/// chosen among and how many are looked for.
fn without_search_options(raw_query: &str) -> String {
    let algorithm = raw_query.split('&').any(|part| part.starts_with("algorithm="));
    let kept: Vec<&str> = raw_query
        .split('&')
        .filter(|part| {
            let name = part.split('=').next().unwrap_or(part);
            !matches!(name, "pixel_selection" | "items_limit" | "scan_limit" | "time_limit" | "exitwhenfull" | "skipcovered")
                && (algorithm || name != "algorithm_params")
        })
        .collect();
    kept.join("&")
}

/// Whether an item request is for a tile or a box whose assets carry band math, or bands
/// of their own on more than one of them. A box is one with a format in its path, as
/// titiler-pgstac's item routes have it.
fn item_picture_by_asset(rest: &[&str], pairs: &[(String, String)]) -> bool {
    let picture = match rest {
        ["tiles", "WebMercatorQuad", _, _, _] => true,
        ["bbox", seg] => matches!(super::parse_bbox4(seg), Some((_, Some(_)))),
        ["bbox", _, dims] => matches!(super::parse_wxh_ext(dims), Ok((_, _, Some(_)))),
        _ => false,
    };
    if !picture {
        return false;
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
    // The routes that take a body.
    let takes_body = |rest: &[&str]| {
        rest == ["statistics"]
            || rest.first().is_some_and(|f| *f == "feature" || f.starts_with("feature."))
    };
    if body.is_some() != takes_body(rest) {
        return if body.is_some() {
            not_found()
        } else {
            detail(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed")
        };
    }
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
                .is_some_and(|f| !matches!(f, "png" | "npy" | "tif" | "jpeg" | "jpg" | "jp2" | "webp" | "pngraw")) =>
        {
            invalid_param("query", "tile_format", "Input should be an image format")
        }
        // A map with nothing to show is refused as its tiles would be.
        [_, "map.html"] if query_value(pairs, "assets").is_none() => missing_query_param("assets"),
        [_, "map.html"] if ["tilesize", "minzoom", "maxzoom"].iter().any(|k| query_value(pairs, k).is_some_and(|v| v.parse::<i64>().is_err())) => {
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
        ["tiles"] | ["tiles", _] if query_value(pairs, "f") == Some("html") => detail(
            StatusCode::BAD_REQUEST,
            "f=html is not supported on tileset routes (use f=json)",
        ),
        ["tiles"] => tilesets(ctx, base, &rec.metadata, pairs),
        ["tiles", tms] => tileset(ctx, base, &rec.metadata, tms, pairs),
        ["tiles", _, zs, xs, ys, "assets"] => {
            let coord = match ys.parse::<u32>() {
                Ok(y) => parse_zxy(zs, xs, y),
                Err(_) => Err(detail(StatusCode::BAD_REQUEST, "invalid z/x/y tile coordinate")),
            };
            match coord {
                Err(r) => r,
                Ok(None) => json_resp(StatusCode::OK, &json!([])),
                Ok(Some(t)) => tile_assets(pg, &rec.hash, t, pairs).await,
            }
        }
        ["tiles", _, zs, xs, ys] => {
            let Some((y, scale, ext)) = parse_y_scale_ext(ys) else {
                return detail(StatusCode::BAD_REQUEST, "invalid z/x/y tile coordinate");
            };
            // Without an extension the format is chosen from the rendered pixels.
            let ext = if ys.contains('.') { ext.unwrap_or("") } else { "" };
            match parse_zxy(zs, xs, y) {
                Err(r) => r,
                Ok(None) if pq.assets.is_empty() => missing_query_param("assets"),
                Ok(None) => no_content(),
                Ok(Some(coord)) => tile(ctx, pg, &rec.hash, coord, scale, ext, pq, pairs).await,
            }
        }
        [first, ..] if *first == "feature" || first.starts_with("feature.") => {
            feature_image(ctx, pg, &rec.hash, rest, body, pq, pairs).await
        }
        ["bbox", seg] => {
            // The format is part of the path here: `{minx},{miny},{maxx},{maxy}.{format}`.
            let Some((bbox, ext)) = parse_bbox4(seg) else {
                return not_found();
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
            let polygon = json!({
                "type": "Polygon",
                "coordinates": [[[w, s], [w, n], [e, n], [e, s], [w, s]]]
            });
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
            let (Some((bbox, None)), Some((w, h, ext))) = (parse_bbox4(seg), parse_wxh_ext(dims).ok())
            else {
                return not_found();
            };
            if w == 0 || h == 0 {
                return not_found();
            }
            let polygon = bbox_polygon(bbox);
            part_image(ctx, pg, &rec.hash, &polygon, bbox, None, PartSize::Exact(w, h), ext, pq, pairs)
                .await
        }
        ["statistics"] => mosaic_statistics(ctx, pg, &rec.hash, body, pq, pairs).await,
        ["point", lonlat] => point_values(ctx, pg, &rec.hash, lonlat, &pq, pairs).await,
        ["point", lonlat, "assets"] => {
            let coords = lonlat
                .split_once(',')
                .and_then(|(x, y)| Some((x.parse::<f64>().ok()?, y.parse::<f64>().ok()?)))
                .filter(|(x, y)| x.is_finite() && y.is_finite());
            let Some((lon, lat)) = coords else {
                return not_found();
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
        assert_eq!(status("tile_format=tif"), StatusCode::BAD_REQUEST);
        assert_eq!(status("use_epsg=maybe"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status("zooms=WebMercatorQuad"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status(""), StatusCode::OK);
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

        let terrain = BandRequest::Terrain { band_1based: 1, encoding: cogzilla_kernel::TerrainEncoding::Terrarium };
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
        // A range of nothing, or one that runs backwards, is not a panic.
        assert_eq!(rescale_byte(3.0, 3.0, 3.0), 0);
        let _ = rescale_byte(3.0, 10.0, 0.0);
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
        use cogzilla_tiff::Affine2D;

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
    fn geotiff_pictures_are_geotiffs() {
        // Bare RGBA pixels under the GeoTIFF content type become a TIFF; anything else,
        // and a body that is not the pixels it should be, is left alone.
        let pixels = Bytes::from(vec![7u8; 4 * 3 * 4]);
        let (tiff, kind) = as_geotiff((pixels.clone(), super::super::GEOTIFF_CONTENT_TYPE), 4, 3, [8.0, 45.0, 9.0, 46.0], 4326, true);
        assert_eq!(kind, super::super::GEOTIFF_CONTENT_TYPE);
        assert_eq!(&tiff[..4], b"II*\0");
        assert!(tiff.len() > pixels.len());
        let (png, _) = as_geotiff((pixels.clone(), "image/png"), 4, 3, [8.0, 45.0, 9.0, 46.0], 4326, true);
        assert_eq!(png, pixels);
        let (short, _) = as_geotiff((pixels.clone(), super::super::GEOTIFF_CONTENT_TYPE), 4, 4, [8.0, 45.0, 9.0, 46.0], 4326, true);
        assert_eq!(short, pixels);
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
        assert_eq!(parse_zxy("x", "0", 0).unwrap_err().status(), StatusCode::BAD_REQUEST);
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
        use cogzilla_tiff::SampleType as T;
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
        // Systems not placed here, tiles that are not tiles, and other routes go without.
        for (sub, query) in [
            (&["bbox", "6.3,43.6,6.5,43.8.png"][..], "dst_crs=epsg:32631"),
            (&["bbox", "6.3,43.6,6.5,43.8.png"][..], "coord_crs=epsg:32631"),
            (&["tiles", "WebMercatorQuad", "1", "2", "0.png"][..], ""),
            (&["tiles", "WGS1984Quad", "0", "0", "0.png"][..], ""),
            (&["preview.png"][..], ""),
            (&["info"][..], ""),
        ] {
            assert_eq!(said(sub, query), None, "{sub:?} {query}");
        }
        // A key the handlers do not read as that parameter is not read as it here either.
        assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst%5Fcrs=EPSG:3857").unwrap().1, crs(4326));
        assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst_crs=epsg:3857&dst_crs=epsg:4326").unwrap().1, crs(4326));
        // Degrees asked for as CRS84 are named so; a buffered tile is not placed.
        let ogc = "<http://www.opengis.net/def/crs/OGC/0/CRS84>";
        assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "coord_crs=OGC:CRS84").unwrap().1, ogc);
        assert_eq!(said(&["bbox", "701000,5404000,720000,5420000.png"], "coord_crs=epsg:3857&dst_crs=OGC%3ACRS84").unwrap().1, ogc);
        assert_eq!(said(&["tiles", "WebMercatorQuad", "0", "0", "0.png"], "buffer=10"), None);
        assert!(said(&["tiles", "WebMercatorQuad", "0", "0", "0.png"], "buffer=0").is_some());
        let missing = Response::builder().status(StatusCode::NOT_FOUND).body(Full::new(Bytes::new())).unwrap();
        assert!(item_placed(missing, &["bbox", "6.3,43.6,6.5,43.8.png"], "").headers().get("content-bbox").is_none());
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
        let doubles = npy_array(&[f32::NAN, 0.5], 1, 2, 1, SampleType::Float64, true, None);
        assert_eq!(&doubles[doubles.len() - 16..doubles.len() - 8], 0f64.to_le_bytes());
        assert_eq!(&doubles[doubles.len() - 8..], f64::MAX.to_le_bytes());
    }

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
    }

}
