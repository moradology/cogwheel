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
use cogzilla_kernel::{OutputFormat, ResampleMode};
use cogzilla_pgstac::{
    MosaicItem, PgStac, PgStacError, SearchLimits, SearchOrder, SearchRecord, jsonb_key_order,
};
use cogzilla_proj::mercator::lonlat_deg_to_mercator;
use cogzilla_proj::{CrsModel, TileCoord, mercator_to_lonlat_deg, tile_bounds_3857};
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CONTENT_TYPE, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{Value, json};

use super::{
    ParsedQuery, apply_unscale_cogs, build_band_request, build_tile_request, parse_bbox4,
    parse_ext_format, parse_wxh_ext, parse_y_scale_ext, percent_decode, read_limited_body,
    to_http_response_cached,
};
use crate::engine::{EngineError, TileEngine};
use crate::stac::{StacItem, rewrite_stac_expr, stac_item_from_json};
use crate::titiler::{
    PolygonMask, build_map_html, encode_rgba_output, query_point, render_wgs84_bbox,
};

/// Most scenes one tile may read. titiler-pgstac's default `items_limit` is 100.
const MAX_SCENES_PER_TILE: usize = 256;
/// The engine composites one to three co-registered assets per scene.
const MAX_ASSETS_PER_SCENE: usize = 3;

/// The parts of the request the routes need.
pub(super) struct Ctx<'a> {
    pub engine: &'a Arc<TileEngine>,
    /// `scheme://host[/prefix]`, for links in responses.
    pub host: &'a str,
    pub forwarded_prefix: &'a str,
    pub path: &'a str,
    pub raw_query: &'a str,
    pub if_none_match: Option<&'a str>,
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

/// `scan_limit`, `items_limit`, `time_limit`, `exitwhenfull` and `skipcovered`.
fn parse_limits(pairs: &[(String, String)]) -> Result<SearchLimits, Resp> {
    let mut limits = SearchLimits::default();
    let int = |name: &str, slot: &mut i32| -> Result<(), Resp> {
        if let Some(v) = query_value(pairs, name) {
            *slot = v
                .parse::<i32>()
                .ok()
                .filter(|n| *n >= 0)
                .ok_or_else(|| invalid_param("query", name, "Input should be a valid integer"))?;
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
    let order = match query_value(pairs, "sortby") {
        None | Some("") => None,
        Some(spec) => {
            let (descending, field) = match spec.as_bytes()[0] {
                b'-' => (true, &spec[1..]),
                b'+' => (false, &spec[1..]),
                _ => (false, spec),
            };
            match field {
                "lastused" => Some(SearchOrder::LastUsed { descending }),
                "usecount" => Some(SearchOrder::UseCount { descending }),
                _ => {
                    return invalid_param("query", "sortby", "sortby must be lastused or usecount");
                }
            }
        }
    };
    let (searches, matched) = match pg.list_searches(limit, offset, order).await {
        Ok(page) => page,
        Err(e) => return db_error(&e),
    };
    let page_url = |off: i64| format!("{}/searches/list?limit={limit}&offset={off}", ctx.host);
    let mut links = vec![json!({ "href": page_url(offset), "rel": "self" })];
    if offset + limit < matched {
        links.push(json!({ "href": page_url(offset + limit), "rel": "next" }));
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
const OPENAPI_ROUTES: [(&str, &str, &str); 45] = [
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
        paths.insert(path.to_string(), json!({ method: { "summary": summary } }));
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
    if let Some(v) = query_value(pairs, "filter") {
        // titiler-pgstac defaults to CQL2 text here and converts it; only JSON is taken yet.
        if query_value(pairs, "filter-lang") != Some("cql2-json") {
            return Err(detail(
                StatusCode::BAD_REQUEST,
                "cql2-text filters are not supported yet: pass the filter as CQL2 JSON with filter-lang=cql2-json",
            ));
        }
        let parsed: Value = serde_json::from_str(v)
            .map_err(|_| invalid_param("query", "filter", "Input should be valid JSON"))?;
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

/// `GET {base}/bbox/{minx},{miny},{maxx},{maxy}/assets` and `GET {base}/point/{lon},{lat}/assets`.
async fn geometry_assets(
    pg: &PgStac,
    hash: &str,
    geometry: Value,
    limits: SearchLimits,
) -> Resp {
    match pg.items_for_geometry(&geometry, hash, limits).await {
        Ok(items) => json_resp(
            StatusCode::OK,
            &Value::Array(items.iter().map(MosaicItem::to_json).collect()),
        ),
        Err(e) => db_error(&e),
    }
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
struct MosaicQuery {
    limits: SearchLimits,
    /// Method for a source already in the tile's projection (`resampling`).
    resampling: ResampleMode,
    /// Method for a source that has to be reprojected (`reproject`).
    reproject: ResampleMode,
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
    // `reproject` is handled here; the shared parser does not know it.
    if let Some(e) = pq.errors.iter().find(|e| !e.contains("'reproject'")) {
        return Err(detail(StatusCode::BAD_REQUEST, e.clone()));
    }
    if pq.assets.iter().any(|a| a.contains('|')) {
        return Err(detail(
            StatusCode::BAD_REQUEST,
            "per-asset options (assets=name|bidx=...|expression=...) are not supported yet",
        ));
    }
    if pq.assets.len() > MAX_ASSETS_PER_SCENE {
        return Err(detail(
            StatusCode::BAD_REQUEST,
            format!("at most {MAX_ASSETS_PER_SCENE} assets per item are supported, got {}", pq.assets.len()),
        ));
    }
    match query_value(pairs, "pixel_selection") {
        None | Some("first") => {}
        Some(other) => {
            return Err(detail(
                StatusCode::BAD_REQUEST,
                format!("pixel_selection '{other}' is not supported yet (use first)"),
            ));
        }
    }
    for unsupported in ["asset_as_band", "algorithm_params"] {
        if query_value(pairs, unsupported).is_some() {
            return Err(detail(
                StatusCode::BAD_REQUEST,
                format!("parameter '{unsupported}' is not supported yet"),
            ));
        }
    }
    // The mosaic engine renders whole tiles at the two native sizes without these.
    if pq.buffer > 0 {
        return Err(detail(StatusCode::BAD_REQUEST, "parameter 'buffer' is not supported on mosaic routes yet"));
    }
    if pq.nodata.is_some() {
        return Err(detail(StatusCode::BAD_REQUEST, "parameter 'nodata' is not supported on mosaic routes yet"));
    }
    if pq.custom_colormap_json.is_some() {
        return Err(detail(StatusCode::BAD_REQUEST, "a JSON 'colormap' is not supported on mosaic routes yet"));
    }
    if pq.unscale && pq.assets.len() > 1 {
        return Err(detail(
            StatusCode::BAD_REQUEST,
            "parameter 'unscale' is not supported with more than one asset yet",
        ));
    }
    Ok(MosaicQuery {
        limits: parse_limits(pairs)?,
        resampling: parse_resample(pairs, "resampling")?,
        reproject: parse_resample(pairs, "reproject")?,
    })
}

async fn tile_items(
    pg: &PgStac,
    id: &str,
    tile: TileCoord,
    limits: SearchLimits,
) -> Result<Arc<Vec<MosaicItem>>, Resp> {
    pg.items_for_geometry(&tile_polygon(tile), id, limits)
        .await
        .map_err(|e| db_error(&e))
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
    match tile_items(pg, id, tile, limits).await {
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
    mut pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    // `width` and `height` are read here; the shared parser does not know them.
    pq.errors.retain(|e| !e.contains("'width'") && !e.contains("'height'"));
    let query = match check_mosaic_query(&pq, pairs) {
        Ok(q) => q,
        Err(r) => return r,
    };
    if let Err(r) = check_coord_crs(pairs) {
        return r;
    }
    if let Some(dst) = query_value(pairs, "dst_crs")
        && !matches!(dst.to_ascii_lowercase().as_str(), "epsg:4326" | "ogc:crs84")
    {
        return detail(
            StatusCode::BAD_REQUEST,
            format!("dst_crs '{dst}' is not supported on pgSTAC mosaic routes (use epsg:4326)"),
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
    if matches!(format, Some((OutputFormat::RawF32 | OutputFormat::Npy, _))) {
        return detail(
            StatusCode::BAD_REQUEST,
            "raw array formats are not supported on pgSTAC bbox and feature routes yet",
        );
    }
    // A JPEG has no mask to carry the polygon, and TiTiler then writes the whole box.
    let mask = if matches!(format, Some((OutputFormat::Jpeg { .. }, _))) { None } else { mask };

    let items = match pg.items_for_geometry(geometry, hash, query.limits).await {
        Ok(items) => items,
        Err(e) => return db_error(&e),
    };
    if items.is_empty() {
        return no_content();
    }
    let mut scenes: Vec<Vec<String>> = Vec::with_capacity(items.len().min(MAX_SCENES_PER_TILE));
    for item in items.iter().take(MAX_SCENES_PER_TILE) {
        let mut hrefs = Vec::with_capacity(pq.assets.len());
        for asset in &pq.assets {
            match item.asset_href(asset) {
                Some(href) => hrefs.push(href.to_string()),
                None => return unknown_asset(asset, item),
            }
        }
        scenes.push(hrefs);
    }
    if let Some(expr) = pq.expr_raw.take() {
        pq.expr_raw = Some(rewrite_stac_expr(&expr, &pq.assets));
    }

    let engine = ctx.engine;
    let rendered: Result<(Bytes, &'static str), EngineError> = async {
        let formula = pq
            .color_formula
            .as_deref()
            .map(cogzilla_kernel::ColorFormula::parse)
            .transpose()
            .map_err(EngineError::BadRequest)?;
        let mut canvas: Vec<u8> = Vec::new();
        let (mut width, mut height) = (0usize, 0usize);
        for (n, scene) in scenes.iter().enumerate() {
            let mut cogs = Vec::with_capacity(scene.len());
            for href in scene {
                cogs.push(engine.store.open_uri(href).await?);
            }
            let crs = engine.get_or_build_crs(&cogs[0])?;
            let full = cogs[0].index.full_res();
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
            let mut bands = build_band_request(&pq, default_bands, full.sample_type)?;
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs);
            }
            // As on tiles: `resampling` for a source already in the output's projection,
            // `reproject` for one that has to be warped into it.
            let resample = match crs {
                CrsModel::Geographic => query.resampling,
                _ => query.reproject,
            };
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
        if let Some(formula) = formula {
            formula.apply_buffer_in_place(&mut canvas, 4);
        }
        let (format, content_type) = match format {
            Some((f, ct)) => (f, if explicit.as_deref() == Some("jpg") { "image/jpg" } else { ct }),
            None => {
                let opaque = canvas.chunks_exact(4).all(|p| p[3] == 255);
                parse_ext_format(if opaque { "jpg" } else { "png" }, None, pq.jpeg_quality)?
            }
        };
        Ok((encode_rgba_output(width, height, &canvas, format, pq.return_mask), content_type))
    }
    .await;
    match rendered {
        Ok((body, content_type)) => {
            let mut resp = Response::new(Full::new(body));
            resp.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
            resp.headers_mut().insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
            resp
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
        ["feature", dims] => match parse_wxh_ext(dims) {
            Some((w, h, ext)) if w > 0 && h > 0 => (Ok(PartSize::Exact(w, h)), ext),
            _ => return detail(StatusCode::NOT_FOUND, "Not Found"),
        },
        _ => return detail(StatusCode::NOT_FOUND, "Not Found"),
    };
    let size = match size {
        Ok(s) => s,
        Err(r) => return r,
    };
    let Some(geojson) = body.and_then(|b| serde_json::from_slice::<Value>(b).ok()) else {
        return invalid_param("body", "body", "Input should be a valid GeoJSON Feature");
    };
    let geometry = match geojson.get("type").and_then(Value::as_str) {
        Some("Feature") => geojson.get("geometry").cloned().unwrap_or(Value::Null),
        _ => geojson.clone(),
    };
    let (mask, bbox) = match PolygonMask::from_geojson(&geojson) {
        Ok(m) => m,
        Err(e) => return engine_error(&e),
    };
    part_image(ctx, pg, hash, &geometry, bbox, Some(&mask), size, ext, pq, pairs).await
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
    if let Some(e) = pq.errors.iter().find(|e| !e.contains("'reproject'")) {
        return detail(StatusCode::BAD_REQUEST, e.clone());
    }
    if pq.assets.iter().any(|a| a.contains('|')) {
        return detail(
            StatusCode::BAD_REQUEST,
            "per-asset options (assets=name|bidx=...|expression=...) are not supported yet",
        );
    }
    if query_value(pairs, "asset_as_band").is_some() {
        return detail(
            StatusCode::BAD_REQUEST,
            "parameter 'asset_as_band' is not supported on pgSTAC point routes yet",
        );
    }
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
    let items = match pg.items_for_geometry(&point, hash, limits).await {
        Ok(items) => items,
        Err(e) => return db_error(&e),
    };
    if items.is_empty() {
        return no_content();
    }

    // Resolve every asset before reading anything, so a bad name fails the request at once.
    let mut jobs: Vec<(String, Vec<(String, String)>)> = Vec::with_capacity(items.len());
    for item in items.iter() {
        let mut hrefs = Vec::with_capacity(pq.assets.len());
        for asset in &pq.assets {
            match item.asset_href(asset) {
                Some(href) => hrefs.push((asset.clone(), href.to_string())),
                None => return unknown_asset(asset, item),
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
            for (asset, href) in &hrefs {
                let cog = engine.store.open_uri(href).await?;
                let pt = match query_point(&engine, &cog, lon, lat, unscale, nodata).await {
                    Ok(pt) => pt,
                    // The item's footprint holds the point but this raster does not.
                    Err(EngineError::NotFound(_)) => return Ok((idx, None)),
                    Err(e) => return Err(e),
                };
                for (i, v) in pt["values"].as_array().into_iter().flatten().enumerate() {
                    values.push(v.clone());
                    descriptions.push(format!("{asset}_b{}", i + 1));
                }
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
    let query = match check_mosaic_query(&pq, pairs) {
        Ok(q) => q,
        Err(r) => return r,
    };
    let items = match tile_items(pg, id, tile, query.limits).await {
        Ok(items) => items,
        Err(r) => return r,
    };
    if items.is_empty() {
        return no_content();
    }

    let mut scenes: Vec<Vec<String>> = Vec::with_capacity(items.len().min(MAX_SCENES_PER_TILE));
    for item in items.iter().take(MAX_SCENES_PER_TILE) {
        let mut hrefs = Vec::with_capacity(pq.assets.len());
        for asset in &pq.assets {
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
    if let Some(expr) = pq.expr_raw.take() {
        pq.expr_raw = Some(rewrite_stac_expr(&expr, &pq.assets));
    }
    let (resampling, reproject) = (query.resampling, query.reproject);
    let return_mask = pq.return_mask;
    let jpeg_quality = pq.jpeg_quality;
    let engine = ctx.engine;
    let cache_key = format!("{}?{}", ctx.path, ctx.raw_query);
    let rendered = engine
        .tile_cache
        .get_or_render(cache_key, || async {
            let first = engine.store.open_uri(&scenes[0][0]).await?;
            let full = first.index.full_res();
            let default_bands = if scenes[0].len() > 1 {
                scenes[0].len() as u16
            } else {
                full.samples_per_pixel
            };
            let is_u8 = full.sample_type == cogzilla_tiff::SampleType::Uint8;
            let (mut treq, content_type) =
                build_tile_request(tile, scale, &explicit, &pq, default_bands, is_u8)?;
            // As in TiTiler, `resampling` applies to a source already in the tile's projection
            // and `reproject` to one that has to be warped into it.
            treq.resample = match engine.get_or_build_crs(&first)? {
                CrsModel::Epsg3857 => resampling,
                _ => reproject,
            };
            if treq.tile_size != 256 && treq.tile_size != 512 {
                return Err(EngineError::BadRequest(format!(
                    "tilesize={} is not supported on mosaic routes (use 256 or 512)",
                    treq.tile_size
                )));
            }
            if !auto_format {
                // TiTiler names the media type after the extension that was asked for.
                let content_type = if explicit == "jpg" { "image/jpg" } else { content_type };
                return Ok((engine.render_mosaic_scenes(&scenes, &treq).await?, content_type));
            }
            treq.format = OutputFormat::RawRgba;
            let rgba = engine.render_mosaic_scenes(&scenes, &treq).await?;
            let dim = treq.tile_size as usize;
            let opaque = rgba.len() == dim * dim * 4 && rgba.chunks_exact(4).all(|p| p[3] == 255);
            let (format, content_type) = if opaque {
                parse_ext_format("jpg", None, jpeg_quality)?
            } else {
                parse_ext_format("png", None, jpeg_quality)?
            };
            Ok((encode_rgba_output(dim, dim, &rgba, format, return_mask), content_type))
        })
        .await;
    match rendered {
        Ok(tile) => to_http_response_cached(engine, Ok(tile), ctx.if_none_match),
        Err(e) => engine_error(&e),
    }
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
    Ok((encode_rgba_output(dim, dim, &rgba, format, pq.return_mask), content_type))
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
            if let Some(expression) = query_value(&pairs, "expression") {
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
    let item = Arc::new(stac_item_from_json(&doc, self_href));
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
    ItemRoute::Stac(item, extent)
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
    // The only route that takes a body.
    let is_feature = |rest: &[&str]| {
        rest.first().is_some_and(|f| *f == "feature" || f.starts_with("feature."))
    };
    if body.is_some() != is_feature(rest) {
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
            let ext = if ys.contains('.') { ext } else { "" };
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
            let limits = match (check_coord_crs(pairs), parse_limits(pairs)) {
                (Err(r), _) | (_, Err(r)) => return r,
                (Ok(()), Ok(l)) => l,
            };
            let polygon = json!({
                "type": "Polygon",
                "coordinates": [[[w, s], [w, n], [e, n], [e, s], [w, s]]]
            });
            geometry_assets(pg, &rec.hash, polygon, limits).await
        }
        ["bbox", seg, dims] => {
            let (Some((bbox, None)), Some((w, h, ext))) = (parse_bbox4(seg), parse_wxh_ext(dims))
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
        ["point", lonlat] => point_values(ctx, pg, &rec.hash, lonlat, &pq, pairs).await,
        ["point", lonlat, "assets"] => {
            let coords = lonlat
                .split_once(',')
                .and_then(|(x, y)| Some((x.parse::<f64>().ok()?, y.parse::<f64>().ok()?)))
                .filter(|(x, y)| x.is_finite() && y.is_finite());
            let Some((lon, lat)) = coords else {
                return not_found();
            };
            let mut limits = match (check_coord_crs(pairs), parse_limits(pairs)) {
                (Err(r), _) | (_, Err(r)) => return r,
                (Ok(()), Ok(l)) => l,
            };
            // A point is never "covered", so every item under it is wanted.
            limits.exitwhenfull = false;
            limits.skipcovered = false;
            let point = json!({ "type": "Point", "coordinates": [lon, lat] });
            geometry_assets(pg, &rec.hash, point, limits).await
        }
        _ => not_found(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(status("filter=eo:cloud_cover%3C20"), StatusCode::BAD_REQUEST);
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
}
