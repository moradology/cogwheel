//! titiler-pgstac's `/searches` and `/collections/{id}` mosaic routes, on the compatibility path.
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
use cogzilla_proj::{CrsModel, TileCoord, mercator_to_lonlat_deg, tile_bounds_3857};
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CONTENT_TYPE, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{Value, json};

use super::{
    ParsedQuery, build_tile_request, parse_bbox4, parse_ext_format, parse_y_scale_ext,
    percent_decode, read_limited_body, to_http_response_cached,
};
use crate::engine::{EngineError, TileEngine};
use crate::stac::rewrite_stac_expr;
use crate::titiler::{build_map_html, encode_rgba_output};

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

fn engine_error(err: &EngineError) -> Resp {
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

/// The pgSTAC routes, as entries for the OpenAPI document's `paths`.
pub(super) fn openapi_paths() -> Value {
    json!({
        "/searches/register": { "post": { "summary": "Register a pgSTAC search" } },
        "/searches/": { "get": { "summary": "List registered pgSTAC searches" } },
        "/searches/{search_id}/info": { "get": { "summary": "pgSTAC search metadata" } },
        "/searches/{search_id}/{tileMatrixSetId}/tilejson.json": { "get": { "summary": "pgSTAC search TileJSON 3.0" } },
        "/searches/{search_id}/{tileMatrixSetId}/map.html": { "get": { "summary": "pgSTAC search map viewer" } },
        "/searches/{search_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}": { "get": { "summary": "Render pgSTAC search mosaic tile" } },
        "/searches/{search_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}/assets": { "get": { "summary": "Items covering a tile" } },
        "/searches/{search_id}/tiles": { "get": { "summary": "pgSTAC search tilesets" } },
        "/searches/{search_id}/tiles/{tileMatrixSetId}": { "get": { "summary": "pgSTAC search tileset description" } },
        "/searches/{search_id}/bbox/{minx},{miny},{maxx},{maxy}/assets": { "get": { "summary": "Items intersecting a bounding box" } },
        "/searches/{search_id}/point/{lon},{lat}/assets": { "get": { "summary": "Items covering a point" } },
        "/collections/{collection_id}/info": { "get": { "summary": "pgSTAC collection mosaic metadata" } },
        "/collections/{collection_id}/{tileMatrixSetId}/tilejson.json": { "get": { "summary": "pgSTAC collection TileJSON 3.0" } },
        "/collections/{collection_id}/{tileMatrixSetId}/map.html": { "get": { "summary": "pgSTAC collection map viewer" } },
        "/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}": { "get": { "summary": "Render pgSTAC collection mosaic tile" } },
        "/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}/assets": { "get": { "summary": "Items covering a tile" } },
        "/collections/{collection_id}/tiles": { "get": { "summary": "pgSTAC collection tilesets" } },
        "/collections/{collection_id}/tiles/{tileMatrixSetId}": { "get": { "summary": "pgSTAC collection tileset description" } },
        "/collections/{collection_id}/bbox/{minx},{miny},{maxx},{maxy}/assets": { "get": { "summary": "Items intersecting a bounding box" } },
        "/collections/{collection_id}/point/{lon},{lat}/assets": { "get": { "summary": "Items covering a point" } }
    })
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
    rec: &SearchRecord,
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
    let meta_i64 = |k: &str| rec.metadata.get(k).and_then(Value::as_i64);
    let minzoom = zoom("minzoom")?.or_else(|| meta_i64("minzoom")).unwrap_or(0);
    let maxzoom = zoom("maxzoom")?.or_else(|| meta_i64("maxzoom")).unwrap_or(24);
    let bounds: [f64; 4] = rec
        .metadata
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
fn tileset(ctx: &Ctx<'_>, base: &str, rec: &SearchRecord, tms: &str, pairs: &[(String, String)]) -> Resp {
    let (minzoom, maxzoom, bounds) = match mosaic_extent(rec, pairs) {
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
fn tilesets(ctx: &Ctx<'_>, base: &str, rec: &SearchRecord, pairs: &[(String, String)]) -> Resp {
    let (_, _, bounds) = match mosaic_extent(rec, pairs) {
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
    let (minzoom, maxzoom, bounds) = match mosaic_extent(rec, pairs) {
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
                None => {
                    let known: Vec<String> =
                        item.assets.keys().map(|k| format!("'{k}'")).collect();
                    return detail(
                        StatusCode::NOT_FOUND,
                        format!("'{asset}' is not valid, should be one of [{}]", known.join(", ")),
                    );
                }
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
            let (mut treq, content_type) = build_tile_request(
                tile,
                scale,
                &explicit,
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
        // The item routes are not served yet.
        ["collections", _, "items", ..] => not_found(),
        [family @ ("searches" | "collections"), id, rest @ ..] if get && !rest.is_empty() => {
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
            mosaic_route(&ctx, pg, &base, &rec, rest, pq, &pairs).await
        }
        _ => not_found(),
    }
}

/// The routes shared by a registered search and a collection's mosaic.
async fn mosaic_route(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    base: &str,
    rec: &SearchRecord,
    rest: &[&str],
    pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    let not_found = || detail(StatusCode::NOT_FOUND, "Not Found");
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
        ["tiles"] | ["tiles", _] if query_value(pairs, "f") == Some("html") => detail(
            StatusCode::BAD_REQUEST,
            "f=html is not supported on tileset routes (use f=json)",
        ),
        ["tiles"] => tilesets(ctx, base, rec, pairs),
        ["tiles", tms] => tileset(ctx, base, rec, tms, pairs),
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
