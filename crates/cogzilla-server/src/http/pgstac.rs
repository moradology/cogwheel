//! titiler-pgstac's `/searches` routes, on the compatibility path.
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
use cogzilla_pgstac::{MosaicItem, PgStac, PgStacError, SearchLimits, SearchOrder, SearchRecord};
use cogzilla_proj::{CrsModel, TileCoord, mercator_to_lonlat_deg, tile_bounds_3857};
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CONTENT_TYPE, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{Value, json};

use super::{
    ParsedQuery, build_tile_request, parse_ext_format, parse_y_scale_ext, percent_decode,
    read_limited_body, to_http_response_cached,
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

fn register_links(host: &str, id: &str) -> Value {
    json!([
        { "href": format!("{host}/searches/{id}/info"), "rel": "metadata", "title": "Mosaic metadata" },
        { "href": format!("{host}/searches/{id}/{{tileMatrixSetId}}/tilejson.json"), "rel": "tilejson", "templated": true, "title": "Link for TileJSON (Template URL)" },
        { "href": format!("{host}/searches/{id}/{{tileMatrixSetId}}/map.html"), "rel": "map", "templated": true, "title": "Link for Map viewer (Template URL)" },
        { "href": format!("{host}/searches/{id}/WMTSCapabilities.xml"), "rel": "wmts", "title": "WMTS Capabilities link." }
    ])
}

fn info_links(host: &str, id: &str) -> Value {
    json!([
        { "href": format!("{host}/searches/{id}/info"), "rel": "self", "title": "Mosaic metadata" },
        { "href": format!("{host}/searches/{id}/{{tileMatrixSetId}}/tilejson.json"), "rel": "tilejson", "templated": true, "title": "TileJSON link (Template URL)." },
        { "href": format!("{host}/searches/{id}/{{tileMatrixSetId}}/map.html"), "rel": "map", "templated": true, "title": "Map viewer link (Template URL)." },
        { "href": format!("{host}/searches/{id}/WMTSCapabilities.xml"), "rel": "wmts", "title": "WMTS Capabilities link." }
    ])
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
            &json!({ "id": rec.hash, "links": register_links(ctx.host, &rec.hash) }),
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

fn tilejson(ctx: &Ctx<'_>, rec: &SearchRecord, tms: &str, pairs: &[(String, String)]) -> Resp {
    if !pairs.iter().any(|(k, _)| k == "assets") {
        return missing_query_param("assets");
    }
    let zoom = |name: &str| -> Result<Option<i64>, Resp> {
        match query_value(pairs, name) {
            None => Ok(None),
            Some(v) => v
                .parse::<i64>()
                .map(Some)
                .map_err(|_| invalid_param("query", name, "Input should be a valid integer")),
        }
    };
    let (minzoom_q, maxzoom_q) = match (zoom("minzoom"), zoom("maxzoom")) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(r), _) | (_, Err(r)) => return r,
    };
    let meta_i64 = |k: &str| rec.metadata.get(k).and_then(Value::as_i64);
    let minzoom = minzoom_q.or_else(|| meta_i64("minzoom")).unwrap_or(0);
    let maxzoom = maxzoom_q.or_else(|| meta_i64("maxzoom")).unwrap_or(24);
    let bounds: [f64; 4] = rec
        .metadata
        .get("bounds")
        .and_then(Value::as_array)
        .filter(|a| a.len() == 4)
        .and_then(|a| Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?, a[3].as_f64()?]))
        .unwrap_or([-180.0, -85.0511287798066, 180.0, 85.0511287798066]);

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
        "{}/searches/{}/tiles/{tms}/{{z}}/{{x}}/{{y}}{ext}?{}",
        ctx.host,
        rec.hash,
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
        ["searches", id, "info"] if get => match find_search(pg, id).await {
            Ok(rec) => json_resp(
                StatusCode::OK,
                &json!({ "search": search_json(&rec), "links": info_links(ctx.host, &rec.hash) }),
            ),
            Err(r) => r,
        },
        ["searches", id, tms, "tilejson.json"] if get => {
            if let Err(r) = check_tms(tms) {
                return r;
            }
            match find_search(pg, id).await {
                Ok(rec) => tilejson(&ctx, &rec, tms, &pairs),
                Err(r) => r,
            }
        }
        ["searches", id, tms, "map.html"] if get => {
            if let Err(r) = check_tms(tms) {
                return r;
            }
            if let Err(r) = find_search(pg, id).await {
                return r;
            }
            let tilejson_url = format!(
                "{}/searches/{id}/{tms}/tilejson.json?{}",
                ctx.forwarded_prefix, ctx.raw_query
            );
            let mut resp = Response::new(Full::new(Bytes::from(build_map_html(&tilejson_url))));
            resp.headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
            resp
        }
        ["searches", id, "tiles", tms, zs, xs, ys, "assets"] if get => {
            if let Err(r) = check_tms(tms) {
                return r;
            }
            let coord = match ys.parse::<u32>() {
                Ok(y) => parse_zxy(zs, xs, y),
                Err(_) => Err(detail(StatusCode::BAD_REQUEST, "invalid z/x/y tile coordinate")),
            };
            match (find_search(pg, id).await, coord) {
                (Err(r), _) | (_, Err(r)) => r,
                (Ok(_), Ok(None)) => json_resp(StatusCode::OK, &json!([])),
                (Ok(_), Ok(Some(t))) => tile_assets(pg, id, t, &pairs).await,
            }
        }
        ["searches", id, "tiles", tms, zs, xs, ys] if get => {
            if let Err(r) = check_tms(tms) {
                return r;
            }
            let Some((y, scale, ext)) = parse_y_scale_ext(ys) else {
                return detail(StatusCode::BAD_REQUEST, "invalid z/x/y tile coordinate");
            };
            // Without an extension the format is chosen from the rendered pixels.
            let ext = if ys.contains('.') { ext } else { "" };
            if let Err(r) = find_search(pg, id).await {
                return r;
            }
            match parse_zxy(zs, xs, y) {
                Err(r) => r,
                Ok(None) if pq.assets.is_empty() => missing_query_param("assets"),
                Ok(None) => no_content(),
                Ok(Some(coord)) => tile(&ctx, pg, id, coord, scale, ext, pq, &pairs).await,
            }
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
