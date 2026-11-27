//! `/zarr/*` HTTP routes, unified failure-to-HTTP-status table (#388), and `cogzilla-io`-backed
//! Zarr store adapter.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use cogzilla_zarr::hierarchy::BoxFuture;
use cogzilla_zarr::{
    ArrayMeta, CfTimeUnits, DataType, FsStore, OpenMode, ShardingConfig, ZarrError, ZarrHierarchy,
    ZarrStore, ZarrVersion, parse_sel_params, read_coord_1d, resolve_geo, resolve_sel_indices,
    select_variable,
};
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CONTENT_TYPE, HeaderValue};
use hyper::{Method, Response, StatusCode};
use serde_json::{Value, json};

use crate::engine::TileEngine;
use super::html::append_vary_accept;
use super::percent_decode;

/// Unified failure -> HTTP status table for every `/zarr/*` route (#388).
///
/// Rules:
/// - Client fault is 4xx, origin/store fault is 5xx, unsupported codec is 501.
/// - Every error message names the offending thing (keys looked for, available variables,
///   available range, rank, dimension + index, CRS, codec + supported list, chunk key, or cap).
/// - No `/zarr/*` route constructs an error status inline; all errors pass through this mapping.
pub fn zarr_error_status(err: &ZarrError) -> StatusCode {
    match err {
        ZarrError::NotAZarrStore { .. }
        | ZarrError::GroupNotFound { .. }
        | ZarrError::VariableNotFound { .. }
        | ZarrError::AmbiguousVariable { .. }
        | ZarrError::InvalidSel { .. }
        | ZarrError::SelOutOfRange { .. }
        | ZarrError::UnsupportedCalendar { .. }
        | ZarrError::UnsupportedRank(_)
        | ZarrError::UnsupportedDataType(_)
        | ZarrError::UnsupportedVersion(_)
        | ZarrError::IrregularGrid { .. }
        | ZarrError::UnsupportedCrs { .. }
        | ZarrError::InvalidSelection { .. }
        | ZarrError::CapExceeded { .. }
        | ZarrError::ChunkTooLarge { .. }
        | ZarrError::MissingField { .. }
        | ZarrError::BadField { .. }
        | ZarrError::Json(_) => StatusCode::BAD_REQUEST,

        ZarrError::NotFound { .. } => StatusCode::NOT_FOUND,
        ZarrError::UpstreamRefused { .. } => StatusCode::FORBIDDEN,
        ZarrError::UpstreamTimeout { .. } => StatusCode::GATEWAY_TIMEOUT,

        ZarrError::UnsupportedCodec { .. } => StatusCode::NOT_IMPLEMENTED,

        ZarrError::MissingChunk { .. }
        | ZarrError::Codec { .. }
        | ZarrError::UpstreamError { .. }
        | ZarrError::Io { .. } => StatusCode::BAD_GATEWAY,
    }
}

/// Convert any [`ZarrError`] into a JSON HTTP error response using [`zarr_error_status`].
pub fn zarr_error_to_response(err: &ZarrError) -> Response<Full<Bytes>> {
    let status = zarr_error_status(err);
    let body = serde_json::to_vec(&json!({ "detail": err.to_string() })).unwrap_or_default();
    let mut resp = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(Bytes::from(body)))
        .unwrap();
    append_vary_accept(resp.headers_mut());
    resp
}

fn json_ok(val: &Value, content_type: &'static str) -> Response<Full<Bytes>> {
    let body = serde_json::to_vec(val).unwrap_or_default();
    let mut resp = Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static(content_type))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(Bytes::from(body)))
        .unwrap();
    append_vary_accept(resp.headers_mut());
    resp
}

/// Store adapter bridging local filesystem paths (`file://` or bare path) and HTTP/HTTPS prefixes
/// into [`ZarrStore`].
pub enum ServerZarrStore {
    Local(FsStore),
    Http {
        base_url: String,
        client: reqwest::Client,
    },
}

impl ServerZarrStore {
    pub fn from_url(raw_url: &str) -> Result<Self, ZarrError> {
        let trimmed = raw_url.trim();
        if trimmed.is_empty() {
            return Err(ZarrError::InvalidSelection {
                detail: "missing required query parameter 'url'".into(),
            });
        }
        // Honour COGZILLA_ALLOWED_URL_PREFIXES allowlist consistently with /cog/*.
        if cogzilla_io::check_uri_allowed(trimmed).is_err() {
            return Err(ZarrError::UpstreamRefused {
                key: trimmed.to_string(),
                status: 403,
            });
        }
        if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .map_err(|e| ZarrError::UpstreamError {
                    key: trimmed.to_string(),
                    detail: e.to_string(),
                })?;
            return Ok(Self::Http {
                base_url: trimmed.trim_end_matches('/').to_string(),
                client,
            });
        }
        if let Some(scheme_pos) = trimmed.find("://") {
            let scheme = &trimmed[..scheme_pos];
            if scheme != "file" {
                return Err(ZarrError::InvalidSelection {
                    detail: format!("unsupported Zarr URL scheme '{scheme}://'"),
                });
            }
        }
        let local_path = match cogzilla_io::strip_file_uri_prefix(trimmed) {
            Ok(Some(p)) => p,
            Ok(None) => trimmed,
            Err(()) => {
                return Err(ZarrError::InvalidSelection {
                    detail: format!("invalid file:// URI {trimmed:?}"),
                });
            }
        };
        let p = PathBuf::from(local_path);
        if !p.exists() {
            return Err(ZarrError::NotFound {
                key: trimmed.to_string(),
            });
        }
        Ok(Self::Local(FsStore::new(p)))
    }
}

impl ZarrStore for ServerZarrStore {
    fn get<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>> {
        match self {
            Self::Local(fs) => fs.get(key),
            Self::Http { base_url, client } => {
                let url = format!("{base_url}/{}", key.trim_start_matches('/'));
                let client = client.clone();
                let key_owned = key.to_string();
                Box::pin(async move {
                    let resp = client.get(&url).send().await.map_err(|e| {
                        if e.is_timeout() {
                            ZarrError::UpstreamTimeout {
                                key: key_owned.clone(),
                                detail: e.to_string(),
                            }
                        } else {
                            ZarrError::UpstreamError {
                                key: key_owned.clone(),
                                detail: e.to_string(),
                            }
                        }
                    })?;
                    let status = resp.status().as_u16();
                    if status == 404 {
                        return Ok(None);
                    }
                    if status == 401 || status == 403 {
                        return Err(ZarrError::UpstreamRefused {
                            key: key_owned,
                            status,
                        });
                    }
                    if !resp.status().is_success() {
                        return Err(ZarrError::UpstreamError {
                            key: key_owned,
                            detail: format!("HTTP {status}"),
                        });
                    }
                    let bytes = resp.bytes().await.map_err(|e| ZarrError::UpstreamError {
                        key: key_owned,
                        detail: e.to_string(),
                    })?;
                    Ok(Some(bytes.to_vec()))
                })
            }
        }
    }

    fn get_range<'a>(
        &'a self,
        key: &'a str,
        start: u64,
        end: u64,
    ) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>> {
        match self {
            Self::Local(fs) => fs.get_range(key, start, end),
            Self::Http { base_url, client } => {
                let url = format!("{base_url}/{}", key.trim_start_matches('/'));
                let client = client.clone();
                let key_owned = key.to_string();
                Box::pin(async move {
                    if start >= end {
                        return Ok(Some(Vec::new()));
                    }
                    let range_hdr = format!("bytes={start}-{}", end - 1);
                    let resp = client
                        .get(&url)
                        .header(reqwest::header::RANGE, range_hdr)
                        .send()
                        .await
                        .map_err(|e| {
                            if e.is_timeout() {
                                ZarrError::UpstreamTimeout {
                                    key: key_owned.clone(),
                                    detail: e.to_string(),
                                }
                            } else {
                                ZarrError::UpstreamError {
                                    key: key_owned.clone(),
                                    detail: e.to_string(),
                                }
                            }
                        })?;
                    let status = resp.status().as_u16();
                    if status == 404 {
                        return Ok(None);
                    }
                    if status == 401 || status == 403 {
                        return Err(ZarrError::UpstreamRefused {
                            key: key_owned,
                            status,
                        });
                    }
                    if status != 206 && status != 200 {
                        return Err(ZarrError::UpstreamError {
                            key: key_owned,
                            detail: format!("HTTP {status} on ranged read"),
                        });
                    }
                    let bytes = resp.bytes().await.map_err(|e| ZarrError::UpstreamError {
                        key: key_owned.clone(),
                        detail: e.to_string(),
                    })?;
                    if status == 200 {
                        let s = start as usize;
                        let e = end as usize;
                        if e > bytes.len() {
                            return Err(ZarrError::UpstreamError {
                                key: key_owned,
                                detail: "origin returned truncated body".into(),
                            });
                        }
                        return Ok(Some(bytes[s..e].to_vec()));
                    }
                    Ok(Some(bytes.to_vec()))
                })
            }
        }
    }

    fn size<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<u64>, ZarrError>> {
        match self {
            Self::Local(fs) => fs.size(key),
            Self::Http { base_url, client } => {
                let url = format!("{base_url}/{}", key.trim_start_matches('/'));
                let client = client.clone();
                let key_owned = key.to_string();
                Box::pin(async move {
                    let resp = client.head(&url).send().await.map_err(|e| {
                        if e.is_timeout() {
                            ZarrError::UpstreamTimeout {
                                key: key_owned.clone(),
                                detail: e.to_string(),
                            }
                        } else {
                            ZarrError::UpstreamError {
                                key: key_owned.clone(),
                                detail: e.to_string(),
                            }
                        }
                    })?;
                    let status = resp.status().as_u16();
                    if status == 404 {
                        return Ok(None);
                    }
                    if status == 401 || status == 403 {
                        return Err(ZarrError::UpstreamRefused {
                            key: key_owned,
                            status,
                        });
                    }
                    let len = resp
                        .headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok())
                        .ok_or_else(|| ZarrError::UpstreamError {
                            key: key_owned,
                            detail: "missing Content-Length on HEAD".into(),
                        })?;
                    Ok(Some(len))
                })
            }
        }
    }

    fn list_dir<'a>(
        &'a self,
        prefix: &'a str,
    ) -> BoxFuture<'a, Result<Option<Vec<String>>, ZarrError>> {
        match self {
            Self::Local(fs) => fs.list_dir(prefix),
            Self::Http { .. } => Box::pin(async { Ok(None) }),
        }
    }
}

/// Parsed `/zarr/*` query parameters.
#[derive(Debug, Default)]
struct ZarrQuery {
    url: Option<String>,
    variable: Option<String>,
    group: Option<String>,
    sel: Vec<String>,
    decode_times: bool,
    show_times: bool,
}

impl ZarrQuery {
    fn parse(raw_query: &str) -> Self {
        let mut q = Self {
            decode_times: true,
            ..Default::default()
        };
        for pair in raw_query.split('&').filter(|s| !s.is_empty()) {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            let val = percent_decode(v);
            match k {
                "url" => q.url = Some(val),
                "variable" => q.variable = Some(val),
                "group" => q.group = Some(val),
                "sel" => q.sel.push(val),
                "decode_times" => {
                    q.decode_times = !matches!(
                        val.to_ascii_lowercase().as_str(),
                        "false" | "0" | "no"
                    );
                }
                "show_times" => {
                    q.show_times = matches!(
                        val.to_ascii_lowercase().as_str(),
                        "true" | "1" | "yes"
                    );
                }
                _ => {}
            }
        }
        q
    }
}

/// Main dispatcher for all `/zarr/*` routes.
pub(crate) async fn handle_zarr_request(
    _engine: &Arc<TileEngine>,
    method: Method,
    sub: &[&str],
    raw_query: &str,
) -> Response<Full<Bytes>> {
    let q = ZarrQuery::parse(raw_query);
    let result = match (method, sub) {
        (Method::GET, ["info"]) => handle_info(&q, false).await,
        (Method::GET, ["info.geojson"]) => handle_info(&q, true).await,
        (Method::GET, ["bounds"]) => handle_bounds(&q).await,
        (Method::GET, ["validate"]) => handle_validate(&q).await,
        (Method::GET, ["dataset"]) | (Method::GET, ["dataset", "keys"]) => {
            handle_dataset_keys(&q).await
        }
        (Method::GET, ["dataset", "dict"]) => handle_dataset_dict(&q).await,
        (Method::GET, ["dataset", "dimensions"]) => handle_dataset_dimensions(&q).await,
        (Method::GET, ["dataset", "coordinates", name]) => {
            handle_dataset_coordinate(&q, &percent_decode(name)).await
        }
        _ => {
            return Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
                .body(Full::new(Bytes::from(
                    r#"{"detail":"Zarr route not found"}"#,
                )))
                .unwrap();
        }
    };

    match result {
        Ok(resp) => resp,
        Err(err) => zarr_error_to_response(&err),
    }
}

async fn open_store_and_hierarchy(q: &ZarrQuery) -> Result<(ServerZarrStore, ZarrHierarchy), ZarrError> {
    let url = q.url.as_deref().unwrap_or("");
    let store = ServerZarrStore::from_url(url)?;
    let hierarchy = ZarrHierarchy::open(&store).await?;
    Ok((store, hierarchy))
}

fn dtype_str(dt: DataType) -> &'static str {
    match dt {
        DataType::Bool => "bool",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::UInt8 => "uint8",
        DataType::UInt16 => "uint16",
        DataType::UInt32 => "uint32",
        DataType::UInt64 => "uint64",
        DataType::Float16 => "float16",
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
    }
}

fn wgs84_bounds_for_geo(geo: &cogzilla_zarr::ZarrGeo, width: u32, height: u32) -> [f64; 4] {
    geo.crs.model.wgs84_bounds(&geo.affine, width, height)
}

async fn handle_info(q: &ZarrQuery, as_geojson: bool) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;

    let (height, width, count) = if sel_var.meta.rank() == 2 {
        (sel_var.meta.shape[0] as u32, sel_var.meta.shape[1] as u32, 1u32)
    } else {
        let mut c = sel_var.meta.shape[sel_var.dims.extra_axis.unwrap_or(0)] as u32;
        if !q.sel.is_empty() {
            let specs = parse_sel_params(&q.sel)?;
            for spec in &specs {
                if sel_var.dims.extra_name.as_deref() == Some(spec.dim.as_str())
                    && let Some(coord_meta) = group.array(&spec.dim)
                {
                    let vals = read_coord_1d(&store, &group.child_prefix(&spec.dim), coord_meta).await?;
                    let cf = if q.decode_times {
                        CfTimeUnits::from_meta(coord_meta)?
                    } else {
                        None
                    };
                    let idxs = resolve_sel_indices(spec, &vals, cf.as_ref())?;
                    c = idxs.len() as u32;
                }
            }
        }
        (
            sel_var.meta.shape[sel_var.dims.y_axis] as u32,
            sel_var.meta.shape[sel_var.dims.x_axis] as u32,
            c,
        )
    };

    let wgs84 = wgs84_bounds_for_geo(&geo, width, height);
    let band_descriptions: Vec<(String, String)> = (1..=count)
        .map(|i| (format!("b{i}"), sel_var.name.clone()))
        .collect();
    let band_metadata: Vec<(String, Value)> = (1..=count)
        .map(|i| (format!("b{i}"), json!(sel_var.meta.attributes)))
        .collect();
    let colorinterp: Vec<&str> = (0..count).map(|_| "undefined").collect();

    let nodata_val = sel_var.meta.effective_nodata();
    let (nodata_type, nodata_json) = match nodata_val {
        None => ("None", Value::Null),
        Some(v) if v.is_nan() => ("Nodata", Value::String("nan".into())),
        Some(v) => ("Nodata", json!(v)),
    };

    let mut info = json!({
        "bounds": geo.bounds,
        "wgs84_bounds": wgs84,
        "crs": geo.crs.description,
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype_str(sel_var.meta.dtype),
        "driver": "Zarr",
        "nodata_type": nodata_type,
        "nodata_value": nodata_json,
        "band_metadata": band_metadata,
        "band_descriptions": band_descriptions,
        "colorinterp": colorinterp,
        "variable": sel_var.name,
        "group": q.group.as_deref().unwrap_or(""),
        "dimensions": sel_var.meta.dimension_names,
        "attributes": sel_var.meta.attributes,
    });

    if let Some(vmin) = sel_var.valid_min {
        info["valid_min"] = json!(vmin);
    }
    if let Some(vmax) = sel_var.valid_max {
        info["valid_max"] = json!(vmax);
    }

    if q.show_times
        && let Some(ref extra_dim) = sel_var.dims.extra_name
        && let Some(t_meta) = group.array(extra_dim)
    {
        let raw_vals = read_coord_1d(&store, &group.child_prefix(extra_dim), t_meta).await?;
        if q.decode_times
            && let Some(cf) = CfTimeUnits::from_meta(t_meta)?
        {
            let iso: Vec<String> = raw_vals.iter().map(|&v| cf.decode_to_iso(v)).collect();
            info["times"] = json!(iso);
        } else {
            info["times"] = json!(raw_vals);
        }
    }

    if as_geojson {
        let [minx, miny, maxx, maxy] = wgs84;
        let feature = json!({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy],
                    [minx, miny],
                ]],
            },
            "properties": info,
        });
        Ok(json_ok(&feature, "application/geo+json"))
    } else {
        Ok(json_ok(&info, "application/json"))
    }
}

async fn handle_bounds(q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let width = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let height = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, width, height);
    let body = json!({
        "bounds": wgs84,
        "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
    });
    Ok(json_ok(&body, "application/json"))
}

async fn handle_validate(q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;

    let version = match hierarchy.version {
        ZarrVersion::V2 => 2,
        ZarrVersion::V3 => 3,
    };
    let open_mode = match hierarchy.open_mode {
        OpenMode::V3Consolidated => "v3_consolidated",
        OpenMode::V3Walked => "v3_walked",
        OpenMode::V3RootArray => "v3_root_array",
        OpenMode::V2Consolidated => "v2_consolidated",
        OpenMode::V2Walked => "v2_walked",
        OpenMode::V2RootArray => "v2_root_array",
    };
    let sharding = ShardingConfig::from_meta(sel_var.meta)?;
    let codecs: Vec<&str> = sel_var.meta.codecs.iter().map(|c| c.name.as_str()).collect();
    let geo_res = resolve_geo(&store, &group, sel_var.meta).await;
    let mut warnings = Vec::new();
    if !hierarchy.is_consolidated() {
        warnings.push("store has no consolidated metadata; opening requires walking groups");
    }

    let (valid, regular_grid, crs_desc, bounds_val, geo_error) = match geo_res {
        Ok(g) => (true, true, Some(g.crs.description), Some(g.bounds), None),
        Err(e) => (false, false, None, None, Some(e.to_string())),
    };

    let body = json!({
        "valid": valid,
        "version": version,
        "consolidated": hierarchy.is_consolidated(),
        "open_mode": open_mode,
        "group": q.group.as_deref().unwrap_or(""),
        "variable": sel_var.name,
        "shape": sel_var.meta.shape,
        "chunk_shape": sel_var.meta.chunk_shape,
        "inner_chunk_shape": sharding.as_ref().map(|s| &s.inner_chunk_shape),
        "sharded": sharding.is_some(),
        "dtype": dtype_str(sel_var.meta.dtype),
        "codecs": codecs,
        "regular_grid": regular_grid,
        "crs": crs_desc,
        "bounds": bounds_val,
        "geo_error": geo_error,
        "warnings": warnings,
    });
    Ok(json_ok(&body, "application/json"))
}

async fn handle_dataset_keys(q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (_store, hierarchy) = open_store_and_hierarchy(q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    let keys = group.data_variables();
    Ok(json_ok(&json!(keys), "application/json"))
}

async fn handle_dataset_dimensions(q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (_store, hierarchy) = open_store_and_hierarchy(q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    Ok(json_ok(&json!(group.dimensions()), "application/json"))
}

async fn handle_dataset_dict(q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (_store, hierarchy) = open_store_and_hierarchy(q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    let dims = group.dimensions();
    let data_vars_set: std::collections::BTreeSet<String> =
        group.data_variables().into_iter().collect();

    let mut coords = BTreeMap::new();
    let mut data_vars = BTreeMap::new();
    for (name, meta) in group.arrays() {
        let entry = array_summary_json(meta);
        if data_vars_set.contains(&name) {
            data_vars.insert(name, entry);
        } else {
            coords.insert(name, entry);
        }
    }

    let body = json!({
        "dims": dims,
        "attrs": group.attributes(),
        "coords": coords,
        "data_vars": data_vars,
        "groups": group.subgroups(),
    });
    Ok(json_ok(&body, "application/json"))
}

fn array_summary_json(meta: &ArrayMeta) -> Value {
    json!({
        "dims": meta.dimension_names,
        "shape": meta.shape,
        "chunks": meta.chunk_shape,
        "dtype": dtype_str(meta.dtype),
        "attrs": meta.attributes,
    })
}

async fn handle_dataset_coordinate(
    q: &ZarrQuery,
    coord_name: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy) = open_store_and_hierarchy(q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    let meta = group
        .array(coord_name)
        .ok_or_else(|| ZarrError::VariableNotFound {
            requested: coord_name.to_string(),
            available: group.variable_names(),
        })?;
    let raw_vals = read_coord_1d(&store, &group.child_prefix(coord_name), meta).await?;
    let data_val = if q.decode_times
        && let Some(cf) = CfTimeUnits::from_meta(meta)?
    {
        let iso: Vec<String> = raw_vals.iter().map(|&v| cf.decode_to_iso(v)).collect();
        json!(iso)
    } else {
        json!(raw_vals)
    };
    let body = json!({
        "name": coord_name,
        "dims": meta.dimension_names,
        "dtype": dtype_str(meta.dtype),
        "attrs": meta.attributes,
        "data": data_val,
    });
    Ok(json_ok(&body, "application/json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_zarr::SUPPORTED_CODECS;

    /// Asserts every single row of #388's failure-to-HTTP-status table: both the status code AND
    /// that the JSON `detail` message names the offending thing.
    #[test]
    fn status_table_covers_every_row_of_issue_388() {
        let cases: Vec<(ZarrError, StatusCode, &str)> = vec![
            (
                ZarrError::NotAZarrStore {
                    searched: "zarr.json, .zmetadata, .zgroup, .zarray",
                },
                StatusCode::BAD_REQUEST,
                "zarr.json",
            ),
            (
                ZarrError::NotFound {
                    key: "s3://bucket/missing.zarr".into(),
                },
                StatusCode::NOT_FOUND,
                "missing.zarr",
            ),
            (
                ZarrError::UpstreamRefused {
                    key: "s3://private/store".into(),
                    status: 403,
                },
                StatusCode::FORBIDDEN,
                "403",
            ),
            (
                ZarrError::UpstreamTimeout {
                    key: "https://example.com/zarr.json".into(),
                    detail: "timed out".into(),
                },
                StatusCode::GATEWAY_TIMEOUT,
                "zarr.json",
            ),
            (
                ZarrError::UpstreamError {
                    key: "https://example.com/c/0/0".into(),
                    detail: "truncated".into(),
                },
                StatusCode::BAD_GATEWAY,
                "c/0/0",
            ),
            (
                ZarrError::AmbiguousVariable {
                    available: vec!["temp".into(), "precip".into()],
                },
                StatusCode::BAD_REQUEST,
                "precip",
            ),
            (
                ZarrError::SelOutOfRange {
                    dim: "time".into(),
                    requested: "2025-01-01".into(),
                    min: "2000-01-01".into(),
                    max: "2000-06-01".into(),
                },
                StatusCode::BAD_REQUEST,
                "2000-06-01",
            ),
            (
                ZarrError::InvalidSel {
                    detail: "conflicting sel methods for dimension \"time\"".into(),
                },
                StatusCode::BAD_REQUEST,
                "conflicting",
            ),
            (
                ZarrError::UnsupportedRank(4),
                StatusCode::BAD_REQUEST,
                "4",
            ),
            (
                ZarrError::IrregularGrid {
                    dim: "x".into(),
                    index: 3,
                    detail: "non-uniform spacing".into(),
                },
                StatusCode::BAD_REQUEST,
                "index 3",
            ),
            (
                ZarrError::UnsupportedCrs {
                    crs: "geostationary".into(),
                    detail: "not modelled".into(),
                },
                StatusCode::BAD_REQUEST,
                "geostationary",
            ),
            (
                ZarrError::UnsupportedCodec {
                    codec: "mystery".into(),
                    chunk_key: "c/0/0".into(),
                    supported: SUPPORTED_CODECS,
                },
                StatusCode::NOT_IMPLEMENTED,
                "zstd",
            ),
            (
                ZarrError::MissingChunk {
                    chunk_key: "temp/c/1/2".into(),
                },
                StatusCode::BAD_GATEWAY,
                "temp/c/1/2",
            ),
            (
                ZarrError::Codec {
                    codec: "sharding_indexed".into(),
                    chunk_key: "data/c/0/0".into(),
                    detail: "crc32c mismatch".into(),
                },
                StatusCode::BAD_GATEWAY,
                "data/c/0/0",
            ),
            (
                ZarrError::CapExceeded {
                    kind: "covering chunk count",
                    computed: 512,
                    cap: 256,
                },
                StatusCode::BAD_REQUEST,
                "512",
            ),
        ];

        for (err, expected_status, expected_substr) in cases {
            let resp = zarr_error_to_response(&err);
            assert_eq!(resp.status(), expected_status, "wrong status for {err}");
            assert!(
                err.to_string().contains(expected_substr),
                "message {:?} missing {:?}",
                err.to_string(),
                expected_substr
            );
        }
    }
}
