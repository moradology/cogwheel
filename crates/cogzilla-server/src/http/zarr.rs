//! `/zarr/*` HTTP routes, unified failure-to-HTTP-status table (#388), and `cogzilla-io`-backed
//! Zarr store adapter.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, UNIX_EPOCH};

use bytes::Bytes;
use cogzilla_kernel::{
    ColormapName, OutputFormat, ResampleMode, SingleBandStyle, render_single_band, sample_general,
    wrap_raw_f32_as_npy_with_mask,
};
use cogzilla_proj::{CoordPlan, CrsModel, TileCoord, normalize_geographic_lon};
use cogzilla_zarr::hierarchy::BoxFuture;
use cogzilla_zarr::{
    ArrayMeta, CfTimeUnits, DataType, FsStore, MultiscalePyramid, OpenMode, ShardingConfig,
    ZarrCaps, ZarrError, ZarrGeo, ZarrHierarchy, ZarrStore, ZarrVersion, parse_sel_params,
    read_chunk_window_for_band_with_caps, read_coord_1d, resolve_geo, resolve_sel_indices,
    select_variable,
};
use dashmap::DashMap;
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CACHE_CONTROL, CONTENT_TYPE, ETAG, HeaderValue};
use hyper::{Method, Response, StatusCode};
use serde_json::{Value, json};

use crate::engine::TileEngine;
use crate::titiler::{
    PolygonMask, StatsOptions, build_map_html, compute_f32_band_stats, encode_geotiff_f32,
    encode_rgba_output,
};
use crate::tms::{TmsDef, lookup_tms};
use crate::zonal::{
    PixelAreaModel, ZonalGeomKind, compute_weighted_band_stats_with_geom,
    densify_ring_lonlat_to_pixel, rasterize_exact_coverage,
};
use super::html::append_vary_accept;
use super::{parse_u8_like, parse_u32_like, parse_y_scale_ext, percent_decode};

/// Global observability counters for `/zarr/*` exposed on `/metrics` (#386).
struct ZarrMetrics {
    stores_opened: AtomicU64,
    chunks_fetched: AtomicU64,
    chunk_holes: AtomicU64,
    bytes_decoded: AtomicU64,
    cap_refusals: AtomicU64,
}

static ZARR_METRICS: ZarrMetrics = ZarrMetrics {
    stores_opened: AtomicU64::new(0),
    chunks_fetched: AtomicU64::new(0),
    chunk_holes: AtomicU64::new(0),
    bytes_decoded: AtomicU64::new(0),
    cap_refusals: AtomicU64::new(0),
};

/// Append Prometheus `/metrics` lines for Zarr reads, holes, decoded bytes, and cap refusals (#386).
pub(crate) fn append_zarr_metrics(out: &mut String) {
    let _ = writeln!(out, "# HELP cogzilla_zarr_stores_opened_total Total Zarr hierarchies opened.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_stores_opened_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_stores_opened_total {}",
        ZARR_METRICS.stores_opened.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_chunks_fetched_total Total Zarr chunks decoded.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_chunks_fetched_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_chunks_fetched_total {}",
        ZARR_METRICS.chunks_fetched.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_chunk_holes_total Unwritten Zarr chunk holes short-circuited.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_chunk_holes_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_chunk_holes_total {}",
        ZARR_METRICS.chunk_holes.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_bytes_decoded_total Resident bytes decoded from Zarr chunks.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_bytes_decoded_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_bytes_decoded_total {}",
        ZARR_METRICS.bytes_decoded.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_cap_refusals_total Zarr requests refused by fan-out or decode caps.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_cap_refusals_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_cap_refusals_total {}",
        ZARR_METRICS.cap_refusals.load(Ordering::Relaxed)
    );
}

struct CachedZarrStore {
    hierarchy: Arc<ZarrHierarchy>,
    /// Store freshness validator (local `mtime_ns:size` or remote `ETag`/`Last-Modified`).
    validator: String,
    fetched_at: Instant,
}

fn zarr_hierarchy_cache() -> &'static DashMap<String, CachedZarrStore> {
    static CACHE: OnceLock<DashMap<String, CachedZarrStore>> = OnceLock::new();
    CACHE.get_or_init(DashMap::new)
}

/// Flush all cached Zarr store hierarchies (called by `POST /admin/clear-cache`, #387).
pub(crate) fn clear_zarr_cache() {
    zarr_hierarchy_cache().clear();
}

fn zarr_cache_ttl() -> Duration {
    let secs = std::env::var("COGZILLA_CACHE_TTL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5);
    Duration::from_secs(secs)
}

fn zarr_max_open_uris() -> usize {
    std::env::var("COGZILLA_MAX_OPEN_URIS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(4096)
}

fn zarr_cache_control_header() -> HeaderValue {
    let max_age = std::env::var("COGZILLA_ZARR_CACHE_CONTROL_MAX_AGE")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(60);
    if max_age == 0 {
        HeaderValue::from_static("no-cache")
    } else {
        HeaderValue::from_str(&format!("public, max-age={max_age}"))
            .unwrap_or_else(|_| HeaderValue::from_static("public, max-age=60"))
    }
}

fn local_store_validator(root: &std::path::Path) -> String {
    // Hash mtime + len of whichever root metadata file exists (plus root dir mtime if present).
    for name in ["zarr.json", ".zmetadata", ".zgroup", ".zarray"] {
        let p = root.join(name);
        if let Ok(meta) = std::fs::metadata(&p) {
            let mtime_ns = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            return format!("fs:{name}:{}:{mtime_ns}", meta.len());
        }
    }
    "fs:none".into()
}

fn record_window_metrics(wb: &cogzilla_zarr::ZarrWindowBlocks) {
    let mut fetched = 0u64;
    let mut holes = 0u64;
    let mut bytes = 0u64;
    for blk in &wb.blocks {
        match blk {
            Some(b) => {
                fetched += 1;
                bytes += b.data.byte_len() as u64;
            }
            None => holes += 1,
        }
    }
    ZARR_METRICS.chunks_fetched.fetch_add(fetched, Ordering::Relaxed);
    ZARR_METRICS.chunk_holes.fetch_add(holes, Ordering::Relaxed);
    ZARR_METRICS.bytes_decoded.fetch_add(bytes, Ordering::Relaxed);
}

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
    if matches!(err, ZarrError::CapExceeded { .. } | ZarrError::ChunkTooLarge { .. }) {
        ZARR_METRICS.cap_refusals.fetch_add(1, Ordering::Relaxed);
    }
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
    bidx: Option<u64>,
    decode_times: bool,
    show_times: bool,
    rescale: Option<(f32, f32)>,
    colormap_name: Option<String>,
    resampling: Option<String>,
    nodata: Option<f32>,
    unscale: bool,
    return_mask: bool,
    max_size: Option<u32>,
    tilesize: Option<u32>,
    format: Option<String>,
    coord_crs: Option<u16>,
}

impl ZarrQuery {
    fn parse(raw_query: &str) -> Self {
        let mut q = Self {
            decode_times: true,
            return_mask: true,
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
                "bidx" => {
                    if let Ok(b) = val.parse::<u64>()
                        && b >= 1
                    {
                        q.bidx = Some(b - 1);
                    }
                }
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
                "rescale" => {
                    if let Some((a, b)) = val.split_once(',')
                        && let (Ok(lo), Ok(hi)) = (a.trim().parse::<f32>(), b.trim().parse::<f32>())
                        && lo.is_finite()
                        && hi.is_finite()
                    {
                        q.rescale = Some((lo, hi));
                    }
                }
                "colormap_name" => q.colormap_name = Some(val),
                "resampling" | "reproject" => q.resampling = Some(val),
                "nodata" => {
                    if let Ok(nd) = val.parse::<f32>() {
                        q.nodata = Some(nd);
                    }
                }
                "unscale" => {
                    q.unscale = matches!(
                        val.to_ascii_lowercase().as_str(),
                        "true" | "1" | "yes"
                    );
                }
                "return_mask" => {
                    q.return_mask = !matches!(
                        val.to_ascii_lowercase().as_str(),
                        "false" | "0" | "no"
                    );
                }
                "max_size" => {
                    if let Ok(ms) = val.parse::<u32>() {
                        q.max_size = Some(ms);
                    }
                }
                "tilesize" => {
                    if let Ok(ts) = val.parse::<u32>() {
                        q.tilesize = Some(ts);
                    }
                }
                "format" | "f" => q.format = Some(val),
                "coord_crs" => {
                    let up = val.trim().to_ascii_uppercase();
                    if let Some(rest) = up.strip_prefix("EPSG:")
                        && let Ok(code) = rest.parse::<u16>()
                    {
                        q.coord_crs = Some(code);
                    }
                }
                _ => {}
            }
        }
        q
    }

    fn resample_mode(&self) -> ResampleMode {
        match self.resampling.as_deref().map(str::to_ascii_lowercase).as_deref() {
            Some("bilinear") => ResampleMode::Bilinear,
            Some("cubic") => ResampleMode::Cubic,
            Some("cubic_spline") => ResampleMode::CubicSpline,
            Some("lanczos") => ResampleMode::Lanczos,
            Some("average") => ResampleMode::Average,
            Some("mode") => ResampleMode::Mode,
            Some("max") => ResampleMode::Max,
            Some("min") => ResampleMode::Min,
            Some("med") => ResampleMode::Med,
            _ => ResampleMode::Nearest,
        }
    }

    fn colormap(&self) -> ColormapName {
        self.colormap_name
            .as_deref()
            .and_then(ColormapName::from_str_opt)
            .unwrap_or(ColormapName::Grayscale)
    }
}

fn parse_output_format(ext: Option<&str>, q_fmt: Option<&str>) -> (OutputFormat, &'static str) {
    let raw = ext.or(q_fmt).unwrap_or("png").to_ascii_lowercase();
    match raw.as_str() {
        "jpg" | "jpeg" => (OutputFormat::Jpeg { quality: 85 }, "image/jpeg"),
        "webp" => (OutputFormat::Webp { quality: Some(80.0) }, "image/webp"),
        "npy" => (OutputFormat::Npy, "application/x-binary"),
        "tif" | "tiff" => (OutputFormat::RawF32, "image/tiff; application=geotiff"),
        _ => (OutputFormat::Png, "image/png"),
    }
}

fn compute_zarr_etag(key: &str, body: &[u8]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    body.len().hash(&mut h);
    let sample = body.len().min(512);
    body[..sample].hash(&mut h);
    format!("\"{:016x}\"", h.finish())
}

fn binary_ok_with_etag(
    body: Bytes,
    content_type: &'static str,
    etag_key: &str,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    let etag = compute_zarr_etag(etag_key, &body);
    let cc = zarr_cache_control_header();
    if let Some(inm) = if_none_match {
        let matched = inm.trim() == "*"
            || inm
                .split(',')
                .any(|t| t.trim().trim_start_matches("W/") == etag);
        if matched {
            return Response::builder()
                .status(StatusCode::NOT_MODIFIED)
                .header(ETAG, HeaderValue::from_str(&etag).unwrap())
                .header(CACHE_CONTROL, cc)
                .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
                .body(Full::new(Bytes::new()))
                .unwrap();
        }
    }
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static(content_type))
        .header(ETAG, HeaderValue::from_str(&etag).unwrap())
        .header(CACHE_CONTROL, cc)
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(body))
        .unwrap()
}

/// Main dispatcher for all `/zarr/*` routes.
pub(crate) async fn handle_zarr_request(
    _engine: &Arc<TileEngine>,
    method: Method,
    sub: &[&str],
    raw_query: &str,
    host: &str,
    if_none_match: Option<&str>,
    post_body: Option<Bytes>,
) -> Response<Full<Bytes>> {
    let q = ZarrQuery::parse(raw_query);
    let result = match (method, sub) {
        // Metadata & introspection family (#372)
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

        // Raster family (#373): OGC Tilesets, TileJSON, Map HTML, WMTS, Tiles, Point, Preview, BBox, GET Statistics
        (Method::GET, ["tiles"]) => handle_tileset_list(&q, host, raw_query).await,
        (Method::GET, ["tiles", tms]) => handle_tileset(&q, tms, host, raw_query).await,
        (Method::GET, ["tilejson.json"]) => {
            handle_tilejson(&q, "WebMercatorQuad", host, raw_query).await
        }
        (Method::GET, [tms, "tilejson.json"]) => handle_tilejson(&q, tms, host, raw_query).await,
        (Method::GET, ["map.html"]) => handle_map_html("WebMercatorQuad", host, raw_query),
        (Method::GET, [tms, "map.html"]) => handle_map_html(tms, host, raw_query),
        (Method::GET, ["WMTSCapabilities.xml"]) => {
            handle_wmts(&q, "WebMercatorQuad", host, raw_query).await
        }
        (Method::GET, [tms, "WMTSCapabilities.xml"]) => handle_wmts(&q, tms, host, raw_query).await,
        (Method::GET, ["tiles", tms, zs, xs, ys]) => {
            handle_tile(_engine, &q, tms, zs, xs, ys, raw_query, if_none_match).await
        }
        (Method::GET, ["tiles", zs, xs, ys]) => {
            handle_tile(
                _engine,
                &q,
                "WebMercatorQuad",
                zs,
                xs,
                ys,
                raw_query,
                if_none_match,
            )
            .await
        }
        (Method::GET, ["point", coords]) => handle_point(&q, &percent_decode(coords)).await,
        (Method::GET, ["preview"]) => handle_preview(&q, None, None, raw_query, if_none_match).await,
        (Method::GET, [seg]) if seg.starts_with("preview.") => {
            let ext = seg.strip_prefix("preview.");
            handle_preview(&q, None, ext, raw_query, if_none_match).await
        }
        (Method::GET, ["preview", dims_seg]) => match parse_dims_ext(dims_seg) {
            Ok((dims, ext)) => handle_preview(&q, Some(dims), ext, raw_query, if_none_match).await,
            Err(e) => Err(e),
        },
        (Method::GET, ["bbox", bbox_seg]) => {
            let (bbox_str, ext) = split_bbox_ext(&percent_decode(bbox_seg));
            handle_bbox(&q, &bbox_str, None, ext.as_deref(), raw_query, if_none_match).await
        }
        (Method::GET, ["bbox", bbox_seg, dims_seg]) => match parse_dims_ext(dims_seg) {
            Ok((dims, ext)) => {
                handle_bbox(
                    &q,
                    &percent_decode(bbox_seg),
                    Some(dims),
                    ext,
                    raw_query,
                    if_none_match,
                )
                .await
            }
            Err(e) => Err(e),
        },
        (Method::GET, ["statistics"]) => handle_get_statistics(&q).await,

        // POST zonal statistics & feature routes (#374)
        (Method::POST, ["statistics"]) => {
            handle_post_statistics(&q, post_body.unwrap_or_default()).await
        }
        (Method::POST, ["feature"]) => {
            handle_post_feature(&q, None, None, post_body.unwrap_or_default()).await
        }
        (Method::POST, [seg]) if seg.starts_with("feature.") => {
            let ext = seg.strip_prefix("feature.");
            handle_post_feature(&q, None, ext, post_body.unwrap_or_default()).await
        }
        (Method::POST, ["feature", dims_seg]) => match parse_dims_ext(dims_seg) {
            Ok((dims, ext)) => {
                handle_post_feature(&q, Some(dims), ext, post_body.unwrap_or_default()).await
            }
            Err(e) => Err(e),
        },
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

fn parse_dims_ext(seg: &str) -> Result<((u32, u32), Option<&str>), ZarrError> {
    let (dims_part, ext) = match seg.rfind('.') {
        Some(pos) => (&seg[..pos], Some(&seg[pos + 1..])),
        None => (seg, None),
    };
    let Some((ws, hs)) = dims_part.split_once('x').or_else(|| dims_part.split_once('X')) else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid dimensions segment {seg:?} (expected WxH)"),
        });
    };
    let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>()) else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid width/height in {seg:?}"),
        });
    };
    if w == 0 || h == 0 || w > 2048 || h > 2048 {
        return Err(ZarrError::InvalidSelection {
            detail: format!("dimensions {w}x{h} out of allowed range 1..=2048"),
        });
    }
    Ok(((w, h), ext))
}

fn split_bbox_ext(seg: &str) -> (String, Option<String>) {
    // e.g. "-123,37,-121.5,38.5.png" -> ("-123,37,-121.5,38.5", Some("png"))
    if let Some(pos) = seg.rfind('.') {
        let after = &seg[pos + 1..];
        if !after.is_empty() && after.bytes().all(|b| b.is_ascii_alphabetic()) {
            return (seg[..pos].to_string(), Some(after.to_string()));
        }
    }
    (seg.to_string(), None)
}

async fn open_store_and_hierarchy(
    q: &ZarrQuery,
) -> Result<(ServerZarrStore, Arc<ZarrHierarchy>, String), ZarrError> {
    let url = q.url.as_deref().unwrap_or("");
    let store = ServerZarrStore::from_url(url)?;
    let cache = zarr_hierarchy_cache();

    match &store {
        ServerZarrStore::Local(fs) => {
            let validator = local_store_validator(fs.root());
            if let Some(entry) = cache.get(url)
                && entry.validator == validator
            {
                return Ok((store, Arc::clone(&entry.hierarchy), validator));
            }
            let hierarchy = Arc::new(ZarrHierarchy::open(&store).await?);
            ZARR_METRICS.stores_opened.fetch_add(1, Ordering::Relaxed);
            if cache.len() >= zarr_max_open_uris() {
                cache.clear();
            }
            cache.insert(
                url.to_string(),
                CachedZarrStore {
                    hierarchy: Arc::clone(&hierarchy),
                    validator: validator.clone(),
                    fetched_at: Instant::now(),
                },
            );
            Ok((store, hierarchy, validator))
        }
        ServerZarrStore::Http { base_url, client } => {
            let ttl = zarr_cache_ttl();
            if ttl > Duration::ZERO
                && let Some(entry) = cache.get(url)
                && entry.fetched_at.elapsed() < ttl
            {
                return Ok((store, Arc::clone(&entry.hierarchy), entry.validator.clone()));
            }

            // Probe root metadata validator cheaply via HEAD (ETag / Last-Modified)
            let mut remote_validator = String::new();
            for meta_name in ["zarr.json", ".zmetadata"] {
                if let Ok(resp) = client.head(format!("{base_url}/{meta_name}")).send().await
                    && resp.status().is_success()
                {
                    let et = resp
                        .headers()
                        .get(reqwest::header::ETAG)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("");
                    let lm = resp
                        .headers()
                        .get(reqwest::header::LAST_MODIFIED)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("");
                    let cl = resp
                        .headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("");
                    remote_validator = format!("http:{meta_name}:{et}:{lm}:{cl}");
                    break;
                }
            }

            if !remote_validator.is_empty()
                && let Some(mut entry) = cache.get_mut(url)
                && entry.validator == remote_validator
            {
                entry.fetched_at = Instant::now();
                return Ok((store, Arc::clone(&entry.hierarchy), remote_validator));
            }

            let hierarchy = Arc::new(ZarrHierarchy::open(&store).await?);
            ZARR_METRICS.stores_opened.fetch_add(1, Ordering::Relaxed);
            if cache.len() >= zarr_max_open_uris() {
                cache.clear();
            }
            let validator = if remote_validator.is_empty() {
                format!("http:gen:{}", ZARR_METRICS.stores_opened.load(Ordering::Relaxed))
            } else {
                remote_validator
            };
            cache.insert(
                url.to_string(),
                CachedZarrStore {
                    hierarchy: Arc::clone(&hierarchy),
                    validator: validator.clone(),
                    fetched_at: Instant::now(),
                },
            );
            Ok((store, hierarchy, validator))
        }
    }
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
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
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
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
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
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
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
    let (_store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    let keys = group.data_variables();
    Ok(json_ok(&json!(keys), "application/json"))
}

async fn handle_dataset_dimensions(q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (_store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    Ok(json_ok(&json!(group.dimensions()), "application/json"))
}

async fn handle_dataset_dict(q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (_store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
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
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
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

// ============================================================================
// Raster route family (#373) — all pixels flow through BandSource + cogzilla-kernel
// ============================================================================

async fn resolve_band_index(
    store: &ServerZarrStore,
    group: &cogzilla_zarr::GroupView<'_>,
    sel_var: &cogzilla_zarr::SelectedVariable<'_>,
    q: &ZarrQuery,
) -> Result<Option<u64>, ZarrError> {
    if sel_var.meta.rank() == 2 {
        return Ok(None);
    }
    if !q.sel.is_empty() {
        let specs = parse_sel_params(&q.sel)?;
        for spec in &specs {
            if sel_var.dims.extra_name.as_deref() == Some(spec.dim.as_str())
                && let Some(coord_meta) = group.array(&spec.dim)
            {
                let vals = read_coord_1d(store, &group.child_prefix(&spec.dim), coord_meta).await?;
                let cf = if q.decode_times {
                    CfTimeUnits::from_meta(coord_meta)?
                } else {
                    None
                };
                let idxs = resolve_sel_indices(spec, &vals, cf.as_ref())?;
                return Ok(idxs.first().map(|&i| i as u64));
            }
        }
    }
    Ok(Some(q.bidx.unwrap_or(0)))
}

fn require_tms(tms_id: &str) -> Result<&'static TmsDef, ZarrError> {
    lookup_tms(tms_id).ok_or_else(|| ZarrError::InvalidSelection {
        detail: format!("unknown TileMatrixSet {tms_id:?}"),
    })
}

fn zoom_range_for_zarr(tms: &TmsDef, geo: &ZarrGeo, width: u32, height: u32) -> (u8, u8) {
    tms.zoom_range_for_cog(&geo.crs.model, &geo.affine, width, height, 1.0)
}

async fn handle_tileset_list(
    q: &ZarrQuery,
    host: &str,
    raw_query: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, w, h);
    let qs = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let tilesets: Vec<Value> = crate::tms::ALL_TMS
        .iter()
        .map(|tms| {
            json!({
                "title": format!("Zarr tileset ({})", tms.id),
                "dataType": "map",
                "crs": tms.crs_uri,
                "tileMatrixSetURI": format!("http://www.opengis.net/def/tilematrixset/OGC/1.0/{}", tms.id),
                "boundingBox": {
                    "lowerLeft": [wgs84[0], wgs84[1]],
                    "upperRight": [wgs84[2], wgs84[3]],
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
                "links": [
                    {
                        "href": format!("{host}/zarr/tiles/{}{qs}", tms.id),
                        "rel": "self",
                        "type": "application/json",
                    },
                    {
                        "href": format!("{host}/zarr/tiles/{}/{{z}}/{{x}}/{{y}}{qs}", tms.id),
                        "rel": "item",
                        "templated": true,
                    },
                ],
            })
        })
        .collect();
    Ok(json_ok(&json!({ "tilesets": tilesets }), "application/json"))
}

async fn handle_tileset(
    q: &ZarrQuery,
    tms_id: &str,
    host: &str,
    raw_query: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, w, h);
    let (minz, maxz) = zoom_range_for_zarr(tms, &geo, w, h);
    let limits: Vec<Value> = (minz..=maxz)
        .filter_map(|z| {
            let m = tms.matrix(z)?;
            let (min_col, max_col, min_row, max_row) = tms.tile_limits_for_wgs84_bounds(m, wgs84);
            Some(json!({
                "tileMatrix": m.id,
                "minTileRow": min_row,
                "maxTileRow": max_row,
                "minTileCol": min_col,
                "maxTileCol": max_col,
            }))
        })
        .collect();
    let qs = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let body = json!({
        "title": format!("Zarr tileset ({})", tms.id),
        "dataType": "map",
        "crs": tms.crs_uri,
        "tileMatrixSetURI": format!("http://www.opengis.net/def/tilematrixset/OGC/1.0/{}", tms.id),
        "tileMatrixSetLimits": limits,
        "boundingBox": {
            "lowerLeft": [wgs84[0], wgs84[1]],
            "upperRight": [wgs84[2], wgs84[3]],
            "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
        },
        "links": [
            {
                "href": format!("{host}/zarr/tiles/{}{qs}", tms.id),
                "rel": "self",
                "type": "application/json",
            },
            {
                "href": format!("{host}/zarr/tiles/{}/{{z}}/{{x}}/{{y}}{qs}", tms.id),
                "rel": "item",
                "templated": true,
            },
        ],
    });
    Ok(json_ok(&body, "application/json"))
}

async fn handle_tilejson(
    q: &ZarrQuery,
    tms_id: &str,
    host: &str,
    raw_query: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, w, h);
    let (minz, maxz) = zoom_range_for_zarr(tms, &geo, w, h);
    let qs = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let tile_url = format!("{host}/zarr/tiles/{}/{{z}}/{{x}}/{{y}}{qs}", tms.id);
    let body = json!({
        "tilejson": "3.0.0",
        "name": sel_var.name,
        "version": "1.0.0",
        "scheme": "xyz",
        "tiles": [tile_url],
        "minzoom": minz,
        "maxzoom": maxz,
        "bounds": wgs84,
        "center": [0.5 * (wgs84[0] + wgs84[2]), 0.5 * (wgs84[1] + wgs84[3]), minz],
    });
    Ok(json_ok(&body, "application/json"))
}

fn handle_map_html(tms_id: &str, host: &str, raw_query: &str) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let qs = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let tj_url = format!("{host}/zarr/{}/tilejson.json{qs}", tms.id);
    let html = build_map_html(&tj_url);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(Bytes::from(html)))
        .unwrap())
}

async fn handle_wmts(
    q: &ZarrQuery,
    tms_id: &str,
    host: &str,
    raw_query: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, w, h);
    let (minz, maxz) = zoom_range_for_zarr(tms, &geo, w, h);
    let qs_esc = raw_query.replace('&', "&amp;");
    let tile_href = format!(
        "{host}/zarr/tiles/{}/{{TileMatrix}}/{{TileCol}}/{{TileRow}}.png?{qs_esc}",
        tms.id
    );
    let xml = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Capabilities xmlns="http://www.opengis.net/wmts/1.0" xmlns:ows="http://www.opengis.net/ows/1.1" version="1.0.0">
  <Contents>
    <Layer>
      <ows:Title>{title}</ows:Title>
      <ows:Identifier>{title}</ows:Identifier>
      <ows:WGS84BoundingBox>
        <ows:LowerCorner>{minx} {miny}</ows:LowerCorner>
        <ows:UpperCorner>{maxx} {maxy}</ows:UpperCorner>
      </ows:WGS84BoundingBox>
      <Format>image/png</Format>
      <TileMatrixSetLink><TileMatrixSet>{tms_id}</TileMatrixSet></TileMatrixSetLink>
      <ResourceURL format="image/png" resourceType="tile" template="{tile_href}"/>
    </Layer>
    <TileMatrixSet>
      <ows:Identifier>{tms_id}</ows:Identifier>
      <ows:SupportedCRS>{crs_uri}</ows:SupportedCRS>
      <!-- minzoom={minz} maxzoom={maxz} -->
    </TileMatrixSet>
  </Contents>
</Capabilities>"#,
        title = sel_var.name,
        minx = wgs84[0],
        miny = wgs84[1],
        maxx = wgs84[2],
        maxy = wgs84[3],
        tms_id = tms.id,
        crs_uri = tms.crs_uri,
    );
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static("application/xml"))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(Bytes::from(xml)))
        .unwrap())
}

/// Build an exact `CoordPlan<N>` from any TMS bounds `[minx, miny, maxx, maxy]` in `dst_crs` into
/// the Zarr array's pixel grid (`geo.crs.model` + `geo.affine`).
fn build_general_tile_plan<const N: usize>(
    dst_crs: &CrsModel,
    dst_bounds: [f64; 4],
    geo: &ZarrGeo,
    ifd_w: u32,
    ifd_h: u32,
) -> Option<CoordPlan<N>> {
    let crs_to_pixel = geo.affine.invert()?;
    let [min_x, min_y, max_x, max_y] = dst_bounds;
    let step_x = (max_x - min_x) / (N as f64);
    let step_y = (max_y - min_y) / (N as f64);

    let mut u = vec![f32::NAN; N * N].into_boxed_slice();
    let mut v = vec![f32::NAN; N * N].into_boxed_slice();
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut any_in = false;
    let fw = ifd_w as f32;
    let fh = ifd_h as f32;

    let same_3857 = matches!(dst_crs, CrsModel::Epsg3857) && matches!(geo.crs.model, CrsModel::Epsg3857);
    let same_geo = matches!(dst_crs, CrsModel::Geographic) && matches!(geo.crs.model, CrsModel::Geographic);

    for py in 0..N {
        let dy = max_y - (py as f64 + 0.5) * step_y;
        let row_off = py * N;
        for px in 0..N {
            let dx = min_x + (px as f64 + 0.5) * step_x;
            let (sx, sy) = if same_3857 || same_geo {
                (dx, dy)
            } else {
                let (lon, lat) = dst_crs.to_lonlat_deg(dx, dy);
                if !lon.is_finite() || !lat.is_finite() {
                    continue;
                }
                geo.crs.model.from_lonlat_deg(lon, lat)
            };
            if sx.is_finite() && sy.is_finite() {
                let (col, row) = crs_to_pixel.forward(sx, sy);
                let uu = (col - 0.5) as f32;
                let vv = (row - 0.5) as f32;
                u[row_off + px] = uu;
                v[row_off + px] = vv;
                if uu >= -1.0 && uu <= fw && vv >= -1.0 && vv <= fh {
                    any_in = true;
                    min_u = min_u.min(uu);
                    max_u = max_u.max(uu);
                    min_v = min_v.min(vv);
                    max_v = max_v.max(vv);
                }
            }
        }
    }
    if !any_in {
        return None;
    }
    let src_min_col = (min_u.floor() as i32).max(0);
    let src_max_col = (max_u.floor() as i32)
        .saturating_add(1)
        .min(ifd_w as i32 - 1);
    let src_min_row = (min_v.floor() as i32).max(0);
    let src_max_row = (max_v.floor() as i32)
        .saturating_add(1)
        .min(ifd_h as i32 - 1);
    if src_min_col > src_max_col || src_min_row > src_max_row {
        return None;
    }
    Some(CoordPlan::Exact2D {
        u,
        v,
        src_min_col,
        src_max_col,
        src_min_row,
        src_max_row,
        all_in_bounds: false,
    })
}

#[allow(clippy::too_many_arguments)]
async fn handle_tile(
    engine: &Arc<TileEngine>,
    q: &ZarrQuery,
    tms_id: &str,
    zs: &str,
    xs: &str,
    ys: &str,
    raw_query: &str,
    if_none_match: Option<&str>,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let (Some(z), Some(x), Some((y, scale_opt, ext_opt))) =
        (parse_u8_like(zs), parse_u32_like(xs), parse_y_scale_ext(ys))
    else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid tile coordinate z={zs} x={xs} y={ys}"),
        });
    };

    let tile_size = q
        .tilesize
        .unwrap_or_else(|| 256 * scale_opt.unwrap_or(1).clamp(1, 4));
    if tile_size == 0 || tile_size > 2048 {
        return Err(ZarrError::InvalidSelection {
            detail: format!("tilesize {tile_size} out of allowed range 1..=2048"),
        });
    }

    let Some(tile_bounds) = tms.tile_bounds(z, x, y, 0.0, tile_size) else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("tile {z}/{x}/{y} is out of matrix bounds for {tms_id}"),
        });
    };

    let (store, hierarchy, validator) = open_store_and_hierarchy(q).await?;
    // Honour GeoZarr / OME-Zarr `multiscales` when `?group=` is not explicitly pinned (#375).
    let mut effective_group = q.group.clone();
    if effective_group.is_none()
        && let Ok(root_g) = hierarchy.group(None)
        && let Ok(Some(pyr)) = MultiscalePyramid::discover(&hierarchy, &root_g, q.variable.as_deref())
    {
        let base_lvl = &pyr.levels[0];
        if let Ok(base_g) = hierarchy.group(Some(&base_lvl.group_path))
            && let Some(base_meta) = base_g.array(&base_lvl.variable)
            && let Ok(base_geo) = resolve_geo(&store, &base_g, base_meta).await
        {
            let tile_res = ((tile_bounds[2] - tile_bounds[0]).abs() / tile_size as f64)
                .min((tile_bounds[3] - tile_bounds[1]).abs() / tile_size as f64);
            let base_res = base_geo.affine.a.abs().min(base_geo.affine.e.abs()).max(1e-9);
            let req_downsample = tile_res / base_res;
            let chosen = pyr.select_for_downsample(req_downsample);
            effective_group = Some(chosen.group_path.clone());
        }
    }
    let sel_var = select_variable(&hierarchy, effective_group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(effective_group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;
    let dst_crs = tms.crs_model().map_err(|e| ZarrError::UnsupportedCrs {
        crs: tms_id.to_string(),
        detail: e.to_string(),
    })?;

    let (fmt, content_type) = parse_output_format(ext_opt, q.format.as_deref());
    let (rmin, rmax) = q.rescale.unwrap_or((0.0, 255.0));
    let style = SingleBandStyle {
        rescale_min: rmin,
        rescale_max: rmax,
        colormap: q.colormap(),
        resample: q.resample_mode(),
        format: fmt,
    };

    let caps = ZarrCaps::from_env();
    let etag_key = format!("zarr:tile:{validator}:{tms_id}/{z}/{x}/{y}/{tile_size}?{raw_query}");
    let tile_cache_key = format!("zarr_tile:{validator}:{tms_id}/{z}/{x}/{y}/{tile_size}?{raw_query}");

    // Fast warm-tile cache hit (keyed by store validator so mutated stores never serve stale tiles).
    if let Some(cached) = engine.tile_cache.get(&tile_cache_key) {
        return Ok(binary_ok_with_etag(
            cached.bytes.clone(),
            content_type,
            &etag_key,
            if_none_match,
        ));
    }

    // Fast path for standard 256x256 and 512x512 tiles directly through `render_single_band`
    if tile_size == 256 {
        let plan = if tms_id == "WebMercatorQuad" {
            CoordPlan::<256>::build(
                &geo.crs.model,
                &crs_to_pixel,
                ifd_w,
                ifd_h,
                TileCoord { z, x, y },
            )
        } else {
            build_general_tile_plan::<256>(&dst_crs, tile_bounds, &geo, ifd_w, ifd_h)
        };
        let Some(plan) = plan else {
            return Err(ZarrError::NotFound {
                key: format!("tile {tms_id}/{z}/{x}/{y} outside Zarr dataset bounds"),
            });
        };
        let (c0, c1, r0, r1) = plan.src_pixel_bounds();
        let pad = 2u32;
        let wx0 = (c0.max(0) as u32).saturating_sub(pad);
        let wy0 = (r0.max(0) as u32).saturating_sub(pad);
        let wx1 = ((c1.max(0) as u32) + 1 + pad).min(ifd_w);
        let wy1 = ((r1.max(0) as u32) + 1 + pad).min(ifd_h);

        let wb = read_chunk_window_for_band_with_caps(
            &store,
            &sel_var.store_prefix,
            sel_var.meta,
            band_idx,
            wx0,
            wy0,
            wx1,
            wy1,
            16,
            &caps,
        )
        .await?;
        let refs = wb.block_refs();
        let mut src = wb.as_band_source(&refs);
        if !q.unscale {
            src.scale_offset = None;
        }
        if let Some(nd) = q.nodata {
            src = src.with_nodata_override(Some(nd));
        }
        let body = if matches!(fmt, OutputFormat::RawF32) {
            let raw = render_single_band::<256>(
                &plan,
                &src,
                &SingleBandStyle {
                    format: OutputFormat::RawF32,
                    ..style
                },
            );
            encode_geotiff_f32(256, 256, &raw, tile_bounds, tms.epsg, q.return_mask)
        } else {
            render_single_band::<256>(&plan, &src, &style)
        };
        engine.tile_cache.insert(tile_cache_key, body.clone(), content_type);
        return Ok(binary_ok_with_etag(body, content_type, &etag_key, if_none_match));
    }

    // General tile size (e.g. 512x512, @2x, custom tilesize): sample through `sample_general`
    render_zarr_rect(
        &store,
        &sel_var,
        &geo,
        band_idx,
        &dst_crs,
        tile_bounds,
        tms.epsg,
        tile_size,
        tile_size,
        q,
        fmt,
        content_type,
        &etag_key,
        if_none_match,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn render_zarr_rect(
    store: &ServerZarrStore,
    sel_var: &cogzilla_zarr::SelectedVariable<'_>,
    geo: &ZarrGeo,
    band_idx: Option<u64>,
    dst_crs: &CrsModel,
    dst_bounds: [f64; 4],
    dst_epsg: u16,
    out_w: u32,
    out_h: u32,
    q: &ZarrQuery,
    fmt: OutputFormat,
    content_type: &'static str,
    etag_key: &str,
    if_none_match: Option<&str>,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;

    let [min_x, min_y, max_x, max_y] = dst_bounds;
    let step_x = (max_x - min_x) / (out_w as f64);
    let step_y = (max_y - min_y) / (out_h as f64);
    let same_3857 = matches!(dst_crs, CrsModel::Epsg3857) && matches!(geo.crs.model, CrsModel::Epsg3857);
    let same_geo = matches!(dst_crs, CrsModel::Geographic) && matches!(geo.crs.model, CrsModel::Geographic);

    let n_px = (out_w as usize) * (out_h as usize);
    let mut u_coords = vec![f32::NAN; n_px];
    let mut v_coords = vec![f32::NAN; n_px];
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut any_in = false;
    let fw = ifd_w as f32;
    let fh = ifd_h as f32;

    for py in 0..out_h as usize {
        let dy = max_y - (py as f64 + 0.5) * step_y;
        let row_off = py * (out_w as usize);
        for px in 0..out_w as usize {
            let dx = min_x + (px as f64 + 0.5) * step_x;
            let (sx, sy) = if same_3857 || same_geo {
                (dx, dy)
            } else {
                let (lon, lat) = dst_crs.to_lonlat_deg(dx, dy);
                if !lon.is_finite() || !lat.is_finite() {
                    continue;
                }
                geo.crs.model.from_lonlat_deg(lon, lat)
            };
            if sx.is_finite() && sy.is_finite() {
                let (col, row) = crs_to_pixel.forward(sx, sy);
                let uu = (col - 0.5) as f32;
                let vv = (row - 0.5) as f32;
                u_coords[row_off + px] = uu;
                v_coords[row_off + px] = vv;
                if uu >= -1.0 && uu <= fw && vv >= -1.0 && vv <= fh {
                    any_in = true;
                    min_u = min_u.min(uu);
                    max_u = max_u.max(uu);
                    min_v = min_v.min(vv);
                    max_v = max_v.max(vv);
                }
            }
        }
    }

    if !any_in {
        return Err(ZarrError::NotFound {
            key: "requested bbox/tile lies outside Zarr dataset bounds".into(),
        });
    }

    let pad = 2u32;
    let wx0 = (min_u.floor().max(0.0) as u32).saturating_sub(pad);
    let wy0 = (min_v.floor().max(0.0) as u32).saturating_sub(pad);
    let wx1 = ((max_u.ceil().max(0.0) as u32) + 1 + pad).min(ifd_w);
    let wy1 = ((max_v.ceil().max(0.0) as u32) + 1 + pad).min(ifd_h);

    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        wx0,
        wy0,
        wx1,
        wy1,
        16,
        &caps,
    )
    .await?;
    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let mode = q.resample_mode();
    let mut plane = vec![f32::NAN; n_px];
    for i in 0..n_px {
        let u = u_coords[i];
        let v = v_coords[i];
        if u.is_finite() && v.is_finite() {
            plane[i] = sample_general(&src, u, v, mode);
        }
    }

    let body = encode_zarr_plane(&plane, out_w, out_h, dst_bounds, dst_epsg, q, fmt);
    Ok(binary_ok_with_etag(body, content_type, etag_key, if_none_match))
}

fn encode_zarr_plane(
    plane: &[f32],
    out_w: u32,
    out_h: u32,
    dst_bounds: [f64; 4],
    dst_epsg: u16,
    q: &ZarrQuery,
    fmt: OutputFormat,
) -> Bytes {
    match fmt {
        OutputFormat::Npy => {
            let mut raw = Vec::with_capacity(plane.len() * 4);
            for &v in plane {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            wrap_raw_f32_as_npy_with_mask(Bytes::from(raw), 1, out_h, out_w, q.return_mask)
        }
        OutputFormat::RawF32 => {
            let mut raw = Vec::with_capacity(plane.len() * 4);
            for &v in plane {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            encode_geotiff_f32(out_w, out_h, &raw, dst_bounds, dst_epsg, q.return_mask)
        }
        _ => {
            let (rmin, rmax) = q.rescale.unwrap_or((0.0, 255.0));
            let denom = if (rmax - rmin).abs() < 1e-12 {
                1.0
            } else {
                rmax - rmin
            };
            let lut = q.colormap().lut();
            let mut rgba = vec![0u8; plane.len() * 4];
            for (i, &val) in plane.iter().enumerate() {
                if val.is_finite() {
                    let norm = ((val - rmin) / denom).clamp(0.0, 1.0);
                    let idx = (norm * 255.0).round() as usize;
                    let c = lut.rgb[idx];
                    rgba[i * 4] = c[0];
                    rgba[i * 4 + 1] = c[1];
                    rgba[i * 4 + 2] = c[2];
                    rgba[i * 4 + 3] = 255;
                }
            }
            encode_rgba_output(out_w as usize, out_h as usize, &rgba, fmt, q.return_mask)
        }
    }
}

async fn handle_point(q: &ZarrQuery, coords: &str) -> Result<Response<Full<Bytes>>, ZarrError> {
    let Some((xs, ys)) = coords.split_once(',') else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("expected /zarr/point/{{lon}},{{lat}}, got {coords:?}"),
        });
    };
    let (Ok(px), Ok(py)) = (xs.trim().parse::<f64>(), ys.trim().parse::<f64>()) else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid coordinates {coords:?}"),
        });
    };
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let (sx, sy) = if let Some(epsg) = q.coord_crs {
        let src_crs = CrsModel::from_epsg(epsg).map_err(|e| ZarrError::UnsupportedCrs {
            crs: format!("EPSG:{epsg}"),
            detail: e.to_string(),
        })?;
        let (lon, lat) = src_crs.to_lonlat_deg(px, py);
        geo.crs.model.from_lonlat_deg(lon, lat)
    } else {
        let lon = normalize_geographic_lon(px).ok_or_else(|| ZarrError::NotFound {
            key: format!("longitude {px} out of range"),
        })?;
        geo.crs.model.from_lonlat_deg(lon, py)
    };

    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;
    let (col_f, row_f) = crs_to_pixel.forward(sx, sy);
    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as i32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as i32;
    let col = col_f.floor() as i32;
    let row = row_f.floor() as i32;
    if col < 0 || row < 0 || col >= ifd_w || row >= ifd_h {
        return Err(ZarrError::NotFound {
            key: format!("point ({px}, {py}) outside Zarr dataset bounds"),
        });
    }

    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        &store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        col as u32,
        row as u32,
        (col as u32) + 1,
        (row as u32) + 1,
        1,
        &caps,
    )
    .await?;
    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let val_json = if sel_var.meta.dtype.needs_wide_path() {
        let v = src.sample_f64_checked(col, row);
        if v.is_nan() { Value::Null } else { json!(v) }
    } else {
        let v = src.sample_checked(col, row);
        if v.is_nan() { Value::Null } else { json!(v) }
    };

    let body = json!({
        "coordinates": [px, py],
        "values": [val_json],
        "band_names": ["b1"],
    });
    Ok(json_ok(&body, "application/json"))
}

async fn handle_preview(
    q: &ZarrQuery,
    dims: Option<(u32, u32)>,
    ext: Option<&str>,
    raw_query: &str,
    if_none_match: Option<&str>,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let (out_w, out_h) = match dims {
        Some((w, h)) => (w, h),
        None => {
            let max_s = q.max_size.unwrap_or(1024);
            if max_s == 0 || (ifd_w <= max_s && ifd_h <= max_s) {
                (ifd_w.min(2048), ifd_h.min(2048))
            } else {
                let scale = (max_s as f64) / (ifd_w.max(ifd_h) as f64);
                (
                    ((ifd_w as f64 * scale).round() as u32).max(1),
                    ((ifd_h as f64 * scale).round() as u32).max(1),
                )
            }
        }
    };

    let (fmt, content_type) = parse_output_format(ext, q.format.as_deref());
    let etag_key = format!("zarr:preview:{out_w}x{out_h}?{raw_query}");
    render_zarr_rect(
        &store,
        &sel_var,
        &geo,
        band_idx,
        &geo.crs.model,
        geo.bounds,
        geo.crs.epsg.unwrap_or(4326),
        out_w,
        out_h,
        q,
        fmt,
        content_type,
        &etag_key,
        if_none_match,
    )
    .await
}

async fn handle_bbox(
    q: &ZarrQuery,
    bbox_str: &str,
    dims: Option<(u32, u32)>,
    ext: Option<&str>,
    raw_query: &str,
    if_none_match: Option<&str>,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let parts: Vec<f64> = bbox_str
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    if parts.len() != 4 || parts.iter().any(|v| !v.is_finite()) {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid bbox {bbox_str:?} (expected minx,miny,maxx,maxy)"),
        });
    }
    let (mut minx, miny, mut maxx, maxy) = (parts[0], parts[1], parts[2], parts[3]);
    let dst_epsg = q.coord_crs.unwrap_or(4326);
    let dst_crs = CrsModel::from_epsg(dst_epsg).map_err(|e| ZarrError::UnsupportedCrs {
        crs: format!("EPSG:{dst_epsg}"),
        detail: e.to_string(),
    })?;
    if matches!(dst_crs, CrsModel::Geographic) && minx > maxx {
        maxx += 360.0;
    }
    if minx >= maxx || miny >= maxy {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid bbox ordering ({minx},{miny},{maxx},{maxy})"),
        });
    }
    let _ = &mut minx;

    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let (out_w, out_h) = dims.unwrap_or((256, 256));
    let (fmt, content_type) = parse_output_format(ext, q.format.as_deref());
    let etag_key = format!("zarr:bbox:{bbox_str}:{out_w}x{out_h}?{raw_query}");
    render_zarr_rect(
        &store,
        &sel_var,
        &geo,
        band_idx,
        &dst_crs,
        [minx, miny, maxx, maxy],
        dst_epsg,
        out_w,
        out_h,
        q,
        fmt,
        content_type,
        &etag_key,
        if_none_match,
    )
    .await
}

async fn handle_get_statistics(q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        &store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        0,
        0,
        ifd_w,
        ifd_h,
        16,
        &caps,
    )
    .await?;
    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let mut pixels = Vec::with_capacity((ifd_w as usize) * (ifd_h as usize));
    for r in 0..ifd_h as i32 {
        for c in 0..ifd_w as i32 {
            pixels.push(src.sample_checked(c, r));
        }
    }
    let opts = StatsOptions::default();
    let b1 = compute_f32_band_stats(&pixels, &sel_var.name, &opts);
    record_window_metrics(&wb);
    Ok(json_ok(&json!({ "b1": b1 }), "application/json"))
}

// ============================================================================
// POST /zarr/statistics & POST /zarr/feature (#374) — reuses zonal.rs directly
// ============================================================================

async fn handle_post_statistics(
    q: &ZarrQuery,
    body: Bytes,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let geojson: Value = serde_json::from_slice(&body)?;
    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let coord_crs_model = match q.coord_crs {
        Some(code) => Some(CrsModel::from_epsg(code).map_err(|e| ZarrError::UnsupportedCrs {
            crs: format!("EPSG:{code}"),
            detail: e.to_string(),
        })?),
        None => None,
    };

    if geojson.get("type").and_then(Value::as_str) == Some("FeatureCollection") {
        let mut out = geojson.clone();
        if let Some(feats) = out.get_mut("features").and_then(Value::as_array_mut) {
            for feat in feats.iter_mut() {
                let stats = compute_zarr_feature_stats(
                    &store,
                    &sel_var,
                    &geo,
                    band_idx,
                    feat,
                    coord_crs_model.as_ref(),
                    q,
                )
                .await?;
                if !feat.get("properties").is_some_and(Value::is_object) {
                    feat["properties"] = json!({});
                }
                feat["properties"]["statistics"] = stats;
            }
        }
        return Ok(json_ok(&out, "application/geo+json"));
    }

    let stats = compute_zarr_feature_stats(
        &store,
        &sel_var,
        &geo,
        band_idx,
        &geojson,
        coord_crs_model.as_ref(),
        q,
    )
    .await?;

    if geojson.get("type").and_then(Value::as_str) == Some("Feature") {
        let mut out = geojson;
        if !out.get("properties").is_some_and(Value::is_object) {
            out["properties"] = json!({});
        }
        out["properties"]["statistics"] = stats;
        Ok(json_ok(&out, "application/geo+json"))
    } else {
        let out = json!({
            "type": "Feature",
            "geometry": geojson,
            "properties": { "statistics": stats },
        });
        Ok(json_ok(&out, "application/geo+json"))
    }
}

async fn compute_zarr_feature_stats(
    store: &ServerZarrStore,
    sel_var: &cogzilla_zarr::SelectedVariable<'_>,
    geo: &ZarrGeo,
    band_idx: Option<u64>,
    geojson: &Value,
    coord_crs: Option<&CrsModel>,
    q: &ZarrQuery,
) -> Result<Value, ZarrError> {
    let (mask, _bbox) =
        PolygonMask::from_geojson_in_crs(geojson, coord_crs).map_err(|e| ZarrError::InvalidSelection {
            detail: e.to_string(),
        })?;
    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as usize;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as usize;
    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;

    // Densify polygon rings directly into Zarr native pixel space
    let is_geo = matches!(geo.crs.model, CrsModel::Geographic);
    let mut px_polys: Vec<Vec<Vec<(f64, f64)>>> = Vec::new();
    let mut min_c = f64::INFINITY;
    let mut max_c = f64::NEG_INFINITY;
    let mut min_r = f64::INFINITY;
    let mut max_r = f64::NEG_INFINITY;
    for poly in &mask.polygons {
        let mut rings = Vec::with_capacity(poly.len());
        for ring in poly {
            let densified = densify_ring_lonlat_to_pixel(
                ring,
                &geo.crs.model,
                &crs_to_pixel,
                None,
                is_geo,
                !is_geo,
                0.0,
                ring.len(),
            );
            for &(c, r) in &densified {
                if c.is_finite() && r.is_finite() {
                    min_c = min_c.min(c);
                    max_c = max_c.max(c);
                    min_r = min_r.min(r);
                    max_r = max_r.max(r);
                }
            }
            rings.push(densified);
        }
        px_polys.push(rings);
    }

    if !min_c.is_finite() || max_c < 0.0 || max_r < 0.0 || min_c >= ifd_w as f64 || min_r >= ifd_h as f64 {
        return Err(ZarrError::NotFound {
            key: "GeoJSON feature does not intersect Zarr dataset bounds".into(),
        });
    }

    let c0 = (min_c.floor().max(0.0) as usize).min(ifd_w.saturating_sub(1));
    let r0 = (min_r.floor().max(0.0) as usize).min(ifd_h.saturating_sub(1));
    let c1 = (max_c.ceil().max(0.0) as usize).min(ifd_w);
    let r1 = (max_r.ceil().max(0.0) as usize).min(ifd_h);
    let win_w = c1.saturating_sub(c0).max(1);
    let win_h = r1.saturating_sub(r0).max(1);

    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        c0 as u32,
        r0 as u32,
        (c0 + win_w) as u32,
        (r0 + win_h) as u32,
        16,
        &caps,
    )
    .await?;
    record_window_metrics(&wb);

    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let mut pixels = Vec::with_capacity(win_w * win_h);
    for r in r0..r0 + win_h {
        for c in c0..c0 + win_w {
            pixels.push(src.sample_checked(c as i32, r as i32));
        }
    }

    let cov = rasterize_exact_coverage(&px_polys, &[], &[], c0, r0, win_w, win_h);
    let area_model = PixelAreaModel::build(&geo.crs.model, &geo.affine, c0, r0, win_w, win_h);
    let opts = StatsOptions::default();
    let b1 = compute_weighted_band_stats_with_geom(
        &pixels,
        &cov,
        win_w,
        win_h,
        &area_model,
        true,
        ZonalGeomKind::Polygon,
        None,
        None,
        &sel_var.name,
        &opts,
    );
    Ok(json!({ "b1": b1 }))
}

async fn handle_post_feature(
    q: &ZarrQuery,
    dims: Option<(u32, u32)>,
    ext: Option<&str>,
    body: Bytes,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let geojson: Value = serde_json::from_slice(&body)?;
    let coord_crs_model = match q.coord_crs {
        Some(code) => Some(CrsModel::from_epsg(code).map_err(|e| ZarrError::UnsupportedCrs {
            crs: format!("EPSG:{code}"),
            detail: e.to_string(),
        })?),
        None => None,
    };
    let (mask, wgs_bbox) =
        PolygonMask::from_geojson_in_crs(&geojson, coord_crs_model.as_ref()).map_err(|e| {
            ZarrError::InvalidSelection {
                detail: e.to_string(),
            }
        })?;

    let (store, hierarchy, _validator) = open_store_and_hierarchy(q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let (out_w, out_h) = dims.unwrap_or((256, 256));
    let dst_crs = CrsModel::Geographic;
    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;

    let [min_x, min_y, max_x, max_y] = wgs_bbox;
    let step_x = (max_x - min_x) / (out_w as f64);
    let step_y = (max_y - min_y) / (out_h as f64);
    let n_px = (out_w as usize) * (out_h as usize);
    let mut u_coords = vec![f32::NAN; n_px];
    let mut v_coords = vec![f32::NAN; n_px];
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut any_in = false;
    let fw = ifd_w as f32;
    let fh = ifd_h as f32;

    for py in 0..out_h as usize {
        let lat = max_y - (py as f64 + 0.5) * step_y;
        let row_off = py * (out_w as usize);
        for px in 0..out_w as usize {
            let lon = min_x + (px as f64 + 0.5) * step_x;
            let (sx, sy) = geo.crs.model.from_lonlat_deg(lon, lat);
            if sx.is_finite() && sy.is_finite() {
                let (col, row) = crs_to_pixel.forward(sx, sy);
                let uu = (col - 0.5) as f32;
                let vv = (row - 0.5) as f32;
                u_coords[row_off + px] = uu;
                v_coords[row_off + px] = vv;
                if uu >= -1.0 && uu <= fw && vv >= -1.0 && vv <= fh {
                    any_in = true;
                    min_u = min_u.min(uu);
                    max_u = max_u.max(uu);
                    min_v = min_v.min(vv);
                    max_v = max_v.max(vv);
                }
            }
        }
    }
    if !any_in {
        return Err(ZarrError::NotFound {
            key: "GeoJSON feature does not intersect Zarr dataset bounds".into(),
        });
    }

    let pad = 2u32;
    let wx0 = (min_u.floor().max(0.0) as u32).saturating_sub(pad);
    let wy0 = (min_v.floor().max(0.0) as u32).saturating_sub(pad);
    let wx1 = ((max_u.ceil().max(0.0) as u32) + 1 + pad).min(ifd_w);
    let wy1 = ((max_v.ceil().max(0.0) as u32) + 1 + pad).min(ifd_h);

    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        &store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        wx0,
        wy0,
        wx1,
        wy1,
        16,
        &caps,
    )
    .await?;
    record_window_metrics(&wb);
    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let mode = q.resample_mode();
    let poly_mask = mask.rasterize_all_touched(out_w as usize, out_h as usize, wgs_bbox, &dst_crs);
    let mut plane = vec![f32::NAN; n_px];
    for i in 0..n_px {
        if poly_mask[i] && u_coords[i].is_finite() && v_coords[i].is_finite() {
            plane[i] = sample_general(&src, u_coords[i], v_coords[i], mode);
        }
    }

    let (fmt, content_type) = parse_output_format(ext, q.format.as_deref());
    let encoded = encode_zarr_plane(&plane, out_w, out_h, wgs_bbox, 4326, q, fmt);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static(content_type))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(encoded))
        .unwrap())
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
