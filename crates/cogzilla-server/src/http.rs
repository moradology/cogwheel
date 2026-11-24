use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use cogzilla_kernel::{
    ColorFormula, ColormapLut, ColormapName, DemAlgorithm, ExprProgram, OutputFormat, ResampleMode,
    SpectralExpr, TerrainEncoding,
};
use cogzilla_proj::{TileCoord, mercator_to_lonlat_deg, tile_bounds_3857};
use http_body_util::{BodyExt, Full};
use hyper::header::{
    ACCESS_CONTROL_ALLOW_ORIGIN, CACHE_CONTROL, CONTENT_TYPE, ETAG, HeaderValue, IF_NONE_MATCH,
};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use tokio::net::TcpListener;

use crate::catalog::{MosaicJsonCatalog, SceneCatalog, load_catalog_json_str};
use crate::engine::{BandRequest, EngineError, TileEngine, TileRequest};
use crate::stac::{
    get_colormap_image, get_colormap_json, get_tms_json, list_colormaps_json, list_tms_json,
    load_stac_item, resolve_stac_assets,
};
use crate::titiler::{
    PolygonMask, build_cog_info, build_cog_info_geojson, build_map_html, build_tilejson,
    compute_statistics, encode_rgba_output, query_point, render_mercator_bbox, render_wgs84_bbox,
};

fn percent_decode(input: &str) -> String {
    percent_decode_opts(input, true)
}

fn percent_decode_opts(input: &str, plus_as_space: bool) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h1 = (bytes[i + 1] as char).to_digit(16);
            let h2 = (bytes[i + 2] as char).to_digit(16);
            if let (Some(a), Some(b)) = (h1, h2) {
                out.push(((a << 4) | b) as u8);
                i += 3;
                continue;
            }
        } else if plus_as_space && bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parse TiTiler's `{y}[@{scale}x][.{ext}]` segment into `(y, scale_opt, ext_str)`.
fn parse_y_scale_ext(seg: &str) -> Option<(u32, Option<u32>, &str)> {
    let (left, ext) = match seg.split_once('.') {
        Some((l, e)) => (l, e),
        None => (seg, "png"),
    };
    let (y_str, scale) = match left.split_once('@') {
        Some((ys, sc)) => {
            let s_num = sc.trim_end_matches('x').parse::<u32>().ok()?;
            (ys, Some(s_num))
        }
        None => (left, None),
    };
    let y = y_str.parse::<u32>().ok()?;
    Some((y, scale, ext))
}

fn parse_wxh_ext(seg: &str) -> Option<(u32, u32, &str)> {
    let (dims, ext) = match seg.split_once('.') {
        Some((d, e)) => (d, e),
        None => (seg, "png"),
    };
    let (ws, hs) = dims.split_once('x')?;
    let w = ws.parse::<u32>().ok()?;
    let h = hs.parse::<u32>().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h, ext))
}

fn parse_bbox4(seg: &str) -> Option<([f64; 4], Option<&str>)> {
    // Handle `minx,miny,maxx,maxy.png` or `minx,miny,maxx,maxy`
    let (coords_str, ext) = if let Some(dot) = seg.rfind('.') {
        let after = &seg[dot + 1..];
        if matches!(
            after,
            "png" | "webp" | "f32" | "bin" | "raw" | "npy" | "jpg" | "jpeg"
        ) {
            (&seg[..dot], Some(after))
        } else {
            (seg, None)
        }
    } else {
        (seg, None)
    };
    let parts: Vec<&str> = coords_str.split(',').collect();
    if parts.len() != 4 {
        return None;
    }
    let b = [
        parts[0].parse::<f64>().ok()?,
        parts[1].parse::<f64>().ok()?,
        parts[2].parse::<f64>().ok()?,
        parts[3].parse::<f64>().ok()?,
    ];
    if !b.iter().all(|v| v.is_finite()) || b[0] >= b[2] || b[1] >= b[3] {
        return None;
    }
    Some((b, ext))
}

struct ParsedQuery {
    urls: Vec<String>,
    assets: Vec<String>,
    bidx: Vec<u16>,
    rescale: Vec<(f32, f32)>,
    colormap: ColormapName,
    custom_colormap_json: Option<String>,
    color_formula: Option<String>,
    resample: ResampleMode,
    reproject: ResampleMode,
    tile_size: u32,
    max_size: u32,
    webp_quality: Option<f32>,
    jpeg_quality: u8,
    expr_raw: Option<String>,
    nodata: Option<f32>,
    unscale: bool,
    return_mask: bool,
    buffer: u32,
    coord_crs_3857: bool,
    format_param: Option<String>,
    minzoom: Option<u8>,
    maxzoom: Option<u8>,
    tile_format: Option<String>,
    algorithm: Option<String>,
    azimuth: f32,
    altitude: f32,
    z_factor: f32,
    colormap_explicit: bool,
    colormap_name_set: bool,
    /// Parameter names present in the query, for per-route validation.
    keys: Vec<String>,
    /// Values that could not be parsed or name something unsupported.
    errors: Vec<String>,
}

fn parse_query(q: Option<&str>) -> ParsedQuery {
    let mut pq = ParsedQuery {
        urls: Vec::new(),
        assets: Vec::new(),
        bidx: Vec::new(),
        rescale: Vec::new(),
        colormap: ColormapName::Grayscale,
        custom_colormap_json: None,
        color_formula: None,
        resample: ResampleMode::Nearest,
        reproject: ResampleMode::Nearest,
        tile_size: 256,
        max_size: 1024,
        webp_quality: None,
        jpeg_quality: 85,
        expr_raw: None,
        nodata: None,
        unscale: false,
        return_mask: true,
        buffer: 0,
        coord_crs_3857: false,
        format_param: None,
        minzoom: None,
        maxzoom: None,
        tile_format: None,
        algorithm: None,
        azimuth: 315.0,
        altitude: 45.0,
        z_factor: 1.0,
        colormap_explicit: false,
        colormap_name_set: false,
        keys: Vec::new(),
        errors: Vec::new(),
    };
    let Some(q) = q else { return pq };
    let mut pq_errors: Vec<String> = Vec::new();
    for pair in q.split('&') {
        if pair.is_empty() {
            continue;
        }
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        pq.keys.push(k.to_string());
        let mut bad = |what: &str| pq_errors.push(format!("invalid {k}={}: {what}", percent_decode(v)));
        match k {
            // Do not convert `+` to space in `url=` (S3 keys and Azure SAS tokens often contain literal `+`)
            "url" => pq.urls.push(percent_decode_opts(v, false)),
            "assets" => {
                for a in percent_decode(v).split(',') {
                    if !a.is_empty() {
                        pq.assets.push(a.to_string());
                    }
                }
            }
            "format" | "f" => pq.format_param = Some(percent_decode(v).to_ascii_lowercase()),
            "algorithm" => pq.algorithm = Some(percent_decode(v).to_ascii_lowercase()),
            "algorithm_params" => {
                let dec = percent_decode(v);
                match serde_json::from_str::<serde_json::Value>(&dec) {
                    Ok(serde_json::Value::Object(obj)) => {
                        if let Some(val) = obj.get("azimuth") {
                            match val.as_f64().map(|a| a as f32) {
                                Some(a) if a.is_finite() => pq.azimuth = a,
                                _ => bad("azimuth must be a finite number"),
                            }
                        }
                        if let Some(val) = obj
                            .get("angle_altitude")
                            .or_else(|| obj.get("altitude"))
                            .or_else(|| obj.get("angle"))
                        {
                            match val.as_f64().map(|a| a as f32) {
                                Some(a) if a.is_finite() => pq.altitude = a.clamp(0.0, 90.0),
                                _ => bad("altitude must be a finite number"),
                            }
                        }
                        if let Some(val) = obj
                            .get("z_exaggeration")
                            .or_else(|| obj.get("z_factor"))
                            .or_else(|| obj.get("zfactor"))
                        {
                            match val.as_f64().map(|z| z as f32) {
                                Some(z) if z.is_finite() => pq.z_factor = z.max(1e-4),
                                _ => bad("z_factor must be a finite number"),
                            }
                        }
                    }
                    _ => bad("expected a JSON object"),
                }
            }
            "azimuth" => {
                if let Ok(a) = v.parse::<f32>()
                    && a.is_finite()
                {
                    pq.azimuth = a;
                } else {
                    bad("expected a finite number");
                }
            }
            "altitude" | "angle" | "angle_altitude" => {
                if let Ok(a) = v.parse::<f32>()
                    && a.is_finite()
                {
                    pq.altitude = a.clamp(0.0, 90.0);
                } else {
                    bad("expected a finite number");
                }
            }
            "z_factor" | "zfactor" => {
                if let Ok(z) = v.parse::<f32>()
                    && z.is_finite()
                {
                    pq.z_factor = z.max(1e-4);
                } else {
                    bad("expected a finite number");
                }
            }
            "bidx" | "indexes" => {
                for part in percent_decode(v).split(',') {
                    match part.parse::<u16>() {
                        Ok(0) => bad("band indices are 1-based (>= 1)"),
                        Ok(b) => pq.bidx.push(b),
                        Err(_) => bad("expected band numbers"),
                    }
                }
            }
            "rescale" => {
                let decoded = percent_decode(v);
                for group in decoded.split(';') {
                    let cleaned = group.trim().trim_matches(|c| c == '[' || c == ']');
                    match cleaned
                        .split_once(',')
                        .map(|(a, b)| (a.trim().parse::<f32>(), b.trim().parse::<f32>()))
                    {
                        Some((Ok(mn), Ok(mx))) if mn.is_finite() && mx.is_finite() => {
                            pq.rescale.push((mn, mx));
                        }
                        _ => bad("expected finite min,max"),
                    }
                }
            }
            "colormap" => {
                let dec = percent_decode(v);
                if dec.trim_start().starts_with('{') {
                    pq.custom_colormap_json = Some(dec);
                } else if let Some(cm) = ColormapName::from_str_opt(&dec) {
                    pq.colormap = cm;
                    pq.colormap_explicit = true;
                } else {
                    bad("expected a JSON colormap or a colormap name");
                }
            }
            "colormap_name" => {
                if let Some(cm) = ColormapName::from_str_opt(&percent_decode(v)) {
                    pq.colormap = cm;
                    pq.colormap_explicit = true;
                    pq.colormap_name_set = true;
                } else {
                    bad("unknown colormap (see /colorMaps)");
                }
            }
            "color_formula" => {
                pq.color_formula = Some(percent_decode(v));
            }
            "nodata" => {
                let dec = percent_decode(v);
                let trimmed = dec.trim();
                if trimmed.eq_ignore_ascii_case("nan") {
                    pq.nodata = Some(f32::NAN);
                } else if trimmed.eq_ignore_ascii_case("inf") || trimmed.eq_ignore_ascii_case("+inf") {
                    pq.nodata = Some(f32::INFINITY);
                } else if trimmed.eq_ignore_ascii_case("-inf") {
                    pq.nodata = Some(f32::NEG_INFINITY);
                } else if let Ok(nd) = trimmed.parse::<f32>()
                    && nd.is_finite()
                {
                    pq.nodata = Some(nd);
                } else {
                    bad("expected a number, nan, inf, or -inf");
                }
            }
            "unscale" => {
                pq.unscale = v.eq_ignore_ascii_case("true") || v == "1";
            }
            "return_mask" => {
                pq.return_mask = !(v.eq_ignore_ascii_case("false") || v == "0");
            }
            "buffer" => {
                if let Ok(buf) = v.parse::<f32>()
                    && buf.is_finite()
                {
                    pq.buffer = buf.round().clamp(0.0, 256.0) as u32;
                } else {
                    bad("expected a finite number");
                }
            }
            "coord_crs" => {
                let dec = percent_decode(v).to_ascii_lowercase();
                pq.coord_crs_3857 = dec.contains("3857");
                if !(dec.contains("3857") || dec.contains("4326") || dec.contains("crs84")) {
                    bad("only EPSG:4326 and EPSG:3857 are supported");
                }
            }
            "resample" | "resampling" => {
                pq.resample = match v {
                    "nearest" => ResampleMode::Nearest,
                    "cubic" | "cubicspline" | "cubic_spline" | "lanczos" => ResampleMode::Cubic,
                    "bilinear" => ResampleMode::Bilinear,
                    _ => {
                        bad("use nearest, bilinear or cubic");
                        ResampleMode::Nearest
                    }
                };
            }
            "reproject" => {
                pq.reproject = match v {
                    "nearest" => ResampleMode::Nearest,
                    "cubic" | "cubicspline" | "cubic_spline" | "lanczos" => ResampleMode::Cubic,
                    "bilinear" => ResampleMode::Bilinear,
                    _ => {
                        bad("use nearest, bilinear or cubic");
                        ResampleMode::Nearest
                    }
                };
            }
            "tile_size" | "tilesize" | "size" => {
                match v.parse::<u32>() {
                    Ok(sz) if (64..=2048).contains(&sz) => pq.tile_size = sz,
                    _ => bad("expected a tile size from 64 to 2048"),
                }
            }
            "scale" => match v.parse::<u32>() {
                Ok(sc) if (1..=4).contains(&sc) => pq.tile_size = sc * 256,
                _ => bad("expected scale in 1..=4"),
            },
            "max_size" => {
                if let Ok(ms) = v.parse::<u32>() {
                    pq.max_size = ms.clamp(64, 4096);
                } else {
                    bad("expected a number");
                }
            }
            "quality" => {
                if v != "lossless" {
                    if let Ok(qual) = v.parse::<f32>()
                        && qual.is_finite()
                    {
                        pq.webp_quality = Some(qual.clamp(1.0, 100.0));
                        pq.jpeg_quality = qual.round().clamp(1.0, 100.0) as u8;
                    } else {
                        bad("expected a finite quality in 1..=100 or 'lossless'");
                    }
                }
            }
            "expr" | "expression" | "index" => {
                // Standard HTTP clients (httpx, requests, URLSearchParams) form-encode spaces as `+`
                // and literal plus as `%2B`, while raw CLI/browser URLs often write `(b1+b2)` directly.
                // Decode `+` as space when form-encoding markers are present so both work cleanly.
                let plus_is_space = v.contains("%2B")
                    || v.contains("%2b")
                    || v.contains("+-")
                    || v.contains("-+")
                    || v.contains("+*")
                    || v.contains("*+")
                    || v.contains("+/")
                    || v.contains("/+")
                    || v.contains("++")
                    || v.contains("+(")
                    || v.contains(")+")
                    || v.contains("%2F+")
                    || v.contains("+%2F")
                    || v.contains("%2f+")
                    || v.contains("+%2f")
                    || v.contains("%28+")
                    || v.contains("+%29");
                pq.expr_raw = Some(percent_decode_opts(v, plus_is_space));
            }
            "padding" => {
                if v.parse::<f32>().ok() != Some(0.0) {
                    bad("padding is not supported");
                }
            }
            "minzoom" => match v.parse::<u8>() {
                Ok(z) if z <= 30 => pq.minzoom = Some(z),
                _ => bad("expected minzoom in 0..=30"),
            },
            "maxzoom" => match v.parse::<u8>() {
                Ok(z) if z <= 30 => pq.maxzoom = Some(z),
                _ => bad("expected maxzoom in 0..=30"),
            },
            "tile_format" => {
                let fmt = percent_decode(v).to_ascii_lowercase();
                if parse_ext_format(&fmt, None, 85).is_ok() {
                    pq.tile_format = Some(fmt);
                } else {
                    bad("unsupported tile_format (use png, webp, jpg, f32, or npy)");
                }
            }
            "dst_crs" | "crs" | "histogram_bins" | "histogram_range" | "categorical" | "c"
            | "p" | "cover_scale" | "height" | "width" | "orientation" => {
                pq_errors.push(format!("parameter '{k}' is not supported by cogzilla"));
            }
            _ => {}
        }
    }
    // TiTiler precedence: colormap_name wins over inline JSON colormap when both are supplied.
    if pq.colormap_name_set {
        pq.custom_colormap_json = None;
    }
    pq.errors = pq_errors;
    pq
}

const SUPPORTED_ALGORITHMS: &[&str] =
    &["hillshade", "slope", "aspect", "terrarium", "terrainrgb", "terrain-rgb", "mapbox"];
const SUPPORTED_TMS: &[&str] = &["WebMercatorQuad", "WorldCRS84Quad"];

/// Reject requests that ask for something this server would otherwise ignore.
///
/// TiTiler answers these with a 422. Returning a 200 with the option silently
/// dropped gives a client the wrong picture and no way to notice.
fn validate_query(segs: &[&str], pq: &ParsedQuery) -> Result<(), String> {
    let family = segs.first().copied().unwrap_or("");
    if !matches!(family, "cog" | "stac" | "mosaicjson" | "mosaic" | "tiles" | "colorMaps") {
        return Ok(());
    }
    if let Some(e) = pq.errors.first() {
        return Err(e.clone());
    }
    if let Some(alg) = pq.algorithm.as_deref() {
        if !SUPPORTED_ALGORITHMS.contains(&alg) {
            return Err(format!(
                "algorithm '{alg}' is not supported (use hillshade, slope, aspect, terrarium or terrainrgb)"
            ));
        }
    }

    // {family}/tiles/{tms}[/{z}/{x}/{y}], {family}/{tms}/tilejson.json, {family}/{tms}/map.html
    if family != "tiles" && family != "colorMaps" {
        let last = segs.last().copied().unwrap_or("");
        let tms = if (last == "tilejson.json" || last == "map.html") && segs.len() >= 3 {
            Some(segs[segs.len() - 2])
        } else {
            match segs.iter().position(|s| *s == "tiles") {
                Some(i) if matches!(segs.len() - i, 2 | 5) => Some(segs[i + 1]),
                _ => None,
            }
        };
        if let Some(tms) = tms {
            if matches!(family, "mosaic" | "mosaicjson") && tms != "WebMercatorQuad" {
                return Err(format!(
                    "tile matrix set '{tms}' is not supported on /{family} routes (use WebMercatorQuad)"
                ));
            }
            if !SUPPORTED_TMS.contains(&tms) {
                return Err(format!(
                    "tile matrix set '{tms}' is not supported (use WebMercatorQuad or WorldCRS84Quad)"
                ));
            }
        }
    }

    let has = |names: &[&str]| pq.keys.iter().find(|k| names.contains(&k.as_str())).cloned();
    // Only match route sub-endpoints on `/cog/...` and `/stac/...` so dataset/catalog names like
    // `/tiles/statistics/0/0/0.png` or `/mosaic/point/tiles/0/0/0.png` are never misclassified.
    let route = |name: &str| {
        matches!(segs[0], "cog" | "stac") && segs.get(1) == Some(&name)
    };
    if route("statistics") || route("asset_statistics") {
        if let Some(k) = has(&["expression", "expr", "index", "algorithm", "bidx", "indexes"]) {
            return Err(format!("parameter '{k}' is not supported on statistics routes"));
        }
    }
    if route("point") {
        if let Some(k) = has(&["expression", "expr", "index", "algorithm", "bidx", "indexes"]) {
            return Err(format!("parameter '{k}' is not supported on point routes"));
        }
    } else if !route("bbox") {
        if let Some(k) = has(&["coord_crs"]) {
            return Err(format!("parameter '{k}' is not supported on this route"));
        }
    }
    if route("bbox") && segs.len() >= 4 {
        if let Some(k) = has(&["max_size"]) {
            return Err(format!("parameter '{k}' is not supported when explicit {{w}}x{{h}} dimensions are given"));
        }
    }
    let last_seg = segs.last().copied().unwrap_or("");
    if last_seg != "tilejson.json" && last_seg != "map.html" {
        if let Some(k) = has(&["minzoom", "maxzoom", "tile_format"]) {
            return Err(format!(
                "parameter '{k}' is only supported on tilejson.json and map.html routes"
            ));
        }
    }
    Ok(())
}

/// Apply TiTiler `minzoom`, `maxzoom`, and `tile_format` query overrides to a TileJSON document,
/// stripping those TileJSON-only params from the forwarded tile URL template.
fn apply_tilejson_overrides(
    tj: &mut serde_json::Value,
    pq: &ParsedQuery,
    host: &str,
    route_prefix: &str,
    tms: &str,
    raw_query: &str,
) {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.to_string()
    } else {
        format!("http://{host}")
    };
    let filtered_query: String = raw_query
        .split('&')
        .filter(|pair| {
            let k = pair.split('=').next().unwrap_or("");
            !pair.is_empty() && !matches!(k, "minzoom" | "maxzoom" | "tile_format")
        })
        .collect::<Vec<_>>()
        .join("&");
    let ext_suffix = pq
        .tile_format
        .as_deref()
        .map(|f| format!(".{f}"))
        .unwrap_or_default();
    let tile_url = if filtered_query.is_empty() {
        format!("{origin}/{route_prefix}/tiles/{tms}/{{z}}/{{x}}/{{y}}{ext_suffix}")
    } else {
        format!("{origin}/{route_prefix}/tiles/{tms}/{{z}}/{{x}}/{{y}}{ext_suffix}?{filtered_query}")
    };
    tj["tiles"] = serde_json::json!([tile_url]);
    if let Some(minz) = pq.minzoom {
        tj["minzoom"] = serde_json::json!(minz);
        if let Some(center) = tj.get_mut("center").and_then(|v| v.as_array_mut())
            && center.len() >= 3
        {
            center[2] = serde_json::json!(minz);
        }
    }
    if let Some(maxz) = pq.maxzoom {
        tj["maxzoom"] = serde_json::json!(maxz);
    }
}

fn parse_ext_format(
    ext: &str,
    webp_quality: Option<f32>,
    jpeg_quality: u8,
) -> Result<(OutputFormat, &'static str), EngineError> {
    match ext {
        "png" | "" => Ok((OutputFormat::Png, "image/png")),
        "webp" => Ok((
            OutputFormat::Webp {
                quality: webp_quality,
            },
            "image/webp",
        )),
        "jpg" | "jpeg" => Ok((OutputFormat::Jpeg { quality: jpeg_quality }, "image/jpeg")),
        "f32" | "bin" | "raw" => Ok((OutputFormat::RawF32, "application/octet-stream")),
        "npy" => Ok((OutputFormat::Npy, "application/x-npy")),
        other => Err(EngineError::BadRequest(format!(
            "unsupported format '.{other}' (use .png, .webp, .jpg, or .f32)"
        ))),
    }
}

/// Fold GDAL `scale`/`offset` into rescale bounds when `?unscale=true` (zero per-pixel cost!).
/// Supports both single multi-band COGs and multi-asset STAC/multi-COG composites.
fn apply_unscale_cogs(bands: &mut BandRequest, cogs: &[Arc<cogzilla_io::OpenedCog>]) {
    if cogs.is_empty() {
        return;
    }
    let adj = |b1: u16, mn: f32, mx: f32| -> (f32, f32) {
        let i = b1.saturating_sub(1) as usize;
        let (sc, off) = if cogs.len() > 1 {
            let c = &cogs[i.min(cogs.len() - 1)];
            (
                c.index.scales.first().copied().unwrap_or(1.0),
                c.index.offsets.first().copied().unwrap_or(0.0),
            )
        } else {
            (
                cogs[0].index.scales.get(i).copied().unwrap_or(1.0),
                cogs[0].index.offsets.get(i).copied().unwrap_or(0.0),
            )
        };
        if (sc - 1.0).abs() < 1e-9 && off.abs() < 1e-9 || sc.abs() < 1e-12 {
            (mn, mx)
        } else {
            ((mn - off) / sc, (mx - off) / sc)
        }
    };
    match bands {
        BandRequest::Single {
            band_1based,
            rescale,
            ..
        } => {
            *rescale = adj(*band_1based, rescale.0, rescale.1);
        }
        BandRequest::Rgb {
            bands_1based,
            rescale_min,
            rescale_max,
        } => {
            for k in 0..3 {
                let (mn, mx) = adj(bands_1based[k], rescale_min[k], rescale_max[k]);
                rescale_min[k] = mn;
                rescale_max[k] = mx;
            }
        }
        _ => {}
    }
}

/// Compute aspect-preserving `(width, height)` inside `max_dim` for format-only `/bbox` and
/// `POST /feature` routes (matching TiTiler instead of stretching into a fixed square).
fn aspect_preserving_dims(bbox: [f64; 4], max_dim: u32) -> (u32, u32) {
    let max_d = max_dim.clamp(1, 4096) as f64;
    let dx = (bbox[2] - bbox[0]).abs();
    let dy = (bbox[3] - bbox[1]).abs();
    if !dx.is_finite() || !dy.is_finite() || dx <= 0.0 || dy <= 0.0 {
        let d = max_dim.clamp(1, 4096);
        return (d, d);
    }
    if dx >= dy {
        let w = max_d.round() as u32;
        let h = ((max_d * (dy / dx)).round() as u32).clamp(1, max_dim.clamp(1, 4096));
        (w.max(1), h)
    } else {
        let h = max_d.round() as u32;
        let w = ((max_d * (dx / dy)).round() as u32).clamp(1, max_dim.clamp(1, 4096));
        (w, h.max(1))
    }
}

/// Default rescale range matching TiTiler's `dtype_ranges` when `?rescale=` is omitted.
fn default_rescale_for_dtype(st: cogzilla_tiff::SampleType, single: bool) -> (f32, f32) {
    use cogzilla_tiff::SampleType;
    match st {
        SampleType::Uint8 => (0.0, 255.0),
        SampleType::Int8 => (-128.0, 127.0),
        SampleType::Uint16 => (0.0, 65535.0),
        SampleType::Int16 => (-32768.0, 32767.0),
        SampleType::Uint32 => (0.0, 4_294_967_295.0),
        SampleType::Int32 => (-2_147_483_648.0, 2_147_483_647.0),
        SampleType::Float32 | SampleType::Float64 => {
            if single {
                (0.0, 1000.0)
            } else {
                (0.0, 1.0)
            }
        }
    }
}

fn build_band_request(
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_tiff::SampleType,
) -> Result<BandRequest, EngineError> {
    if let Some(alg) = pq.algorithm.as_deref() {
        let band_1based = pq.bidx.first().copied().unwrap_or(1);
        match alg {
            "terrarium" => {
                return Ok(BandRequest::Terrain {
                    band_1based,
                    encoding: TerrainEncoding::Terrarium,
                });
            }
            "terrainrgb" | "terrain-rgb" | "mapbox" => {
                return Ok(BandRequest::Terrain {
                    band_1based,
                    encoding: TerrainEncoding::TerrainRgb,
                });
            }
            "hillshade" => {
                let rescale = pq.rescale.first().copied().unwrap_or((0.0, 255.0));
                let colormap = if pq.colormap_explicit {
                    pq.colormap
                } else {
                    ColormapName::from_str_opt("gray").unwrap_or(ColormapName::Viridis)
                };
                return Ok(BandRequest::Dem {
                    band_1based,
                    algorithm: DemAlgorithm::Hillshade {
                        azimuth_deg: pq.azimuth,
                        altitude_deg: pq.altitude,
                        z_factor: pq.z_factor,
                    },
                    rescale,
                    colormap,
                });
            }
            "slope" => {
                let rescale = pq.rescale.first().copied().unwrap_or((0.0, 90.0));
                return Ok(BandRequest::Dem {
                    band_1based,
                    algorithm: DemAlgorithm::Slope {
                        z_factor: pq.z_factor,
                    },
                    rescale,
                    colormap: pq.colormap,
                });
            }
            "aspect" => {
                let rescale = pq.rescale.first().copied().unwrap_or((0.0, 360.0));
                return Ok(BandRequest::Dem {
                    band_1based,
                    algorithm: DemAlgorithm::Aspect,
                    rescale,
                    colormap: pq.colormap,
                });
            }
            _ => {}
        }
    }
    if let Some(raw_expr) = &pq.expr_raw {
        if let Some(preset) = SpectralExpr::from_str_opt(raw_expr) {
            return Ok(match preset {
                SpectralExpr::Db => {
                    let rescale = pq.rescale.first().copied().unwrap_or((-25.0, 0.0));
                    let b = pq.bidx.first().copied().unwrap_or(1);
                    BandRequest::Spectral1 {
                        expr: preset,
                        band_1based: b,
                        rescale,
                        colormap: pq.colormap,
                    }
                }
                SpectralExpr::SarRgb => {
                    let bs = if pq.bidx.len() >= 2 {
                        [pq.bidx[0], pq.bidx[1]]
                    } else {
                        [1, 2]
                    };
                    let (rmin, rmax) = match pq.rescale.len() {
                        3 => (
                            [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                            [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
                        ),
                        1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
                        _ => ([-20.0, -25.0, 0.0], [0.0, -5.0, 15.0]),
                    };
                    BandRequest::SarRgb {
                        bands_1based: bs,
                        rescale_min: rmin,
                        rescale_max: rmax,
                    }
                }
                SpectralExpr::Evi => {
                    let rescale = pq.rescale.first().copied().unwrap_or((-0.2, 0.8));
                    let bs = if pq.bidx.len() >= 3 {
                        [pq.bidx[0], pq.bidx[1], pq.bidx[2]]
                    } else {
                        [1, 2, 3]
                    };
                    BandRequest::Spectral3 {
                        expr: preset,
                        bands_1based: bs,
                        rescale,
                        colormap: pq.colormap,
                    }
                }
                _ => {
                    let rescale = pq.rescale.first().copied().unwrap_or((-1.0, 1.0));
                    let bs = if pq.bidx.len() >= 2 {
                        [pq.bidx[0], pq.bidx[1]]
                    } else {
                        [1, 2]
                    };
                    BandRequest::Spectral2 {
                        expr: preset,
                        bands_1based: bs,
                        rescale,
                        colormap: pq.colormap,
                    }
                }
            });
        }
        // Arbitrary TiTiler expression -> compile to bytecode VM!
        let prog = ExprProgram::compile(raw_expr).map_err(EngineError::BadRequest)?;
        let rescale = pq.rescale.first().copied().unwrap_or((-1.0, 1.0));
        return Ok(BandRequest::CustomExpr {
            prog,
            rescale,
            colormap: pq.colormap,
        });
    }

    if pq.bidx.len() == 3 || (pq.bidx.is_empty() && default_bands >= 3) {
        let bs = if pq.bidx.len() == 3 {
            [pq.bidx[0], pq.bidx[1], pq.bidx[2]]
        } else {
            [1, 2, 3]
        };
        let (rmin, rmax) = match pq.rescale.len() {
            3 => (
                [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
            ),
            1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
            _ => {
                let (lo, hi) = default_rescale_for_dtype(sample_type, false);
                ([lo; 3], [hi; 3])
            }
        };
        Ok(BandRequest::Rgb {
            bands_1based: bs,
            rescale_min: rmin,
            rescale_max: rmax,
        })
    } else {
        let b = pq.bidx.first().copied().unwrap_or(1);
        let default_range = default_rescale_for_dtype(sample_type, true);
        let rescale = pq.rescale.first().copied().unwrap_or(default_range);
        Ok(BandRequest::Single {
            band_1based: b,
            rescale,
            colormap: pq.colormap,
        })
    }
}

/// Match TiTiler / rio-tiler resampling semantics: when the source CRS matches the target CRS,
/// `resampling` applies; when reprojecting across CRSes, `reproject` applies (both default to
/// `Nearest`).
fn resolve_resample(
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    target_is_4326: bool,
) -> ResampleMode {
    let Some(cog) = source_cog else {
        return pq.resample;
    };
    let same_crs = match cogzilla_proj::CrsModel::from_geo_keys(&cog.index.geo_keys) {
        Ok(cogzilla_proj::CrsModel::Epsg3857) => !target_is_4326,
        Ok(cogzilla_proj::CrsModel::Geographic) => target_is_4326,
        _ => false,
    };
    if same_crs { pq.resample } else { pq.reproject }
}

fn build_tile_request(
    tile: TileCoord,
    scale_opt: Option<u32>,
    ext: &str,
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_tiff::SampleType,
    is_crs84: bool,
    source_cog: Option<&cogzilla_io::OpenedCog>,
) -> Result<(TileRequest, &'static str), EngineError> {
    if tile.z > 30 {
        return Err(EngineError::BadRequest(format!(
            "zoom level z={} exceeds maximum supported zoom (30)",
            tile.z
        )));
    }
    let rows = 1u64 << (tile.z as u32);
    let cols = if is_crs84 { rows << 1 } else { rows };
    if (tile.x as u64) >= cols || (tile.y as u64) >= rows {
        return Err(EngineError::BadRequest(format!(
            "tile coordinate ({}, {}) out of range for z={} (expected x < {cols}, y < {rows})",
            tile.x, tile.y, tile.z
        )));
    }
    let eff_ext = if ext.is_empty() {
        pq.format_param.as_deref().unwrap_or("png")
    } else {
        ext
    };
    let (format, content_type) = parse_ext_format(eff_ext, pq.webp_quality, pq.jpeg_quality)?;
    let bands = build_band_request(pq, default_bands, sample_type)?;
    let tile_size = match scale_opt {
        Some(sc) if (1..=4).contains(&sc) => sc * 256,
        Some(other) => {
            return Err(EngineError::BadRequest(format!(
                "invalid tile scale @{other}x (expected 1..=4)"
            )));
        }
        None => pq.tile_size,
    };
    Ok((
        TileRequest {
            tile,
            tile_size,
            bands,
            resample: resolve_resample(pq, source_cog, is_crs84),
            format,
            unscale: pq.unscale,
        },
        content_type,
    ))
}

fn validate_mosaic_tile_params(pq: &ParsedQuery, tile_size: u32) -> Result<(), EngineError> {
    if tile_size != 256 && tile_size != 512 {
        return Err(EngineError::BadRequest(format!(
            "mosaic tile routes only support tile_size 256 or 512 (got {tile_size})"
        )));
    }
    if pq.nodata.is_some() {
        return Err(EngineError::BadRequest(
            "query parameter 'nodata' is not supported on mosaic tile routes".into(),
        ));
    }
    if pq.custom_colormap_json.is_some() {
        return Err(EngineError::BadRequest(
            "inline JSON 'colormap' is not supported on mosaic tile routes (use 'colormap_name')"
                .into(),
        ));
    }
    if pq.buffer > 0 {
        return Err(EngineError::BadRequest(
            "query parameter 'buffer' is not supported on mosaic tile routes".into(),
        ));
    }
    Ok(())
}

async fn render_mosaic_tile_styled(
    engine: &TileEngine,
    urls: &[String],
    mut treq: TileRequest,
    pq: &ParsedQuery,
) -> Result<Bytes, EngineError> {
    if (pq.color_formula.is_some() || !pq.return_mask)
        && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy)
    {
        let cf = pq
            .color_formula
            .as_deref()
            .map(ColorFormula::parse)
            .transpose()
            .map_err(EngineError::BadRequest)?;
        let orig_format = treq.format;
        treq.format = OutputFormat::RawRgba;
        let raw = engine.render_mosaic_tile(urls, &treq).await?;
        let dim = treq.tile_size as usize;
        let data = if let Some(cf) = cf {
            let mut buf = raw.to_vec();
            cf.apply_buffer_in_place(&mut buf, 4);
            encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
        } else {
            encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
        };
        Ok(data)
    } else {
        engine.render_mosaic_tile(urls, &treq).await
    }
}

fn world_crs84_tile_bbox(z: u8, x: u32, y: u32, buffer: u32, tile_size: u32) -> [f64; 4] {
    let cols = (1u64 << (z as u32 + 1)) as f64;
    let rows = (1u64 << (z as u32)) as f64;
    let lon_span = 360.0 / cols;
    let lat_span = 180.0 / rows;
    let pad_lon = lon_span * (buffer as f64 / tile_size as f64);
    let pad_lat = lat_span * (buffer as f64 / tile_size as f64);
    let min_lon = -180.0 + (x as f64) * lon_span - pad_lon;
    let max_lon = min_lon + lon_span + 2.0 * pad_lon;
    let max_lat = 90.0 - (y as f64) * lat_span + pad_lat;
    let min_lat = max_lat - lat_span - 2.0 * pad_lat;
    [min_lon, min_lat, max_lon, max_lat]
}

fn web_mercator_buffered_3857_bbox(z: u8, x: u32, y: u32, buffer: u32, tile_size: u32) -> [f64; 4] {
    let (min_x, min_y, max_x, max_y) = tile_bounds_3857(TileCoord { z, x, y });
    let pad_x = (max_x - min_x) * (buffer as f64 / tile_size as f64);
    let pad_y = (max_y - min_y) * (buffer as f64 / tile_size as f64);
    [min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y]
}

fn plain_resp(
    status: StatusCode,
    body: impl Into<Bytes>,
    content_type: &'static str,
) -> Response<Full<Bytes>> {
    let mut resp = Response::new(Full::new(body.into()));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    resp.headers_mut()
        .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    resp
}

async fn load_mosaicjson(
    engine: &TileEngine,
    uri: &str,
) -> Result<Arc<MosaicJsonCatalog>, EngineError> {
    cogzilla_io::check_uri_allowed(uri)?;
    if let Some(cached) = engine.mosaicjson_cache.get(uri) {
        return Ok(cached.clone());
    }
    let bytes = engine
        .store
        .fetch_uri_bytes(uri, MAX_POST_BODY_BYTES)
        .await?;
    let text = String::from_utf8_lossy(&bytes);
    let cat = Arc::new(MosaicJsonCatalog::from_json_str(&text).map_err(EngineError::BadRequest)?);
    if engine.mosaicjson_cache.len() >= 256 {
        engine.mosaicjson_cache.clear();
    }
    engine.mosaicjson_cache.insert(uri.to_string(), cat.clone());
    Ok(cat)
}

const MAX_POST_BODY_BYTES: usize = 16 * 1024 * 1024;

async fn read_limited_body<B>(body: B) -> Result<Bytes, EngineError>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    use bytes::Buf;
    let mut out = bytes::BytesMut::new();
    tokio::pin!(body);
    while let Some(frame_res) = body.frame().await {
        let frame =
            frame_res.map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?;
        if let Ok(mut data) = frame.into_data() {
            if out.len().saturating_add(data.remaining()) > MAX_POST_BODY_BYTES {
                return Err(EngineError::BadRequest(format!(
                    "request body exceeds {} MiB limit",
                    MAX_POST_BODY_BYTES / (1024 * 1024)
                )));
            }
            while data.has_remaining() {
                let chunk = data.chunk();
                out.extend_from_slice(chunk);
                let n = chunk.len();
                data.advance(n);
            }
        }
    }
    Ok(out.freeze())
}

async fn open_query_urls(
    engine: &TileEngine,
    urls: &[String],
) -> Result<Vec<Arc<cogzilla_io::OpenedCog>>, EngineError> {
    if urls.is_empty() {
        return Err(EngineError::BadRequest(
            "missing ?url=... query parameter".to_string(),
        ));
    }
    if urls.len() > 16 {
        return Err(EngineError::BadRequest(format!(
            "too many ?url= parameters ({}, maximum 16)",
            urls.len()
        )));
    }
    if urls.len() == 1 {
        return Ok(vec![engine.store.open_uri(&urls[0]).await?]);
    }
    let mut js = tokio::task::JoinSet::new();
    for (idx, u) in urls.iter().enumerate() {
        let store = engine.store.clone();
        let url = u.clone();
        js.spawn(async move { Ok::<_, cogzilla_io::IoError>((idx, store.open_uri(&url).await?)) });
    }
    let mut slots: Vec<Option<Arc<cogzilla_io::OpenedCog>>> = vec![None; urls.len()];
    while let Some(res) = js.join_next().await {
        let (idx, cog) = res.map_err(|e| {
            EngineError::Io(cogzilla_io::IoError::Fs(std::io::Error::other(format!(
                "open_uri task failed: {e}"
            ))))
        })??;
        slots[idx] = Some(cog);
    }
    Ok(slots.into_iter().map(|s| s.unwrap()).collect())
}

/// Convenience entry point for serverless / embedded runtimes (such as AWS Lambda)
/// that already hold the request body in memory as `Bytes`.
pub async fn handle_bytes_request(
    engine: Arc<TileEngine>,
    req: Request<Bytes>,
) -> Response<Bytes> {
    let (parts, body) = req.into_parts();
    let req_full = Request::from_parts(parts, Full::new(body));
    let Ok(resp) = handle_request(engine, req_full).await;
    let (rparts, rbody) = resp.into_parts();
    let bytes = rbody
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();
    Response::from_parts(rparts, bytes)
}

pub async fn handle_request<B>(
    engine: Arc<TileEngine>,
    req: Request<B>,
) -> Result<Response<Full<Bytes>>, Infallible>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let is_head = *req.method() == Method::HEAD;
    let is_post = *req.method() == Method::POST;
    let mut resp = handle_request_inner(engine, req, is_head).await?;
    if is_post {
        // RFC 9110: POST responses depend on the request body and must not carry public Cache-Control/ETag.
        resp.headers_mut().remove(CACHE_CONTROL);
        resp.headers_mut().remove(ETAG);
    }
    if is_head {
        let (mut parts, body) = resp.into_parts();
        if parts.status != StatusCode::NOT_MODIFIED && parts.status != StatusCode::NO_CONTENT {
            if let Some(len) = hyper::body::Body::size_hint(&body).exact() {
                if let Ok(hv) = HeaderValue::from_str(&len.to_string()) {
                    parts.headers.insert(hyper::header::CONTENT_LENGTH, hv);
                }
            }
        }
        return Ok(Response::from_parts(parts, Full::new(Bytes::new())));
    }
    Ok(resp)
}

async fn handle_request_inner<B>(
    engine: Arc<TileEngine>,
    req: Request<B>,
    is_head: bool,
) -> Result<Response<Full<Bytes>>, Infallible>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let proto = req
        .headers()
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim())
        .filter(|&s| s.eq_ignore_ascii_case("https") || s.eq_ignore_ascii_case("http"))
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "http".to_string());
    let host_raw = req
        .headers()
        .get("x-forwarded-host")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            req.headers()
                .get("host")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
        })
        .or_else(|| req.uri().authority().map(|a| a.as_str()))
        .unwrap_or("127.0.0.1:8080");
    let forwarded_prefix = req
        .headers()
        .get("x-forwarded-prefix")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().trim_end_matches('/'))
        .filter(|s| {
            !s.is_empty()
                && s.starts_with('/')
                && !s.contains("..")
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'-' | b'_' | b'.' | b'~'))
        })
        .map(|s| s.to_string())
        .unwrap_or_default();
    let host = format!("{proto}://{host_raw}{forwarded_prefix}");
    let if_none_match = req
        .headers()
        .get(IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let method = if is_head {
        Method::GET
    } else {
        req.method().clone()
    };
    let is_post = method == Method::POST;
    let if_none_match = if is_post { None } else { if_none_match };
    let path = req.uri().path().to_string();
    let raw_query = req.uri().query().unwrap_or("").to_string();

    if method == Method::OPTIONS {
        let mut resp = plain_resp(StatusCode::NO_CONTENT, Bytes::new(), "text/plain");
        resp.headers_mut().insert(
            hyper::header::ACCESS_CONTROL_ALLOW_METHODS,
            HeaderValue::from_static("GET, POST, HEAD, OPTIONS"),
        );
        resp.headers_mut().insert(
            hyper::header::ACCESS_CONTROL_ALLOW_HEADERS,
            HeaderValue::from_static("Content-Type, If-None-Match, Authorization"),
        );
        return Ok(resp);
    }

    if method == Method::GET && path == "/healthz" {
        return Ok(plain_resp(StatusCode::OK, Bytes::from_static(b"ok"), "text/plain"));
    }

    if method == Method::GET && (path == "/" || path == "/index.html") {
        let base_path_js = serde_json::to_string(&forwarded_prefix)
            .unwrap_or_else(|_| "\"\"".to_string())
            .replace("</", "<\\/");
        let html = include_str!("landing.html")
            .replace("{{VERSION}}", env!("CARGO_PKG_VERSION"))
            .replace("{{BASE_PATH}}", &forwarded_prefix)
            .replace("{{BASE_PATH_JS}}", &base_path_js);
        return Ok(plain_resp(StatusCode::OK, Bytes::from(html), "text/html; charset=utf-8"));
    }

    if method == Method::GET && path == "/openapi.json" {
        let spec = serde_json::json!({
            "openapi": "3.1.0",
            "info": {
                "title": "cogzilla",
                "version": env!("CARGO_PKG_VERSION"),
                "description": "Zero-GDAL Rust + SIMD drop-in TiTiler COG & STAC tile server"
            },
            "paths": {
                "/": { "get": { "summary": "Landing page with live demo maps" } },
                "/healthz": { "get": { "summary": "Health check" } },
                "/cog/info": { "get": { "summary": "COG metadata" } },
                "/cog/info.geojson": { "get": { "summary": "COG GeoJSON metadata" } },
                "/cog/bounds": { "get": { "summary": "COG WGS84 bounds" } },
                "/cog/statistics": { "get": { "summary": "COG statistics" }, "post": { "summary": "Polygon-clipped COG statistics" } },
                "/cog/tiles": { "get": { "summary": "List COG tilesets" } },
                "/cog/tiles/{tileMatrixSetId}": { "get": { "summary": "COG tileset metadata" } },
                "/cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}": { "get": { "summary": "Render COG map tile" } },
                "/cog/{tileMatrixSetId}/tilejson.json": { "get": { "summary": "COG TileJSON 3.0" } },
                "/cog/{tileMatrixSetId}/map.html": { "get": { "summary": "Interactive Leaflet map viewer" } },
                "/cog/point/{lon},{lat}": { "get": { "summary": "Sample pixel values at point" } },
                "/cog/preview.{format}": { "get": { "summary": "Render COG preview" } },
                "/cog/bbox/{minx},{miny},{maxx},{maxy}/{width}x{height}.{format}": { "get": { "summary": "Render COG bbox crop" } },
                "/cog/feature/{width}x{height}.{format}": { "post": { "summary": "Render GeoJSON feature crop" } },
                "/stac/assets": { "get": { "summary": "List STAC Item raster assets" } },
                "/stac/info": { "get": { "summary": "STAC assets metadata" } },
                "/stac/info.geojson": { "get": { "summary": "STAC GeoJSON metadata" } },
                "/stac/asset_statistics": { "get": { "summary": "Per-asset STAC statistics" } },
                "/stac/statistics": { "get": { "summary": "Merged STAC statistics" }, "post": { "summary": "Polygon-clipped STAC statistics" } },
                "/stac/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}": { "get": { "summary": "Render STAC multi-asset tile" } },
                "/stac/{tileMatrixSetId}/tilejson.json": { "get": { "summary": "STAC TileJSON 3.0" } },
                "/stac/point/{lon},{lat}": { "get": { "summary": "Sample STAC asset values at point" } },
                "/stac/preview/{width}x{height}.{format}": { "get": { "summary": "Render STAC preview" } },
                "/colorMaps": { "get": { "summary": "List all 211 rio-tiler colormaps" } },
                "/colorMaps/{colorMapId}": { "get": { "summary": "Get colormap RGBA LUT JSON or ?f=png colorbar" } },
                "/tileMatrixSets": { "get": { "summary": "List OGC TileMatrixSets" } },
                "/tileMatrixSets/{tileMatrixSetId}": { "get": { "summary": "Get OGC TileMatrixSet 2.0 definition" } }
            }
        })
        .to_string();
        return Ok(plain_resp(StatusCode::OK, Bytes::from(spec), "application/json"));
    }

    if method == Method::GET && path == "/docs" {
        let spec_url_js = serde_json::to_string(&format!("{forwarded_prefix}/openapi.json"))
            .unwrap_or_else(|_| "\"/openapi.json\"".to_string())
            .replace("</", "<\\/");
        let html = format!(
            r#"<!DOCTYPE html>
<html><head><title>cogzilla API Docs</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"/>
</head><body><div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({{url:{spec_url_js},dom_id:'#swagger-ui'}});</script>
</body></html>"#
        );
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from(html),
            "text/html; charset=utf-8",
        ));
    }

    if method == Method::GET && path == "/stats" {
        let disk = engine.store.disk_cache.as_ref().map(|dc| {
            let s = dc.stats();
            serde_json::json!({
                "hits": s.hits,
                "misses": s.misses,
                "bytes_used": s.bytes_used,
                "max_bytes": s.max_bytes,
                "evictions": s.evictions,
                "write_errors": s.write_errors,
                "read_errors": s.read_errors,
            })
        });
        let body = serde_json::json!({
            "tiles_served": engine.stats.tiles_served.load(Ordering::Relaxed),
            "empty_short_circuits": engine.stats.empty_short_circuits.load(Ordering::Relaxed),
            "passthrough_hits": engine.stats.passthrough_hits.load(Ordering::Relaxed),
            "http_304_not_modified": engine.stats.http_304_not_modified.load(Ordering::Relaxed),
            "cache_hits": engine.store.cache.hits.load(Ordering::Relaxed),
            "cache_misses": engine.store.cache.misses.load(Ordering::Relaxed),
            "singleflight_coalesced": engine.store.cache.coalesced_waiters.load(Ordering::Relaxed),
            "cache_bytes": engine.store.cache.current_bytes(),
            "tile_cache_hits": engine.tile_cache.hits.load(Ordering::Relaxed),
            "tile_cache_misses": engine.tile_cache.misses.load(Ordering::Relaxed),
            "tile_cache_coalesced": engine.tile_cache.coalesced.load(Ordering::Relaxed),
            "tile_cache_bytes": engine.tile_cache.current_bytes(),
            "upstream_limit": engine.store.upstream_limiter.current_limit(),
            "upstream_in_flight": engine.store.upstream_limiter.in_flight(),
            "upstream_throttles": engine.store.upstream_limiter.throttle_events(),
            "remote_range_requests": engine.store.metrics.remote_range_requests.load(Ordering::Relaxed),
            "remote_bytes_fetched": engine.store.metrics.remote_bytes_fetched.load(Ordering::Relaxed),
            "disk_cache": disk,
        })
        .to_string();
        return Ok(plain_resp(StatusCode::OK, Bytes::from(body), "application/json"));
    }

    if method == Method::GET && path == "/metrics" {
        let prom = render_prometheus_metrics(&engine);
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from(prom),
            "text/plain; version=0.0.4; charset=utf-8",
        ));
    }

    if method == Method::POST && path == "/admin/clear-cache" {
        engine.store.cache.clear();
        engine.tile_cache.clear();
        return Ok(plain_resp(StatusCode::OK, Bytes::from_static(b"cleared"), "text/plain"));
    }

    let pq = parse_query(if raw_query.is_empty() {
        None
    } else {
        Some(&raw_query)
    });

    let segs: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    if let Err(msg) = validate_query(&segs, &pq) {
        return Ok(plain_resp(StatusCode::BAD_REQUEST, Bytes::from(msg), "text/plain"));
    }

    // ------------------------------------------------------------------------
    // Dynamic registration of spatial catalogs (MosaicJSON, STAC FeatureCollection,
    // JSON array, or live STAC-API POST /search):
    //   POST /mosaic/register
    // ------------------------------------------------------------------------
    if method == Method::POST && path == "/mosaic/register" {
        let res: Result<(Bytes, &'static str), EngineError> = async {
            let body_bytes = read_limited_body(req.into_body()).await?;
            let spec: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid JSON body: {e}")))?;
            let name = spec
                .get("name")
                .and_then(|v| v.as_str())
                .filter(|s| {
                    !s.is_empty()
                        && *s != "tiles"
                        && *s != "register"
                        && !s.contains('/')
                        && !s.contains('?')
                        && !s.contains('#')
                        && !s.chars().any(|c| c.is_whitespace())
                })
                .ok_or_else(|| {
                    EngineError::BadRequest("missing valid 'name' for mosaic catalog".into())
                })?
                .to_string();

            let catalog: Arc<dyn SceneCatalog> = if let Some(cat_val) = spec.get("catalog") {
                load_catalog_json_str(&cat_val.to_string()).map_err(EngineError::BadRequest)?
            } else if let Some(search_url) = spec.get("stac_api_url").and_then(|v| v.as_str()) {
                cogzilla_io::check_uri_allowed(search_url)?;
                let mut search_body = serde_json::Map::new();
                for k in ["collections", "bbox", "datetime", "limit", "query", "filter"] {
                    if let Some(v) = spec.get(k) {
                        search_body.insert(k.to_string(), v.clone());
                    }
                }
                let resp = engine
                    .store
                    .client
                    .post(search_url)
                    .header("content-type", "application/json")
                    .body(serde_json::Value::Object(search_body).to_string())
                    .send()
                    .await
                    .map_err(|e| {
                        EngineError::BadRequest(format!(
                            "STAC API search failed: {}",
                            e.without_url()
                        ))
                    })?;
                if !resp.status().is_success() {
                    return Err(EngineError::BadRequest(format!(
                        "STAC API search returned HTTP {}",
                        resp.status()
                    )));
                }
                let body_bytes = cogzilla_io::read_response_capped(resp, MAX_POST_BODY_BYTES)
                    .await
                    .map_err(|e| {
                        EngineError::BadRequest(format!("STAC API response read failed: {e}"))
                    })?;
                let text = String::from_utf8_lossy(&body_bytes);
                load_catalog_json_str(&text).map_err(EngineError::BadRequest)?
            } else {
                return Err(EngineError::BadRequest(
                    "expected 'catalog' or 'stac_api_url' in /mosaic/register body".into(),
                ));
            };

            let count = catalog.len();
            if engine.catalog_count() >= 256 && engine.get_catalog(&name).is_none() {
                return Err(EngineError::BadRequest(
                    "maximum registered mosaic catalogs (256) reached".into(),
                ));
            }
            engine.register_catalog(name.clone(), catalog);
            engine.tile_cache.clear();
            let body = serde_json::json!({
                "name": name,
                "scenes": count,
                "tiles_url": format!("{forwarded_prefix}/mosaic/{name}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"),
            })
            .to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await;
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler MosaicJSON Router:
    //   GET /mosaicjson/info?url=...
    //   GET /mosaicjson/{tms}/tilejson.json?url=...
    //   GET /mosaicjson/tiles/{tms}/{z}/{x}/{y}[@scale][.{ext}]?url=...
    //   GET /mosaicjson/tiles/{z}/{x}/{y}[@scale][.{ext}]?url=...
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"mosaicjson") {
        let mj_url = pq.urls.first().cloned();
        let res: Result<(Bytes, &'static str), EngineError> = match segs.as_slice() {
            ["mosaicjson", "info"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                Ok((Bytes::from(mj.info_json().to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", tms, "tilejson.json"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let tile_url =
                    format!("{host}/mosaicjson/tiles/{tms}/{{z}}/{{x}}/{{y}}?{raw_query}");
                let tj = serde_json::json!({
                    "tilejson": "3.0.0",
                    "version": mj.version,
                    "scheme": "xyz",
                    "tiles": [tile_url],
                    "minzoom": mj.minzoom,
                    "maxzoom": mj.maxzoom,
                    "bounds": mj.bounds,
                })
                .to_string();
                Ok((Bytes::from(tj), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles", "WebMercatorQuad", zs, xs, ys]
            | ["mosaicjson", "tiles", zs, xs, ys] => {
                let cache_key = format!("{path}?{raw_query}");
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        let u = mj_url
                            .ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                        let mj = load_mosaicjson(&engine, &u).await?;
                        let (Some(z), Some(x), Some((y, sc, ext))) = (
                            zs.parse::<u8>().ok(),
                            xs.parse::<u32>().ok(),
                            parse_y_scale_ext(ys),
                        ) else {
                            return Err(EngineError::BadRequest(
                                "invalid /mosaicjson/tiles coordinate path".into(),
                            ));
                        };
                        if z > 30 {
                            return Err(EngineError::BadRequest(format!(
                                "invalid zoom level z={z} (expected 0..=30)"
                            )));
                        }
                        let tile = TileCoord { z, x, y };
                        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
                        let urls = mj.query_bbox_3857(min_x, min_y, max_x, max_y);
                        if urls.is_empty() {
                            let (treq, ct) = build_tile_request(
                                tile,
                                sc,
                                ext,
                                &pq,
                                3,
                                cogzilla_tiff::SampleType::Uint8,
                                false,
                                None,
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let nb = match treq.bands {
                                BandRequest::Rgb { .. } | BandRequest::SarRgb { .. } => 3,
                                _ => 1,
                            };
                            Ok((
                                cogzilla_kernel::empty_tile_bytes(treq.tile_size, treq.format, nb),
                                ct,
                            ))
                        } else {
                            let first_cog = engine.store.open_uri(&urls[0]).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let st = first_cog.index.full_res().sample_type;
                            let (treq, ct) = build_tile_request(
                                tile,
                                sc,
                                ext,
                                &pq,
                                db,
                                st,
                                false,
                                Some(&first_cog),
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let b = render_mosaic_tile_styled(&engine, &urls, treq, &pq).await?;
                            Ok((b, ct))
                        }
                    })
                    .await;
                return Ok(to_http_response_cached(&engine, cached, if_none_match.as_deref()));
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // Dynamic early-exit COG mosaic:
    //   GET /mosaic/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@scale][.{ext}]?url=a&url=b&...
    //   GET /mosaic/tiles/{z}/{x}/{y}[@scale][.{ext}]?url=a&url=b&...
    // Indexed spatial-catalog mosaic:
    //   GET /mosaic/{name}/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@scale][.{ext}]
    //   GET /mosaic/{name}/tiles/{z}/{x}/{y}[@scale][.{ext}]
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"mosaic") && segs.get(1) == Some(&"tiles") {
        let coords = match segs.as_slice() {
            ["mosaic", "tiles", "WebMercatorQuad", zs, xs, ys]
            | ["mosaic", "tiles", zs, xs, ys] => (
                zs.parse::<u8>().ok(),
                xs.parse::<u32>().ok(),
                parse_y_scale_ext(ys),
            ),
            _ => (None, None, None),
        };
        let cache_key = format!("{path}?{raw_query}");
        let res = match coords {
            (Some(z), Some(x), Some((y, sc, ext))) => {
                if z > 30 {
                    Err(EngineError::BadRequest(format!(
                        "invalid zoom level z={z} (expected 0..=30)"
                    )))
                } else if pq.urls.is_empty() {
                    Err(EngineError::BadRequest(
                        "missing required ?url= parameter(s) for /mosaic/tiles".into(),
                    ))
                } else if pq.urls.len() > 16 {
                    Err(EngineError::BadRequest(format!(
                        "too many ?url= parameters ({}; max 16)",
                        pq.urls.len()
                    )))
                } else {
                    let cached = engine
                        .tile_cache
                        .get_or_render(cache_key, || async {
                            let first_cog = engine.store.open_uri(&pq.urls[0]).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let st = first_cog.index.full_res().sample_type;
                            let (treq, ct) = build_tile_request(
                                TileCoord { z, x, y },
                                sc,
                                ext,
                                &pq,
                                db,
                                st,
                                false,
                                Some(&first_cog),
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let b = render_mosaic_tile_styled(&engine, &pq.urls, treq, &pq).await?;
                            Ok((b, ct))
                        })
                        .await;
                    return Ok(to_http_response_cached(&engine, cached, if_none_match.as_deref()));
                }
            }
            _ => Err(EngineError::BadRequest(
                "invalid /mosaic/tiles coordinate path".into(),
            )),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    if method == Method::GET && segs.first() == Some(&"mosaic") && segs.get(2) == Some(&"tiles") {
        let cat_name = segs[1];
        let coords = match segs.as_slice() {
            ["mosaic", _, "tiles", "WebMercatorQuad", zs, xs, ys]
            | ["mosaic", _, "tiles", zs, xs, ys] => (
                zs.parse::<u8>().ok(),
                xs.parse::<u32>().ok(),
                parse_y_scale_ext(ys),
            ),
            _ => (None, None, None),
        };
        let res = match (engine.get_catalog(cat_name), coords) {
            (None, _) => Err(EngineError::NotFound(format!("mosaic catalog '{cat_name}'"))),
            (Some(catalog), (Some(z), Some(x), Some((y, sc, ext)))) => {
                let cache_key = format!("{path}?{raw_query}");
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        if z > 30 {
                            return Err(EngineError::BadRequest(format!(
                                "invalid zoom level z={z} (expected 0..=30)"
                            )));
                        }
                        let tile = TileCoord { z, x, y };
                        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
                        let urls = catalog.query_bbox_3857(min_x, min_y, max_x, max_y);
                        if urls.is_empty() {
                            let (treq, ct) = build_tile_request(
                                tile,
                                sc,
                                ext,
                                &pq,
                                3,
                                cogzilla_tiff::SampleType::Uint8,
                                false,
                                None,
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let nb = match treq.bands {
                                BandRequest::Rgb { .. } | BandRequest::SarRgb { .. } => 3,
                                _ => 1,
                            };
                            Ok((
                                cogzilla_kernel::empty_tile_bytes(treq.tile_size, treq.format, nb),
                                ct,
                            ))
                        } else {
                            let first_cog = engine.store.open_uri(&urls[0]).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let st = first_cog.index.full_res().sample_type;
                            let (treq, ct) = build_tile_request(
                                tile,
                                sc,
                                ext,
                                &pq,
                                db,
                                st,
                                false,
                                Some(&first_cog),
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let b = render_mosaic_tile_styled(&engine, &urls, treq, &pq).await?;
                            Ok((b, ct))
                        }
                    })
                    .await;
                return Ok(to_http_response_cached(&engine, cached, if_none_match.as_deref()));
            }
            _ => Err(EngineError::BadRequest(
                "invalid /mosaic/{name}/tiles coordinate path".into(),
            )),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // Pre-registered fast path: GET /tiles/{dataset_or_scene}/{z}/{x}/{y}[@scale][.{ext}]
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"tiles") && segs.len() == 5 {
        let ds = segs[1];
        let parsed = (
            segs[2].parse::<u8>().ok(),
            segs[3].parse::<u32>().ok(),
            parse_y_scale_ext(segs[4]),
        );
        let cache_key = format!("{path}?{raw_query}");
        let res = match parsed {
            (Some(z), Some(x), Some((y, sc, ext))) => {
                if z > 30 {
                    Err(EngineError::BadRequest(format!(
                        "invalid zoom level z={z} (expected 0..=30)"
                    )))
                } else {
                    let cached = engine
                        .tile_cache
                        .get_or_render(cache_key, || async {
                            let cogs: Vec<Arc<cogzilla_io::OpenedCog>> =
                                if let Some(scene) = engine.get_scene(ds) {
                                    scene
                                } else if let Some(cog) = engine.store.get_registered(ds) {
                                    vec![cog]
                                } else {
                                    return Err(EngineError::NotFound(ds.to_string()));
                                };
                            let db = if cogs.len() > 1 {
                                cogs.len() as u16
                            } else {
                                cogs[0].index.full_res().samples_per_pixel
                            };
                            let st = cogs[0].index.full_res().sample_type;
                            let (mut treq, ct) = build_tile_request(
                                TileCoord { z, x, y },
                                sc,
                                ext,
                                &pq,
                                db,
                                st,
                                false,
                                Some(&cogs[0]),
                            )?;
                            if pq.unscale {
                                apply_unscale_cogs(&mut treq.bands, &cogs);
                            }
                            let needs_general_path = pq.nodata.is_some()
                                || pq.custom_colormap_json.is_some()
                                || pq.buffer > 0
                                || (treq.tile_size != 256 && treq.tile_size != 512);
                            if needs_general_path {
                                let custom_lut = pq
                                    .custom_colormap_json
                                    .as_deref()
                                    .map(ColormapLut::from_json_str)
                                    .transpose()
                                    .map_err(EngineError::BadRequest)?;
                                let cf = pq
                                    .color_formula
                                    .as_deref()
                                    .map(ColorFormula::parse)
                                    .transpose()
                                    .map_err(EngineError::BadRequest)?;
                                let out_dim = treq.tile_size + 2 * pq.buffer;
                                let bbox_3857 = web_mercator_buffered_3857_bbox(
                                    z,
                                    x,
                                    y,
                                    pq.buffer,
                                    treq.tile_size,
                                );
                                let data = render_mercator_bbox(
                                    &engine,
                                    &cogs,
                                    bbox_3857,
                                    out_dim,
                                    out_dim,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq.return_mask,
                                    pq.unscale,
                                )
                                .await?;
                                Ok((data, ct))
                            } else if (pq.color_formula.is_some() || !pq.return_mask)
                                && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy)
                            {
                                let cf = pq
                                    .color_formula
                                    .as_deref()
                                    .map(ColorFormula::parse)
                                    .transpose()
                                    .map_err(EngineError::BadRequest)?;
                                let orig_format = treq.format;
                                treq.format = OutputFormat::RawRgba;
                                let raw = engine.render_multi_cog_tile(&cogs, &treq).await?;
                                let dim = treq.tile_size as usize;
                                let data = if let Some(cf) = cf {
                                    let mut buf = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut buf, 4);
                                    encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
                                } else {
                                    encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
                                };
                                Ok((data, ct))
                            } else {
                                let b = engine.render_multi_cog_tile(&cogs, &treq).await?;
                                Ok((b, ct))
                            }
                        })
                        .await;
                    return Ok(to_http_response_cached(&engine, cached, if_none_match.as_deref()));
                }
            }
            _ => Err(EngineError::BadRequest(
                "expected /tiles/{dataset}/{z}/{x}/{y}.{png|webp|f32}".to_string(),
            )),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler ColorMaps Router: GET /colorMaps, GET /colorMaps/{colorMapId}
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"colorMaps") {
        let res = match segs.as_slice() {
            ["colorMaps"] => Ok((
                Bytes::from(list_colormaps_json(&host).to_string()),
                "application/json",
            )),
            ["colorMaps", id] => match pq.format_param.as_deref() {
                None | Some("json") => {
                    get_colormap_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
                }
                Some("png") => get_colormap_image(id, OutputFormat::Png).map(|b| (b, "image/png")),
                Some("jpg" | "jpeg") => {
                    get_colormap_image(id, OutputFormat::Jpeg { quality: pq.jpeg_quality })
                        .map(|b| (b, "image/jpeg"))
                }
                Some("webp") => {
                    get_colormap_image(id, OutputFormat::Webp { quality: pq.webp_quality })
                        .map(|b| (b, "image/webp"))
                }
                Some(other) => Err(EngineError::BadRequest(format!(
                    "unsupported colormap format '{other}' (use json, png, jpg, or webp)"
                ))),
            },
            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler OGC TileMatrixSets Router: /tileMatrixSets, /tileMatrixSets/{id}
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"tileMatrixSets") {
        let res = match segs.as_slice() {
            ["tileMatrixSets"] => Ok((
                Bytes::from(list_tms_json(&host).to_string()),
                "application/json",
            )),
            ["tileMatrixSets", id] => {
                get_tms_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // Full TiTiler STAC Item Router: /stac/* (MultiBaseTilerFactory compatible!)
    // ------------------------------------------------------------------------
    if segs.first() == Some(&"stac") {
        let sub = &segs[1..];
        let stac_url = pq.urls.first().cloned();
        let res: Result<(Bytes, &'static str), EngineError> = match (method.clone(), sub) {
            // GET /stac/assets?url=item.json
            (Method::GET, ["assets"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let keys: Vec<&String> = item.assets.keys().collect();
                Ok((Bytes::from(serde_json::json!(keys).to_string()), "application/json"))
            }
            .await,

            // GET /stac/info?url=item.json&assets=...
            (Method::GET, ["info"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let mut map = serde_json::Map::new();
                for (name, cog) in names.into_iter().zip(cogs.into_iter()) {
                    map.insert(name, build_cog_info(&engine, &cog)?);
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,

            // GET /stac/info.geojson?url=item.json&assets=...
            (Method::GET, ["info.geojson"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let mut props = serde_json::Map::new();
                for (name, cog) in names.into_iter().zip(cogs.into_iter()) {
                    props.insert(name, build_cog_info(&engine, &cog)?);
                }
                let geom = item.geometry.clone().unwrap_or_else(|| {
                    let b = item.bbox.unwrap_or([-180.0, -90.0, 180.0, 90.0]);
                    serde_json::json!({
                        "type": "Polygon",
                        "coordinates": [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]]
                    })
                });
                let feat = serde_json::json!({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props,
                });
                Ok((Bytes::from(feat.to_string()), "application/geo+json"))
            }
            .await,

            // GET /stac/asset_statistics (per-asset dict: {asset: {b1: ...}})
            (Method::GET, ["asset_statistics"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let n = names.len();
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    set.spawn(async move {
                        let st =
                            compute_statistics(&eng, &cog, max_size, None, unscale, nodata).await?;
                        Ok::<_, EngineError>((idx, st))
                    });
                }
                let mut ordered: Vec<Option<serde_json::Value>> = vec![None; n];
                while let Some(res) = set.join_next().await {
                    let (idx, st) = res.map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                    ordered[idx] = Some(st);
                }
                let mut map = serde_json::Map::new();
                for (name, st) in names.into_iter().zip(ordered.into_iter()) {
                    if let Some(st) = st {
                        map.insert(name, st);
                    }
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,
            // GET /stac/statistics (merged band dict: {b1: ..., b2: ...})
            (Method::GET, ["statistics"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (_, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let n = cogs.len();
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    set.spawn(async move {
                        let nb = cog.index.full_res().samples_per_pixel as usize;
                        let st =
                            compute_statistics(&eng, &cog, max_size, None, unscale, nodata).await?;
                        Ok::<_, EngineError>((idx, nb, st))
                    });
                }
                let mut ordered: Vec<Option<(usize, serde_json::Value)>> = vec![None; n];
                while let Some(res) = set.join_next().await {
                    let (idx, nb, st) = res.map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                    ordered[idx] = Some((nb, st));
                }
                let mut map = serde_json::Map::new();
                let mut band_idx = 1usize;
                for (nb, st) in ordered.into_iter().flatten() {
                    if let Some(obj) = st.as_object() {
                        for i in 1..=nb {
                            if let Some(v) = obj.get(&format!("b{i}")) {
                                map.insert(format!("b{band_idx}"), v.clone());
                                band_idx += 1;
                            }
                        }
                    }
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,
            (Method::POST, ["statistics"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;

                let compute_stac_merged_stats = |mask: Arc<PolygonMask>| {
                    let cogs_c = cogs.clone();
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let n = names.len();
                    async move {
                        let mut set = tokio::task::JoinSet::new();
                        for (idx, cog) in cogs_c.into_iter().enumerate() {
                            let eng_c = eng.clone();
                            let mask_c = mask.clone();
                            set.spawn(async move {
                                let st = compute_statistics(
                                    &eng_c,
                                    &cog,
                                    max_size,
                                    Some(&mask_c),
                                    unscale,
                                    nodata,
                                )
                                .await?;
                                Ok::<_, EngineError>((idx, st))
                            });
                        }
                        let mut ordered: Vec<Option<serde_json::Value>> = vec![None; n];
                        while let Some(res) = set.join_next().await {
                            let (idx, st) = res
                                .map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                            ordered[idx] = Some(st);
                        }
                        // Merge bands across assets into b1..bN to match TiTiler's POST /stac/statistics.
                        let mut merged = serde_json::Map::new();
                        let mut b_counter = 1usize;
                        for st in ordered.into_iter().flatten() {
                            if let Some(obj) = st.as_object() {
                                for (_k, v) in obj {
                                    merged.insert(format!("b{b_counter}"), v.clone());
                                    b_counter += 1;
                                }
                            }
                        }
                        Ok::<_, EngineError>(serde_json::Value::Object(merged))
                    }
                };

                if geojson.get("type").and_then(|v| v.as_str()) == Some("FeatureCollection") {
                    let features = geojson
                        .get_mut("features")
                        .and_then(|v| v.as_array_mut())
                        .ok_or_else(|| {
                            EngineError::BadRequest("FeatureCollection missing 'features' array".into())
                        })?;
                    if features.len() > 64 {
                        return Err(EngineError::BadRequest(format!(
                            "FeatureCollection exceeds max 64 features (got {})",
                            features.len()
                        )));
                    }
                    for feat in features.iter_mut() {
                        let (mask, _) = PolygonMask::from_geojson(feat)?;
                        let stats = compute_stac_merged_stats(Arc::new(mask)).await?;
                        if let Some(props) =
                            feat.get_mut("properties").and_then(|v| v.as_object_mut())
                        {
                            props.insert("statistics".into(), stats);
                        } else if let Some(obj) = feat.as_object_mut() {
                            obj.insert(
                                "properties".into(),
                                serde_json::json!({"statistics": stats}),
                            );
                        }
                    }
                    Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                } else {
                    let (mask, _) = PolygonMask::from_geojson(&geojson)?;
                    let stats = compute_stac_merged_stats(Arc::new(mask)).await?;
                    if geojson.get("type").and_then(|v| v.as_str()) == Some("Feature") {
                        if let Some(props) =
                            geojson.get_mut("properties").and_then(|v| v.as_object_mut())
                        {
                            props.insert("statistics".into(), stats);
                        } else if let Some(obj) = geojson.as_object_mut() {
                            obj.insert(
                                "properties".into(),
                                serde_json::json!({"statistics": stats}),
                            );
                        }
                        Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                    } else {
                        let feat = serde_json::json!({
                            "type": "Feature",
                            "geometry": geojson,
                            "properties": { "statistics": stats },
                        });
                        Ok((Bytes::from(feat.to_string()), "application/geo+json"))
                    }
                }
            }
            .await,

            // GET /stac/tiles & /stac/tiles/{tms}
            (Method::GET, ["tiles"]) => Ok((
                Bytes::from(list_tms_json(&host).to_string()),
                "application/json",
            )),
            (Method::GET, ["tiles", tms]) => {
                get_tms_json(tms).map(|v| (Bytes::from(v.to_string()), "application/json"))
            }

            // GET /stac/{tms}/tilejson.json & /stac/{tms}/map.html
            (Method::GET, [tms, "tilejson.json"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (_, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut tj = build_tilejson(&engine, &cogs[0], &host, tms, &raw_query)?;
                apply_tilejson_overrides(&mut tj, &pq, &host, "stac", tms, &raw_query);
                tj["name"] = serde_json::json!(item.id);
                Ok((Bytes::from(tj.to_string()), "application/json"))
            }
            .await,
            (Method::GET, [tms, "map.html"]) => {
                let tj_url = format!("{forwarded_prefix}/stac/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }

            // GET /stac/tiles/{tms}/{z}/{x}/{y}[@scale][.{format}]
            // and alias GET /stac/tiles/{z}/{x}/{y}[@scale][.{format}] (implicit WebMercatorQuad)
            (Method::GET, ["tiles", _, zs, xs, ys]) | (Method::GET, ["tiles", zs, xs, ys]) => {
                let tms = if sub.len() == 5 { sub[1] } else { "WebMercatorQuad" };
                let cache_key = format!("{path}?{raw_query}");
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        let u = stac_url
                            .ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                        let item =
                            load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                        let (_, cogs, rewritten_expr) = resolve_stac_assets(
                            &engine,
                            &item,
                            &pq.assets,
                            pq.expr_raw.as_deref(),
                        )
                        .await?;
                        let (Some(z), Some(x), Some((y, sc, ext))) = (
                            zs.parse::<u8>().ok(),
                            xs.parse::<u32>().ok(),
                            parse_y_scale_ext(ys),
                        ) else {
                            return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                        };
                        let mut pq_stac = pq;
                        pq_stac.expr_raw = rewritten_expr;
                        let db = if cogs.len() > 1 {
                            cogs.len() as u16
                        } else {
                            cogs[0].index.full_res().samples_per_pixel
                        };
                        let st = cogs[0].index.full_res().sample_type;
                        let (mut treq, ct) = build_tile_request(
                            TileCoord { z, x, y },
                            sc,
                            ext,
                            &pq_stac,
                            db,
                            st,
                            tms == "WorldCRS84Quad",
                            Some(&cogs[0]),
                        )?;
                        if pq_stac.unscale {
                            apply_unscale_cogs(&mut treq.bands, &cogs);
                        }
                        let needs_general = tms == "WorldCRS84Quad"
                            || pq_stac.buffer > 0
                            || pq_stac.nodata.is_some()
                            || pq_stac.custom_colormap_json.is_some()
                            || (treq.tile_size != 256 && treq.tile_size != 512);
                        if needs_general {
                            let custom_lut = pq_stac
                                .custom_colormap_json
                                .as_deref()
                                .map(ColormapLut::from_json_str)
                                .transpose()
                                .map_err(EngineError::BadRequest)?;
                            let cf = pq_stac
                                .color_formula
                                .as_deref()
                                .map(ColorFormula::parse)
                                .transpose()
                                .map_err(EngineError::BadRequest)?;
                            let out_dim = treq.tile_size + 2 * pq_stac.buffer;
                            let data = if tms == "WorldCRS84Quad" {
                                let bbox = world_crs84_tile_bbox(
                                    z,
                                    x,
                                    y,
                                    pq_stac.buffer,
                                    treq.tile_size,
                                );
                                render_wgs84_bbox(
                                    &engine,
                                    &cogs,
                                    bbox,
                                    out_dim,
                                    out_dim,
                                    None,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq_stac.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq_stac.return_mask,
                                    pq_stac.unscale,
                                )
                                .await?
                            } else {
                                let bbox_3857 = web_mercator_buffered_3857_bbox(
                                    z,
                                    x,
                                    y,
                                    pq_stac.buffer,
                                    treq.tile_size,
                                );
                                render_mercator_bbox(
                                    &engine,
                                    &cogs,
                                    bbox_3857,
                                    out_dim,
                                    out_dim,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq_stac.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq_stac.return_mask,
                                    pq_stac.unscale,
                                )
                                .await?
                            };
                            Ok((data, ct))
                        } else if (pq_stac.color_formula.is_some() || !pq_stac.return_mask)
                            && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy)
                        {
                            let cf = pq_stac
                                .color_formula
                                .as_deref()
                                .map(ColorFormula::parse)
                                .transpose()
                                .map_err(EngineError::BadRequest)?;
                            let orig_format = treq.format;
                            treq.format = OutputFormat::RawRgba;
                            let raw = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            let dim = treq.tile_size as usize;
                            let data = if let Some(cf) = cf {
                                let mut buf = raw.to_vec();
                                cf.apply_buffer_in_place(&mut buf, 4);
                                encode_rgba_output(dim, dim, &buf, orig_format, pq_stac.return_mask)
                            } else {
                                encode_rgba_output(dim, dim, &raw, orig_format, pq_stac.return_mask)
                            };
                            Ok((data, ct))
                        } else {
                            let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            Ok((data, ct))
                        }
                    })
                    .await;
                return Ok(to_http_response_cached(&engine, cached, if_none_match.as_deref()));
            }

            // GET /stac/point/{lon},{lat}
            (Method::GET, ["point", coord]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let Some((lons, lats)) = coord.split_once(',') else {
                    return Err(EngineError::BadRequest("expected /stac/point/{lon},{lat}".into()));
                };
                let (Ok(x), Ok(y)) = (lons.parse::<f64>(), lats.parse::<f64>()) else {
                    return Err(EngineError::BadRequest("invalid lon,lat floats".into()));
                };
                let (lon, lat) = if pq.coord_crs_3857 {
                    mercator_to_lonlat_deg(x, y)
                } else {
                    (x, y)
                };
                let n = names.len();
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    set.spawn(async move {
                        let pt = query_point(&eng, &cog, lon, lat, unscale, nodata).await?;
                        Ok::<_, EngineError>((idx, pt))
                    });
                }
                let mut ordered: Vec<Option<serde_json::Value>> = vec![None; n];
                while let Some(res) = set.join_next().await {
                    let (idx, pt) = res.map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                    ordered[idx] = Some(pt);
                }
                let mut all_vals = Vec::new();
                let mut all_names = Vec::new();
                for (name, pt) in names.into_iter().zip(ordered.into_iter()) {
                    if let Some(vs) = pt.as_ref().and_then(|p| p["values"].as_array()) {
                        for (i, v) in vs.iter().enumerate() {
                            all_vals.push(v.clone());
                            all_names.push(format!("{name}_b{}", i + 1));
                        }
                    }
                }
                let body = serde_json::json!({
                    "coordinates": [x, y],
                    "values": all_vals,
                    "band_names": all_names,
                    "band_descriptions": all_names,
                })
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // GET /stac/preview[/{w}x{h}][.{ext}]
            (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
                let ext_opt = p.strip_prefix("preview.");
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let scale = (pq_stac.max_size as f64) / (full.width.max(full.height) as f64);
                let out_w = ((full.width as f64 * scale.min(1.0)).round() as u32).max(1);
                let out_h = ((full.height as f64 * scale.min(1.0)).round() as u32).max(1);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq_stac, db, st)?;
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq_stac
                    .custom_colormap_json
                    .as_deref()
                    .map(ColormapLut::from_json_str)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["preview", dims_ext]) => async {
                let Some((out_w, out_h, ext)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected /stac/preview/{w}x{h}.{ext}".into()));
                };
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq_stac, db, st)?;
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq_stac
                    .custom_colormap_json
                    .as_deref()
                    .map(ColormapLut::from_json_str)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,

            // GET /stac/bbox/{minx},{miny},{maxx},{maxy}[/{w}x{h}][.{ext}]
            (Method::GET, ["bbox", bbox_seg]) => async {
                let Some((bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let (fmt, ct) = parse_ext_format(
                    ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png"),
                    pq_stac.webp_quality,
                    pq_stac.jpeg_quality,
                )?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq_stac, db, st)?;
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq_stac
                    .custom_colormap_json
                    .as_deref()
                    .map(ColormapLut::from_json_str)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let (out_w, out_h) = aspect_preserving_dims(bbox, pq_stac.max_size.min(1024));
                let data = if pq_stac.coord_crs_3857 {
                    let resample = resolve_resample(&pq_stac, Some(&cogs[0]), false);
                    render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?
                } else {
                    let resample = resolve_resample(&pq_stac, Some(&cogs[0]), true);
                    render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?
                };
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["bbox", bbox_seg, dims_ext]) => async {
                let Some((bbox, _)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let Some((out_w, out_h, ext)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected {w}x{h}.{ext}".into()));
                };
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq_stac, db, st)?;
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq_stac
                    .custom_colormap_json
                    .as_deref()
                    .map(ColormapLut::from_json_str)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let data = if pq_stac.coord_crs_3857 {
                    let resample = resolve_resample(&pq_stac, Some(&cogs[0]), false);
                    render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?
                } else {
                    let resample = resolve_resample(&pq_stac, Some(&cogs[0]), true);
                    render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?
                };
                Ok((data, ct))
            }
            .await,

            // POST /stac/feature[.{ext}] and POST /stac/feature/{w}x{h}.{ext}
            (Method::POST, ["feature"]) | (Method::POST, ["feature", _]) => async {
                let (out_w, out_h, ext) = if sub.len() == 2 {
                    parse_wxh_ext(sub[1]).ok_or_else(|| {
                        EngineError::BadRequest("expected /stac/feature/{w}x{h}.{ext}".into())
                    })?
                } else {
                    let s = pq.max_size.min(1024);
                    (s, s, "png")
                };
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson(&geojson)?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq_stac, db, st)?;
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq_stac
                    .custom_colormap_json
                    .as_deref()
                    .map(ColormapLut::from_json_str)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let resample = resolve_resample(&pq_stac, Some(&cogs[0]), true);
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, Some(&mask), &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::POST, [feat_ext])
                if *feat_ext == "feature" || feat_ext.starts_with("feature.") =>
            async {
                let ext_opt = feat_ext.strip_prefix("feature.");
                let s = pq.max_size.min(1024);
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson(&geojson)?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq_stac, db, st)?;
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq_stac
                    .custom_colormap_json
                    .as_deref()
                    .map(ColormapLut::from_json_str)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let (out_w, out_h) = aspect_preserving_dims(bbox, s);
                let resample = resolve_resample(&pq_stac, Some(&cogs[0]), true);
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, Some(&mask), &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,

            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // Full TiTiler Drop-In API: /cog/* (all 19 TiTiler TilerFactory routes!)
    // ------------------------------------------------------------------------
    if segs.first() == Some(&"cog") {
        let sub = &segs[1..];
        let res: Result<(Bytes, &'static str), EngineError> = match (method.clone(), sub) {
            // 1. GET /cog/info
            (Method::GET, ["info"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let json = build_cog_info(&engine, &cogs[0])?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 2. GET /cog/info.geojson
            (Method::GET, ["info.geojson"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let json = build_cog_info_geojson(&engine, &cogs[0])?;
                Ok((Bytes::from(json.to_string()), "application/geo+json"))
            }
            .await,

            // Alias: GET /cog/bounds
            (Method::GET, ["bounds"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bounds = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let body = serde_json::json!({"bounds": bounds}).to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // 3. GET / POST /cog/statistics (POST supports GeoJSON Polygon clipping!)
            (Method::GET, ["statistics"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let json = compute_statistics(
                    &engine,
                    &cogs[0],
                    pq.max_size,
                    None,
                    pq.unscale,
                    pq.nodata,
                )
                .await?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,
            (Method::POST, ["statistics"]) => async {
                let body_bytes = read_limited_body(req.into_body()).await?;
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                if body_bytes.is_empty() {
                    let json = compute_statistics(
                        &engine,
                        &cogs[0],
                        pq.max_size,
                        None,
                        pq.unscale,
                        pq.nodata,
                    )
                    .await?;
                    Ok((Bytes::from(json.to_string()), "application/json"))
                } else {
                    let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                        .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                    if geojson.get("type").and_then(|v| v.as_str()) == Some("FeatureCollection") {
                        let features = geojson
                            .get_mut("features")
                            .and_then(|v| v.as_array_mut())
                            .ok_or_else(|| {
                                EngineError::BadRequest(
                                    "FeatureCollection missing 'features' array".into(),
                                )
                            })?;
                        if features.len() > 64 {
                            return Err(EngineError::BadRequest(format!(
                                "FeatureCollection exceeds max 64 features (got {})",
                                features.len()
                            )));
                        }
                        for feat in features.iter_mut() {
                            let (mask, _) = PolygonMask::from_geojson(feat)?;
                            let stats = compute_statistics(
                                &engine,
                                &cogs[0],
                                pq.max_size,
                                Some(&mask),
                                pq.unscale,
                                pq.nodata,
                            )
                            .await?;
                            if let Some(props) =
                                feat.get_mut("properties").and_then(|v| v.as_object_mut())
                            {
                                props.insert("statistics".into(), stats);
                            } else if let Some(obj) = feat.as_object_mut() {
                                obj.insert(
                                    "properties".into(),
                                    serde_json::json!({"statistics": stats}),
                                );
                            }
                        }
                        Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                    } else {
                        let (mask, _) = PolygonMask::from_geojson(&geojson)?;
                        let stats = compute_statistics(
                            &engine,
                            &cogs[0],
                            pq.max_size,
                            Some(&mask),
                            pq.unscale,
                            pq.nodata,
                        )
                        .await?;
                        if geojson.get("type").and_then(|v| v.as_str()) == Some("Feature") {
                            if let Some(props) =
                                geojson.get_mut("properties").and_then(|v| v.as_object_mut())
                            {
                                props.insert("statistics".into(), stats);
                            } else if let Some(obj) = geojson.as_object_mut() {
                                obj.insert(
                                    "properties".into(),
                                    serde_json::json!({"statistics": stats}),
                                );
                            }
                            Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                        } else {
                            let feat = serde_json::json!({
                                "type": "Feature",
                                "geometry": geojson,
                                "properties": { "statistics": stats },
                            });
                            Ok((Bytes::from(feat.to_string()), "application/geo+json"))
                        }
                    }
                }
            }
            .await,

            // 4. GET /cog/tiles (OGC Tiles list)
            (Method::GET, ["tiles"]) => {
                let body = serde_json::json!({
                    "tilesets": [
                        {
                            "title": "WebMercatorQuad",
                            "id": "WebMercatorQuad",
                            "crs": "http://www.opengis.net/def/crs/EPSG/0/3857"
                        },
                        {
                            "title": "WorldCRS84Quad",
                            "id": "WorldCRS84Quad",
                            "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                        }
                    ]
                })
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }

            // 5. GET /cog/tiles/{tileMatrixSetId}
            (Method::GET, ["tiles", tms]) => {
                let crs_uri = if *tms == "WorldCRS84Quad" {
                    "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                } else {
                    "http://www.opengis.net/def/crs/EPSG/0/3857"
                };
                let body = serde_json::json!({
                    "id": tms,
                    "title": tms,
                    "crs": crs_uri,
                    "dataType": "map",
                })
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }

            // 6 & 7. GET /cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@scale][.{format}]
            //        and alias GET /cog/tiles/{z}/{x}/{y}[@scale][.{format}] (implicit WebMercatorQuad)
            (Method::GET, ["tiles", _, zs, xs, ys]) | (Method::GET, ["tiles", zs, xs, ys]) => {
                let tms = if sub.len() == 5 { sub[1] } else { "WebMercatorQuad" };
                let cache_key = format!("{path}?{raw_query}");
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        let cogs = open_query_urls(&engine, &pq.urls).await?;
                        let (Some(z), Some(x), Some((y, sc, ext))) = (
                            zs.parse::<u8>().ok(),
                            xs.parse::<u32>().ok(),
                            parse_y_scale_ext(ys),
                        ) else {
                            return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                        };
                        let db = if cogs.len() > 1 {
                            cogs.len() as u16
                        } else {
                            cogs[0].index.full_res().samples_per_pixel
                        };
                        let st = cogs[0].index.full_res().sample_type;
                        let (mut treq, ct) = build_tile_request(
                            TileCoord { z, x, y },
                            sc,
                            ext,
                            &pq,
                            db,
                            st,
                            tms == "WorldCRS84Quad",
                            Some(&cogs[0]),
                        )?;
                        if pq.unscale {
                            apply_unscale_cogs(&mut treq.bands, &cogs);
                        }

                        let needs_general_path = tms == "WorldCRS84Quad"
                            || pq.buffer > 0
                            || pq.nodata.is_some()
                            || pq.custom_colormap_json.is_some()
                            || (treq.tile_size != 256 && treq.tile_size != 512);

                        if needs_general_path {
                            let custom_lut = pq
                                .custom_colormap_json
                                .as_deref()
                                .map(ColormapLut::from_json_str)
                                .transpose()
                                .map_err(EngineError::BadRequest)?;
                            let cf = pq
                                .color_formula
                                .as_deref()
                                .map(ColorFormula::parse)
                                .transpose()
                                .map_err(EngineError::BadRequest)?;
                            let out_dim = treq.tile_size + 2 * pq.buffer;
                            let data = if tms == "WorldCRS84Quad" {
                                let bbox =
                                    world_crs84_tile_bbox(z, x, y, pq.buffer, treq.tile_size);
                                render_wgs84_bbox(
                                    &engine,
                                    &cogs,
                                    bbox,
                                    out_dim,
                                    out_dim,
                                    None,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq.return_mask,
                                    pq.unscale,
                                )
                                .await?
                            } else {
                                let bbox_3857 = web_mercator_buffered_3857_bbox(
                                    z,
                                    x,
                                    y,
                                    pq.buffer,
                                    treq.tile_size,
                                );
                                render_mercator_bbox(
                                    &engine,
                                    &cogs,
                                    bbox_3857,
                                    out_dim,
                                    out_dim,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq.return_mask,
                                    pq.unscale,
                                )
                                .await?
                            };
                            Ok((data, ct))
                        } else if (pq.color_formula.is_some() || !pq.return_mask)
                            && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy)
                        {
                            let cf = pq
                                .color_formula
                                .as_deref()
                                .map(ColorFormula::parse)
                                .transpose()
                                .map_err(EngineError::BadRequest)?;
                            let orig_format = treq.format;
                            treq.format = OutputFormat::RawRgba;
                            let raw = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            let dim = treq.tile_size as usize;
                            let data = if let Some(cf) = cf {
                                let mut buf = raw.to_vec();
                                cf.apply_buffer_in_place(&mut buf, 4);
                                encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
                            } else {
                                encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
                            };
                            Ok((data, ct))
                        } else {
                            let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            Ok((data, ct))
                        }
                    })
                    .await;
                return Ok(to_http_response_cached(&engine, cached, if_none_match.as_deref()));
            }

            // 8. GET /cog/{tileMatrixSetId}/tilejson.json & /cog/tilejson.json
            (Method::GET, [tms, "tilejson.json"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let mut json = build_tilejson(&engine, &cogs[0], &host, tms, &raw_query)?;
                apply_tilejson_overrides(&mut json, &pq, &host, "cog", tms, &raw_query);
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,
            (Method::GET, ["tilejson.json"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let mut json =
                    build_tilejson(&engine, &cogs[0], &host, "WebMercatorQuad", &raw_query)?;
                apply_tilejson_overrides(&mut json, &pq, &host, "cog", "WebMercatorQuad", &raw_query);
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 9. GET /cog/{tileMatrixSetId}/map.html
            (Method::GET, [tms, "map.html"]) => {
                let tj_url = format!("{forwarded_prefix}/cog/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }

            // 10. GET /cog/point/{lon},{lat} (supports coord_crs=EPSG:3857)
            (Method::GET, ["point", coord]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let Some((lons, lats)) = coord.split_once(',') else {
                    return Err(EngineError::BadRequest("expected /cog/point/{lon},{lat}".into()));
                };
                let (Ok(x), Ok(y)) = (lons.parse::<f64>(), lats.parse::<f64>()) else {
                    return Err(EngineError::BadRequest("invalid lon,lat floats".into()));
                };
                let (lon, lat) = if pq.coord_crs_3857 {
                    mercator_to_lonlat_deg(x, y)
                } else {
                    (x, y)
                };
                let mut json =
                    query_point(&engine, &cogs[0], lon, lat, pq.unscale, pq.nodata).await?;
                if pq.coord_crs_3857 {
                    json["coordinates"] = serde_json::json!([x, y]);
                }
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 11. GET /cog/preview, /cog/preview.{format}, /cog/preview/{w}x{h}.{format}
            (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
                let ext = p
                    .strip_prefix("preview.")
                    .or(pq.format_param.as_deref())
                    .unwrap_or("png");
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                let out_w = ((full.width as f64 * scale.min(1.0)).round() as u32).max(1);
                let out_h = ((full.height as f64 * scale.min(1.0)).round() as u32).max(1);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq.resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["preview", dims_ext]) => async {
                let Some((out_w, out_h, ext)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected /cog/preview/{w}x{h}.{ext}".into()));
                };
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq.resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,

            // 12. GET /cog/bbox/{minx},{miny},{maxx},{maxy}[.{format}] & /cog/bbox/{...}/{w}x{h}.{format}
            (Method::GET, ["bbox", bbox_seg]) => async {
                let Some((bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("expected /cog/bbox/{minx},{miny},{maxx},{maxy}.{ext}".into()));
                };
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(
                    ext_opt.or(pq.format_param.as_deref()).unwrap_or("png"),
                    pq.webp_quality,
                    pq.jpeg_quality,
                )?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let (out_w, out_h) = aspect_preserving_dims(bbox, pq.max_size.min(1024));
                let data = if pq.coord_crs_3857 {
                    let resample = resolve_resample(&pq, Some(&cogs[0]), false);
                    render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?
                } else {
                    let resample = resolve_resample(&pq, Some(&cogs[0]), true);
                    render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?
                };
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["bbox", bbox_seg, dims_ext]) => async {
                let Some((bbox, _)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox coordinates".into()));
                };
                let Some((out_w, out_h, ext)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected {w}x{h}.{ext}".into()));
                };
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = if pq.coord_crs_3857 {
                    let resample = resolve_resample(&pq, Some(&cogs[0]), false);
                    render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?
                } else {
                    let resample = resolve_resample(&pq, Some(&cogs[0]), true);
                    render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?
                };
                Ok((data, ct))
            }
            .await,

            // 13. POST /cog/feature, /cog/feature.{format}, /cog/feature/{w}x{h}.{format}
            (Method::POST, [f]) if *f == "feature" || f.starts_with("feature.") => async {
                let ext = f
                    .strip_prefix("feature.")
                    .or(pq.format_param.as_deref())
                    .unwrap_or("png");
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson(&geojson)?;
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let (out_w, out_h) = aspect_preserving_dims(bbox, pq.max_size.min(1024));
                let resample = resolve_resample(&pq, Some(&cogs[0]), true);
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, Some(&mask), &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::POST, ["feature", dims_ext]) => async {
                let Some((out_w, out_h, ext)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected /cog/feature/{w}x{h}.{ext}".into()));
                };
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson(&geojson)?;
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let resample = resolve_resample(&pq, Some(&cogs[0]), true);
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, Some(&mask), &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,

            _ => Err(EngineError::NotFound(path)),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    Ok(plain_resp(
        StatusCode::NOT_FOUND,
        Bytes::from(path),
        "text/plain",
    ))
}

fn cache_control_header() -> &'static HeaderValue {
    static VAL: std::sync::OnceLock<HeaderValue> = std::sync::OnceLock::new();
    VAL.get_or_init(|| {
        let secs: u64 = std::env::var("COGZILLA_CACHE_CONTROL_MAX_AGE")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(3600);
        let s = if secs == 0 {
            "no-cache".to_string()
        } else {
            format!("public, max-age={secs}")
        };
        HeaderValue::from_str(&s).unwrap_or_else(|_| HeaderValue::from_static("public, max-age=3600"))
    })
}

fn etag_matches(if_none_match: &str, etag: &str) -> bool {
    let trimmed = if_none_match.trim();
    if trimmed == "*" {
        return true;
    }
    let target = etag.strip_prefix("W/").unwrap_or(etag);
    for part in trimmed.split(',') {
        let cand = part.trim().strip_prefix("W/").unwrap_or(part.trim());
        if cand == target {
            return true;
        }
    }
    false
}

fn render_prometheus_metrics(engine: &TileEngine) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(2048);
    let _ = writeln!(out, "# HELP cogzilla_tiles_served_total Total tiles rendered or served.");
    let _ = writeln!(out, "# TYPE cogzilla_tiles_served_total counter");
    let _ = writeln!(out, "cogzilla_tiles_served_total {}", engine.stats.tiles_served.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_empty_short_circuits_total Empty/nodata tiles short-circuited.");
    let _ = writeln!(out, "# TYPE cogzilla_empty_short_circuits_total counter");
    let _ = writeln!(out, "cogzilla_empty_short_circuits_total {}", engine.stats.empty_short_circuits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_passthrough_hits_total Zero-decode JPEG/WebP pass-through tiles.");
    let _ = writeln!(out, "# TYPE cogzilla_passthrough_hits_total counter");
    let _ = writeln!(out, "cogzilla_passthrough_hits_total {}", engine.stats.passthrough_hits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_http_304_not_modified_total Conditional GET 304 Not Modified responses.");
    let _ = writeln!(out, "# TYPE cogzilla_http_304_not_modified_total counter");
    let _ = writeln!(out, "cogzilla_http_304_not_modified_total {}", engine.stats.http_304_not_modified.load(Ordering::Relaxed));

    let _ = writeln!(out, "# TYPE cogzilla_block_cache_hits_total counter");
    let _ = writeln!(out, "cogzilla_block_cache_hits_total {}", engine.store.cache.hits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_misses_total counter");
    let _ = writeln!(out, "cogzilla_block_cache_misses_total {}", engine.store.cache.misses.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_coalesced_total counter");
    let _ = writeln!(out, "cogzilla_block_cache_coalesced_total {}", engine.store.cache.coalesced_waiters.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_bytes gauge");
    let _ = writeln!(out, "cogzilla_block_cache_bytes {}", engine.store.cache.current_bytes());

    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_hits_total counter");
    let _ = writeln!(out, "cogzilla_tile_cache_hits_total {}", engine.tile_cache.hits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_misses_total counter");
    let _ = writeln!(out, "cogzilla_tile_cache_misses_total {}", engine.tile_cache.misses.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_coalesced_total counter");
    let _ = writeln!(out, "cogzilla_tile_cache_coalesced_total {}", engine.tile_cache.coalesced.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_bytes gauge");
    let _ = writeln!(out, "cogzilla_tile_cache_bytes {}", engine.tile_cache.current_bytes());

    let _ = writeln!(out, "# TYPE cogzilla_upstream_limit gauge");
    let _ = writeln!(out, "cogzilla_upstream_limit {}", engine.store.upstream_limiter.current_limit());
    let _ = writeln!(out, "# TYPE cogzilla_upstream_in_flight gauge");
    let _ = writeln!(out, "cogzilla_upstream_in_flight {}", engine.store.upstream_limiter.in_flight());
    let _ = writeln!(out, "# TYPE cogzilla_upstream_throttles_total counter");
    let _ = writeln!(out, "cogzilla_upstream_throttles_total {}", engine.store.upstream_limiter.throttle_events());
    let _ = writeln!(out, "# TYPE cogzilla_remote_range_requests_total counter");
    let _ = writeln!(out, "cogzilla_remote_range_requests_total {}", engine.store.metrics.remote_range_requests.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_remote_bytes_fetched_total counter");
    let _ = writeln!(out, "cogzilla_remote_bytes_fetched_total {}", engine.store.metrics.remote_bytes_fetched.load(Ordering::Relaxed));

    if let Some(dc) = &engine.store.disk_cache {
        let s = dc.stats();
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_hits_total counter");
        let _ = writeln!(out, "cogzilla_disk_cache_hits_total {}", s.hits);
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_misses_total counter");
        let _ = writeln!(out, "cogzilla_disk_cache_misses_total {}", s.misses);
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_bytes gauge");
        let _ = writeln!(out, "cogzilla_disk_cache_bytes {}", s.bytes_used);
    }
    out
}

fn to_http_response(
    engine: &TileEngine,
    res: Result<(Bytes, &'static str), EngineError>,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    match res {
        Ok((bytes, content_type)) => {
            let is_binary_tile = content_type.starts_with("image/")
                || content_type == "application/octet-stream"
                || content_type == "application/x-npy";
            let etag = is_binary_tile.then(|| crate::tile_cache::compute_tile_etag(&bytes));
            binary_or_plain_response(engine, bytes, content_type, etag, if_none_match)
        }
        Err(EngineError::Shared(inner)) => error_to_http_response(&inner),
        Err(e) => error_to_http_response(&e),
    }
}

fn to_http_response_cached(
    engine: &TileEngine,
    res: Result<crate::tile_cache::CachedTile, EngineError>,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    match res {
        Ok(tile) => {
            binary_or_plain_response(engine, tile.bytes, tile.content_type, tile.etag, if_none_match)
        }
        Err(EngineError::Shared(inner)) => error_to_http_response(&inner),
        Err(e) => error_to_http_response(&e),
    }
}

fn binary_or_plain_response(
    engine: &TileEngine,
    bytes: Bytes,
    content_type: &'static str,
    etag: Option<HeaderValue>,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    if let Some(etag_hv) = etag {
        if let Some(inm) = if_none_match {
            if let Ok(etag_str) = etag_hv.to_str() {
                if etag_matches(inm, etag_str) {
                    engine
                        .stats
                        .http_304_not_modified
                        .fetch_add(1, Ordering::Relaxed);
                    let mut resp = Response::new(Full::new(Bytes::new()));
                    *resp.status_mut() = StatusCode::NOT_MODIFIED;
                    resp.headers_mut().insert(ETAG, etag_hv);
                    resp.headers_mut()
                        .insert(CACHE_CONTROL, cache_control_header().clone());
                    resp.headers_mut()
                        .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
                    return resp;
                }
            }
        }
        let mut resp = plain_resp(StatusCode::OK, bytes, content_type);
        resp.headers_mut().insert(ETAG, etag_hv);
        resp.headers_mut()
            .insert(CACHE_CONTROL, cache_control_header().clone());
        resp
    } else {
        plain_resp(StatusCode::OK, bytes, content_type)
    }
}

fn error_to_http_response(err: &EngineError) -> Response<Full<Bytes>> {
    match err {
        EngineError::Shared(inner) => error_to_http_response(inner),
        EngineError::NotFound(msg) => {
            plain_resp(StatusCode::NOT_FOUND, Bytes::from(msg.clone()), "text/plain")
        }
        EngineError::BadRequest(msg) => {
            plain_resp(StatusCode::BAD_REQUEST, Bytes::from(msg.clone()), "text/plain")
        }
        // The file is in a coordinate system this server cannot place. That is a
        // property of the requested source, not a server fault.
        EngineError::Proj(e) => {
            plain_resp(StatusCode::BAD_REQUEST, Bytes::from(e.to_string()), "text/plain")
        }
        EngineError::Io(e @ cogzilla_io::IoError::InvalidUri(_)) => {
            plain_resp(StatusCode::BAD_REQUEST, Bytes::from(e.to_string()), "text/plain")
        }
        EngineError::Io(e @ cogzilla_io::IoError::Tiff(_)) => {
            plain_resp(StatusCode::BAD_REQUEST, Bytes::from(e.to_string()), "text/plain")
        }
        EngineError::Io(e @ cogzilla_io::IoError::Forbidden(_)) => {
            plain_resp(StatusCode::FORBIDDEN, Bytes::from(e.to_string()), "text/plain")
        }
        EngineError::Io(e @ cogzilla_io::IoError::HttpStatus { status: 404, .. }) => {
            plain_resp(StatusCode::NOT_FOUND, Bytes::from(e.to_string()), "text/plain")
        }
        EngineError::Io(e @ cogzilla_io::IoError::Fs(io_err))
            if io_err.kind() == std::io::ErrorKind::NotFound =>
        {
            plain_resp(StatusCode::NOT_FOUND, Bytes::from(e.to_string()), "text/plain")
        }
        e => {
            tracing::warn!(error = %e, "request failed with 500");
            plain_resp(
                StatusCode::INTERNAL_SERVER_ERROR,
                Bytes::from(e.to_string()),
                "text/plain",
            )
        }
    }
}

/// Bind and serve HTTP/1.1 + HTTP/2 requests until shutdown.
pub async fn serve(engine: Arc<TileEngine>, addr: SocketAddr) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    serve_listener(engine, listener).await
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

pub async fn serve_listener(engine: Arc<TileEngine>, listener: TcpListener) -> std::io::Result<()> {
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);
    loop {
        let (stream, _) = tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("shutdown signal received; stopping listener");
                return Ok(());
            }
            accept_res = listener.accept() => match accept_res {
                Ok(pair) => pair,
                Err(e) => {
                    // Transient socket errors (EMFILE/ENFILE/ECONNABORTED) must never crash the server.
                    tracing::warn!(error = %e, "TCP accept error; backing off briefly");
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                    continue;
                }
            }
        };
        let _ = stream.set_nodelay(true);
        let io = TokioIo::new(stream);
        let eng = engine.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req| handle_request(eng.clone(), req));
            let mut builder = ServerBuilder::new(TokioExecutor::new());
            builder
                .http1()
                .timer(hyper_util::rt::TokioTimer::new())
                .header_read_timeout(std::time::Duration::from_secs(15));
            let _ = builder.serve_connection(io, svc).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colormap_param_sets_explicit_for_hillshade_and_stac_bbox_accepts_coord_crs() {
        let pq = parse_query(Some("algorithm=hillshade&colormap=terrain"));
        assert!(pq.colormap_explicit);
        let bands = build_band_request(&pq, 1, cogzilla_tiff::SampleType::Float32)
            .expect("valid hillshade request");
        match bands {
            BandRequest::Dem { colormap, .. } => {
                assert_ne!(colormap, ColormapName::Grayscale);
                assert_eq!(colormap, ColormapName::from_str_opt("terrain").unwrap());
            }
            other => panic!("expected BandRequest::Dem, got {other:?}"),
        }

        // /stac/bbox must accept coord_crs=EPSG:3857 (matching /cog/bbox)
        let pq_bbox = parse_query(Some("url=foo.json&coord_crs=EPSG:3857"));
        let res = validate_query(&["stac", "bbox", "0,0,1000,1000.png"], &pq_bbox);
        assert!(res.is_ok(), "unexpected error: {res:?}");
    }

    #[test]
    fn xyz_bounds_and_http_error_status_mapping() {
        let pq = parse_query(None);
        assert_eq!(pq.resample, ResampleMode::Nearest);
        assert_eq!(pq.reproject, ResampleMode::Nearest);
        let u8t = cogzilla_tiff::SampleType::Uint8;
        // WebMercatorQuad at z=2 has 4x4 tiles (0..=3)
        assert!(build_tile_request(TileCoord { z: 2, x: 3, y: 3 }, None, "png", &pq, 3, u8t, false, None).is_ok());
        assert!(build_tile_request(TileCoord { z: 2, x: 4, y: 0 }, None, "png", &pq, 3, u8t, false, None).is_err());
        assert!(build_tile_request(TileCoord { z: 2, x: 0, y: 4 }, None, "png", &pq, 3, u8t, false, None).is_err());
        // WorldCRS84Quad at z=2 has 8x4 tiles (x: 0..=7, y: 0..=3)
        assert!(build_tile_request(TileCoord { z: 2, x: 7, y: 3 }, None, "png", &pq, 3, u8t, true, None).is_ok());
        assert!(build_tile_request(TileCoord { z: 2, x: 8, y: 0 }, None, "png", &pq, 3, u8t, true, None).is_err());
        assert!(build_tile_request(TileCoord { z: 2, x: 0, y: 4 }, None, "png", &pq, 3, u8t, true, None).is_err());

        // ?reproject= is accepted and parsed independently from ?resampling=
        let pq_rp = parse_query(Some("resampling=bilinear&reproject=cubic"));
        assert!(pq_rp.errors.is_empty());
        assert_eq!(pq_rp.resample, ResampleMode::Bilinear);
        assert_eq!(pq_rp.reproject, ResampleMode::Cubic);
        assert!(validate_query(&["cog", "tiles", "WebMercatorQuad", "0", "0", "0.png"], &pq_rp).is_ok());

        // Single-band default colormap is Grayscale (matching TiTiler)
        assert_eq!(pq.colormap, ColormapName::Grayscale);

        // Form-encoded ?expression= with spaces (`+`) and `%2B` decodes cleanly, while raw `(b1+b2)` still works
        let pq_form = parse_query(Some("expression=%28b1+-+b2%29+%2F+%28b1+%2B+b2%29"));
        assert_eq!(pq_form.expr_raw.as_deref(), Some("(b1 - b2) / (b1 + b2)"));
        let pq_raw = parse_query(Some("expression=(b1-b2)/(b1+b2)"));
        assert_eq!(pq_raw.expr_raw.as_deref(), Some("(b1-b2)/(b1+b2)"));

        // ?format= applies on format-less tile routes
        let pq_fmt = parse_query(Some("format=webp"));
        let (treq, ct) =
            build_tile_request(TileCoord { z: 2, x: 1, y: 1 }, None, "", &pq_fmt, 3, u8t, false, None)
                .unwrap();
        assert_eq!(ct, "image/webp");
        assert!(matches!(treq.format, OutputFormat::Webp { .. }));

        // Default dtype rescale ranges & aspect-preserving dims & hex colormaps
        assert_eq!(default_rescale_for_dtype(cogzilla_tiff::SampleType::Uint16, false), (0.0, 65535.0));
        assert_eq!(default_rescale_for_dtype(cogzilla_tiff::SampleType::Int16, true), (-32768.0, 32767.0));
        assert_eq!(aspect_preserving_dims([0.0, 0.0, 2.0, 1.0], 512), (512, 256));
        assert_eq!(aspect_preserving_dims([0.0, 0.0, 1.0, 4.0], 512), (128, 512));
        let hex_lut = ColormapLut::from_json_str(r##"{"0":"#112233","255":"#ff000080"}"##).unwrap();
        assert_eq!(hex_lut.rgb[0], [0x11, 0x22, 0x33]);
        assert_eq!(hex_lut.rgba_u32[255].to_le_bytes(), [0xff, 0x00, 0x00, 0x80]);

        // TiTiler query param parity: nodata=nan/inf/-inf, z_exaggeration, [0,1000] rescale,
        // colormap_name precedence over inline colormap, and tilejson minzoom/maxzoom/tile_format.
        let pq_nd = parse_query(Some("nodata=nan&rescale=[0,%201000]&algorithm_params=%7B%22z_exaggeration%22:2.5%7D"));
        assert!(pq_nd.errors.is_empty());
        assert!(pq_nd.nodata.unwrap().is_nan());
        assert_eq!(pq_nd.rescale, vec![(0.0, 1000.0)]);
        assert!((pq_nd.z_factor - 2.5).abs() < 1e-6);

        let pq_cm = parse_query(Some("colormap=%7B%220%22:[0,0,0,255]%7D&colormap_name=viridis"));
        assert!(pq_cm.errors.is_empty());
        assert!(pq_cm.custom_colormap_json.is_none());
        assert_eq!(pq_cm.colormap, ColormapName::from_str_opt("viridis").unwrap());

        let pq_tj = parse_query(Some("url=a.tif&minzoom=4&maxzoom=14&tile_format=webp"));
        assert!(pq_tj.errors.is_empty());
        assert!(validate_query(&["cog", "WebMercatorQuad", "tilejson.json"], &pq_tj).is_ok());
        assert!(validate_query(&["cog", "tiles", "WebMercatorQuad", "4", "0", "0.png"], &pq_tj).is_err());
        let mut tj = serde_json::json!({"minzoom": 0, "maxzoom": 10, "center": [0.0, 0.0, 0], "tiles": []});
        apply_tilejson_overrides(&mut tj, &pq_tj, "http://host", "cog", "WebMercatorQuad", "url=a.tif&minzoom=4&maxzoom=14&tile_format=webp");
        assert_eq!(tj["minzoom"], 4);
        assert_eq!(tj["maxzoom"], 14);
        assert_eq!(tj["center"][2], 4);
        assert_eq!(tj["tiles"][0], "http://host/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.webp?url=a.tif");

        // HTTP status mapping for upstream 404, local NotFound, and invalid TIFF
        let r404 = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::HttpStatus {
            status: 404,
            url: "https://example.com/missing.tif".into(),
        }));
        assert_eq!(r404.status(), StatusCode::NOT_FOUND);
        let r_enoent = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::Fs(
            std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        )));
        assert_eq!(r_enoent.status(), StatusCode::NOT_FOUND);
        let r_tiff = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::Tiff(
            cogzilla_tiff::TiffError::Corrupt("not a tiff"),
        )));
        assert_eq!(r_tiff.status(), StatusCode::BAD_REQUEST);
    }
}
