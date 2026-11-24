use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use cogzilla_kernel::{
    ColorFormula, ColormapName, CustomColormap, DemAlgorithm, ExprProgram, OutputFormat,
    ResampleMode, SpectralExpr, TerrainEncoding,
};
use cogzilla_proj::mercator::lonlat_deg_to_mercator;
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
    PolygonMask, StatsOptions, build_cog_info, build_cog_info_geojson, build_cog_validate,
    build_map_html, build_tilejson, build_wmts_capabilities, compute_statistics_with_options,
    encode_rgba_output, get_algorithm_json, list_algorithms_json, query_point,
    render_mercator_bbox, render_native_preview, render_wgs84_bbox,
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

/// Parse TiTiler's `{y}[@{scale}x][.{ext}]` segment into `(y, scale_opt, ext_opt)`.
fn parse_y_scale_ext(seg: &str) -> Option<(u32, Option<u32>, Option<&str>)> {
    let (left, ext) = match seg.split_once('.') {
        Some((l, e)) => (l, Some(e)),
        None => (seg, None),
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

fn parse_wxh_ext(seg: &str) -> Option<(u32, u32, Option<&str>)> {
    let (dims, ext) = match seg.split_once('.') {
        Some((d, e)) => (d, Some(e)),
        None => (seg, None),
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
    width: Option<u32>,
    height: Option<u32>,
    orientation_vertical: bool,
    use_epsg: bool,
    percentiles: Vec<f64>,
    histogram_bins: Option<usize>,
    histogram_range: Option<(f64, f64)>,
    categorical: bool,
    categories: Vec<f64>,
    webp_quality: Option<f32>,
    jpeg_quality: u8,
    expr_raw: Option<String>,
    nodata: Option<f32>,
    unscale: bool,
    strict: bool,
    return_mask: bool,
    buffer: u32,
    coord_crs_3857: bool,
    format_param: Option<String>,
    minzoom: Option<u8>,
    maxzoom: Option<u8>,
    tile_format: Option<String>,
    tile_scale: Option<u32>,
    algorithm: Option<String>,
    azimuth: f32,
    altitude: f32,
    z_factor: f32,
    threshold: f32,
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
        width: None,
        height: None,
        orientation_vertical: false,
        use_epsg: false,
        percentiles: Vec::new(),
        histogram_bins: None,
        histogram_range: None,
        categorical: false,
        categories: Vec::new(),
        webp_quality: None,
        jpeg_quality: 85,
        expr_raw: None,
        nodata: None,
        unscale: false,
        strict: false,
        return_mask: true,
        buffer: 0,
        coord_crs_3857: false,
        format_param: None,
        minzoom: None,
        maxzoom: None,
        tile_format: None,
        tile_scale: None,
        algorithm: None,
        azimuth: 315.0,
        altitude: 45.0,
        z_factor: 1.0,
        threshold: 128.0,
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
                let dec = percent_decode(v);
                if dec.contains('|') {
                    // TiTiler `assets=name|bidx=1,2,3` or `assets=name|indexes=1,2,3`
                    let mut parts = dec.split('|');
                    if let Some(name) = parts.next()
                        && !name.is_empty()
                    {
                        pq.assets.push(name.to_string());
                    }
                    for opt in parts {
                        if let Some(bidx_str) = opt
                            .strip_prefix("bidx=")
                            .or_else(|| opt.strip_prefix("indexes="))
                        {
                            for tok in bidx_str.split(',') {
                                match tok.trim().parse::<u16>() {
                                    Ok(b) if b >= 1 => pq.bidx.push(b),
                                    _ => bad("expected positive integer band indices in |bidx="),
                                }
                            }
                        }
                    }
                } else {
                    for a in dec.split(',') {
                        if !a.is_empty() {
                            pq.assets.push(a.to_string());
                        }
                    }
                }
            }
            "asset_bidx" => {
                let dec = percent_decode(v);
                if let Some((_name, bidx_str)) = dec.split_once('|') {
                    for tok in bidx_str.split(',') {
                        match tok.trim().parse::<u16>() {
                            Ok(b) if b >= 1 => pq.bidx.push(b),
                            _ => bad("expected positive integer band indices in asset_bidx"),
                        }
                    }
                } else {
                    bad("expected asset_bidx=name|1,2,3");
                }
            }
            "asset_as_band" => match percent_decode(v).to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "false" | "0" | "no" => {}
                _ => bad("expected boolean"),
            },
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
                        if let Some(val) = obj.get("threshold") {
                            match val.as_f64().map(|t| t as f32) {
                                Some(t) if t.is_finite() => pq.threshold = t,
                                _ => bad("threshold must be a finite number"),
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
                let trimmed = dec.trim_start();
                if trimmed.starts_with('{') || trimmed.starts_with('[') {
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
            "strict" => {
                pq.strict = v.eq_ignore_ascii_case("true") || v == "1";
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
            "tile_scale" => match v.parse::<u32>() {
                Ok(sc) if (1..=4).contains(&sc) => pq.tile_scale = Some(sc),
                _ => bad("expected tile_scale in 1..=4"),
            },
            "max_size" => {
                if let Ok(ms) = v.parse::<u32>() {
                    pq.max_size = ms.clamp(1, 4096);
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
                let lower = v.to_ascii_lowercase();
                let plus_is_space = lower.contains("%2b")
                    || lower.contains("+-")
                    || lower.contains("-+")
                    || lower.contains("+*")
                    || lower.contains("*+")
                    || lower.contains("+/")
                    || lower.contains("/+")
                    || lower.contains("+^")
                    || lower.contains("^+")
                    || lower.contains("++")
                    || lower.contains("+(")
                    || lower.contains(")+")
                    || lower.contains(",+")
                    || lower.contains("+%2f")
                    || lower.contains("%2f+")
                    || lower.contains("+%2a")
                    || lower.contains("%2a+")
                    || lower.contains("+%5e")
                    || lower.contains("%5e+")
                    || lower.contains("+%28")
                    || lower.contains("%28+")
                    || lower.contains("+%29")
                    || lower.contains("%29+")
                    || lower.contains("%2c+");
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
            "width" => match v.parse::<u32>() {
                Ok(n) if (1..=4096).contains(&n) => pq.width = Some(n),
                _ => bad("expected width in 1..=4096"),
            },
            "height" => match v.parse::<u32>() {
                Ok(n) if (1..=4096).contains(&n) => pq.height = Some(n),
                _ => bad("expected height in 1..=4096"),
            },
            "orientation" => match v.to_ascii_lowercase().as_str() {
                "horizontal" => pq.orientation_vertical = false,
                "vertical" => pq.orientation_vertical = true,
                _ => bad("expected horizontal or vertical"),
            },
            "use_epsg" => {
                pq.use_epsg = v.eq_ignore_ascii_case("true") || v == "1";
            }
            "p" => {
                let dec = percent_decode(v);
                for tok in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    match tok.parse::<f64>() {
                        Ok(p) if p.is_finite() && (0.0..=100.0).contains(&p) => pq.percentiles.push(p),
                        _ => bad("expected percentile p in 0..=100"),
                    }
                }
            }
            "histogram_bins" => match v.parse::<usize>() {
                Ok(n) if (1..=1000).contains(&n) => pq.histogram_bins = Some(n),
                _ => bad("expected histogram_bins in 1..=1000"),
            },
            "histogram_range" => {
                let dec = percent_decode(v);
                let parts: Vec<&str> = dec
                    .trim_matches(|c| c == '[' || c == ']')
                    .split(',')
                    .map(str::trim)
                    .collect();
                if parts.len() == 2
                    && let (Ok(lo), Ok(hi)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>())
                    && lo.is_finite()
                    && hi.is_finite()
                    && lo < hi
                {
                    pq.histogram_range = Some((lo, hi));
                } else {
                    bad("expected histogram_range=min,max with min < max");
                }
            }
            "categorical" => {
                pq.categorical = v.eq_ignore_ascii_case("true") || v == "1";
            }
            "c" => {
                let dec = percent_decode(v);
                for tok in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    match tok.parse::<f64>() {
                        Ok(c) if c.is_finite() => pq.categories.push(c),
                        _ => bad("expected numeric category value"),
                    }
                }
            }
            "dst_crs" | "crs" | "cover_scale" => {
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

const SUPPORTED_ALGORITHMS: &[&str] = &[
    "hillshade",
    "slope",
    "aspect",
    "terrarium",
    "terrainrgb",
    "terrain-rgb",
    "mapbox",
    "normalizedindex",
    "cast",
    "ceil",
    "floor",
    "min",
    "max",
    "median",
    "mean",
    "std",
    "var",
    "sum",
    "grayscale",
    "bitonal",
];
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

    // {family}/tiles/{tms}[/{z}/{x}/{y}], {family}/{tms}/tilejson.json, {family}/{tms}/map.html,
    // {family}/{tms}/WMTSCapabilities.xml
    if family != "tiles" && family != "colorMaps" {
        let last = segs.last().copied().unwrap_or("");
        let tms = if matches!(last, "tilejson.json" | "map.html" | "WMTSCapabilities.xml")
            && segs.len() >= 3
        {
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
    // Only match route sub-endpoints on `/cog/...`, `/stac/...`, and `/mosaicjson/...` so
    // dataset/catalog names like `/tiles/statistics/0/0/0.png` are never misclassified.
    let route = |name: &str| {
        matches!(segs[0], "cog" | "stac" | "mosaicjson")
            && segs
                .get(1)
                .is_some_and(|s| *s == name || s.starts_with(&format!("{name}.")))
    };
    if family != "colorMaps" {
        if let Some(k) = has(&["orientation"]) {
            return Err(format!("parameter '{k}' is only supported on /colorMaps routes"));
        }
    }
    if !(route("preview") || route("bbox") || route("feature")) {
        if let Some(k) = has(&["width", "height"]) {
            return Err(format!(
                "parameter '{k}' is only supported on preview, bbox, and feature routes"
            ));
        }
    }
    if route("statistics") || route("asset_statistics") {
        if let Some(k) = has(&["expression", "expr", "index", "algorithm"]) {
            return Err(format!("parameter '{k}' is not supported on statistics routes"));
        }
    } else if let Some(k) = has(&["p", "histogram_bins", "histogram_range", "categorical", "c"]) {
        return Err(format!("parameter '{k}' is only supported on statistics routes"));
    }
    if route("point") {
        if let Some(k) = has(&["algorithm"]) {
            return Err(format!("parameter '{k}' is not supported on point routes"));
        }
    } else if !route("bbox") {
        if let Some(k) = has(&["coord_crs"]) {
            return Err(format!("parameter '{k}' is not supported on this route"));
        }
    }
    if route("bbox") && segs.len() >= 4 {
        if let Some(k) = has(&["max_size", "width", "height"]) {
            return Err(format!("parameter '{k}' is not supported when explicit {{w}}x{{h}} dimensions are given"));
        }
    }
    if (route("preview") || route("feature")) && segs.len() >= 3 {
        if let Some(k) = has(&["width", "height"]) {
            return Err(format!("parameter '{k}' is not supported when explicit {{w}}x{{h}} dimensions are given"));
        }
    }
    let last_seg = segs.last().copied().unwrap_or("");
    if !matches!(last_seg, "tilejson.json" | "map.html" | "WMTSCapabilities.xml") {
        if let Some(k) = has(&["minzoom", "maxzoom", "tile_format", "tile_scale"]) {
            return Err(format!(
                "parameter '{k}' is only supported on tilejson.json, map.html, and WMTSCapabilities.xml routes"
            ));
        }
    } else if let (Some(minz), Some(maxz)) = (pq.minzoom, pq.maxzoom)
        && minz > maxz
    {
        return Err(format!("minzoom ({minz}) must be <= maxzoom ({maxz})"));
    }
    Ok(())
}

/// Apply optional `?width=` / `?height=` overrides onto aspect-preserving default dimensions.
fn resolve_query_dims(default_w: u32, default_h: u32, q_w: Option<u32>, q_h: Option<u32>) -> (u32, u32) {
    match (q_w, q_h) {
        (Some(w), Some(h)) => (w.clamp(1, 4096), h.clamp(1, 4096)),
        (Some(w), None) => {
            let w = w.clamp(1, 4096);
            let h = ((w as f64 * (default_h as f64 / default_w.max(1) as f64)).round() as u32)
                .clamp(1, 4096);
            (w, h)
        }
        (None, Some(h)) => {
            let h = h.clamp(1, 4096);
            let w = ((h as f64 * (default_w as f64 / default_h.max(1) as f64)).round() as u32)
                .clamp(1, 4096);
            (w, h)
        }
        (None, None) => (default_w, default_h),
    }
}

/// Strip TileJSON/WMTS-only query parameters so they are not forwarded into tile URL templates.
fn filter_tile_template_query(raw_query: &str) -> String {
    raw_query
        .split('&')
        .filter(|pair| {
            let k = pair.split('=').next().unwrap_or("");
            !pair.is_empty()
                && !matches!(
                    k,
                    "minzoom" | "maxzoom" | "tile_format" | "tile_scale" | "use_epsg"
                )
        })
        .collect::<Vec<_>>()
        .join("&")
}

/// Apply TiTiler `minzoom`, `maxzoom`, `tile_format`, and `tile_scale` query overrides to a
/// TileJSON document, stripping those TileJSON-only params from the forwarded tile URL template.
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
            !pair.is_empty() && !matches!(k, "minzoom" | "maxzoom" | "tile_format" | "tile_scale")
        })
        .collect::<Vec<_>>()
        .join("&");
    let scale_suffix = match pq.tile_scale {
        Some(sc) if sc > 1 => format!("@{sc}x"),
        _ => String::new(),
    };
    let ext_suffix = pq
        .tile_format
        .as_deref()
        .map(|f| format!(".{f}"))
        .unwrap_or_default();
    let tile_url = if filtered_query.is_empty() {
        format!("{origin}/{route_prefix}/tiles/{tms}/{{z}}/{{x}}/{{y}}{scale_suffix}{ext_suffix}")
    } else {
        format!(
            "{origin}/{route_prefix}/tiles/{tms}/{{z}}/{{x}}/{{y}}{scale_suffix}{ext_suffix}?{filtered_query}"
        )
    };
    tj["tiles"] = serde_json::json!([tile_url]);
    let cur_min = tj.get("minzoom").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
    let cur_max = tj.get("maxzoom").and_then(|v| v.as_u64()).unwrap_or(22) as u8;
    let (new_min, new_max) = match (pq.minzoom, pq.maxzoom) {
        (Some(mn), Some(mx)) => (mn, mx.max(mn)),
        (Some(mn), None) => (mn, cur_max.max(mn)),
        (None, Some(mx)) => (cur_min.min(mx), mx),
        (None, None) => (cur_min, cur_max),
    };
    tj["minzoom"] = serde_json::json!(new_min);
    tj["maxzoom"] = serde_json::json!(new_max);
    if let Some(center) = tj.get_mut("center").and_then(|v| v.as_array_mut())
        && center.len() >= 3
    {
        center[2] = serde_json::json!(new_min);
    }
}

fn parse_ext_format(
    ext: &str,
    webp_quality: Option<f32>,
    jpeg_quality: u8,
) -> Result<(OutputFormat, &'static str), EngineError> {
    match ext {
        "png" | "pngraw" | "" => Ok((OutputFormat::Png, "image/png")),
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
            "normalizedindex" | "cast" | "ceil" | "floor" | "min" | "max" | "median" | "mean"
            | "std" | "var" | "sum" | "grayscale" | "bitonal" => {
                let bands: Vec<u16> = if !pq.bidx.is_empty() {
                    pq.bidx.clone()
                } else {
                    (1..=default_bands.clamp(1, 16)).collect()
                };
                let (expr_str, default_rescale) = match alg {
                    "normalizedindex" => {
                        if bands.len() < 2 {
                            return Err(EngineError::BadRequest(
                                "algorithm 'normalizedIndex' requires at least 2 bands".into(),
                            ));
                        }
                        (
                            format!(
                                "(b{} - b{}) / (b{} + b{})",
                                bands[0], bands[1], bands[0], bands[1]
                            ),
                            (-1.0, 1.0),
                        )
                    }
                    "cast" => (
                        format!("b{}", bands[0]),
                        default_rescale_for_dtype(sample_type, true),
                    ),
                    "ceil" => (
                        format!("ceil(b{})", bands[0]),
                        default_rescale_for_dtype(sample_type, true),
                    ),
                    "floor" => (
                        format!("floor(b{})", bands[0]),
                        default_rescale_for_dtype(sample_type, true),
                    ),
                    "min" => {
                        let mut acc = format!("b{}", bands[0]);
                        for &b in &bands[1..] {
                            acc = format!("min({acc}, b{b})");
                        }
                        (acc, default_rescale_for_dtype(sample_type, true))
                    }
                    "max" => {
                        let mut acc = format!("b{}", bands[0]);
                        for &b in &bands[1..] {
                            acc = format!("max({acc}, b{b})");
                        }
                        (acc, default_rescale_for_dtype(sample_type, true))
                    }
                    "sum" => {
                        let sum_s = bands
                            .iter()
                            .map(|b| format!("b{b}"))
                            .collect::<Vec<_>>()
                            .join(" + ");
                        let (lo, hi) = default_rescale_for_dtype(sample_type, true);
                        (sum_s, (lo, hi * bands.len() as f32))
                    }
                    "mean" => {
                        let sum_s = bands
                            .iter()
                            .map(|b| format!("b{b}"))
                            .collect::<Vec<_>>()
                            .join(" + ");
                        (
                            format!("({sum_s}) / {}", bands.len()),
                            default_rescale_for_dtype(sample_type, true),
                        )
                    }
                    "var" | "std" => {
                        let expr = if bands.len() <= 1 {
                            "0".to_string()
                        } else {
                            let n = bands.len();
                            let mean_s = format!(
                                "({}) / {n}",
                                bands
                                    .iter()
                                    .map(|b| format!("b{b}"))
                                    .collect::<Vec<_>>()
                                    .join(" + ")
                            );
                            let sq_s = bands
                                .iter()
                                .map(|b| format!("(b{b} - ({mean_s}))^2"))
                                .collect::<Vec<_>>()
                                .join(" + ");
                            if alg == "std" {
                                format!("sqrt(({sq_s}) / {n})")
                            } else {
                                format!("({sq_s}) / {n}")
                            }
                        };
                        (expr, default_rescale_for_dtype(sample_type, true))
                    }
                    "median" => {
                        let expr = match bands.len() {
                            1 => format!("b{}", bands[0]),
                            2 => format!("(b{} + b{}) / 2", bands[0], bands[1]),
                            3 => format!(
                                "b{0} + b{1} + b{2} - min(b{0}, min(b{1}, b{2})) - max(b{0}, max(b{1}, b{2}))",
                                bands[0], bands[1], bands[2]
                            ),
                            _ => {
                                let sum_s = bands
                                    .iter()
                                    .map(|b| format!("b{b}"))
                                    .collect::<Vec<_>>()
                                    .join(" + ");
                                format!("({sum_s}) / {}", bands.len())
                            }
                        };
                        (expr, default_rescale_for_dtype(sample_type, true))
                    }
                    "grayscale" => {
                        let expr = if bands.len() >= 3 {
                            format!(
                                "0.299 * b{} + 0.587 * b{} + 0.114 * b{}",
                                bands[0], bands[1], bands[2]
                            )
                        } else {
                            format!("b{}", bands[0])
                        };
                        (expr, default_rescale_for_dtype(sample_type, true))
                    }
                    "bitonal" => (
                        format!("where(b{} >= {}, 1, 0)", bands[0], pq.threshold),
                        (0.0, 1.0),
                    ),
                    _ => unreachable!(),
                };
                let prog = ExprProgram::compile(&expr_str).map_err(EngineError::BadRequest)?;
                let rescale = pq.rescale.first().copied().unwrap_or(default_rescale);
                return Ok(BandRequest::CustomExpr {
                    prog,
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
                        0 => ([-20.0, -25.0, 0.0], [0.0, -5.0, 15.0]),
                        1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
                        3 => (
                            [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                            [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
                        ),
                        n => {
                            return Err(EngineError::BadRequest(format!(
                                "expected 1 or 3 rescale pairs for 3-band SAR RGB output, got {n}"
                            )));
                        }
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

    if !pq.bidx.is_empty() && pq.bidx.len() != 1 && pq.bidx.len() != 3 {
        return Err(EngineError::BadRequest(format!(
            "unsupported bidx count {} (expected 1 or 3 bands)",
            pq.bidx.len()
        )));
    }

    if pq.bidx.len() == 3 || (pq.bidx.is_empty() && default_bands >= 3) {
        let bs = if pq.bidx.len() == 3 {
            [pq.bidx[0], pq.bidx[1], pq.bidx[2]]
        } else {
            [1, 2, 3]
        };
        let (rmin, rmax) = match pq.rescale.len() {
            0 => {
                let (lo, hi) = default_rescale_for_dtype(sample_type, false);
                ([lo; 3], [hi; 3])
            }
            1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
            3 => (
                [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
            ),
            n => {
                return Err(EngineError::BadRequest(format!(
                    "expected 1 or 3 rescale pairs for 3-band RGB output, got {n}"
                )));
            }
        };
        Ok(BandRequest::Rgb {
            bands_1based: bs,
            rescale_min: rmin,
            rescale_max: rmax,
        })
    } else {
        if pq.rescale.len() > 1 {
            return Err(EngineError::BadRequest(format!(
                "expected 1 rescale pair for single-band output, got {}",
                pq.rescale.len()
            )));
        }
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
/// `Nearest`). Uses `engine.get_or_build_crs` (cached) to avoid rebuilding `CrsModel` per tile.
fn resolve_resample(
    engine: Option<&TileEngine>,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    target_is_4326: bool,
) -> ResampleMode {
    let Some(cog) = source_cog else {
        return pq.resample;
    };
    let crs_res = if let Some(eng) = engine {
        eng.get_or_build_crs(cog).ok()
    } else {
        cogzilla_proj::CrsModel::from_geo_keys(&cog.index.geo_keys).ok()
    };
    let same_crs = match crs_res {
        Some(cogzilla_proj::CrsModel::Epsg3857) => !target_is_4326,
        Some(cogzilla_proj::CrsModel::Geographic) => target_is_4326,
        _ => false,
    };
    if same_crs { pq.resample } else { pq.reproject }
}

fn build_tile_request(
    engine: Option<&TileEngine>,
    tile: TileCoord,
    scale_opt: Option<u32>,
    ext_opt: Option<&str>,
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
    let eff_ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
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
            resample: resolve_resample(engine, pq, source_cog, is_crs84),
            format,
            unscale: pq.unscale,
        },
        content_type,
    ))
}

fn build_stats_options(pq: &ParsedQuery) -> StatsOptions {
    let mut opts = StatsOptions::default();
    if !pq.percentiles.is_empty() {
        opts.percentiles = pq.percentiles.clone();
    }
    if let Some(bins) = pq.histogram_bins {
        opts.histogram_bins = bins;
    }
    opts.histogram_range = pq.histogram_range;
    opts.categorical = pq.categorical;
    if !pq.categories.is_empty() {
        opts.categories = Some(pq.categories.clone());
    }
    if !pq.bidx.is_empty() {
        opts.bidx = Some(pq.bidx.clone());
    }
    opts
}

/// Apply optional `?expression=` or `?bidx=` filtering to a TiTiler `/point` JSON payload.
fn apply_point_filters(
    pt: &mut serde_json::Value,
    expr_raw: Option<&str>,
    bidx: &[u16],
) -> Result<(), EngineError> {
    let Some(vals) = pt.get("values").and_then(|v| v.as_array()).cloned() else {
        return Ok(());
    };
    let names: Vec<serde_json::Value> = pt
        .get("band_names")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let descs: Vec<serde_json::Value> = pt
        .get("band_descriptions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_else(|| names.clone());

    if let Some(expr) = expr_raw {
        let prog = ExprProgram::compile(expr).map_err(EngineError::BadRequest)?;
        let n_bands = vals.len();
        let mut slot_vals = Vec::with_capacity(prog.used_bands.len());
        for &b in &prog.used_bands {
            if b == 0 || (b as usize) > n_bands {
                return Err(EngineError::BadRequest(format!(
                    "expression references b{b} but point query only has {n_bands} bands"
                )));
            }
            let f = vals[(b - 1) as usize]
                .as_f64()
                .map(|x| x as f32)
                .unwrap_or(f32::NAN);
            slot_vals.push(f);
        }
        let res = prog.eval(&slot_vals);
        let out_val = if res.is_finite() {
            serde_json::json!(res)
        } else {
            serde_json::Value::Null
        };
        pt["values"] = serde_json::json!([out_val]);
        pt["band_names"] = serde_json::json!([expr]);
        pt["band_descriptions"] = serde_json::json!([expr]);
        return Ok(());
    }

    if !bidx.is_empty() {
        let n_bands = vals.len();
        let mut f_vals = Vec::with_capacity(bidx.len());
        let mut f_names = Vec::with_capacity(bidx.len());
        let mut f_descs = Vec::with_capacity(bidx.len());
        for &b in bidx {
            if b == 0 || (b as usize) > n_bands {
                return Err(EngineError::BadRequest(format!(
                    "band {b} out of range (dataset has {n_bands} bands)"
                )));
            }
            let idx = (b - 1) as usize;
            f_vals.push(vals[idx].clone());
            f_names.push(names.get(idx).cloned().unwrap_or_else(|| serde_json::json!(format!("b{b}"))));
            f_descs.push(descs.get(idx).cloned().unwrap_or_else(|| serde_json::json!(format!("b{b}"))));
        }
        pt["values"] = serde_json::Value::Array(f_vals);
        pt["band_names"] = serde_json::Value::Array(f_names);
        pt["band_descriptions"] = serde_json::Value::Array(f_descs);
    }
    Ok(())
}

fn validate_mosaic_tile_params(_pq: &ParsedQuery, tile_size: u32) -> Result<(), EngineError> {
    if !(64..=2048).contains(&tile_size) {
        return Err(EngineError::BadRequest(format!(
            "mosaic tile_size must be in 64..=2048 (got {tile_size})"
        )));
    }
    Ok(())
}

async fn render_mosaic_tile_styled(
    engine: &TileEngine,
    urls: &[String],
    mut treq: TileRequest,
    pq: &ParsedQuery,
) -> Result<Bytes, EngineError> {
    let needs_bbox_mosaic = pq.nodata.is_some()
        || pq.custom_colormap_json.is_some()
        || pq.buffer > 0
        || (treq.tile_size != 256 && treq.tile_size != 512);

    if needs_bbox_mosaic && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy) {
        let custom_lut = pq
            .custom_colormap_json
            .as_deref()
            .map(|j| CustomColormap::from_json_str(j, !pq.rescale.is_empty()))
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
            treq.tile.z,
            treq.tile.x,
            treq.tile.y,
            pq.buffer,
            treq.tile_size,
        );
        let n_px = (out_dim as usize) * (out_dim as usize);
        let mut comp = vec![0u8; n_px * 4];
        let mut filled = 0usize;

        for url in urls {
            let cog = engine.store.open_uri(url).await?;
            let cogs = vec![cog];
            let mut scene_bands = treq.bands.clone();
            if pq.unscale {
                apply_unscale_cogs(&mut scene_bands, &cogs);
            }
            let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), false);
            let scene_rgba = render_mercator_bbox(
                engine,
                &cogs,
                bbox_3857,
                out_dim,
                out_dim,
                &scene_bands,
                resample,
                OutputFormat::RawRgba,
                pq.nodata,
                custom_lut.as_ref(),
                None,
                true,
                pq.unscale,
            )
            .await?;
            if scene_rgba.len() != n_px * 4 {
                continue;
            }
            for i in 0..n_px {
                let o = i * 4;
                if comp[o + 3] == 0 && scene_rgba[o + 3] > 0 {
                    comp[o..o + 4].copy_from_slice(&scene_rgba[o..o + 4]);
                    filled += 1;
                }
            }
            if filled >= n_px {
                break;
            }
        }
        if let Some(cf) = cf {
            cf.apply_buffer_in_place(&mut comp, 4);
        }
        return Ok(encode_rgba_output(
            out_dim as usize,
            out_dim as usize,
            &comp,
            treq.format,
            pq.return_mask,
        ));
    }

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
        .or_else(|| {
            req.uri()
                .scheme_str()
                .filter(|&s| s.eq_ignore_ascii_case("https") || s.eq_ignore_ascii_case("http"))
                .map(|s| s.to_ascii_lowercase())
        })
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
                "/cog/validate": { "get": { "summary": "Validate COG structure and layout" } },
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
                "/stac/bounds": { "get": { "summary": "STAC Item WGS84 bounds" } },
                "/stac/info": { "get": { "summary": "STAC assets metadata" } },
                "/stac/info.geojson": { "get": { "summary": "STAC GeoJSON metadata" } },
                "/stac/asset_statistics": { "get": { "summary": "Per-asset STAC statistics" } },
                "/stac/statistics": { "get": { "summary": "Merged STAC statistics" }, "post": { "summary": "Polygon-clipped STAC statistics" } },
                "/stac/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}": { "get": { "summary": "Render STAC multi-asset tile" } },
                "/stac/{tileMatrixSetId}/tilejson.json": { "get": { "summary": "STAC TileJSON 3.0" } },
                "/stac/point/{lon},{lat}": { "get": { "summary": "Sample STAC asset values at point" } },
                "/stac/preview/{width}x{height}.{format}": { "get": { "summary": "Render STAC preview" } },
                "/algorithms": { "get": { "summary": "List available raster algorithms" } },
                "/algorithms/{algorithmId}": { "get": { "summary": "Get algorithm metadata and parameters" } },
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
    //   GET  /mosaicjson/info?url=...
    //   GET  /mosaicjson/bounds?url=...
    //   GET  /mosaicjson/tiles?url=...
    //   GET  /mosaicjson/tiles/{tms}?url=...
    //   GET  /mosaicjson/point/{lon},{lat}?url=...
    //   GET  /mosaicjson/{tms}/tilejson.json?url=...
    //   GET  /mosaicjson/{tms}/WMTSCapabilities.xml?url=...
    //   GET  /mosaicjson/{tms}/map.html?url=...
    //   GET  /mosaicjson/tiles/{tms}/{z}/{x}/{y}[@scale][.{ext}]?url=...
    //   POST /mosaicjson/validate
    // ------------------------------------------------------------------------
    if method == Method::POST && segs.as_slice() == ["mosaicjson", "validate"] {
        let res: Result<(Bytes, &'static str), EngineError> = async {
            let body_bytes = read_limited_body(req.into_body()).await?;
            let text = std::str::from_utf8(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid UTF-8 body: {e}")))?;
            let mj = MosaicJsonCatalog::from_json_str(text).map_err(EngineError::BadRequest)?;
            Ok((Bytes::from(mj.info_json().to_string()), "application/json"))
        }
        .await;
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    if method == Method::GET && segs.first() == Some(&"mosaicjson") {
        let mj_url = pq.urls.first().cloned();
        let res: Result<(Bytes, &'static str), EngineError> = match segs.as_slice() {
            ["mosaicjson", "info"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                Ok((Bytes::from(mj.info_json().to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "bounds"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let body = serde_json::json!({ "bounds": mj.bounds }).to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles"] => {
                let body = serde_json::json!({
                    "tilesets": [
                        {
                            "title": "WebMercatorQuad",
                            "id": "WebMercatorQuad",
                            "crs": "http://www.opengis.net/def/crs/EPSG/0/3857"
                        }
                    ]
                })
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            ["mosaicjson", "tiles", tms] => {
                let body = serde_json::json!({
                    "title": tms,
                    "tileMatrixSetId": tms,
                    "dataType": "map",
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/3857"
                })
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            ["mosaicjson", "point", coord] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let Some((lons, lats)) = coord.split_once(',') else {
                    return Err(EngineError::BadRequest(
                        "expected /mosaicjson/point/{lon},{lat}".into(),
                    ));
                };
                let (Ok(x), Ok(y)) = (lons.parse::<f64>(), lats.parse::<f64>()) else {
                    return Err(EngineError::BadRequest("invalid lon,lat floats".into()));
                };
                let (lon, lat, mx, my) = if pq.coord_crs_3857 {
                    let (lo, la) = mercator_to_lonlat_deg(x, y);
                    (lo, la, x, y)
                } else {
                    let (mx, my) = lonlat_deg_to_mercator(x, y);
                    (x, y, mx, my)
                };
                let urls = mj.query_bbox_3857(mx, my, mx, my);
                let mut assets_out = Vec::new();
                for asset_url in urls.into_iter().take(16) {
                    let cog = engine.store.open_uri(&asset_url).await?;
                    match query_point(&engine, &cog, lon, lat, pq.unscale, pq.nodata).await {
                        Ok(pt) => {
                            let vals = pt["values"].clone();
                            let names = pt["band_names"].clone();
                            assets_out.push(serde_json::json!([asset_url, vals, names]));
                        }
                        Err(EngineError::NotFound(_)) => continue,
                        Err(e) => return Err(e),
                    }
                }
                if assets_out.is_empty() {
                    return Err(EngineError::NotFound(
                        "point outside all MosaicJSON scenes".into(),
                    ));
                }
                let body = serde_json::json!({
                    "coordinates": [x, y],
                    "assets": assets_out,
                })
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,
            ["mosaicjson", _, "WMTSCapabilities.xml"] | ["mosaicjson", "WMTSCapabilities.xml"] => {
                async {
                    let tms = if segs.len() == 3 { segs[1] } else { "WebMercatorQuad" };
                    let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                    let mj = load_mosaicjson(&engine, &u).await?;
                    let minz = pq.minzoom.unwrap_or(mj.minzoom);
                    let maxz = pq.maxzoom.unwrap_or(mj.maxzoom).max(minz);
                    let fmt = pq.tile_format.as_deref().unwrap_or("png");
                    let scale = pq.tile_scale.unwrap_or(1);
                    let tile_q = filter_tile_template_query(&raw_query);
                    let xml = build_wmts_capabilities(
                        &host,
                        "mosaicjson",
                        tms,
                        "MosaicJSON",
                        mj.bounds,
                        minz,
                        maxz,
                        fmt,
                        scale,
                        &tile_q,
                        &raw_query,
                        pq.use_epsg,
                    );
                    Ok((Bytes::from(xml), "application/xml"))
                }
                .await
            }
            ["mosaicjson", _, "tilejson.json"] | ["mosaicjson", "tilejson.json"] => async {
                let tms = if segs.len() == 3 { segs[1] } else { "WebMercatorQuad" };
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let center = serde_json::json!([
                    0.5 * (mj.bounds[0] + mj.bounds[2]),
                    0.5 * (mj.bounds[1] + mj.bounds[3]),
                    mj.minzoom,
                ]);
                let mut tj = serde_json::json!({
                    "tilejson": "3.0.0",
                    "version": mj.version,
                    "scheme": "xyz",
                    "tiles": [],
                    "minzoom": mj.minzoom,
                    "maxzoom": mj.maxzoom,
                    "bounds": mj.bounds,
                    "center": center,
                });
                apply_tilejson_overrides(&mut tj, &pq, &host, "mosaicjson", tms, &raw_query);
                Ok((Bytes::from(tj.to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", _, "map.html"] | ["mosaicjson", "map.html"] => {
                let tms = if segs.len() == 3 { segs[1] } else { "WebMercatorQuad" };
                let tj_url =
                    format!("{forwarded_prefix}/mosaicjson/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
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
                                Some(&engine),
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
                                Some(&engine),
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
                                Some(&engine),
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
                                Some(&engine),
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
                                Some(&engine),
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
                                Some(&engine),
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
                                    .map(|j| CustomColormap::from_json_str(j, !pq.rescale.is_empty()))
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
                Some("png") => {
                    get_colormap_image(id, OutputFormat::Png, pq.orientation_vertical)
                        .map(|b| (b, "image/png"))
                }
                Some("jpg" | "jpeg") => get_colormap_image(
                    id,
                    OutputFormat::Jpeg { quality: pq.jpeg_quality },
                    pq.orientation_vertical,
                )
                .map(|b| (b, "image/jpeg")),
                Some("webp") => get_colormap_image(
                    id,
                    OutputFormat::Webp { quality: pq.webp_quality },
                    pq.orientation_vertical,
                )
                .map(|b| (b, "image/webp")),
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
    // TiTiler AlgorithmFactory Router: /algorithms, /algorithms/{algorithmId}
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"algorithms") {
        let res = match segs.as_slice() {
            ["algorithms"] => Ok((
                Bytes::from(list_algorithms_json(&host).to_string()),
                "application/json",
            )),
            ["algorithms", id] => {
                get_algorithm_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
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

            // GET /stac/bounds?url=item.json
            (Method::GET, ["bounds"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                let bounds = if let Some(b) = item.bbox {
                    b
                } else {
                    let (_, cogs, _) =
                        resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                    let mut acc = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                    for cog in &cogs {
                        let crs = engine.get_or_build_crs(cog)?;
                        let full = cog.index.full_res();
                        let b = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                        acc[0] = acc[0].min(b[0]);
                        acc[1] = acc[1].min(b[1]);
                        acc[2] = acc[2].max(b[2]);
                        acc[3] = acc[3].max(b[3]);
                    }
                    if !acc.iter().all(|v| v.is_finite()) {
                        return Err(EngineError::BadRequest("STAC item has no valid bounds".into()));
                    }
                    acc
                };
                let body = serde_json::json!({"bounds": bounds}).to_string();
                Ok((Bytes::from(body), "application/json"))
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
                let stats_opts = build_stats_options(&pq);
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let opts_c = stats_opts.clone();
                    set.spawn(async move {
                        let st = compute_statistics_with_options(
                            &eng, &cog, max_size, None, unscale, nodata, &opts_c,
                        )
                        .await?;
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
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let n = cogs.len();
                let stats_opts = build_stats_options(&pq);
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let opts_c = stats_opts.clone();
                    set.spawn(async move {
                        let nb = cog.index.full_res().samples_per_pixel as usize;
                        let st = compute_statistics_with_options(
                            &eng, &cog, max_size, None, unscale, nodata, &opts_c,
                        )
                        .await?;
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
                for (name, slot) in names.iter().zip(ordered.into_iter()) {
                    let Some((nb, st)) = slot else { continue };
                    if let Some(obj) = st.as_object() {
                        for i in 1..=nb {
                            if let Some(v) = obj.get(&format!("b{i}")) {
                                let mut band_val = v.clone();
                                if let Some(bobj) = band_val.as_object_mut() {
                                    bobj.insert(
                                        "description".to_string(),
                                        serde_json::json!(format!("{name}_b{i}")),
                                    );
                                }
                                map.insert(format!("b{band_idx}"), band_val);
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
                    let names_c = names.clone();
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let stats_opts = build_stats_options(&pq);
                    let n = names_c.len();
                    async move {
                        let mut set = tokio::task::JoinSet::new();
                        for (idx, cog) in cogs_c.into_iter().enumerate() {
                            let eng_c = eng.clone();
                            let mask_c = mask.clone();
                            let opts_c = stats_opts.clone();
                            let nb = cog.index.full_res().samples_per_pixel as usize;
                            set.spawn(async move {
                                let st = compute_statistics_with_options(
                                    &eng_c,
                                    &cog,
                                    max_size,
                                    Some(&mask_c),
                                    unscale,
                                    nodata,
                                    &opts_c,
                                )
                                .await?;
                                Ok::<_, EngineError>((idx, nb, st))
                            });
                        }
                        let mut ordered: Vec<Option<(usize, serde_json::Value)>> = vec![None; n];
                        while let Some(res) = set.join_next().await {
                            let (idx, nb, st) = res
                                .map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                            ordered[idx] = Some((nb, st));
                        }
                        // Merge bands in numeric order 1..=nb across assets to match TiTiler's POST /stac/statistics.
                        let mut merged = serde_json::Map::new();
                        let mut b_counter = 1usize;
                        for (name, slot) in names_c.iter().zip(ordered.into_iter()) {
                            let Some((nb, st)) = slot else { continue };
                            if let Some(obj) = st.as_object() {
                                for b in 1..=nb {
                                    if let Some(v) = obj.get(&format!("b{b}")) {
                                        let mut band_val = v.clone();
                                        if let Some(bobj) = band_val.as_object_mut() {
                                            bobj.insert(
                                                "description".to_string(),
                                                serde_json::json!(format!("{name}_b{b}")),
                                            );
                                        }
                                        merged.insert(format!("b{b_counter}"), band_val);
                                        b_counter += 1;
                                    }
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

            // GET /stac/tiles & /stac/tiles/{tms} (OGC TilesetList & Tileset, matching /cog/tiles*)
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
            (Method::GET, ["tiles", tms]) => {
                let crs_uri = if *tms == "WorldCRS84Quad" {
                    "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                } else {
                    "http://www.opengis.net/def/crs/EPSG/0/3857"
                };
                let body = serde_json::json!({
                    "title": tms,
                    "id": tms,
                    "crs": crs_uri,
                    "dataType": "map",
                })
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }

            // GET /stac/{tms}/tilejson.json, /stac/tilejson.json, /stac/{tms}/map.html, /stac/map.html
            (Method::GET, [_, "tilejson.json"]) | (Method::GET, ["tilejson.json"]) => async {
                let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
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
            (Method::GET, [_, "map.html"]) | (Method::GET, ["map.html"]) => {
                let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                let tj_url = format!("{forwarded_prefix}/stac/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
            (Method::GET, [_, "WMTSCapabilities.xml"]) | (Method::GET, ["WMTSCapabilities.xml"]) => {
                async {
                    let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                    let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                    let item = load_stac_item(&engine.stac_cache, &engine.store, &u).await?;
                    let (_, cogs, _) =
                        resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref())
                            .await?;
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let bounds = item
                        .bbox
                        .unwrap_or_else(|| crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height));
                    let coarsest = cogs[0].index.ifds.last().unwrap();
                    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
                        .max(full.height as f64 / coarsest.height.max(1) as f64);
                    let (def_min, def_max) = if tms == "WorldCRS84Quad" {
                        crs.world_crs84_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
                    } else {
                        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
                    };
                    let minz = pq.minzoom.unwrap_or(def_min);
                    let maxz = pq.maxzoom.unwrap_or(def_max).max(minz);
                    let fmt = pq.tile_format.as_deref().unwrap_or("png");
                    let scale = pq.tile_scale.unwrap_or(1);
                    let tile_q = filter_tile_template_query(&raw_query);
                    let xml = build_wmts_capabilities(
                        &host,
                        "stac",
                        tms,
                        &item.id,
                        bounds,
                        minz,
                        maxz,
                        fmt,
                        scale,
                        &tile_q,
                        &raw_query,
                        pq.use_epsg,
                    );
                    Ok((Bytes::from(xml), "application/xml"))
                }
                .await
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
                            Some(&engine),
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
                                .map(|j| CustomColormap::from_json_str(j, !pq_stac.rescale.is_empty()))
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
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
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
                let mut pt_json = serde_json::json!({
                    "coordinates": [x, y],
                    "values": all_vals,
                    "band_names": all_names,
                    "band_descriptions": all_names,
                });
                apply_point_filters(&mut pt_json, rewritten_expr.as_deref(), &pq.bidx)?;
                Ok((Bytes::from(pt_json.to_string()), "application/json"))
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
                let def_w = ((full.width as f64 * scale.min(1.0)).round() as u32).max(1);
                let def_h = ((full.height as f64 * scale.min(1.0)).round() as u32).max(1);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
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
                    .map(|j| CustomColormap::from_json_str(j, !pq_stac.rescale.is_empty()))
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let data = render_native_preview(
                    &engine, &cogs, bbox, out_w, out_h, &bands, pq_stac.resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["preview", dims_ext]) => async {
                let Some((out_w, out_h, ext_opt)) = parse_wxh_ext(dims_ext) else {
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
                    .map(|j| CustomColormap::from_json_str(j, !pq_stac.rescale.is_empty()))
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let data = render_native_preview(
                    &engine, &cogs, bbox, out_w, out_h, &bands, pq_stac.resample, fmt,
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
                    .map(|j| CustomColormap::from_json_str(j, !pq_stac.rescale.is_empty()))
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let (def_w, def_h) = aspect_preserving_dims(bbox, pq_stac.max_size);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                let data = if pq_stac.coord_crs_3857 {
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), false);
                    render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?
                } else {
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), true);
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
                let Some((out_w, out_h, ext_opt)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected {w}x{h}.{ext}".into()));
                };
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
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
                    .map(|j| CustomColormap::from_json_str(j, !pq_stac.rescale.is_empty()))
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let data = if pq_stac.coord_crs_3857 {
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), false);
                    render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?
                } else {
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), true);
                    render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?
                };
                Ok((data, ct))
            }
            .await,

            // POST /stac/feature/{w}x{h}[.{ext}] and POST /stac/feature[.{ext}]
            (Method::POST, ["feature", dims_ext]) => async {
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).ok_or_else(|| {
                    EngineError::BadRequest("expected /stac/feature/{w}x{h}.{ext}".into())
                })?;
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
                    .map(|j| CustomColormap::from_json_str(j, !pq_stac.rescale.is_empty()))
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), true);
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
                let s = pq.max_size;
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
                    .map(|j| CustomColormap::from_json_str(j, !pq_stac.rescale.is_empty()))
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let (def_w, def_h) = aspect_preserving_dims(bbox, s);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), true);
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

            // GET /cog/validate?url=...&strict=false
            (Method::GET, ["validate"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let url = pq.urls.first().map(String::as_str).unwrap_or("");
                let json = build_cog_validate(&engine, &cogs[0], url, pq.strict)?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 3. GET / POST /cog/statistics (POST supports GeoJSON Polygon clipping!)
            (Method::GET, ["statistics"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let stats_opts = build_stats_options(&pq);
                let json = compute_statistics_with_options(
                    &engine,
                    &cogs[0],
                    pq.max_size,
                    None,
                    pq.unscale,
                    pq.nodata,
                    &stats_opts,
                )
                .await?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,
            (Method::POST, ["statistics"]) => async {
                let body_bytes = read_limited_body(req.into_body()).await?;
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let stats_opts = build_stats_options(&pq);
                if body_bytes.is_empty() {
                    let json = compute_statistics_with_options(
                        &engine,
                        &cogs[0],
                        pq.max_size,
                        None,
                        pq.unscale,
                        pq.nodata,
                        &stats_opts,
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
                            let stats = compute_statistics_with_options(
                                &engine,
                                &cogs[0],
                                pq.max_size,
                                Some(&mask),
                                pq.unscale,
                                pq.nodata,
                                &stats_opts,
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
                        let stats = compute_statistics_with_options(
                            &engine,
                            &cogs[0],
                            pq.max_size,
                            Some(&mask),
                            pq.unscale,
                            pq.nodata,
                            &stats_opts,
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
                            Some(&engine),
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
                                .map(|j| CustomColormap::from_json_str(j, !pq.rescale.is_empty()))
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

            // 9. GET /cog/{tileMatrixSetId}/map.html & /cog/map.html
            (Method::GET, [_, "map.html"]) | (Method::GET, ["map.html"]) => {
                let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                let tj_url = format!("{forwarded_prefix}/cog/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
            (Method::GET, [_, "WMTSCapabilities.xml"]) | (Method::GET, ["WMTSCapabilities.xml"]) => {
                async {
                    let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                    let cogs = open_query_urls(&engine, &pq.urls).await?;
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let bounds = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let coarsest = cogs[0].index.ifds.last().unwrap();
                    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
                        .max(full.height as f64 / coarsest.height.max(1) as f64);
                    let (def_min, def_max) = if tms == "WorldCRS84Quad" {
                        crs.world_crs84_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
                    } else {
                        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
                    };
                    let minz = pq.minzoom.unwrap_or(def_min);
                    let maxz = pq.maxzoom.unwrap_or(def_max).max(minz);
                    let fmt = pq.tile_format.as_deref().unwrap_or("png");
                    let scale = pq.tile_scale.unwrap_or(1);
                    let tile_q = filter_tile_template_query(&raw_query);
                    let xml = build_wmts_capabilities(
                        &host,
                        "cog",
                        tms,
                        "Cloud Optimized GeoTIFF",
                        bounds,
                        minz,
                        maxz,
                        fmt,
                        scale,
                        &tile_q,
                        &raw_query,
                        pq.use_epsg,
                    );
                    Ok((Bytes::from(xml), "application/xml"))
                }
                .await
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
                apply_point_filters(&mut json, pq.expr_raw.as_deref(), &pq.bidx)?;
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
                let def_w = ((full.width as f64 * scale.min(1.0)).round() as u32).max(1);
                let def_h = ((full.height as f64 * scale.min(1.0)).round() as u32).max(1);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(|j| CustomColormap::from_json_str(j, !pq.rescale.is_empty())).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_native_preview(
                    &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["preview", dims_ext]) => async {
                let Some((out_w, out_h, ext_opt)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected /cog/preview/{w}x{h}.{ext}".into()));
                };
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(|j| CustomColormap::from_json_str(j, !pq.rescale.is_empty())).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_native_preview(
                    &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
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
                let custom_lut = pq.custom_colormap_json.as_deref().map(|j| CustomColormap::from_json_str(j, !pq.rescale.is_empty())).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let (def_w, def_h) = aspect_preserving_dims(bbox, pq.max_size);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let data = if pq.coord_crs_3857 {
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), false);
                    render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?
                } else {
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), true);
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
                let Some((out_w, out_h, ext_opt)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected {w}x{h}.{ext}".into()));
                };
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(|j| CustomColormap::from_json_str(j, !pq.rescale.is_empty())).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = if pq.coord_crs_3857 {
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), false);
                    render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?
                } else {
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), true);
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
                let custom_lut = pq.custom_colormap_json.as_deref().map(|j| CustomColormap::from_json_str(j, !pq.rescale.is_empty())).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let (def_w, def_h) = aspect_preserving_dims(bbox, pq.max_size);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), true);
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, Some(&mask), &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::POST, ["feature", dims_ext]) => async {
                let Some((out_w, out_h, ext_opt)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected /cog/feature/{w}x{h}.{ext}".into()));
                };
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson(&geojson)?;
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request(&pq, db, st)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(|j| CustomColormap::from_json_str(j, !pq.rescale.is_empty())).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), true);
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
        EngineError::Io(io_e) => match io_e.root() {
            e @ cogzilla_io::IoError::InvalidUri(_) => {
                plain_resp(StatusCode::BAD_REQUEST, Bytes::from(e.to_string()), "text/plain")
            }
            e @ cogzilla_io::IoError::Tiff(_) => {
                plain_resp(StatusCode::BAD_REQUEST, Bytes::from(e.to_string()), "text/plain")
            }
            e @ cogzilla_io::IoError::Forbidden(_) => {
                plain_resp(StatusCode::FORBIDDEN, Bytes::from(e.to_string()), "text/plain")
            }
            e @ cogzilla_io::IoError::HttpStatus { status: 404, .. } => {
                plain_resp(StatusCode::NOT_FOUND, Bytes::from(e.to_string()), "text/plain")
            }
            e @ cogzilla_io::IoError::Fs(io_err)
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
        },
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
        assert!(build_tile_request(None, TileCoord { z: 2, x: 3, y: 3 }, None, Some("png"), &pq, 3, u8t, false, None).is_ok());
        assert!(build_tile_request(None, TileCoord { z: 2, x: 4, y: 0 }, None, Some("png"), &pq, 3, u8t, false, None).is_err());
        assert!(build_tile_request(None, TileCoord { z: 2, x: 0, y: 4 }, None, Some("png"), &pq, 3, u8t, false, None).is_err());
        // WorldCRS84Quad at z=2 has 8x4 tiles (x: 0..=7, y: 0..=3)
        assert!(build_tile_request(None, TileCoord { z: 2, x: 7, y: 3 }, None, Some("png"), &pq, 3, u8t, true, None).is_ok());
        assert!(build_tile_request(None, TileCoord { z: 2, x: 8, y: 0 }, None, Some("png"), &pq, 3, u8t, true, None).is_err());
        assert!(build_tile_request(None, TileCoord { z: 2, x: 0, y: 4 }, None, Some("png"), &pq, 3, u8t, true, None).is_err());

        // ?reproject= is accepted and parsed independently from ?resampling=
        let pq_rp = parse_query(Some("resampling=bilinear&reproject=cubic"));
        assert!(pq_rp.errors.is_empty());
        assert_eq!(pq_rp.resample, ResampleMode::Bilinear);
        assert_eq!(pq_rp.reproject, ResampleMode::Cubic);
        assert!(validate_query(&["cog", "tiles", "WebMercatorQuad", "0", "0", "0.png"], &pq_rp).is_ok());

        // Single-band default colormap is Grayscale (matching TiTiler)
        assert_eq!(pq.colormap, ColormapName::Grayscale);

        // Form-encoded ?expression= with spaces (`+`), `%2B`, `%2A`, `%5E`, `%28`/`%29` decodes cleanly, while raw `(b1+b2)` still works
        let pq_form = parse_query(Some("expression=%28b1+-+b2%29+%2F+%28b1+%2B+b2%29"));
        assert_eq!(pq_form.expr_raw.as_deref(), Some("(b1 - b2) / (b1 + b2)"));
        let pq_mul = parse_query(Some("expression=10+%2A+log10%28b1%29"));
        assert_eq!(pq_mul.expr_raw.as_deref(), Some("10 * log10(b1)"));
        let pq_raw = parse_query(Some("expression=(b1-b2)/(b1+b2)"));
        assert_eq!(pq_raw.expr_raw.as_deref(), Some("(b1-b2)/(b1+b2)"));

        // ?format= applies on format-less tile routes parsed via parse_y_scale_ext
        let (y_parsed, sc_parsed, ext_opt) = parse_y_scale_ext("1@2x").unwrap();
        assert_eq!((y_parsed, sc_parsed, ext_opt), (1, Some(2), None));
        let pq_fmt = parse_query(Some("format=webp"));
        let (treq, ct) =
            build_tile_request(None, TileCoord { z: 2, x: 1, y: y_parsed }, sc_parsed, ext_opt, &pq_fmt, 3, u8t, false, None)
                .unwrap();
        assert_eq!(ct, "image/webp");
        assert!(matches!(treq.format, OutputFormat::Webp { .. }));

        // Invalid minzoom > maxzoom rejected on tilejson.json
        let pq_bad_z = parse_query(Some("minzoom=14&maxzoom=8"));
        assert!(validate_query(&["cog", "WebMercatorQuad", "tilejson.json"], &pq_bad_z).is_err());

        // Default dtype rescale ranges & aspect-preserving dims & hex colormaps
        assert_eq!(default_rescale_for_dtype(cogzilla_tiff::SampleType::Uint16, false), (0.0, 65535.0));
        assert_eq!(default_rescale_for_dtype(cogzilla_tiff::SampleType::Int16, true), (-32768.0, 32767.0));
        assert_eq!(aspect_preserving_dims([0.0, 0.0, 2.0, 1.0], 512), (512, 256));
        assert_eq!(aspect_preserving_dims([0.0, 0.0, 1.0, 4.0], 512), (128, 512));
        let hex_lut = cogzilla_kernel::ColormapLut::from_json_str(r##"{"0":"#112233","255":"#ff000080"}"##).unwrap();
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

        // AlgorithmFactory JSON responses (/algorithms and /algorithms/{id})
        let algs = list_algorithms_json("http://localhost:8080");
        let list = algs["algorithms"].as_array().unwrap();
        assert_eq!(list.len(), 18);
        assert_eq!(list[0]["id"], "hillshade");
        assert_eq!(
            list[0]["links"][0]["href"],
            "http://localhost:8080/algorithms/hillshade"
        );
        let hs = get_algorithm_json("hillshade").unwrap();
        assert_eq!(hs["title"], "Hillshade");
        assert_eq!(hs["inputs"]["nbands"], 1);
        assert_eq!(hs["outputs"]["dtype"], "uint8");
        assert_eq!(get_algorithm_json("normalizedIndex").unwrap()["title"], "Normalized Index");
        assert_eq!(get_algorithm_json("grayscale").unwrap()["title"], "Grayscale");
        assert_eq!(get_algorithm_json("bitonal").unwrap()["title"], "Bitonal");
        assert!(matches!(
            get_algorithm_json("nonexistent"),
            Err(EngineError::NotFound(_))
        ));

        // WMTSCapabilities.xml generation & XML escaping
        let xml = build_wmts_capabilities(
            "http://localhost:8080",
            "cog",
            "WebMercatorQuad",
            "Test & <Layer>",
            [-122.7, 37.7, -122.5, 37.8],
            8,
            10,
            "webp",
            2,
            "url=https://ex.com/a.tif&rescale=0,100",
            "url=https://ex.com/a.tif&rescale=0,100&tile_format=webp&tile_scale=2",
            false,
        );
        assert!(xml.contains("xmlns=\"http://www.opengis.net/wmts/1.0\""));
        assert!(xml.contains("Test &amp; &lt;Layer&gt;"));
        assert!(xml.contains(
            "http://localhost:8080/cog/tiles/WebMercatorQuad/{TileMatrix}/{TileCol}/{TileRow}@2x.webp?url=https://ex.com/a.tif&amp;rescale=0,100"
        ));
        assert!(xml.contains("<TileWidth>512</TileWidth>"));
        assert!(xml.contains("<ows:Identifier>8</ows:Identifier>"));
        assert!(xml.contains("<ows:Identifier>10</ows:Identifier>"));

        // Vertical colormap PNG (20x256) + orientation/width/height query validation
        let vpng = get_colormap_image("viridis", OutputFormat::Png, true).unwrap();
        assert_eq!(&vpng[0..8], b"\x89PNG\r\n\x1a\n");
        let w = u32::from_be_bytes([vpng[16], vpng[17], vpng[18], vpng[19]]);
        let h = u32::from_be_bytes([vpng[20], vpng[21], vpng[22], vpng[23]]);
        assert_eq!((w, h), (20, 256));

        let pq_wh = parse_query(Some("url=a.tif&width=320&height=240"));
        assert!(validate_query(&["cog", "preview.png"], &pq_wh).is_ok());
        assert!(validate_query(&["cog", "preview", "256x256.png"], &pq_wh).is_err());
        assert_eq!(resolve_query_dims(512, 256, Some(200), None), (200, 100));
        assert_eq!(resolve_query_dims(512, 256, None, Some(100)), (200, 100));

        let pq_ori = parse_query(Some("f=png&orientation=vertical"));
        assert!(validate_query(&["colorMaps", "viridis"], &pq_ori).is_ok());
        assert!(validate_query(&["cog", "info"], &pq_ori).is_err());

        let pq_wmts = parse_query(Some("url=a.tif&minzoom=4&maxzoom=12&tile_format=png&tile_scale=2"));
        assert!(validate_query(&["cog", "WebMercatorQuad", "WMTSCapabilities.xml"], &pq_wmts).is_ok());
        assert!(validate_query(&["mosaicjson", "WorldCRS84Quad", "WMTSCapabilities.xml"], &pq_wmts).is_err());
    }
}
