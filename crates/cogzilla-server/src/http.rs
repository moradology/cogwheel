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
    ACCESS_CONTROL_ALLOW_ORIGIN, CACHE_CONTROL, CONTENT_TYPE, ETAG, HeaderName, HeaderValue,
    IF_NONE_MATCH,
};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use tokio::net::TcpListener;

mod pgstac;
mod pixel_selection;
mod cql2_text;
mod point_expr;
mod zonal;

use crate::catalog::{MosaicJsonCatalog, SceneCatalog, load_catalog_json_str};
use crate::engine::{BandRequest, EngineError, TileEngine, TileRequest};
use crate::stac::{
    StacItem, StacSource, get_colormap_image, get_colormap_json, get_tms_json,
    list_colormaps_json, list_tms_json, resolve_stac_assets,
};
use crate::titiler::{
    PolygonMask, StatsOptions, build_cog_info, build_cog_info_geojson, build_cog_validate,
    build_map_html, build_tilejson, build_wmts_capabilities, compute_f32_band_stats,
    compute_statistics_with_options,
    encode_auto_image, encode_geotiff_f32, encode_geotiff_rgba, encode_rgba_output,
    get_algorithm_json,
    list_algorithms_json, native_bbox_dims_in_crs, query_point,
    render_crs_bbox, render_mercator_bbox, render_multi_asset_bbox, render_native_preview,
    render_wgs84_bbox,
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
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
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
            "png" | "pngraw" | "webp" | "f32" | "bin" | "raw" | "npy" | "jpg" | "jpeg" | "tif" | "tiff"
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PixelSelection {
    First,
    Highest,
    Lowest,
    Mean,
    Median,
    Stdev,
    Count,
    LastBandLow,
    LastBandHigh,
}

impl PixelSelection {
    fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "first" => Some(Self::First),
            "highest" => Some(Self::Highest),
            "lowest" => Some(Self::Lowest),
            "mean" => Some(Self::Mean),
            "median" => Some(Self::Median),
            "stdev" | "std" => Some(Self::Stdev),
            "count" => Some(Self::Count),
            "lastbandlow" => Some(Self::LastBandLow),
            "lastbandhigh" => Some(Self::LastBandHigh),
            _ => None,
        }
    }
}

#[derive(Clone)]
struct ParsedQuery {
    urls: Vec<String>,
    assets: Vec<String>,
    asset_bidx: Vec<(String, Vec<u16>)>,
    asset_exprs: Vec<(String, String)>,
    asset_as_band: bool,
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
    pixel_selection: PixelSelection,
    use_epsg: bool,
    percentiles: Vec<f64>,
    histogram_bins: Option<usize>,
    histogram_edges: Option<Vec<f64>>,
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
    shape_mask: bool,
    buffer: f32,
    alg_buffer: Option<f32>,
    coord_crs_3857: bool,
    coord_crs: Option<cogzilla_proj::CrsModel>,
    coord_crs_epsg: Option<u16>,
    dst_crs: Option<(cogzilla_proj::CrsModel, u16)>,
    bounds_crs: Option<(cogzilla_proj::CrsModel, u16)>,
    bbox_crs: Option<(cogzilla_proj::CrsModel, u16)>,
    bbox_param: Option<[f64; 4]>,
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
    contour_increment: f32,
    contour_thickness: u32,
    contour_minz: f32,
    contour_maxz: f32,
    cast_dtype: Option<String>,
    colormap_explicit: bool,
    colormap_name_set: bool,
    colormap_type_linear: bool,
    /// Parameter names present in the query, for per-route validation.
    keys: Vec<String>,
    /// Values that could not be parsed or name something unsupported.
    errors: Vec<String>,
}

fn parse_query(q: Option<&str>) -> ParsedQuery {
    let mut pq = ParsedQuery {
        urls: Vec::new(),
        assets: Vec::new(),
        asset_bidx: Vec::new(),
        asset_exprs: Vec::new(),
        asset_as_band: false,
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
        pixel_selection: PixelSelection::First,
        use_epsg: false,
        percentiles: Vec::new(),
        histogram_bins: None,
        histogram_edges: None,
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
        shape_mask: true,
        buffer: 0.0,
        alg_buffer: None,
        coord_crs_3857: false,
        coord_crs: None,
        coord_crs_epsg: None,
        dst_crs: None,
        bounds_crs: None,
        bbox_crs: None,
        bbox_param: None,
        format_param: None,
        minzoom: None,
        maxzoom: None,
        tile_format: None,
        tile_scale: None,
        algorithm: None,
        azimuth: 315.0,
        altitude: 45.0,
        z_factor: 1.0,
        threshold: 127.0,
        contour_increment: 35.0,
        contour_thickness: 1,
        contour_minz: -12000.0,
        contour_maxz: 8000.0,
        cast_dtype: None,
        colormap_explicit: false,
        colormap_name_set: false,
        colormap_type_linear: false,
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
                // Preserve literal `+` in per-asset expressions (`assets=red|expression=b1+1`).
                let dec = percent_decode_opts(v, false);
                if dec.contains('|') {
                    // TiTiler `assets=name|bidx=1,2,3`, `assets=name|indexes=1,2,3`, or `assets=name|expression=b1*2`
                    let mut parts = dec.split('|');
                    let asset_name = parts.next().map(str::trim).unwrap_or("");
                    if !asset_name.is_empty() {
                        pq.assets.push(asset_name.to_string());
                    }
                    let mut parsed = Vec::new();
                    for opt in parts {
                        if let Some(bidx_str) = opt
                            .strip_prefix("bidx=")
                            .or_else(|| opt.strip_prefix("indexes="))
                        {
                            for tok in bidx_str.split(',') {
                                match tok.trim().parse::<u16>() {
                                    Ok(b) if b >= 1 => parsed.push(b),
                                    _ => bad("expected positive integer band indices in |bidx="),
                                }
                            }
                        } else if let Some(expr_str) = opt.strip_prefix("expression=") {
                            let expr_str = expr_str.trim();
                            if !asset_name.is_empty() && !expr_str.is_empty() {
                                if let Some((_, existing)) =
                                    pq.asset_exprs.iter_mut().find(|(n, _)| n == asset_name)
                                {
                                    *existing = expr_str.to_string();
                                } else {
                                    pq.asset_exprs
                                        .push((asset_name.to_string(), expr_str.to_string()));
                                }
                            }
                        }
                    }
                    if !asset_name.is_empty() && !parsed.is_empty() {
                        if let Some((_, existing)) =
                            pq.asset_bidx.iter_mut().find(|(n, _)| n == asset_name)
                        {
                            existing.extend(parsed);
                        } else {
                            pq.asset_bidx.push((asset_name.to_string(), parsed));
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
                if let Some((name, bidx_str)) = dec.split_once('|') {
                    let name = name.trim();
                    let mut parsed = Vec::new();
                    for tok in bidx_str.split(',') {
                        match tok.trim().parse::<u16>() {
                            Ok(b) if b >= 1 => parsed.push(b),
                            _ => bad("expected positive integer band indices in asset_bidx"),
                        }
                    }
                    if !name.is_empty() {
                        if !pq.assets.iter().any(|a| a == name) {
                            pq.assets.push(name.to_string());
                        }
                        if !parsed.is_empty() {
                            if let Some((_, existing)) =
                                pq.asset_bidx.iter_mut().find(|(n, _)| n == name)
                            {
                                existing.extend(parsed);
                            } else {
                                pq.asset_bidx.push((name.to_string(), parsed));
                            }
                        }
                    }
                } else {
                    bad("expected asset_bidx=name|1,2,3");
                }
            }
            "asset_as_band" => match percent_decode(v).to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" => pq.asset_as_band = true,
                "false" | "0" | "no" => pq.asset_as_band = false,
                _ => bad("expected boolean"),
            },
            "format" | "f" => pq.format_param = Some(percent_decode(v).to_ascii_lowercase()),
            "bbox" => {
                let dec = percent_decode(v);
                if let Some((b, None)) = parse_bbox4(&dec) {
                    pq.bbox_param = Some(b);
                } else {
                    bad("expected bbox=minx,miny,maxx,maxy");
                }
            }
            "algorithm" => pq.algorithm = Some(percent_decode(v).to_ascii_lowercase()),
            "algorithm_params" => {
                let dec = percent_decode(v);
                match serde_json::from_str::<serde_json::Value>(&dec) {
                    Ok(serde_json::Value::Object(obj)) => {
                        if let Some(val) = obj.get("azimuth") {
                            match val.as_f64().map(|a| a as f32) {
                                Some(a) if a.is_finite() => {
                                    // TiTiler's HillShade uses a 90°-rotated azimuth convention
                                    // (default 45° == compass 315°); convert to compass degrees
                                    // while preserving cogzilla's native `?azimuth=` compass param.
                                    pq.azimuth = (a - 90.0).rem_euclid(360.0);
                                }
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
                        if let Some(val) = obj.get("dtype") {
                            match val.as_str() {
                                Some(dt) => pq.cast_dtype = Some(dt.to_ascii_lowercase()),
                                None => bad("dtype must be a string"),
                            }
                        }
                        if let Some(val) = obj.get("increment") {
                            match val.as_f64().map(|x| x as f32) {
                                Some(x) if x.is_finite() && x > 0.0 => pq.contour_increment = x,
                                _ => bad("increment must be a positive finite number"),
                            }
                        }
                        if let Some(val) = obj.get("thickness") {
                            match val.as_u64() {
                                Some(t) if (1..=16).contains(&t) => pq.contour_thickness = t as u32,
                                _ => bad("thickness must be an integer in 1..=16"),
                            }
                        }
                        if let Some(val) = obj.get("minz") {
                            match val.as_f64().map(|x| x as f32) {
                                Some(x) if x.is_finite() => pq.contour_minz = x,
                                _ => bad("minz must be a finite number"),
                            }
                        }
                        if let Some(val) = obj.get("buffer") {
                            match val.as_f64().map(|x| x as f32) {
                                Some(b) if b.is_finite() && (0.0..=99.0).contains(&b) => {
                                    pq.alg_buffer = Some(b);
                                }
                                _ => bad("buffer must be a number in 0..=99"),
                            }
                        }
                        if let Some(val) = obj.get("maxz") {
                            match val.as_f64().map(|x| x as f32) {
                                Some(x) if x.is_finite() => pq.contour_maxz = x,
                                _ => bad("maxz must be a finite number"),
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
            "shape_mask" => {
                pq.shape_mask = !(v.eq_ignore_ascii_case("false") || v == "0");
            }
            "buffer" => {
                if let Ok(buf) = v.parse::<f32>()
                    && buf.is_finite()
                {
                    pq.buffer = buf.clamp(0.0, 256.0);
                } else {
                    bad("expected a finite number");
                }
            }
            "coord_crs" => {
                let dec = percent_decode(v);
                match parse_coord_crs_with_epsg(&dec) {
                    Ok((crs, epsg)) => {
                        pq.coord_crs_3857 = matches!(crs, cogzilla_proj::CrsModel::Epsg3857);
                        pq.coord_crs = Some(crs);
                        pq.coord_crs_epsg = Some(epsg);
                    }
                    Err(msg) => bad(&msg),
                }
            }
            "resample" | "resampling" => {
                match parse_resample_mode(v) {
                    Some(m) => pq.resample = m,
                    None => {
                        bad("unsupported resampling method");
                        pq.resample = ResampleMode::Nearest;
                    }
                }
            }
            "reproject" => {
                match parse_resample_mode(v) {
                    Some(m) => pq.reproject = m,
                    None => {
                        bad("unsupported reproject method");
                        pq.reproject = ResampleMode::Nearest;
                    }
                }
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
                    // TiTiler treats max_size=0 as full native resolution (capped at 4096).
                    pq.max_size = if ms == 0 { 4096 } else { ms.clamp(1, 4096) };
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
                    || lower.contains("%2c+")
                    || lower.contains("+%3c")
                    || lower.contains("%3c+")
                    || lower.contains("+%3e")
                    || lower.contains("%3e+")
                    || lower.contains("+%3d")
                    || lower.contains("%3d+")
                    || lower.contains("+%21")
                    || lower.contains("%21+")
                    || lower.contains("+%3b")
                    || lower.contains("%3b+")
                    || lower.contains("+<")
                    || lower.contains("<+")
                    || lower.contains("+>")
                    || lower.contains(">+")
                    || lower.contains("+=")
                    || lower.contains("=+")
                    || lower.contains("+!")
                    || lower.contains("+;")
                    || lower.contains(";+");
                pq.expr_raw = Some(percent_decode_opts(v, plus_is_space));
            }
            "padding" => match v.parse::<u32>() {
                Ok(p) if p <= 64 => {}
                _ => bad("expected padding in 0..=64"),
            },
            "dst_crs" => {
                let dec = percent_decode(v);
                match parse_coord_crs_with_epsg(&dec) {
                    Ok(pair) => pq.dst_crs = Some(pair),
                    Err(msg) => bad(&msg),
                }
            }
            "crs" => {
                let dec = percent_decode(v);
                match parse_coord_crs_with_epsg(&dec) {
                    Ok(pair) => pq.bounds_crs = Some(pair),
                    Err(msg) => bad(&msg),
                }
            }
            "bbox-crs" | "bbox_crs" => {
                let dec = percent_decode(v);
                match parse_coord_crs_with_epsg(&dec) {
                    Ok(pair) => pq.bbox_crs = Some(pair),
                    Err(msg) => bad(&msg),
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
            "histogram_bins" => {
                let dec = percent_decode(v);
                if dec.contains(',') {
                    let mut edges = Vec::new();
                    let mut ok = true;
                    for tok in dec
                        .trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                    {
                        match tok.parse::<f64>() {
                            Ok(e) if e.is_finite() => edges.push(e),
                            _ => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    let monotonic = edges.windows(2).all(|w| w[0] < w[1]);
                    if ok && edges.len() >= 2 && edges.len() <= 1001 && monotonic {
                        pq.histogram_edges = Some(edges);
                    } else {
                        bad("expected strictly increasing histogram_bins edge list (2..=1001 edges)");
                    }
                } else {
                    match dec.trim().parse::<usize>() {
                        Ok(n) if (1..=1000).contains(&n) => pq.histogram_bins = Some(n),
                        _ => bad("expected histogram_bins in 1..=1000 or comma-separated edge list"),
                    }
                }
            }
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
            "pixel_selection" => match PixelSelection::from_str_opt(&v) {
                Some(ps) => pq.pixel_selection = ps,
                None => pq_errors.push(format!(
                    "unsupported pixel_selection '{v}' (expected first, highest, lowest, mean, median, stdev, count, lastbandlow, or lastbandhigh)"
                )),
            },
            "colormap_type" => {
                let dec = percent_decode(v);
                match dec.to_ascii_lowercase().as_str() {
                    "linear" => pq.colormap_type_linear = true,
                    "explicit" | "discrete" | "" => pq.colormap_type_linear = false,
                    _ => bad("expected colormap_type=explicit or linear"),
                }
            }
            "cover_scale" => {
                pq_errors.push(format!("parameter '{k}' is not supported by cogzilla"));
            }
            _ => {}
        }
    }
    // TiTiler precedence: colormap_name wins over inline JSON colormap when both are supplied.
    if pq.colormap_name_set {
        pq.custom_colormap_json = None;
    }
    // TiTiler HillShade/Slope crop `algorithm_params.buffer` (default 3) back out of the buffered
    // input tile; since cogzilla's Horn 3×3 already fetches its own neighbor halo seamlessly,
    // subtract that crop from `pq.buffer` so `?algorithm=hillshade&buffer=3` outputs 256×256.
    if pq.buffer > 0.0
        && pq
            .algorithm
            .as_deref()
            .is_some_and(|a| a.eq_ignore_ascii_case("hillshade") || a.eq_ignore_ascii_case("slope"))
    {
        let crop = pq.alg_buffer.unwrap_or(3.0).min(pq.buffer);
        pq.buffer = (pq.buffer - crop).max(0.0);
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
    "contours",
];

fn parse_resample_mode(raw: &str) -> Option<ResampleMode> {
    match raw {
        "nearest" => Some(ResampleMode::Nearest),
        "bilinear" => Some(ResampleMode::Bilinear),
        "cubic" => Some(ResampleMode::Cubic),
        "cubic_spline" | "cubicspline" => Some(ResampleMode::CubicSpline),
        "lanczos" => Some(ResampleMode::Lanczos),
        "average" => Some(ResampleMode::Average),
        "mode" => Some(ResampleMode::Mode),
        "gauss" => Some(ResampleMode::Gauss),
        "rms" => Some(ResampleMode::Rms),
        "min" => Some(ResampleMode::Min),
        "max" => Some(ResampleMode::Max),
        "med" => Some(ResampleMode::Med),
        "q1" => Some(ResampleMode::Q1),
        "q3" => Some(ResampleMode::Q3),
        "sum" => Some(ResampleMode::Sum),
        _ => None,
    }
}

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
            if crate::tms::lookup_tms(tms).is_none() {
                return Err(format!(
                    "tile matrix set '{tms}' is not supported (see /tileMatrixSets)"
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
    let is_ogc_map =
        matches!(segs[0], "cog" | "stac" | "mosaicjson") && segs.get(1) == Some(&"map");
    let is_tileset_meta = matches!(segs[0], "cog" | "stac" | "mosaicjson")
        && segs.get(1) == Some(&"tiles")
        && segs.len() <= 3;
    if family != "colorMaps" {
        if let Some(k) = has(&["orientation"]) {
            return Err(format!("parameter '{k}' is only supported on /colorMaps routes"));
        }
    }
    if !(route("preview") || route("bbox") || route("feature") || is_ogc_map || route("statistics") || family == "colorMaps") {
        if let Some(k) = has(&["width", "height"]) {
            return Err(format!(
                "parameter '{k}' is only supported on preview, bbox, feature, map, statistics, and colorMaps routes"
            ));
        }
    }
    if !(route("preview") || route("bbox") || route("feature") || is_ogc_map || route("statistics")) {
        if let Some(k) = has(&["dst_crs"]) {
            return Err(format!(
                "parameter '{k}' is only supported on preview, bbox, feature, map, and statistics routes"
            ));
        }
    }
    if !(route("bounds") || route("info") || is_ogc_map || is_tileset_meta) {
        if let Some(k) = has(&["crs"]) {
            return Err(format!(
                "parameter '{k}' is only supported on bounds, info, tiles, and map routes"
            ));
        }
    }
    if route("statistics") || route("asset_statistics") {
        if let Some(k) = has(&["algorithm"]) {
            if !(segs[0] == "cog" || (segs[0] == "stac" && route("statistics"))) {
                return Err(format!(
                    "parameter '{k}' is only supported on /cog/statistics and /stac/statistics routes"
                ));
            }
        }
        if !(segs[0] == "cog" || (segs[0] == "stac" && route("statistics"))) {
            if let Some(k) = has(&["expression", "expr", "index"]) {
                return Err(format!(
                    "parameter '{k}' is only supported on /cog/statistics and /stac/statistics routes"
                ));
            }
        }
    } else if let Some(k) = has(&["p", "histogram_bins", "histogram_range", "categorical", "c"]) {
        return Err(format!("parameter '{k}' is only supported on statistics routes"));
    }
    if route("point") {
        if let Some(k) = has(&["algorithm"]) {
            return Err(format!("parameter '{k}' is not supported on point routes"));
        }
    } else if !(route("bbox") || route("feature") || route("statistics")) {
        if let Some(k) = has(&["coord_crs"]) {
            return Err(format!("parameter '{k}' is not supported on this route"));
        }
    }
    if route("bbox") && segs.len() >= 4 {
        if let Some(k) = has(&["width", "height"]) {
            return Err(format!("parameter '{k}' is not supported when explicit {{w}}x{{h}} dimensions are given"));
        }
    }
    if (route("preview") || route("feature")) && segs.len() >= 3 {
        if let Some(k) = has(&["width", "height"]) {
            return Err(format!("parameter '{k}' is not supported when explicit {{w}}x{{h}} dimensions are given"));
        }
    }
    let last_seg = segs.last().copied().unwrap_or("");
    if !matches!(last_seg, "tilejson.json" | "map.html" | "WMTSCapabilities.xml") && !is_tileset_meta {
        if let Some(k) = has(&["minzoom", "maxzoom", "tile_format", "tile_scale"]) {
            return Err(format!(
                "parameter '{k}' is only supported on tilejson.json, map.html, tiles, and WMTSCapabilities.xml routes"
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
    let (new_min, new_max) = resolve_zoom_range(cur_min, cur_max, pq.minzoom, pq.maxzoom);
    tj["minzoom"] = serde_json::json!(new_min);
    tj["maxzoom"] = serde_json::json!(new_max);
    if let Some(center) = tj.get_mut("center").and_then(|v| v.as_array_mut())
        && center.len() >= 3
    {
        center[2] = serde_json::json!(new_min);
    }
}

fn resolve_zoom_range(
    def_min: u8,
    def_max: u8,
    req_min: Option<u8>,
    req_max: Option<u8>,
) -> (u8, u8) {
    match (req_min, req_max) {
        (Some(mn), Some(mx)) => (mn, mx.max(mn)),
        (Some(mn), None) => (mn, def_max.max(mn)),
        (None, Some(mx)) => (def_min.min(mx), mx),
        (None, None) => (def_min, def_max),
    }
}

fn apply_info_bounds_crs(
    info: &mut serde_json::Value,
    wgs84_bounds: [f64; 4],
    bounds_crs: Option<&(cogzilla_proj::CrsModel, u16)>,
) -> Result<(), EngineError> {
    if let Some((dst_crs, dst_epsg)) = bounds_crs {
        let proj_bounds = reproject_wgs84_bounds(wgs84_bounds, dst_crs)?;
        info["bounds"] = serde_json::json!(proj_bounds);
        info["crs"] = serde_json::json!(format!(
            "http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"
        ));
    }
    Ok(())
}

const GEOTIFF_CONTENT_TYPE: &str = "image/tiff; application=geotiff";

/// Parse a TiTiler CRS string (`EPSG:32610`, `EPSG:900913`, `CRS84`, `http://.../EPSG/0/32610`,
/// `urn:ogc:def:crs:EPSG::32610`, etc.) into `(CrsModel, epsg_code)` using our 3,260+ EPSG table.
fn parse_coord_crs_with_epsg(raw: &str) -> Result<(cogzilla_proj::CrsModel, u16), String> {
    let raw_trimmed = raw.trim();
    let trimmed = raw_trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(raw_trimmed)
        .trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower == "crs84" || lower.ends_with(":crs84") || lower.ends_with("/crs84") {
        return Ok((cogzilla_proj::CrsModel::Geographic, 4326));
    }
    let digits = if let Some(rest) = lower.strip_prefix("epsg:") {
        rest.trim()
    } else if let Some(pos) = lower.find("epsg:") {
        // e.g. urn:ogc:def:crs:epsg::32610 or urn:ogc:def:crs:epsg:6.3:32610
        lower[pos + 5..].rsplit(':').next().unwrap_or("").trim()
    } else if let Some(pos) = lower.find("/epsg/") {
        // e.g. http://www.opengis.net/def/crs/EPSG/0/32610
        lower[pos + 6..].rsplit('/').next().unwrap_or("").trim()
    } else {
        return Err(format!(
            "unsupported CRS '{raw}' (expected EPSG:<code> or CRS84)"
        ));
    };
    if digits == "900913" {
        return Ok((cogzilla_proj::CrsModel::Epsg3857, 3857));
    }
    let Ok(code) = digits.parse::<u16>() else {
        return Err(format!("unsupported CRS '{raw}' (expected EPSG:<code> or CRS84)"));
    };
    let model = cogzilla_proj::CrsModel::from_epsg(code)
        .map_err(|e| format!("unsupported CRS '{raw}': {e}"))?;
    Ok((model, code))
}

fn dataset_native_crs_and_bounds(
    engine: &TileEngine,
    cog: &cogzilla_io::OpenedCog,
) -> Result<(cogzilla_proj::CrsModel, u16, [f64; 4]), EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let full = cog.index.full_res();
    let gk = &cog.index.geo_keys;
    let std_proj_epsg = gk.projected_cs_type.filter(|&c| c > 0 && c != 32767);
    let is_user_defined_projected = std_proj_epsg.is_none()
        && (gk.model_type == Some(1)
            || gk.projected_cs_type == Some(32767)
            || gk.proj_coord_trans.is_some());
    let native_epsg = if let Some(epsg) = std_proj_epsg {
        epsg
    } else if !is_user_defined_projected
        && let Some(epsg) = gk.geographic_type.filter(|&c| c > 0 && c != 32767)
    {
        epsg
    } else {
        4326
    };
    let (x00, y00) = full.pixel_to_crs.forward(0.0, 0.0);
    let (x10, y10) = full.pixel_to_crs.forward(full.width as f64, 0.0);
    let (x01, y01) = full.pixel_to_crs.forward(0.0, full.height as f64);
    let (x11, y11) = full.pixel_to_crs.forward(full.width as f64, full.height as f64);
    let native_bounds = [
        x00.min(x10).min(x01).min(x11),
        y00.min(y10).min(y01).min(y11),
        x00.max(x10).max(x01).max(x11),
        y00.max(y10).max(y01).max(y11),
    ];
    Ok((crs, native_epsg, native_bounds))
}

/// Transform a WGS84 `[min_lon, min_lat, max_lon, max_lat]` bbox into `dst` CRS by densifying
/// along the 4 edges plus an interior lattice so curved/limited-domain projections (e.g. UTM,
/// NZTM, LAEA on global COGs) preserve their full projected envelope.
fn reproject_wgs84_bounds(
    wgs84: [f64; 4],
    dst: &cogzilla_proj::CrsModel,
) -> Result<[f64; 4], EngineError> {
    if matches!(dst, cogzilla_proj::CrsModel::Geographic) {
        return Ok(wgs84);
    }
    let [w, mut s, e, mut n] = wgs84;
    // TransverseMercator::forward_raw rejects |lat| > 1.56 rad (~89.38°); clamp polar edges
    // to [-89.35°, 89.35°] so global COGs don't drop the 67.5°..89.3° band, while rejecting
    // bboxes wholly outside the TM domain (s >= n after clamping).
    if matches!(dst, cogzilla_proj::CrsModel::TransverseMercator(_)) {
        const MAX_TM_LAT: f64 = 89.35;
        s = s.clamp(-MAX_TM_LAT, MAX_TM_LAT);
        n = n.clamp(-MAX_TM_LAT, MAX_TM_LAT);
        if s >= n {
            return Err(EngineError::BadRequest(
                "bounds could not be projected to target CRS".into(),
            ));
        }
    }
    let antimeridian = e < w;
    let crosses_180 = antimeridian || e > 180.0;
    let lon_span = if antimeridian { (e + 360.0) - w } else { e - w };
    let wrap_lon = |lo: f64| if lo > 180.0 { lo - 360.0 } else { lo };
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut accum = |lo: f64, la: f64| {
        let (x, y) = dst.from_lonlat_deg(lo, la);
        if x.is_finite() && y.is_finite() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    };
    const SAMPLES: usize = 16;
    for i in 0..=SAMPLES {
        let t = i as f64 / SAMPLES as f64;
        let lon = wrap_lon(w + lon_span * t);
        let lat = s + (n - s) * t;
        accum(lon, s);
        accum(lon, n);
        accum(wrap_lon(w), lat);
        accum(wrap_lon(e), lat);
        if crosses_180 {
            accum(180.0, lat);
            accum(-180.0, lat);
        }
    }
    const INTERIOR: usize = 8;
    for iy in 1..INTERIOR {
        let lat = s + (n - s) * (iy as f64 / INTERIOR as f64);
        for ix in 1..INTERIOR {
            let lon = wrap_lon(w + lon_span * (ix as f64 / INTERIOR as f64));
            accum(lon, lat);
        }
    }
    if !min_x.is_finite() || min_x >= max_x || min_y >= max_y {
        return Err(EngineError::BadRequest(
            "bounds could not be projected to target CRS".into(),
        ));
    }
    Ok([min_x, min_y, max_x, max_y])
}

/// Validate a target rendering bbox and unwrap antimeridian-crossing geographic bboxes (`w > e`)
/// into `[w, s, e + 360, n]` so aspect ratio, `ModelPixelScale`, and pixel stepping are positive.
fn normalize_render_bbox(
    mut bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
) -> Result<[f64; 4], EngineError> {
    if !bbox.iter().all(|v| v.is_finite()) || bbox[1] >= bbox[3] {
        return Err(EngineError::BadRequest(format!(
            "invalid bbox: [{}, {}, {}, {}]",
            bbox[0], bbox[1], bbox[2], bbox[3]
        )));
    }
    if matches!(dst_crs, cogzilla_proj::CrsModel::Geographic) && bbox[0] > bbox[2] {
        bbox[2] += 360.0;
    }
    if bbox[0] >= bbox[2] {
        return Err(EngineError::BadRequest(format!(
            "invalid bbox: [{}, {}, {}, {}]",
            bbox[0], bbox[1], bbox[2], bbox[3]
        )));
    }
    Ok(bbox)
}

/// Return the standard EPSG code for a COG's GeoKeys, or `None` for user-defined projected CRSes
/// (`ProjectedCSTypeGeoKey == 32767` or `ModelType == 1` / `ProjCoordTrans` without a standard EPSG).
fn cog_standard_epsg(gk: &cogzilla_tiff::GeoKeys) -> Option<u16> {
    if let Some(epsg) = gk.projected_cs_type.filter(|&c| c > 0 && c != 32767) {
        return Some(epsg);
    }
    if gk.model_type == Some(1)
        || gk.projected_cs_type == Some(32767)
        || gk.proj_coord_trans.is_some()
    {
        return None;
    }
    Some(
        gk.geographic_type
            .filter(|&c| c > 0 && c != 32767)
            .unwrap_or(4326),
    )
}

/// Return the COG's native `(standard_epsg_opt, [minx, miny, maxx, maxy])` in its own CRS.
fn cog_native_epsg_and_bounds(cog: &cogzilla_io::OpenedCog) -> (Option<u16>, [f64; 4]) {
    let full = cog.index.full_res();
    let epsg = cog_standard_epsg(&cog.index.geo_keys);
    let w = full.width as f64;
    let h = full.height as f64;
    let corners = [
        full.pixel_to_crs.forward(0.0, 0.0),
        full.pixel_to_crs.forward(w, 0.0),
        full.pixel_to_crs.forward(0.0, h),
        full.pixel_to_crs.forward(w, h),
    ];
    let min_x = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
    let max_x = corners.iter().map(|c| c.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let max_y = corners.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
    (epsg, [min_x, min_y, max_x, max_y])
}

fn coord_to_lonlat(
    x: f64,
    y: f64,
    coord_crs: Option<&cogzilla_proj::CrsModel>,
) -> Result<(f64, f64), EngineError> {
    let (lon, lat) = match coord_crs {
        None | Some(cogzilla_proj::CrsModel::Geographic) => (x, y),
        Some(cogzilla_proj::CrsModel::Epsg3857) => mercator_to_lonlat_deg(x, y),
        Some(crs) => crs.to_lonlat_deg(x, y),
    };
    if !lon.is_finite() || !lat.is_finite() {
        return Err(EngineError::BadRequest(
            "point coordinates could not be transformed to WGS84".into(),
        ));
    }
    Ok((lon, lat))
}

fn bbox_to_wgs84(
    bbox: [f64; 4],
    coord_crs: Option<&cogzilla_proj::CrsModel>,
) -> Result<[f64; 4], EngineError> {
    let Some(crs) = coord_crs else {
        return Ok(bbox);
    };
    if matches!(crs, cogzilla_proj::CrsModel::Geographic) {
        return Ok(bbox);
    }
    let [x0, y0, x1, y1] = bbox;
    let mx = 0.5 * (x0 + x1);
    let my = 0.5 * (y0 + y1);
    let pts = [
        (x0, y0),
        (x1, y0),
        (x0, y1),
        (x1, y1),
        (mx, y0),
        (mx, y1),
        (x0, my),
        (x1, my),
    ];
    let (center_lon, center_lat) = crs.to_lonlat_deg(mx, my);
    let mut ref_lon = if center_lon.is_finite() && center_lat.is_finite() {
        Some(center_lon)
    } else {
        None
    };
    let mut min_lon = f64::INFINITY;
    let mut min_lat = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    for (x, y) in pts {
        let (lo, la) = crs.to_lonlat_deg(x, y);
        if lo.is_finite() && la.is_finite() {
            let r = *ref_lon.get_or_insert(lo);
            let mut delta = lo - r;
            if delta > 180.0 {
                delta -= 360.0;
            } else if delta < -180.0 {
                delta += 360.0;
            }
            let unwrapped = r + delta;
            min_lon = min_lon.min(unwrapped);
            max_lon = max_lon.max(unwrapped);
            min_lat = min_lat.min(la);
            max_lat = max_lat.max(la);
        }
    }
    if min_lon < -180.0 {
        min_lon += 360.0;
        max_lon += 360.0;
    }
    // Polar-centered projected bboxes (e.g. EPSG:3413 / EPSG:3031) contain a pole in their interior:
    // if either pole projects inside [x0..=x1, y0..=y1], span full [-180, 180] and extend to ±90°.
    if !matches!(
        crs,
        cogzilla_proj::CrsModel::Geographic | cogzilla_proj::CrsModel::Epsg3857
    ) {
        let (bx0, bx1) = (x0.min(x1), x0.max(x1));
        let (by0, by1) = (y0.min(y1), y0.max(y1));
        for pole_lat in [90.0_f64, -90.0_f64] {
            let (px, py) = crs.from_lonlat_deg(0.0, pole_lat);
            if px.is_finite()
                && py.is_finite()
                && px >= bx0
                && px <= bx1
                && py >= by0
                && py <= by1
            {
                min_lon = -180.0;
                max_lon = 180.0;
                if pole_lat > 0.0 {
                    max_lat = 90.0;
                } else {
                    min_lat = -90.0;
                }
            }
        }
    }
    if !min_lon.is_finite() || min_lon >= max_lon || min_lat >= max_lat {
        return Err(EngineError::BadRequest(
            "bbox coordinates could not be transformed to WGS84".into(),
        ));
    }
    Ok([min_lon, min_lat, max_lon, max_lat])
}

/// Query a MosaicJSON's WebMercator quadkey index for a WGS84 bbox, splitting antimeridian-crossing
/// bboxes (`e > 180` or `w > e`) into eastern and western hemisphere halves deduplicated in order.
fn query_mosaicjson_wgs84_bbox(mj: &crate::catalog::MosaicJsonCatalog, wgs_bbox: [f64; 4]) -> Vec<String> {
    let [w, s, e, n] = wgs_bbox;
    let query_half = |lo0: f64, lo1: f64| {
        let (mx0, my0) = lonlat_deg_to_mercator(lo0, s);
        let (mx1, my1) = lonlat_deg_to_mercator(lo1, n);
        mj.query_bbox_3857(mx0.min(mx1), my0.min(my1), mx0.max(mx1), my0.max(my1))
    };
    if e > 180.0 || w > e {
        let e_wrapped = if e > 180.0 { e - 360.0 } else { e };
        let mut urls = query_half(w.clamp(-180.0, 180.0), 180.0);
        for u in query_half(-180.0, e_wrapped.clamp(-180.0, 180.0)) {
            if !urls.contains(&u) {
                urls.push(u);
            }
        }
        urls
    } else {
        query_half(w, e)
    }
}

/// Check whether a requested tile intersects the dataset's WGS84 footprint (used to return
/// TiTiler-compatible `404 Not Found` on out-of-bounds `/cog/tiles` and `/stac/tiles` requests).
fn tile_intersects_cog(
    engine: &TileEngine,
    cog: &cogzilla_io::OpenedCog,
    tms: &str,
    tile: TileCoord,
    buffer: f32,
    tile_size: u32,
) -> Result<bool, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let full = cog.index.full_res();
    let [cw, cs, ce, cn] = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
    if !cw.is_finite() || !cs.is_finite() || !ce.is_finite() || !cn.is_finite() {
        return Ok(true);
    }
    let [tw, ts, te, tn] = if let Some(tms_def) = crate::tms::lookup_tms(tms) {
        let Some(tb) = tms_def.tile_bounds(tile.z, tile.x, tile.y, buffer, tile_size) else {
            return Ok(false);
        };
        let Ok(dst_crs) = tms_def.crs_model() else {
            return Ok(true);
        };
        if !matches!(
            dst_crs,
            cogzilla_proj::CrsModel::Geographic | cogzilla_proj::CrsModel::Epsg3857
        ) {
            // Check intersection directly in the TMS's native projected CRS so polar
            // (UPSArctic/Antarctic) and antimeridian (LINZ/NZTM) tiles never falsely 404.
            return match reproject_wgs84_bounds([cw, cs, ce, cn], &dst_crs) {
                Ok([cx0, cy0, cx1, cy1]) => {
                    Ok(tb[2] >= cx0 && tb[0] <= cx1 && tb[3] >= cy0 && tb[1] <= cy1)
                }
                Err(_) => Ok(true),
            };
        }
        if matches!(dst_crs, cogzilla_proj::CrsModel::Epsg3857) {
            let (lon0, lat0) = mercator_to_lonlat_deg(tb[0], tb[1]);
            let (lon1, lat1) = mercator_to_lonlat_deg(tb[2], tb[3]);
            let (ts, tn) = (lat0.min(lat1), lat0.max(lat1));
            if tn < cs || ts > cn {
                return Ok(false);
            }
            let (raw_w, raw_e) = (lon0.min(lon1), lon0.max(lon1));
            let overlaps = |w: f64, e: f64| -> bool {
                if cw <= ce {
                    e >= cw && w <= ce
                } else {
                    e >= cw || w <= ce
                }
            };
            let primary = overlaps(raw_w.max(-180.0), raw_e.min(180.0));
            let wrap_west = raw_w < -180.0 && overlaps((raw_w + 360.0).max(-180.0), 180.0);
            let wrap_east = raw_e > 180.0 && overlaps(-180.0, (raw_e - 360.0).min(180.0));
            return Ok(primary || wrap_west || wrap_east);
        }
        match bbox_to_wgs84(tb, Some(&dst_crs)) {
            Ok(w) => w,
            Err(_) => return Ok(true),
        }
    } else {
        let [mx0, my0, mx1, my1] =
            web_mercator_buffered_3857_bbox(tile.z, tile.x, tile.y, buffer, tile_size);
        let (lon0, lat0) = mercator_to_lonlat_deg(mx0, my0);
        let (lon1, lat1) = mercator_to_lonlat_deg(mx1, my1);
        [lon0.min(lon1), lat0.min(lat1), lon0.max(lon1), lat0.max(lat1)]
    };
    if tn < cs || ts > cn {
        return Ok(false);
    }
    let lon_overlaps = if cw <= ce {
        te >= cw && tw <= ce
    } else {
        // Antimeridian-crossing dataset: [cw..180] U [-180..ce]
        te >= cw || tw <= ce
    };
    Ok(lon_overlaps)
}

/// Return true when a `.tif`/`.tiff` request should emit a native IEEE Float32 GeoTIFF
/// (`SampleFormat=3`, `GDAL_NODATA="nan"`) instead of an 8-bit visual RGB/RGBA GeoTIFF.
fn wants_f32_geotiff(
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_tiff::SampleType,
) -> bool {
    if ct != GEOTIFF_CONTENT_TYPE {
        return false;
    }
    if !pq.rescale.is_empty()
        || pq.colormap_explicit
        || pq.custom_colormap_json.is_some()
        || pq.color_formula.is_some()
    {
        return false;
    }
    if source_cog.is_some_and(|c| c.index.color_map.is_some()) {
        return false;
    }
    match bands {
        BandRequest::Terrain { .. } => false,
        BandRequest::Dem { algorithm, .. } => !matches!(
            algorithm,
            cogzilla_kernel::DemAlgorithm::Hillshade { .. }
                | cogzilla_kernel::DemAlgorithm::Contours { .. }
        ),
        BandRequest::Rgb { .. } | BandRequest::Single { .. } => {
            !matches!(sample_type, cogzilla_tiff::SampleType::Uint8)
        }
        BandRequest::CustomExpr { .. }
        | BandRequest::CustomExprRgb { .. }
        | BandRequest::MultiBand { .. }
        | BandRequest::Spectral1 { .. }
        | BandRequest::Spectral2 { .. }
        | BandRequest::Spectral3 { .. }
        | BandRequest::SarRgb { .. } => true,
    }
}

fn geotiff_render_format(
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_tiff::SampleType,
) -> OutputFormat {
    if wants_f32_geotiff(ct, pq, source_cog, bands, sample_type) {
        OutputFormat::RawF32
    } else {
        fmt
    }
}

fn wrap_geotiff_fmt(
    data: Bytes,
    ct: &'static str,
    fmt: OutputFormat,
    w: u32,
    h: u32,
    bbox: [f64; 4],
    epsg: u16,
    return_mask: bool,
) -> Bytes {
    if ct == GEOTIFF_CONTENT_TYPE {
        if matches!(fmt, OutputFormat::RawF32) {
            encode_geotiff_f32(w, h, &data, bbox, epsg, return_mask)
        } else {
            encode_geotiff_rgba(w, h, &data, bbox, epsg, return_mask)
        }
    } else {
        data
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_image_output(
    data: Bytes,
    ct: &'static str,
    fmt: OutputFormat,
    auto_fmt: bool,
    out_w: u32,
    out_h: u32,
    bbox: [f64; 4],
    epsg: u16,
    jpeg_quality: u8,
    return_mask: bool,
) -> (Bytes, &'static str) {
    if auto_fmt {
        encode_auto_image(out_w as usize, out_h as usize, &data, jpeg_quality, return_mask)
    } else {
        (wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, bbox, epsg, return_mask), ct)
    }
}

fn attach_x_assets(resp: &mut Response<Full<Bytes>>, assets: &[String]) {
    if assets.is_empty() || !resp.status().is_success() {
        return;
    }
    let joined = assets.join(",");
    if let Ok(hv) = hyper::header::HeaderValue::from_str(&joined) {
        resp.headers_mut().insert("x-assets", hv);
    }
}

fn parse_ext_format(
    ext: &str,
    webp_quality: Option<f32>,
    jpeg_quality: u8,
) -> Result<(OutputFormat, &'static str), EngineError> {
    match ext {
        "png" | "pngraw" | "" => Ok((OutputFormat::Png, "image/png")),
        "tif" | "tiff" => Ok((OutputFormat::RawRgba, GEOTIFF_CONTENT_TYPE)),
        "webp" => Ok((
            OutputFormat::Webp {
                quality: webp_quality,
            },
            "image/webp",
        )),
        "jpg" | "jpeg" => Ok((OutputFormat::Jpeg { quality: jpeg_quality }, "image/jpeg")),
        "f32" | "bin" | "raw" => Ok((OutputFormat::RawF32, "application/octet-stream")),
        "npy" => Ok((OutputFormat::Npy, "application/x-binary")),
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

/// Compute native-resolution `(width, height)` for `dst_bbox` capped at `max_dim`,
/// matching TiTiler / GDAL `WarpedVRT` without upsampling past native pixel resolution.
fn capped_native_bbox_dims(
    engine: &TileEngine,
    cog: &cogzilla_io::OpenedCog,
    wgs_bbox: [f64; 4],
    dst_bbox: [f64; 4],
    max_dim: u32,
) -> (u32, u32) {
    let cap = max_dim.clamp(1, 4096);
    let Ok(crs) = engine.get_or_build_crs(cog) else {
        return aspect_preserving_dims(dst_bbox, cap);
    };
    let (nat_w, nat_h) = native_bbox_dims_in_crs(cog.index.full_res(), &crs, wgs_bbox, dst_bbox);
    if nat_w.max(nat_h) <= cap {
        (nat_w.max(1), nat_h.max(1))
    } else if nat_w >= nat_h {
        let h = ((nat_h as f64 / nat_w as f64) * cap as f64)
            .ceil()
            .clamp(1.0, cap as f64) as u32;
        (cap, h)
    } else {
        let w = ((nat_w as f64 / nat_h as f64) * cap as f64)
            .ceil()
            .clamp(1.0, cap as f64) as u32;
        (w, cap)
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
            "contours" => {
                return Ok(BandRequest::Terrain {
                    band_1based,
                    encoding: TerrainEncoding::Contours {
                        increment: pq.contour_increment.round() as i32,
                        thickness: pq.contour_thickness.min(255) as u8,
                        minz: pq.contour_minz.round() as i32,
                        maxz: pq.contour_maxz.round() as i32,
                    },
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
                        // Match TiTiler's NormalizedIndex: (b2 - b1) / (b2 + b1)
                        (
                            format!(
                                "(b{} - b{}) / (b{} + b{})",
                                bands[1], bands[0], bands[1], bands[0]
                            ),
                            (-1.0, 1.0),
                        )
                    }
                    "cast" => {
                        let cast_one = |b: u16| -> (String, (f32, f32)) {
                            match pq.cast_dtype.as_deref() {
                                None | Some("uint8") => {
                                    (format!("clip(floor(b{b}), 0, 255)"), (0.0, 255.0))
                                }
                                Some("int8") => {
                                    (format!("clip(floor(b{b}), -128, 127)"), (-128.0, 127.0))
                                }
                                Some("uint16") => {
                                    (format!("clip(floor(b{b}), 0, 65535)"), (0.0, 65535.0))
                                }
                                Some("int16") => {
                                    (format!("clip(floor(b{b}), -32768, 32767)"), (-32768.0, 32767.0))
                                }
                                Some("uint32") => {
                                    (format!("max(floor(b{b}), 0)"), (0.0, 4_294_967_295.0))
                                }
                                Some("int32") => {
                                    (format!("floor(b{b})"), (-2_147_483_648.0, 2_147_483_647.0))
                                }
                                _ => (format!("b{b}"), default_rescale_for_dtype(sample_type, true)),
                            }
                        };
                        if bands.len() >= 3 {
                            let (e0, dr) = cast_one(bands[0]);
                            let (e1, _) = cast_one(bands[1]);
                            let (e2, _) = cast_one(bands[2]);
                            let progs = [
                                ExprProgram::compile(&e0).map_err(EngineError::BadRequest)?,
                                ExprProgram::compile(&e1).map_err(EngineError::BadRequest)?,
                                ExprProgram::compile(&e2).map_err(EngineError::BadRequest)?,
                            ];
                            let r0 = pq.rescale.first().copied().unwrap_or(dr);
                            let r1 = pq.rescale.get(1).copied().unwrap_or(r0);
                            let r2 = pq.rescale.get(2).copied().unwrap_or(r0);
                            return Ok(BandRequest::CustomExprRgb {
                                progs,
                                rescale_min: [r0.0, r1.0, r2.0],
                                rescale_max: [r0.1, r1.1, r2.1],
                            });
                        }
                        cast_one(bands[0])
                    }
                    "ceil" | "floor" => {
                        let dr = (0.0, 255.0);
                        if bands.len() >= 3 {
                            let progs = [
                                ExprProgram::compile(&format!("{alg}(b{})", bands[0])).map_err(EngineError::BadRequest)?,
                                ExprProgram::compile(&format!("{alg}(b{})", bands[1])).map_err(EngineError::BadRequest)?,
                                ExprProgram::compile(&format!("{alg}(b{})", bands[2])).map_err(EngineError::BadRequest)?,
                            ];
                            let r0 = pq.rescale.first().copied().unwrap_or(dr);
                            let r1 = pq.rescale.get(1).copied().unwrap_or(r0);
                            let r2 = pq.rescale.get(2).copied().unwrap_or(r0);
                            return Ok(BandRequest::CustomExprRgb {
                                progs,
                                rescale_min: [r0.0, r1.0, r2.0],
                                rescale_max: [r0.1, r1.1, r2.1],
                            });
                        }
                        (format!("{alg}(b{})", bands[0]), dr)
                    }
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
                        let args = bands
                            .iter()
                            .map(|b| format!("b{b}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        (format!("{alg}({args})"), default_rescale_for_dtype(sample_type, true))
                    }
                    "median" => {
                        let args = bands
                            .iter()
                            .map(|b| format!("b{b}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        (format!("median({args})"), default_rescale_for_dtype(sample_type, true))
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
                    "bitonal" => {
                        let src_expr = if bands.len() >= 3 {
                            format!(
                                "(0.299 * b{} + 0.587 * b{} + 0.114 * b{})",
                                bands[0], bands[1], bands[2]
                            )
                        } else {
                            format!("b{}", bands[0])
                        };
                        (
                            format!("where({src_expr} > {}, 255, 0)", pq.threshold),
                            (0.0, 255.0),
                        )
                    }
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
        // Supports single expressions (`(b1-b2)/(b1+b2)`) and `;`-separated 3-band RGB
        // expressions (`b3*2;b2*2;b1*2`).
        let parts: Vec<&str> = raw_expr
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() == 3 {
            if pq.colormap_explicit || pq.custom_colormap_json.is_some() {
                return Err(EngineError::BadRequest(
                    "colormap cannot be applied to 3-band RGB output".into(),
                ));
            }
            let p0 = ExprProgram::compile(parts[0]).map_err(EngineError::BadRequest)?;
            let p1 = ExprProgram::compile(parts[1]).map_err(EngineError::BadRequest)?;
            let p2 = ExprProgram::compile(parts[2]).map_err(EngineError::BadRequest)?;
            let (rmin, rmax) = match pq.rescale.len() {
                0 => {
                    let (lo, hi) = default_rescale_for_dtype(sample_type, false);
                    ([lo; 3], [hi; 3])
                }
                3 => (
                    [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                    [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
                ),
                // rio-tiler applies rescale[0] across all bands when len != count
                _ => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
            };
            return Ok(BandRequest::CustomExprRgb {
                progs: [p0, p1, p2],
                rescale_min: rmin,
                rescale_max: rmax,
            });
        } else if parts.len() != 1 {
            return Err(EngineError::BadRequest(format!(
                "multi-band tile expression must have 1 or 3 semicolon-separated expressions (got {})",
                parts.len()
            )));
        }
        let prog = ExprProgram::compile(parts[0]).map_err(EngineError::BadRequest)?;
        let rescale = pq.rescale.first().copied().unwrap_or((-1.0, 1.0));
        return Ok(BandRequest::CustomExpr {
            prog,
            rescale,
            colormap: pq.colormap,
        });
    }

    if !pq.bidx.is_empty() && pq.bidx.len() != 1 && pq.bidx.len() != 3 {
        if pq.bidx.len() > 15 {
            return Err(EngineError::BadRequest(format!(
                "bidx count {} exceeds maximum of 15 bands",
                pq.bidx.len()
            )));
        }
        return Ok(BandRequest::MultiBand {
            bands_1based: pq.bidx.clone(),
        });
    }

    if pq.bidx.len() == 3 || (pq.bidx.is_empty() && default_bands >= 3) {
        if pq.colormap_explicit || pq.custom_colormap_json.is_some() {
            return Err(EngineError::BadRequest(
                "colormap cannot be applied to 3-band RGB output".into(),
            ));
        }
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
            3 => (
                [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
            ),
            // rio-tiler applies rescale[0] across all bands when len != count
            _ => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
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
    target_epsg: u16,
) -> ResampleMode {
    let Some(cog) = source_cog else {
        return pq.resample;
    };
    let gk = &cog.index.geo_keys;
    let is_projected = gk.model_type == Some(1)
        || gk.projected_cs_type.is_some()
        || gk.proj_coord_trans.is_some();
    if (target_epsg != 32767 && gk.projected_cs_type == Some(target_epsg))
        || (!is_projected && gk.geographic_type == Some(target_epsg))
    {
        return pq.resample;
    }
    let crs_res = if let Some(eng) = engine {
        eng.get_or_build_crs(cog).ok()
    } else {
        cogzilla_proj::CrsModel::from_geo_keys(gk).ok()
    };
    let same_crs = match crs_res {
        Some(cogzilla_proj::CrsModel::Epsg3857) => target_epsg == 3857,
        Some(cogzilla_proj::CrsModel::Geographic) => matches!(
            cogzilla_proj::CrsModel::from_epsg(target_epsg),
            Ok(cogzilla_proj::CrsModel::Geographic)
        ),
        _ => false,
    };
    if same_crs { pq.resample } else { pq.reproject }
}

fn build_ogc_tileset_json(
    host: &str,
    forwarded_prefix: &str,
    route_prefix: &str,
    tms_def: &crate::tms::TmsDef,
    raw_query: &str,
    bounds: Option<[f64; 4]>,
    minzoom: Option<u8>,
    maxzoom: Option<u8>,
) -> serde_json::Value {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        format!("{host}{forwarded_prefix}")
    } else {
        format!("http://{host}{forwarded_prefix}")
    };
    let q_suffix = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let [w, s, e, n] = bounds.unwrap_or([-180.0, -90.0, 180.0, 90.0]);
    let limits: Vec<serde_json::Value> = tms_def
        .matrices
        .iter()
        .filter(|m| {
            let z = m.id.parse::<u8>().unwrap_or(0);
            minzoom.is_none_or(|mn| z >= mn) && maxzoom.is_none_or(|mx| z <= mx)
        })
        .map(|m| {
            serde_json::json!({
                "tileMatrix": m.id,
                "minTileRow": 0,
                "maxTileRow": m.matrix_height.saturating_sub(1),
                "minTileCol": 0,
                "maxTileCol": m.matrix_width.saturating_sub(1),
            })
        })
        .collect();
    serde_json::json!({
        "title": tms_def.id,
        "id": tms_def.id,
        "tileMatrixSetId": tms_def.id,
        "tileMatrixSetURI": format!("http://www.opengis.net/def/tilematrixset/OGC/1.0/{}", tms_def.id),
        "dataType": "map",
        "crs": tms_def.crs_uri,
        "accessConstraints": "unclassified",
        "boundingBox": {
            "lowerLeft": [w, s],
            "upperRight": [e, n],
            "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
        },
        "tileMatrixSetLimits": limits,
        "links": [
            {
                "href": format!("{origin}/{route_prefix}/tiles/{}{q_suffix}", tms_def.id),
                "rel": "self",
                "type": "application/json",
                "title": tms_def.id,
            },
            {
                "href": format!("{origin}/tileMatrixSets/{}", tms_def.id),
                "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                "type": "application/json",
                "title": format!("Definition of '{}' tile matrix set", tms_def.id),
            },
            {
                "href": format!("{origin}/{route_prefix}/tiles/{}/{{tileMatrix}}/{{tileCol}}/{{tileRow}}{q_suffix}", tms_def.id),
                "rel": "item",
                "type": "image/png",
                "title": "Tiles",
                "templated": true,
            },
        ],
    })
}

fn build_ogc_tilesets_list(
    host: &str,
    forwarded_prefix: &str,
    route_prefix: &str,
    raw_query: &str,
    bounds: Option<[f64; 4]>,
) -> serde_json::Value {
    let tilesets: Vec<serde_json::Value> = crate::tms::ALL_TMS
        .iter()
        .map(|t| {
            build_ogc_tileset_json(
                host,
                forwarded_prefix,
                route_prefix,
                t,
                raw_query,
                bounds,
                None,
                None,
            )
        })
        .collect();
    serde_json::json!({ "tilesets": tilesets })
}

fn method_not_allowed_resp(allow: &'static str) -> Response<Full<Bytes>> {
    let body = serde_json::json!({ "detail": "Method Not Allowed" }).to_string();
    let mut resp = plain_resp(
        StatusCode::METHOD_NOT_ALLOWED,
        Bytes::from(body),
        "application/json",
    );
    resp.headers_mut()
        .insert(hyper::header::ALLOW, HeaderValue::from_static(allow));
    resp
}

fn strip_null_object_fields(mut v: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut() {
        obj.retain(|_, val| !val.is_null());
    }
    v
}

fn utc_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    // Howard Hinnant's civil-from-days algorithm
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let ss = rem % 60;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn build_stac_render_entry(
    host: &str,
    forwarded_prefix: &str,
    stac_url: &str,
    render_id: &str,
    render: &serde_json::Value,
) -> serde_json::Value {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        format!("{host}{forwarded_prefix}")
    } else {
        format!("http://{host}{forwarded_prefix}")
    };
    let mut qs_parts = vec![format!("url={stac_url}")];
    if let Some(obj) = render.as_object() {
        for (k, v) in obj {
            match k.as_str() {
                "title" | "minmax_zoom" | "tilematrixsets" => continue,
                "assets" => {
                    if let Some(arr) = v.as_array() {
                        for a in arr {
                            if let Some(s) = a.as_str() {
                                qs_parts.push(format!("assets={s}"));
                            }
                        }
                    }
                }
                "rescale" => {
                    if let Some(arr) = v.as_array() {
                        for pair in arr {
                            if let Some(p) = pair.as_array()
                                && p.len() == 2
                            {
                                qs_parts.push(format!("rescale={},{}", p[0], p[1]));
                            }
                        }
                    }
                }
                "bidx" => {
                    if let Some(arr) = v.as_array() {
                        for b in arr {
                            qs_parts.push(format!("bidx={b}"));
                        }
                    }
                }
                _ => match v {
                    serde_json::Value::String(s) => qs_parts.push(format!("{k}={s}")),
                    serde_json::Value::Number(n) => qs_parts.push(format!("{k}={n}")),
                    serde_json::Value::Bool(b) => qs_parts.push(format!("{k}={b}")),
                    _ => {}
                },
            }
        }
    }
    let render_qs = qs_parts.join("&");
    serde_json::json!({
        "valid": true,
        "params": render,
        "links": [
            {
                "rel": "self",
                "type": "application/json",
                "href": format!("{origin}/stac/renders/{render_id}?url={stac_url}"),
            },
            {
                "rel": "tilesets-map",
                "type": "application/json",
                "title": "TileJSON",
                "href": format!("{origin}/stac/WebMercatorQuad/tilejson.json?{render_qs}"),
            },
            {
                "rel": "tilesets-map",
                "type": "application/xml",
                "title": "WMTS",
                "href": format!("{origin}/stac/WebMercatorQuad/WMTSCapabilities.xml?{render_qs}"),
            },
        ],
    })
}

fn data_band_count(ifd: &cogzilla_tiff::IfdMeta) -> u16 {
    if cogzilla_codec::alpha_channel_index(ifd.samples_per_pixel as usize, &ifd.extra_samples)
        .is_some()
    {
        ifd.samples_per_pixel.saturating_sub(1).max(1)
    } else {
        ifd.samples_per_pixel
    }
}

fn effective_bbox_max_size(pq: &ParsedQuery) -> u32 {
    if pq.keys.iter().any(|k| k == "max_size") {
        pq.max_size
    } else {
        // TiTiler defaults max_size=None on /bbox and /feature (full native resolution, capped at 4096)
        4096
    }
}

fn maybe_upgrade_multiband_default(
    bands: BandRequest,
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    default_bands: u16,
) -> BandRequest {
    if pq.bidx.is_empty()
        && pq.expr_raw.is_none()
        && pq.algorithm.is_none()
        && (default_bands == 2 || (4..=15).contains(&default_bands))
        && (matches!(fmt, OutputFormat::Npy | OutputFormat::RawF32)
            || (ct == GEOTIFF_CONTENT_TYPE
                && pq.rescale.is_empty()
                && !pq.colormap_explicit
                && pq.custom_colormap_json.is_none()
                && pq.color_formula.is_none()))
    {
        return BandRequest::MultiBand {
            bands_1based: (1..=default_bands).collect(),
        };
    }
    bands
}

fn build_band_request_for_output(
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_tiff::SampleType,
    fmt: OutputFormat,
    ct: &str,
) -> Result<BandRequest, EngineError> {
    let bands = build_band_request(pq, default_bands, sample_type)?;
    Ok(maybe_upgrade_multiband_default(
        bands,
        fmt,
        ct,
        pq,
        default_bands,
    ))
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
    let tms_id = if is_crs84 {
        "WorldCRS84Quad"
    } else {
        "WebMercatorQuad"
    };
    build_tile_request_for_tms(
        engine,
        tile,
        scale_opt,
        ext_opt,
        pq,
        default_bands,
        sample_type,
        tms_id,
        source_cog,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_tile_request_for_tms(
    engine: Option<&TileEngine>,
    tile: TileCoord,
    scale_opt: Option<u32>,
    ext_opt: Option<&str>,
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_tiff::SampleType,
    tms_id: &str,
    source_cog: Option<&cogzilla_io::OpenedCog>,
) -> Result<(TileRequest, &'static str), EngineError> {
    let (cols, rows, target_epsg, base_tile_size) =
        if let Some(tms_def) = crate::tms::lookup_tms(tms_id) {
            let Some(m) = tms_def.matrix(tile.z) else {
                return Err(EngineError::BadRequest(format!(
                    "zoom level z={} is not defined for tile matrix set '{}'",
                    tile.z, tms_id
                )));
            };
            (m.matrix_width, m.matrix_height, tms_def.epsg, m.tile_width)
        } else {
            if tile.z > 30 {
                return Err(EngineError::BadRequest(format!(
                    "zoom level z={} exceeds maximum supported zoom (30)",
                    tile.z
                )));
            }
            let rows = 1u64 << (tile.z as u32);
            (rows, rows, 3857u16, 256u32)
        };
    if (tile.x as u64) >= cols || (tile.y as u64) >= rows {
        return Err(EngineError::BadRequest(format!(
            "tile coordinate ({}, {}) out of range for {tms_id} z={} (expected x < {cols}, y < {rows})",
            tile.x, tile.y, tile.z
        )));
    }
    let eff_ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
    let (format, content_type) = parse_ext_format(eff_ext, pq.webp_quality, pq.jpeg_quality)?;
    let eff_default_bands = match source_cog {
        Some(c) if default_bands == c.index.full_res().samples_per_pixel => {
            data_band_count(c.index.full_res())
        }
        _ => default_bands,
    };
    let bands = build_band_request_for_output(pq, eff_default_bands, sample_type, format, content_type)?;
    let format = geotiff_render_format(format, content_type, pq, source_cog, &bands, sample_type);
    if matches!(bands, BandRequest::MultiBand { .. })
        && !matches!(format, OutputFormat::RawF32 | OutputFormat::Npy)
    {
        return Err(EngineError::BadRequest(
            "multi-band bidx (2 or 4..=15 bands) is only supported for .tif, .npy, or .f32 output"
                .into(),
        ));
    }
    let has_explicit_tile_size = pq
        .keys
        .iter()
        .any(|k| matches!(k.as_str(), "tile_size" | "tilesize" | "size" | "scale"));
    let tile_size = match scale_opt {
        Some(sc) if (1..=4).contains(&sc) => {
            let sz = sc.saturating_mul(base_tile_size);
            if sz > 4096 {
                return Err(EngineError::BadRequest(format!(
                    "scaled tile size {sz} exceeds maximum (4096) for {tms_id}"
                )));
            }
            sz
        }
        Some(other) => {
            return Err(EngineError::BadRequest(format!(
                "invalid tile scale @{other}x (expected 1..=4)"
            )));
        }
        None => {
            if has_explicit_tile_size {
                pq.tile_size
            } else {
                base_tile_size
            }
        }
    };
    let buffered_dim = (tile_size as f32 + 2.0 * pq.buffer).round() as u32;
    if buffered_dim > 4096 {
        return Err(EngineError::BadRequest(format!(
            "buffered tile size {buffered_dim} (tile_size={tile_size}, buffer={}) exceeds maximum (4096) for {tms_id}",
            pq.buffer
        )));
    }
    Ok((
        TileRequest {
            tile,
            tile_size,
            bands,
            resample: resolve_resample(engine, pq, source_cog, target_epsg),
            format,
            unscale: pq.unscale,
        },
        content_type,
    ))
}

fn build_stats_options(pq: &ParsedQuery, allow_expr: bool) -> Result<StatsOptions, EngineError> {
    build_stats_options_with_bands(pq, allow_expr, 0)
}

fn build_stats_options_with_bands(
    pq: &ParsedQuery,
    allow_expr: bool,
    default_bands: u16,
) -> Result<StatsOptions, EngineError> {
    let mut opts = StatsOptions::default();
    if !pq.percentiles.is_empty() {
        opts.percentiles = pq.percentiles.clone();
    }
    if let Some(bins) = pq.histogram_bins {
        opts.histogram_bins = bins;
    }
    opts.histogram_edges = pq.histogram_edges.clone();
    opts.histogram_range = pq.histogram_range;
    opts.categorical = pq.categorical;
    if !pq.categories.is_empty() {
        opts.categories = Some(pq.categories.clone());
    }
    if !pq.bidx.is_empty() {
        opts.bidx = Some(pq.bidx.clone());
    }
    opts.width = pq.width;
    opts.height = pq.height;
    opts.max_size_explicit = pq.keys.iter().any(|k| k == "max_size");
    if allow_expr
        && let Some(raw) = pq.expr_raw.as_deref()
    {
        let mut compiled = Vec::new();
        for part in raw.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let prog = ExprProgram::compile(part).map_err(EngineError::BadRequest)?;
            compiled.push((part.to_string(), prog));
            if compiled.len() > 16 {
                return Err(EngineError::BadRequest(
                    "expression exceeds max 16 semicolon-separated outputs".into(),
                ));
            }
        }
        if !compiled.is_empty() {
            opts.expressions = Some(compiled);
        }
    } else if allow_expr
        && let Some(alg_raw) = pq.algorithm.as_deref()
    {
        let bands: Vec<u16> = if !pq.bidx.is_empty() {
            pq.bidx.clone()
        } else {
            (1..=default_bands.clamp(1, 16)).collect()
        };
        let alg = alg_raw.to_ascii_lowercase();
        let expr_str = match alg.as_str() {
            "normalizedindex" => {
                if bands.len() < 2 {
                    return Err(EngineError::BadRequest(
                        "algorithm 'normalizedIndex' requires at least 2 bands".into(),
                    ));
                }
                format!("(b{} - b{}) / (b{} + b{})", bands[1], bands[0], bands[1], bands[0])
            }
            "min" | "max" => {
                let mut acc = format!("b{}", bands[0]);
                for &b in &bands[1..] {
                    acc = format!("{alg}({acc}, b{b})");
                }
                acc
            }
            "sum" => bands.iter().map(|b| format!("b{b}")).collect::<Vec<_>>().join(" + "),
            "mean" => {
                let s = bands.iter().map(|b| format!("b{b}")).collect::<Vec<_>>().join(" + ");
                format!("({s}) / {}", bands.len())
            }
            "var" | "std" | "median" => {
                let args = bands.iter().map(|b| format!("b{b}")).collect::<Vec<_>>().join(", ");
                format!("{alg}({args})")
            }
            "grayscale" => {
                if bands.len() >= 3 {
                    format!("0.299 * b{} + 0.587 * b{} + 0.114 * b{}", bands[0], bands[1], bands[2])
                } else {
                    format!("b{}", bands[0])
                }
            }
            "bitonal" => {
                let src = if bands.len() >= 3 {
                    format!("(0.299 * b{} + 0.587 * b{} + 0.114 * b{})", bands[0], bands[1], bands[2])
                } else {
                    format!("b{}", bands[0])
                };
                format!("where({src} > {}, 255, 0)", pq.threshold)
            }
            "cast" => format!("clip(floor(b{}), 0, 255)", bands[0]),
            "ceil" | "floor" => format!("{alg}(b{})", bands[0]),
            _ => {
                return Err(EngineError::BadRequest(format!(
                    "algorithm '{alg_raw}' is not supported on statistics routes"
                )));
            }
        };
        let prog = ExprProgram::compile(&expr_str).map_err(EngineError::BadRequest)?;
        opts.expressions = Some(vec![(alg_raw.to_string(), prog)]);
    }
    Ok(opts)
}

/// Apply optional `?expression=` (supporting `;`-separated multi-band outputs) or `?bidx=`
/// filtering to a TiTiler `/point` JSON payload.
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

    if let Some(raw_expr) = expr_raw {
        let parts: Vec<&str> = raw_expr
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(EngineError::BadRequest("empty expression".into()));
        }
        if parts.len() > 16 {
            return Err(EngineError::BadRequest(
                "expression exceeds max 16 semicolon-separated outputs".into(),
            ));
        }
        let n_bands = vals.len();
        let mut out_vals = Vec::with_capacity(parts.len());
        let mut out_names = Vec::with_capacity(parts.len());
        for expr in parts {
            let prog = ExprProgram::compile(expr).map_err(EngineError::BadRequest)?;
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
            out_vals.push(if res.is_finite() {
                serde_json::json!(res)
            } else {
                serde_json::Value::Null
            });
            out_names.push(serde_json::json!(expr));
        }
        let band_names: Vec<serde_json::Value> = (1..=out_vals.len())
            .map(|i| serde_json::json!(format!("b{i}")))
            .collect();
        pt["values"] = serde_json::Value::Array(out_vals);
        pt["band_names"] = serde_json::Value::Array(band_names);
        pt["band_descriptions"] = serde_json::Value::Array(out_names);
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

fn default_raw_band_count(bands: &BandRequest, pixel_selection: PixelSelection) -> usize {
    if pixel_selection == PixelSelection::Count {
        return 1;
    }
    match bands {
        BandRequest::Rgb { .. }
        | BandRequest::SarRgb { .. }
        | BandRequest::Terrain { .. }
        | BandRequest::CustomExprRgb { .. } => 3,
        BandRequest::MultiBand { bands_1based } => bands_1based.len().max(1),
        _ => 1,
    }
}

/// Hard memory budget (64 MiB) for `pixel_selection=median`, which must buffer
/// per-scene planes in order to compute exact per-pixel medians.
const MAX_MEDIAN_BUFFER_BYTES: usize = 64 * 1024 * 1024;

async fn render_mosaic_tile_pixel_selection(
    engine: &TileEngine,
    urls: &[String],
    treq: TileRequest,
    pq: &ParsedQuery,
) -> Result<Bytes, EngineError> {
    let urls = &urls[..urls.len().min(64)];
    let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
    let bbox_3857 = web_mercator_buffered_3857_bbox(
        treq.tile.z,
        treq.tile.x,
        treq.tile.y,
        pq.buffer,
        treq.tile_size,
    );
    let n_px = (out_dim as usize) * (out_dim as usize);
    let mut first_lut: Option<cogzilla_kernel::CustomColormap> = None;

    // O(1)-in-scenes streaming accumulator for all non-Median modes; bounded
    // Vec<Vec<f32>> only for Median.
    enum PixelAcc {
        Highest { nb: usize, best: Vec<f32> },
        Lowest { nb: usize, best: Vec<f32> },
        Mean { nb: usize, sum: Vec<f64>, cnt: Vec<u32> },
        Stdev { nb: usize, mean: Vec<f64>, m2: Vec<f64>, cnt: Vec<u32> },
        Count { nb: usize, cnt: Vec<u32> },
        LastBand { nb: usize, want_high: bool, best_key: Vec<f32>, best_vals: Vec<f32> },
        Median { nb: usize, scenes: Vec<Vec<f32>> },
    }

    let mut acc: Option<PixelAcc> = None;

    for (scene_i, url) in urls.iter().enumerate() {
        if scene_i > 0 {
            tokio::task::consume_budget().await;
        }
        let Ok(cog) = engine.store.open_uri(url).await else {
            continue;
        };
        let cogs = vec![cog];
        let mut scene_bands = treq.bands.clone();
        if pq.unscale {
            apply_unscale_cogs(&mut scene_bands, &cogs);
        }
        if first_lut.is_none() {
            first_lut = resolve_custom_lut(pq, &cogs[0], &scene_bands)?;
        }
        let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), 3857);
        let raw = render_mercator_bbox(
            engine,
            &cogs,
            bbox_3857,
            out_dim,
            out_dim,
            &scene_bands,
            resample,
            OutputFormat::RawF32,
            pq.nodata,
            None,
            None,
            pq.return_mask,
            pq.unscale,
        )
        .await?;
        if n_px == 0 || raw.len() % (n_px * 4) != 0 {
            continue;
        }
        let nb = raw.len() / (n_px * 4);
        if nb == 0 {
            continue;
        }
        let total = nb * n_px;
        let read_f32 = |idx: usize| -> f32 {
            let off = idx * 4;
            f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
        };

        let state = acc.get_or_insert_with(|| match pq.pixel_selection {
            PixelSelection::Highest => PixelAcc::Highest { nb, best: vec![f32::NAN; total] },
            PixelSelection::Lowest => PixelAcc::Lowest { nb, best: vec![f32::NAN; total] },
            PixelSelection::Mean => PixelAcc::Mean {
                nb,
                sum: vec![0.0f64; total],
                cnt: vec![0u32; total],
            },
            PixelSelection::Stdev => PixelAcc::Stdev {
                nb,
                mean: vec![0.0f64; total],
                m2: vec![0.0f64; total],
                cnt: vec![0u32; total],
            },
            PixelSelection::Count => PixelAcc::Count { nb, cnt: vec![0u32; n_px] },
            PixelSelection::LastBandLow | PixelSelection::LastBandHigh => PixelAcc::LastBand {
                nb,
                want_high: pq.pixel_selection == PixelSelection::LastBandHigh,
                best_key: vec![f32::NAN; n_px],
                best_vals: vec![f32::NAN; total],
            },
            PixelSelection::Median => PixelAcc::Median { nb, scenes: Vec::new() },
            PixelSelection::First => unreachable!(),
        });

        match state {
            PixelAcc::Highest { nb: acc_nb, best } if *acc_nb == nb => {
                for (idx, dst) in best.iter_mut().enumerate() {
                    let v = read_f32(idx);
                    if v.is_finite() && (dst.is_nan() || v > *dst) {
                        *dst = v;
                    }
                }
            }
            PixelAcc::Lowest { nb: acc_nb, best } if *acc_nb == nb => {
                for (idx, dst) in best.iter_mut().enumerate() {
                    let v = read_f32(idx);
                    if v.is_finite() && (dst.is_nan() || v < *dst) {
                        *dst = v;
                    }
                }
            }
            PixelAcc::Mean { nb: acc_nb, sum, cnt } if *acc_nb == nb => {
                for idx in 0..total {
                    let v = read_f32(idx);
                    if v.is_finite() {
                        sum[idx] += v as f64;
                        cnt[idx] += 1;
                    }
                }
            }
            PixelAcc::Stdev { nb: acc_nb, mean, m2, cnt } if *acc_nb == nb => {
                for idx in 0..total {
                    let v = read_f32(idx);
                    if v.is_finite() {
                        let c = cnt[idx] + 1;
                        cnt[idx] = c;
                        let vf = v as f64;
                        let delta = vf - mean[idx];
                        let new_mean = mean[idx] + delta / (c as f64);
                        mean[idx] = new_mean;
                        let delta2 = vf - new_mean;
                        m2[idx] += delta * delta2;
                    }
                }
            }
            PixelAcc::Count { nb: acc_nb, cnt } if *acc_nb == nb => {
                for (i, c) in cnt.iter_mut().enumerate() {
                    if read_f32(i).is_finite() {
                        *c += 1;
                    }
                }
            }
            PixelAcc::LastBand {
                nb: acc_nb,
                want_high,
                best_key,
                best_vals,
            } if *acc_nb == nb => {
                let last_off = (nb - 1) * n_px;
                for i in 0..n_px {
                    let kv = read_f32(last_off + i);
                    let cur = best_key[i];
                    if kv.is_finite()
                        && (cur.is_nan()
                            || (*want_high && kv > cur)
                            || (!*want_high && kv < cur))
                    {
                        best_key[i] = kv;
                        for b in 0..nb {
                            best_vals[b * n_px + i] = read_f32(b * n_px + i);
                        }
                    }
                }
            }
            PixelAcc::Median { nb: acc_nb, scenes } if *acc_nb == nb => {
                let next_bytes = (scenes.len() + 1)
                    .saturating_mul(total)
                    .saturating_mul(4);
                if next_bytes > MAX_MEDIAN_BUFFER_BYTES {
                    return Err(EngineError::BadRequest(format!(
                        "pixel_selection=median exceeds memory budget ({} MiB limit); reduce tile_size, buffer, or band count",
                        MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
                    )));
                }
                let vals: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                scenes.push(vals);
            }
            _ => {}
        }
    }

    let (out_nb, combined): (usize, Vec<f32>) = match acc {
        None => {
            let out_nb = default_raw_band_count(&treq.bands, pq.pixel_selection);
            (out_nb, vec![f32::NAN; out_nb * n_px])
        }
        Some(PixelAcc::Highest { nb, best }) | Some(PixelAcc::Lowest { nb, best }) => (nb, best),
        Some(PixelAcc::Mean { nb, sum, cnt }) => {
            let out = sum
                .into_iter()
                .zip(cnt)
                .map(|(s, c)| if c > 0 { (s / c as f64) as f32 } else { f32::NAN })
                .collect();
            (nb, out)
        }
        Some(PixelAcc::Stdev { nb, m2, cnt, .. }) => {
            let out = m2
                .into_iter()
                .zip(cnt)
                .map(|(s2, c)| {
                    if c > 0 {
                        (s2 / c as f64).max(0.0).sqrt() as f32
                    } else {
                        f32::NAN
                    }
                })
                .collect();
            (nb, out)
        }
        Some(PixelAcc::Count { cnt, .. }) => {
            let out = cnt
                .into_iter()
                .map(|c| if c > 0 { c as f32 } else { f32::NAN })
                .collect();
            (1, out)
        }
        Some(PixelAcc::LastBand { nb, best_vals, .. }) => (nb, best_vals),
        Some(PixelAcc::Median { nb, scenes }) => {
            let mut out = vec![f32::NAN; nb * n_px];
            let mut scratch = Vec::with_capacity(scenes.len());
            for b in 0..nb {
                let b_off = b * n_px;
                for i in 0..n_px {
                    scratch.clear();
                    for s in &scenes {
                        let v = s[b_off + i];
                        if v.is_finite() {
                            scratch.push(v);
                        }
                    }
                    let k = scratch.len();
                    if k > 0 {
                        scratch.sort_unstable_by(|a, b| a.total_cmp(b));
                        out[b_off + i] = if k % 2 == 1 {
                            scratch[k / 2]
                        } else {
                            0.5 * (scratch[k / 2 - 1] + scratch[k / 2])
                        };
                    }
                }
            }
            (nb, out)
        }
    };

    if matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy) {
        let mut out = bytes::BytesMut::with_capacity(combined.len() * 4);
        for v in combined {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let raw = out.freeze();
        if matches!(treq.format, OutputFormat::Npy) {
            return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                raw,
                out_nb,
                out_dim,
                out_dim,
                pq.return_mask,
            ));
        }
        return Ok(raw);
    }

    let cf = pq
        .color_formula
        .as_deref()
        .map(ColorFormula::parse)
        .transpose()
        .map_err(EngineError::BadRequest)?;
    let mut rgba = vec![0u8; n_px * 4];
    if out_nb >= 3 {
        let (rmin, rmax) = match &treq.bands {
            BandRequest::Rgb {
                rescale_min,
                rescale_max,
                ..
            }
            | BandRequest::SarRgb {
                rescale_min,
                rescale_max,
                ..
            }
            | BandRequest::CustomExprRgb {
                rescale_min,
                rescale_max,
                ..
            } => (*rescale_min, *rescale_max),
            _ => ([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]),
        };
        let inv = [
            255.0 / cogzilla_kernel::safe_rescale_denom(rmin[0], rmax[0]),
            255.0 / cogzilla_kernel::safe_rescale_denom(rmin[1], rmax[1]),
            255.0 / cogzilla_kernel::safe_rescale_denom(rmin[2], rmax[2]),
        ];
        for i in 0..n_px {
            let r = combined[i];
            let g = combined[n_px + i];
            let b = combined[2 * n_px + i];
            if r.is_finite() && g.is_finite() && b.is_finite() {
                rgba[i * 4] = ((r - rmin[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                rgba[i * 4 + 1] = ((g - rmin[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                rgba[i * 4 + 2] = ((b - rmin[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                rgba[i * 4 + 3] = 255;
            }
        }
        if let Some(ref cf) = cf {
            cf.apply_buffer_in_place(&mut rgba, 4);
        }
    } else {
        let (rescale, cmap) = match &treq.bands {
            BandRequest::Single {
                rescale, colormap, ..
            }
            | BandRequest::Dem {
                rescale, colormap, ..
            }
            | BandRequest::CustomExpr {
                rescale, colormap, ..
            }
            | BandRequest::Spectral1 {
                rescale, colormap, ..
            }
            | BandRequest::Spectral2 {
                rescale, colormap, ..
            }
            | BandRequest::Spectral3 {
                rescale, colormap, ..
            } => (*rescale, *colormap),
            _ => ((0.0, 255.0), pq.colormap),
        };
        let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
        let cf_lut = cf.as_ref().map(|c| {
            let mut t = [0u8; 256];
            for i in 0..256 {
                let mut px = [i as u8, i as u8, i as u8, 255];
                c.apply_buffer_in_place(&mut px, 4);
                t[i] = px[0];
            }
            t
        });
        let lut = cmap.lut();
        for i in 0..n_px {
            let v = combined[i];
            if v.is_finite() {
                let px = if let Some(ref custom) = first_lut {
                    custom.lookup(v, rescale.0, inv)
                } else {
                    let mut idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                    if let Some(ref cl) = cf_lut {
                        idx = cl[idx] as usize;
                    }
                    lut.rgba_u32[idx].to_le_bytes()
                };
                rgba[i * 4..i * 4 + 4].copy_from_slice(&px);
            }
        }
    }
    Ok(encode_rgba_output(
        out_dim as usize,
        out_dim as usize,
        &rgba,
        treq.format,
        pq.return_mask,
    ))
}

async fn render_mosaic_tile_styled(
    engine: &TileEngine,
    urls: &[String],
    mut treq: TileRequest,
    pq: &ParsedQuery,
    first_has_colormap: bool,
) -> Result<Bytes, EngineError> {
    if pq.pixel_selection != PixelSelection::First {
        return render_mosaic_tile_pixel_selection(engine, urls, treq, pq).await;
    }
    let mut needs_bbox_mosaic = pq.nodata.is_some()
        || pq.custom_colormap_json.is_some()
        || pq.buffer > 0.0
        || matches!(treq.format, OutputFormat::Npy)
        || (pq.color_formula.is_some()
            && !matches!(
                treq.bands,
                BandRequest::Rgb { .. }
                    | BandRequest::SarRgb { .. }
                    | BandRequest::CustomExprRgb { .. }
                    | BandRequest::Terrain { .. }
            ))
        || matches!(treq.bands, BandRequest::CustomExprRgb { .. } | BandRequest::MultiBand { .. })
        || (treq.tile_size != 256 && treq.tile_size != 512);
    if !needs_bbox_mosaic
        && !pq.colormap_explicit
        && pq.custom_colormap_json.is_none()
        && first_has_colormap
    {
        needs_bbox_mosaic = true;
    }

    if needs_bbox_mosaic {
        let urls = &urls[..urls.len().min(64)];
        let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
        let bbox_3857 = web_mercator_buffered_3857_bbox(
            treq.tile.z,
            treq.tile.x,
            treq.tile.y,
            pq.buffer,
            treq.tile_size,
        );
        let n_px = (out_dim as usize) * (out_dim as usize);
        if matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut acc_f32: Option<(usize, Vec<f32>)> = None;
            let mut filled = 0usize;
            for (scene_i, url) in urls.iter().enumerate() {
                if scene_i > 0 {
                    tokio::task::consume_budget().await;
                }
                let Ok(cog) = engine.store.open_uri(url).await else {
                    continue;
                };
                let cogs = vec![cog];
                let mut scene_bands = treq.bands.clone();
                if pq.unscale {
                    apply_unscale_cogs(&mut scene_bands, &cogs);
                }
                let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), 3857);
                let raw = render_mercator_bbox(
                    engine,
                    &cogs,
                    bbox_3857,
                    out_dim,
                    out_dim,
                    &scene_bands,
                    resample,
                    OutputFormat::RawF32,
                    pq.nodata,
                    None,
                    None,
                    pq.return_mask,
                    pq.unscale,
                )
                .await?;
                if n_px == 0 || raw.len() % (n_px * 4) != 0 {
                    continue;
                }
                let nb = raw.len() / (n_px * 4);
                if nb == 0 {
                    continue;
                }
                let (acc_nb, acc) = acc_f32.get_or_insert_with(|| (nb, vec![f32::NAN; nb * n_px]));
                if *acc_nb != nb {
                    continue;
                }
                for i in 0..n_px {
                    if acc[i].is_nan() {
                        let mut all_valid = true;
                        for b in 0..nb {
                            let boff = (b * n_px + i) * 4;
                            let v = f32::from_le_bytes([
                                raw[boff],
                                raw[boff + 1],
                                raw[boff + 2],
                                raw[boff + 3],
                            ]);
                            if v.is_nan() {
                                all_valid = false;
                                break;
                            }
                        }
                        if all_valid {
                            for b in 0..nb {
                                let boff = (b * n_px + i) * 4;
                                acc[b * n_px + i] = f32::from_le_bytes([
                                    raw[boff],
                                    raw[boff + 1],
                                    raw[boff + 2],
                                    raw[boff + 3],
                                ]);
                            }
                            filled += 1;
                        }
                    }
                }
                if filled >= n_px {
                    break;
                }
            }
            let default_nb = default_raw_band_count(&treq.bands, pq.pixel_selection);
            let (nb, acc) = acc_f32.unwrap_or_else(|| (default_nb, vec![f32::NAN; default_nb * n_px]));
            let mut out = bytes::BytesMut::with_capacity(acc.len() * 4);
            for v in acc {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let raw = out.freeze();
            if matches!(treq.format, OutputFormat::Npy) {
                return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                    raw,
                    nb,
                    out_dim,
                    out_dim,
                    pq.return_mask,
                ));
            }
            return Ok(raw);
        }
        let cf = pq
            .color_formula
            .as_deref()
            .map(ColorFormula::parse)
            .transpose()
            .map_err(EngineError::BadRequest)?;
        let mut comp = vec![0u8; n_px * 4];
        let mut filled = 0usize;

        for (scene_i, url) in urls.iter().enumerate() {
            if scene_i > 0 {
                tokio::task::consume_budget().await;
            }
            let Ok(cog) = engine.store.open_uri(url).await else {
                continue;
            };
            let cogs = vec![cog];
            let mut scene_bands = treq.bands.clone();
            if pq.unscale {
                apply_unscale_cogs(&mut scene_bands, &cogs);
            }
            let scene_lut = resolve_custom_lut(pq, &cogs[0], &scene_bands)?;
            let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), 3857);
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
                scene_lut.as_ref(),
                cf.as_ref(),
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

fn stac_asset_expr<'a>(pq: &'a ParsedQuery, asset_name: &str) -> Option<&'a str> {
    pq.asset_exprs
        .iter()
        .find(|(n, _)| n == asset_name)
        .map(|(_, e)| e.as_str())
}

fn expand_stac_asset_bidx(
    pq: &mut ParsedQuery,
    names: &[String],
    cogs: Vec<Arc<cogzilla_io::OpenedCog>>,
) -> Result<(Vec<Arc<cogzilla_io::OpenedCog>>, Option<Vec<u16>>), EngineError> {
    if pq.expr_raw.is_none() && !pq.asset_exprs.is_empty() {
        if names.len() == 1 {
            if let Some(expr) = stac_asset_expr(pq, &names[0]) {
                pq.expr_raw = Some(expr.to_string());
            }
        } else if names.len() == 3
            && names.iter().any(|n| stac_asset_expr(pq, n).is_some())
        {
            let parts: Vec<String> = names
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    let slot = format!("b{}", i + 1);
                    match stac_asset_expr(pq, n) {
                        Some(expr) => expr.replace("b1", &slot),
                        None => slot,
                    }
                })
                .collect();
            pq.expr_raw = Some(parts.join(";"));
        }
    }
    if pq.bidx.is_empty() && names.len() == 1 {
        if let Some((_, bands)) = pq.asset_bidx.iter().find(|(n, _)| n == &names[0]) {
            pq.bidx = bands.clone();
        }
        return Ok((cogs, None));
    }
    if names.len() > 1 {
        let needs_per_asset = names.iter().any(|name| {
            pq.asset_bidx
                .iter()
                .find(|(n, _)| n == name)
                .is_some_and(|(_, bands)| bands.as_slice() != [1])
        });
        if needs_per_asset {
            let mut expanded_cogs = Vec::new();
            let mut slot_bands = Vec::new();
            for (name, cog) in names.iter().zip(cogs.into_iter()) {
                let nb = cog.index.full_res().samples_per_pixel;
                let sel: &[u16] = stac_asset_selected_bands(pq, name).unwrap_or(&[1]);
                for &b in sel {
                    if b == 0 || b > nb {
                        return Err(EngineError::BadRequest(format!(
                            "band {b} out of range for asset '{name}' (dataset has {nb} bands)"
                        )));
                    }
                    expanded_cogs.push(cog.clone());
                    slot_bands.push(b);
                }
            }
            if expanded_cogs.len() > 16 {
                return Err(EngineError::BadRequest(format!(
                    "expanded asset_bidx band count {} exceeds maximum of 16",
                    expanded_cogs.len()
                )));
            }
            return Ok((expanded_cogs, Some(slot_bands)));
        }
    }
    Ok((cogs, None))
}

#[allow(clippy::too_many_arguments)]
async fn render_stac_bbox_opt(
    engine: &TileEngine,
    cogs: &[Arc<cogzilla_io::OpenedCog>],
    slot_bands: Option<&[u16]>,
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    out_w: u32,
    out_h: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    fmt: OutputFormat,
    nodata: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    cf: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    if let Some(ab) = slot_bands {
        render_multi_asset_bbox(
            engine,
            cogs,
            ab,
            bbox,
            dst_crs,
            native_preview,
            out_w,
            out_h,
            poly_mask,
            bands,
            resample,
            fmt,
            nodata,
            custom_lut,
            cf,
            return_mask,
            unscale,
        )
        .await
    } else if native_preview {
        render_native_preview(
            engine,
            cogs,
            bbox,
            out_w,
            out_h,
            bands,
            resample,
            fmt,
            nodata,
            custom_lut,
            cf,
            return_mask,
            unscale,
        )
        .await
    } else {
        render_crs_bbox(
            engine,
            cogs,
            bbox,
            dst_crs,
            out_w,
            out_h,
            poly_mask,
            bands,
            resample,
            fmt,
            nodata,
            custom_lut,
            cf,
            return_mask,
            unscale,
        )
        .await
    }
}

/// Resolve the effective output `(CrsModel, epsg)` for `/{cog,stac,mosaicjson}/bbox` routes:
/// explicit `?dst_crs=` wins, otherwise default to `?coord_crs=` (matching TiTiler 2.4 / `rio-tiler`),
/// falling back to `EPSG:4326`.
fn resolve_bbox_dst_crs(pq: &ParsedQuery) -> Option<(cogzilla_proj::CrsModel, u16)> {
    pq.dst_crs
        .clone()
        .or_else(|| pq.coord_crs.clone().zip(pq.coord_crs_epsg))
}

fn stac_asset_selected_bands<'a>(pq: &'a ParsedQuery, asset_name: &str) -> Option<&'a [u16]> {
    pq.asset_bidx
        .iter()
        .find(|(n, _)| n == asset_name)
        .map(|(_, bands)| bands.as_slice())
}

/// Composite up to 16 intersecting MosaicJSON scenes for `/mosaicjson/bbox` and `/mosaicjson/feature`
/// using first-non-transparent alpha-over (or first-non-NaN for f32/npy) blending in target CRS.
#[allow(clippy::too_many_arguments)]
async fn render_mosaic_bbox_composite(
    engine: &TileEngine,
    urls: &[String],
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    out_w: u32,
    out_h: u32,
    poly_mask: Option<&PolygonMask>,
    pq: &ParsedQuery,
    fmt: OutputFormat,
    ct: &'static str,
    epsg: u16,
) -> Result<Bytes, EngineError> {
    if urls.is_empty() {
        return Err(EngineError::NotFound("no mosaic scenes intersect requested bounds".into()));
    }
    let out_w = out_w.clamp(1, 4096);
    let out_h = out_h.clamp(1, 4096);
    let urls = &urls[..urls.len().min(16)];
    let mut eff_fmt: Option<OutputFormat> = None;
    let mut acc_rgba: Option<Vec<u8>> = None;
    let mut acc_f32: Option<(usize, Vec<f32>)> = None;
    let total_px = (out_w as usize).saturating_mul(out_h as usize);
    let mut opaque_count = 0usize;
    let mut any_ok = false;
    let mut last_err: Option<EngineError> = None;

    for u in urls {
        let cog = match engine.store.open_uri(u).await {
            Ok(c) => c,
            Err(e) => {
                last_err = Some(e.into());
                continue;
            }
        };
        let cogs = [cog];
        let full = cogs[0].index.full_res();
        let mut bands =
            build_band_request_for_output(pq, full.samples_per_pixel, full.sample_type, fmt, ct)?;
        let scene_fmt = *eff_fmt.get_or_insert_with(|| {
            geotiff_render_format(fmt, ct, pq, Some(&cogs[0]), &bands, full.sample_type)
        });
        let is_f32_out = matches!(scene_fmt, OutputFormat::RawF32 | OutputFormat::Npy);
        if pq.unscale {
            apply_unscale_cogs(&mut bands, &cogs);
        }
        let custom_lut = resolve_custom_lut(pq, &cogs[0], &bands)?;
        let cf = pq
            .color_formula
            .as_deref()
            .map(ColorFormula::parse)
            .transpose()
            .map_err(EngineError::BadRequest)?;
        let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), epsg);
        if urls.len() == 1 {
            let d = render_crs_bbox(
                engine,
                &cogs,
                bbox,
                dst_crs,
                out_w,
                out_h,
                poly_mask,
                &bands,
                resample,
                scene_fmt,
                pq.nodata,
                custom_lut.as_ref(),
                cf.as_ref(),
                pq.return_mask,
                pq.unscale,
            )
            .await?;
            return Ok(wrap_geotiff_fmt(
                d,
                ct,
                scene_fmt,
                out_w,
                out_h,
                bbox,
                epsg,
                pq.return_mask,
            ));
        }
        if is_f32_out {
            let raw = render_crs_bbox(
                engine,
                &cogs,
                bbox,
                dst_crs,
                out_w,
                out_h,
                poly_mask,
                &bands,
                resample,
                OutputFormat::RawF32,
                pq.nodata,
                custom_lut.as_ref(),
                cf.as_ref(),
                pq.return_mask,
                pq.unscale,
            )
            .await?;
            any_ok = true;
            if total_px == 0 || raw.len() % (total_px * 4) != 0 {
                continue;
            }
            let nb = raw.len() / (total_px * 4);
            if nb == 0 {
                continue;
            }
            let (acc_nb, acc) = acc_f32.get_or_insert_with(|| (nb, vec![f32::NAN; nb * total_px]));
            if *acc_nb != nb {
                continue;
            }
            for i in 0..total_px {
                if acc[i].is_nan() {
                    let mut all_valid = true;
                    for b in 0..nb {
                        let boff = (b * total_px + i) * 4;
                        let v = f32::from_le_bytes([
                            raw[boff],
                            raw[boff + 1],
                            raw[boff + 2],
                            raw[boff + 3],
                        ]);
                        if v.is_nan() {
                            all_valid = false;
                            break;
                        }
                    }
                    if all_valid {
                        for b in 0..nb {
                            let boff = (b * total_px + i) * 4;
                            acc[b * total_px + i] = f32::from_le_bytes([
                                raw[boff],
                                raw[boff + 1],
                                raw[boff + 2],
                                raw[boff + 3],
                            ]);
                        }
                        opaque_count += 1;
                    }
                }
            }
            if opaque_count >= total_px {
                break;
            }
            continue;
        }
        let raw = render_crs_bbox(
            engine,
            &cogs,
            bbox,
            dst_crs,
            out_w,
            out_h,
            poly_mask,
            &bands,
            resample,
            OutputFormat::RawRgba,
            pq.nodata,
            custom_lut.as_ref(),
            cf.as_ref(),
            true,
            pq.unscale,
        )
        .await?;
        any_ok = true;
        if raw.len() != total_px * 4 {
            continue;
        }
        let acc = acc_rgba.get_or_insert_with(|| vec![0u8; total_px * 4]);
        for i in 0..total_px {
            let off = i * 4;
            if acc[off + 3] == 0 && raw[off + 3] > 0 {
                acc[off..off + 4].copy_from_slice(&raw[off..off + 4]);
                if raw[off + 3] == 255 {
                    opaque_count += 1;
                }
            }
        }
        if opaque_count >= total_px {
            break;
        }
    }
    if !any_ok {
        return Err(last_err.unwrap_or_else(|| {
            EngineError::NotFound("no readable mosaic scenes intersect requested bounds".into())
        }));
    }
    let final_fmt = eff_fmt.unwrap_or(fmt);
    if matches!(final_fmt, OutputFormat::RawF32 | OutputFormat::Npy) {
        let (nb, acc) = acc_f32.unwrap_or_else(|| (1, vec![f32::NAN; total_px]));
        let mut raw_buf = bytes::BytesMut::with_capacity(acc.len() * 4);
        for v in acc {
            raw_buf.extend_from_slice(&v.to_le_bytes());
        }
        let raw = raw_buf.freeze();
        if matches!(final_fmt, OutputFormat::Npy) {
            return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                raw,
                nb,
                out_h,
                out_w,
                pq.return_mask,
            ));
        }
        return Ok(wrap_geotiff_fmt(
            raw,
            ct,
            final_fmt,
            out_w,
            out_h,
            bbox,
            epsg,
            pq.return_mask,
        ));
    }
    let acc = acc_rgba.unwrap_or_else(|| vec![0u8; total_px * 4]);
    let data = encode_rgba_output(out_w as usize, out_h as usize, &acc, final_fmt, pq.return_mask);
    Ok(wrap_geotiff_fmt(
        data,
        ct,
        final_fmt,
        out_w,
        out_h,
        bbox,
        epsg,
        pq.return_mask,
    ))
}

fn web_mercator_buffered_3857_bbox(z: u8, x: u32, y: u32, buffer: f32, tile_size: u32) -> [f64; 4] {
    let (min_x, min_y, max_x, max_y) = tile_bounds_3857(TileCoord { z, x, y });
    let pad_x = (max_x - min_x) * (buffer as f64 / tile_size as f64);
    let pad_y = (max_y - min_y) * (buffer as f64 / tile_size as f64);
    [min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y]
}

fn resolve_custom_lut(
    pq: &ParsedQuery,
    cog: &cogzilla_io::OpenedCog,
    bands: &BandRequest,
) -> Result<Option<CustomColormap>, EngineError> {
    if let Some(ref j) = pq.custom_colormap_json {
        return CustomColormap::from_json_str_with_type(
            j,
            !pq.rescale.is_empty(),
            pq.colormap_type_linear,
        )
        .map(Some)
        .map_err(EngineError::BadRequest);
    }
    if !pq.colormap_explicit
        && pq.rescale.is_empty()
        && matches!(bands, BandRequest::Single { .. })
        && let Some(ref pal) = cog.index.color_map
    {
        return Ok(Some(CustomColormap::from_rgba_palette(pal)));
    }
    Ok(None)
}

fn has_embedded_palette(
    pq: &ParsedQuery,
    cog: &cogzilla_io::OpenedCog,
    bands: &BandRequest,
) -> bool {
    pq.custom_colormap_json.is_none()
        && !pq.colormap_explicit
        && pq.rescale.is_empty()
        && matches!(bands, BandRequest::Single { .. })
        && cog.index.color_map.is_some()
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
    let decoded_buf: Vec<u8>;
    let json_bytes: &[u8] = if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        decoded_buf = cogzilla_codec::decompress_gzip_limited(&bytes, MAX_POST_BODY_BYTES)
            .map_err(|e| EngineError::BadRequest(format!("invalid gzipped MosaicJSON: {e}")))?;
        &decoded_buf
    } else {
        &bytes
    };
    let text = String::from_utf8_lossy(json_bytes);
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

/// Open the first readable scene in a mosaic candidate list so a single missing/corrupt scene at
/// index 0 does not fail the entire mosaic tile when subsequent scenes are valid.
async fn open_first_mosaic_cog(
    engine: &TileEngine,
    urls: &[String],
) -> Result<Arc<cogzilla_io::OpenedCog>, EngineError> {
    let mut last_err: Option<EngineError> = None;
    for u in urls.iter().take(16) {
        match engine.store.open_uri(u).await {
            Ok(c) => return Ok(c),
            Err(e) => last_err = Some(e.into()),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        EngineError::NotFound("mosaic contains no readable scenes".into())
    }))
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
    let t0 = std::time::Instant::now();
    let is_head = *req.method() == Method::HEAD;
    let is_post = *req.method() == Method::POST;
    let mut resp = handle_request_inner(engine, req, is_head).await?;
    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if let Ok(hv) = HeaderValue::from_str(&format!("total;dur={dur_ms:.2}")) {
        resp.headers_mut().insert(
            hyper::header::HeaderName::from_static("server-timing"),
            hv,
        );
    }
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

    if method == Method::GET && (path == "/conformance" || path == "/conformance/") {
        let conf = serde_json::json!({
            "conformsTo": [
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/landing-page",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/oas30",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/json",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/html",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tileset",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tilesets-list",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/dataset-tilesets",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/geodata-tilesets",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/png",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/jpeg",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tiff",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/crs",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/scaling",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/spatial-subsetting",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/png",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/jpeg",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/tiff",
                "http://www.opengis.net/spec/tms/2.0/conf/tilematrixset",
                "http://www.opengis.net/spec/tms/2.0/conf/json-tilematrixset",
                "http://www.opengis.net/spec/tms/2.0/conf/tilematrixsetlimits"
            ]
        });
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from(conf.to_string()),
            "application/json",
        ));
    }

    if method == Method::GET && (path == "/" || path == "/index.html") {
        let wants_json = raw_query.split('&').any(|kv| kv == "f=json")
            || (path == "/"
                && req
                    .headers()
                    .get("accept")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|a| a.contains("application/json") && !a.contains("text/html")));
        if wants_json {
            let landing = serde_json::json!({
                "title": "cogzilla",
                "description": "Zero-GDAL Rust + SIMD drop-in TiTiler COG, STAC & MosaicJSON tile server",
                "links": [
                    { "href": format!("{host}/"), "rel": "self", "type": "application/json", "title": "Landing Page" },
                    { "href": format!("{host}/api"), "rel": "service-desc", "type": "application/vnd.oai.openapi+json;version=3.0", "title": "OpenAPI definition" },
                    { "href": format!("{host}/api.html"), "rel": "service-doc", "type": "text/html", "title": "Swagger UI" },
                    { "href": format!("{host}/conformance"), "rel": "conformance", "type": "application/json", "title": "OGC API conformance classes" },
                    { "href": format!("{host}/tileMatrixSets"), "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-schemes", "type": "application/json", "title": "TileMatrixSets" },
                    { "href": format!("{host}/colorMaps"), "rel": "data", "type": "application/json", "title": "ColorMaps" },
                    { "href": format!("{host}/algorithms"), "rel": "data", "type": "application/json", "title": "Algorithms" }
                ]
            });
            return Ok(plain_resp(
                StatusCode::OK,
                Bytes::from(landing.to_string()),
                "application/json",
            ));
        }
        let base_path_js = serde_json::to_string(&forwarded_prefix)
            .unwrap_or_else(|_| "\"\"".to_string())
            .replace("</", "<\\/");
        let html = include_str!("landing.html")
            .replace("{{VERSION}}", env!("CARGO_PKG_VERSION"))
            .replace("{{BASE_PATH}}", &forwarded_prefix)
            .replace("{{BASE_PATH_JS}}", &base_path_js);
        return Ok(plain_resp(StatusCode::OK, Bytes::from(html), "text/html; charset=utf-8"));
    }

    if method == Method::GET && (path == "/openapi.json" || path == "/api") {
        let mut spec = serde_json::json!({
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
        });
        if let (Some(paths), serde_json::Value::Object(extra)) =
            (spec["paths"].as_object_mut(), pgstac::openapi_paths())
        {
            paths.extend(extra);
        }
        return Ok(plain_resp(StatusCode::OK, Bytes::from(spec.to_string()), "application/json"));
    }

    if method == Method::GET && (path == "/docs" || path == "/api.html") {
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

    let mut pq = parse_query(if raw_query.is_empty() {
        None
    } else {
        Some(&raw_query)
    });

    let segs: Vec<&str> = path.trim_matches('/').split('/').collect();

    // titiler-pgstac's item routes, /collections/{c}/items/{i}/..., are the /stac router over
    // an item read from pgSTAC instead of from `url`.
    let mut pgstac_item: Option<Arc<StacItem>> = None;
    let mut pgstac_item_extent: Option<pgstac::ItemExtent> = None;
    let mut stac_prefix = String::from("stac");
    let item_rest: Option<Vec<&str>> = match segs.as_slice() {
        ["collections", c, "items", i, rest @ ..] if !rest.is_empty() => {
            let ctx = pgstac::Ctx {
                engine: &engine,
                host: &host,
                forwarded_prefix: &forwarded_prefix,
                path: &path,
                raw_query: &raw_query,
                if_none_match: if_none_match.as_deref(),
                only: None,
            };
            match pgstac::item_route(&ctx, &method, c, i, rest, &pq).await {
                pgstac::ItemRoute::Done(resp) => return Ok(pgstac_cacheable(resp, &method)),
                pgstac::ItemRoute::Stac(item, extent) => {
                    pgstac_item = Some(item);
                    pgstac_item_extent = extent;
                    // Band math on an item's point is applied to the `/stac` answer by
                    // `pgstac::item_fixup`, with TiTiler's names: `/stac` must not apply it first.
                    // An expression is over every band read, so the request's `bidx` does
                    // not narrow them first; each asset's own selection still applies.
                    match pgstac::item_asset_options(rest, &raw_query) {
                        Err(resp) => return Ok(resp),
                        Ok(Some((per_asset, own))) => {
                            pq.bidx = own;
                            pq.asset_bidx = per_asset;
                        }
                        Ok(None) => {}
                    }
                    if matches!(rest, ["point", _]) && pq.expr_raw.take().is_some() {
                        pq.bidx.clear();
                    }
                    stac_prefix = format!("collections/{c}/items/{i}");
                    let mut as_stac = vec!["stac"];
                    as_stac.extend_from_slice(rest);
                    Some(as_stac)
                }
            }
        }
        _ => None,
    };
    let segs = item_rest.unwrap_or(segs);

    if let Err(msg) = validate_query(&segs, &pq) {
        if pgstac_item.is_none() {
            let body = serde_json::json!({
                "detail": [
                    {
                        "type": "value_error",
                        "loc": ["query"],
                        "msg": msg,
                    }
                ]
            })
            .to_string();
            return Ok(plain_resp(
                StatusCode::UNPROCESSABLE_ENTITY,
                Bytes::from(body),
                "application/json",
            ));
        }
        // Item routes answer in JSON, and do band math on point values themselves.
        let point_math = segs.get(1) == Some(&"point") && msg.contains("'expression'");
        if !point_math {
            return Ok(pgstac::bad_request(msg));
        }
    }

    // titiler-pgstac's routes: /searches/... and /collections/...
    if matches!(segs.first(), Some(&("searches" | "collections"))) {
        let ctx = pgstac::Ctx {
            engine: &engine,
            host: &host,
            forwarded_prefix: &forwarded_prefix,
            path: &path,
            raw_query: &raw_query,
            if_none_match: if_none_match.as_deref(),
            only: None,
        };
        let list = matches!(segs[..], ["searches"] | ["searches", "" | "list"]);
        let cacheable = method == Method::GET && !list;
        let resp = pgstac::handle(ctx, &method, &segs, pq, req).await;
        return Ok(if cacheable { pgstac_cacheable(resp, &Method::GET) } else { resp });
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
    if method == Method::POST && segs.first() == Some(&"mosaicjson") {
        let mj_url = pq.urls.first().cloned();
        let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
        let res: Result<(Bytes, &'static str), EngineError> = match segs.as_slice() {
            ["mosaicjson", "validate"] => async {
                let body_bytes = read_limited_body(req.into_body()).await?;
                let text = std::str::from_utf8(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid UTF-8 body: {e}")))?;
                let mj = MosaicJsonCatalog::from_json_str(text).map_err(EngineError::BadRequest)?;
                let clean = strip_null_object_fields(mj.raw_json.clone());
                Ok((Bytes::from(clean.to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "statistics"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let stats_opts = build_stats_options(&pq, false)?;
                let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let compute_for_geom = |geom: &serde_json::Value| {
                    let eng = engine.clone();
                    let mj_c = mj.clone();
                    let opts = stats_opts.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let coord_crs_c = pq.coord_crs.clone();
                    let geom_owned = geom.clone();
                    async move {
                        let (mask, bbox) =
                            PolygonMask::from_geojson_in_crs(&geom_owned, coord_crs_c.as_ref())?;
                        let (mx0, my0) = lonlat_deg_to_mercator(bbox[0], bbox[1]);
                        let (mx1, my1) = lonlat_deg_to_mercator(bbox[2], bbox[3]);
                        let urls = mj_c.query_bbox_3857(
                            mx0.min(mx1),
                            my0.min(my1),
                            mx0.max(mx1),
                            my0.max(my1),
                        );
                        let mut last_err: Option<EngineError> = None;
                        let mut first_empty: Option<serde_json::Value> = None;
                        for cand in urls.iter().take(16) {
                            match eng.store.open_uri(cand).await {
                                Ok(cog) => {
                                    match compute_statistics_with_options(
                                        &eng,
                                        &cog,
                                        max_size,
                                        Some(&mask),
                                        unscale,
                                        nodata,
                                        &opts,
                                    )
                                    .await
                                    {
                                        Ok(stats) => {
                                            let has_valid = stats
                                                .as_object()
                                                .is_some_and(|obj| {
                                                    obj.values().any(|b| {
                                                        b.get("valid_pixels")
                                                            .and_then(|v| v.as_u64())
                                                            .unwrap_or(0)
                                                            > 0
                                                    })
                                                });
                                            if has_valid {
                                                return Ok(stats);
                                            }
                                            if first_empty.is_none() {
                                                first_empty = Some(stats);
                                            }
                                        }
                                        Err(e) => last_err = Some(e),
                                    }
                                }
                                Err(e) => last_err = Some(e.into()),
                            }
                        }
                        if let Some(empty) = first_empty {
                            return Ok(empty);
                        }
                        Err(last_err.unwrap_or_else(|| {
                            EngineError::NotFound(
                                "no mosaic scenes intersect requested geometry".into(),
                            )
                        }))
                    }
                };
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
                        let stats = compute_for_geom(feat).await?;
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
                    let stats = compute_for_geom(&geojson).await?;
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
            ["mosaicjson", feat_ext]
                if *feat_ext == "feature" || feat_ext.starts_with("feature.") =>
            async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let (mx0, my0) = lonlat_deg_to_mercator(bbox[0], bbox[1]);
                let (mx1, my1) = lonlat_deg_to_mercator(bbox[2], bbox[3]);
                let urls = mj.query_bbox_3857(mx0.min(mx1), my0.min(my1), mx0.max(mx1), my0.max(my1));
                let ext = feat_ext
                    .strip_prefix("feature.")
                    .or(pq.format_param.as_deref())
                    .unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq)
                    .unwrap_or((cogzilla_proj::CrsModel::Geographic, 4326));
                let out_bbox = reproject_wgs84_bounds(bbox, &dst_crs)?;
                let (def_w, def_h) = aspect_preserving_dims(out_bbox, pq.max_size);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let mask_opt = pq.shape_mask.then_some(&mask);
                let data = render_mosaic_bbox_composite(
                    &engine, &urls, out_bbox, &dst_crs, out_w, out_h, mask_opt, &pq, fmt, ct, dst_epsg,
                )
                .await?;
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            ["mosaicjson", "feature", dims_ext] => async {
                let Some((out_w, out_h, ext_opt)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected /mosaicjson/feature/{w}x{h}.{ext}".into()));
                };
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let (mx0, my0) = lonlat_deg_to_mercator(bbox[0], bbox[1]);
                let (mx1, my1) = lonlat_deg_to_mercator(bbox[2], bbox[3]);
                let urls = mj.query_bbox_3857(mx0.min(mx1), my0.min(my1), mx0.max(mx1), my0.max(my1));
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq)
                    .unwrap_or((cogzilla_proj::CrsModel::Geographic, 4326));
                let out_bbox = reproject_wgs84_bounds(bbox, &dst_crs)?;
                let mask_opt = pq.shape_mask.then_some(&mask);
                let data = render_mosaic_bbox_composite(
                    &engine, &urls, out_bbox, &dst_crs, out_w, out_h, mask_opt, &pq, fmt, ct, dst_epsg,
                )
                .await?;
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            _ => Err(EngineError::NotFound(path.clone())),
        };
        let mut resp = to_http_response(&engine, res, if_none_match.as_deref());
        if let Some((bbox, epsg, is_crs84)) = geo_meta {
            attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
        }
        return Ok(resp);
    }

    if method == Method::GET && segs.first() == Some(&"mosaicjson") {
        let mj_url = pq.urls.first().cloned();
        let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
        let res: Result<(Bytes, &'static str), EngineError> = match segs.as_slice() {
            ["mosaicjson"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let clean = strip_null_object_fields(mj.raw_json.clone());
                Ok((Bytes::from(clean.to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "viewer"] => {
                let tj_url = format!("{forwarded_prefix}/mosaicjson/WebMercatorQuad/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
            ["mosaicjson", "info"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let mut info = mj.info_json();
                apply_info_bounds_crs(&mut info, mj.bounds, pq.bounds_crs.as_ref())?;
                Ok((Bytes::from(info.to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "info.geojson"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let mut props = mj.info_json();
                apply_info_bounds_crs(&mut props, mj.bounds, pq.bounds_crs.as_ref())?;
                let [w, s, e, n] = mj.bounds;
                let feat = serde_json::json!({
                    "type": "Feature",
                    "bbox": mj.bounds,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
                    },
                    "properties": props
                });
                Ok((Bytes::from(feat.to_string()), "application/geo+json"))
            }
            .await,
            ["mosaicjson", "assets"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let assets: Vec<String> = if let Some(b) = pq.bbox_param {
                    let [mx0, my0, mx1, my1] = crate::catalog::wgs84_bbox_to_3857(b);
                    mj.query_bbox_3857(mx0, my0, mx1, my1)
                } else {
                    let [mx0, my0, mx1, my1] = crate::catalog::wgs84_bbox_to_3857(mj.bounds);
                    mj.query_bbox_3857(mx0, my0, mx1, my1)
                };
                Ok((Bytes::from(serde_json::json!(assets).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "bounds"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let bounds = if let Some((ref dst, _)) = pq.bounds_crs {
                    reproject_wgs84_bounds(mj.bounds, dst)?
                } else {
                    mj.bounds
                };
                let body = serde_json::json!({ "bounds": bounds }).to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,
            ["mosaicjson", "statistics"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let [mx0, my0, mx1, my1] = crate::catalog::wgs84_bbox_to_3857(mj.bounds);
                let urls = mj.query_bbox_3857(mx0, my0, mx1, my1);
                let stats_opts = build_stats_options(&pq, false)?;
                let mut last_err: Option<EngineError> = None;
                let mut first_empty: Option<serde_json::Value> = None;
                for cand in urls.iter().take(16) {
                    match engine.store.open_uri(cand).await {
                        Ok(cog) => {
                            match compute_statistics_with_options(
                                &engine, &cog, pq.max_size, None, pq.unscale, pq.nodata, &stats_opts,
                            )
                            .await
                            {
                                Ok(json) => {
                                    let has_valid = json.as_object().is_some_and(|obj| {
                                        obj.values().any(|b| {
                                            b.get("valid_pixels")
                                                .and_then(|v| v.as_u64())
                                                .unwrap_or(0)
                                                > 0
                                        })
                                    });
                                    if has_valid {
                                        return Ok((Bytes::from(json.to_string()), "application/json"));
                                    }
                                    if first_empty.is_none() {
                                        first_empty = Some(json);
                                    }
                                }
                                Err(e) => last_err = Some(e),
                            }
                        }
                        Err(e) => last_err = Some(e.into()),
                    }
                }
                if let Some(empty) = first_empty {
                    return Ok((Bytes::from(empty.to_string()), "application/json"));
                }
                Err(last_err.unwrap_or_else(|| {
                    EngineError::NotFound("mosaicjson contains no readable raster scenes".into())
                }))
            }
            .await,
            ["mosaicjson", "bbox", bbox_seg] => async {
                let Some((bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                let urls = query_mosaicjson_wgs84_bbox(&mj, wgs_bbox);
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq).unwrap_or_else(|| {
                    if pq.coord_crs_3857 {
                        (cogzilla_proj::CrsModel::Epsg3857, 3857)
                    } else {
                        (cogzilla_proj::CrsModel::Geographic, 4326)
                    }
                });
                let out_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                };
                let (def_w, def_h) = aspect_preserving_dims(out_bbox, pq.max_size);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let data = render_mosaic_bbox_composite(
                    &engine, &urls, out_bbox, &dst_crs, out_w, out_h, None, &pq, fmt, ct, dst_epsg,
                )
                .await?;
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            ["mosaicjson", "bbox", bbox_seg, "assets"] => async {
                let Some((bbox, _)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                let urls = query_mosaicjson_wgs84_bbox(&mj, wgs_bbox);
                Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "bbox", bbox_seg, dims_ext] => async {
                let Some((bbox, _)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let Some((out_w, out_h, ext_opt)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected {w}x{h}.{ext}".into()));
                };
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                let urls = query_mosaicjson_wgs84_bbox(&mj, wgs_bbox);
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq).unwrap_or_else(|| {
                    if pq.coord_crs_3857 {
                        (cogzilla_proj::CrsModel::Epsg3857, 3857)
                    } else {
                        (cogzilla_proj::CrsModel::Geographic, 4326)
                    }
                });
                let out_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                };
                let data = render_mosaic_bbox_composite(
                    &engine, &urls, out_bbox, &dst_crs, out_w, out_h, None, &pq, fmt, ct, dst_epsg,
                )
                .await?;
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            ["mosaicjson", "tiles"] => async {
                let bounds = if let Some(ref u) = mj_url {
                    load_mosaicjson(&engine, u).await.ok().map(|m| m.bounds)
                } else {
                    None
                };
                let body = build_ogc_tilesets_list(
                    &host,
                    &forwarded_prefix,
                    "mosaicjson",
                    &raw_query,
                    bounds,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles", tms] => async {
                if let Some(tms_def) = crate::tms::lookup_tms(tms) {
                    let (bounds, minz, maxz) = if let Some(ref u) = mj_url
                        && let Ok(m) = load_mosaicjson(&engine, u).await
                    {
                        (
                            Some(m.bounds),
                            pq.minzoom.or(Some(m.minzoom)),
                            pq.maxzoom.or(Some(m.maxzoom)),
                        )
                    } else {
                        (None, pq.minzoom, pq.maxzoom)
                    };
                    let body = build_ogc_tileset_json(
                        &host,
                        &forwarded_prefix,
                        "mosaicjson",
                        tms_def,
                        &raw_query,
                        bounds,
                        minz,
                        maxz,
                    )
                    .to_string();
                    Ok((Bytes::from(body), "application/json"))
                } else {
                    Err(EngineError::NotFound(format!(
                        "unknown TileMatrixSet '{tms}'"
                    )))
                }
            }
            .await,
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
                let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
                let (mx, my) = if pq.coord_crs_3857 {
                    (x, y)
                } else {
                    lonlat_deg_to_mercator(lon, lat)
                };
                let urls = mj.query_bbox_3857(mx, my, mx, my);
                let mut assets_out = Vec::new();
                for asset_url in urls.into_iter().take(16) {
                    let Ok(cog) = engine.store.open_uri(&asset_url).await else {
                        continue;
                    };
                    match query_point(&engine, &cog, lon, lat, pq.unscale, pq.nodata).await {
                        Ok(mut pt) => {
                            apply_point_filters(&mut pt, pq.expr_raw.as_deref(), &pq.bidx)?;
                            let vals = pt["values"].clone();
                            let names = pt["band_names"].clone();
                            let descs = pt
                                .get("band_descriptions")
                                .cloned()
                                .unwrap_or_else(|| names.clone());
                            assets_out.push(serde_json::json!({
                                "name": asset_url,
                                "values": vals,
                                "band_names": names,
                                "band_descriptions": descs,
                            }));
                        }
                        Err(EngineError::NotFound(_)) => continue,
                        Err(e) => return Err(e),
                    }
                }
                if assets_out.is_empty() {
                    return Err(EngineError::NoContent);
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
                    if crate::tms::lookup_tms(tms).is_none() {
                        return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                    }
                    let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                    let mj = load_mosaicjson(&engine, &u).await?;
                    let (minz, maxz) =
                        resolve_zoom_range(mj.minzoom, mj.maxzoom, pq.minzoom, pq.maxzoom);
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
                    "version": mj.doc_version,
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
            ["mosaicjson", "point", coord, "assets"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let Some((lons, lats)) = coord.split_once(',') else {
                    return Err(EngineError::BadRequest("expected /mosaicjson/point/{lon},{lat}/assets".into()));
                };
                let (Ok(x), Ok(y)) = (lons.parse::<f64>(), lats.parse::<f64>()) else {
                    return Err(EngineError::BadRequest("invalid lon,lat floats".into()));
                };
                let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
                let (mx, my) = lonlat_deg_to_mercator(lon, lat);
                let eps = 1e-6;
                let urls = mj.query_bbox_3857(mx - eps, my - eps, mx + eps, my + eps);
                Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles", tms, zs, xs, ys, "assets"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let (Ok(z), Ok(x), Ok(y)) = (zs.parse::<u8>(), xs.parse::<u32>(), ys.parse::<u32>()) else {
                    return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                };
                if z > 30 {
                    return Err(EngineError::BadRequest(format!("invalid zoom level z={z} (expected 0..=30)")));
                }
                let urls = if *tms == "WebMercatorQuad" {
                    let max_xy = 1u64 << (z as u32);
                    if (x as u64) >= max_xy || (y as u64) >= max_xy {
                        return Err(EngineError::BadRequest(format!(
                            "tile coordinate ({x}, {y}) out of range for WebMercatorQuad z={z} (expected x, y < {max_xy})"
                        )));
                    }
                    let b = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, 256);
                    mj.query_bbox_3857(b[0], b[1], b[2], b[3])
                } else {
                    let Some(tms_def) = crate::tms::lookup_tms(tms) else {
                        return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                    };
                    let Some(tb) = tms_def.tile_bounds(z, x, y, pq.buffer, 256) else {
                        return Err(EngineError::BadRequest(format!(
                            "tile coordinate ({z}/{x}/{y}) out of range for TileMatrixSet '{tms}'"
                        )));
                    };
                    let dst_crs = tms_def.crs_model()?;
                    let wgs = bbox_to_wgs84(tb, Some(&dst_crs))?;
                    query_mosaicjson_wgs84_bbox(&mj, wgs)
                };
                Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles", zs, xs, ys, "assets"]
            | ["mosaicjson", zs, xs, ys, "assets"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let (Ok(z), Ok(x), Ok(y)) = (zs.parse::<u8>(), xs.parse::<u32>(), ys.parse::<u32>()) else {
                    return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                };
                if z > 30 {
                    return Err(EngineError::BadRequest(format!("invalid zoom level z={z} (expected 0..=30)")));
                }
                let max_xy = 1u64 << (z as u32);
                if (x as u64) >= max_xy || (y as u64) >= max_xy {
                    return Err(EngineError::BadRequest(format!(
                        "tile coordinate ({x}, {y}) out of range for WebMercatorQuad z={z} (expected x, y < {max_xy})"
                    )));
                }
                let b = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, 256);
                let urls = mj.query_bbox_3857(b[0], b[1], b[2], b[3]);
                Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles", _, zs, xs, ys]
            | ["mosaicjson", "tiles", zs, xs, ys] => {
                let tms = if segs.len() == 6 { segs[2] } else { "WebMercatorQuad" };
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
                        let auto_fmt = ext.is_none() && pq.format_param.is_none();
                        if tms != "WebMercatorQuad" {
                            let tms_def = crate::tms::lookup_tms(tms).ok_or_else(|| {
                                EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'"))
                            })?;
                            let query_ts = match sc {
                                Some(s) if (1..=4).contains(&s) => s * 256,
                                _ => pq.tile_size,
                            };
                            let tb = tms_def
                                .tile_bounds(z, x, y, pq.buffer, query_ts)
                                .ok_or_else(|| {
                                    EngineError::BadRequest(format!(
                                        "tile coordinate ({z}/{x}/{y}) out of range for TileMatrixSet '{tms}'"
                                    ))
                                })?;
                            let dst_crs = tms_def.crs_model()?;
                            let wgs = bbox_to_wgs84(tb, Some(&dst_crs))?;
                            let urls = query_mosaicjson_wgs84_bbox(&mj, wgs);
                            let empty_default_bands = if pq.bidx.is_empty()
                                && (pq.colormap_explicit || pq.custom_colormap_json.is_some())
                            {
                                1
                            } else {
                                3
                            };
                            let (mut treq, ct) = build_tile_request_for_tms(
                                Some(&engine),
                                tile,
                                sc,
                                ext,
                                &pq,
                                empty_default_bands,
                                cogzilla_tiff::SampleType::Uint8,
                                tms,
                                None,
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                            if auto_fmt {
                                treq.format = OutputFormat::RawRgba;
                            }
                            if urls.is_empty() {
                                return Err(EngineError::NoContent);
                            }
                            let data = render_mosaic_bbox_composite(
                                &engine,
                                &urls,
                                tb,
                                &dst_crs,
                                out_dim,
                                out_dim,
                                None,
                                &pq,
                                treq.format,
                                if auto_fmt { "application/octet-stream" } else { ct },
                                tms_def.epsg,
                            )
                            .await?;
                            if auto_fmt {
                                return Ok(encode_auto_image(
                                    out_dim as usize,
                                    out_dim as usize,
                                    &data,
                                    pq.jpeg_quality,
                                    pq.return_mask,
                                ));
                            }
                            return Ok((data, ct));
                        }
                        let query_ts = match sc {
                            Some(s) if (1..=4).contains(&s) => s * 256,
                            _ => pq.tile_size,
                        };
                        let buf_bbox =
                            web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, query_ts);
                        let urls = mj.query_bbox_3857(
                            buf_bbox[0],
                            buf_bbox[1],
                            buf_bbox[2],
                            buf_bbox[3],
                        );
                        if urls.is_empty() {
                            let empty_default_bands = if pq.bidx.is_empty()
                                && (pq.colormap_explicit || pq.custom_colormap_json.is_some())
                            {
                                1
                            } else {
                                3
                            };
                            let (treq, _) = build_tile_request(
                                Some(&engine),
                                tile,
                                sc,
                                ext,
                                &pq,
                                empty_default_bands,
                                cogzilla_tiff::SampleType::Uint8,
                                false,
                                None,
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            Err(EngineError::NoContent)
                        } else {
                            let first_cog = open_first_mosaic_cog(&engine, &urls).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let st = first_cog.index.full_res().sample_type;
                            let (mut treq, ct) = build_tile_request(
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
                            let ts = treq.tile_size;
                            if auto_fmt {
                                treq.format = OutputFormat::RawRgba;
                            }
                            let fmt = treq.format;
                            let out_dim = (ts as f32 + 2.0 * pq.buffer).round() as u32;
                            let buf_bbox = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, ts);
                            let b = render_mosaic_tile_styled(
                                &engine,
                                &urls,
                                treq,
                                &pq,
                                first_cog.index.color_map.is_some(),
                            )
                            .await?;
                            Ok(finalize_image_output(
                                b,
                                ct,
                                fmt,
                                auto_fmt,
                                out_dim,
                                out_dim,
                                buf_bbox,
                                3857,
                                pq.jpeg_quality,
                                pq.return_mask,
                            ))
                        }
                    })
                    .await;
                let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                if let (Ok(z), Ok(x), Some((y, sc, _))) =
                    (zs.parse::<u8>(), xs.parse::<u32>(), parse_y_scale_ext(ys))
                {
                    attach_tile_geo_headers(&mut resp, tms, z, x, y, sc, &pq);
                }
                return Ok(resp);
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        let mut resp = to_http_response(&engine, res, if_none_match.as_deref());
        if let Some((bbox, epsg, is_crs84)) = geo_meta {
            attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
        }
        return Ok(resp);
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
                            let first_cog = open_first_mosaic_cog(&engine, &pq.urls).await?;
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
                            let ts = treq.tile_size;
                            let fmt = treq.format;
                            let out_dim = (ts as f32 + 2.0 * pq.buffer).round() as u32;
                            let buf_bbox = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, ts);
                            let b = render_mosaic_tile_styled(
                                &engine,
                                &pq.urls,
                                treq,
                                &pq,
                                first_cog.index.color_map.is_some(),
                            )
                            .await?;
                            let b = wrap_geotiff_fmt(
                                b,
                                ct,
                                fmt,
                                out_dim,
                                out_dim,
                                buf_bbox,
                                3857,
                                pq.return_mask,
                            );
                            Ok((b, ct))
                        })
                        .await;
                    let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                    attach_tile_geo_headers(&mut resp, "WebMercatorQuad", z, x, y, sc, &pq);
                    return Ok(resp);
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
                        let query_ts = match sc {
                            Some(s) if (1..=4).contains(&s) => s * 256,
                            _ => pq.tile_size,
                        };
                        let buf_bbox =
                            web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, query_ts);
                        let urls = catalog.query_bbox_3857(
                            buf_bbox[0],
                            buf_bbox[1],
                            buf_bbox[2],
                            buf_bbox[3],
                        );
                        if urls.is_empty() {
                            let empty_default_bands = if pq.bidx.is_empty()
                                && (pq.colormap_explicit || pq.custom_colormap_json.is_some())
                            {
                                1
                            } else {
                                3
                            };
                            let (treq, ct) = build_tile_request(
                                Some(&engine),
                                tile,
                                sc,
                                ext,
                                &pq,
                                empty_default_bands,
                                cogzilla_tiff::SampleType::Uint8,
                                false,
                                None,
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                            let nb = default_raw_band_count(&treq.bands, pq.pixel_selection);
                            let b = if treq.format == OutputFormat::Npy {
                                let raw = cogzilla_kernel::empty_tile_bytes(
                                    out_dim,
                                    OutputFormat::RawF32,
                                    nb,
                                );
                                cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                                    raw,
                                    nb,
                                    out_dim,
                                    out_dim,
                                    pq.return_mask,
                                )
                            } else {
                                cogzilla_kernel::empty_tile_bytes(out_dim, treq.format, nb)
                            };
                            let b = wrap_geotiff_fmt(
                                b,
                                ct,
                                treq.format,
                                out_dim,
                                out_dim,
                                buf_bbox,
                                3857,
                                pq.return_mask,
                            );
                            Ok((b, ct))
                        } else {
                            let first_cog = open_first_mosaic_cog(&engine, &urls).await?;
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
                            let ts = treq.tile_size;
                            let fmt = treq.format;
                            let out_dim = (ts as f32 + 2.0 * pq.buffer).round() as u32;
                            let buf_bbox = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, ts);
                            let b = render_mosaic_tile_styled(
                                &engine,
                                &urls,
                                treq,
                                &pq,
                                first_cog.index.color_map.is_some(),
                            )
                            .await?;
                            let b = wrap_geotiff_fmt(
                                b,
                                ct,
                                fmt,
                                out_dim,
                                out_dim,
                                buf_bbox,
                                3857,
                                pq.return_mask,
                            );
                            Ok((b, ct))
                        }
                    })
                    .await;
                let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                attach_tile_geo_headers(&mut resp, "WebMercatorQuad", z, x, y, sc, &pq);
                return Ok(resp);
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
                                || pq.buffer > 0.0
                                || (!pq.return_mask && matches!(treq.format, OutputFormat::Npy))
                                || (pq.color_formula.is_some()
                                    && !matches!(
                                        treq.bands,
                                        BandRequest::Rgb { .. }
                                            | BandRequest::SarRgb { .. }
                                            | BandRequest::CustomExprRgb { .. }
                                            | BandRequest::Terrain { .. }
                                    ))
                                || (treq.tile_size != 256 && treq.tile_size != 512)
                                || matches!(treq.bands, BandRequest::CustomExprRgb { .. } | BandRequest::MultiBand { .. })
                                || has_embedded_palette(&pq, &cogs[0], &treq.bands)
                                || !crate::engine::cogs_share_base_grid(&cogs);
                            if needs_general_path {
                                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &treq.bands)?;
                                let cf = pq
                                    .color_formula
                                    .as_deref()
                                    .map(ColorFormula::parse)
                                    .transpose()
                                    .map_err(EngineError::BadRequest)?;
                                let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
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
                                let data = wrap_geotiff_fmt(
                                    data,
                                    ct,
                                    treq.format,
                                    out_dim,
                                    out_dim,
                                    bbox_3857,
                                    3857,
                                    pq.return_mask,
                                );
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
                                let (min_x, min_y, max_x, max_y) = tile_bounds_3857(treq.tile);
                                let data = wrap_geotiff_fmt(
                                    data,
                                    ct,
                                    orig_format,
                                    treq.tile_size,
                                    treq.tile_size,
                                    [min_x, min_y, max_x, max_y],
                                    3857,
                                    pq.return_mask,
                                );
                                Ok((data, ct))
                            } else {
                                let (min_x, min_y, max_x, max_y) = tile_bounds_3857(treq.tile);
                                let b = engine.render_multi_cog_tile(&cogs, &treq).await?;
                                let b = wrap_geotiff_fmt(
                                    b,
                                    ct,
                                    treq.format,
                                    treq.tile_size,
                                    treq.tile_size,
                                    [min_x, min_y, max_x, max_y],
                                    3857,
                                    pq.return_mask,
                                );
                                Ok((b, ct))
                            }
                        })
                        .await;
                    let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                    attach_tile_geo_headers(&mut resp, "WebMercatorQuad", z, x, y, sc, &pq);
                    return Ok(resp);
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
                Some("png") => get_colormap_image(
                    id,
                    OutputFormat::Png,
                    pq.orientation_vertical,
                    pq.width,
                    pq.height,
                )
                .map(|b| (b, "image/png")),
                Some("jpg" | "jpeg") => get_colormap_image(
                    id,
                    OutputFormat::Jpeg { quality: pq.jpeg_quality },
                    pq.orientation_vertical,
                    pq.width,
                    pq.height,
                )
                .map(|b| (b, "image/jpeg")),
                Some("webp") => get_colormap_image(
                    id,
                    OutputFormat::Webp { quality: pq.webp_quality },
                    pq.orientation_vertical,
                    pq.width,
                    pq.height,
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
        let stac_x_assets = pq.assets.clone();
        let stac_url_hdr = pq.urls.first().cloned();
        // An item read from pgSTAC is held here; one named by `url` is in the cache by then.
        let stac_item_hdr = pgstac_item.clone();
        let stac_req_assets = pq.assets.clone();
        let stac_req_expr = pq.expr_raw.clone();
        let attach_resolved_stac_assets = |resp: &mut Response<Full<Bytes>>| {
            if !resp.status().is_success() {
                return;
            }
            let item = stac_item_hdr
                .clone()
                .or_else(|| stac_url_hdr.as_ref().and_then(|u| engine.stac_cache.get(u).map(|item| item.value().clone())));
            if let Some(item) = item
                && let Ok(names) = crate::stac::select_stac_asset_names(
                    &item,
                    &stac_req_assets,
                    stac_req_expr.as_deref(),
                )
            {
                attach_x_assets(resp, &names);
            }
        };
        let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
        let stac_source = match pgstac_item.take() {
            Some(item) => StacSource::Item(item),
            None => StacSource::Url(pq.urls.first().cloned()),
        };
        let res: Result<(Bytes, &'static str), EngineError> = match (method.clone(), sub) {
            // GET /stac/assets?url=item.json
            (Method::GET, ["assets"]) => async {
                let item = stac_source.load(&engine).await?;
                let keys: Vec<&String> = item.assets.keys().collect();
                Ok((Bytes::from(serde_json::json!(keys).to_string()), "application/json"))
            }
            .await,

            // GET /stac/bounds?url=item.json
            (Method::GET, ["bounds"]) => async {
                let item = stac_source.load(&engine).await?;
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
                let bounds = if let Some((ref dst, _)) = pq.bounds_crs {
                    reproject_wgs84_bounds(bounds, dst)?
                } else {
                    bounds
                };
                let body = serde_json::json!({"bounds": bounds}).to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // GET /stac/info?url=item.json&assets=...
            (Method::GET, ["info"]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let mut map = serde_json::Map::new();
                for (name, cog) in names.into_iter().zip(cogs.into_iter()) {
                    let mut info = build_cog_info(&engine, &cog)?;
                    if pq.bounds_crs.is_some() {
                        let crs = engine.get_or_build_crs(&cog)?;
                        let full = cog.index.full_res();
                        let wgs84 = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                        apply_info_bounds_crs(&mut info, wgs84, pq.bounds_crs.as_ref())?;
                    }
                    map.insert(name, info);
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,

            // GET /stac/info.geojson?url=item.json&assets=...
            (Method::GET, ["info.geojson"]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let mut props = serde_json::Map::new();
                for (name, cog) in names.into_iter().zip(cogs.into_iter()) {
                    let mut info = build_cog_info(&engine, &cog)?;
                    if pq.bounds_crs.is_some() {
                        let crs = engine.get_or_build_crs(&cog)?;
                        let full = cog.index.full_res();
                        let wgs84 = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                        apply_info_bounds_crs(&mut info, wgs84, pq.bounds_crs.as_ref())?;
                    }
                    props.insert(name, info);
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
                let item = stac_source.load(&engine).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let n = names.len();
                let stats_opts = build_stats_options(&pq, false)?;
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let mut opts_c = stats_opts.clone();
                    if let Some(expr_str) = stac_asset_expr(&pq, &names[idx]) {
                        let prog = ExprProgram::compile(expr_str).map_err(EngineError::BadRequest)?;
                        opts_c.expressions = Some(vec![(expr_str.to_string(), prog)]);
                    } else if let Some(sel) = stac_asset_selected_bands(&pq, &names[idx]) {
                        opts_c.bidx = Some(sel.to_vec());
                    }
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
                    if let Some(mut st) = st {
                        if let Some(sel) = stac_asset_selected_bands(&pq, &name) {
                            if let Some(obj) = st.as_object() {
                                let mut filtered = serde_json::Map::new();
                                for &b in sel {
                                    let k = format!("b{b}");
                                    if let Some(v) = obj.get(&k) {
                                        filtered.insert(k, v.clone());
                                    }
                                }
                                st = serde_json::Value::Object(filtered);
                            }
                        }
                        map.insert(name, st);
                    }
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,
            // GET /stac/statistics (merged band dict: {b1: ..., b2: ...})
            (Method::GET, ["statistics"]) => async {
                // An item's statistics take band math on the planes road or not at all:
                // the names this branch reads (`red_b1`) are not titiler-pgstac's.
                if stac_prefix != "stac" && pq.expr_raw.is_some() {
                    return Err(EngineError::BadRequest(
                        "parameter 'expression' is only supported on /cog/statistics routes".into(),
                    ));
                }
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let is_dem_alg = pq.algorithm.as_deref().is_some_and(|a| {
                    matches!(
                        a.to_ascii_lowercase().as_str(),
                        "hillshade" | "slope" | "aspect" | "contours" | "terrarium" | "terrainrgb" | "terrain-rgb" | "mapbox"
                    )
                });
                if is_dem_alg {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                    let mut bands = build_band_request(&pq, db, full.sample_type)?;
                    if pq.unscale {
                        apply_unscale_cogs(&mut bands, &cogs);
                    }
                    let stats_opts = build_stats_options(&pq, false)?;
                    let raw = render_native_preview(
                        &engine,
                        &cogs,
                        bbox,
                        out_w,
                        out_h,
                        &bands,
                        pq.resample,
                        OutputFormat::RawF32,
                        pq.nodata,
                        None,
                        None,
                        true,
                        pq.unscale,
                    )
                    .await?;
                    let n_px = (out_w as usize) * (out_h as usize);
                    let nb = if n_px > 0 && raw.len() >= n_px * 4 {
                        (raw.len() / (n_px * 4)).clamp(1, 16)
                    } else {
                        1
                    };
                    let alg_name = pq.algorithm.as_deref().unwrap_or("algorithm");
                    let mut map = serde_json::Map::new();
                    for b in 0..nb {
                        let start = b * n_px * 4;
                        let end = (start + n_px * 4).min(raw.len());
                        let plane: Vec<f32> = raw[start..end]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let desc = if nb == 1 {
                            alg_name.to_string()
                        } else {
                            format!("{alg_name}_b{}", b + 1)
                        };
                        map.insert(
                            format!("b{}", b + 1),
                            compute_f32_band_stats(&plane, &desc, &stats_opts),
                        );
                    }
                    return Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"));
                }
                let mut expr_progs: Option<Vec<(String, ExprProgram)>> = None;
                if let Some(ref expr_str) = rewritten_expr {
                    let parts: Vec<&str> = expr_str
                        .split(';')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .collect();
                    if parts.len() > 16 {
                        return Err(EngineError::BadRequest(
                            "expression exceeds max 16 semicolon-separated outputs".into(),
                        ));
                    }
                    let raw_parts: Vec<&str> = pq
                        .expr_raw
                        .as_deref()
                        .unwrap_or(expr_str)
                        .split(';')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .collect();
                    let mut compiled = Vec::with_capacity(parts.len());
                    for (idx, part) in parts.iter().enumerate() {
                        let prog = ExprProgram::compile(part).map_err(EngineError::BadRequest)?;
                        let desc = raw_parts.get(idx).copied().unwrap_or(part).to_string();
                        compiled.push((desc, prog));
                    }
                    expr_progs = Some(compiled);
                } else if pq.algorithm.is_some() {
                    let db = if cogs.len() > 1 {
                        cogs.len() as u16
                    } else {
                        cogs[0].index.full_res().samples_per_pixel
                    };
                    expr_progs = build_stats_options_with_bands(&pq, true, db)?.expressions;
                }
                if let Some(compiled) = expr_progs {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let stats_opts = build_stats_options(&pq, false)?;
                    let mut map = serde_json::Map::new();
                    for (idx, (desc, prog)) in compiled.into_iter().enumerate() {
                        let mut bands = BandRequest::CustomExpr {
                            prog,
                            rescale: (0.0, 1.0),
                            colormap: ColormapName::Grayscale,
                        };
                        if pq.unscale {
                            apply_unscale_cogs(&mut bands, &cogs);
                        }
                        let raw = render_native_preview(
                            &engine,
                            &cogs,
                            bbox,
                            out_w,
                            out_h,
                            &bands,
                            pq.resample,
                            OutputFormat::RawF32,
                            pq.nodata,
                            None,
                            None,
                            true,
                            pq.unscale,
                        )
                        .await?;
                        let pixels: Vec<f32> = raw
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        map.insert(
                            format!("b{}", idx + 1),
                            compute_f32_band_stats(&pixels, &desc, &stats_opts),
                        );
                    }
                    return Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"));
                }
                let n = cogs.len();
                let stats_opts = build_stats_options(&pq, false)?;
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let mut opts_c = stats_opts.clone();
                    if let Some(expr_str) = stac_asset_expr(&pq, &names[idx]) {
                        let prog = ExprProgram::compile(expr_str).map_err(EngineError::BadRequest)?;
                        opts_c.expressions = Some(vec![(expr_str.to_string(), prog)]);
                    } else if let Some(sel) = stac_asset_selected_bands(&pq, &names[idx]) {
                        opts_c.bidx = Some(sel.to_vec());
                    }
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
                        let bands_iter: Vec<usize> = match stac_asset_selected_bands(&pq, name) {
                            Some(sel) => sel.iter().map(|&b| b as usize).collect(),
                            None if pq.asset_as_band => vec![1],
                            None if !pq.bidx.is_empty() => {
                                pq.bidx.iter().map(|&b| b as usize).collect()
                            }
                            None => (1..=nb).collect(),
                        };
                        for i in bands_iter {
                            if let Some(v) = obj.get(&format!("b{i}")) {
                                let mut band_val = v.clone();
                                if let Some(bobj) = band_val.as_object_mut() {
                                    let desc = if pq.asset_as_band {
                                        name.clone()
                                    } else {
                                        format!("{name}_b{i}")
                                    };
                                    bobj.insert(
                                        "description".to_string(),
                                        serde_json::json!(desc),
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
                let item = stac_source.load(&engine).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                // As on GET: an item's statistics take no band math in these names.
                if stac_prefix != "stac" && pq.expr_raw.is_some() {
                    return Err(EngineError::BadRequest(
                        "parameter 'expression' is only supported on /cog/statistics routes".into(),
                    ));
                }
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;

                let is_dem_alg = pq.algorithm.as_deref().is_some_and(|a| {
                    matches!(
                        a.to_ascii_lowercase().as_str(),
                        "hillshade" | "slope" | "aspect" | "contours" | "terrarium" | "terrainrgb" | "terrain-rgb" | "mapbox"
                    )
                });
                let post_expr_progs: Option<Vec<(String, ExprProgram)>> = if let Some(ref expr_str) = rewritten_expr {
                    let parts: Vec<&str> = expr_str.split(';').map(str::trim).filter(|s| !s.is_empty()).collect();
                    if parts.len() > 16 {
                        return Err(EngineError::BadRequest(
                            "expression exceeds max 16 semicolon-separated outputs".into(),
                        ));
                    }
                    let mut compiled = Vec::with_capacity(parts.len());
                    for part in parts {
                        compiled.push((part.to_string(), ExprProgram::compile(part).map_err(EngineError::BadRequest)?));
                    }
                    Some(compiled)
                } else if pq.algorithm.is_some() && !is_dem_alg {
                    let db = if cogs.len() > 1 {
                        cogs.len() as u16
                    } else {
                        cogs[0].index.full_res().samples_per_pixel
                    };
                    build_stats_options_with_bands(&pq, true, db)?.expressions
                } else {
                    None
                };
                let stats_opts_post = build_stats_options(&pq, false)?;
                let compute_stac_merged_stats = |mask: Arc<PolygonMask>, poly_bbox: [f64; 4]| {
                    let cogs_c = cogs.clone();
                    let names_c = names.clone();
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let width_opt = pq.width;
                    let height_opt = pq.height;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let resample = pq.resample;
                    let stats_opts = stats_opts_post.clone();
                    let asset_bidx_c = pq.asset_bidx.clone();
                    let pq_bidx_c = pq.bidx.clone();
                    let asset_as_band = pq.asset_as_band;
                    let expr_progs_c = post_expr_progs.clone();
                    let alg_c = pq.algorithm.clone();
                    let pq_clone = pq.clone();
                    let n = names_c.len();
                    async move {
                        if is_dem_alg || expr_progs_c.is_some() {
                            let (def_w, def_h) = aspect_preserving_dims(poly_bbox, max_size);
                            let (out_w, out_h) = resolve_query_dims(def_w, def_h, width_opt, height_opt);
                            let n_px = (out_w as usize) * (out_h as usize);
                            if is_dem_alg {
                                let full = cogs_c[0].index.full_res();
                                let db = if cogs_c.len() > 1 { cogs_c.len() as u16 } else { full.samples_per_pixel };
                                let mut bands = build_band_request(&pq_clone, db, full.sample_type)?;
                                if unscale {
                                    apply_unscale_cogs(&mut bands, &cogs_c);
                                }
                                let raw = render_wgs84_bbox(
                                    &eng, &cogs_c, poly_bbox, out_w, out_h, Some(&mask), &bands,
                                    resample, OutputFormat::RawF32, nodata, None, None, true, unscale,
                                )
                                .await?;
                                let nb = if n_px > 0 && raw.len() >= n_px * 4 {
                                    (raw.len() / (n_px * 4)).clamp(1, 16)
                                } else {
                                    1
                                };
                                let alg_name = alg_c.as_deref().unwrap_or("algorithm");
                                let mut map = serde_json::Map::new();
                                for b in 0..nb {
                                    let start = b * n_px * 4;
                                    let end = (start + n_px * 4).min(raw.len());
                                    let plane: Vec<f32> = raw[start..end]
                                        .chunks_exact(4)
                                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                        .collect();
                                    let desc = if nb == 1 { alg_name.to_string() } else { format!("{alg_name}_b{}", b + 1) };
                                    map.insert(format!("b{}", b + 1), compute_f32_band_stats(&plane, &desc, &stats_opts));
                                }
                                return Ok::<_, EngineError>(serde_json::Value::Object(map));
                            }
                            let compiled = expr_progs_c.unwrap();
                            let mut map = serde_json::Map::new();
                            for (idx, (desc, prog)) in compiled.into_iter().enumerate() {
                                let mut bands = BandRequest::CustomExpr {
                                    prog,
                                    rescale: (0.0, 1.0),
                                    colormap: ColormapName::Grayscale,
                                };
                                if unscale {
                                    apply_unscale_cogs(&mut bands, &cogs_c);
                                }
                                let raw = render_wgs84_bbox(
                                    &eng, &cogs_c, poly_bbox, out_w, out_h, Some(&mask), &bands,
                                    resample, OutputFormat::RawF32, nodata, None, None, true, unscale,
                                )
                                .await?;
                                let pixels: Vec<f32> = raw
                                    .chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect();
                                map.insert(format!("b{}", idx + 1), compute_f32_band_stats(&pixels, &desc, &stats_opts));
                            }
                            return Ok::<_, EngineError>(serde_json::Value::Object(map));
                        }
                        let mut set = tokio::task::JoinSet::new();
                        for (idx, cog) in cogs_c.into_iter().enumerate() {
                            let eng_c = eng.clone();
                            let mask_c = mask.clone();
                            let mut opts_c = stats_opts.clone();
                            if let Some((_, sel)) =
                                asset_bidx_c.iter().find(|(n, _)| n == &names_c[idx])
                            {
                                opts_c.bidx = Some(sel.to_vec());
                            }
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
                                let bands_iter: Vec<usize> = match asset_bidx_c
                                    .iter()
                                    .find(|(n, _)| n == name)
                                {
                                    Some((_, sel)) => sel.iter().map(|&b| b as usize).collect(),
                                    None if asset_as_band => vec![1],
                                    None if !pq_bidx_c.is_empty() => {
                                        pq_bidx_c.iter().map(|&b| b as usize).collect()
                                    }
                                    None => (1..=nb).collect(),
                                };
                                for b in bands_iter {
                                    if let Some(v) = obj.get(&format!("b{b}")) {
                                        let mut band_val = v.clone();
                                        if let Some(bobj) = band_val.as_object_mut() {
                                            let desc = if asset_as_band {
                                                name.clone()
                                            } else {
                                                format!("{name}_b{b}")
                                            };
                                            bobj.insert(
                                                "description".to_string(),
                                                serde_json::json!(desc),
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
                        let (mask, poly_bbox) =
                            PolygonMask::from_geojson_in_crs(feat, pq.coord_crs.as_ref())?;
                        let stats = compute_stac_merged_stats(Arc::new(mask), poly_bbox).await?;
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
                    let (mask, poly_bbox) =
                        PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                    let stats = compute_stac_merged_stats(Arc::new(mask), poly_bbox).await?;
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
            (Method::GET, ["tiles"]) => async {
                let bounds = stac_source.load(&engine).await.ok().and_then(|item| item.bbox);
                let body = build_ogc_tilesets_list(
                    &host,
                    &forwarded_prefix,
                    "stac",
                    &raw_query,
                    bounds,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,
            (Method::GET, ["tiles", tms]) => async {
                let tms_def = crate::tms::lookup_tms(tms)
                    .ok_or_else(|| EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")))?;
                let bounds = stac_source.load(&engine).await.ok().and_then(|item| item.bbox);
                let body = build_ogc_tileset_json(
                    &host,
                    &forwarded_prefix,
                    "stac",
                    tms_def,
                    &raw_query,
                    bounds,
                    pq.minzoom,
                    pq.maxzoom,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // GET /stac/{tms}/tilejson.json, /stac/tilejson.json, /stac/{tms}/map.html, /stac/map.html
            (Method::GET, [_, "tilejson.json"]) | (Method::GET, ["tilejson.json"]) => async {
                let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                let item = stac_source.load(&engine).await?;
                let (_, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut tj = build_tilejson(&engine, &cogs[0], &host, tms, &raw_query)?;
                apply_tilejson_overrides(&mut tj, &pq, &host, &stac_prefix, tms, &raw_query);
                tj["name"] = serde_json::json!(item.id);
                Ok((Bytes::from(tj.to_string()), "application/json"))
            }
            .await,
            // titiler-pgstac has no viewer on an item.
            (Method::GET, ["viewer"]) if stac_prefix != "stac" => {
                Err(EngineError::NotFound("Not Found".into()))
            }
            (Method::GET, [_, "map.html"]) | (Method::GET, ["map.html"]) | (Method::GET, ["viewer"]) => {
                let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                let tj_url =
                    format!("{forwarded_prefix}/{stac_prefix}/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
            (Method::GET, ["renders"]) => async {
                // The links name the item's address, which only an item given by `url` has.
                let u = match &stac_source {
                    StacSource::Url(Some(u)) => u.clone(),
                    _ => return Err(EngineError::BadRequest("missing ?url=".into())),
                };
                let item = stac_source.load(&engine).await?;
                let origin = if host.starts_with("http://") || host.starts_with("https://") {
                    format!("{host}{forwarded_prefix}")
                } else {
                    format!("http://{host}{forwarded_prefix}")
                };
                let mut renders_map = serde_json::Map::new();
                for (rid, rval) in &item.renders {
                    renders_map.insert(
                        rid.clone(),
                        build_stac_render_entry(&host, &forwarded_prefix, &u, rid, rval),
                    );
                }
                let out = serde_json::json!({
                    "renders": serde_json::Value::Object(renders_map),
                    "links": [
                        {
                            "rel": "self",
                            "type": "application/json",
                            "href": format!("{origin}/stac/renders?url={u}"),
                        }
                    ],
                });
                Ok((Bytes::from(out.to_string()), "application/json"))
            }
            .await,
            (Method::GET, ["renders", render_id]) => async {
                let u = match &stac_source {
                    StacSource::Url(Some(u)) => u.clone(),
                    _ => return Err(EngineError::BadRequest("missing ?url=".into())),
                };
                let item = stac_source.load(&engine).await?;
                let render = item
                    .renders
                    .get(*render_id)
                    .ok_or_else(|| EngineError::NotFound(format!("render '{render_id}' not found")))?;
                let out = build_stac_render_entry(&host, &forwarded_prefix, &u, render_id, render);
                Ok((Bytes::from(out.to_string()), "application/json"))
            }
            .await,
            // GET /stac/map (OGC API - Maps)
            (Method::GET, ["map"]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs)?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let (dst_crs, dst_epsg) = pq_stac
                    .dst_crs
                    .clone()
                    .or_else(|| pq_stac.bounds_crs.clone())
                    .unwrap_or((cogzilla_proj::CrsModel::Geographic, 4326));
                let out_bbox = if let Some(raw_bbox) = pq_stac.bbox_param {
                    if pq_stac.bbox_crs.as_ref().map(|(_, e)| *e) == Some(dst_epsg)
                        || (pq_stac.bbox_crs.is_none() && dst_epsg == 4326)
                    {
                        normalize_render_bbox(raw_bbox, &dst_crs)?
                    } else {
                        let wb = bbox_to_wgs84(raw_bbox, pq_stac.bbox_crs.as_ref().map(|(c, _)| c))?;
                        normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                    }
                } else {
                    let wb = item.bbox.unwrap_or_else(|| {
                        crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height)
                    });
                    normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                };
                let (def_w, def_h) = aspect_preserving_dims(out_bbox, pq_stac.max_size);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = pq_stac.format_param.as_deref().unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let mut bands =
                    build_band_request_for_output(&pq_stac, db, full.sample_type, fmt, ct)?;
                let fmt = geotiff_render_format(fmt, ct, &pq_stac, Some(&cogs[0]), &bands, full.sample_type);
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                let data = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), out_bbox, &dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, dst_epsg, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::GET, [_, "WMTSCapabilities.xml"]) | (Method::GET, ["WMTSCapabilities.xml"]) => {
                async {
                    let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                    if crate::tms::lookup_tms(tms).is_none() {
                        return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                    }
                    let item = stac_source.load(&engine).await?;
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
                    let (def_min, def_max) = if let Some(tms_def) = crate::tms::lookup_tms(tms) {
                        tms_def.zoom_range_for_cog(&crs, &full.pixel_to_crs, full.width, full.height, ov_factor)
                    } else {
                        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
                    };
                    let (minz, maxz) =
                        resolve_zoom_range(def_min, def_max, pq.minzoom, pq.maxzoom);
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
                // An item read from the database can change under the same path: its tile
                // is of the assets it has now.
                let cache_key = match &stac_source {
                    StacSource::Item(item) => {
                        format!("{path}?{raw_query}#{:x}", pgstac::source_identity(&item.assets))
                    }
                    StacSource::Url(_) => format!("{path}?{raw_query}"),
                };
                let stac_tile_geo = TileGeoOpts::from_pq(&pq);
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        let item = stac_source.load(&engine).await?;
                        let (names, cogs, rewritten_expr) = resolve_stac_assets(
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
                        let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs)?;
                        let db = if cogs.len() > 1 {
                            cogs.len() as u16
                        } else {
                            cogs[0].index.full_res().samples_per_pixel
                        };
                        let st = cogs[0].index.full_res().sample_type;
                        let (mut treq, ct) = build_tile_request_for_tms(
                            Some(&engine),
                            TileCoord { z, x, y },
                            sc,
                            ext,
                            &pq_stac,
                            db,
                            st,
                            tms,
                            Some(&cogs[0]),
                        )?;
                        // titiler-pgstac's item routes name the media type after the extension
                        // and, without one, choose the format from the pixels.
                        let item_route = stac_prefix != "stac";
                        let ct = if item_route && ext == Some("jpg") { "image/jpg" } else { ct };
                        let item_auto_format =
                            item_route && ext.is_none() && pq_stac.format_param.is_none();
                        if pq_stac.unscale {
                            apply_unscale_cogs(&mut treq.bands, &cogs);
                        }
                        let any_intersects = cogs.iter().any(|c| {
                            tile_intersects_cog(&engine, c, tms, TileCoord { z, x, y }, pq_stac.buffer, treq.tile_size)
                                .unwrap_or(false)
                        });
                        if !any_intersects {
                            return Err(EngineError::NotFound(format!(
                                "Tile {z}/{x}/{y} is outside image bounds"
                            )));
                        }
                        let needs_general = tms != "WebMercatorQuad"
                            || slot_bands.is_some()
                            || pq_stac.buffer > 0.0
                            || pq_stac.nodata.is_some()
                            || pq_stac.custom_colormap_json.is_some()
                            || (!pq_stac.return_mask && matches!(treq.format, OutputFormat::Npy))
                            || (pq_stac.color_formula.is_some()
                                && !matches!(
                                    treq.bands,
                                    BandRequest::Rgb { .. }
                                        | BandRequest::SarRgb { .. }
                                        | BandRequest::CustomExprRgb { .. }
                                        | BandRequest::Terrain { .. }
                                ))
                            || matches!(treq.bands, BandRequest::CustomExprRgb { .. } | BandRequest::MultiBand { .. })
                            || (treq.tile_size != 256 && treq.tile_size != 512)
                            || has_embedded_palette(&pq_stac, &cogs[0], &treq.bands)
                            || !crate::engine::cogs_share_base_grid(&cogs);
                        let auto_fmt = ext.is_none() && pq_stac.format_param.is_none();
                        if auto_fmt {
                            treq.format = OutputFormat::RawRgba;
                        }
                        if needs_general {
                            let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &treq.bands)?;
                            let cf = pq_stac
                                .color_formula
                                .as_deref()
                                .map(ColorFormula::parse)
                                .transpose()
                                .map_err(EngineError::BadRequest)?;
                            let out_dim = (treq.tile_size as f32 + 2.0 * pq_stac.buffer).round() as u32;
                            let (data, bbox, epsg) = if tms != "WebMercatorQuad" {
                                let tms_def = crate::tms::lookup_tms(tms).ok_or_else(|| {
                                    EngineError::BadRequest(format!("unsupported TMS '{tms}'"))
                                })?;
                                let bbox = tms_def
                                    .tile_bounds(z, x, y, pq_stac.buffer, treq.tile_size)
                                    .ok_or_else(|| {
                                        EngineError::NotFound(format!(
                                            "Tile {z}/{x}/{y} is outside {tms} matrix"
                                        ))
                                    })?;
                                let dst_crs = tms_def.crs_model()?;
                                let d = render_stac_bbox_opt(
                                    &engine,
                                    &cogs,
                                    slot_bands.as_deref(),
                                    bbox,
                                    &dst_crs,
                                    false,
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
                                .await?;
                                (d, bbox, tms_def.epsg)
                            } else {
                                let bbox_3857 = web_mercator_buffered_3857_bbox(
                                    z,
                                    x,
                                    y,
                                    pq_stac.buffer,
                                    treq.tile_size,
                                );
                                let d = render_stac_bbox_opt(
                                    &engine,
                                    &cogs,
                                    slot_bands.as_deref(),
                                    bbox_3857,
                                    &cogzilla_proj::CrsModel::Epsg3857,
                                    false,
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
                                .await?;
                                (d, bbox_3857, 3857u16)
                            };
                            // An item's picture with pixels missing is a PNG whether or not
                            // the mask was asked for: the format is chosen from the pixels.
                            if item_auto_format
                                && !pq_stac.return_mask
                                && data.len() == (out_dim as usize).pow(2) * 4
                                && data.chunks_exact(4).any(|p| p[3] != 255)
                            {
                                let (w, h) = (out_dim as usize, out_dim as usize);
                                return Ok((encode_rgba_output(w, h, &data, OutputFormat::Png, false), "image/png"));
                            }
                            Ok(finalize_image_output(data, ct, treq.format, auto_fmt, out_dim, out_dim, bbox, epsg, pq_stac.jpeg_quality, pq_stac.return_mask))
                        } else if item_auto_format {
                            pgstac::item_tile_auto_format(&engine, &cogs, treq, &pq_stac).await
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
                            let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                            if auto_fmt {
                                let buf = if let Some(cf) = cf {
                                    let mut b = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut b, 4);
                                    Bytes::from(b)
                                } else {
                                    raw
                                };
                                Ok(encode_auto_image(dim, dim, &buf, pq_stac.jpeg_quality, pq_stac.return_mask))
                            } else {
                                let data = if let Some(cf) = cf {
                                    let mut buf = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut buf, 4);
                                    encode_rgba_output(dim, dim, &buf, orig_format, pq_stac.return_mask)
                                } else {
                                    encode_rgba_output(dim, dim, &raw, orig_format, pq_stac.return_mask)
                                };
                                let data = wrap_geotiff_fmt(data, ct, orig_format, treq.tile_size, treq.tile_size, bbox, 3857, pq_stac.return_mask);
                                Ok((data, ct))
                            }
                        } else {
                            let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                            Ok(finalize_image_output(data, ct, treq.format, auto_fmt, treq.tile_size, treq.tile_size, bbox, 3857, pq_stac.jpeg_quality, pq_stac.return_mask))
                        }
                    })
                    .await;
                if stac_prefix != "stac"
                    && let Err(e) = &cached
                {
                    return Ok(pgstac::engine_error(e));
                }
                let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                attach_resolved_stac_assets(&mut resp);
                if let (Ok(z), Ok(x), Some((y, sc, _))) =
                    (zs.parse::<u8>(), xs.parse::<u32>(), parse_y_scale_ext(ys))
                {
                    attach_tile_geo_headers_opts(&mut resp, tms, z, x, y, sc, stac_tile_geo);
                }
                // An item's tile says where it lies as titiler-pgstac writes it.
                if stac_prefix != "stac" {
                    resp = pgstac::item_placed(resp, &segs[1..], &raw_query);
                }
                return Ok(resp);
            }

            // GET /stac/point/{lon},{lat}
            (Method::GET, ["point", coord]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let Some((lons, lats)) = coord.split_once(',') else {
                    return Err(EngineError::BadRequest("expected /stac/point/{lon},{lat}".into()));
                };
                let (Ok(x), Ok(y)) = (lons.parse::<f64>(), lats.parse::<f64>()) else {
                    return Err(EngineError::BadRequest("invalid lon,lat floats".into()));
                };
                let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
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
                let single_asset_expr = names.len() == 1 && rewritten_expr.is_some();
                let single_band_per_asset = (names.len() > 1 && rewritten_expr.is_some()) || pq.asset_as_band;
                let mut used_asset_bidx = false;
                for (name, pt) in names.into_iter().zip(ordered.into_iter()) {
                    if let Some(vs) = pt.as_ref().and_then(|p| p["values"].as_array()) {
                        if !single_asset_expr
                            && let Some(expr_str) = stac_asset_expr(&pq, &name)
                        {
                            let parts: Vec<&str> = expr_str
                                .split(';')
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                                .collect();
                            for expr in parts {
                                let prog =
                                    ExprProgram::compile(expr).map_err(EngineError::BadRequest)?;
                                let mut slot_vals = Vec::with_capacity(prog.used_bands.len());
                                for &b in &prog.used_bands {
                                    if b == 0 || (b as usize) > vs.len() {
                                        return Err(EngineError::BadRequest(format!(
                                            "expression references b{b} out of range for asset '{name}' (dataset has {} bands)",
                                            vs.len()
                                        )));
                                    }
                                    let f = vs[(b - 1) as usize]
                                        .as_f64()
                                        .map(|x| x as f32)
                                        .unwrap_or(f32::NAN);
                                    slot_vals.push(f);
                                }
                                let res = prog.eval(&slot_vals);
                                all_vals.push(if res.is_finite() {
                                    serde_json::json!(res)
                                } else {
                                    serde_json::Value::Null
                                });
                                all_names.push(format!("{name}_{expr}"));
                            }
                        } else if !single_asset_expr
                            && let Some(sel) = stac_asset_selected_bands(&pq, &name)
                        {
                            used_asset_bidx = true;
                            let iter_sel = if single_band_per_asset {
                                &sel[..sel.len().min(1)]
                            } else {
                                sel
                            };
                            for &b in iter_sel {
                                if b == 0 || (b as usize) > vs.len() {
                                    return Err(EngineError::BadRequest(format!(
                                        "band {b} out of range for asset '{name}' (dataset has {} bands)",
                                        vs.len()
                                    )));
                                }
                                all_vals.push(vs[(b as usize) - 1].clone());
                                all_names.push(if pq.asset_as_band {
                                    name.clone()
                                } else {
                                    format!("{name}_b{b}")
                                });
                            }
                        } else if single_band_per_asset {
                            if let Some(v) = vs.first() {
                                all_vals.push(v.clone());
                                all_names.push(if pq.asset_as_band {
                                    name.clone()
                                } else {
                                    format!("{name}_b1")
                                });
                            }
                        } else {
                            for (i, v) in vs.iter().enumerate() {
                                all_vals.push(v.clone());
                                all_names.push(format!("{name}_b{}", i + 1));
                            }
                        }
                    }
                }
                let band_names: Vec<String> =
                    (1..=all_vals.len()).map(|i| format!("b{i}")).collect();
                let mut pt_json = serde_json::json!({
                    "coordinates": [x, y],
                    "values": all_vals,
                    "band_names": band_names,
                    "band_descriptions": all_names,
                });
                let point_bidx: &[u16] = if used_asset_bidx && rewritten_expr.is_none() {
                    &[]
                } else {
                    &pq.bidx
                };
                apply_point_filters(&mut pt_json, rewritten_expr.as_deref(), point_bidx)?;
                Ok((Bytes::from(pt_json.to_string()), "application/json"))
            }
            .await,

            // GET /stac/preview[/{w}x{h}][.{ext}]
            (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
                let ext_opt = p.strip_prefix("preview.");
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs)?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    geotiff_render_format(fmt, ct, &pq_stac, Some(&cogs[0]), &bands, st)
                };
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], bbox, dst_bbox, pq_stac.max_size);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands,
                        rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else {
                    let scale = (pq_stac.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                    let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                        && full.pixel_to_crs.a > 0.0
                        && full.pixel_to_crs.e < 0.0;
                    if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                        let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands,
                            rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, bbox, out_w, out_h, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                        let epsg = nat_epsg.unwrap_or(4326);
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), nat_bbox, &crs, false, out_w, out_h, None, &bands,
                            pq_stac.resample, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, nat_bbox, out_w, out_h, epsg)
                    } else {
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, true, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                            pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, nat_bbox, out_w, out_h, nat_epsg.unwrap_or(4326))
                    }
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.jpeg_quality, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["preview", dims_ext]) => async {
                let Some((out_w, out_h, ext_opt)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected /stac/preview/{w}x{h}.{ext}".into()));
                };
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs)?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = geotiff_render_format(fmt, ct, &pq_stac, Some(&cogs[0]), &bands, st);
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands,
                        rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else {
                    let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                    let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                        && full.pixel_to_crs.a > 0.0
                        && full.pixel_to_crs.e < 0.0;
                    if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                        let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands,
                            rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, bbox, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                        let epsg = nat_epsg.unwrap_or(4326);
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), nat_bbox, &crs, false, out_w, out_h, None, &bands,
                            pq_stac.resample, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, nat_bbox, epsg)
                    } else {
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, true, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                            pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, nat_bbox, nat_epsg.unwrap_or(4326))
                    }
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,

            // GET /stac/bbox/{minx},{miny},{maxx},{maxy}[/{w}x{h}][.{ext}]
            (Method::GET, ["bbox", bbox_seg]) => async {
                let Some((bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs)?;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let (fmt, ct) = parse_ext_format(
                    ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png"),
                    pq_stac.webp_quality,
                    pq_stac.jpeg_quality,
                )?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    geotiff_render_format(fmt, ct, &pq_stac, Some(&cogs[0]), &bands, st)
                };
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let wgs_bbox_src = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                let bbox_max = effective_bbox_max_size(&pq_stac);
                let (data, out_bbox, out_w, out_h, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq_stac).filter(|(_, e)| *e != 4326 && *e != 3857) {
                    let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        reproject_wgs84_bounds(wgs_bbox_src, &dst_crs)?
                    };
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], wgs_bbox_src, dst_bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, &dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                    let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        reproject_wgs84_bounds(wgs_bbox_src, dst_crs)?
                    };
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], wgs_bbox_src, dst_bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else if pq_stac.coord_crs_3857 {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], wgs_bbox_src, bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 3857);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Epsg3857, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 3857u16)
                } else {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], wgs_bbox_src, wgs_bbox_src, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), wgs_bbox_src, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, wgs_bbox_src, out_w, out_h, 4326u16)
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, epsg, pq_stac.jpeg_quality, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(epsg), false));
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
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs)?;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = geotiff_render_format(fmt, ct, &pq_stac, Some(&cogs[0]), &bands, st);
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let (data, out_bbox, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq_stac).filter(|(_, e)| *e != 4326 && *e != 3857) {
                    let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        let wgs_bbox = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                        reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                    };
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, &dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                    let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        let wgs_bbox = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                        reproject_wgs84_bounds(wgs_bbox, dst_crs)?
                    };
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else if pq_stac.coord_crs_3857 {
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 3857);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Epsg3857, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, 3857u16)
                } else {
                    let wgs_bbox = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), wgs_bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, wgs_bbox, 4326u16)
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, epsg, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(epsg), false));
                Ok((data, ct))
            }
            .await,

            // POST /stac/feature/{w}x{h}[.{ext}] and POST /stac/feature[.{ext}]
            (Method::POST, ["feature", dims_ext]) => async {
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).ok_or_else(|| {
                    EngineError::BadRequest("expected /stac/feature/{w}x{h}.{ext}".into())
                })?;
                let item = stac_source.load(&engine).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs)?;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = geotiff_render_format(fmt, ct, &pq_stac, Some(&cogs[0]), &bands, st);
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let eff_dst = resolve_bbox_dst_crs(&pq_stac);
                let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), target_epsg);
                let mask_opt = pq_stac.shape_mask.then_some(&mask);
                let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else {
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, 4326u16)
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::POST, [feat_ext])
                if *feat_ext == "feature" || feat_ext.starts_with("feature.") =>
            async {
                let ext_opt = feat_ext.strip_prefix("feature.");
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let s = effective_bbox_max_size(&pq);
                let item = stac_source.load(&engine).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs)?;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    geotiff_render_format(fmt, ct, &pq_stac, Some(&cogs[0]), &bands, st)
                };
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac
                    .color_formula
                    .as_deref()
                    .map(ColorFormula::parse)
                    .transpose()
                    .map_err(EngineError::BadRequest)?;
                let eff_dst = resolve_bbox_dst_crs(&pq_stac);
                let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), target_epsg);
                let mask_opt = pq_stac.shape_mask.then_some(&mask);
                let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], bbox, dst_bbox, s);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], bbox, bbox, s);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 4326u16)
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.jpeg_quality, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,

            (_, ["statistics"]) => Err(EngineError::MethodNotAllowed("GET, POST, HEAD")),
            (_, [f]) if *f == "feature" || f.starts_with("feature.") => {
                Err(EngineError::MethodNotAllowed("POST"))
            }
            (_, ["feature", _]) => Err(EngineError::MethodNotAllowed("POST")),
            (_, ["info"] | ["info.geojson"] | ["bounds"] | ["assets"] | ["asset_statistics"] | ["tiles"] | ["tiles", ..] | ["point", _] | ["preview", ..] | ["bbox", ..] | ["map"] | ["viewer"] | ["renders"] | ["renders", _] | ["tilejson.json"] | [_, "tilejson.json"] | ["map.html"] | [_, "map.html"] | ["WMTSCapabilities.xml"] | [_, "WMTSCapabilities.xml"]) => {
                Err(EngineError::MethodNotAllowed("GET, HEAD"))
            }

            _ => Err(EngineError::NotFound(path.clone())),
        };
        // titiler-pgstac answers errors as JSON `{"detail": ...}`.
        if stac_prefix != "stac" {
            return Ok(match res {
                Err(e) => pgstac::engine_error(&e),
                Ok((body, ct)) => {
                    match pgstac::item_fixup(
                        &segs[1..],
                        pgstac_item_extent.as_ref(),
                        &raw_query,
                        body,
                        ct,
                    ) {
                        Ok(fixed) => {
                            let mut resp = to_http_response(&engine, Ok(fixed), if_none_match.as_deref());
                            attach_x_assets(&mut resp, &stac_x_assets);
                            let resp = if method == Method::GET {
                                pgstac::item_placed(resp, &segs[1..], &raw_query)
                            } else {
                                resp
                            };
                            pgstac_cacheable(resp, &method)
                        }
                        Err(resp) => resp,
                    }
                }
            });
        }
        let mut resp = to_http_response(&engine, res, if_none_match.as_deref());
        if !matches!(
            sub,
            ["assets"]
                | ["tiles"]
                | ["tiles", _]
                | [_, "map.html"]
                | ["map.html"]
                | ["viewer"]
                | ["renders"]
                | ["renders", _]
        ) {
            attach_resolved_stac_assets(&mut resp);
        }
        if let Some((bbox, epsg, is_crs84)) = geo_meta {
            attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
        }
        return Ok(resp);
    }

    // ------------------------------------------------------------------------
    // Full TiTiler Drop-In API: /cog/* (all 19 TiTiler TilerFactory routes!)
    // ------------------------------------------------------------------------
    if segs.first() == Some(&"cog") {
        let sub = &segs[1..];
        let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
        let res: Result<(Bytes, &'static str), EngineError> = match (method.clone(), sub) {
            // 1. GET /cog/info
            (Method::GET, ["info"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let mut json = build_cog_info(&engine, &cogs[0])?;
                if let Some((ref dst, dst_epsg)) = pq.bounds_crs {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let wgs = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let rep = reproject_wgs84_bounds(wgs, dst)?;
                    json["bounds"] = serde_json::json!(rep);
                    json["crs"] = serde_json::json!(format!("http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"));
                }
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 2. GET /cog/info.geojson
            (Method::GET, ["info.geojson"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let mut json = build_cog_info_geojson(&engine, &cogs[0])?;
                if let Some((ref dst, dst_epsg)) = pq.bounds_crs {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let wgs = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let rep = reproject_wgs84_bounds(wgs, dst)?;
                    json["properties"]["bounds"] = serde_json::json!(rep);
                    json["properties"]["crs"] = serde_json::json!(format!("http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"));
                }
                Ok((Bytes::from(json.to_string()), "application/geo+json"))
            }
            .await,

            // Alias: GET /cog/bounds
            (Method::GET, ["bounds"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bounds = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let bounds = if let Some((ref dst, _)) = pq.bounds_crs {
                    reproject_wgs84_bounds(bounds, dst)?
                } else {
                    bounds
                };
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
                let full = cogs[0].index.full_res();
                let db = full.samples_per_pixel;
                let is_dem_alg = pq.algorithm.as_deref().is_some_and(|a| {
                    matches!(
                        a.to_ascii_lowercase().as_str(),
                        "hillshade" | "slope" | "aspect" | "contours" | "terrarium" | "terrainrgb" | "terrain-rgb" | "mapbox"
                    )
                });
                if is_dem_alg {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let mut bands = build_band_request(&pq, db, full.sample_type)?;
                    if pq.unscale {
                        apply_unscale_cogs(&mut bands, &cogs);
                    }
                    let stats_opts = build_stats_options(&pq, false)?;
                    let raw = render_native_preview(
                        &engine,
                        &cogs,
                        bbox,
                        out_w,
                        out_h,
                        &bands,
                        pq.resample,
                        OutputFormat::RawF32,
                        pq.nodata,
                        None,
                        None,
                        true,
                        pq.unscale,
                    )
                    .await?;
                    let n_px = (out_w as usize) * (out_h as usize);
                    let nb = if n_px > 0 && raw.len() >= n_px * 4 {
                        (raw.len() / (n_px * 4)).clamp(1, 16)
                    } else {
                        1
                    };
                    let alg_name = pq.algorithm.as_deref().unwrap_or("algorithm");
                    let mut map = serde_json::Map::new();
                    for b in 0..nb {
                        let start = b * n_px * 4;
                        let end = (start + n_px * 4).min(raw.len());
                        let plane: Vec<f32> = raw[start..end]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let desc = if nb == 1 {
                            alg_name.to_string()
                        } else {
                            format!("{alg_name}_b{}", b + 1)
                        };
                        map.insert(
                            format!("b{}", b + 1),
                            compute_f32_band_stats(&plane, &desc, &stats_opts),
                        );
                    }
                    return Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"));
                }
                let stats_opts = build_stats_options_with_bands(&pq, true, db)?;
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
                let db = cogs[0].index.full_res().samples_per_pixel;
                let stats_opts = build_stats_options_with_bands(&pq, true, db)?;
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
                            let (mask, _) =
                                PolygonMask::from_geojson_in_crs(feat, pq.coord_crs.as_ref())?;
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
                        let (mask, _) =
                            PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
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
            (Method::GET, ["tiles"]) => async {
                let bounds = if let Ok(cogs) = open_query_urls(&engine, &pq.urls).await
                    && let Ok(crs) = engine.get_or_build_crs(&cogs[0])
                {
                    let full = cogs[0].index.full_res();
                    Some(crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height))
                } else {
                    None
                };
                let body = build_ogc_tilesets_list(
                    &host,
                    &forwarded_prefix,
                    "cog",
                    &raw_query,
                    bounds,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // 5. GET /cog/tiles/{tileMatrixSetId}
            (Method::GET, ["tiles", tms]) => async {
                let tms_def = crate::tms::lookup_tms(tms)
                    .ok_or_else(|| EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")))?;
                let (bounds, minz, maxz) = if let Ok(cogs) = open_query_urls(&engine, &pq.urls).await
                    && let Ok(crs) = engine.get_or_build_crs(&cogs[0])
                {
                    let full = cogs[0].index.full_res();
                    let coarsest = cogs[0].index.ifds.last().unwrap();
                    let ov = (full.width as f64 / coarsest.width.max(1) as f64)
                        .max(full.height as f64 / coarsest.height.max(1) as f64);
                    let (dmin, dmax) =
                        tms_def.zoom_range_for_cog(&crs, &full.pixel_to_crs, full.width, full.height, ov);
                    let (mn, mx) = resolve_zoom_range(dmin, dmax, pq.minzoom, pq.maxzoom);
                    (
                        Some(crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height)),
                        Some(mn),
                        Some(mx),
                    )
                } else {
                    (None, pq.minzoom, pq.maxzoom)
                };
                let body = build_ogc_tileset_json(
                    &host,
                    &forwarded_prefix,
                    "cog",
                    tms_def,
                    &raw_query,
                    bounds,
                    minz,
                    maxz,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

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
                        let (mut treq, ct) = build_tile_request_for_tms(
                            Some(&engine),
                            TileCoord { z, x, y },
                            sc,
                            ext,
                            &pq,
                            db,
                            st,
                            tms,
                            Some(&cogs[0]),
                        )?;
                        if pq.unscale {
                            apply_unscale_cogs(&mut treq.bands, &cogs);
                        }

                        let any_intersects = cogs.iter().any(|c| {
                            tile_intersects_cog(&engine, c, tms, TileCoord { z, x, y }, pq.buffer, treq.tile_size)
                                .unwrap_or(false)
                        });
                        if !any_intersects {
                            return Err(EngineError::NotFound(format!(
                                "Tile {z}/{x}/{y} is outside image bounds"
                            )));
                        }
                        let needs_general_path = tms != "WebMercatorQuad"
                            || pq.buffer > 0.0
                            || pq.nodata.is_some()
                            || pq.custom_colormap_json.is_some()
                            || (!pq.return_mask && matches!(treq.format, OutputFormat::Npy))
                            || (pq.color_formula.is_some()
                                && !matches!(
                                    treq.bands,
                                    BandRequest::Rgb { .. }
                                        | BandRequest::SarRgb { .. }
                                        | BandRequest::CustomExprRgb { .. }
                                        | BandRequest::Terrain { .. }
                                ))
                            || matches!(treq.bands, BandRequest::CustomExprRgb { .. } | BandRequest::MultiBand { .. })
                            || (treq.tile_size != 256 && treq.tile_size != 512)
                            || has_embedded_palette(&pq, &cogs[0], &treq.bands)
                            || !crate::engine::cogs_share_base_grid(&cogs);

                        let auto_fmt = ext.is_none() && pq.format_param.is_none();
                        if auto_fmt {
                            treq.format = OutputFormat::RawRgba;
                        }

                        if needs_general_path {
                            let custom_lut = resolve_custom_lut(&pq, &cogs[0], &treq.bands)?;
                            let cf = pq
                                .color_formula
                                .as_deref()
                                .map(ColorFormula::parse)
                                .transpose()
                                .map_err(EngineError::BadRequest)?;
                            let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                            let (data, bbox, epsg) = if tms != "WebMercatorQuad" {
                                let tms_def = crate::tms::lookup_tms(tms).ok_or_else(|| {
                                    EngineError::BadRequest(format!("unsupported TMS '{tms}'"))
                                })?;
                                let bbox = tms_def
                                    .tile_bounds(z, x, y, pq.buffer, treq.tile_size)
                                    .ok_or_else(|| {
                                        EngineError::NotFound(format!(
                                            "Tile {z}/{x}/{y} is outside {tms} matrix"
                                        ))
                                    })?;
                                let dst_crs = tms_def.crs_model()?;
                                let d = render_crs_bbox(
                                    &engine,
                                    &cogs,
                                    bbox,
                                    &dst_crs,
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
                                .await?;
                                (d, bbox, tms_def.epsg)
                            } else {
                                let bbox_3857 = web_mercator_buffered_3857_bbox(
                                    z,
                                    x,
                                    y,
                                    pq.buffer,
                                    treq.tile_size,
                                );
                                let d = render_mercator_bbox(
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
                                (d, bbox_3857, 3857u16)
                            };
                            Ok(finalize_image_output(data, ct, treq.format, auto_fmt, out_dim, out_dim, bbox, epsg, pq.jpeg_quality, pq.return_mask))
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
                            let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                            if auto_fmt {
                                let buf = if let Some(cf) = cf {
                                    let mut b = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut b, 4);
                                    Bytes::from(b)
                                } else {
                                    raw
                                };
                                Ok(encode_auto_image(dim, dim, &buf, pq.jpeg_quality, pq.return_mask))
                            } else {
                                let data = if let Some(cf) = cf {
                                    let mut buf = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut buf, 4);
                                    encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
                                } else {
                                    encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
                                };
                                let data = wrap_geotiff_fmt(data, ct, orig_format, treq.tile_size, treq.tile_size, bbox, 3857, pq.return_mask);
                                Ok((data, ct))
                            }
                        } else {
                            let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                            Ok(finalize_image_output(data, ct, treq.format, auto_fmt, treq.tile_size, treq.tile_size, bbox, 3857, pq.jpeg_quality, pq.return_mask))
                        }
                    })
                    .await;
                let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                if let (Ok(z), Ok(x), Some((y, sc, _))) =
                    (zs.parse::<u8>(), xs.parse::<u32>(), parse_y_scale_ext(ys))
                {
                    attach_tile_geo_headers(&mut resp, tms, z, x, y, sc, &pq);
                }
                return Ok(resp);
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
            (Method::GET, [_, "map.html"]) | (Method::GET, ["map.html"]) | (Method::GET, ["viewer"]) => {
                let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                let tj_url = format!("{forwarded_prefix}/cog/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
            // GET /cog/stac — build a STAC 1.1.0 Item JSON from a COG (matching TiTiler / rio-stac)
            (Method::GET, ["stac"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let cog = &cogs[0];
                let (crs, native_epsg, native_bounds) =
                    dataset_native_crs_and_bounds(&engine, cog)?;
                let full = cog.index.full_res();
                let wgs_bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let [w, s, e, n] = wgs_bbox;
                let [nw, ns, ne, nn] = native_bounds;
                let mut q_id: Option<String> = None;
                let mut q_collection: Option<String> = None;
                let mut q_collection_url: Option<String> = None;
                let mut q_asset_name: Option<String> = None;
                let mut q_asset_href: Option<String> = None;
                let mut q_asset_media_type: Option<String> = None;
                let mut q_asset_roles: Vec<String> = Vec::new();
                let mut q_extensions: Vec<String> = Vec::new();
                let mut q_datetime: Option<String> = None;
                let mut with_proj = true;
                let mut with_raster = true;
                let mut with_eo = true;
                for pair in raw_query.split('&').filter(|p| !p.is_empty()) {
                    if let Some((k, v)) = pair.split_once('=') {
                        let dec = percent_decode_opts(v, false);
                        match k {
                            "id" if !dec.is_empty() => q_id = Some(dec),
                            "collection" if !dec.is_empty() => q_collection = Some(dec),
                            "collection_url" if !dec.is_empty() => q_collection_url = Some(dec),
                            "asset_name" if !dec.is_empty() => q_asset_name = Some(dec),
                            "asset_href" if !dec.is_empty() => q_asset_href = Some(dec),
                            "asset_media_type" if !dec.is_empty() => q_asset_media_type = Some(dec),
                            "asset_roles" => {
                                for r in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                                    q_asset_roles.push(r.to_string());
                                }
                            }
                            "extensions" => {
                                for ext in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                                    q_extensions.push(ext.to_string());
                                }
                            }
                            "datetime" if !dec.is_empty() => q_datetime = Some(dec),
                            "with_proj" => with_proj = !dec.eq_ignore_ascii_case("false") && dec != "0",
                            "with_raster" => with_raster = !dec.eq_ignore_ascii_case("false") && dec != "0",
                            "with_eo" => with_eo = !dec.eq_ignore_ascii_case("false") && dec != "0",
                            _ => {}
                        }
                    }
                }
                let item_id = q_id.unwrap_or_else(|| {
                    std::path::Path::new(&cog.uri)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("cog-item")
                        .to_string()
                });
                let asset_name = q_asset_name.unwrap_or_else(|| "data".to_string());
                let asset_href = q_asset_href.unwrap_or_else(|| cog.uri.clone());
                let media_type = match q_asset_media_type.as_deref() {
                    None | Some("auto") => "image/tiff; application=geotiff".to_string(),
                    Some("cog") => "image/tiff; application=geotiff; profile=cloud-optimized".to_string(),
                    Some(other) => other.to_string(),
                };
                let mut extensions: Vec<String> = Vec::new();
                let mut push_ext = |uri: &str| {
                    if !extensions.iter().any(|e| e == uri) {
                        extensions.push(uri.to_string());
                    }
                };
                let mut props = serde_json::Map::new();
                props.insert(
                    "datetime".into(),
                    serde_json::json!(q_datetime.unwrap_or_else(utc_now_rfc3339)),
                );
                if with_proj {
                    push_ext("https://stac-extensions.github.io/projection/v2.0.0/schema.json");
                    props.insert("proj:epsg".into(), serde_json::json!(native_epsg));
                    props.insert("proj:shape".into(), serde_json::json!([full.height, full.width]));
                    props.insert(
                        "proj:transform".into(),
                        serde_json::json!([
                            full.pixel_to_crs.a,
                            full.pixel_to_crs.b,
                            full.pixel_to_crs.c,
                            full.pixel_to_crs.d,
                            full.pixel_to_crs.e,
                            full.pixel_to_crs.f,
                            0.0,
                            0.0,
                            1.0
                        ]),
                    );
                    props.insert("proj:bbox".into(), serde_json::json!(native_bounds));
                    props.insert(
                        "proj:geometry".into(),
                        serde_json::json!({
                            "type": "Polygon",
                            "coordinates": [[[nw, ns], [ne, ns], [ne, nn], [nw, nn], [nw, ns]]]
                        }),
                    );
                }
                let count = full.samples_per_pixel as usize;
                let mut asset_obj = serde_json::json!({
                    "href": asset_href,
                    "type": media_type,
                    "roles": q_asset_roles,
                });
                if with_raster {
                    push_ext("https://stac-extensions.github.io/raster/v1.1.0/schema.json");
                    let sampling = if cog.index.geo_keys.raster_type == Some(2) {
                        "point"
                    } else {
                        "area"
                    };
                    let stats_json = compute_statistics_with_options(
                        &engine,
                        cog,
                        128,
                        None,
                        false,
                        None,
                        &StatsOptions::default(),
                    )
                    .await
                    .ok();
                    let rb: Vec<serde_json::Value> = (0..count)
                        .map(|i| {
                            let mut b = serde_json::Map::new();
                            b.insert("data_type".into(), serde_json::json!(full.sample_type.gdal_name()));
                            b.insert("scale".into(), serde_json::json!(cog.index.scales_f64.get(i).copied().unwrap_or(1.0)));
                            b.insert("offset".into(), serde_json::json!(cog.index.offsets_f64.get(i).copied().unwrap_or(0.0)));
                            b.insert("sampling".into(), serde_json::json!(sampling));
                            if let Some(nd) = cog.index.nodata {
                                if nd.is_nan() {
                                    b.insert("nodata".into(), serde_json::json!("nan"));
                                } else {
                                    b.insert("nodata".into(), serde_json::json!(nd));
                                }
                            }
                            if let Some(bs) = stats_json.as_ref().and_then(|s| s.get(format!("b{}", i + 1))) {
                                b.insert(
                                    "statistics".into(),
                                    serde_json::json!({
                                        "mean": bs["mean"],
                                        "minimum": bs["min"],
                                        "maximum": bs["max"],
                                        "stddev": bs["std"],
                                        "valid_percent": bs["valid_percent"],
                                    }),
                                );
                                if let Some(h) = bs.get("histogram").and_then(|h| h.as_array())
                                    && h.len() == 2
                                    && let (Some(counts), Some(edges)) = (h[0].as_array(), h[1].as_array())
                                    && edges.len() >= 2
                                {
                                    b.insert(
                                        "histogram".into(),
                                        serde_json::json!({
                                            "count": counts.len() + 1,
                                            "min": edges.first(),
                                            "max": edges.last(),
                                            "buckets": counts,
                                        }),
                                    );
                                }
                            }
                            serde_json::Value::Object(b)
                        })
                        .collect();
                    asset_obj["raster:bands"] = serde_json::Value::Array(rb);
                }
                if with_eo {
                    push_ext("https://stac-extensions.github.io/eo/v1.1.0/schema.json");
                    let eb: Vec<serde_json::Value> = (0..count)
                        .map(|i| {
                            let name = format!("b{}", i + 1);
                            let desc = cog
                                .index
                                .band_descriptions
                                .get(i)
                                .filter(|s| !s.is_empty())
                                .cloned()
                                .unwrap_or_else(|| name.clone());
                            serde_json::json!({ "name": name, "description": desc })
                        })
                        .collect();
                    asset_obj["eo:bands"] = serde_json::Value::Array(eb);
                }
                for ext in &q_extensions {
                    push_ext(ext);
                }
                let mut links: Vec<serde_json::Value> = Vec::new();
                if let Some(col_url) = q_collection_url {
                    links.push(serde_json::json!({
                        "rel": "collection",
                        "type": "application/json",
                        "href": col_url,
                    }));
                }
                let mut assets_map = serde_json::Map::new();
                assets_map.insert(asset_name, asset_obj);
                let mut item = serde_json::json!({
                    "type": "Feature",
                    "stac_version": "1.1.0",
                    "stac_extensions": extensions,
                    "id": item_id,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
                    },
                    "bbox": wgs_bbox,
                    "properties": serde_json::Value::Object(props),
                    "links": links,
                    "assets": serde_json::Value::Object(assets_map),
                });
                if let Some(col) = q_collection {
                    item["collection"] = serde_json::json!(col);
                }
                Ok((Bytes::from(item.to_string()), "application/geo+json"))
            }
            .await,
            // GET /cog/map (OGC API - Maps)
            (Method::GET, ["map"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let (native_crs, native_epsg, native_bounds) =
                    dataset_native_crs_and_bounds(&engine, &cogs[0])?;
                let full = cogs[0].index.full_res();
                let explicit_crs = pq.dst_crs.clone().or_else(|| pq.bounds_crs.clone());
                let (dst_crs, dst_epsg) = explicit_crs
                    .clone()
                    .unwrap_or((native_crs.clone(), native_epsg));
                let (wgs_bbox, out_bbox) = if let Some(raw_bbox) = pq.bbox_param {
                    let wb = bbox_to_wgs84(
                        raw_bbox,
                        pq.bbox_crs.as_ref().map(|(c, _)| c),
                    )?;
                    let ob = if pq.bbox_crs.as_ref().map(|(_, e)| *e) == Some(dst_epsg)
                        || (pq.bbox_crs.is_none() && dst_epsg == 4326)
                    {
                        normalize_render_bbox(raw_bbox, &dst_crs)?
                    } else {
                        normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                    };
                    (wb, ob)
                } else {
                    let wb = native_crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let ob = if explicit_crs.is_some() {
                        normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                    } else {
                        native_bounds
                    };
                    (wb, ob)
                };
                let (def_w, def_h) = if dst_epsg == native_epsg {
                    let px_x = full.pixel_to_crs.a.hypot(full.pixel_to_crs.d).max(1e-12);
                    let px_y = full.pixel_to_crs.b.hypot(full.pixel_to_crs.e).max(1e-12);
                    let nat_w = ((out_bbox[2] - out_bbox[0]).abs() / px_x)
                        .round()
                        .clamp(1.0, 4096.0) as u32;
                    let nat_h = ((out_bbox[3] - out_bbox[1]).abs() / px_y)
                        .round()
                        .clamp(1.0, 4096.0) as u32;
                    let cap = pq.max_size.clamp(1, 4096);
                    if nat_w.max(nat_h) <= cap {
                        (nat_w, nat_h)
                    } else if nat_w >= nat_h {
                        (
                            cap,
                            ((nat_h as f64 / nat_w as f64) * cap as f64)
                                .ceil()
                                .clamp(1.0, cap as f64) as u32,
                        )
                    } else {
                        (
                            ((nat_w as f64 / nat_h as f64) * cap as f64)
                                .ceil()
                                .clamp(1.0, cap as f64) as u32,
                            cap,
                        )
                    }
                } else {
                    capped_native_bbox_dims(
                        &engine,
                        &cogs[0],
                        wgs_bbox,
                        out_bbox,
                        pq.max_size,
                    )
                };
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = pq.format_param.as_deref().unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let mut bands = build_band_request_for_output(&pq, db, full.sample_type, fmt, ct)?;
                let fmt = geotiff_render_format(fmt, ct, &pq, Some(&cogs[0]), &bands, full.sample_type);
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                let data = render_crs_bbox(
                    &engine, &cogs, out_bbox, &dst_crs, out_w, out_h, None, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, dst_epsg, pq.return_mask);
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::GET, [_, "WMTSCapabilities.xml"]) | (Method::GET, ["WMTSCapabilities.xml"]) => {
                async {
                    let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                    if crate::tms::lookup_tms(tms).is_none() {
                        return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                    }
                    let cogs = open_query_urls(&engine, &pq.urls).await?;
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let bounds = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let coarsest = cogs[0].index.ifds.last().unwrap();
                    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
                        .max(full.height as f64 / coarsest.height.max(1) as f64);
                    let (def_min, def_max) = if let Some(tms_def) = crate::tms::lookup_tms(tms) {
                        tms_def.zoom_range_for_cog(&crs, &full.pixel_to_crs, full.width, full.height, ov_factor)
                    } else {
                        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
                    };
                    let (minz, maxz) =
                        resolve_zoom_range(def_min, def_max, pq.minzoom, pq.maxzoom);
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
                let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
                let mut json =
                    query_point(&engine, &cogs[0], lon, lat, pq.unscale, pq.nodata).await?;
                if pq.coord_crs.is_some() {
                    json["coordinates"] = serde_json::json!([x, y]);
                }
                apply_point_filters(&mut json, pq.expr_raw.as_deref(), &pq.bidx)?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 11. GET /cog/preview, /cog/preview.{format}, /cog/preview/{w}x{h}.{format}
            (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
                let ext_opt = p.strip_prefix("preview.");
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let ext = ext_opt
                    .or(pq.format_param.as_deref())
                    .unwrap_or("png");
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    geotiff_render_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st)
                };
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], bbox, dst_bbox, pq.max_size);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands,
                        rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else {
                    let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                    let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                        && full.pixel_to_crs.a > 0.0
                        && full.pixel_to_crs.e < 0.0;
                    if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                        let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                        let d = render_wgs84_bbox(
                            &engine, &cogs, bbox, out_w, out_h, None, &bands,
                            rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, bbox, out_w, out_h, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                        let epsg = nat_epsg.unwrap_or(4326);
                        let d = render_crs_bbox(
                            &engine, &cogs, nat_bbox, &crs, out_w, out_h, None, &bands,
                            pq.resample, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, nat_bbox, out_w, out_h, epsg)
                    } else {
                        let d = render_native_preview(
                            &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                            pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, nat_bbox, out_w, out_h, nat_epsg.unwrap_or(4326))
                    }
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq.jpeg_quality, pq.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
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
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = geotiff_render_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st);
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands,
                        rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else {
                    let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                    let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                        && full.pixel_to_crs.a > 0.0
                        && full.pixel_to_crs.e < 0.0;
                    if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                        let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                        let d = render_wgs84_bbox(
                            &engine, &cogs, bbox, out_w, out_h, None, &bands,
                            rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, bbox, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                        let epsg = nat_epsg.unwrap_or(4326);
                        let d = render_crs_bbox(
                            &engine, &cogs, nat_bbox, &crs, out_w, out_h, None, &bands,
                            pq.resample, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, nat_bbox, epsg)
                    } else {
                        let d = render_native_preview(
                            &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                            pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, nat_bbox, nat_epsg.unwrap_or(4326))
                    }
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,

            // 12. GET /cog/bbox/{minx},{miny},{maxx},{maxy}[.{format}] & /cog/bbox/{...}/{w}x{h}.{format}
            (Method::GET, ["bbox", bbox_seg]) => async {
                let Some((bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("expected /cog/bbox/{minx},{miny},{maxx},{maxy}.{ext}".into()));
                };
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let (fmt, ct) = parse_ext_format(
                    ext_opt.or(pq.format_param.as_deref()).unwrap_or("png"),
                    pq.webp_quality,
                    pq.jpeg_quality,
                )?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    geotiff_render_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st)
                };
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let wgs_bbox_src = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                let bbox_max = effective_bbox_max_size(&pq);
                let (data, out_bbox, out_w, out_h, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq).filter(|(_, e)| *e != 4326 && *e != 3857) {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        reproject_wgs84_bounds(wgs_bbox_src, &dst_crs)?
                    };
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], wgs_bbox_src, dst_bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, &dst_crs, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        reproject_wgs84_bounds(wgs_bbox_src, dst_crs)?
                    };
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], wgs_bbox_src, dst_bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else if pq.coord_crs_3857 {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], wgs_bbox_src, bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 3857);
                    let d = render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 3857u16)
                } else {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], wgs_bbox_src, wgs_bbox_src, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                    let d = render_wgs84_bbox(
                        &engine, &cogs, wgs_bbox_src, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, wgs_bbox_src, out_w, out_h, 4326u16)
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, epsg, pq.jpeg_quality, pq.return_mask);
                geo_meta = Some((out_bbox, Some(epsg), false));
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
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = geotiff_render_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st);
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let (data, out_bbox, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq).filter(|(_, e)| *e != 4326 && *e != 3857) {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                        reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                    };
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, &dst_crs, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                        reproject_wgs84_bounds(wgs_bbox, dst_crs)?
                    };
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else if pq.coord_crs_3857 {
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 3857);
                    let d = render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, 3857u16)
                } else {
                    let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                    let d = render_wgs84_bbox(
                        &engine, &cogs, wgs_bbox, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, wgs_bbox, 4326u16)
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, epsg, pq.return_mask);
                geo_meta = Some((out_bbox, Some(epsg), false));
                Ok((data, ct))
            }
            .await,

            // 13. POST /cog/feature, /cog/feature.{format}, /cog/feature/{w}x{h}.{format}
            (Method::POST, [f]) if *f == "feature" || f.starts_with("feature.") => async {
                let ext_opt = f.strip_prefix("feature.");
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let ext = ext_opt
                    .or(pq.format_param.as_deref())
                    .unwrap_or("png");
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    geotiff_render_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st)
                };
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let eff_dst = resolve_bbox_dst_crs(&pq);
                let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), target_epsg);
                let mask_opt = pq.shape_mask.then_some(&mask);
                let feat_max = effective_bbox_max_size(&pq);
                let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], bbox, dst_bbox, feat_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], bbox, bbox, feat_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let d = render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 4326u16)
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq.jpeg_quality, pq.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
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
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = cogs[0].index.full_res().sample_type;
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = geotiff_render_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st);
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let eff_dst = resolve_bbox_dst_crs(&pq);
                let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), target_epsg);
                let mask_opt = pq.shape_mask.then_some(&mask);
                let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else {
                    let d = render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, 4326u16)
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,

            (_, ["statistics"]) => Err(EngineError::MethodNotAllowed("GET, POST, HEAD")),
            (_, [f]) if *f == "feature" || f.starts_with("feature.") => {
                Err(EngineError::MethodNotAllowed("POST"))
            }
            (_, ["feature", _]) => Err(EngineError::MethodNotAllowed("POST")),
            (_, ["info"] | ["info.geojson"] | ["bounds"] | ["validate"] | ["stac"] | ["tiles"] | ["tiles", ..] | ["point", _] | ["preview", ..] | ["bbox", ..] | ["map"] | ["viewer"] | ["tilejson.json"] | [_, "tilejson.json"] | ["map.html"] | [_, "map.html"] | ["WMTSCapabilities.xml"] | [_, "WMTSCapabilities.xml"]) => {
                Err(EngineError::MethodNotAllowed("GET, HEAD"))
            }

            _ => Err(EngineError::NotFound(path)),
        };
        let mut resp = to_http_response(&engine, res, if_none_match.as_deref());
        if let Some((bbox, epsg, is_crs84)) = geo_meta {
            attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
        }
        return Ok(resp);
    }

    // 405 Method Not Allowed when a known route is requested with an unsupported HTTP verb
    if matches!(
        path.as_str(),
        "/" | "/index.html"
            | "/healthz"
            | "/metrics"
            | "/conformance"
            | "/conformance/"
            | "/openapi.json"
            | "/api"
            | "/docs"
            | "/api.html"
    ) || matches!(
        segs.first().copied(),
        Some("algorithms" | "colorMaps" | "tileMatrixSets" | "mosaicjson")
    ) {
        let allow = if matches!(segs.as_slice(), ["mosaicjson", "validate" | "statistics" | "feature", ..]) {
            "POST"
        } else {
            "GET, HEAD"
        };
        return Ok(method_not_allowed_resp(allow));
    }

    Ok(json_error_resp(StatusCode::NOT_FOUND, &path))
}

/// A pgSTAC route's answer with the `Cache-Control` the tiles carry, as titiler-pgstac
/// puts one on what it serves: on what a `GET` found, never on an error, and not where
/// one is already set.
fn pgstac_cacheable(mut resp: Response<Full<Bytes>>, method: &Method) -> Response<Full<Bytes>> {
    if *method == Method::GET && resp.status().is_success() && !resp.headers().contains_key(CACHE_CONTROL) {
        resp.headers_mut().insert(CACHE_CONTROL, cache_control_header().clone());
    }
    resp
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

fn attach_geo_headers(
    resp: &mut Response<Full<Bytes>>,
    bbox: [f64; 4],
    epsg: Option<u16>,
    is_crs84: bool,
) {
    if resp.status() != StatusCode::OK {
        return;
    }
    if let Ok(hv) = HeaderValue::from_str(&format!(
        "{},{},{},{}",
        bbox[0], bbox[1], bbox[2], bbox[3]
    )) {
        resp.headers_mut()
            .insert(HeaderName::from_static("content-bbox"), hv);
    }
    let crs_hv = if is_crs84 {
        Some(HeaderValue::from_static(
            "<http://www.opengis.net/def/crs/OGC/0/CRS84>",
        ))
    } else if let Some(code) = epsg {
        match code {
            3857 => Some(HeaderValue::from_static(
                "<http://www.opengis.net/def/crs/EPSG/0/3857>",
            )),
            4326 => Some(HeaderValue::from_static(
                "<http://www.opengis.net/def/crs/EPSG/0/4326>",
            )),
            other => HeaderValue::from_str(&format!(
                "<http://www.opengis.net/def/crs/EPSG/0/{other}>"
            ))
            .ok(),
        }
    } else {
        None
    };
    if let Some(hv) = crs_hv {
        resp.headers_mut()
            .insert(HeaderName::from_static("content-crs"), hv);
    }
}

#[derive(Clone, Copy)]
struct TileGeoOpts {
    buffer: f32,
    tile_size: u32,
    has_explicit_ts: bool,
}

impl TileGeoOpts {
    fn from_pq(pq: &ParsedQuery) -> Self {
        let has_explicit_ts = pq
            .keys
            .iter()
            .any(|k| matches!(k.as_str(), "tile_size" | "tilesize" | "size" | "scale"));
        Self {
            buffer: pq.buffer,
            tile_size: pq.tile_size,
            has_explicit_ts,
        }
    }
}

fn attach_tile_geo_headers(
    resp: &mut Response<Full<Bytes>>,
    tms: &str,
    z: u8,
    x: u32,
    y: u32,
    scale_opt: Option<u32>,
    pq: &ParsedQuery,
) {
    attach_tile_geo_headers_opts(resp, tms, z, x, y, scale_opt, TileGeoOpts::from_pq(pq));
}

fn attach_tile_geo_headers_opts(
    resp: &mut Response<Full<Bytes>>,
    tms: &str,
    z: u8,
    x: u32,
    y: u32,
    scale_opt: Option<u32>,
    opts: TileGeoOpts,
) {
    if resp.status() != StatusCode::OK {
        return;
    }
    if tms == "WebMercatorQuad" {
        let ts = scale_opt
            .map(|sc| sc.saturating_mul(256))
            .unwrap_or(if opts.has_explicit_ts { opts.tile_size } else { 256 });
        let bbox = web_mercator_buffered_3857_bbox(z, x, y, opts.buffer, ts);
        attach_geo_headers(resp, bbox, Some(3857), false);
    } else if let Some(tms_def) = crate::tms::lookup_tms(tms)
        && let Some(m) = tms_def.matrix(z)
    {
        let base_ts = m.tile_width.clamp(64, 4096);
        let ts = scale_opt
            .map(|sc| sc.saturating_mul(base_ts))
            .unwrap_or(if opts.has_explicit_ts { opts.tile_size } else { base_ts });
        if let Some(bbox) = tms_def.tile_bounds(z, x, y, opts.buffer, ts) {
            let is_crs84 = matches!(tms, "WorldCRS84Quad" | "GNOSISGlobalGrid" | "WGS1984Quad");
            attach_geo_headers(resp, bbox, Some(tms_def.epsg), is_crs84);
        }
    }
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
                || content_type == "application/x-binary";
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

fn json_error_resp(status: StatusCode, msg: &str) -> Response<Full<Bytes>> {
    let body = serde_json::json!({ "detail": msg }).to_string();
    plain_resp(status, Bytes::from(body), "application/json")
}

fn error_to_http_response(err: &EngineError) -> Response<Full<Bytes>> {
    match err {
        EngineError::Shared(inner) => error_to_http_response(inner),
        EngineError::NoContent => Response::builder()
            .status(StatusCode::NO_CONTENT)
            .header("access-control-allow-origin", "*")
            .body(Full::new(Bytes::new()))
            .unwrap(),
        EngineError::MethodNotAllowed(allow) => method_not_allowed_resp(allow),
        EngineError::NotFound(msg) => json_error_resp(StatusCode::NOT_FOUND, msg),
        EngineError::BadRequest(msg) => json_error_resp(StatusCode::BAD_REQUEST, msg),
        // The file is in a coordinate system this server cannot place. That is a
        // property of the requested source, not a server fault.
        EngineError::Proj(e) => json_error_resp(StatusCode::BAD_REQUEST, &e.to_string()),
        EngineError::Io(io_e) => match io_e.root() {
            e @ cogzilla_io::IoError::InvalidUri(_) => {
                json_error_resp(StatusCode::BAD_REQUEST, &e.to_string())
            }
            e @ cogzilla_io::IoError::Tiff(_) => {
                json_error_resp(StatusCode::BAD_REQUEST, &e.to_string())
            }
            e @ cogzilla_io::IoError::Forbidden(_) => {
                json_error_resp(StatusCode::FORBIDDEN, &e.to_string())
            }
            e @ cogzilla_io::IoError::HttpStatus { status: 404, .. } => {
                json_error_resp(StatusCode::NOT_FOUND, &e.to_string())
            }
            e @ cogzilla_io::IoError::Fs(io_err)
                if io_err.kind() == std::io::ErrorKind::NotFound =>
            {
                json_error_resp(StatusCode::NOT_FOUND, &e.to_string())
            }
            e => {
                tracing::warn!(error = %e, "request failed with 500");
                json_error_resp(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string())
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
        assert_eq!(list.len(), 19);
        assert_eq!(list[0]["id"], "hillshade");
        assert_eq!(get_algorithm_json("contours").unwrap()["title"], "Contours");
        assert_eq!(
            list[0]["links"][0]["href"],
            "http://localhost:8080/algorithms/hillshade"
        );
        let hs = get_algorithm_json("hillshade").unwrap();
        assert_eq!(hs["title"], "Hillshade");
        assert_eq!(hs["inputs"]["nbands"], 1);
        assert_eq!(hs["outputs"]["dtype"], "uint8");
        assert_eq!(get_algorithm_json("normalizedIndex").unwrap()["title"], "Normalized Difference Index");
        assert_eq!(get_algorithm_json("grayscale").unwrap()["title"], "Transform a RGB Image to Grayscale");
        assert_eq!(get_algorithm_json("bitonal").unwrap()["title"], "Transform an Image to Bitonal");
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
        let vpng = get_colormap_image("viridis", OutputFormat::Png, true, None, None).unwrap();
        let sized = get_colormap_image("viridis", OutputFormat::Png, false, Some(128), Some(16)).unwrap();
        assert_eq!(&sized[16..20], &128u32.to_be_bytes());
        assert_eq!(&sized[20..24], &16u32.to_be_bytes());
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
        assert!(validate_query(&["mosaicjson", "WorldCRS84Quad", "WMTSCapabilities.xml"], &pq_wmts).is_ok());
        assert!(validate_query(&["mosaicjson", "NotATms", "WMTSCapabilities.xml"], &pq_wmts).is_err());
    }

    #[test]
    fn geospatial_bounds_reproject_and_user_defined_crs_parity() {
        // 1. reproject_wgs84_bounds on global COG into UTM31 (EPSG:32631), NZTM (EPSG:2193),
        //    and LAEA (EPSG:3035) must include interior points (Rome/Berlin/NZ/Europe), not just edges.
        let global = [-180.0, -85.0, 180.0, 85.0];
        let utm31 = cogzilla_proj::CrsModel::from_epsg(32631).unwrap();
        let utm_b = reproject_wgs84_bounds(global, &utm31).unwrap();
        let (rome_x, rome_y) = utm31.from_lonlat_deg(12.5, 41.9);
        assert!(
            rome_x >= utm_b[0] && rome_x <= utm_b[2] && rome_y >= utm_b[1] && rome_y <= utm_b[3],
            "Rome ({rome_x}, {rome_y}) outside projected UTM31 bounds {utm_b:?}"
        );

        let nztm = cogzilla_proj::CrsModel::from_epsg(2193).unwrap();
        let nztm_b = reproject_wgs84_bounds(global, &nztm).unwrap();
        let (nz_x, nz_y) = nztm.from_lonlat_deg(170.5, -43.5);
        assert!(
            nz_x >= nztm_b[0] && nz_x <= nztm_b[2] && nz_y >= nztm_b[1] && nz_y <= nztm_b[3],
            "NZ ({nz_x}, {nz_y}) outside projected NZTM bounds {nztm_b:?}"
        );

        let laea = cogzilla_proj::CrsModel::from_epsg(3035).unwrap();
        let laea_b = reproject_wgs84_bounds([-180.0, -90.0, 180.0, 90.0], &laea).unwrap();
        let (ber_x, ber_y) = laea.from_lonlat_deg(13.4, 52.5);
        assert!(
            ber_x >= laea_b[0] && ber_x <= laea_b[2] && ber_y >= laea_b[1] && ber_y <= laea_b[3],
            "Berlin ({ber_x}, {ber_y}) outside projected LAEA bounds {laea_b:?}"
        );

        // 2. cog_standard_epsg returns None for user-defined projected CRS (32767) even with geographic_type=4326
        let mut gk = cogzilla_tiff::GeoKeys::default();
        gk.model_type = Some(1);
        gk.projected_cs_type = Some(32767);
        gk.geographic_type = Some(4326);
        assert_eq!(cog_standard_epsg(&gk), None);

        // Standard projected EPSG returns Some(32610)
        gk.projected_cs_type = Some(32610);
        assert_eq!(cog_standard_epsg(&gk), Some(32610));

        // Pure geographic returns Some(4326)
        let mut gk_geo = cogzilla_tiff::GeoKeys::default();
        gk_geo.model_type = Some(2);
        gk_geo.geographic_type = Some(4326);
        assert_eq!(cog_standard_epsg(&gk_geo), Some(4326));

        // 3. GEOTIFF_CONTENT_TYPE matches parse_ext_format("tif", ...)
        let (_, tif_ct) = parse_ext_format("tif", None, 85).unwrap();
        assert_eq!(tif_ct, GEOTIFF_CONTENT_TYPE);

        // 4. PolygonMask::check_work_budget rejects O(W*H*V) DoS payloads
        let ring: Vec<serde_json::Value> = (0..10_000)
            .map(|i| {
                let a = (i as f64) * std::f64::consts::TAU / 10_000.0;
                serde_json::json!([a.cos(), a.sin()])
            })
            .collect();
        let poly_json = serde_json::json!({
            "type": "Polygon",
            "coordinates": [ring],
        });
        let (pm, _) = PolygonMask::from_geojson(&poly_json).unwrap();
        assert!(pm.check_work_budget(256 * 256).is_err());
        assert!(pm.check_work_budget(64 * 64).is_ok());

        // 5. ?assets=visual|bidx=2&assets=nir stores per-asset bidx without leaking into global bidx
        let pq_ab = parse_query(Some("assets=visual|bidx=2&assets=nir&asset_as_band=true"));
        assert_eq!(pq_ab.assets, vec!["visual", "nir"]);
        assert!(pq_ab.bidx.is_empty());
        assert_eq!(pq_ab.asset_bidx, vec![("visual".to_string(), vec![2])]);
        assert!(pq_ab.asset_as_band);

        // 6. CrsModel::linear_unit_metres returns US survey foot for EPSG:2227 and 1.0 for EPSG:32610
        let ca_ft = cogzilla_proj::CrsModel::from_epsg(2227).unwrap();
        assert!((ca_ft.linear_unit_metres() - 1200.0 / 3937.0).abs() < 1e-6);
        let utm10 = cogzilla_proj::CrsModel::from_epsg(32610).unwrap();
        assert!((utm10.linear_unit_metres() - 1.0).abs() < 1e-9);

        // 7. resolve_bbox_dst_crs defaults to coord_crs when dst_crs is omitted
        let pq_utm = parse_query(Some("url=a.tif&coord_crs=EPSG:32610"));
        assert_eq!(resolve_bbox_dst_crs(&pq_utm).map(|(_, e)| e), Some(32610));

        // 8. Form-encoded comparison & semicolon expressions decode `+` as space
        let pq_cmp = parse_query(Some("expression=b1+%3E+100"));
        assert_eq!(pq_cmp.expr_raw.as_deref(), Some("b1 > 100"));
        let pq_semi = parse_query(Some("expression=b1%3B+b2%3B+b3"));
        assert_eq!(pq_semi.expr_raw.as_deref(), Some("b1; b2; b3"));

        // 9. resolve_zoom_range honors ?maxzoom < def_min and WMTS tif MIME type
        assert_eq!(resolve_zoom_range(10, 15, None, Some(5)), (5, 5));
        let tif_xml = build_wmts_capabilities(
            "http://localhost:8080",
            "cog",
            "WebMercatorQuad",
            "Test",
            [-122.7, 37.7, -122.5, 37.8],
            5,
            5,
            "tif",
            1,
            "url=a.tif",
            "url=a.tif&tile_format=tif",
            false,
        );
        assert!(tif_xml.contains("<Format>image/tiff; application=geotiff</Format>"));

        // 10. /{family}/map.html must reject OGC Maps-only width/dst_crs params
        let pq_map_html = parse_query(Some("url=a.tif&width=256"));
        assert!(validate_query(&["cog", "map.html"], &pq_map_html).is_err());
        assert!(validate_query(&["cog", "map"], &pq_map_html).is_ok());

        // 11. Antimeridian-crossing WGS84 bbox (w > e, Fiji) projects across 180°, not 0°
        let utm01s = cogzilla_proj::CrsModel::from_epsg(32701).unwrap();
        let fiji_b = reproject_wgs84_bounds([177.0, -20.0, -178.0, -16.0], &utm01s).unwrap();
        let (fiji_x, fiji_y) = utm01s.from_lonlat_deg(179.5, -18.0);
        assert!(
            fiji_x >= fiji_b[0] && fiji_x <= fiji_b[2] && fiji_y >= fiji_b[1] && fiji_y <= fiji_b[3],
            "Fiji ({fiji_x}, {fiji_y}) outside antimeridian projected bounds {fiji_b:?}"
        );
        assert!(
            (fiji_b[2] - fiji_b[0]) < 1_000_000.0,
            "antimeridian bounds swept wrong hemisphere: {fiji_b:?}"
        );

        // 12. normalize_render_bbox unwraps antimeridian Geographic bboxes & rejects inverted boxes
        let fiji_geo = normalize_render_bbox(
            [177.0, -20.0, -178.0, -16.0],
            &cogzilla_proj::CrsModel::Geographic,
        )
        .unwrap();
        assert_eq!(fiji_geo, [177.0, -20.0, 182.0, -16.0]);
        assert_eq!(aspect_preserving_dims(fiji_geo, 512), (512, 410));
        assert!(normalize_render_bbox(
            [0.0, 10.0, 1.0, -10.0],
            &cogzilla_proj::CrsModel::Geographic,
        )
        .is_err());
        assert!(normalize_render_bbox(
            [10.0, 0.0, 5.0, 1.0],
            &cogzilla_proj::CrsModel::Epsg3857,
        )
        .is_err());

        // 13. bbox_to_wgs84 unwraps antimeridian-crossing projected bboxes instead of inverting
        let utm60n = cogzilla_proj::CrsModel::from_epsg(32660).unwrap();
        let am_wgs = bbox_to_wgs84([800_000.0, 0.0, 900_000.0, 100_000.0], Some(&utm60n)).unwrap();
        assert!(
            am_wgs[0] > 179.0 && am_wgs[2] > 180.0 && (am_wgs[2] - am_wgs[0]) < 2.0,
            "antimeridian UTM60N bbox inverted to complement: {am_wgs:?}"
        );

        // 14. reproject_wgs84_bounds into UTM on a global [-180,-90,180,90] COG covers >67.5°N
        let utm31_global = reproject_wgs84_bounds([-180.0, -90.0, 180.0, 90.0], &utm31).unwrap();
        let (_, sval_y) = utm31.from_lonlat_deg(3.0, 78.0);
        assert!(
            sval_y <= utm31_global[3],
            "global UTM31 bounds truncated below 78°N ({sval_y} > {})",
            utm31_global[3]
        );

        // 15. Buffered CDB1GlobalGrid @4x tile exceeding 4096 px must return 400 (not silent clamp)
        let pq_cdb_buf = parse_query(Some("url=a.tif&buffer=64"));
        assert!(build_tile_request_for_tms(
            None,
            TileCoord { z: 0, x: 0, y: 0 },
            Some(4),
            Some("png"),
            &pq_cdb_buf,
            1,
            cogzilla_tiff::SampleType::Uint8,
            "CDB1GlobalGrid",
            None,
        )
        .is_err());

        // 16. bbox_to_wgs84 preserves full [-180, 180] on global EPSG:3857 and polar EPSG:3413,
        // and reproject_wgs84_bounds rejects polar-cap bboxes wholly outside TM's ±89.35° domain
        let m = 20_037_508.342_789_244;
        let glob_3857 =
            bbox_to_wgs84([-m, -m, m, m], Some(&cogzilla_proj::CrsModel::Epsg3857)).unwrap();
        assert!(
            (glob_3857[0] + 180.0).abs() < 1e-6 && (glob_3857[2] - 180.0).abs() < 1e-6,
            "global EPSG:3857 bbox must span [-180, 180], got {glob_3857:?}"
        );
        let ps_north = cogzilla_proj::CrsModel::from_epsg(3413).unwrap();
        let polar_wgs =
            bbox_to_wgs84([-2_000_000.0, -2_000_000.0, 2_000_000.0, 2_000_000.0], Some(&ps_north))
                .unwrap();
        assert_eq!(polar_wgs[0], -180.0);
        assert_eq!(polar_wgs[2], 180.0);
        assert_eq!(polar_wgs[3], 90.0);
        assert!(reproject_wgs84_bounds([0.0, 89.5, 10.0, 90.0], &utm31).is_err());
    }
}
