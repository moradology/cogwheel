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
use hyper::body::Incoming;
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
    get_colormap_json, get_colormap_png, get_tms_json, list_colormaps_json, list_tms_json,
    load_stac_item, resolve_stac_assets,
};
use crate::titiler::{
    PolygonMask, build_cog_info, build_cog_info_geojson, build_map_html, build_tilejson,
    compute_statistics, query_point, render_wgs84_bbox,
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
    Some((w, h, ext))
}

fn parse_bbox4(seg: &str) -> Option<([f64; 4], Option<&str>)> {
    // Handle `minx,miny,maxx,maxy.png` or `minx,miny,maxx,maxy`
    let (coords_str, ext) = if let Some(dot) = seg.rfind('.') {
        let after = &seg[dot + 1..];
        if matches!(after, "png" | "webp" | "f32" | "bin" | "raw" | "jpg" | "jpeg") {
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
    algorithm: Option<String>,
    azimuth: f32,
    altitude: f32,
    z_factor: f32,
    colormap_explicit: bool,
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
        colormap: ColormapName::Viridis,
        custom_colormap_json: None,
        color_formula: None,
        resample: ResampleMode::Bilinear,
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
        algorithm: None,
        azimuth: 315.0,
        altitude: 45.0,
        z_factor: 1.0,
        colormap_explicit: false,
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
            "url" => pq.urls.push(percent_decode(v)),
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
                if serde_json::from_str::<serde_json::Value>(&dec).is_err() {
                    bad("expected a JSON object");
                }
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&dec) {
                    if let Some(a) = obj.get("azimuth").and_then(|x| x.as_f64()) {
                        pq.azimuth = a as f32;
                    }
                    if let Some(a) = obj
                        .get("angle_altitude")
                        .or_else(|| obj.get("altitude"))
                        .or_else(|| obj.get("angle"))
                        .and_then(|x| x.as_f64())
                    {
                        pq.altitude = (a as f32).clamp(0.0, 90.0);
                    }
                    if let Some(z) = obj
                        .get("z_factor")
                        .or_else(|| obj.get("zfactor"))
                        .and_then(|x| x.as_f64())
                    {
                        pq.z_factor = (z as f32).max(1e-4);
                    }
                }
            }
            "azimuth" => {
                if let Ok(a) = v.parse::<f32>() {
                    pq.azimuth = a;
                } else {
                    bad("expected a number");
                }
            }
            "altitude" | "angle" | "angle_altitude" => {
                if let Ok(a) = v.parse::<f32>() {
                    pq.altitude = a.clamp(0.0, 90.0);
                } else {
                    bad("expected a number");
                }
            }
            "z_factor" | "zfactor" => {
                if let Ok(z) = v.parse::<f32>() {
                    pq.z_factor = z.max(1e-4);
                } else {
                    bad("expected a number");
                }
            }
            "bidx" | "indexes" => {
                for part in percent_decode(v).split(',') {
                    if let Ok(b) = part.parse::<u16>() {
                        pq.bidx.push(b);
                    } else {
                        bad("expected band numbers");
                    }
                }
            }
            "rescale" => {
                let decoded = percent_decode(v);
                for group in decoded.split(';') {
                    match group.split_once(',').map(|(a, b)| (a.parse::<f32>(), b.parse::<f32>())) {
                        Some((Ok(mn), Ok(mx))) => pq.rescale.push((mn, mx)),
                        _ => bad("expected min,max"),
                    }
                }
            }
            "colormap" => {
                let dec = percent_decode(v);
                if dec.trim_start().starts_with('{') {
                    pq.custom_colormap_json = Some(dec);
                } else if let Some(cm) = ColormapName::from_str_opt(&dec) {
                    pq.colormap = cm;
                } else {
                    bad("expected a JSON colormap or a colormap name");
                }
            }
            "colormap_name" => {
                if let Some(cm) = ColormapName::from_str_opt(&percent_decode(v)) {
                    pq.colormap = cm;
                    pq.colormap_explicit = true;
                } else {
                    bad("unknown colormap (see /colorMaps)");
                }
            }
            "color_formula" => {
                pq.color_formula = Some(percent_decode(v));
            }
            "nodata" => {
                let dec = percent_decode(v);
                if let Ok(nd) = dec.parse::<f32>() {
                    pq.nodata = Some(nd);
                } else {
                    bad("expected a number");
                }
            }
            "unscale" => {
                pq.unscale = v.eq_ignore_ascii_case("true") || v == "1";
            }
            "return_mask" => {
                pq.return_mask = !(v.eq_ignore_ascii_case("false") || v == "0");
            }
            "buffer" => {
                if let Ok(buf) = v.parse::<f32>() {
                    pq.buffer = buf.round().clamp(0.0, 256.0) as u32;
                } else {
                    bad("expected a number");
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
                        ResampleMode::Bilinear
                    }
                };
            }
            "tile_size" | "tilesize" | "size" => {
                match v.parse::<u32>() {
                    Ok(sz) if (64..=2048).contains(&sz) => pq.tile_size = sz,
                    _ => bad("expected a tile size from 64 to 2048"),
                }
            }
            "scale" => {
                if let Ok(sc) = v.parse::<u32>() {
                    pq.tile_size = match sc {
                        2 => 512,
                        _ => 256,
                    };
                }
            }
            "max_size" => {
                if let Ok(ms) = v.parse::<u32>() {
                    pq.max_size = ms.clamp(64, 4096);
                } else {
                    bad("expected a number");
                }
            }
            "quality" => {
                if v != "lossless" {
                    if let Ok(qual) = v.parse::<f32>() {
                        pq.webp_quality = Some(qual.clamp(1.0, 100.0));
                        pq.jpeg_quality = qual.round().clamp(1.0, 100.0) as u8;
                    }
                }
            }
            "expr" | "expression" | "index" => {
                pq.expr_raw = Some(percent_decode_opts(v, false));
            }
            "padding" => {
                if v.parse::<f32>().ok() != Some(0.0) {
                    bad("padding is not supported");
                }
            }
            "reproject" | "dst_crs" | "crs" | "histogram_bins" | "histogram_range"
            | "categorical" | "c" | "p" | "cover_scale" | "height" | "width" | "minzoom"
            | "maxzoom" | "tile_format" | "orientation" => {
                pq_errors.push(format!("parameter '{k}' is not supported by cogzilla"));
            }
            _ => {}
        }
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
            if !SUPPORTED_TMS.contains(&tms) {
                return Err(format!(
                    "tile matrix set '{tms}' is not supported (use WebMercatorQuad or WorldCRS84Quad)"
                ));
            }
        }
    }

    let has = |names: &[&str]| pq.keys.iter().find(|k| names.contains(&k.as_str())).cloned();
    let route = |name: &str| segs[1..].iter().any(|s| *s == name);
    if route("statistics") || route("asset_statistics") {
        if let Some(k) = has(&["expression", "expr", "index", "algorithm", "bidx", "indexes"]) {
            return Err(format!("parameter '{k}' is not supported on statistics routes"));
        }
    }
    if route("point") {
        if let Some(k) = has(&["expression", "expr", "index", "algorithm", "bidx", "indexes"]) {
            return Err(format!("parameter '{k}' is not supported on point routes"));
        }
    } else if family == "stac" || !route("bbox") {
        if let Some(k) = has(&["coord_crs"]) {
            return Err(format!("parameter '{k}' is not supported on this route"));
        }
    }
    if route("bbox") {
        if let Some(k) = has(&["max_size"]) {
            return Err(format!("parameter '{k}' is not supported on bbox routes"));
        }
    }
    Ok(())
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
        "f32" | "bin" | "raw" | "npy" => Ok((OutputFormat::RawF32, "application/octet-stream")),
        other => Err(EngineError::BadRequest(format!(
            "unsupported format '.{other}' (use .png, .webp, .jpg, or .f32)"
        ))),
    }
}

/// Fold GDAL `scale`/`offset` into rescale bounds when `?unscale=true` (zero per-pixel cost!).
fn apply_unscale(bands: &mut BandRequest, scales: &[f32], offsets: &[f32]) {
    let adj = |b1: u16, mn: f32, mx: f32| -> (f32, f32) {
        let i = b1.saturating_sub(1) as usize;
        let sc = scales.get(i).copied().unwrap_or(1.0);
        let off = offsets.get(i).copied().unwrap_or(0.0);
        if (sc - 1.0).abs() < 1e-9 && off.abs() < 1e-9 {
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

fn build_band_request(
    pq: &ParsedQuery,
    default_bands: u16,
    is_u8: bool,
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
                if is_u8 {
                    ([0.0; 3], [255.0; 3])
                } else {
                    ([0.0; 3], [1.0; 3])
                }
            }
        };
        Ok(BandRequest::Rgb {
            bands_1based: bs,
            rescale_min: rmin,
            rescale_max: rmax,
        })
    } else {
        let b = pq.bidx.first().copied().unwrap_or(1);
        let default_range = if is_u8 { (0.0, 255.0) } else { (0.0, 1000.0) };
        let rescale = pq.rescale.first().copied().unwrap_or(default_range);
        Ok(BandRequest::Single {
            band_1based: b,
            rescale,
            colormap: pq.colormap,
        })
    }
}

fn build_tile_request(
    tile: TileCoord,
    scale_opt: Option<u32>,
    ext: &str,
    pq: &ParsedQuery,
    default_bands: u16,
    is_u8: bool,
) -> Result<(TileRequest, &'static str), EngineError> {
    let (format, content_type) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
    let bands = build_band_request(pq, default_bands, is_u8)?;
    let tile_size = match scale_opt {
        Some(2) => 512,
        Some(1) => 256,
        _ => pq.tile_size,
    };
    Ok((
        TileRequest {
            tile,
            tile_size,
            bands,
            resample: pq.resample,
            format,
        },
        content_type,
    ))
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

fn web_mercator_buffered_wgs84_bbox(z: u8, x: u32, y: u32, buffer: u32, tile_size: u32) -> [f64; 4] {
    let (min_x, min_y, max_x, max_y) = tile_bounds_3857(TileCoord { z, x, y });
    let pad_x = (max_x - min_x) * (buffer as f64 / tile_size as f64);
    let pad_y = (max_y - min_y) * (buffer as f64 / tile_size as f64);
    let (w, s) = mercator_to_lonlat_deg(min_x - pad_x, min_y - pad_y);
    let (e, n) = mercator_to_lonlat_deg(max_x + pad_x, max_y + pad_y);
    [w, s, e, n]
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
    let text = if uri.starts_with("http://") || uri.starts_with("https://") {
        let resp = engine
            .store
            .client
            .get(uri)
            .send()
            .await
            .map_err(cogzilla_io::IoError::Http)?;
        if !resp.status().is_success() {
            return Err(EngineError::Io(cogzilla_io::IoError::HttpStatus {
                status: resp.status().as_u16(),
                url: uri.to_string(),
            }));
        }
        resp.text().await.map_err(cogzilla_io::IoError::Http)?
    } else if let Some((bucket, key)) = cogzilla_io::parse_s3_uri(uri) {
        let cfg = engine
            .store
            .s3_cfg
            .clone()
            .unwrap_or_else(cogzilla_io::S3Config::from_env_full);
        let bytes =
            cogzilla_io::fetch_s3_object(&engine.store.client, bucket, key, &cfg, None, None)
                .await?;
        String::from_utf8_lossy(&bytes).into_owned()
    } else {
        let local_path = uri.strip_prefix("file://").unwrap_or(uri);
        std::fs::read_to_string(local_path).map_err(cogzilla_io::IoError::Fs)?
    };
    let cat = Arc::new(MosaicJsonCatalog::from_json_str(&text).map_err(EngineError::BadRequest)?);
    engine.mosaicjson_cache.insert(uri.to_string(), cat.clone());
    Ok(cat)
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
    let mut opened = Vec::with_capacity(urls.len());
    for u in urls {
        opened.push(engine.store.open_uri(u).await?);
    }
    Ok(opened)
}

pub async fn handle_request(
    engine: Arc<TileEngine>,
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let host = req
        .headers()
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("127.0.0.1:8080")
        .to_string();
    let if_none_match = req
        .headers()
        .get(IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let raw_query = req.uri().query().unwrap_or("").to_string();

    if method == Method::OPTIONS {
        return Ok(plain_resp(StatusCode::NO_CONTENT, Bytes::new(), "text/plain"));
    }

    if method == Method::GET && path == "/healthz" {
        return Ok(plain_resp(StatusCode::OK, Bytes::from_static(b"ok"), "text/plain"));
    }

    if method == Method::GET && (path == "/" || path == "/index.html") {
        let html = include_str!("landing.html").replace("{{VERSION}}", env!("CARGO_PKG_VERSION"));
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
        const SWAGGER_HTML: &str = r#"<!DOCTYPE html>
<html><head><title>cogzilla API Docs</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"/>
</head><body><div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({url:'/openapi.json',dom_id:'#swagger-ui'});</script>
</body></html>"#;
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from_static(SWAGGER_HTML.as_bytes()),
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
            let body_bytes = req
                .into_body()
                .collect()
                .await
                .map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?
                .to_bytes();
            let spec: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid JSON body: {e}")))?;
            let name = spec
                .get("name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty() && *s != "tiles" && *s != "register")
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
                        EngineError::BadRequest(format!("STAC API search failed: {e}"))
                    })?;
                if !resp.status().is_success() {
                    return Err(EngineError::BadRequest(format!(
                        "STAC API search returned HTTP {}",
                        resp.status()
                    )));
                }
                let text = resp.text().await.map_err(|e| {
                    EngineError::BadRequest(format!("STAC API response read failed: {e}"))
                })?;
                load_catalog_json_str(&text).map_err(EngineError::BadRequest)?
            } else {
                return Err(EngineError::BadRequest(
                    "expected 'catalog' or 'stac_api_url' in /mosaic/register body".into(),
                ));
            };

            let count = catalog.len();
            engine.register_catalog(name.clone(), catalog);
            let body = serde_json::json!({
                "name": name,
                "scenes": count,
                "tiles_url": format!("/mosaic/{name}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"),
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
                    format!("http://{host}/mosaicjson/tiles/{tms}/{{z}}/{{x}}/{{y}}?{raw_query}");
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
            ["mosaicjson", "tiles", _, zs, xs, ys] | ["mosaicjson", "tiles", zs, xs, ys] => {
                let cache_key = format!("{path}?{raw_query}");
                engine
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
                        let tile = TileCoord { z, x, y };
                        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
                        let urls = mj.query_bbox_3857(min_x, min_y, max_x, max_y);
                        if urls.is_empty() {
                            let (treq, ct) = build_tile_request(tile, sc, ext, &pq, 3, true)?;
                            Ok((
                                cogzilla_kernel::empty_tile_bytes(treq.tile_size, treq.format, 3),
                                ct,
                            ))
                        } else {
                            let first_cog = engine.store.open_uri(&urls[0]).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let is_u8 = first_cog.index.full_res().sample_type
                                == cogzilla_tiff::SampleType::Uint8;
                            let (treq, ct) = build_tile_request(tile, sc, ext, &pq, db, is_u8)?;
                            let b = engine.render_mosaic_tile(&urls, &treq).await?;
                            Ok((b, ct))
                        }
                    })
                    .await
                    .map(|t| (t.bytes, t.content_type))
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
            ["mosaic", "tiles", _, zs, xs, ys] | ["mosaic", "tiles", zs, xs, ys] => (
                zs.parse::<u8>().ok(),
                xs.parse::<u32>().ok(),
                parse_y_scale_ext(ys),
            ),
            _ => (None, None, None),
        };
        let cache_key = format!("{path}?{raw_query}");
        let res = match coords {
            (Some(z), Some(x), Some((y, sc, ext))) => {
                if pq.urls.is_empty() {
                    Err(EngineError::BadRequest(
                        "missing required ?url= parameter(s) for /mosaic/tiles".into(),
                    ))
                } else {
                    engine
                        .tile_cache
                        .get_or_render(cache_key, || async {
                            let first_cog = engine.store.open_uri(&pq.urls[0]).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let is_u8 = first_cog.index.full_res().sample_type
                                == cogzilla_tiff::SampleType::Uint8;
                            let (treq, ct) =
                                build_tile_request(TileCoord { z, x, y }, sc, ext, &pq, db, is_u8)?;
                            let b = engine.render_mosaic_tile(&pq.urls, &treq).await?;
                            Ok((b, ct))
                        })
                        .await
                        .map(|t| (t.bytes, t.content_type))
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
            ["mosaic", _, "tiles", _, zs, xs, ys] | ["mosaic", _, "tiles", zs, xs, ys] => (
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
                engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        let tile = TileCoord { z, x, y };
                        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
                        let urls = catalog.query_bbox_3857(min_x, min_y, max_x, max_y);
                        if urls.is_empty() {
                            let (treq, ct) = build_tile_request(tile, sc, ext, &pq, 3, true)?;
                            Ok((
                                cogzilla_kernel::empty_tile_bytes(treq.tile_size, treq.format, 3),
                                ct,
                            ))
                        } else {
                            let first_cog = engine.store.open_uri(&urls[0]).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let is_u8 = first_cog.index.full_res().sample_type
                                == cogzilla_tiff::SampleType::Uint8;
                            let (treq, ct) = build_tile_request(tile, sc, ext, &pq, db, is_u8)?;
                            let b = engine.render_mosaic_tile(&urls, &treq).await?;
                            Ok((b, ct))
                        }
                    })
                    .await
                    .map(|t| (t.bytes, t.content_type))
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
                engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        if let Some(scene) = engine.get_scene(ds) {
                            let db = scene.len() as u16;
                            let is_u8 = scene[0].index.full_res().sample_type
                                == cogzilla_tiff::SampleType::Uint8;
                            let (treq, ct) =
                                build_tile_request(TileCoord { z, x, y }, sc, ext, &pq, db, is_u8)?;
                            let b = engine.render_multi_cog_tile(&scene, &treq).await?;
                            Ok((b, ct))
                        } else if let Some(cog) = engine.store.get_registered(ds) {
                            let db = cog.index.full_res().samples_per_pixel;
                            let is_u8 = cog.index.full_res().sample_type
                                == cogzilla_tiff::SampleType::Uint8;
                            let (treq, ct) =
                                build_tile_request(TileCoord { z, x, y }, sc, ext, &pq, db, is_u8)?;
                            let b = engine.render_tile(&cog, &treq).await?;
                            Ok((b, ct))
                        } else {
                            Err(EngineError::NotFound(ds.to_string()))
                        }
                    })
                    .await
                    .map(|t| (t.bytes, t.content_type))
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
            ["colorMaps", id] => {
                if pq.format_param.as_deref() == Some("png") {
                    get_colormap_png(id).map(|b| (b, "image/png"))
                } else {
                    get_colormap_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
                }
            }
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
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let keys: Vec<&String> = item.assets.keys().collect();
                Ok((Bytes::from(serde_json::json!(keys).to_string()), "application/json"))
            }
            .await,

            // GET /stac/info?url=item.json&assets=...
            (Method::GET, ["info"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
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
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
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
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let mut map = serde_json::Map::new();
                for (name, cog) in names.into_iter().zip(cogs.into_iter()) {
                    let st = compute_statistics(&engine, &cog, pq.max_size, None, pq.unscale).await?;
                    map.insert(name, st);
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,
            // GET /stac/statistics (merged band dict: {b1: ..., b2: ...})
            (Method::GET, ["statistics"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let (_, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let mut map = serde_json::Map::new();
                let mut band_idx = 1usize;
                for cog in cogs {
                    let st = compute_statistics(&engine, &cog, pq.max_size, None, pq.unscale).await?;
                    if let Some(obj) = st.as_object() {
                        for (_, v) in obj {
                            map.insert(format!("b{band_idx}"), v.clone());
                            band_idx += 1;
                        }
                    }
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,
            (Method::POST, ["statistics"]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let body_bytes = req
                    .into_body()
                    .collect()
                    .await
                    .map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?
                    .to_bytes();
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, _) = PolygonMask::from_geojson(&geojson)?;
                let mut map = serde_json::Map::new();
                for (name, cog) in names.into_iter().zip(cogs.into_iter()) {
                    let st =
                        compute_statistics(&engine, &cog, pq.max_size, Some(&mask), pq.unscale)
                            .await?;
                    map.insert(name, st);
                }
                let feat = serde_json::json!({
                    "type": "Feature",
                    "geometry": geojson.get("geometry").cloned().unwrap_or(geojson),
                    "properties": { "statistics": map },
                });
                Ok((Bytes::from(feat.to_string()), "application/geo+json"))
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
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let (_, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut tj = build_tilejson(&engine, &cogs[0], &host, tms, &raw_query)?;
                let tile_url = format!("http://{host}/stac/tiles/{tms}/{{z}}/{{x}}/{{y}}?{raw_query}");
                tj["tiles"] = serde_json::json!([tile_url]);
                tj["name"] = serde_json::json!(item.id);
                Ok((Bytes::from(tj.to_string()), "application/json"))
            }
            .await,
            (Method::GET, [tms, "map.html"]) => {
                let tj_url = format!("/stac/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }

            // GET /stac/tiles/{tms}/{z}/{x}/{y}[@scale][.{format}]
            (Method::GET, ["tiles", tms, zs, xs, ys]) => {
                let cache_key = format!("{path}?{raw_query}");
                engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        let u = stac_url
                            .ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                        let item =
                            load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
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
                        let is_u8 = cogs[0].index.full_res().sample_type
                            == cogzilla_tiff::SampleType::Uint8;
                        let (mut treq, ct) = build_tile_request(
                            TileCoord { z, x, y },
                            sc,
                            ext,
                            &pq_stac,
                            db,
                            is_u8,
                        )?;
                        if pq_stac.unscale {
                            apply_unscale(
                                &mut treq.bands,
                                &cogs[0].index.scales,
                                &cogs[0].index.offsets,
                            );
                        }
                        let needs_general = *tms == "WorldCRS84Quad"
                            || pq_stac.buffer > 0
                            || pq_stac.nodata.is_some()
                            || pq_stac.custom_colormap_json.is_some()
                            || pq_stac.color_formula.is_some()
                            || !pq_stac.return_mask;
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
                            let bbox = if *tms == "WorldCRS84Quad" {
                                world_crs84_tile_bbox(z, x, y, pq_stac.buffer, treq.tile_size)
                            } else {
                                web_mercator_buffered_wgs84_bbox(
                                    z,
                                    x,
                                    y,
                                    pq_stac.buffer,
                                    treq.tile_size,
                                )
                            };
                            let data = render_wgs84_bbox(
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
                            )
                            .await?;
                            Ok((data, ct))
                        } else {
                            let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            Ok((data, ct))
                        }
                    })
                    .await
                    .map(|t| (t.bytes, t.content_type))
            }

            // GET /stac/point/{lon},{lat}
            (Method::GET, ["point", coord]) => async {
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
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
                let mut all_vals = Vec::new();
                let mut all_names = Vec::new();
                for (name, cog) in names.into_iter().zip(cogs.into_iter()) {
                    let pt = query_point(&engine, &cog, lon, lat).await?;
                    if let Some(vs) = pt["values"].as_array() {
                        for (i, v) in vs.iter().enumerate() {
                            all_vals.push(v.clone());
                            all_names.push(format!("{name}_b{}", i + 1));
                        }
                    }
                }
                let body = serde_json::json!({
                    "coordinates": [lon, lat],
                    "values": all_vals,
                    "band_names": all_names,
                })
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // GET /stac/preview[/{w}x{h}][.{ext}]
            (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
                let ext = p.strip_prefix("preview.").unwrap_or("png");
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
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
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let bands = build_band_request(&pq_stac, db, is_u8)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                    pq_stac.nodata, None, None, pq_stac.return_mask,
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
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let bands = build_band_request(&pq_stac, db, is_u8)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                    pq_stac.nodata, None, None, pq_stac.return_mask,
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
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext_opt.unwrap_or("png"), pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let bands = build_band_request(&pq_stac, db, is_u8)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, 512, 512, None, &bands, pq_stac.resample, fmt,
                    pq_stac.nodata, None, None, pq_stac.return_mask,
                )
                .await?;
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
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let bands = build_band_request(&pq_stac, db, is_u8)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                    pq_stac.nodata, None, None, pq_stac.return_mask,
                )
                .await?;
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
                let body_bytes = req
                    .into_body()
                    .collect()
                    .await
                    .map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?
                    .to_bytes();
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson(&geojson)?;
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let bands = build_band_request(&pq_stac, db, is_u8)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, Some(&mask), &bands, pq_stac.resample, fmt,
                    pq_stac.nodata, None, None, pq_stac.return_mask,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::POST, [feat_ext]) if feat_ext.starts_with("feature.") => async {
                let ext = feat_ext.strip_prefix("feature.").unwrap_or("png");
                let s = pq.max_size.min(1024);
                let u = stac_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let body_bytes = req
                    .into_body()
                    .collect()
                    .await
                    .map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?
                    .to_bytes();
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson(&geojson)?;
                let item = load_stac_item(&engine.stac_cache, &engine.store.client, &u).await?;
                let (_, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref()).await?;
                let mut pq_stac = pq;
                pq_stac.expr_raw = rewritten_expr;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { cogs[0].index.full_res().samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let bands = build_band_request(&pq_stac, db, is_u8)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, s, s, Some(&mask), &bands, pq_stac.resample, fmt,
                    pq_stac.nodata, None, None, pq_stac.return_mask,
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
                let json = compute_statistics(&engine, &cogs[0], pq.max_size, None, pq.unscale).await?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,
            (Method::POST, ["statistics"]) => async {
                let body_bytes = req
                    .into_body()
                    .collect()
                    .await
                    .map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?
                    .to_bytes();
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                if body_bytes.is_empty() {
                    let json =
                        compute_statistics(&engine, &cogs[0], pq.max_size, None, pq.unscale).await?;
                    Ok((Bytes::from(json.to_string()), "application/json"))
                } else {
                    let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                        .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                    let (mask, _) = PolygonMask::from_geojson(&geojson)?;
                    let stats =
                        compute_statistics(&engine, &cogs[0], pq.max_size, Some(&mask), pq.unscale)
                            .await?;
                    if geojson.get("type").and_then(|v| v.as_str()) == Some("Feature") {
                        if let Some(props) = geojson.get_mut("properties").and_then(|v| v.as_object_mut()) {
                            props.insert("statistics".into(), stats);
                        } else if let Some(obj) = geojson.as_object_mut() {
                            obj.insert("properties".into(), serde_json::json!({"statistics": stats}));
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
            (Method::GET, ["tiles", tms, zs, xs, ys]) => {
                let cache_key = format!("{path}?{raw_query}");
                engine
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
                        let is_u8 = cogs[0].index.full_res().sample_type
                            == cogzilla_tiff::SampleType::Uint8;
                        let (mut treq, ct) =
                            build_tile_request(TileCoord { z, x, y }, sc, ext, &pq, db, is_u8)?;
                        if pq.unscale {
                            apply_unscale(
                                &mut treq.bands,
                                &cogs[0].index.scales,
                                &cogs[0].index.offsets,
                            );
                        }

                        let needs_general_path = *tms == "WorldCRS84Quad"
                            || pq.buffer > 0
                            || pq.nodata.is_some()
                            || pq.custom_colormap_json.is_some()
                            || pq.color_formula.is_some()
                            || !pq.return_mask;

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
                            let bbox = if *tms == "WorldCRS84Quad" {
                                world_crs84_tile_bbox(z, x, y, pq.buffer, treq.tile_size)
                            } else {
                                web_mercator_buffered_wgs84_bbox(z, x, y, pq.buffer, treq.tile_size)
                            };
                            let data = render_wgs84_bbox(
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
                            )
                            .await?;
                            Ok((data, ct))
                        } else {
                            let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            Ok((data, ct))
                        }
                    })
                    .await
                    .map(|t| (t.bytes, t.content_type))
            }

            // Alias: GET /cog/tiles/{z}/{x}/{y}[@scale][.{format}] (implicit WebMercatorQuad)
            (Method::GET, ["tiles", zs, xs, ys]) => {
                let cache_key = format!("{path}?{raw_query}");
                engine
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
                        let is_u8 = cogs[0].index.full_res().sample_type
                            == cogzilla_tiff::SampleType::Uint8;
                        let (treq, ct) =
                            build_tile_request(TileCoord { z, x, y }, sc, ext, &pq, db, is_u8)?;
                        let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                        Ok((data, ct))
                    })
                    .await
                    .map(|t| (t.bytes, t.content_type))
            }

            // 8. GET /cog/{tileMatrixSetId}/tilejson.json & /cog/tilejson.json
            (Method::GET, [tms, "tilejson.json"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let json = build_tilejson(&engine, &cogs[0], &host, tms, &raw_query)?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,
            (Method::GET, ["tilejson.json"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let json = build_tilejson(&engine, &cogs[0], &host, "WebMercatorQuad", &raw_query)?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 9. GET /cog/{tileMatrixSetId}/map.html
            (Method::GET, [tms, "map.html"]) => {
                let tj_url = format!("/cog/{tms}/tilejson.json?{raw_query}");
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
                let json = query_point(&engine, &cogs[0], lon, lat).await?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 11. GET /cog/preview, /cog/preview.{format}, /cog/preview/{w}x{h}.{format}
            (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
                let ext = p.strip_prefix("preview.").unwrap_or("png");
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                let out_w = ((full.width as f64 * scale.min(1.0)).round() as u32).max(1);
                let out_h = ((full.height as f64 * scale.min(1.0)).round() as u32).max(1);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let mut bands = build_band_request(&pq, db, is_u8)?;
                if pq.unscale {
                    apply_unscale(&mut bands, &cogs[0].index.scales, &cogs[0].index.offsets);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq.resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask,
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
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let mut bands = build_band_request(&pq, db, is_u8)?;
                if pq.unscale {
                    apply_unscale(&mut bands, &cogs[0].index.scales, &cogs[0].index.offsets);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq.resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask,
                )
                .await?;
                Ok((data, ct))
            }
            .await,

            // 12. GET /cog/bbox/{minx},{miny},{maxx},{maxy}[.{format}] & /cog/bbox/{...}/{w}x{h}.{format}
            (Method::GET, ["bbox", bbox_seg]) => async {
                let Some((mut bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("expected /cog/bbox/{minx},{miny},{maxx},{maxy}.{ext}".into()));
                };
                if pq.coord_crs_3857 {
                    let (w, s) = mercator_to_lonlat_deg(bbox[0], bbox[1]);
                    let (e, n) = mercator_to_lonlat_deg(bbox[2], bbox[3]);
                    bbox = [w, s, e, n];
                }
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext_opt.unwrap_or("png"), pq.webp_quality, pq.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let mut bands = build_band_request(&pq, db, is_u8)?;
                if pq.unscale {
                    apply_unscale(&mut bands, &cogs[0].index.scales, &cogs[0].index.offsets);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, 512, 512, None, &bands, pq.resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["bbox", bbox_seg, dims_ext]) => async {
                let Some((mut bbox, _)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox coordinates".into()));
                };
                if pq.coord_crs_3857 {
                    let (w, s) = mercator_to_lonlat_deg(bbox[0], bbox[1]);
                    let (e, n) = mercator_to_lonlat_deg(bbox[2], bbox[3]);
                    bbox = [w, s, e, n];
                }
                let Some((out_w, out_h, ext)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected {w}x{h}.{ext}".into()));
                };
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let mut bands = build_band_request(&pq, db, is_u8)?;
                if pq.unscale {
                    apply_unscale(&mut bands, &cogs[0].index.scales, &cogs[0].index.offsets);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, None, &bands, pq.resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask,
                )
                .await?;
                Ok((data, ct))
            }
            .await,

            // 13. POST /cog/feature, /cog/feature.{format}, /cog/feature/{w}x{h}.{format}
            (Method::POST, [f]) if *f == "feature" || f.starts_with("feature.") => async {
                let ext = f.strip_prefix("feature.").unwrap_or("png");
                let body_bytes = req
                    .into_body()
                    .collect()
                    .await
                    .map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?
                    .to_bytes();
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson(&geojson)?;
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let mut bands = build_band_request(&pq, db, is_u8)?;
                if pq.unscale {
                    apply_unscale(&mut bands, &cogs[0].index.scales, &cogs[0].index.offsets);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, 512, 512, Some(&mask), &bands, pq.resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask,
                )
                .await?;
                Ok((data, ct))
            }
            .await,
            (Method::POST, ["feature", dims_ext]) => async {
                let Some((out_w, out_h, ext)) = parse_wxh_ext(dims_ext) else {
                    return Err(EngineError::BadRequest("expected /cog/feature/{w}x{h}.{ext}".into()));
                };
                let body_bytes = req
                    .into_body()
                    .collect()
                    .await
                    .map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?
                    .to_bytes();
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson(&geojson)?;
                let cogs = open_query_urls(&engine, &pq.urls).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let is_u8 = cogs[0].index.full_res().sample_type == cogzilla_tiff::SampleType::Uint8;
                let mut bands = build_band_request(&pq, db, is_u8)?;
                if pq.unscale {
                    apply_unscale(&mut bands, &cogs[0].index.scales, &cogs[0].index.offsets);
                }
                let custom_lut = pq.custom_colormap_json.as_deref().map(ColormapLut::from_json_str).transpose().map_err(EngineError::BadRequest)?;
                let cf = pq.color_formula.as_deref().map(ColorFormula::parse).transpose().map_err(EngineError::BadRequest)?;
                let data = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, Some(&mask), &bands, pq.resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask,
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

/// Fast, deterministic FNV-1a 64-bit strong ETag for rendered tile bytes.
fn compute_etag(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("\"cgz-{h:016x}-{}\"", bytes.len())
}

fn etag_matches(if_none_match: &str, etag: &str) -> bool {
    let trimmed = if_none_match.trim();
    if trimmed == "*" {
        return true;
    }
    for part in trimmed.split(',') {
        let cand = part.trim().strip_prefix("W/").unwrap_or(part.trim());
        if cand == etag {
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
                || content_type == "application/octet-stream";
            if is_binary_tile {
                let etag = compute_etag(&bytes);
                if let Some(inm) = if_none_match {
                    if etag_matches(inm, &etag) {
                        engine
                            .stats
                            .http_304_not_modified
                            .fetch_add(1, Ordering::Relaxed);
                        let mut resp = Response::new(Full::new(Bytes::new()));
                        *resp.status_mut() = StatusCode::NOT_MODIFIED;
                        if let Ok(hv) = HeaderValue::from_str(&etag) {
                            resp.headers_mut().insert(ETAG, hv);
                        }
                        resp.headers_mut()
                            .insert(CACHE_CONTROL, cache_control_header().clone());
                        resp.headers_mut()
                            .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
                        return resp;
                    }
                }
                let mut resp = plain_resp(StatusCode::OK, bytes, content_type);
                if let Ok(hv) = HeaderValue::from_str(&etag) {
                    resp.headers_mut().insert(ETAG, hv);
                }
                resp.headers_mut()
                    .insert(CACHE_CONTROL, cache_control_header().clone());
                resp
            } else {
                plain_resp(StatusCode::OK, bytes, content_type)
            }
        }
        Err(EngineError::NotFound(msg)) => {
            plain_resp(StatusCode::NOT_FOUND, Bytes::from(msg), "text/plain")
        }
        Err(EngineError::BadRequest(msg)) => {
            plain_resp(StatusCode::BAD_REQUEST, Bytes::from(msg), "text/plain")
        }
        Err(EngineError::Io(e @ cogzilla_io::IoError::Forbidden(_))) => {
            plain_resp(StatusCode::FORBIDDEN, Bytes::from(e.to_string()), "text/plain")
        }
        Err(e) => {
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

pub async fn serve_listener(engine: Arc<TileEngine>, listener: TcpListener) -> std::io::Result<()> {
    loop {
        let (stream, _) = listener.accept().await?;
        let _ = stream.set_nodelay(true);
        let io = TokioIo::new(stream);
        let eng = engine.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req| handle_request(eng.clone(), req));
            let _ = ServerBuilder::new(TokioExecutor::new())
                .serve_connection(io, svc)
                .await;
        });
    }
}
