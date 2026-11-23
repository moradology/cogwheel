use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use cogwheel_kernel::{
    ColorFormula, ColormapLut, ColormapName, ExprProgram, OutputFormat, ResampleMode, SpectralExpr,
};
use cogwheel_proj::{TileCoord, mercator_to_lonlat_deg, tile_bounds_3857};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CONTENT_TYPE, HeaderValue};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use tokio::net::TcpListener;

use crate::engine::{BandRequest, EngineError, TileEngine, TileRequest};
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
}

fn parse_query(q: Option<&str>) -> ParsedQuery {
    let mut pq = ParsedQuery {
        urls: Vec::new(),
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
    };
    let Some(q) = q else { return pq };
    for pair in q.split('&') {
        if pair.is_empty() {
            continue;
        }
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        match k {
            "url" => pq.urls.push(percent_decode(v)),
            "bidx" | "indexes" => {
                for part in percent_decode(v).split(',') {
                    if let Ok(b) = part.parse::<u16>() {
                        pq.bidx.push(b);
                    }
                }
            }
            "rescale" => {
                let decoded = percent_decode(v);
                for group in decoded.split(';') {
                    if let Some((a, b)) = group.split_once(',') {
                        if let (Ok(mn), Ok(mx)) = (a.parse::<f32>(), b.parse::<f32>()) {
                            pq.rescale.push((mn, mx));
                        }
                    }
                }
            }
            "colormap" => {
                let dec = percent_decode(v);
                if dec.trim_start().starts_with('{') {
                    pq.custom_colormap_json = Some(dec);
                } else if let Some(cm) = ColormapName::from_str_opt(&dec) {
                    pq.colormap = cm;
                }
            }
            "colormap_name" => {
                if let Some(cm) = ColormapName::from_str_opt(&percent_decode(v)) {
                    pq.colormap = cm;
                }
            }
            "color_formula" => {
                pq.color_formula = Some(percent_decode(v));
            }
            "nodata" => {
                let dec = percent_decode(v);
                if let Ok(nd) = dec.parse::<f32>() {
                    pq.nodata = Some(nd);
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
                }
            }
            "coord_crs" => {
                let dec = percent_decode(v);
                pq.coord_crs_3857 = dec.contains("3857");
            }
            "resample" | "resampling" => {
                pq.resample = match v {
                    "nearest" => ResampleMode::Nearest,
                    _ => ResampleMode::Bilinear,
                };
            }
            "tile_size" | "size" => {
                if let Ok(sz) = v.parse::<u32>() {
                    pq.tile_size = sz;
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
            _ => {}
        }
    }
    pq
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

fn build_band_request(pq: &ParsedQuery, default_bands: u16) -> Result<BandRequest, EngineError> {
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
            _ => ([0.0; 3], [1.0; 3]),
        };
        Ok(BandRequest::Rgb {
            bands_1based: bs,
            rescale_min: rmin,
            rescale_max: rmax,
        })
    } else {
        let b = pq.bidx.first().copied().unwrap_or(1);
        let rescale = pq.rescale.first().copied().unwrap_or((0.0, 1000.0));
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
) -> Result<(TileRequest, &'static str), EngineError> {
    let (format, content_type) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
    let bands = build_band_request(pq, default_bands)?;
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

async fn open_query_urls(
    engine: &TileEngine,
    urls: &[String],
) -> Result<Vec<Arc<cogwheel_io::OpenedCog>>, EngineError> {
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
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let raw_query = req.uri().query().unwrap_or("").to_string();

    if method == Method::OPTIONS {
        return Ok(plain_resp(StatusCode::NO_CONTENT, Bytes::new(), "text/plain"));
    }

    if method == Method::GET && path == "/healthz" {
        return Ok(plain_resp(StatusCode::OK, Bytes::from_static(b"ok"), "text/plain"));
    }

    if method == Method::GET && path == "/stats" {
        let body = serde_json::json!({
            "tiles_served": engine.stats.tiles_served.load(Ordering::Relaxed),
            "empty_short_circuits": engine.stats.empty_short_circuits.load(Ordering::Relaxed),
            "cache_hits": engine.store.cache.hits.load(Ordering::Relaxed),
            "cache_misses": engine.store.cache.misses.load(Ordering::Relaxed),
            "singleflight_coalesced": engine.store.cache.coalesced_waiters.load(Ordering::Relaxed),
            "cache_bytes": engine.store.cache.current_bytes(),
            "remote_range_requests": engine.store.metrics.remote_range_requests.load(Ordering::Relaxed),
            "remote_bytes_fetched": engine.store.metrics.remote_bytes_fetched.load(Ordering::Relaxed),
        })
        .to_string();
        return Ok(plain_resp(StatusCode::OK, Bytes::from(body), "application/json"));
    }

    if method == Method::POST && path == "/admin/clear-cache" {
        engine.store.cache.clear();
        return Ok(plain_resp(StatusCode::OK, Bytes::from_static(b"cleared"), "text/plain"));
    }

    let pq = parse_query(if raw_query.is_empty() {
        None
    } else {
        Some(&raw_query)
    });

    let segs: Vec<&str> = path.trim_start_matches('/').split('/').collect();

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
        let res = match parsed {
            (Some(z), Some(x), Some((y, sc, ext))) => {
                if let Some(scene) = engine.get_scene(ds) {
                    let db = scene.len() as u16;
                    match build_tile_request(TileCoord { z, x, y }, sc, ext, &pq, db) {
                        Ok((treq, ct)) => engine.render_multi_cog_tile(&scene, &treq).await.map(|b| (b, ct)),
                        Err(e) => Err(e),
                    }
                } else if let Some(cog) = engine.store.get_registered(ds) {
                    let db = cog.index.full_res().samples_per_pixel;
                    match build_tile_request(TileCoord { z, x, y }, sc, ext, &pq, db) {
                        Ok((treq, ct)) => engine.render_tile(&cog, &treq).await.map(|b| (b, ct)),
                        Err(e) => Err(e),
                    }
                } else {
                    Err(EngineError::NotFound(ds.to_string()))
                }
            }
            _ => Err(EngineError::BadRequest(
                "expected /tiles/{dataset}/{z}/{x}/{y}.{png|webp|f32}".to_string(),
            )),
        };
        return Ok(to_http_response(res));
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
            (Method::GET, ["tiles", tms, zs, xs, ys]) => async {
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
                let (mut treq, ct) = build_tile_request(TileCoord { z, x, y }, sc, ext, &pq, db)?;
                if pq.unscale {
                    apply_unscale(&mut treq.bands, &cogs[0].index.scales, &cogs[0].index.offsets);
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
            }
            .await,

            // Alias: GET /cog/tiles/{z}/{x}/{y}[@scale][.{format}] (implicit WebMercatorQuad)
            (Method::GET, ["tiles", zs, xs, ys]) => async {
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
                let (treq, ct) = build_tile_request(TileCoord { z, x, y }, sc, ext, &pq, db)?;
                let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                Ok((data, ct))
            }
            .await,

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
                let mut bands = build_band_request(&pq, db)?;
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
                let mut bands = build_band_request(&pq, db)?;
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
                let mut bands = build_band_request(&pq, db)?;
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
                let mut bands = build_band_request(&pq, db)?;
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
                let mut bands = build_band_request(&pq, db)?;
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
                let mut bands = build_band_request(&pq, db)?;
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
        return Ok(to_http_response(res));
    }

    Ok(plain_resp(
        StatusCode::NOT_FOUND,
        Bytes::from(path),
        "text/plain",
    ))
}

fn to_http_response(res: Result<(Bytes, &'static str), EngineError>) -> Response<Full<Bytes>> {
    match res {
        Ok((bytes, content_type)) => plain_resp(StatusCode::OK, bytes, content_type),
        Err(EngineError::NotFound(msg)) => {
            plain_resp(StatusCode::NOT_FOUND, Bytes::from(msg), "text/plain")
        }
        Err(EngineError::BadRequest(msg)) => {
            plain_resp(StatusCode::BAD_REQUEST, Bytes::from(msg), "text/plain")
        }
        Err(e) => plain_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            Bytes::from(e.to_string()),
            "text/plain",
        ),
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
