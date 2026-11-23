use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use cogwheel_kernel::{ColormapName, OutputFormat, ResampleMode, SpectralExpr};
use cogwheel_proj::TileCoord;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::header::{CONTENT_TYPE, HeaderValue};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use tokio::net::TcpListener;

use crate::engine::{BandRequest, EngineError, TileEngine, TileRequest};

fn percent_decode(input: &str) -> String {
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
        } else if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn parse_y_and_ext(seg: &str) -> Option<(u32, &str)> {
    let (y_str, ext) = seg.split_once('.')?;
    let y = y_str.parse::<u32>().ok()?;
    Some((y, ext))
}

struct ParsedQuery {
    urls: Vec<String>,
    bidx: Vec<u16>,
    rescale: Vec<(f32, f32)>,
    colormap: ColormapName,
    resample: ResampleMode,
    tile_size: u32,
    webp_quality: Option<f32>,
    expr: Option<SpectralExpr>,
}

fn parse_query(q: Option<&str>) -> ParsedQuery {
    let mut pq = ParsedQuery {
        urls: Vec::new(),
        bidx: Vec::new(),
        rescale: Vec::new(),
        colormap: ColormapName::Viridis,
        resample: ResampleMode::Bilinear,
        tile_size: 256,
        webp_quality: None,
        expr: None,
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
            "bidx" => {
                for part in v.split(',') {
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
            "colormap" | "colormap_name" => {
                if let Some(cm) = ColormapName::from_str_opt(v) {
                    pq.colormap = cm;
                }
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
            "quality" => {
                if v != "lossless" {
                    if let Ok(qual) = v.parse::<f32>() {
                        pq.webp_quality = Some(qual.clamp(1.0, 100.0));
                    }
                }
            }
            "expr" | "expression" | "index" => {
                pq.expr = SpectralExpr::from_str_opt(v);
            }
            _ => {}
        }
    }
    pq
}

fn build_tile_request(
    tile: TileCoord,
    ext: &str,
    pq: &ParsedQuery,
    default_bands: u16,
) -> Result<(TileRequest, &'static str), EngineError> {
    let (format, content_type) = match ext {
        "png" => (OutputFormat::Png, "image/png"),
        "webp" => (
            OutputFormat::Webp {
                quality: pq.webp_quality,
            },
            "image/webp",
        ),
        "f32" | "bin" | "raw" => (OutputFormat::RawF32, "application/octet-stream"),
        other => {
            return Err(EngineError::BadRequest(format!(
                "unsupported tile extension '.{other}' (use .png, .webp, or .f32)"
            )));
        }
    };

    let bands = if let Some(expr) = pq.expr {
        match expr {
            SpectralExpr::Db => {
                let rescale = pq.rescale.first().copied().unwrap_or((-25.0, 0.0));
                let b = pq.bidx.first().copied().unwrap_or(1);
                BandRequest::Spectral1 {
                    expr,
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
                    // Standard ASF / OPERA RTC-S1 dual-pol dB stretch: VV[-20,0], VH[-25,-5], VV-VH[0,15]
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
                    expr,
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
                    expr,
                    bands_1based: bs,
                    rescale,
                    colormap: pq.colormap,
                }
            }
        }
    } else if pq.bidx.len() == 3 || (pq.bidx.is_empty() && default_bands >= 3) {
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
        BandRequest::Rgb {
            bands_1based: bs,
            rescale_min: rmin,
            rescale_max: rmax,
        }
    } else {
        let b = pq.bidx.first().copied().unwrap_or(1);
        let rescale = pq.rescale.first().copied().unwrap_or((0.0, 1000.0));
        BandRequest::Single {
            band_1based: b,
            rescale,
            colormap: pq.colormap,
        }
    };

    Ok((
        TileRequest {
            tile,
            tile_size: pq.tile_size,
            bands,
            resample: pq.resample,
            format,
        },
        content_type,
    ))
}

fn plain_resp(status: StatusCode, body: impl Into<Bytes>, content_type: &'static str) -> Response<Full<Bytes>> {
    let mut resp = Response::new(Full::new(body.into()));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    resp
}

pub async fn handle_request(
    engine: Arc<TileEngine>,
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let path = req.uri().path();
    let method = req.method();

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

    if method != Method::GET {
        return Ok(plain_resp(
            StatusCode::METHOD_NOT_ALLOWED,
            Bytes::from_static(b"method not allowed"),
            "text/plain",
        ));
    }

    let pq = parse_query(req.uri().query());

    // Zero-alloc path matching:
    // 1) /tiles/{dataset}/{z}/{x}/{y}.{ext}
    // 2) /cog/tiles/{z}/{x}/{y}.{ext}?url=...
    let mut segs = path.trim_start_matches('/').split('/');
    let first = segs.next().unwrap_or("");

    let result = match first {
        "tiles" => {
            let dataset = segs.next();
            let z = segs.next().and_then(|s| s.parse::<u8>().ok());
            let x = segs.next().and_then(|s| s.parse::<u32>().ok());
            let y_ext = segs.next().and_then(parse_y_and_ext);
            match (dataset, z, x, y_ext) {
                (Some(ds), Some(z), Some(x), Some((y, ext))) => {
                    if let Some(scene) = engine.get_scene(ds) {
                        let default_bands = scene.len() as u16;
                        match build_tile_request(TileCoord { z, x, y }, ext, &pq, default_bands) {
                            Ok((treq, ct)) => engine
                                .render_multi_cog_tile(&scene, &treq)
                                .await
                                .map(|b| (b, ct)),
                            Err(e) => Err(e),
                        }
                    } else if let Some(cog) = engine.store.get_registered(ds) {
                        let default_bands = cog.index.full_res().samples_per_pixel;
                        match build_tile_request(TileCoord { z, x, y }, ext, &pq, default_bands) {
                            Ok((treq, ct)) => engine
                                .render_tile(&cog, &treq)
                                .await
                                .map(|b| (b, ct)),
                            Err(e) => Err(e),
                        }
                    } else {
                        Err(EngineError::NotFound(ds.to_string()))
                    }
                }
                _ => Err(EngineError::BadRequest(
                    "expected /tiles/{dataset}/{z}/{x}/{y}.{png|webp|f32}".to_string(),
                )),
            }
        }
        "cog" if segs.next() == Some("tiles") => {
            let z = segs.next().and_then(|s| s.parse::<u8>().ok());
            let x = segs.next().and_then(|s| s.parse::<u32>().ok());
            let y_ext = segs.next().and_then(parse_y_and_ext);
            match (pq.urls.as_slice(), z, x, y_ext) {
                ([], _, _, _) => Err(EngineError::BadRequest(
                    "missing ?url=... query parameter (repeatable for multi-band Landsat/SAR)"
                        .to_string(),
                )),
                (urls, Some(z), Some(x), Some((y, ext))) => {
                    let mut opened = Vec::with_capacity(urls.len());
                    let mut open_err = None;
                    for u in urls {
                        match engine.store.open_uri(u).await {
                            Ok(c) => opened.push(c),
                            Err(e) => {
                                open_err = Some(EngineError::Io(e));
                                break;
                            }
                        }
                    }
                    if let Some(e) = open_err {
                        Err(e)
                    } else {
                        let default_bands = if opened.len() > 1 {
                            opened.len() as u16
                        } else {
                            opened[0].index.full_res().samples_per_pixel
                        };
                        match build_tile_request(TileCoord { z, x, y }, ext, &pq, default_bands) {
                            Ok((treq, ct)) => engine
                                .render_multi_cog_tile(&opened, &treq)
                                .await
                                .map(|b| (b, ct)),
                            Err(e) => Err(e),
                        }
                    }
                }
                _ => Err(EngineError::BadRequest(
                    "expected /cog/tiles/{z}/{x}/{y}.{png|webp|f32}?url=...".to_string(),
                )),
            }
        }
        _ => Err(EngineError::NotFound(path.to_string())),
    };

    match result {
        Ok((bytes, content_type)) => Ok(plain_resp(StatusCode::OK, bytes, content_type)),
        Err(EngineError::NotFound(msg)) => Ok(plain_resp(
            StatusCode::NOT_FOUND,
            Bytes::from(msg),
            "text/plain",
        )),
        Err(EngineError::BadRequest(msg)) => Ok(plain_resp(
            StatusCode::BAD_REQUEST,
            Bytes::from(msg),
            "text/plain",
        )),
        Err(e) => Ok(plain_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            Bytes::from(e.to_string()),
            "text/plain",
        )),
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
