use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use cogzilla_io::{CogStore, DiskBlockCache};
use cogzilla_server::{TileEngine, handle_bytes_request, load_catalog_json_file};
use lambda_http::{
    Body, Error, Request, RequestExt, Response, request::RequestContext, run, service_fn,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn build_engine_from_env() -> Arc<TileEngine> {
    let cache_mb = env_usize("COGZILLA_CACHE_MB", 256);
    let tile_cache_mb = env_usize("COGZILLA_TILE_CACHE_MB", 0);
    let disk_cache_mb = env_usize("COGZILLA_DISK_CACHE_MB", 0) as u64;

    let mut store = CogStore::new(cache_mb * 1024 * 1024);
    if disk_cache_mb > 0 {
        let dir = std::env::var("COGZILLA_DISK_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp/cogzilla-disk-cache"));
        match DiskBlockCache::open(&dir, disk_cache_mb * 1024 * 1024) {
            Ok(dc) => {
                tracing::info!(
                    dir = %dir.display(),
                    disk_cache_mb,
                    "enabled tier-2 compressed-block disk cache on Lambda /tmp"
                );
                store = store.with_disk_cache(dc);
            }
            Err(e) => {
                tracing::warn!(
                    dir = %dir.display(),
                    error = %e,
                    "failed to open Lambda disk cache; continuing with tier 2 disabled"
                );
            }
        }
    }

    let engine = Arc::new(TileEngine::with_store_and_tile_cache(
        Arc::new(store),
        tile_cache_mb * 1024 * 1024,
    ));

    match engine.attach_pgstac_from_env() {
        Ok(true) => tracing::info!("pgSTAC database configured; serving /searches"),
        Ok(false) => {}
        Err(e) => tracing::warn!(error = %e, "invalid pgSTAC database settings; /searches disabled"),
    }

    if let Ok(specs) = std::env::var("COGZILLA_MOSAIC") {
        for spec in specs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if let Some((name, path)) = spec.split_once('=') {
                match load_catalog_json_file(path) {
                    Ok(cat) => {
                        let scenes = cat.len();
                        engine.register_catalog(name, cat);
                        tracing::info!(mosaic = name, scenes, "registered Lambda mosaic catalog");
                    }
                    Err(e) => {
                        tracing::warn!(mosaic = name, path, error = %e, "failed to load mosaic catalog");
                    }
                }
            }
        }
    }

    engine
}

fn is_binary_content_type(ct: &str) -> bool {
    ct.starts_with("image/")
        || ct.starts_with("application/octet-stream")
        || ct.starts_with("application/x-npy")
}

/// Strip a leading `/{stage}` prefix added by API Gateway v1 (REST) or API Gateway v2 named stages
/// (anything other than `$default`), preserving any query string and mapping `/{stage}` -> `/`.
/// Returns `(rewritten_uri, Some("/{stage}"))` when a stage prefix was stripped so downstream
/// URL builders (TileJSON, `/docs`, `/colorMaps`, etc.) preserve the public stage prefix.
fn strip_stage_prefix(
    uri: &lambda_http::http::Uri,
    stage: Option<&str>,
) -> (lambda_http::http::Uri, Option<String>) {
    let Some(s) = stage.filter(|s| !s.is_empty() && *s != "$default") else {
        return (uri.clone(), None);
    };
    let path = uri.path();
    let prefix = format!("/{s}");
    let stripped = if let Some(rest) = path.strip_prefix(&prefix) {
        if rest.is_empty() {
            "/"
        } else if rest.starts_with('/') {
            rest
        } else {
            return (uri.clone(), None);
        }
    } else {
        return (uri.clone(), None);
    };
    let new_pq = match uri.query() {
        Some(q) => format!("{stripped}?{q}"),
        None => stripped.to_string(),
    };
    let mut parts = uri.clone().into_parts();
    if let Ok(pq) = new_pq.parse() {
        parts.path_and_query = Some(pq);
        if let Ok(u) = lambda_http::http::Uri::from_parts(parts) {
            return (u, Some(prefix));
        }
    }
    (uri.clone(), None)
}

/// Core AWS Lambda HTTP handler: translates `lambda_http::Request` into `cogzilla-server`'s
/// shared `handle_bytes_request` and encodes binary image/raster payloads as `Body::Binary`
/// (automatic `isBase64Encoded: true` for Function URLs, API Gateway v1/v2, and ALB).
async fn lambda_handler(engine: Arc<TileEngine>, event: Request) -> Result<Response<Body>, Error> {
    let stage = event.request_context_ref().and_then(|ctx| match ctx {
        RequestContext::ApiGatewayV1(c) => c.stage.clone(),
        RequestContext::ApiGatewayV2(c) => c.stage.clone(),
        _ => None,
    });
    let (mut parts, body) = event.into_parts();
    let (new_uri, stage_prefix) = strip_stage_prefix(&parts.uri, stage.as_deref());
    parts.uri = new_uri;
    if let Some(prefix) = stage_prefix {
        if !parts.headers.contains_key("x-forwarded-prefix") {
            if let Ok(hv) = lambda_http::http::HeaderValue::from_str(&prefix) {
                parts.headers.insert("x-forwarded-prefix", hv);
            }
        }
    }
    let req_bytes = match body {
        Body::Empty => Bytes::new(),
        Body::Text(s) => Bytes::from(s),
        Body::Binary(b) => Bytes::from(b),
    };
    let req = lambda_http::http::Request::from_parts(parts, req_bytes);
    let resp = handle_bytes_request(engine, req).await;

    let (rparts, rbytes) = resp.into_parts();
    let ct = rparts
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let out_body = if rbytes.is_empty() {
        Body::Empty
    } else if is_binary_content_type(ct) {
        Body::Binary(Vec::from(rbytes))
    } else {
        match String::from_utf8(Vec::from(rbytes)) {
            Ok(s) => Body::Text(s),
            Err(e) => Body::Binary(e.into_bytes()),
        }
    };

    Ok(Response::from_parts(rparts, out_body))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .without_time()
        .init();

    let engine = build_engine_from_env();
    tracing::info!(
        cache_mb = env_usize("COGZILLA_CACHE_MB", 256),
        "cogzilla-lambda initialized"
    );

    run(service_fn(move |event: Request| {
        let eng = engine.clone();
        async move { lambda_handler(eng, event).await }
    }))
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../benches/fixtures")
            .join(name)
    }

    #[tokio::test]
    async fn lambda_handler_serves_healthz_and_metrics() {
        let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
        let req = lambda_http::http::Request::builder()
            .method("GET")
            .uri("/healthz")
            .body(Body::Empty)
            .unwrap();
        let resp = lambda_handler(engine.clone(), req).await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.body(), &Body::Text("ok".to_string()));

        let mreq = lambda_http::http::Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::Empty)
            .unwrap();
        let mresp = lambda_handler(engine, mreq).await.unwrap();
        assert_eq!(mresp.status(), 200);
        match mresp.body() {
            Body::Text(s) => assert!(s.contains("cogzilla_tiles_served_total")),
            other => panic!("expected Body::Text for /metrics, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn lambda_handler_serves_binary_png_tile_and_304_not_modified() {
        let merc_path = fixture("mercator_4k_zstd_p3.tif");
        if !merc_path.exists() {
            return;
        }
        let merc_url = merc_path.canonicalize().unwrap().to_string_lossy().to_string();
        let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));

        let uri = format!(
            "/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&algorithm=hillshade&resampling=cubic"
        );
        let req = lambda_http::http::Request::builder()
            .method("GET")
            .uri(&uri)
            .body(Body::Empty)
            .unwrap();
        let resp = lambda_handler(engine.clone(), req).await.unwrap();
        assert_eq!(resp.status(), 200);
        let etag = resp
            .headers()
            .get("etag")
            .expect("ETag missing")
            .to_str()
            .unwrap()
            .to_string();
        match resp.body() {
            Body::Binary(b) => {
                assert!(b.len() > 100);
                assert_eq!(&b[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            }
            other => panic!("expected Body::Binary for PNG tile, got {other:?}"),
        }

        // Conditional GET with If-None-Match -> 304 Not Modified with Body::Empty
        let req304 = lambda_http::http::Request::builder()
            .method("GET")
            .uri(&uri)
            .header("if-none-match", etag)
            .body(Body::Empty)
            .unwrap();
        let resp304 = lambda_handler(engine, req304).await.unwrap();
        assert_eq!(resp304.status(), 304);
        assert_eq!(resp304.body(), &Body::Empty);
    }

    #[test]
    fn strip_stage_prefix_handles_apigw_named_stages_and_default() {
        let u1: lambda_http::http::Uri = "/prod/cog/info?url=s3://b/k.tif".parse().unwrap();
        let (r1, p1) = strip_stage_prefix(&u1, Some("prod"));
        assert_eq!(r1.to_string(), "/cog/info?url=s3://b/k.tif");
        assert_eq!(p1.as_deref(), Some("/prod"));
        // $default stage must NOT strip
        let (r2, p2) = strip_stage_prefix(&u1, Some("$default"));
        assert_eq!(r2.to_string(), "/prod/cog/info?url=s3://b/k.tif");
        assert_eq!(p2, None);
        // Exact /{stage} maps to /
        let u2: lambda_http::http::Uri = "/v1".parse().unwrap();
        let (r3, p3) = strip_stage_prefix(&u2, Some("v1"));
        assert_eq!(r3.to_string(), "/");
        assert_eq!(p3.as_deref(), Some("/v1"));
        // Partial prefix match (/v10 vs stage v1) must NOT strip
        let u3: lambda_http::http::Uri = "/v10/healthz".parse().unwrap();
        let (r4, p4) = strip_stage_prefix(&u3, Some("v1"));
        assert_eq!(r4.to_string(), "/v10/healthz");
        assert_eq!(p4, None);
    }
}
