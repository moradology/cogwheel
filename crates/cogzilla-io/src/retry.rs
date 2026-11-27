//! Unified ranged-read & whole-object HTTP retry pipeline for all remote backends (#429).
//!
//! One definition of:
//! - `GDAL_HTTP_MAX_RETRY` parsing ([`RetryPolicy`])
//! - `AimdLimiter` permit acquisition, throttle, and success signaling
//! - `IoMetrics::remote_range_requests` (counted once per upstream attempt, uniformly across
//!   HTTP, S3, GCS, and Azure) and `IoMetrics::remote_bytes_fetched`
//! - Transient transport & `429`/`5xx` retries honoring `Retry-After`
//! - `Range` / `Content-Range` / `If-Match` / ETag / `Last-Modified` validation
//! - Streaming byte-cap body reads via `read_response_capped`

use std::sync::atomic::Ordering;

use bytes::Bytes;

use crate::backend::{
    IoError, IoMetrics, RangeResponse, RemoteValidator, io_error_from_reqwest,
    validate_content_range_start,
};
use crate::cloud::read_response_capped;
use crate::limiter::AimdLimiter;
use crate::s3::{is_retryable_status, is_retryable_transport_error, retry_backoff_duration};

/// Retry policy parsed once from `GDAL_HTTP_MAX_RETRY` (clamped to `0..=10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_retries: u32,
}

impl RetryPolicy {
    /// Parse `GDAL_HTTP_MAX_RETRY` with the given fallback default (clamped to `10`).
    pub fn from_env_with_default(default_retries: u32) -> Self {
        let max_retries = std::env::var("GDAL_HTTP_MAX_RETRY")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(default_retries)
            .min(10);
        Self { max_retries }
    }

    /// Standard default (`3`, matching `GDAL_HTTP_MAX_RETRY`).
    pub fn from_env() -> Self {
        Self::from_env_with_default(3)
    }
}

/// Options governing a single unified GET execution through [`execute_get_with_retry`].
pub(crate) struct FetchSpec<'a> {
    /// When `Some((start, end))`, validate HTTP 200 vs 206 and `Content-Range` start.
    pub range: Option<(u64, u64)>,
    /// Hard streaming body cap in bytes (if `None` and `range` is `Some`, defaults to
    /// `(end - start).min(64 MiB)`).
    pub max_bytes: Option<usize>,
    /// Optional ETag / Last-Modified freshness validator (used by `RemoteHttp`).
    pub validator: Option<&'a RemoteValidator>,
    /// Retry limit.
    pub max_retries: u32,
    /// When true, strip the URL from `reqwest::Error` (used by cloud backends with signed URLs);
    /// when false, preserve redirect-allowlist `IoError::Forbidden` via `io_error_from_reqwest`.
    pub strip_url_on_http_err: bool,
}

/// Execute an HTTP GET (ranged or whole-object) through the single shared retry, AIMD concurrency
/// limiter, `IoMetrics`, range-validation, and byte-cap pipeline.
pub(crate) async fn execute_get_with_retry<F>(
    mut build_req: F,
    spec: FetchSpec<'_>,
    limiter: &AimdLimiter,
    metrics: &IoMetrics,
) -> Result<RangeResponse, IoError>
where
    F: FnMut() -> Result<(reqwest::RequestBuilder, String), IoError>,
{
    let map_http_err = |e: reqwest::Error| -> IoError {
        if spec.strip_url_on_http_err {
            IoError::Http(e.without_url())
        } else {
            io_error_from_reqwest(e)
        }
    };

    let eff_cap = spec.max_bytes.or_else(|| {
        spec.range
            .map(|(s, e)| e.saturating_sub(s).min(64 * 1024 * 1024) as usize)
    });

    for attempt in 0..=spec.max_retries {
        let (req, redacted_url) = build_req()?;
        let permit = limiter.acquire().await;
        metrics.remote_range_requests.fetch_add(1, Ordering::Relaxed);

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                if is_retryable_transport_error(&e) {
                    permit.on_throttle();
                    if attempt < spec.max_retries {
                        tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                        continue;
                    }
                }
                return Err(map_http_err(e));
            }
        };

        let status = resp.status().as_u16();
        if status == 412 && spec.validator.is_some() {
            return Err(IoError::SourceChanged { url: redacted_url });
        }

        if is_retryable_status(status) {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok());
            permit.on_throttle();
            if attempt < spec.max_retries {
                tokio::time::sleep(retry_backoff_duration(attempt, retry_after)).await;
                continue;
            }
            return Err(IoError::HttpStatus {
                status,
                url: redacted_url,
            });
        }

        let resp_etag = resp
            .headers()
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let resp_lm = resp
            .headers()
            .get("last-modified")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        if (status == 200 || status == 206)
            && let Some(v) = spec.validator
        {
            let etag_changed = matches!(
                (&v.etag, &resp_etag),
                (Some(exp), Some(got)) if exp != got
            );
            let lm_changed = matches!(
                (&v.last_modified, &resp_lm),
                (Some(exp), Some(got)) if exp != got
            );
            if etag_changed || lm_changed {
                return Err(IoError::SourceChanged { url: redacted_url });
            }
        }

        if status == 200
            && let Some((start, end)) = spec.range
            && (start > 0
                || resp
                    .content_length()
                    .is_some_and(|cl| cl > end.saturating_sub(start).min(64 * 1024 * 1024)))
        {
            return Err(IoError::RangeNotSupported { url: redacted_url });
        }

        if status == 206
            && let Some((start, _)) = spec.range
            && !validate_content_range_start(&resp, start)
        {
            return Err(IoError::RangeNotSupported { url: redacted_url });
        }

        if status != 200 && status != 206 {
            return Err(IoError::HttpStatus {
                status,
                url: redacted_url,
            });
        }

        let body_res = if let Some(cap) = eff_cap {
            read_response_capped(resp, cap).await
        } else {
            resp.bytes().await.map_err(IoError::Http)
        };

        match body_res {
            Ok(data) => {
                permit.on_success();
                metrics
                    .remote_bytes_fetched
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                return Ok(RangeResponse {
                    bytes: data,
                    etag: resp_etag,
                    last_modified: resp_lm,
                });
            }
            Err(IoError::ResponseTooLarge { .. }) if status == 200 && spec.range.is_some() => {
                return Err(IoError::RangeNotSupported { url: redacted_url });
            }
            Err(IoError::Http(e)) => {
                if is_retryable_transport_error(&e) {
                    permit.on_throttle();
                    if attempt < spec.max_retries {
                        tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                        continue;
                    }
                }
                return Err(map_http_err(e));
            }
            Err(e) => return Err(e),
        }
    }

    Err(IoError::HttpStatus {
        status: 503,
        url: String::new(),
    })
}

/// Helper for callers that only need the raw `Bytes` from [`execute_get_with_retry`].
pub(crate) async fn execute_get_bytes_with_retry<F>(
    build_req: F,
    spec: FetchSpec<'_>,
    limiter: &AimdLimiter,
    metrics: &IoMetrics,
) -> Result<Bytes, IoError>
where
    F: FnMut() -> Result<(reqwest::RequestBuilder, String), IoError>,
{
    Ok(execute_get_with_retry(build_req, spec, limiter, metrics)
        .await?
        .bytes)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bytes::Bytes;
    use http_body_util::Full;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper::{Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    use crate::backend::{CogSource, IoMetrics};
    use crate::cloud::{AzureConfig, AzureTarget, GcsConfig, GcsTokenCache, fetch_azure_range, fetch_gcs_range};
    use crate::limiter::AimdLimiter;
    use crate::s3::{S3Config, fetch_s3_object_capped};

    async fn spawn_flaky_server(fail_count: usize, body: &'static [u8]) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else { break };
                let hits_c = Arc::clone(&hits_srv);
                tokio::spawn(async move {
                    let svc = service_fn(move |_req| {
                        let n = hits_c.fetch_add(1, Ordering::SeqCst);
                        async move {
                            if n < fail_count {
                                Ok::<_, std::convert::Infallible>(
                                    Response::builder()
                                        .status(StatusCode::SERVICE_UNAVAILABLE)
                                        .header("retry-after", "0")
                                        .body(Full::new(Bytes::from_static(b"busy")))
                                        .unwrap(),
                                )
                            } else {
                                Ok::<_, std::convert::Infallible>(
                                    Response::builder()
                                        .status(StatusCode::PARTIAL_CONTENT)
                                        .header("content-range", format!("bytes 0-{}/{}", body.len() - 1, body.len()))
                                        .body(Full::new(Bytes::from_static(body)))
                                        .unwrap(),
                                )
                            }
                        }
                    });
                    let _ = http1::Builder::new().serve_connection(TokioIo::new(stream), svc).await;
                });
            }
        });
        (format!("http://127.0.0.1:{}", addr.port()), hits)
    }

    /// Pin that `remote_range_requests` counts upstream attempts identically across
    /// HTTP, S3, GCS, and Azure: 2× 503 + 1× 206 == 3 attempts and exact byte count (#429).
    #[tokio::test]
    async fn remote_range_requests_counts_attempts_uniformly_across_all_backends() {
        let client = reqwest::Client::new();
        let payload = b"01234567";

        // 1. RemoteHttp
        let (ep_http, _) = spawn_flaky_server(2, payload).await;
        let m_http = Arc::new(IoMetrics::default());
        let lim_http = Arc::new(AimdLimiter::new(1, 8));
        let src_http = CogSource::open_remote_limited(
            &format!("{ep_http}/tile.tif"),
            client.clone(),
            None,
            Arc::clone(&m_http),
            Arc::clone(&lim_http),
        )
        .unwrap();
        let b = src_http.fetch_range(0, 8).await.unwrap();
        assert_eq!(&b[..], payload);
        assert_eq!(m_http.remote_range_requests.load(Ordering::Relaxed), 3);
        assert_eq!(m_http.remote_bytes_fetched.load(Ordering::Relaxed), 8);

        // 2. S3
        let (ep_s3, _) = spawn_flaky_server(2, payload).await;
        let m_s3 = Arc::new(IoMetrics::default());
        let lim_s3 = Arc::new(AimdLimiter::new(1, 8));
        let mut s3_cfg = S3Config::from_env_full();
        s3_cfg.endpoint_url = Some(ep_s3);
        s3_cfg.force_path_style = true;
        s3_cfg.no_sign_request = true;
        s3_cfg.max_retries = 3;
        let b = fetch_s3_object_capped(
            &client,
            "bkt",
            "tile.tif",
            &s3_cfg,
            None,
            Some("bytes=0-7"),
            8,
            Some(&lim_s3),
            Some(&m_s3),
        )
        .await
        .unwrap();
        assert_eq!(&b[..], payload);
        assert_eq!(m_s3.remote_range_requests.load(Ordering::Relaxed), 3);
        assert_eq!(m_s3.remote_bytes_fetched.load(Ordering::Relaxed), 8);

        // 3. GCS
        let (ep_gcs, _) = spawn_flaky_server(2, payload).await;
        let m_gcs = Arc::new(IoMetrics::default());
        let lim_gcs = Arc::new(AimdLimiter::new(1, 8));
        let gcs_cfg = GcsConfig {
            endpoint: ep_gcs,
            no_sign: true,
            bearer_token: None,
            hmac_access_key: None,
            hmac_secret_key: None,
            user_project: None,
            max_retries: 3,
        };
        let tc = GcsTokenCache::default();
        let b = fetch_gcs_range(&client, "bkt", "tile.tif", &gcs_cfg, &tc, 0, 8, &m_gcs, &lim_gcs)
            .await
            .unwrap();
        assert_eq!(&b[..], payload);
        assert_eq!(m_gcs.remote_range_requests.load(Ordering::Relaxed), 3);
        assert_eq!(m_gcs.remote_bytes_fetched.load(Ordering::Relaxed), 8);

        // 4. Azure
        let (ep_az, _) = spawn_flaky_server(2, payload).await;
        let m_az = Arc::new(IoMetrics::default());
        let lim_az = Arc::new(AimdLimiter::new(1, 8));
        let az_cfg = AzureConfig {
            endpoint_override: Some(ep_az),
            no_sign: true,
            sas_token: None,
            bearer_token: None,
            access_key: None,
            max_retries: 3,
        };
        let az_target = AzureTarget {
            account: "acct".into(),
            container: "cont".into(),
            blob: "tile.tif".into(),
        };
        let b = fetch_azure_range(&client, &az_target, &az_cfg, 0, 8, &m_az, &lim_az)
            .await
            .unwrap();
        assert_eq!(&b[..], payload);
        assert_eq!(m_az.remote_range_requests.load(Ordering::Relaxed), 3);
        assert_eq!(m_az.remote_bytes_fetched.load(Ordering::Relaxed), 8);
    }
}
