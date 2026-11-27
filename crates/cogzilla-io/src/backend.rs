use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use memmap2::Mmap;
use thiserror::Error;

use crate::cloud::{
    AzureConfig, AzureTarget, GcsConfig, GcsTokenCache, fetch_azure_range, fetch_gcs_range,
    parse_azure_uri, parse_gcs_uri,
};
use crate::limiter::AimdLimiter;
use crate::s3::{
    DynamicCredsCache, S3Config, is_retryable_status, is_retryable_transport_error, parse_s3_uri,
    retry_backoff_duration,
};

#[derive(Debug, Error)]
pub enum IoError {
    #[error("filesystem error: {0}")]
    Fs(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("HTTP status {status} fetching {url}")]
    HttpStatus { status: u16, url: String },
    #[error("origin does not support HTTP range requests: {url}")]
    RangeNotSupported { url: String },
    #[error("remote source changed while reading: {url}")]
    SourceChanged { url: String },
    #[error("remote response exceeded limit of {limit} bytes")]
    ResponseTooLarge { limit: usize, actual: Option<u64> },
    #[error("TIFF parse error: {0}")]
    Tiff(#[from] cogzilla_tiff::TiffError),
    #[error("codec error: {0}")]
    Codec(#[from] cogzilla_codec::CodecError),
    #[error("invalid URI: {0}")]
    InvalidUri(String),
    #[error("location is not in COGZILLA_ALLOWED_URL_PREFIXES: {0}")]
    Forbidden(String),
    #[error("{0}")]
    Shared(Arc<IoError>),
}

impl IoError {
    /// Unwrap any [`IoError::Shared`] singleflight wrapper to inspect the underlying cause.
    pub fn root(&self) -> &IoError {
        let mut cur = self;
        while let IoError::Shared(inner) = cur {
            cur = inner;
        }
        cur
    }

    /// An error that must never be absorbed by a per-asset fallback:
    /// allowlist rejection (`COGZILLA_ALLOWED_URL_PREFIXES`), where silence
    /// would hide a security refusal.
    pub fn is_fatal(&self) -> bool {
        matches!(self.root(), IoError::Forbidden(_))
    }
}

#[derive(Default)]
pub struct IoMetrics {
    pub remote_range_requests: AtomicU64,
    pub remote_bytes_fetched: AtomicU64,
}

/// HTTP cache validators (`ETag` and `Last-Modified`) captured when opening a remote source.
#[derive(Debug, Clone, Default)]
pub struct RemoteValidator {
    pub etag: Option<String>,
    pub last_modified: Option<String>,
}

/// Byte slice returned by a range read together with any HTTP validators present on the response.
#[derive(Debug, Clone)]
pub struct RangeResponse {
    pub bytes: Bytes,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
}

/// Convert a `reqwest::Error` into `IoError`, surfacing any nested `IoError::Forbidden` thrown
/// by the redirect-allowlist policy hook as top-level `IoError::Forbidden` so every caller maps
/// cross-allowlist redirects to `403 Forbidden`.
pub fn io_error_from_reqwest(e: reqwest::Error) -> IoError {
    let mut src: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(&e);
    while let Some(s) = src {
        if let Some(IoError::Forbidden(u)) = s.downcast_ref::<IoError>() {
            return IoError::Forbidden(u.clone());
        }
        src = s.source();
    }
    IoError::Http(e.without_url())
}

#[derive(Clone)]
pub enum CogSource {
    LocalMmap {
        mmap: Arc<Mmap>,
    },
    RemoteS3 {
        bucket: String,
        key: String,
        client: reqwest::Client,
        s3_cfg: S3Config,
        dynamic_creds: Arc<DynamicCredsCache>,
        metrics: Arc<IoMetrics>,
        limiter: Arc<AimdLimiter>,
    },
    RemoteHttp {
        url: String,
        client: reqwest::Client,
        max_retries: u32,
        metrics: Arc<IoMetrics>,
        limiter: Arc<AimdLimiter>,
    },
    RemoteGcs {
        bucket: String,
        key: String,
        client: reqwest::Client,
        cfg: GcsConfig,
        token_cache: Arc<GcsTokenCache>,
        metrics: Arc<IoMetrics>,
        limiter: Arc<AimdLimiter>,
    },
    RemoteAzure {
        target: AzureTarget,
        client: reqwest::Client,
        cfg: AzureConfig,
        metrics: Arc<IoMetrics>,
        limiter: Arc<AimdLimiter>,
    },
}

/// Strip `file://` / `file://localhost` for local paths, or reject any unrecognized `<scheme>://`
/// URI with [`IoError::InvalidUri`] naming the unsupported scheme and the supported ones.
pub(crate) fn resolve_local_or_reject_scheme(uri: &str) -> Result<&str, IoError> {
    match crate::allowlist::strip_file_uri_prefix(uri) {
        Ok(Some(local)) => return Ok(local),
        Ok(None) => {}
        Err(()) => {
            return Err(IoError::InvalidUri(format!(
                "non-local host in file:// URI '{uri}' (expected file:///path or file://localhost/path)"
            )));
        }
    }
    if let Some(idx) = uri.find("://") {
        let scheme = &uri[..idx];
        let is_valid_scheme = !scheme.is_empty()
            && scheme.as_bytes()[0].is_ascii_alphabetic()
            && scheme
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'));
        if is_valid_scheme {
            return Err(IoError::InvalidUri(format!(
                "unsupported URI scheme '{scheme}://' (supported: http, https, s3, gs, az, abfs, abfss, file, or local path)"
            )));
        }
    }
    Ok(uri)
}

/// Strip `?query`, `#fragment`, and `user:pass@` userinfo from an HTTP(S) URL before surfacing it
/// in `IoError::HttpStatus` so presigned S3 / Azure SAS / basic-auth credentials never leak.
pub fn redact_http_url(url: &str) -> String {
    let no_query = url.split(['?', '#']).next().unwrap_or(url);
    if let Some(scheme_end) = no_query.find("://") {
        let after_scheme = &no_query[scheme_end + 3..];
        let authority_end = after_scheme.find('/').unwrap_or(after_scheme.len());
        let authority = &after_scheme[..authority_end];
        if let Some(at) = authority.rfind('@') {
            return format!(
                "{}{}{}",
                &no_query[..scheme_end + 3],
                &authority[at + 1..],
                &after_scheme[authority_end..]
            );
        }
    }
    no_query.to_string()
}

/// A requested tile byte range `(slot_idx, file_offset, byte_len)`.
#[derive(Debug, Clone, Copy)]
pub struct TileByteReq {
    pub slot: usize,
    pub offset: u64,
    pub len: u32,
}

/// A coalesced contiguous HTTP byte range `[start, end)` covering one or more `TileByteReq`s.
#[derive(Debug, Clone)]
pub struct CoalescedSpan {
    pub start: u64,
    pub end: u64,
    pub items: Vec<TileByteReq>,
}

/// Coalesce nearby byte ranges separated by `<= gap_threshold` bytes (up to `max_span_bytes`)
/// to minimize HTTP/S3 round trips.
pub fn coalesce_ranges(
    mut reqs: Vec<TileByteReq>,
    gap_threshold: u64,
    max_span_bytes: u64,
) -> Vec<CoalescedSpan> {
    if reqs.is_empty() {
        return Vec::new();
    }
    reqs.sort_unstable_by_key(|r| r.offset);

    let mut spans: Vec<CoalescedSpan> = Vec::with_capacity(reqs.len());
    for r in reqs {
        let r_end = r.offset.saturating_add(r.len as u64);
        if let Some(last) = spans.last_mut() {
            let new_end = last.end.max(r_end);
            let contained = r.offset >= last.start && r_end <= last.end;
            if contained
                || (r.offset <= last.end.saturating_add(gap_threshold)
                    && new_end.saturating_sub(last.start) <= max_span_bytes)
            {
                last.end = new_end;
                last.items.push(r);
                continue;
            }
        }
        spans.push(CoalescedSpan {
            start: r.offset,
            end: r_end,
            items: vec![r],
        });
    }
    spans
}

pub(crate) fn shared_dynamic_creds() -> Arc<DynamicCredsCache> {
    static CREDS: std::sync::OnceLock<Arc<DynamicCredsCache>> = std::sync::OnceLock::new();
    CREDS
        .get_or_init(|| Arc::new(DynamicCredsCache::default()))
        .clone()
}

pub(crate) fn shared_gcs_token_cache() -> Arc<GcsTokenCache> {
    static CACHE: std::sync::OnceLock<Arc<GcsTokenCache>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Arc::new(GcsTokenCache::default())).clone()
}

impl CogSource {
    pub fn open_local(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let file = File::open(path)?;
        // SAFETY: COG files are treated as immutable read-only datasets.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self::LocalMmap {
            mmap: Arc::new(mmap),
        })
    }

    pub fn open_remote(
        uri: &str,
        client: reqwest::Client,
        s3_cfg: Option<S3Config>,
        metrics: Arc<IoMetrics>,
    ) -> Result<Self, IoError> {
        Self::open_remote_limited(uri, client, s3_cfg, metrics, Arc::new(AimdLimiter::from_env()))
    }

    pub fn open_remote_limited(
        uri: &str,
        client: reqwest::Client,
        s3_cfg: Option<S3Config>,
        metrics: Arc<IoMetrics>,
        limiter: Arc<AimdLimiter>,
    ) -> Result<Self, IoError> {
        if let Some((bucket, key)) = parse_s3_uri(uri) {
            let cfg = s3_cfg.unwrap_or_else(S3Config::from_env_full);
            Ok(Self::RemoteS3 {
                bucket: bucket.to_string(),
                key: key.to_string(),
                client,
                s3_cfg: cfg,
                dynamic_creds: shared_dynamic_creds(),
                metrics,
                limiter,
            })
        } else if uri.starts_with("s3://") || uri.starts_with("/vsis3/") {
            Err(IoError::InvalidUri(uri.to_string()))
        } else if let Some((bucket, key)) = parse_gcs_uri(uri) {
            Ok(Self::RemoteGcs {
                bucket: bucket.to_string(),
                key: key.to_string(),
                client,
                cfg: GcsConfig::from_env(),
                token_cache: shared_gcs_token_cache(),
                metrics,
                limiter,
            })
        } else if uri.starts_with("gs://") || uri.starts_with("/vsigs/") {
            Err(IoError::InvalidUri(uri.to_string()))
        } else if let Some(target) = parse_azure_uri(uri, None) {
            Ok(Self::RemoteAzure {
                target,
                client,
                cfg: AzureConfig::from_env(),
                metrics,
                limiter,
            })
        } else if uri.starts_with("az://")
            || uri.starts_with("abfs://")
            || uri.starts_with("abfss://")
            || uri.starts_with("/vsiaz/")
        {
            Err(IoError::InvalidUri(uri.to_string()))
        } else if let Some(http_url) = uri
            .strip_prefix("/vsicurl/")
            .filter(|u| u.starts_with("http://") || u.starts_with("https://"))
            .or_else(|| {
                (uri.starts_with("http://") || uri.starts_with("https://")).then_some(uri)
            })
        {
            let max_retries = std::env::var("GDAL_HTTP_MAX_RETRY")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(3)
                .min(10);
            Ok(Self::RemoteHttp {
                url: http_url.to_string(),
                client,
                max_retries,
                metrics,
                limiter,
            })
        } else if uri.starts_with("/vsicurl/") {
            Err(IoError::InvalidUri(uri.to_string()))
        } else {
            let local_path = resolve_local_or_reject_scheme(uri)?;
            Self::open_local(local_path)
        }
    }

    /// Fetch a byte range `[start, end)` (exclusive end) from a remote source.
    pub async fn fetch_range(&self, start: u64, end: u64) -> Result<Bytes, IoError> {
        Ok(self.fetch_range_with_meta(start, end, None).await?.bytes)
    }

    /// Fetch a byte range `[start, end)` (exclusive end), optionally sending `If-Match` and
    /// verifying that the response's `ETag` / `Last-Modified` match `validator` so a single read
    /// can never mix bytes from two versions of a replaced remote file.
    pub async fn fetch_range_with_meta(
        &self,
        start: u64,
        end: u64,
        validator: Option<&RemoteValidator>,
    ) -> Result<RangeResponse, IoError> {
        if start >= end {
            return Ok(RangeResponse {
                bytes: Bytes::new(),
                etag: None,
                last_modified: None,
            });
        }
        match self {
            Self::LocalMmap { mmap } => {
                let s = usize::try_from(start).unwrap_or(usize::MAX).min(mmap.len());
                let e = usize::try_from(end).unwrap_or(usize::MAX).min(mmap.len());
                Ok(RangeResponse {
                    bytes: Bytes::copy_from_slice(&mmap[s..e]),
                    etag: None,
                    last_modified: None,
                })
            }
            Self::RemoteS3 {
                bucket,
                key,
                client,
                s3_cfg,
                dynamic_creds,
                metrics,
                limiter,
            } => {
                let range_hdr = format!("bytes={start}-{}", end.saturating_sub(1));
                let expected_len = end.saturating_sub(start).min(64 * 1024 * 1024) as usize;
                let data = crate::s3::fetch_s3_object_capped(
                    client,
                    bucket,
                    key,
                    s3_cfg,
                    Some(dynamic_creds),
                    Some(&range_hdr),
                    expected_len,
                    Some(limiter),
                    Some(metrics),
                )
                .await?;
                Ok(RangeResponse {
                    bytes: data,
                    etag: None,
                    last_modified: None,
                })
            }
            Self::RemoteHttp {
                url,
                client,
                max_retries,
                metrics,
                limiter,
            } => {
                let range_hdr = format!("bytes={start}-{}", end.saturating_sub(1));
                let strong_etag = validator
                    .and_then(|v| v.etag.as_deref())
                    .filter(|e| !e.trim_start().starts_with("W/"));
                let redacted = redact_http_url(url);
                crate::retry::execute_get_with_retry(
                    || {
                        let mut req = client.get(url).header("Range", &range_hdr);
                        if let Some(etag) = strong_etag {
                            req = req.header("If-Match", etag);
                        }
                        Ok((req, redacted.clone()))
                    },
                    crate::retry::FetchSpec {
                        range: Some((start, end)),
                        max_bytes: None,
                        validator,
                        max_retries: *max_retries,
                        strip_url_on_http_err: false,
                    },
                    limiter,
                    metrics,
                )
                .await
            }
            Self::RemoteGcs {
                bucket,
                key,
                client,
                cfg,
                token_cache,
                metrics,
                limiter,
            } => {
                let bytes = fetch_gcs_range(
                    client,
                    bucket,
                    key,
                    cfg,
                    token_cache,
                    start,
                    end,
                    metrics,
                    limiter,
                )
                .await?;
                Ok(RangeResponse {
                    bytes,
                    etag: None,
                    last_modified: None,
                })
            }
            Self::RemoteAzure {
                target,
                client,
                cfg,
                metrics,
                limiter,
            } => {
                let bytes =
                    fetch_azure_range(client, target, cfg, start, end, metrics, limiter).await?;
                Ok(RangeResponse {
                    bytes,
                    etag: None,
                    last_modified: None,
                })
            }
        }
    }

    /// Revalidate a cached remote HTTP source using `If-None-Match` / `If-Modified-Since` on a
    /// 1-byte `Range: bytes=0-0` GET (avoiding `HEAD`, which some origins forbid).
    /// Returns `Ok(true)` if unchanged, `Ok(false)` if modified and must be re-opened, or `Err`
    /// if the origin reports an error (e.g. `404` when deleted).
    pub async fn revalidate_remote(&self, validator: &RemoteValidator) -> Result<bool, IoError> {
        let Self::RemoteHttp {
            url,
            client,
            metrics,
            limiter,
            max_retries,
        } = self
        else {
            return Ok(false);
        };
        if validator.etag.is_none() && validator.last_modified.is_none() {
            return Ok(false);
        }
        let mut attempt = 0u32;
        loop {
            let permit = limiter.acquire().await;
            metrics.remote_range_requests.fetch_add(1, Ordering::Relaxed);
            let mut req = client.get(url).header("Range", "bytes=0-0");
            if let Some(ref etag) = validator.etag {
                req = req.header("If-None-Match", etag);
            } else if let Some(ref lm) = validator.last_modified {
                req = req.header("If-Modified-Since", lm);
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) if attempt < *max_retries && is_retryable_transport_error(&e) => {
                    permit.on_throttle();
                    tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                    attempt += 1;
                    continue;
                }
                Err(e) => {
                    if is_retryable_transport_error(&e) {
                        permit.on_throttle();
                    }
                    return Err(io_error_from_reqwest(e));
                }
            };
            let status = resp.status().as_u16();
            if status == 304 {
                permit.on_success();
                return Ok(true);
            }
            if status == 200 || status == 206 {
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
                permit.on_success();
                let unchanged = match (&validator.etag, &resp_etag) {
                    (Some(exp), Some(got)) => exp == got,
                    _ => match (&validator.last_modified, &resp_lm) {
                        (Some(exp), Some(got)) => exp == got,
                        _ => false,
                    },
                };
                // Drain the tiny 1-byte body on `206 Partial Content` (for `Range: bytes=0-0`) so
                // reqwest can return the HTTP/1.1 keep-alive connection to the pool instead of
                // tearing it down. Never drain a `200` (origin may have ignored Range on a huge file).
                if status == 206 && resp.content_length().is_some_and(|n| n <= 64) {
                    let _ = resp.bytes().await;
                }
                return Ok(unchanged);
            }
            if is_retryable_status(status) && attempt < *max_retries {
                let retry_after = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.trim().parse::<u64>().ok());
                permit.on_throttle();
                tokio::time::sleep(retry_backoff_duration(attempt, retry_after)).await;
                attempt += 1;
                continue;
            }
            if is_retryable_status(status) {
                permit.on_throttle();
            }
            return Err(IoError::HttpStatus {
                status,
                url: redact_http_url(url),
            });
        }
    }

    /// Zero-copy slice read when backed by local `mmap`.
    #[inline(always)]
    pub fn local_slice(&self, offset: u64, len: u32) -> Option<&[u8]> {
        match self {
            Self::LocalMmap { mmap } => {
                let start = usize::try_from(offset).ok()?;
                let end = start.checked_add(len as usize)?;
                mmap.get(start..end)
            }
            Self::RemoteS3 { .. }
            | Self::RemoteHttp { .. }
            | Self::RemoteGcs { .. }
            | Self::RemoteAzure { .. } => None,
        }
    }
}

/// Validate that a `206 Partial Content` response's `Content-Range` (when present) starts at
/// `expected_start` in `bytes` units and is not `multipart/byteranges`. Prevents misconfigured
/// CDNs or proxies from silently serving offset-0 or misaligned bytes into TIFF/tile caches.
pub(crate) fn validate_content_range_headers(
    content_type: Option<&str>,
    content_range: Option<&str>,
    expected_start: u64,
) -> bool {
    if content_type
        .is_some_and(|s| s.trim_start().to_ascii_lowercase().starts_with("multipart/byteranges"))
    {
        return false;
    }
    let Some(cr) = content_range else {
        return true;
    };
    let s = cr.trim();
    if !s.get(..6).is_some_and(|p| p.eq_ignore_ascii_case("bytes ")) {
        return false;
    }
    let rest = s[6..].trim_start();
    let Some((start_str, _)) = rest.split_once('-') else {
        return false;
    };
    start_str.trim().parse::<u64>().ok() == Some(expected_start)
}

pub(crate) fn validate_content_range_start(resp: &reqwest::Response, expected_start: u64) -> bool {
    let ct = match resp.headers().get(reqwest::header::CONTENT_TYPE) {
        Some(v) => match v.to_str() {
            Ok(s) => Some(s),
            Err(_) => return false,
        },
        None => None,
    };
    let cr = match resp.headers().get(reqwest::header::CONTENT_RANGE) {
        Some(v) => match v.to_str() {
            Ok(s) => Some(s),
            Err(_) => return false,
        },
        None => None,
    };
    validate_content_range_headers(ct, cr, expected_start)
}

/// Extract the numeric start offset from a `Range: bytes=<start>-<end>` header value.
pub(crate) fn parse_range_header_start(range_header: &str) -> Option<u64> {
    let rest = range_header.strip_prefix("bytes=")?;
    let (start_str, _) = rest.split_once('-')?;
    start_str.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::{parse_range_header_start, redact_http_url, validate_content_range_headers};

    #[test]
    fn redact_http_url_strips_credentials_query_and_fragment() {
        assert_eq!(
            redact_http_url("https://user:secret@example.com/a/b.tif?X-Amz-Signature=deadbeef#frag"),
            "https://example.com/a/b.tif"
        );
        assert_eq!(
            redact_http_url("https://example.com/cog.tif?sig=123"),
            "https://example.com/cog.tif"
        );
        assert_eq!(
            redact_http_url("https://example.com/cog.tif"),
            "https://example.com/cog.tif"
        );
    }

    #[test]
    fn content_range_validation_rejects_mismatched_start_and_multipart() {
        assert!(validate_content_range_headers(None, Some("bytes 4096-8191/65536"), 4096));
        assert!(validate_content_range_headers(Some("image/tiff"), Some("BYTES 0-1023/*"), 0));
        assert!(validate_content_range_headers(Some("image/tiff"), None, 4096));
        assert!(!validate_content_range_headers(None, Some("bytes 0-4095/65536"), 4096));
        assert!(!validate_content_range_headers(None, Some("items 4096-8191/65536"), 4096));
        assert!(!validate_content_range_headers(
            Some("multipart/byteranges; boundary=abc"),
            None,
            4096
        ));
        assert_eq!(parse_range_header_start("bytes=262144-524287"), Some(262144));
        assert_eq!(parse_range_header_start("bytes=0-1023"), Some(0));
        assert_eq!(parse_range_header_start("items=0-10"), None);
    }

    #[test]
    fn io_error_is_fatal_identifies_forbidden_through_shared_wrappers() {
        use super::IoError;
        use std::sync::Arc;

        let direct = IoError::Forbidden("https://evil.example/a.tif".into());
        assert!(direct.is_fatal());
        let wrapped = IoError::Shared(Arc::new(IoError::Shared(Arc::new(direct))));
        assert!(wrapped.is_fatal());

        assert!(!IoError::HttpStatus {
            status: 404,
            url: "https://ok.example/missing.tif".into(),
        }
        .is_fatal());
        assert!(!IoError::InvalidUri("ftp://example/a.tif".into()).is_fatal());
        assert!(!IoError::Fs(std::io::Error::new(std::io::ErrorKind::NotFound, "nope")).is_fatal());
    }
}
