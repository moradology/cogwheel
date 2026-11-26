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
}

#[derive(Default)]
pub struct IoMetrics {
    pub remote_range_requests: AtomicU64,
    pub remote_bytes_fetched: AtomicU64,
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

/// Strip `file://` for local paths, or reject any unrecognized `<scheme>://` URI with
/// [`IoError::InvalidUri`] naming the unsupported scheme and the supported ones.
pub(crate) fn resolve_local_or_reject_scheme(uri: &str) -> Result<&str, IoError> {
    if let Some(local) = uri.strip_prefix("file://") {
        return Ok(local);
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
        if start >= end {
            return Ok(Bytes::new());
        }
        match self {
            Self::LocalMmap { mmap } => {
                let s = usize::try_from(start).unwrap_or(usize::MAX).min(mmap.len());
                let e = usize::try_from(end).unwrap_or(usize::MAX).min(mmap.len());
                Ok(Bytes::copy_from_slice(&mmap[s..e]))
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
                metrics.remote_range_requests.fetch_add(1, Ordering::Relaxed);
                let data = crate::s3::fetch_s3_object_capped(
                    client,
                    bucket,
                    key,
                    s3_cfg,
                    Some(dynamic_creds),
                    Some(&range_hdr),
                    expected_len,
                    Some(limiter),
                )
                .await?;
                metrics
                    .remote_bytes_fetched
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                Ok(data)
            }
            Self::RemoteHttp {
                url,
                client,
                max_retries,
                metrics,
                limiter,
            } => {
                let range_hdr = format!("bytes={start}-{}", end.saturating_sub(1));
                let mut attempt = 0u32;
                loop {
                    let permit = limiter.acquire().await;
                    metrics.remote_range_requests.fetch_add(1, Ordering::Relaxed);
                    let resp = match client.get(url).header("Range", &range_hdr).send().await {
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
                            return Err(IoError::Http(e.without_url()));
                        }
                    };
                    let status = resp.status().as_u16();
                    let expected_len = end.saturating_sub(start).min(64 * 1024 * 1024) as usize;
                    if status == 200
                        && (start > 0
                            || resp
                                .content_length()
                                .is_some_and(|cl| cl > expected_len as u64))
                    {
                        // Server ignored Range header and sent full object:
                        // returning those bytes would silently corrupt tile/IFD data or OOM.
                        return Err(IoError::RangeNotSupported {
                            url: redact_http_url(url),
                        });
                    }
                    if status == 200 || status == 206 {
                        let data = match crate::cloud::read_response_capped(resp, expected_len).await
                        {
                            Ok(d) => d,
                            Err(IoError::ResponseTooLarge { .. }) if status == 200 => {
                                return Err(IoError::RangeNotSupported {
                                    url: redact_http_url(url),
                                });
                            }
                            Err(IoError::Http(e))
                                if attempt < *max_retries && is_retryable_transport_error(&e) =>
                            {
                                permit.on_throttle();
                                tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                                attempt += 1;
                                continue;
                            }
                            Err(IoError::Http(e)) => {
                                if is_retryable_transport_error(&e) {
                                    permit.on_throttle();
                                }
                                return Err(IoError::Http(e.without_url()));
                            }
                            Err(e) => return Err(e),
                        };
                        permit.on_success();
                        metrics
                            .remote_bytes_fetched
                            .fetch_add(data.len() as u64, Ordering::Relaxed);
                        return Ok(data);
                    }
                    if attempt < *max_retries && is_retryable_status(status) {
                        let retry_after = resp
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.parse::<u64>().ok());
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
            Self::RemoteGcs {
                bucket,
                key,
                client,
                cfg,
                token_cache,
                metrics,
                limiter,
            } => {
                fetch_gcs_range(
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
                .await
            }
            Self::RemoteAzure {
                target,
                client,
                cfg,
                metrics,
                limiter,
            } => fetch_azure_range(client, target, cfg, start, end, metrics, limiter).await,
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

#[cfg(test)]
mod tests {
    use super::redact_http_url;

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
}
