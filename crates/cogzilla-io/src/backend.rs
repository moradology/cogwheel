use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use memmap2::Mmap;
use thiserror::Error;

use crate::limiter::AimdLimiter;
use crate::s3::{
    DynamicCredsCache, S3Config, fetch_s3_object_limited, is_retryable_status,
    is_retryable_transport_error, parse_s3_uri, retry_backoff_duration,
};

#[derive(Debug, Error)]
pub enum IoError {
    #[error("filesystem error: {0}")]
    Fs(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("HTTP status {status} fetching {url}")]
    HttpStatus { status: u16, url: String },
    #[error("TIFF parse error: {0}")]
    Tiff(#[from] cogzilla_tiff::TiffError),
    #[error("codec error: {0}")]
    Codec(#[from] cogzilla_codec::CodecError),
    #[error("invalid URI: {0}")]
    InvalidUri(String),
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
        let r_end = r.offset + r.len as u64;
        if let Some(last) = spans.last_mut() {
            let new_end = last.end.max(r_end);
            if r.offset <= last.end + gap_threshold && (new_end - last.start) <= max_span_bytes {
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

fn shared_dynamic_creds() -> Arc<DynamicCredsCache> {
    static CREDS: std::sync::OnceLock<Arc<DynamicCredsCache>> = std::sync::OnceLock::new();
    CREDS
        .get_or_init(|| Arc::new(DynamicCredsCache::default()))
        .clone()
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
        } else if uri.starts_with("http://") || uri.starts_with("https://") {
            let max_retries = std::env::var("GDAL_HTTP_MAX_RETRY")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(3)
                .min(10);
            Ok(Self::RemoteHttp {
                url: uri.to_string(),
                client,
                max_retries,
                metrics,
                limiter,
            })
        } else {
            let local_path = uri.strip_prefix("file://").unwrap_or(uri);
            Self::open_local(local_path)
        }
    }

    /// Fetch a byte range `[start, end)` (exclusive end) from a remote source.
    pub async fn fetch_range(&self, start: u64, end: u64) -> Result<Bytes, IoError> {
        match self {
            Self::LocalMmap { mmap } => {
                let s = (start as usize).min(mmap.len());
                let e = (end as usize).min(mmap.len());
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
                metrics.remote_range_requests.fetch_add(1, Ordering::Relaxed);
                let data = fetch_s3_object_limited(
                    client,
                    bucket,
                    key,
                    s3_cfg,
                    Some(dynamic_creds),
                    Some(&range_hdr),
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
                            return Err(IoError::Http(e));
                        }
                    };
                    let status = resp.status().as_u16();
                    if status == 200 || status == 206 {
                        let data = match resp.bytes().await {
                            Ok(d) => d,
                            Err(e)
                                if attempt < *max_retries && is_retryable_transport_error(&e) =>
                            {
                                permit.on_throttle();
                                tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                                attempt += 1;
                                continue;
                            }
                            Err(e) => {
                                if is_retryable_transport_error(&e) {
                                    permit.on_throttle();
                                }
                                return Err(IoError::Http(e));
                            }
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
                        url: url.clone(),
                    });
                }
            }
        }
    }

    /// Zero-copy slice read when backed by local `mmap`.
    #[inline(always)]
    pub fn local_slice(&self, offset: u64, len: u32) -> Option<&[u8]> {
        match self {
            Self::LocalMmap { mmap } => {
                let start = offset as usize;
                let end = start + len as usize;
                mmap.get(start..end)
            }
            Self::RemoteS3 { .. } | Self::RemoteHttp { .. } => None,
        }
    }
}
