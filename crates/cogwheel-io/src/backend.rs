use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use hmac::{Hmac, Mac};
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("filesystem error: {0}")]
    Fs(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("HTTP status {status} fetching {url}")]
    HttpStatus { status: u16, url: String },
    #[error("TIFF parse error: {0}")]
    Tiff(#[from] cogwheel_tiff::TiffError),
    #[error("codec error: {0}")]
    Codec(#[from] cogwheel_codec::CodecError),
    #[error("invalid URI: {0}")]
    InvalidUri(String),
}

#[derive(Debug, Clone)]
pub struct S3Config {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
    pub region: String,
    pub endpoint_url: Option<String>,
}

impl S3Config {
    pub fn from_env() -> Option<Self> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").ok()?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok()?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());
        let endpoint_url = std::env::var("AWS_ENDPOINT_URL").ok();
        Some(Self {
            access_key,
            secret_key,
            session_token,
            region,
            endpoint_url,
        })
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
    RemoteHttp {
        url: String,
        host: String,
        canonical_uri: String,
        client: reqwest::Client,
        s3_auth: Option<S3Config>,
        metrics: Arc<IoMetrics>,
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

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC key");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn sign_s3_get(
    cfg: &S3Config,
    host: &str,
    canonical_uri: &str,
    range_header: &str,
) -> HashMap<&'static str, String> {
    let now = time::OffsetDateTime::now_utc();
    let date_stamp = format!("{:04}{:02}{:02}", now.year(), now.month() as u8, now.day());
    let amz_date = format!(
        "{date_stamp}T{:02}{:02}{:02}Z",
        now.hour(),
        now.minute(),
        now.second()
    );
    let empty_hash = sha256_hex(b"");

    let canonical_headers = format!(
        "host:{host}\nrange:{range_header}\nx-amz-content-sha256:{empty_hash}\nx-amz-date:{amz_date}\n"
    );
    let signed_headers = "host;range;x-amz-content-sha256;x-amz-date";
    let canonical_request = format!(
        "GET\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{empty_hash}"
    );
    let scope = format!("{date_stamp}/{}/s3/aws4_request", cfg.region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    let k_date = hmac_sha256(format!("AWS4{}", cfg.secret_key).as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, cfg.region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"s3");
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    let auth = format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
        cfg.access_key
    );

    let mut headers = HashMap::new();
    headers.insert("Authorization", auth);
    headers.insert("x-amz-date", amz_date);
    headers.insert("x-amz-content-sha256", empty_hash);
    if let Some(tok) = &cfg.session_token {
        headers.insert("x-amz-security-token", tok.clone());
    }
    headers
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
        if let Some(rest) = uri.strip_prefix("s3://") {
            let (bucket, key) = rest
                .split_once('/')
                .ok_or_else(|| IoError::InvalidUri(uri.to_string()))?;
            let cfg = s3_cfg.or_else(S3Config::from_env);
            let region = cfg
                .as_ref()
                .map(|c| c.region.as_str())
                .unwrap_or("us-east-1");
            let (url, host, canonical_uri) = if let Some(ep) =
                cfg.as_ref().and_then(|c| c.endpoint_url.as_deref())
            {
                let ep_clean = ep.trim_end_matches('/');
                let host = ep_clean
                    .split("://")
                    .nth(1)
                    .unwrap_or(ep_clean)
                    .to_string();
                let path = format!("/{bucket}/{key}");
                (format!("{ep_clean}{path}"), host, path)
            } else {
                let host = format!("{bucket}.s3.{region}.amazonaws.com");
                let path = format!("/{key}");
                (format!("https://{host}{path}"), host, path)
            };
            Ok(Self::RemoteHttp {
                url,
                host,
                canonical_uri,
                client,
                s3_auth: cfg,
                metrics,
            })
        } else if uri.starts_with("http://") || uri.starts_with("https://") {
            let after_scheme = uri.split("://").nth(1).unwrap_or("");
            let (host, path) = match after_scheme.find('/') {
                Some(idx) => (&after_scheme[..idx], &after_scheme[idx..]),
                None => (after_scheme, "/"),
            };
            Ok(Self::RemoteHttp {
                url: uri.to_string(),
                host: host.to_string(),
                canonical_uri: path.to_string(),
                client,
                s3_auth: None,
                metrics,
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
            Self::RemoteHttp {
                url,
                host,
                canonical_uri,
                client,
                s3_auth,
                metrics,
            } => {
                let range_hdr = format!("bytes={start}-{}", end.saturating_sub(1));
                let mut req = client.get(url).header("Range", &range_hdr);
                if let Some(cfg) = s3_auth {
                    for (k, v) in sign_s3_get(cfg, host, canonical_uri, &range_hdr) {
                        req = req.header(k, v);
                    }
                }
                metrics.remote_range_requests.fetch_add(1, Ordering::Relaxed);
                let resp = req.send().await?;
                let status = resp.status().as_u16();
                if status != 200 && status != 206 {
                    return Err(IoError::HttpStatus {
                        status,
                        url: url.clone(),
                    });
                }
                let data = resp.bytes().await?;
                metrics
                    .remote_bytes_fetched
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                Ok(data)
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
            Self::RemoteHttp { .. } => None,
        }
    }
}
