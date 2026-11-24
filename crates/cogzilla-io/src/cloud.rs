use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use bytes::Bytes;
use hmac::{Hmac, Mac};
use parking_lot::Mutex;
use sha2::Sha256;

use crate::backend::{IoError, IoMetrics};
use crate::limiter::AimdLimiter;
use crate::s3::{is_retryable_status, is_retryable_transport_error, retry_backoff_duration};

type HmacSha256 = Hmac<Sha256>;

const AZURE_API_VERSION: &str = "2020-10-02";

/// Validate a GCS bucket name (3..=63 chars, `[a-z0-9._-]`, alphanumeric ends, no `..`).
fn is_valid_gcs_bucket(bucket: &str) -> bool {
    let bytes = bucket.as_bytes();
    (3..=63).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes[bytes.len() - 1].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'-' | b'_'))
        && !bucket.contains("..")
}

/// Validate an Azure storage account name (3..=24 lowercase ASCII alphanumeric chars).
fn is_valid_azure_account(account: &str) -> bool {
    let bytes = account.as_bytes();
    (3..=24).contains(&bytes.len())
        && bytes
            .iter()
            .all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit())
}

/// Validate an Azure blob container name (3..=63 `[a-z0-9-]`, alphanumeric ends, no `--`,
/// plus standard system containers `$web`, `$logs`, `$root`).
fn is_valid_azure_container(container: &str) -> bool {
    if matches!(container, "$web" | "$logs" | "$root") {
        return true;
    }
    let bytes = container.as_bytes();
    (3..=63).contains(&bytes.len())
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && (bytes[bytes.len() - 1].is_ascii_lowercase() || bytes[bytes.len() - 1].is_ascii_digit())
        && bytes
            .iter()
            .all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !container.contains("--")
}

/// Parse `gs://bucket/key` or `/vsigs/bucket/key` into `(bucket, key)`.
pub fn parse_gcs_uri(uri: &str) -> Option<(&str, &str)> {
    let rest = uri
        .strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("/vsigs/"))?;
    let (bucket, key) = rest.split_once('/')?;
    if !is_valid_gcs_bucket(bucket)
        || key.is_empty()
        || key.split(['/', '\\']).any(|seg| seg == "..")
    {
        return None;
    }
    Some((bucket, key))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AzureTarget {
    pub account: String,
    pub container: String,
    pub blob: String,
}

/// Parse `az://container/blob`, `/vsiaz/container/blob`, or
/// `abfs[s]://container@account.dfs.core.windows.net/blob` into an `AzureTarget`.
///
/// For `az://` and `/vsiaz/`, `account` is resolved from `account_override` or
/// `AZURE_STORAGE_ACCOUNT` (or `az://account/container/blob` if `AZURE_STORAGE_ACCOUNT` is unset
/// and 3 segments are present).
pub fn parse_azure_uri(uri: &str, account_override: Option<&str>) -> Option<AzureTarget> {
    if let Some(rest) = uri
        .strip_prefix("abfss://")
        .or_else(|| uri.strip_prefix("abfs://"))
    {
        let (auth, blob) = rest.split_once('/')?;
        let (container, host) = auth.split_once('@')?;
        if !host
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
        {
            return None;
        }
        let account = host.split('.').next()?;
        if !is_valid_azure_account(account)
            || !is_valid_azure_container(container)
            || blob.is_empty()
            || blob.split(['/', '\\']).any(|seg| seg == "..")
        {
            return None;
        }
        return Some(AzureTarget {
            account: account.to_string(),
            container: container.to_string(),
            blob: blob.to_string(),
        });
    }

    let rest = uri
        .strip_prefix("az://")
        .or_else(|| uri.strip_prefix("/vsiaz/"))?;
    let (first, tail) = rest.split_once('/')?;
    if first.is_empty() || tail.is_empty() || tail.split(['/', '\\']).any(|seg| seg == "..") {
        return None;
    }
    if let Some(acct) = account_override.filter(|s| !s.is_empty()) {
        if !is_valid_azure_account(acct) || !is_valid_azure_container(first) {
            return None;
        }
        return Some(AzureTarget {
            account: acct.to_string(),
            container: first.to_string(),
            blob: tail.to_string(),
        });
    }
    if let Ok(acct) = std::env::var("AZURE_STORAGE_ACCOUNT") {
        let acct = acct.trim();
        if !acct.is_empty() {
            if !is_valid_azure_account(acct) || !is_valid_azure_container(first) {
                return None;
            }
            return Some(AzureTarget {
                account: acct.to_string(),
                container: first.to_string(),
                blob: tail.to_string(),
            });
        }
    }
    // Fallback: az://account/container/blob
    let (container, blob) = tail.split_once('/')?;
    if !is_valid_azure_account(first)
        || !is_valid_azure_container(container)
        || blob.is_empty()
    {
        return None;
    }
    Some(AzureTarget {
        account: first.to_string(),
        container: container.to_string(),
        blob: blob.to_string(),
    })
}

enum CachedGcsToken {
    Token(String, Instant),
    /// Negative-cache failed GCE metadata probes so non-GCE hosts don't serialize
    /// every public `gs://` range request behind a 2-second timeout.
    Unavailable(Instant),
}

/// Cached GCE / GKE / Cloud Run metadata OAuth2 bearer token with singleflighted refresh.
#[derive(Default)]
pub struct GcsTokenCache {
    cached: Mutex<Option<CachedGcsToken>>,
    refresh_lock: tokio::sync::Mutex<()>,
}

impl GcsTokenCache {
    async fn get_bearer(&self, client: &reqwest::Client) -> Option<String> {
        if let Ok(tok) = std::env::var("GS_OAUTH2_ACCESS_TOKEN")
            .or_else(|_| std::env::var("GOOGLE_OAUTH_ACCESS_TOKEN"))
        {
            let t = tok.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
        if std::env::var("GCE_METADATA_DISABLED")
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false)
        {
            return None;
        }
        {
            let lock = self.cached.lock();
            match lock.as_ref() {
                Some(CachedGcsToken::Token(tok, exp))
                    if Instant::now() + Duration::from_secs(60) < *exp =>
                {
                    return Some(tok.clone());
                }
                Some(CachedGcsToken::Unavailable(retry_after)) if Instant::now() < *retry_after => {
                    return None;
                }
                _ => {}
            }
        }
        let _refresh_guard = self.refresh_lock.lock().await;
        {
            let lock = self.cached.lock();
            match lock.as_ref() {
                Some(CachedGcsToken::Token(tok, exp))
                    if Instant::now() + Duration::from_secs(60) < *exp =>
                {
                    return Some(tok.clone());
                }
                Some(CachedGcsToken::Unavailable(retry_after)) if Instant::now() < *retry_after => {
                    return None;
                }
                _ => {}
            }
        }
        let fetched = async {
            let resp = client
                .get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
                .header("Metadata-Flavor", "Google")
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .ok()?;
            if !resp.status().is_success() {
                return None;
            }
            let text = resp.text().await.ok()?;
            let tok = extract_json_str(&text, "access_token")?;
            let ttl = extract_json_u64(&text, "expires_in").unwrap_or(300);
            Some((tok, ttl))
        }
        .await;
        match fetched {
            Some((tok, ttl)) => {
                let exp = Instant::now() + Duration::from_secs(ttl);
                *self.cached.lock() = Some(CachedGcsToken::Token(tok.clone(), exp));
                Some(tok)
            }
            None => {
                *self.cached.lock() =
                    Some(CachedGcsToken::Unavailable(Instant::now() + Duration::from_secs(60)));
                None
            }
        }
    }
}

#[derive(Clone)]
pub struct GcsConfig {
    pub endpoint: String,
    pub no_sign: bool,
    pub bearer_token: Option<String>,
    pub user_project: Option<String>,
    pub max_retries: u32,
}

impl GcsConfig {
    pub fn from_env() -> Self {
        let endpoint = std::env::var("GS_ENDPOINT_URL")
            .or_else(|_| std::env::var("STORAGE_EMULATOR_HOST"))
            .ok()
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .map(|s| {
                if s.starts_with("http://") || s.starts_with("https://") {
                    s
                } else {
                    format!("http://{s}")
                }
            })
            .unwrap_or_else(|| "https://storage.googleapis.com".to_string());
        let no_sign = std::env::var("GS_NO_SIGN_REQUEST")
            .map(|v| v.eq_ignore_ascii_case("YES") || v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false);
        let bearer_token = std::env::var("GS_OAUTH2_ACCESS_TOKEN")
            .or_else(|_| std::env::var("GOOGLE_OAUTH_ACCESS_TOKEN"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let user_project = std::env::var("GS_USER_PROJECT")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let max_retries = std::env::var("GDAL_HTTP_MAX_RETRY")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(3)
            .min(10);
        Self {
            endpoint,
            no_sign,
            bearer_token,
            user_project,
            max_retries,
        }
    }
}

#[derive(Clone)]
pub struct AzureConfig {
    pub endpoint_override: Option<String>,
    pub no_sign: bool,
    pub sas_token: Option<String>,
    pub bearer_token: Option<String>,
    pub access_key: Option<String>,
    pub max_retries: u32,
}

impl AzureConfig {
    pub fn from_env() -> Self {
        let endpoint_override = std::env::var("AZURE_STORAGE_ENDPOINT")
            .ok()
            .map(|s| s.trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty());
        let no_sign = std::env::var("AZURE_NO_SIGN_REQUEST")
            .map(|v| v.eq_ignore_ascii_case("YES") || v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false);
        let sas_token = std::env::var("AZURE_STORAGE_SAS_TOKEN")
            .ok()
            .map(|s| s.trim().trim_start_matches('?').to_string())
            .filter(|s| !s.is_empty());
        let bearer_token = std::env::var("AZURE_STORAGE_ACCESS_TOKEN")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let access_key = std::env::var("AZURE_STORAGE_ACCESS_KEY")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let max_retries = std::env::var("GDAL_HTTP_MAX_RETRY")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(3)
            .min(10);
        Self {
            endpoint_override,
            no_sign,
            sas_token,
            bearer_token,
            access_key,
            max_retries,
        }
    }
}

/// Stream a `reqwest::Response` body up to `max_bytes`, failing cleanly before OOM if exceeded.
pub async fn read_response_capped(
    mut resp: reqwest::Response,
    max_bytes: usize,
) -> Result<Bytes, IoError> {
    if let Some(cl) = resp.content_length()
        && cl > max_bytes as u64
    {
        return Err(IoError::Fs(std::io::Error::other(format!(
            "remote response Content-Length {cl} exceeds limit of {max_bytes} bytes"
        ))));
    }
    let init = resp
        .content_length()
        .map(|n| (n as usize).min(max_bytes).min(1024 * 1024))
        .unwrap_or(0);
    let mut out = bytes::BytesMut::with_capacity(init);
    while let Some(chunk) = resp.chunk().await.map_err(|e| IoError::Http(e.without_url()))? {
        if out.len().saturating_add(chunk.len()) > max_bytes {
            return Err(IoError::Fs(std::io::Error::other(format!(
                "remote response exceeded limit of {max_bytes} bytes"
            ))));
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out.freeze())
}

/// Fetch a byte range `[start, end)` from Google Cloud Storage with AIMD concurrency control and retry.
pub async fn fetch_gcs_range(
    client: &reqwest::Client,
    bucket: &str,
    key: &str,
    cfg: &GcsConfig,
    token_cache: &GcsTokenCache,
    start: u64,
    end: u64,
    metrics: &IoMetrics,
    limiter: &AimdLimiter,
) -> Result<Bytes, IoError> {
    fetch_gcs_inner(
        client,
        bucket,
        key,
        cfg,
        token_cache,
        Some((start, end)),
        None,
        metrics,
        limiter,
    )
    .await
}

/// Fetch a full object from Google Cloud Storage with AIMD concurrency control, retry, and a hard byte cap.
pub async fn fetch_gcs_object_capped(
    client: &reqwest::Client,
    bucket: &str,
    key: &str,
    cfg: &GcsConfig,
    token_cache: &GcsTokenCache,
    max_bytes: usize,
    metrics: &IoMetrics,
    limiter: &AimdLimiter,
) -> Result<Bytes, IoError> {
    fetch_gcs_inner(
        client,
        bucket,
        key,
        cfg,
        token_cache,
        None,
        Some(max_bytes),
        metrics,
        limiter,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn fetch_gcs_inner(
    client: &reqwest::Client,
    bucket: &str,
    key: &str,
    cfg: &GcsConfig,
    token_cache: &GcsTokenCache,
    range: Option<(u64, u64)>,
    max_bytes: Option<usize>,
    metrics: &IoMetrics,
    limiter: &AimdLimiter,
) -> Result<Bytes, IoError> {
    let url = format!("{}/{}/{}", cfg.endpoint, bucket, encode_path(key));
    let range_hdr = range.map(|(start, end)| format!("bytes={start}-{}", end.saturating_sub(1)));
    let bearer = if cfg.no_sign {
        None
    } else if let Some(t) = &cfg.bearer_token {
        Some(t.clone())
    } else {
        token_cache.get_bearer(client).await
    };

    for attempt in 0..=cfg.max_retries {
        let permit = limiter.acquire().await;
        metrics.remote_range_requests.fetch_add(1, Ordering::Relaxed);

        let mut req = client.get(&url);
        if let Some(rh) = &range_hdr {
            req = req.header("Range", rh);
        }
        if let Some(tok) = &bearer {
            req = req.header("Authorization", format!("Bearer {tok}"));
        }
        if let Some(proj) = &cfg.user_project {
            req = req.header("x-goog-user-project", proj);
        }

        let send_res = req.send().await;
        let resp = match send_res {
            Ok(r) => r,
            Err(e) => {
                if is_retryable_transport_error(&e) {
                    permit.on_throttle();
                    if attempt < cfg.max_retries {
                        tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                        continue;
                    }
                }
                return Err(IoError::Http(e));
            }
        };
        let status = resp.status().as_u16();
        if is_retryable_status(status) {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok());
            permit.on_throttle();
            if attempt < cfg.max_retries {
                tokio::time::sleep(retry_backoff_duration(attempt, retry_after)).await;
                continue;
            }
            return Err(IoError::HttpStatus { status, url });
        }
        if status == 200
            && let Some((start, _)) = range
            && start > 0
        {
            return Err(IoError::HttpStatus { status: 200, url });
        }
        if status != 200 && status != 206 {
            return Err(IoError::HttpStatus { status, url });
        }
        let eff_cap = max_bytes.or_else(|| {
            range.map(|(s, e)| e.saturating_sub(s).min(64 * 1024 * 1024) as usize)
        });
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
                return Ok(data);
            }
            Err(IoError::Http(e)) => {
                if is_retryable_transport_error(&e) {
                    permit.on_throttle();
                    if attempt < cfg.max_retries {
                        tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                        continue;
                    }
                }
                return Err(IoError::Http(e));
            }
            Err(e) => return Err(e),
        }
    }
    Err(IoError::HttpStatus { status: 503, url })
}

/// Compute Azure Blob SharedKey `Authorization` header (`SharedKey {account}:{b64_hmac_sha256}`).
pub fn sign_azure_shared_key(
    account: &str,
    container: &str,
    blob: &str,
    range_hdr: &str,
    x_ms_date: &str,
    access_key_b64: &str,
) -> Option<String> {
    let key_bytes = base64_decode(access_key_b64)?;
    // CanonicalizedHeaders (sorted lowercase) + CanonicalizedResource
    let canonical_headers = format!("x-ms-date:{x_ms_date}\nx-ms-version:{AZURE_API_VERSION}\n");
    let canonical_resource = format!("/{account}/{container}/{}", encode_path(blob));
    // Azure Blob SharedKey StringToSign (GET, empty Content-Length, Range in slot 11):
    // VERB\nContent-Encoding\nContent-Language\nContent-Length\nContent-MD5\nContent-Type\nDate\nIf-Modified-Since\nIf-Match\nIf-None-Match\nIf-Unmodified-Since\nRange\nCanonicalizedHeaders + CanonicalizedResource
    let string_to_sign = format!(
        "GET\n\n\n\n\n\n\n\n\n\n\n{range_hdr}\n{canonical_headers}{canonical_resource}"
    );
    let mut mac = HmacSha256::new_from_slice(&key_bytes).ok()?;
    mac.update(string_to_sign.as_bytes());
    let sig = base64_encode(&mac.finalize().into_bytes());
    Some(format!("SharedKey {account}:{sig}"))
}

/// Fetch a byte range `[start, end)` from Azure Blob Storage with AIMD concurrency control and retry.
pub async fn fetch_azure_range(
    client: &reqwest::Client,
    target: &AzureTarget,
    cfg: &AzureConfig,
    start: u64,
    end: u64,
    metrics: &IoMetrics,
    limiter: &AimdLimiter,
) -> Result<Bytes, IoError> {
    fetch_azure_inner(client, target, cfg, Some((start, end)), None, metrics, limiter).await
}

/// Fetch a full object from Azure Blob Storage with AIMD concurrency control, retry, and a hard byte cap.
pub async fn fetch_azure_object_capped(
    client: &reqwest::Client,
    target: &AzureTarget,
    cfg: &AzureConfig,
    max_bytes: usize,
    metrics: &IoMetrics,
    limiter: &AimdLimiter,
) -> Result<Bytes, IoError> {
    fetch_azure_inner(client, target, cfg, None, Some(max_bytes), metrics, limiter).await
}

async fn fetch_azure_inner(
    client: &reqwest::Client,
    target: &AzureTarget,
    cfg: &AzureConfig,
    range: Option<(u64, u64)>,
    max_bytes: Option<usize>,
    metrics: &IoMetrics,
    limiter: &AimdLimiter,
) -> Result<Bytes, IoError> {
    let base = cfg
        .endpoint_override
        .clone()
        .unwrap_or_else(|| format!("https://{}.blob.core.windows.net", target.account));
    let clean_url = format!("{}/{}/{}", base, target.container, encode_path(&target.blob));
    let mut url = clean_url.clone();
    if !cfg.no_sign {
        if let Some(sas) = &cfg.sas_token {
            url.push('?');
            url.push_str(sas);
        }
    }
    let range_hdr = range.map(|(start, end)| format!("bytes={start}-{}", end.saturating_sub(1)));

    for attempt in 0..=cfg.max_retries {
        let permit = limiter.acquire().await;
        metrics.remote_range_requests.fetch_add(1, Ordering::Relaxed);

        let mut req = client.get(&url).header("x-ms-version", AZURE_API_VERSION);
        if let Some(rh) = &range_hdr {
            req = req.header("Range", rh);
        }

        if !cfg.no_sign && cfg.sas_token.is_none() {
            if let Some(tok) = &cfg.bearer_token {
                req = req.header("Authorization", format!("Bearer {tok}"));
            } else if let Some(key) = &cfg.access_key {
                let date_hdr = format_rfc1123_now();
                req = req.header("x-ms-date", &date_hdr);
                let Some(auth) = sign_azure_shared_key(
                    &target.account,
                    &target.container,
                    &target.blob,
                    range_hdr.as_deref().unwrap_or(""),
                    &date_hdr,
                    key,
                ) else {
                    return Err(IoError::InvalidUri(
                        "invalid base64 AZURE_STORAGE_ACCESS_KEY".to_string(),
                    ));
                };
                req = req.header("Authorization", auth);
            }
        }

        let send_res = req.send().await;
        let resp = match send_res {
            Ok(r) => r,
            Err(e) => {
                if is_retryable_transport_error(&e) {
                    permit.on_throttle();
                    if attempt < cfg.max_retries {
                        tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                        continue;
                    }
                }
                return Err(IoError::Http(e.without_url()));
            }
        };
        let status = resp.status().as_u16();
        if is_retryable_status(status) {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok());
            permit.on_throttle();
            if attempt < cfg.max_retries {
                tokio::time::sleep(retry_backoff_duration(attempt, retry_after)).await;
                continue;
            }
            return Err(IoError::HttpStatus {
                status,
                url: clean_url,
            });
        }
        if status == 200
            && let Some((start, _)) = range
            && start > 0
        {
            return Err(IoError::HttpStatus {
                status: 200,
                url: clean_url,
            });
        }
        if status != 200 && status != 206 {
            return Err(IoError::HttpStatus {
                status,
                url: clean_url,
            });
        }
        let eff_cap = max_bytes.or_else(|| {
            range.map(|(s, e)| e.saturating_sub(s).min(64 * 1024 * 1024) as usize)
        });
        let body_res = if let Some(cap) = eff_cap {
            read_response_capped(resp, cap).await
        } else {
            resp.bytes().await.map_err(|e| IoError::Http(e.without_url()))
        };
        match body_res {
            Ok(data) => {
                permit.on_success();
                metrics
                    .remote_bytes_fetched
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                return Ok(data);
            }
            Err(IoError::Http(e)) => {
                if is_retryable_transport_error(&e) {
                    permit.on_throttle();
                    if attempt < cfg.max_retries {
                        tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                        continue;
                    }
                }
                return Err(IoError::Http(e.without_url()));
            }
            Err(e) => return Err(e),
        }
    }
    Err(IoError::HttpStatus {
        status: 503,
        url: clean_url,
    })
}

fn encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for &b in path.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn format_rfc1123_now() -> String {
    let now = time::OffsetDateTime::now_utc();
    let fmt = time::format_description::well_known::Rfc2822;
    // RFC2822 ends with "+0000"; Azure x-ms-date prefers "GMT".
    now.format(&fmt)
        .unwrap_or_else(|_| "Thu, 01 Jan 1970 00:00:00 GMT".to_string())
        .replace("+0000", "GMT")
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let after = after.strip_prefix('"')?;
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let num: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

const B64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_TABLE[((n >> 18) & 63) as usize] as char);
        out.push(B64_TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            B64_TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut vals = Vec::with_capacity(input.len());
    for b in input.bytes() {
        if b == b'=' || b.is_ascii_whitespace() {
            continue;
        }
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        vals.push(v);
    }
    let mut out = Vec::with_capacity((vals.len() * 3) / 4);
    for chunk in vals.chunks(4) {
        if chunk.len() < 2 {
            break;
        }
        out.push((chunk[0] << 2) | (chunk[1] >> 4));
        if chunk.len() > 2 {
            out.push((chunk[1] << 4) | (chunk[2] >> 2));
        }
        if chunk.len() > 3 {
            out.push((chunk[2] << 6) | chunk[3]);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::Full;
    use hyper::service::service_fn;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder as ServerBuilder;
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use tokio::net::TcpListener;

    #[test]
    fn parses_gcs_and_azure_uris() {
        assert_eq!(parse_gcs_uri("gs://my-bkt/a/b.tif"), Some(("my-bkt", "a/b.tif")));
        assert_eq!(parse_gcs_uri("/vsigs/my-bkt/a/b.tif"), Some(("my-bkt", "a/b.tif")));
        assert_eq!(parse_gcs_uri("s3://my-bkt/a.tif"), None);

        let az1 = parse_azure_uri("az://cont/dir/cog.tif", Some("myacct")).unwrap();
        assert_eq!(az1.account, "myacct");
        assert_eq!(az1.container, "cont");
        assert_eq!(az1.blob, "dir/cog.tif");

        let az2 = parse_azure_uri(
            "abfss://sentinel2@pcdata.dfs.core.windows.net/tiles/TCI.tif",
            None,
        )
        .unwrap();
        assert_eq!(az2.account, "pcdata");
        assert_eq!(az2.container, "sentinel2");
        assert_eq!(az2.blob, "tiles/TCI.tif");

        // Verify base64 round-trip and SharedKey signing
        let key = base64_encode(b"0123456789abcdef0123456789abcdef");
        let auth = sign_azure_shared_key(
            "myacct",
            "cont",
            "cog.tif",
            "bytes=0-15",
            "Thu, 01 Jan 1970 00:00:00 GMT",
            &key,
        )
        .unwrap();
        assert!(auth.starts_with("SharedKey myacct:"));
    }

    #[tokio::test]
    async fn gcs_and_azure_range_fetches_retry_503_and_send_auth_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let gcs_attempts = Arc::new(AtomicUsize::new(0));
        let gcs_att_srv = gcs_attempts.clone();

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let io = TokioIo::new(stream);
                let att = gcs_att_srv.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<hyper::body::Incoming>| {
                        let att = att.clone();
                        async move {
                            let path = req.uri().path().to_string();
                            let query = req.uri().query().unwrap_or("").to_string();
                            let range = req
                                .headers()
                                .get("range")
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or("");
                            assert_eq!(range, "bytes=0-7");

                            if path == "/gcs-bkt/scene.tif" {
                                let auth = req
                                    .headers()
                                    .get("authorization")
                                    .and_then(|v| v.to_str().ok())
                                    .unwrap_or("");
                                assert_eq!(auth, "Bearer gcs-secret-token");
                                let proj = req
                                    .headers()
                                    .get("x-goog-user-project")
                                    .and_then(|v| v.to_str().ok())
                                    .unwrap_or("");
                                assert_eq!(proj, "my-gcp-proj");
                                if att.fetch_add(1, Ordering::SeqCst) == 0 {
                                    let mut r = Response::new(Full::new(Bytes::from_static(b"slow")));
                                    *r.status_mut() = StatusCode::SERVICE_UNAVAILABLE;
                                    return Ok::<_, Infallible>(r);
                                }
                                let mut r = Response::new(Full::new(Bytes::from_static(b"GCS_DATA")));
                                *r.status_mut() = StatusCode::PARTIAL_CONTENT;
                                return Ok(r);
                            }

                            if path == "/az-cont/scene.tif" {
                                assert_eq!(query, "sv=2020-10-02&sig=test");
                                assert_eq!(
                                    req.headers()
                                        .get("x-ms-version")
                                        .and_then(|v| v.to_str().ok()),
                                    Some(AZURE_API_VERSION)
                                );
                                let mut r = Response::new(Full::new(Bytes::from_static(b"AZR_DATA")));
                                *r.status_mut() = StatusCode::PARTIAL_CONTENT;
                                return Ok(r);
                            }

                            let mut r = Response::new(Full::new(Bytes::new()));
                            *r.status_mut() = StatusCode::NOT_FOUND;
                            Ok(r)
                        }
                    });
                    let _ = ServerBuilder::new(TokioExecutor::new())
                        .serve_connection(io, svc)
                        .await;
                });
            }
        });

        let client = reqwest::Client::new();
        let metrics = IoMetrics::default();
        let limiter = AimdLimiter::new(16, 4);

        let gcs_cfg = GcsConfig {
            endpoint: format!("http://{addr}"),
            no_sign: false,
            bearer_token: Some("gcs-secret-token".into()),
            user_project: Some("my-gcp-proj".into()),
            max_retries: 2,
        };
        let gcs_bytes = fetch_gcs_range(
            &client,
            "gcs-bkt",
            "scene.tif",
            &gcs_cfg,
            &GcsTokenCache::default(),
            0,
            8,
            &metrics,
            &limiter,
        )
        .await
        .unwrap();
        assert_eq!(gcs_bytes.as_ref(), b"GCS_DATA");
        assert_eq!(gcs_attempts.load(Ordering::SeqCst), 2);
        assert_eq!(limiter.throttle_events(), 1);

        let az_cfg = AzureConfig {
            endpoint_override: Some(format!("http://{addr}")),
            no_sign: false,
            sas_token: Some("sv=2020-10-02&sig=test".into()),
            bearer_token: None,
            access_key: None,
            max_retries: 1,
        };
        let az_target = AzureTarget {
            account: "pcdata".into(),
            container: "az-cont".into(),
            blob: "scene.tif".into(),
        };
        let az_bytes = fetch_azure_range(&client, &az_target, &az_cfg, 0, 8, &metrics, &limiter)
            .await
            .unwrap();
        assert_eq!(az_bytes.as_ref(), b"AZR_DATA");
    }

    #[test]
    fn parse_gcs_and_azure_uri_reject_ssrf_and_invalid_names() {
        assert_eq!(parse_gcs_uri("gs://gcp-public-data-landsat/a.tif"), Some(("gcp-public-data-landsat", "a.tif")));
        for bad in [
            "gs://attacker.example?/k",
            "gs://bucket#frag/k",
            "gs://bad..bucket/k",
            "gs://UPPERCASE/k",
            "gs://ab/k",
        ] {
            assert!(parse_gcs_uri(bad).is_none(), "expected GCS rejection for {bad}");
        }

        let ok = parse_azure_uri("abfss://sentinel2@pcdata.dfs.core.windows.net/a/b.tif", None).unwrap();
        assert_eq!(ok.account, "pcdata");
        assert_eq!(ok.container, "sentinel2");
        assert_eq!(ok.blob, "a/b.tif");

        for bad in [
            "az://attacker.example?/c/b",
            "az://attacker.example#/c/b",
            "az://bad_acct/cont/b",
            "az://validacct/bad?cont/b",
            "abfss://cont@internal:8443?/b",
            "abfss://cont@attacker.example?/b",
            "abfss://bad..cont@pcdata.dfs.core.windows.net/b",
        ] {
            assert!(parse_azure_uri(bad, None).is_none(), "expected Azure rejection for {bad}");
        }
    }
}
