use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use hmac::{Hmac, Mac};
use parking_lot::RwLock;
use sha2::{Digest, Sha256};

use crate::backend::IoError;

/// Resolved AWS credentials (static or temporary STS/IAM role).
#[derive(Debug, Clone)]
pub struct AwsCredentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
    pub expires_at: Option<Instant>,
}

impl AwsCredentials {
    pub fn is_expired_soon(&self) -> bool {
        match self.expires_at {
            Some(exp) => Instant::now() + Duration::from_secs(60) >= exp,
            None => false,
        }
    }
}

/// Complete S3 runtime configuration matching GDAL/VSIS3/TiTiler environment conventions.
#[derive(Debug, Clone)]
pub struct S3Config {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
    pub region: String,
    pub endpoint_url: Option<String>,
    /// When `true` (`AWS_NO_SIGN_REQUEST=YES`), requests are sent unsigned even if credentials exist.
    pub no_sign_request: bool,
    /// When `true` (`AWS_REQUEST_PAYER=requester`), sends & signs `x-amz-request-payer: requester`
    /// (required for Landsat C2 `usgs-landsat`, Sentinel-2 L1C, etc.).
    pub requester_pays: bool,
    /// When `true` (`AWS_VIRTUAL_HOSTING=FALSE` or custom endpoint), uses path-style `/bucket/key`.
    pub force_path_style: bool,
    /// Max transient retries on `429`/`500`/`502`/`503`/`504` (default: 3, matching `GDAL_HTTP_MAX_RETRY`).
    pub max_retries: u32,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            access_key: String::new(),
            secret_key: String::new(),
            session_token: None,
            region: "us-east-1".to_string(),
            endpoint_url: None,
            no_sign_request: false,
            requester_pays: false,
            force_path_style: false,
            max_retries: 3,
        }
    }
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn env_falsy(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
        .unwrap_or(false)
}

impl S3Config {
    /// Load S3 configuration from standard AWS & GDAL (`VSIS3`) environment variables
    /// plus `~/.aws/credentials` fallback.
    ///
    /// Always returns a valid `S3Config` (even for anonymous public buckets with custom endpoints,
    /// regions, or `AWS_NO_SIGN_REQUEST=YES`).
    pub fn from_env_full() -> Self {
        let no_sign_request = env_truthy("AWS_NO_SIGN_REQUEST");
        let requester_pays = std::env::var("AWS_REQUEST_PAYER")
            .map(|v| v.trim().eq_ignore_ascii_case("requester"))
            .unwrap_or(false);
        let region = std::env::var("AWS_REGION")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                std::env::var("AWS_DEFAULT_REGION")
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_else(|| "us-east-1".to_string());

        // Support both standard AWS_ENDPOINT_URL and GDAL's AWS_S3_ENDPOINT
        let endpoint_url = std::env::var("AWS_ENDPOINT_URL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                std::env::var("AWS_S3_ENDPOINT")
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            })
            .map(|ep| {
                if ep.starts_with("http://") || ep.starts_with("https://") {
                    ep
                } else if env_falsy("AWS_HTTPS") {
                    format!("http://{ep}")
                } else {
                    format!("https://{ep}")
                }
            });

        let force_path_style = if env_truthy("AWS_VIRTUAL_HOSTING") {
            false
        } else {
            env_falsy("AWS_VIRTUAL_HOSTING") || endpoint_url.is_some()
        };

        let max_retries = std::env::var("GDAL_HTTP_MAX_RETRY")
            .or_else(|_| std::env::var("AWS_MAX_ATTEMPTS"))
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(3)
            .min(10);

        let (access_key, secret_key, session_token) = if no_sign_request {
            (String::new(), String::new(), None)
        } else if let (Ok(ak), Ok(sk)) = (
            std::env::var("AWS_ACCESS_KEY_ID"),
            std::env::var("AWS_SECRET_ACCESS_KEY"),
        ) {
            (
                ak,
                sk,
                std::env::var("AWS_SESSION_TOKEN")
                    .ok()
                    .filter(|s| !s.is_empty()),
            )
        } else if let Some(creds) = load_shared_credentials_file(None, None) {
            (creds.access_key, creds.secret_key, creds.session_token)
        } else {
            (String::new(), String::new(), None)
        };

        Self {
            access_key,
            secret_key,
            session_token,
            region,
            endpoint_url,
            no_sign_request,
            requester_pays,
            force_path_style,
            max_retries,
        }
    }

    /// Backwards-compatible constructor returning `Some(S3Config)` only when static env credentials exist.
    pub fn from_env() -> Option<Self> {
        let cfg = Self::from_env_full();
        if cfg.has_credentials() || cfg.endpoint_url.is_some() || cfg.requester_pays || cfg.no_sign_request {
            Some(cfg)
        } else {
            None
        }
    }

    #[inline]
    pub fn has_credentials(&self) -> bool {
        !self.no_sign_request && !self.access_key.is_empty() && !self.secret_key.is_empty()
    }
}

/// Parse `~/.aws/credentials` (or `AWS_SHARED_CREDENTIALS_FILE`) for the active `AWS_PROFILE` (default: `default`).
pub fn load_shared_credentials_file(
    explicit_path: Option<&std::path::Path>,
    explicit_profile: Option<&str>,
) -> Option<AwsCredentials> {
    let path: PathBuf = match explicit_path {
        Some(p) => p.to_path_buf(),
        None => {
            if let Ok(p) = std::env::var("AWS_SHARED_CREDENTIALS_FILE") {
                PathBuf::from(p)
            } else {
                let home = std::env::var("HOME").ok()?;
                PathBuf::from(home).join(".aws").join("credentials")
            }
        }
    };
    let profile = explicit_profile
        .map(|s| s.to_string())
        .or_else(|| std::env::var("AWS_PROFILE").ok())
        .or_else(|| std::env::var("AWS_DEFAULT_PROFILE").ok())
        .unwrap_or_else(|| "default".to_string());

    let text = std::fs::read_to_string(path).ok()?;
    parse_aws_credentials_ini(&text, &profile)
}

/// Parse INI-formatted AWS credentials text for `profile`.
pub fn parse_aws_credentials_ini(text: &str, profile: &str) -> Option<AwsCredentials> {
    let mut in_section = false;
    let mut ak: Option<String> = None;
    let mut sk: Option<String> = None;
    let mut tok: Option<String> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let sec = line[1..line.len() - 1].trim();
            let sec_clean = sec.strip_prefix("profile ").unwrap_or(sec).trim();
            in_section = sec_clean == profile;
            continue;
        }
        if !in_section {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim();
            if val.is_empty() {
                continue;
            }
            match key.as_str() {
                "aws_access_key_id" => ak = Some(val.to_string()),
                "aws_secret_access_key" => sk = Some(val.to_string()),
                "aws_session_token" => tok = Some(val.to_string()),
                _ => {}
            }
        }
    }

    Some(AwsCredentials {
        access_key: ak?,
        secret_key: sk?,
        session_token: tok,
        expires_at: None,
    })
}

enum CachedDynamicCreds {
    Creds(AwsCredentials),
    /// Negative-cache failed ECS/IMDSv2 probes so non-EC2 hosts reading public `s3://` buckets
    /// don't serialize every range fetch behind a 500ms timeout.
    Unavailable(Instant),
}

/// Dynamic credential cache supporting ECS container metadata (`AWS_CONTAINER_CREDENTIALS_*`)
/// and EC2 IMDSv2 (`169.254.169.254`), with singleflighted TTL refresh.
pub struct DynamicCredsCache {
    cached: RwLock<Option<CachedDynamicCreds>>,
    refresh_lock: tokio::sync::Mutex<()>,
}

impl Default for DynamicCredsCache {
    fn default() -> Self {
        Self {
            cached: RwLock::new(None),
            refresh_lock: tokio::sync::Mutex::new(()),
        }
    }
}

impl DynamicCredsCache {
    /// Resolve active credentials: returns cached temporary creds if still fresh, otherwise
    /// singleflights an ECS container credentials or EC2 IMDSv2 refresh.
    pub async fn resolve_dynamic(&self, client: &reqwest::Client) -> Option<AwsCredentials> {
        {
            let guard = self.cached.read();
            match guard.as_ref() {
                Some(CachedDynamicCreds::Creds(c)) if !c.is_expired_soon() => {
                    return Some(c.clone());
                }
                Some(CachedDynamicCreds::Unavailable(retry_after))
                    if Instant::now() < *retry_after =>
                {
                    return None;
                }
                _ => {}
            }
        }

        // Singleflight metadata refresh so a burst of concurrent requests never hammers IMDSv2.
        let _refresh_guard = self.refresh_lock.lock().await;
        {
            let guard = self.cached.read();
            match guard.as_ref() {
                Some(CachedDynamicCreds::Creds(c)) if !c.is_expired_soon() => {
                    return Some(c.clone());
                }
                Some(CachedDynamicCreds::Unavailable(retry_after))
                    if Instant::now() < *retry_after =>
                {
                    return None;
                }
                _ => {}
            }
        }

        // 1. ECS Task Role (AWS_CONTAINER_CREDENTIALS_RELATIVE_URI or FULL_URI)
        let ecs_url = match std::env::var("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") {
            Ok(rel) if !rel.trim().is_empty() => Some(format!("http://169.254.170.2{}", rel.trim())),
            _ => std::env::var("AWS_CONTAINER_CREDENTIALS_FULL_URI")
                .ok()
                .filter(|s| !s.trim().is_empty()),
        };

        if let Some(url) = ecs_url {
            if let Some(creds) = fetch_ecs_credentials(client, &url).await {
                *self.cached.write() = Some(CachedDynamicCreds::Creds(creds.clone()));
                return Some(creds);
            }
        }

        // 2. EC2 IMDSv2 (only when not explicitly disabled via AWS_EC2_METADATA_DISABLED=true)
        if !env_truthy("AWS_EC2_METADATA_DISABLED") {
            if let Some(creds) = fetch_imds_v2_credentials(client).await {
                *self.cached.write() = Some(CachedDynamicCreds::Creds(creds.clone()));
                return Some(creds);
            }
        }

        *self.cached.write() = Some(CachedDynamicCreds::Unavailable(
            Instant::now() + Duration::from_secs(60),
        ));
        None
    }
}

pub async fn fetch_ecs_credentials(client: &reqwest::Client, url: &str) -> Option<AwsCredentials> {
    let mut req = client.get(url).timeout(Duration::from_secs(2));
    if let Ok(tok) = std::env::var("AWS_CONTAINER_AUTHORIZATION_TOKEN") {
        req = req.header("Authorization", tok);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body = resp.text().await.ok()?;
    parse_aws_json_credentials(&body)
}

async fn fetch_imds_v2_credentials(client: &reqwest::Client) -> Option<AwsCredentials> {
    let token_resp = client
        .put("http://169.254.169.254/latest/api/token")
        .header("X-aws-ec2-metadata-token-ttl-seconds", "21600")
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .ok()?;
    if !token_resp.status().is_success() {
        return None;
    }
    let imds_token = token_resp.text().await.ok()?;

    let role_resp = client
        .get("http://169.254.169.254/latest/meta-data/iam/security-credentials/")
        .header("X-aws-ec2-metadata-token", &imds_token)
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .ok()?;
    if !role_resp.status().is_success() {
        return None;
    }
    let role_text = role_resp.text().await.ok()?;
    let role = role_text.lines().next()?.trim();
    if role.is_empty() {
        return None;
    }

    let creds_resp = client
        .get(format!(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/{role}"
        ))
        .header("X-aws-ec2-metadata-token", &imds_token)
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .ok()?;
    if !creds_resp.status().is_success() {
        return None;
    }
    let body = creds_resp.text().await.ok()?;
    parse_aws_json_credentials(&body)
}

/// Minimal dependency-free JSON parser for AWS ECS / IMDSv2 credential responses
/// (`AccessKeyId`, `SecretAccessKey`, `Token`, `Expiration`).
pub fn parse_aws_json_credentials(body: &str) -> Option<AwsCredentials> {
    fn extract_str(json: &str, key: &str) -> Option<String> {
        let needle = format!("\"{key}\"");
        let idx = json.find(&needle)?;
        let rest = &json[idx + needle.len()..];
        let colon = rest.find(':')?;
        let after = rest[colon + 1..].trim_start();
        if !after.starts_with('"') {
            return None;
        }
        let inner = &after[1..];
        let mut out = String::new();
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(esc) = chars.next() {
                    out.push(esc);
                }
            } else if c == '"' {
                return Some(out);
            } else {
                out.push(c);
            }
        }
        None
    }

    let access_key = extract_str(body, "AccessKeyId")?;
    let secret_key = extract_str(body, "SecretAccessKey")?;
    let session_token = extract_str(body, "Token").filter(|s| !s.is_empty());
    let ttl_secs = extract_str(body, "Expiration")
        .and_then(|s| parse_iso8601_utc_epoch(&s))
        .and_then(|exp_epoch| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_secs();
            Some(exp_epoch.saturating_sub(now))
        })
        .unwrap_or(900)
        .min(21_600);
    let expires_at = Some(Instant::now() + Duration::from_secs(ttl_secs));
    Some(AwsCredentials {
        access_key,
        secret_key,
        session_token,
        expires_at,
    })
}

/// Minimal ISO-8601 UTC timestamp parser (`YYYY-MM-DDTHH:MM:SSZ`) returning Unix epoch seconds.
fn parse_iso8601_utc_epoch(s: &str) -> Option<u64> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() < 19
        || !bytes[..19].is_ascii()
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return None;
    }
    let year: i64 = s[0..4].parse().ok()?;
    let month: i64 = s[5..7].parse().ok()?;
    let day: i64 = s[8..10].parse().ok()?;
    let hour: u64 = s[11..13].parse().ok()?;
    let min: u64 = s[14..16].parse().ok()?;
    let sec: u64 = s[17..19].parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || min > 59 || sec > 60 {
        return None;
    }
    // Howard Hinnant's civil-from-days inverse (days since 1970-01-01)
    let y = year - if month <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m_adj = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * m_adj + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    if days < 0 {
        return None;
    }
    Some((days as u64) * 86_400 + hour * 3600 + min * 60 + sec)
}

/// Global lock-free cache mapping `bucket -> discovered AWS region` (populated automatically
/// whenever S3 returns `301` or `400` with `x-amz-bucket-region`).
pub fn bucket_region_cache() -> &'static DashMap<String, String, ahash::RandomState> {
    static CACHE: std::sync::OnceLock<DashMap<String, String, ahash::RandomState>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| DashMap::with_hasher(ahash::RandomState::new()))
}

/// RFC 3986 URI-encode an S3 path for SigV4 `canonical_uri` (preserving `/`, encoding spaces,
/// `+`, `:`, `=`, `(`, `)`, `%`, etc. as uppercase `%XX`).
pub fn encode_s3_canonical_uri(path: &str) -> String {
    let mut out = String::with_capacity(path.len() + 16);
    if !path.starts_with('/') {
        out.push('/');
    }
    for &b in path.as_bytes() {
        let unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b'/');
        if unreserved {
            out.push(b as char);
        } else {
            out.push('%');
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0F) as usize] as char);
        }
    }
    out
}

/// Validate an S3 bucket name to prevent SSRF host/query injection (`?`, `#`, `@`, `:`, `%`, `..`).
fn is_valid_s3_bucket(bucket: &str) -> bool {
    let bytes = bucket.as_bytes();
    (3..=63).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes[bytes.len() - 1].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
        && !bucket.contains("..")
}

/// Parse `s3://bucket/key` or `/vsis3/bucket/key` into `(bucket, key)`.
pub fn parse_s3_uri(uri: &str) -> Option<(&str, &str)> {
    let rest = uri
        .strip_prefix("s3://")
        .or_else(|| uri.strip_prefix("/vsis3/"))?;
    let (bucket, key) = rest.split_once('/')?;
    if !is_valid_s3_bucket(bucket)
        || key.is_empty()
        || key.split(['/', '\\']).any(|seg| seg == "..")
    {
        return None;
    }
    Some((bucket, key))
}

/// Validate an AWS region identifier (e.g. `us-west-2`, `eu-central-1`) to block SSRF
/// via spoofed `x-amz-bucket-region` response headers.
fn is_valid_aws_region(region: &str) -> bool {
    let bytes = region.as_bytes();
    (3..=32).contains(&bytes.len())
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && (bytes[bytes.len() - 1].is_ascii_lowercase() || bytes[bytes.len() - 1].is_ascii_digit())
        && bytes
            .iter()
            .all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !region.contains("--")
}

/// Normalize a host[:port] for SigV4 signing (lowercase + strip default `:443` / `:80`).
fn normalize_sigv4_host(scheme: &str, host_str: &str) -> String {
    let lower = host_str.to_ascii_lowercase();
    if scheme.eq_ignore_ascii_case("https") {
        lower.strip_suffix(":443").unwrap_or(&lower).to_string()
    } else if scheme.eq_ignore_ascii_case("http") {
        lower.strip_suffix(":80").unwrap_or(&lower).to_string()
    } else {
        lower
    }
}

/// Build `(url, host, canonical_uri)` for an S3 bucket/key given `region` and `S3Config`.
pub fn build_s3_endpoint(
    bucket: &str,
    key: &str,
    region: &str,
    cfg: &S3Config,
) -> (String, String, String) {
    let encoded_key = encode_s3_canonical_uri(key);
    let bucket_needs_path_style = bucket
        .bytes()
        .any(|b| b == b'.' || b == b'_' || b.is_ascii_uppercase());
    if let Some(ep) = cfg.endpoint_url.as_deref() {
        let ep_clean = ep.trim_end_matches('/');
        let scheme = ep_clean.split("://").next().unwrap_or("https");
        let after_scheme = ep_clean.split("://").nth(1).unwrap_or(ep_clean);
        let (host_str, ep_path) = match after_scheme.find('/') {
            Some(idx) => (&after_scheme[..idx], &after_scheme[idx..]),
            None => (after_scheme, ""),
        };
        let host = normalize_sigv4_host(scheme, host_str);
        if cfg.force_path_style || bucket_needs_path_style {
            let bucket_key = format!("/{bucket}{encoded_key}");
            let canonical_uri = format!("{ep_path}{bucket_key}");
            (format!("{scheme}://{host}{canonical_uri}"), host, canonical_uri)
        } else {
            let vhost = format!("{bucket}.{host}");
            let canonical_uri = format!("{ep_path}{encoded_key}");
            (
                format!("{scheme}://{vhost}{canonical_uri}"),
                vhost,
                canonical_uri,
            )
        }
    } else if cfg.force_path_style || bucket_needs_path_style {
        // Dotted, underscored, or uppercase AWS bucket names fail wildcard TLS validation or
        // SigV4 host normalization under virtual-hosted addressing, so automatically use path-style.
        let host = format!("s3.{region}.amazonaws.com");
        let canonical_uri = format!("/{bucket}{encoded_key}");
        (format!("https://{host}{canonical_uri}"), host, canonical_uri)
    } else {
        let host = format!("{bucket}.s3.{region}.amazonaws.com");
        (format!("https://{host}{encoded_key}"), host, encoded_key)
    }
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

/// Sign an S3 `GET` request using AWS Signature Version 4.
///
/// Properly sorts and signs ALL required headers including `x-amz-security-token` (for STS/IAM roles)
/// and `x-amz-request-payer` (for Requester Pays buckets like Landsat Collection 2), plus optional `Range`.
pub fn sign_s3_get(
    access_key: &str,
    secret_key: &str,
    session_token: Option<&str>,
    region: &str,
    host: &str,
    canonical_uri: &str,
    range_header: Option<&str>,
    requester_pays: bool,
) -> Vec<(&'static str, String)> {
    let now = time::OffsetDateTime::now_utc();
    let date_stamp = format!("{:04}{:02}{:02}", now.year(), now.month() as u8, now.day());
    let amz_date = format!(
        "{date_stamp}T{:02}{:02}{:02}Z",
        now.hour(),
        now.minute(),
        now.second()
    );
    let empty_hash = sha256_hex(b"");

    // BTreeMap guarantees strict lexicographical sorting by lowercase header name (required by SigV4!)
    let mut canon = BTreeMap::<&'static str, String>::new();
    canon.insert("host", host.to_string());
    if let Some(rng) = range_header {
        canon.insert("range", rng.to_string());
    }
    canon.insert("x-amz-content-sha256", empty_hash.clone());
    canon.insert("x-amz-date", amz_date.clone());
    if requester_pays {
        canon.insert("x-amz-request-payer", "requester".to_string());
    }
    if let Some(tok) = session_token {
        canon.insert("x-amz-security-token", tok.to_string());
    }

    let mut canonical_headers = String::with_capacity(256);
    let mut signed_headers = String::with_capacity(128);
    for (i, (k, v)) in canon.iter().enumerate() {
        canonical_headers.push_str(k);
        canonical_headers.push(':');
        canonical_headers.push_str(v.trim());
        canonical_headers.push('\n');
        if i > 0 {
            signed_headers.push(';');
        }
        signed_headers.push_str(k);
    }

    let canonical_request = format!(
        "GET\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{empty_hash}"
    );
    let scope = format!("{date_stamp}/{region}/s3/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    let k_date = hmac_sha256(format!("AWS4{secret_key}").as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"s3");
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    let auth = format!(
        "AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
    );

    let mut out = Vec::with_capacity(6);
    out.push(("Authorization", auth));
    out.push(("x-amz-date", amz_date));
    out.push(("x-amz-content-sha256", empty_hash));
    if requester_pays {
        out.push(("x-amz-request-payer", "requester".to_string()));
    }
    if let Some(tok) = session_token {
        out.push(("x-amz-security-token", tok.to_string()));
    }
    out
}

/// Check if an HTTP status code is transient and safe to retry (`429`, `500`, `502`, `503`, `504`).
#[inline]
pub fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504)
}

/// Transport-level failures worth retrying for an idempotent (ranged) `GET`: connect/read timeouts,
/// connections dropped before or during the response, and truncated or undecodable bodies.
pub fn is_retryable_transport_error(e: &reqwest::Error) -> bool {
    e.is_timeout() || e.is_connect() || e.is_request() || e.is_body() || e.is_decode()
}

/// Compute exponential backoff delay with deterministic jitter (100ms -> 250ms -> 600ms...).
pub fn retry_backoff_duration(attempt: u32, retry_after_secs: Option<u64>) -> Duration {
    if let Some(secs) = retry_after_secs {
        return Duration::from_secs(secs.min(10));
    }
    let base_ms = 100u64.saturating_mul(1u64 << attempt.min(5));
    // Lightweight jitter from current subsec nanos
    let jitter = (time::OffsetDateTime::now_utc().nanosecond() as u64) % 50;
    Duration::from_millis((base_ms + jitter).min(4000))
}

/// Execute a robust GET request against an S3 object (optional `Range` header), handling:
/// - Automatic bucket region discovery & caching on `301` / `400` with `x-amz-bucket-region`
/// - Full SigV4 signing (including `x-amz-security-token` and `x-amz-request-payer`)
/// - Unsigned public access (`AWS_NO_SIGN_REQUEST=YES` or no creds)
/// - Transient `429`/`500`/`502`/`503`/`504` and transport-error retries with exponential backoff
pub async fn fetch_s3_object(
    client: &reqwest::Client,
    bucket: &str,
    key: &str,
    cfg: &S3Config,
    dynamic_creds: Option<&Arc<DynamicCredsCache>>,
    range_header: Option<&str>,
) -> Result<bytes::Bytes, IoError> {
    fetch_s3_object_limited(client, bucket, key, cfg, dynamic_creds, range_header, None).await
}

/// Same as [`fetch_s3_object`], gated per-attempt by an optional [`crate::limiter::AimdLimiter`]
/// so retries release their concurrency permit before sleeping on backoff.
pub async fn fetch_s3_object_limited(
    client: &reqwest::Client,
    bucket: &str,
    key: &str,
    cfg: &S3Config,
    dynamic_creds: Option<&Arc<DynamicCredsCache>>,
    range_header: Option<&str>,
    limiter: Option<&Arc<crate::limiter::AimdLimiter>>,
) -> Result<bytes::Bytes, IoError> {
    fetch_s3_inner(client, bucket, key, cfg, dynamic_creds, range_header, None, limiter).await
}

/// Same as [`fetch_s3_object_limited`], with a hard streaming byte cap (`max_bytes`) to prevent OOM.
#[allow(clippy::too_many_arguments)]
pub async fn fetch_s3_object_capped(
    client: &reqwest::Client,
    bucket: &str,
    key: &str,
    cfg: &S3Config,
    dynamic_creds: Option<&Arc<DynamicCredsCache>>,
    range_header: Option<&str>,
    max_bytes: usize,
    limiter: Option<&Arc<crate::limiter::AimdLimiter>>,
) -> Result<bytes::Bytes, IoError> {
    fetch_s3_inner(
        client,
        bucket,
        key,
        cfg,
        dynamic_creds,
        range_header,
        Some(max_bytes),
        limiter,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn fetch_s3_inner(
    client: &reqwest::Client,
    bucket: &str,
    key: &str,
    cfg: &S3Config,
    dynamic_creds: Option<&Arc<DynamicCredsCache>>,
    range_header: Option<&str>,
    max_bytes: Option<usize>,
    limiter: Option<&Arc<crate::limiter::AimdLimiter>>,
) -> Result<bytes::Bytes, IoError> {
    let mut region = bucket_region_cache()
        .get(bucket)
        .map(|r| r.clone())
        .unwrap_or_else(|| cfg.region.clone());

    let mut region_redirected = false;
    let mut attempt = 0u32;

    loop {
        // Resolve dynamic credentials BEFORE acquiring an AimdLimiter permit so IMDSv2/ECS
        // metadata hops never consume upstream S3 concurrency slots.
        let resolved_dynamic = if !cfg.no_sign_request && !cfg.has_credentials() {
            match dynamic_creds {
                Some(dc) => dc.resolve_dynamic(client).await,
                None => None,
            }
        } else {
            None
        };

        let permit = match limiter {
            Some(l) => Some(l.acquire().await),
            None => None,
        };

        let (url, host, canonical_uri) = build_s3_endpoint(bucket, key, &region, cfg);
        let mut req = client.get(&url);
        if let Some(rng) = range_header {
            req = req.header("Range", rng);
        }

        // Sign request using static config or pre-resolved dynamic ECS/IMDSv2 creds
        if !cfg.no_sign_request {
            if cfg.has_credentials() {
                for (k, v) in sign_s3_get(
                    &cfg.access_key,
                    &cfg.secret_key,
                    cfg.session_token.as_deref(),
                    &region,
                    &host,
                    &canonical_uri,
                    range_header,
                    cfg.requester_pays,
                ) {
                    req = req.header(k, v);
                }
            } else if let Some(ref creds) = resolved_dynamic {
                for (k, v) in sign_s3_get(
                    &creds.access_key,
                    &creds.secret_key,
                    creds.session_token.as_deref(),
                    &region,
                    &host,
                    &canonical_uri,
                    range_header,
                    cfg.requester_pays,
                ) {
                    req = req.header(k, v);
                }
            } else if cfg.requester_pays {
                req = req.header("x-amz-request-payer", "requester");
            }
        } else if cfg.requester_pays {
            req = req.header("x-amz-request-payer", "requester");
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) if attempt < cfg.max_retries && is_retryable_transport_error(&e) => {
                if let Some(p) = permit {
                    p.on_throttle();
                }
                tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                attempt += 1;
                continue;
            }
            Err(e) => {
                if let Some(p) = permit {
                    if is_retryable_transport_error(&e) {
                        p.on_throttle();
                    }
                }
                return Err(IoError::Http(e.without_url()));
            }
        };

        let status = resp.status().as_u16();
        // Reject HTTP 200 when a non-zero start range was requested: the upstream ignored Range
        // and sent bytes from offset 0, which would silently corrupt tile/IFD parsing.
        if status == 200
            && let Some(rng) = range_header
            && !rng.starts_with("bytes=0-")
        {
            return Err(IoError::HttpStatus {
                status: 200,
                url: crate::backend::redact_http_url(&url),
            });
        }
        if status == 200 || status == 206 {
            // The body can still fail after a good status line (connection reset mid-transfer).
            let body_res = if let Some(cap) = max_bytes {
                crate::cloud::read_response_capped(resp, cap).await
            } else {
                resp.bytes().await.map_err(IoError::Http)
            };
            match body_res {
                Ok(body) => {
                    if let Some(p) = permit {
                        p.on_success();
                    }
                    return Ok(body);
                }
                Err(IoError::Http(e))
                    if attempt < cfg.max_retries && is_retryable_transport_error(&e) =>
                {
                    if let Some(p) = permit {
                        p.on_throttle();
                    }
                    tokio::time::sleep(retry_backoff_duration(attempt, None)).await;
                    attempt += 1;
                    continue;
                }
                Err(IoError::Http(e)) => {
                    if let Some(p) = permit {
                        if is_retryable_transport_error(&e) {
                            p.on_throttle();
                        }
                    }
                    return Err(IoError::Http(e.without_url()));
                }
                Err(e) => return Err(e),
            }
        }

        // Automatic S3 bucket region discovery on 301 / 307 / 400 / 403 with `x-amz-bucket-region`.
        // Strictly validate region syntax to prevent SSRF/credential exfiltration via spoofed headers.
        if !region_redirected && matches!(status, 301 | 307 | 400 | 403) {
            if let Some(discovered) = resp
                .headers()
                .get("x-amz-bucket-region")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim().to_string())
                .filter(|s| is_valid_aws_region(s) && s != &region)
            {
                let cache = bucket_region_cache();
                if cache.len() < 4096 {
                    cache.insert(bucket.to_string(), discovered.clone());
                }
                region = discovered;
                region_redirected = true;
                drop(permit);
                continue;
            }
        }

        // Transient S3 retry (429 SlowDown, 500, 502, 503 ServiceUnavailable, 504)
        if attempt < cfg.max_retries && is_retryable_status(status) {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            if let Some(p) = permit {
                p.on_throttle();
            }
            tokio::time::sleep(retry_backoff_duration(attempt, retry_after)).await;
            attempt += 1;
            continue;
        }

        if let Some(p) = permit {
            if is_retryable_status(status) {
                p.on_throttle();
            }
        }
        return Err(IoError::HttpStatus {
            status,
            url: crate::backend::redact_http_url(&url),
        });
    }
}

#[cfg(test)]
mod uri_tests {
    use super::*;

    #[test]
    fn parse_s3_uri_rejects_ssrf_and_invalid_buckets() {
        assert_eq!(parse_s3_uri("s3://sentinel-cogs/path/to/tile.tif"), Some(("sentinel-cogs", "path/to/tile.tif")));
        assert_eq!(parse_s3_uri("/vsis3/my.dotted.bucket/a.tif"), Some(("my.dotted.bucket", "a.tif")));
        for bad in [
            "s3://attacker.example?/k",
            "s3://attacker.example#/k",
            "s3://attacker.example:8443/k",
            "s3://user@attacker.example/k",
            "s3://bad..bucket/k",
            "s3://-leading-dash/k",
            "s3://ab/k",
            "s3://valid-bucket/",
        ] {
            assert!(parse_s3_uri(bad).is_none(), "expected rejection for {bad}");
        }
    }

    #[test]
    fn build_s3_endpoint_uses_path_style_for_dotted_aws_buckets() {
        let cfg = S3Config::default();
        let (url, host, canonical) = build_s3_endpoint("my.dotted.bucket", "a/b.tif", "us-west-2", &cfg);
        assert_eq!(host, "s3.us-west-2.amazonaws.com");
        assert_eq!(canonical, "/my.dotted.bucket/a/b.tif");
        assert_eq!(url, "https://s3.us-west-2.amazonaws.com/my.dotted.bucket/a/b.tif");

        // Underscored/uppercase buckets also use path-style and custom endpoints normalize host/port
        let (_, host_u, _) = build_s3_endpoint("Legacy_Bucket", "a.tif", "us-east-1", &cfg);
        assert_eq!(host_u, "s3.us-east-1.amazonaws.com");
        let ep_cfg = S3Config {
            endpoint_url: Some("https://MinIO.Local:443".into()),
            force_path_style: true,
            ..Default::default()
        };
        let (ep_url, ep_host, _) = build_s3_endpoint("bkt", "a.tif", "us-east-1", &ep_cfg);
        assert_eq!(ep_host, "minio.local");
        assert_eq!(ep_url, "https://minio.local/bkt/a.tif");

        // Region header validation blocks SSRF payloads
        assert!(is_valid_aws_region("us-west-2"));
        assert!(is_valid_aws_region("eu-central-1"));
        for bad in ["attacker.example?", "us-east-1/x#", "US-EAST-1", "a--b", "ab", ""] {
            assert!(!is_valid_aws_region(bad), "expected region rejection for {bad}");
        }
    }
}
