//! Optional allowlist for user-supplied source locations.
//!
//! By default the server opens any `url=` it is handed: remote objects, local
//! paths, anything the process can reach. That is right for a private
//! deployment and wrong for a public one, where it makes the server an open
//! proxy onto its own filesystem and network. Setting
//! `COGZILLA_ALLOWED_URL_PREFIXES` to a comma-separated list of prefixes
//! restricts every dynamically opened location to that list. Datasets
//! registered at startup (`--dataset`, `--scene`, `--mosaic`) are not affected.

use std::sync::OnceLock;

use crate::backend::IoError;

pub const ALLOWED_URL_PREFIXES_ENV: &str = "COGZILLA_ALLOWED_URL_PREFIXES";
pub const ALLOWLIST_RESOLVE_SYMLINKS_ENV: &str = "COGZILLA_ALLOWLIST_RESOLVE_SYMLINKS";

/// Normalize `file:///path` and `file://localhost/path` to `/path` (rejecting non-local hosts
/// like `file://evil/share`). Returns `Ok(Some(path))` if `uri` is a local `file://` URI,
/// `Ok(None)` if `uri` does not start with `file://`, or `Err(())` for a non-localhost `file://` host.
pub fn strip_file_uri_prefix(uri: &str) -> Result<Option<&str>, ()> {
    let Some(rest) = uri.strip_prefix("file://") else {
        return Ok(None);
    };
    if rest.starts_with("localhost/") {
        return Ok(Some(&rest["localhost".len()..]));
    }
    if rest == "localhost" {
        return Ok(Some("/"));
    }
    if rest.starts_with('/') {
        return Ok(Some(rest));
    }
    Err(())
}

fn resolve_symlinks_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(ALLOWLIST_RESOLVE_SYMLINKS_ENV)
            .map(|v| {
                let s = v.trim();
                s == "1" || s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("yes")
            })
            .unwrap_or(false)
    })
}

fn is_vsi_uri(uri: &str) -> bool {
    uri.starts_with("/vsicurl/")
        || uri.starts_with("/vsis3/")
        || uri.starts_with("/vsigs/")
        || uri.starts_with("/vsiaz/")
}

fn is_local_candidate(uri: &str) -> bool {
    !uri.contains("://") && !is_vsi_uri(uri)
}

fn parse_prefixes(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// `None` when the variable is unset or empty (no restriction).
fn configured() -> Option<&'static [String]> {
    static PREFIXES: OnceLock<Option<Vec<String>>> = OnceLock::new();
    PREFIXES
        .get_or_init(|| {
            let list = parse_prefixes(&std::env::var(ALLOWED_URL_PREFIXES_ENV).ok()?);
            (!list.is_empty()).then_some(list)
        })
        .as_deref()
}

fn decode_percent_once(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h1), Some(h2)) = (
                (b[i + 1] as char).to_digit(16),
                (b[i + 2] as char).to_digit(16),
            ) {
                let byte = ((h1 << 4) | h2) as u8;
                if byte.is_ascii() {
                    out.push(byte as char);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// True when `uri` falls under one of `prefixes`.
///
/// A prefix only matches at a path boundary, so `https://bucket.example.com`
/// allows `https://bucket.example.com/a.tif` but not
/// `https://bucket.example.com.evil.test/a.tif`. Locations containing raw or
/// percent-encoded `..` segments (e.g. `%2e%2e`, `..%2f`, `%252e%252e`) or
/// encoded path separators (`%2f`, `%5c`) never match, since they can climb
/// out of an allowed prefix on downstream HTTP servers/proxies.
pub fn matches_any_prefix(prefixes: &[String], uri: &str) -> bool {
    matches_any_prefix_with_opts(prefixes, uri, false)
}

/// Same as [`matches_any_prefix`], with opt-in symlink canonicalization for local paths
/// (`COGZILLA_ALLOWLIST_RESOLVE_SYMLINKS=1`).
pub fn matches_any_prefix_with_opts(
    prefixes: &[String],
    uri: &str,
    resolve_symlinks: bool,
) -> bool {
    // Reject ASCII control characters (including `\t`, `\r`, `\n`), which WHATWG URL parsers
    // (`url` / `reqwest`) silently strip before dot-segment resolution, enabling `.%09./` bypasses.
    if uri.bytes().any(|b| b <= 0x1f || b == 0x7f) {
        return false;
    }

    // Normalize `file:///path` and `file://localhost/path` to `/path` (Z1); reject remote
    // `file://otherhost/...` URIs.
    let (is_file_uri, normalized_uri) = match strip_file_uri_prefix(uri) {
        Ok(Some(local)) => (true, local),
        Ok(None) => (false, uri),
        Err(()) => return false,
    };

    // Unwrap GDAL `/vsicurl/http(s)://...` only for non-`file://` URIs so `file:///vsicurl/...`
    // is never treated as HTTP by the allowlist while opened as a local path by the backend.
    let (is_vsicurl, effective_uri) = if !is_file_uri {
        if let Some(rest) = normalized_uri.strip_prefix("/vsicurl/") {
            if !(rest.starts_with("http://") || rest.starts_with("https://")) {
                return false;
            }
            (true, rest)
        } else {
            (false, normalized_uri)
        }
    } else {
        (false, normalized_uri)
    };

    // For HTTP(S) URLs, inspect only the path component (before `?query` / `#fragment`) so valid
    // AWS SigV4 presigned URLs (`%2F` in `X-Amz-Credential`) and Azure SAS URLs (`%2F` in base64
    // `sig=`) are allowed. For cloud object URIs (`s3://`, `gs://`, `az://`, `abfs(s)://`, `/vsi*`)
    // and local paths, `?` and `#` are literal path/key bytes — inspect the entire URI so
    // `gs://allowed/public/?/../../../secret/key.tif` cannot bypass traversal checks.
    let is_http = effective_uri.starts_with("http://") || effective_uri.starts_with("https://");
    let inspect = if is_http {
        effective_uri.split(['?', '#']).next().unwrap_or(effective_uri)
    } else {
        effective_uri
    };
    let mut cur = inspect.to_string();
    for _ in 0..3 {
        if cur.bytes().any(|b| b <= 0x1f || b == 0x7f) {
            return false;
        }
        let lower = cur.to_ascii_lowercase();
        if lower.contains("%2e") || lower.contains("%2f") || lower.contains("%5c") {
            return false;
        }
        if cur.split(['/', '\\']).any(|seg| seg == "..") {
            return false;
        }
        let next = decode_percent_once(&cur);
        if next == cur {
            break;
        }
        cur = next;
    }

    let matches_boundary = |candidate: &str, local_only: bool| -> bool {
        prefixes.iter().any(|raw_p| {
            let p = match strip_file_uri_prefix(raw_p.as_str()) {
                Ok(Some(local_p)) => local_p,
                _ => raw_p.as_str(),
            };
            if local_only && !is_local_candidate(p) {
                return false;
            }
            match candidate.strip_prefix(p) {
                Some(rest) => rest.is_empty() || p.ends_with('/') || rest.starts_with('/'),
                None => false,
            }
        })
    };

    let matched = if is_file_uri {
        matches_boundary(normalized_uri, true)
    } else {
        matches_boundary(uri, false) || (is_vsicurl && matches_boundary(effective_uri, false))
    };
    if !matched {
        return false;
    }

    // Opt-in symlink hardening (Z3): canonicalize local paths (and, when the leaf does not exist,
    // the deepest existing ancestor) so symlinks inside an allowed directory pointing outside it
    // are refused with 403 without leaking existence of files under an escaping symlinked parent.
    if resolve_symlinks && (is_file_uri || is_local_candidate(effective_uri)) {
        let canon_matches_local_prefix = |canon_str: &str| -> bool {
            prefixes.iter().any(|raw_p| {
                let p = match strip_file_uri_prefix(raw_p.as_str()) {
                    Ok(Some(local_p)) => local_p,
                    _ => raw_p.as_str(),
                };
                if !is_local_candidate(p) {
                    return false;
                }
                let cp = std::fs::canonicalize(p)
                    .map(|pb| pb.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| p.to_string());
                match canon_str.strip_prefix(cp.as_str()) {
                    Some(rest) => rest.is_empty() || cp.ends_with('/') || rest.starts_with('/'),
                    None => false,
                }
            })
        };
        match std::fs::canonicalize(effective_uri) {
            Ok(canon) => {
                if !canon_matches_local_prefix(&canon.to_string_lossy()) {
                    return false;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Reject dangling symlinks outright (target could be outside the allowlist).
                if let Ok(meta) = std::fs::symlink_metadata(effective_uri)
                    && meta.file_type().is_symlink()
                {
                    return false;
                }
                // Walk up to the deepest existing ancestor and verify it stays inside an allowed
                // local prefix (prevents `/allowed/link/missing` -> 404 vs `/allowed/link/exist` -> 403).
                let mut cur_path = std::path::Path::new(effective_uri);
                while let Some(parent) = cur_path.parent() {
                    if parent.as_os_str().is_empty() {
                        break;
                    }
                    match std::fs::canonicalize(parent) {
                        Ok(canon_parent) => {
                            if !canon_matches_local_prefix(&canon_parent.to_string_lossy()) {
                                return false;
                            }
                            break;
                        }
                        Err(pe) if pe.kind() == std::io::ErrorKind::NotFound => {
                            cur_path = parent;
                        }
                        Err(_) => return false,
                    }
                }
            }
            Err(_) => return false,
        }
    }

    // For HTTP(S) URLs, also canonicalize with `reqwest::Url::parse` (the exact parser used by
    // outbound requests) so WHATWG normalization cannot escape the allowed prefix.
    if is_http {
        let Ok(parsed) = reqwest::Url::parse(effective_uri) else {
            return false;
        };
        if parsed.path().split('/').any(|seg| seg == "..") {
            return false;
        }
        let parsed_ok = matches_boundary(parsed.as_str(), false)
            || (is_vsicurl && matches_boundary(&format!("/vsicurl/{}", parsed.as_str()), false));
        if !parsed_ok {
            return false;
        }
    }

    true
}

/// Reject `uri` with [`IoError::Forbidden`] when an allowlist is configured
/// and the location is outside it.
pub fn check_uri_allowed(uri: &str) -> Result<(), IoError> {
    match configured() {
        Some(prefixes) if !matches_any_prefix_with_opts(prefixes, uri, resolve_symlinks_enabled()) => {
            Err(IoError::Forbidden(uri.to_string()))
        }
        _ => Ok(()),
    }
}

/// Whether an allowlist is in force.
pub fn allowlist_enabled() -> bool {
    configured().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_comma_separated_prefixes() {
        assert_eq!(
            parse_prefixes(" https://a.example/ , s3://bucket/,, "),
            list(&["https://a.example/", "s3://bucket/"])
        );
    }

    #[test]
    fn matches_only_at_a_path_boundary() {
        let p = list(&["https://bucket.example.com", "s3://imagery/public/"]);
        assert!(matches_any_prefix(&p, "https://bucket.example.com/a.tif"));
        assert!(matches_any_prefix(&p, "https://bucket.example.com"));
        assert!(matches_any_prefix(&p, "s3://imagery/public/scene/B4.tif"));
        assert!(!matches_any_prefix(&p, "https://bucket.example.com.evil.test/a.tif"));
        assert!(!matches_any_prefix(&p, "https://bucket.example.com@evil.test/a.tif"));
        assert!(!matches_any_prefix(&p, "s3://imagery/private/a.tif"));
        assert!(!matches_any_prefix(&p, "/etc/passwd"));
        assert!(!matches_any_prefix(&p, "http://169.254.169.254/latest/meta-data/"));
    }

    #[test]
    fn rejects_parent_directory_segments() {
        let p = list(&["/data/", "https://bucket.example.com/"]);
        assert!(matches_any_prefix(&p, "/data/dem.tif"));
        assert!(!matches_any_prefix(&p, "/data/../etc/passwd"));
        assert!(!matches_any_prefix(&p, "https://bucket.example.com/a/../../b.tif"));
        assert!(!matches_any_prefix(&p, "https://bucket.example.com/a/%2e%2e/b.tif"));
        assert!(!matches_any_prefix(&p, "https://bucket.example.com/a/..%2fb.tif"));
        assert!(!matches_any_prefix(&p, "https://bucket.example.com/a/%252e%252e/b.tif"));
        assert!(!matches_any_prefix(&p, "https://bucket.example.com/public/..?x=1"));
        assert!(!matches_any_prefix(&p, "https://bucket.example.com/public/..#frag"));
        assert!(matches_any_prefix(&p, "https://bucket.example.com/a..b.tif"));
        // Presigned S3 URLs (`%2F` in X-Amz-Credential) and Azure SAS URLs (`%2F` in base64 sig)
        // must be allowed while `%2F` in the path remains rejected.
        assert!(matches_any_prefix(
            &p,
            "https://bucket.example.com/scene.tif?X-Amz-Credential=AKIA%2F20261124%2Fus-west-2%2Fs3%2Faws4_request"
        ));
        assert!(matches_any_prefix(
            &p,
            "https://bucket.example.com/scene.tif?sig=abc%2Fdef%2B123%3D"
        ));
        // Cloud object URIs treat `?`/`#` as literal key bytes, so `?/../..` must be rejected!
        let cloud_p = list(&["gs://allowed/public/", "az://acct/public/", "s3://allowed/public/"]);
        assert!(!matches_any_prefix(&cloud_p, "gs://allowed/public/?/../../../secret-bucket/key.tif"));
        assert!(!matches_any_prefix(&cloud_p, "az://acct/public/?/../../../private/blob.tif"));
        assert!(!matches_any_prefix(&cloud_p, "s3://allowed/public/#/../../secret/key.tif"));
        assert!(matches_any_prefix(&cloud_p, "gs://allowed/public/scene.tif"));

        // WHATWG URL parser strips ASCII `\t`/`\r`/`\n` before dot-segment resolution:
        // both raw control chars and percent-encoded `%09`/`%0a`/`%0d` must be rejected.
        let gcs_http = list(&["https://storage.googleapis.com/public-bucket/"]);
        assert!(!matches_any_prefix(
            &gcs_http,
            "https://storage.googleapis.com/public-bucket/.\t./private-bucket/secret.tif"
        ));
        assert!(!matches_any_prefix(
            &gcs_http,
            "https://storage.googleapis.com/public-bucket/.\r./private-bucket/secret.tif"
        ));
        assert!(!matches_any_prefix(
            &gcs_http,
            "https://storage.googleapis.com/public-bucket/.\n./private-bucket/secret.tif"
        ));
        assert!(!matches_any_prefix(
            &gcs_http,
            "https://storage.googleapis.com/public-bucket/.%09./private-bucket/secret.tif"
        ));
        assert!(!matches_any_prefix(
            &gcs_http,
            "https://storage.googleapis.com/public-bucket/%2%09e%2%09e/private-bucket/secret.tif"
        ));
        assert!(matches_any_prefix(
            &gcs_http,
            "https://storage.googleapis.com/public-bucket/scene.tif"
        ));

        // GDAL `/vsicurl/http(s)://...` prefix unwrapping and traversal protection.
        assert!(matches_any_prefix(
            &gcs_http,
            "/vsicurl/https://storage.googleapis.com/public-bucket/scene.tif"
        ));
        assert!(!matches_any_prefix(
            &gcs_http,
            "/vsicurl/https://storage.googleapis.com/public-bucket/../private/secret.tif"
        ));
        assert!(!matches_any_prefix(
            &list(&["/vsicurl/"]),
            "/vsicurl/file:///etc/passwd"
        ));
    }

    #[test]
    fn normalizes_file_uri_and_rejects_remote_file_hosts() {
        let p = list(&["/allowed/dir"]);
        assert!(matches_any_prefix(&p, "/allowed/dir/a.tif"));
        assert!(matches_any_prefix(&p, "file:///allowed/dir/a.tif"));
        assert!(matches_any_prefix(&p, "file://localhost/allowed/dir/a.tif"));
        assert!(!matches_any_prefix(&p, "file:///allowed/dir2/a.tif"));
        assert!(!matches_any_prefix(&p, "file:///other/a.tif"));
        assert!(!matches_any_prefix(&p, "file:///allowed/dir/../etc/passwd"));
        assert!(!matches_any_prefix(&p, "file://localhost/allowed/dir/../etc/passwd"));
        assert!(!matches_any_prefix(&p, "file://evil/allowed/dir/a.tif"));

        // Configured `file://` prefix also matches bare local paths and `file://localhost`.
        let p_file = list(&["file:///allowed/dir/"]);
        assert!(matches_any_prefix(&p_file, "/allowed/dir/a.tif"));
        assert!(matches_any_prefix(&p_file, "file:///allowed/dir/a.tif"));
        assert!(matches_any_prefix(&p_file, "file://localhost/allowed/dir/a.tif"));
        assert!(!matches_any_prefix(&p_file, "/allowed/other/a.tif"));

        // `file://` URIs must never match remote HTTP or GDAL VSI prefixes (parser differential).
        let p_remote = list(&["https://allowed.example/", "/vsis3/bucket/"]);
        assert!(!matches_any_prefix(
            &p_remote,
            "file:///vsicurl/https://allowed.example/a.tif"
        ));
        assert!(!matches_any_prefix(
            &p_remote,
            "file://localhost/vsicurl/https://allowed.example/a.tif"
        ));
        assert!(!matches_any_prefix(&p_remote, "file:///vsis3/bucket/a.tif"));
    }

    #[cfg(unix)]
    #[test]
    fn opt_in_symlink_resolution_blocks_escape() {
        use std::os::unix::fs::symlink;

        let base = std::env::temp_dir().join(format!(
            "cogzilla-allowlist-symlink-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let allowed = base.join("allowed");
        let outside = base.join("outside");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&outside).unwrap();

        let ok_file = allowed.join("ok.tif");
        let secret_file = outside.join("secret.tif");
        std::fs::write(&ok_file, b"ok").unwrap();
        std::fs::write(&secret_file, b"secret").unwrap();

        let escape_link = allowed.join("escape.tif");
        symlink(&secret_file, &escape_link).unwrap();

        let prefixes = vec![allowed.to_string_lossy().into_owned()];
        let ok_str = ok_file.to_string_lossy();
        let link_str = escape_link.to_string_lossy();
        let link_file_uri = format!("file://{link_str}");

        // Default (resolve_symlinks = false): lexical match allows both.
        assert!(matches_any_prefix_with_opts(&prefixes, &ok_str, false));
        assert!(matches_any_prefix_with_opts(&prefixes, &link_str, false));

        // Opt-in (resolve_symlinks = true): regular file allowed, escaping symlink refused.
        assert!(matches_any_prefix_with_opts(&prefixes, &ok_str, true));
        assert!(!matches_any_prefix_with_opts(&prefixes, &link_str, true));
        assert!(!matches_any_prefix_with_opts(&prefixes, &link_file_uri, true));

        // Non-existent file under allowed prefix falls through to normal 404.
        let missing = allowed.join("missing.tif");
        assert!(matches_any_prefix_with_opts(
            &prefixes,
            &missing.to_string_lossy(),
            true
        ));

        // Symlinked parent dir pointing outside: both existing and missing leaves are refused (403),
        // preventing a 403-vs-404 existence oracle.
        let dir_link = allowed.join("dir_link");
        symlink(&outside, &dir_link).unwrap();
        assert!(!matches_any_prefix_with_opts(
            &prefixes,
            &dir_link.join("secret.tif").to_string_lossy(),
            true
        ));
        assert!(!matches_any_prefix_with_opts(
            &prefixes,
            &dir_link.join("does_not_exist.tif").to_string_lossy(),
            true
        ));

        // Dangling symlink pointing outside is refused (403).
        let dangling = allowed.join("dangling.tif");
        symlink(outside.join("nonexistent.tif"), &dangling).unwrap();
        assert!(!matches_any_prefix_with_opts(
            &prefixes,
            &dangling.to_string_lossy(),
            true
        ));

        let _ = std::fs::remove_dir_all(&base);
    }
}
