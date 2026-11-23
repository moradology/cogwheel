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

/// True when `uri` falls under one of `prefixes`.
///
/// A prefix only matches at a path boundary, so `https://bucket.example.com`
/// allows `https://bucket.example.com/a.tif` but not
/// `https://bucket.example.com.evil.test/a.tif`. Locations containing a `..`
/// segment never match, since they can climb out of an allowed prefix.
pub fn matches_any_prefix(prefixes: &[String], uri: &str) -> bool {
    if uri.split(['/', '\\']).any(|seg| seg == "..") {
        return false;
    }
    prefixes.iter().any(|p| match uri.strip_prefix(p.as_str()) {
        Some(rest) => rest.is_empty() || p.ends_with('/') || rest.starts_with('/'),
        None => false,
    })
}

/// Reject `uri` with [`IoError::Forbidden`] when an allowlist is configured
/// and the location is outside it.
pub fn check_uri_allowed(uri: &str) -> Result<(), IoError> {
    match configured() {
        Some(prefixes) if !matches_any_prefix(prefixes, uri) => {
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
        assert!(matches_any_prefix(&p, "https://bucket.example.com/a..b.tif"));
    }
}
