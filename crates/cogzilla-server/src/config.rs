//! Single server configuration parsed once from environment variables.
//!
//! Every `COGZILLA_*` server setting has one named default constant and one reader here.
//! Unparseable values are reported as [`ConfigWarning`]s rather than silently swallowed.

use std::fmt;
use std::sync::OnceLock;

use hyper::header::HeaderValue;

pub const CACHE_CONTROL_MAX_AGE_ENV: &str = "COGZILLA_CACHE_CONTROL_MAX_AGE";
pub const ZARR_CACHE_CONTROL_MAX_AGE_ENV: &str = "COGZILLA_CACHE_CONTROL_MAX_AGE_ZARR";
pub const MAX_STATISTICS_FEATURES_ENV: &str = "COGZILLA_MAX_STATISTICS_FEATURES";
pub const BODY_TIMEOUT_SECS_ENV: &str = "COGZILLA_BODY_TIMEOUT_SECS";
pub const SHUTDOWN_TIMEOUT_SECS_ENV: &str = "COGZILLA_SHUTDOWN_TIMEOUT_SECS";
pub const AIMD_ENV: &str = "COGZILLA_AIMD";
pub const AIMD_TARGET_MS_ENV: &str = "COGZILLA_AIMD_TARGET_MS";
pub const AIMD_MIN_LIMIT_ENV: &str = "COGZILLA_AIMD_MIN_LIMIT";
pub const AIMD_MAX_LIMIT_ENV: &str = "COGZILLA_AIMD_MAX_LIMIT";

/// Default `Cache-Control: public, max-age=<N>` seconds on raster & pgSTAC responses.
pub const DEFAULT_CACHE_CONTROL_MAX_AGE_SECS: u64 = 3600;
/// Default cap on features in a `POST /*/statistics` `FeatureCollection` (`0` = unlimited).
pub const DEFAULT_MAX_STATISTICS_FEATURES: usize = 10_000;
/// Default timeout in seconds for reading a `POST` request body (`0` = unlimited).
pub const DEFAULT_BODY_TIMEOUT_SECS: u64 = 30;
/// Default graceful shutdown drain timeout in seconds on `SIGINT`/`SIGTERM`.
pub const DEFAULT_SHUTDOWN_TIMEOUT_SECS: u64 = 15;
/// Default AIMD target RTT in milliseconds.
pub const DEFAULT_AIMD_TARGET_MS: u64 = 150;
/// Default AIMD maximum concurrency ceiling.
pub const DEFAULT_AIMD_MAX_LIMIT: usize = 4096;

/// Warning emitted when a `COGZILLA_*` environment variable is set to an unparseable value
/// and falls back to its default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigWarning {
    pub var: &'static str,
    pub raw: String,
    pub reason: String,
}

impl fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:?} ignored ({})", self.var, self.raw, self.reason)
    }
}

/// Effective server configuration, parsed once at startup.
#[derive(Clone, Debug)]
pub struct Config {
    pub cache_control_max_age_secs: u64,
    pub zarr_cache_control_max_age_secs: u64,
    pub max_statistics_features: usize,
    pub body_timeout_secs: u64,
    pub shutdown_timeout_secs: u64,
    pub aimd_enabled: bool,
    pub aimd_target_ms: u64,
    pub aimd_min_limit: Option<usize>,
    pub aimd_max_limit: usize,
    cache_control: HeaderValue,
    zarr_cache_control: HeaderValue,
}

fn format_cache_control(secs: u64) -> HeaderValue {
    if secs == 0 {
        HeaderValue::from_static("no-cache")
    } else {
        HeaderValue::from_str(&format!("public, max-age={secs}"))
            .unwrap_or_else(|_| HeaderValue::from_static("public, max-age=3600"))
    }
}

fn parse_u64(
    get: &dyn Fn(&str) -> Option<String>,
    var: &'static str,
    default: u64,
    warnings: &mut Vec<ConfigWarning>,
) -> u64 {
    let Some(raw) = get(var) else {
        return default;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return default;
    }
    match trimmed.parse::<u64>() {
        Ok(v) => v,
        Err(_) => {
            warnings.push(ConfigWarning {
                var,
                raw,
                reason: format!("expected non-negative integer; using default {default}"),
            });
            default
        }
    }
}

fn parse_usize(
    get: &dyn Fn(&str) -> Option<String>,
    var: &'static str,
    default: usize,
    warnings: &mut Vec<ConfigWarning>,
) -> usize {
    let Some(raw) = get(var) else {
        return default;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return default;
    }
    match trimmed.parse::<usize>() {
        Ok(v) => v,
        Err(_) => {
            warnings.push(ConfigWarning {
                var,
                raw,
                reason: format!("expected non-negative integer; using default {default}"),
            });
            default
        }
    }
}

fn parse_optional_usize(
    get: &dyn Fn(&str) -> Option<String>,
    var: &'static str,
    warnings: &mut Vec<ConfigWarning>,
) -> Option<usize> {
    let raw = get(var)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<usize>() {
        Ok(v) => Some(v),
        Err(_) => {
            warnings.push(ConfigWarning {
                var,
                raw,
                reason: "expected non-negative integer; using dynamic default".into(),
            });
            None
        }
    }
}

fn parse_bool_enabled(
    get: &dyn Fn(&str) -> Option<String>,
    var: &'static str,
    default: bool,
    warnings: &mut Vec<ConfigWarning>,
) -> bool {
    let Some(raw) = get(var) else {
        return default;
    };
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return default;
    }
    match s.as_str() {
        "1" | "on" | "true" | "yes" | "enabled" => true,
        "0" | "off" | "false" | "no" | "disabled" | "none" => false,
        _ => {
            warnings.push(ConfigWarning {
                var,
                raw,
                reason: format!("expected on/off or true/false; using default {default}"),
            });
            default
        }
    }
}

impl Config {
    /// Parse server configuration from the process environment, returning any warnings for
    /// unparseable variable values.
    pub fn from_env() -> (Self, Vec<ConfigWarning>) {
        Self::from_lookup(&|k| std::env::var(k).ok())
    }

    /// Parse server configuration from an arbitrary key lookup (used by unit tests).
    pub fn from_lookup(get: &dyn Fn(&str) -> Option<String>) -> (Self, Vec<ConfigWarning>) {
        let mut warnings = Vec::new();
        let cache_control_max_age_secs = parse_u64(
            get,
            CACHE_CONTROL_MAX_AGE_ENV,
            DEFAULT_CACHE_CONTROL_MAX_AGE_SECS,
            &mut warnings,
        );
        // Optional per-family override for `/zarr/*`, falling back to `COGZILLA_CACHE_CONTROL_MAX_AGE`.
        let zarr_cache_control_max_age_secs = parse_u64(
            get,
            ZARR_CACHE_CONTROL_MAX_AGE_ENV,
            cache_control_max_age_secs,
            &mut warnings,
        );
        let max_statistics_features = parse_usize(
            get,
            MAX_STATISTICS_FEATURES_ENV,
            DEFAULT_MAX_STATISTICS_FEATURES,
            &mut warnings,
        );
        let body_timeout_secs = parse_u64(
            get,
            BODY_TIMEOUT_SECS_ENV,
            DEFAULT_BODY_TIMEOUT_SECS,
            &mut warnings,
        );
        let shutdown_timeout_secs = parse_u64(
            get,
            SHUTDOWN_TIMEOUT_SECS_ENV,
            DEFAULT_SHUTDOWN_TIMEOUT_SECS,
            &mut warnings,
        );
        let aimd_enabled = parse_bool_enabled(get, AIMD_ENV, true, &mut warnings);
        let aimd_target_ms = parse_u64(
            get,
            AIMD_TARGET_MS_ENV,
            DEFAULT_AIMD_TARGET_MS,
            &mut warnings,
        )
        .max(10);
        let aimd_min_limit =
            parse_optional_usize(get, AIMD_MIN_LIMIT_ENV, &mut warnings).map(|v| v.max(1));
        let aimd_max_limit = parse_usize(
            get,
            AIMD_MAX_LIMIT_ENV,
            DEFAULT_AIMD_MAX_LIMIT,
            &mut warnings,
        );

        let cfg = Self {
            cache_control_max_age_secs,
            zarr_cache_control_max_age_secs,
            max_statistics_features,
            body_timeout_secs,
            shutdown_timeout_secs,
            aimd_enabled,
            aimd_target_ms,
            aimd_min_limit,
            aimd_max_limit,
            cache_control: format_cache_control(cache_control_max_age_secs),
            zarr_cache_control: format_cache_control(zarr_cache_control_max_age_secs),
        };
        (cfg, warnings)
    }

    /// Process-wide configuration parsed once from the environment, logging any [`ConfigWarning`]s.
    pub fn global() -> &'static Self {
        static CFG: OnceLock<Config> = OnceLock::new();
        CFG.get_or_init(|| {
            let (cfg, warnings) = Self::from_env();
            for w in &warnings {
                tracing::warn!(var = w.var, raw = %w.raw, reason = %w.reason, "invalid environment variable");
            }
            cfg
        })
    }

    /// Pre-formatted `Cache-Control` header for COG/STAC/pgSTAC raster responses.
    pub fn cache_control_header(&self) -> &HeaderValue {
        &self.cache_control
    }

    /// Pre-formatted `Cache-Control` header for `/zarr/*` raster responses.
    pub fn zarr_cache_control_header(&self) -> &HeaderValue {
        &self.zarr_cache_control
    }

    /// Enumerable list of `(ENV_VAR, effective_value)` pairs for `--print-config` and diagnostics.
    pub fn effective_entries(&self) -> Vec<(&'static str, String)> {
        vec![
            (CACHE_CONTROL_MAX_AGE_ENV, self.cache_control_max_age_secs.to_string()),
            (ZARR_CACHE_CONTROL_MAX_AGE_ENV, self.zarr_cache_control_max_age_secs.to_string()),
            (MAX_STATISTICS_FEATURES_ENV, self.max_statistics_features.to_string()),
            (BODY_TIMEOUT_SECS_ENV, self.body_timeout_secs.to_string()),
            (SHUTDOWN_TIMEOUT_SECS_ENV, self.shutdown_timeout_secs.to_string()),
            (AIMD_ENV, self.aimd_enabled.to_string()),
            (AIMD_TARGET_MS_ENV, self.aimd_target_ms.to_string()),
            (
                AIMD_MIN_LIMIT_ENV,
                self.aimd_min_limit
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "auto".into()),
            ),
            (AIMD_MAX_LIMIT_ENV, self.aimd_max_limit.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    fn lookup(pairs: &[(&str, &str)]) -> (Config, Vec<ConfigWarning>) {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        Config::from_lookup(&|k| map.get(k).cloned())
    }

    #[test]
    fn defaults_and_zarr_cache_control_fallback() {
        let (cfg, warnings) = lookup(&[]);
        assert!(warnings.is_empty());
        assert_eq!(cfg.cache_control_max_age_secs, DEFAULT_CACHE_CONTROL_MAX_AGE_SECS);
        assert_eq!(cfg.zarr_cache_control_max_age_secs, DEFAULT_CACHE_CONTROL_MAX_AGE_SECS);
        assert_eq!(cfg.cache_control_header(), "public, max-age=3600");
        assert_eq!(cfg.zarr_cache_control_header(), "public, max-age=3600");
        assert_eq!(cfg.max_statistics_features, DEFAULT_MAX_STATISTICS_FEATURES);
        assert_eq!(cfg.body_timeout_secs, DEFAULT_BODY_TIMEOUT_SECS);
        assert_eq!(cfg.shutdown_timeout_secs, DEFAULT_SHUTDOWN_TIMEOUT_SECS);
        assert!(cfg.aimd_enabled);
        assert_eq!(cfg.aimd_target_ms, DEFAULT_AIMD_TARGET_MS);
        assert_eq!(cfg.aimd_min_limit, None);
        assert_eq!(cfg.aimd_max_limit, DEFAULT_AIMD_MAX_LIMIT);
        assert_eq!(cfg.effective_entries().len(), 9);
    }

    #[test]
    fn trimmed_parsing_override_and_zero_no_cache() {
        let (cfg, warnings) = lookup(&[
            (CACHE_CONTROL_MAX_AGE_ENV, " 86400 "),
            (MAX_STATISTICS_FEATURES_ENV, " 250 "),
            (BODY_TIMEOUT_SECS_ENV, " 45 "),
            (SHUTDOWN_TIMEOUT_SECS_ENV, " 20 "),
            (AIMD_ENV, " OFF "),
            (AIMD_TARGET_MS_ENV, " 5 "),
            (AIMD_MIN_LIMIT_ENV, " 8 "),
            (AIMD_MAX_LIMIT_ENV, " 512 "),
        ]);
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(cfg.cache_control_max_age_secs, 86400);
        // Unset Zarr override inherits COGZILLA_CACHE_CONTROL_MAX_AGE.
        assert_eq!(cfg.zarr_cache_control_max_age_secs, 86400);
        assert_eq!(cfg.cache_control_header(), "public, max-age=86400");
        assert_eq!(cfg.zarr_cache_control_header(), "public, max-age=86400");
        assert_eq!(cfg.max_statistics_features, 250);
        assert_eq!(cfg.body_timeout_secs, 45);
        assert_eq!(cfg.shutdown_timeout_secs, 20);
        assert!(!cfg.aimd_enabled);
        assert_eq!(cfg.aimd_target_ms, 10); // clamped floor
        assert_eq!(cfg.aimd_min_limit, Some(8));
        assert_eq!(cfg.aimd_max_limit, 512);

        // Explicit Zarr override to 0 emits `no-cache` without affecting the general header.
        let (cfg2, warnings2) = lookup(&[
            (CACHE_CONTROL_MAX_AGE_ENV, "3600"),
            (ZARR_CACHE_CONTROL_MAX_AGE_ENV, "0"),
        ]);
        assert!(warnings2.is_empty());
        assert_eq!(cfg2.cache_control_header(), "public, max-age=3600");
        assert_eq!(cfg2.zarr_cache_control_header(), "no-cache");
    }

    #[test]
    fn unparseable_values_emit_named_warnings_and_use_defaults() {
        let (cfg, warnings) = lookup(&[
            (CACHE_CONTROL_MAX_AGE_ENV, "1OO"),
            (ZARR_CACHE_CONTROL_MAX_AGE_ENV, "nope"),
            (MAX_STATISTICS_FEATURES_ENV, "-5"),
            (BODY_TIMEOUT_SECS_ENV, "abc"),
            (SHUTDOWN_TIMEOUT_SECS_ENV, "3.5"),
            (AIMD_ENV, "maybe"),
            (AIMD_TARGET_MS_ENV, "fast"),
            (AIMD_MIN_LIMIT_ENV, "low"),
            (AIMD_MAX_LIMIT_ENV, "high"),
        ]);
        let warned_vars: Vec<&str> = warnings.iter().map(|w| w.var).collect();
        assert_eq!(
            warned_vars,
            vec![
                CACHE_CONTROL_MAX_AGE_ENV,
                ZARR_CACHE_CONTROL_MAX_AGE_ENV,
                MAX_STATISTICS_FEATURES_ENV,
                BODY_TIMEOUT_SECS_ENV,
                SHUTDOWN_TIMEOUT_SECS_ENV,
                AIMD_ENV,
                AIMD_TARGET_MS_ENV,
                AIMD_MIN_LIMIT_ENV,
                AIMD_MAX_LIMIT_ENV,
            ]
        );
        assert_eq!(cfg.cache_control_max_age_secs, DEFAULT_CACHE_CONTROL_MAX_AGE_SECS);
        assert_eq!(cfg.zarr_cache_control_max_age_secs, DEFAULT_CACHE_CONTROL_MAX_AGE_SECS);
        assert_eq!(cfg.max_statistics_features, DEFAULT_MAX_STATISTICS_FEATURES);
        assert!(cfg.aimd_enabled);
    }

    /// Recursively collect all `.rs` files under `dir`.
    fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                collect_rs_files(&p, out);
            } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }

    /// Guard against re-introducing duplicate `COGZILLA_*` environment readers across workspace crates.
    #[test]
    fn no_duplicate_cogzilla_env_var_readers_in_workspace() {
        let workspace_crates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/ directory");
        let mut files = Vec::new();
        for entry in std::fs::read_dir(workspace_crates).unwrap().flatten() {
            let src = entry.path().join("src");
            if src.is_dir() {
                collect_rs_files(&src, &mut files);
            }
        }
        assert!(!files.is_empty());

        let mut readers: HashMap<String, Vec<String>> = HashMap::new();
        for file in files {
            let content = std::fs::read_to_string(&file).unwrap();
            // Stop before `#[cfg(test)]` modules so test fixtures don't count as production readers.
            let prod = content.split("#[cfg(test)]").next().unwrap_or(&content);
            let rel = file
                .strip_prefix(workspace_crates)
                .unwrap_or(&file)
                .display()
                .to_string();
            for (idx, line) in prod.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                if !code.contains("env::var(") && !code.contains("env_usize(") {
                    continue;
                }
                let mut rest = code;
                while let Some(pos) = rest.find("\"COGZILLA_") {
                    let after = &rest[pos + 1..];
                    if let Some(end) = after.find('"') {
                        let var = &after[..end];
                        readers
                            .entry(var.to_string())
                            .or_default()
                            .push(format!("{rel}:{}", idx + 1));
                        rest = &after[end + 1..];
                    } else {
                        break;
                    }
                }
            }
        }

        // `cogzilla-lambda` is a separate binary entrypoint that reads `COGZILLA_CACHE_MB`
        // once for boot and once for its startup log; every other COGZILLA_* var must have
        // at most one production reader across all workspace crates.
        let mut duplicates: Vec<String> = readers
            .into_iter()
            .filter(|(var, locs)| locs.len() > 1 && var != "COGZILLA_CACHE_MB")
            .map(|(var, locs)| format!("{var}: {}", locs.join(", ")))
            .collect();
        duplicates.sort();
        assert!(
            duplicates.is_empty(),
            "duplicate COGZILLA_* env var readers found:\n{}",
            duplicates.join("\n")
        );
    }
}
