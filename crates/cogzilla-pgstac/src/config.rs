//! Connection settings, read the way titiler-pgstac 2+ reads them.
//!
//! `DATABASE_URL` wins when set. Otherwise the libpq variables `PGHOST`, `PGPORT`,
//! `PGUSER`, `PGPASSWORD` and `PGDATABASE` are used, and pgSTAC support stays off
//! unless at least `PGHOST` or `PGDATABASE` is present.

use std::time::Duration;

use crate::PgStacError;
use crate::tls::TlsVerify;

/// Everything needed to open the pool.
#[derive(Clone)]
pub struct PgStacConfig {
    pub(crate) pg: tokio_postgres::Config,
    pub(crate) verify: TlsVerify,
    /// Largest number of open connections (`DB_MAX_CONN_SIZE`, default 10).
    pub max_connections: usize,
    /// How long a request waits for a free connection before failing.
    pub acquire_timeout: Duration,
    /// Whether pictures list their items with the lean lookup (`COGZILLA_PGSTAC_LEAN`).
    pub lean: LeanMode,
}

/// When the lean item lookup is used in place of pgSTAC's `geojsonsearch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeanMode {
    /// On the pgSTAC versions it was checked against.
    #[default]
    Auto,
    /// On every version. A request it cannot answer still goes to `geojsonsearch`.
    On,
    Off,
}

impl std::fmt::Debug for PgStacConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the password or the full connection string.
        f.debug_struct("PgStacConfig")
            .field("hosts", &self.pg.get_hosts())
            .field("dbname", &self.pg.get_dbname())
            .field("verify", &self.verify)
            .field("max_connections", &self.max_connections)
            .finish()
    }
}

fn env_nonempty(get: &dyn Fn(&str) -> Option<String>, key: &str) -> Option<String> {
    get(key).map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// tokio-postgres knows `disable`, `prefer` and `require`. libpq's two verifying modes
/// are mapped to `require` plus certificate verification on our side.
fn split_sslmode(mode: &str) -> Result<(&'static str, TlsVerify), PgStacError> {
    match mode.to_ascii_lowercase().as_str() {
        "disable" => Ok(("disable", TlsVerify::None)),
        "allow" | "prefer" => Ok(("prefer", TlsVerify::None)),
        "require" => Ok(("require", TlsVerify::None)),
        "verify-ca" | "verify-full" => Ok(("require", TlsVerify::Full)),
        other => Err(PgStacError::Config(format!("unsupported sslmode '{other}'"))),
    }
}

/// Pull `sslmode=<value>` out of a connection string so the verifying modes, which
/// tokio-postgres rejects, can be handled here. Works for both URL and key/value forms.
fn extract_sslmode(conn: &str) -> (String, Option<String>) {
    let lower = conn.to_ascii_lowercase();
    let Some(pos) = lower.find("sslmode=") else {
        return (conn.to_string(), None);
    };
    let value_start = pos + "sslmode=".len();
    let value_end = conn[value_start..]
        .find(|c: char| c == '&' || c.is_whitespace())
        .map_or(conn.len(), |i| value_start + i);
    let value = conn[value_start..value_end].to_string();
    let mut out = String::with_capacity(conn.len());
    out.push_str(&conn[..pos]);
    out.push_str(conn[value_end..].trim_start_matches(|c: char| c == '&' || c.is_whitespace()));
    let out = out.trim_end_matches(['?', '&', ' ']).to_string();
    (out, Some(value))
}

impl PgStacConfig {
    /// Build from a `postgresql://` URL or a libpq key/value string.
    pub fn from_connection_string(conn: &str) -> Result<Self, PgStacError> {
        Self::build(conn, &|k| std::env::var(k).ok())
    }

    fn build(conn: &str, get: &dyn Fn(&str) -> Option<String>) -> Result<Self, PgStacError> {
        let (stripped, url_mode) = extract_sslmode(conn);
        // The parse error can echo the string, password included, so it is not passed on.
        let mut pg: tokio_postgres::Config = stripped
            .parse()
            .map_err(|_| PgStacError::Config("invalid database connection string".into()))?;
        let mode = url_mode
            .or_else(|| env_nonempty(get, "PGSSLMODE"))
            .unwrap_or_else(|| "prefer".to_string());
        let (pg_mode, verify) = split_sslmode(&mode)?;
        pg.ssl_mode(match pg_mode {
            "disable" => tokio_postgres::config::SslMode::Disable,
            "require" => tokio_postgres::config::SslMode::Require,
            _ => tokio_postgres::config::SslMode::Prefer,
        });
        // pgSTAC's functions and tables are found through the search path, as in titiler-pgstac.
        let schemas = env_nonempty(get, "PGSTAC_SEARCH_PATH").unwrap_or_else(|| "pgstac,public".into());
        if !schemas
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b',' | b' ' | b'$'))
        {
            return Err(PgStacError::Config("invalid PGSTAC_SEARCH_PATH".into()));
        }
        // No parallel workers: every query here is an index lookup of a few rows to a few
        // thousand, and starting workers costs more than such a query takes.
        pg.options(format!(
            "-c search_path={} -c max_parallel_workers_per_gather=0",
            schemas.replace(' ', "")
        ));
        if pg.get_application_name().is_none() {
            pg.application_name("cogzilla");
        }
        if pg.get_connect_timeout().is_none() {
            pg.connect_timeout(Duration::from_secs(10));
        }
        let max_connections = env_nonempty(get, "DB_MAX_CONN_SIZE")
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(10);
        Ok(Self {
            pg,
            verify,
            max_connections,
            acquire_timeout: Duration::from_secs(10),
            lean: match env_nonempty(get, "COGZILLA_PGSTAC_LEAN").map(|v| v.to_ascii_lowercase()).as_deref() {
                None | Some("auto") => LeanMode::Auto,
                Some("on" | "1" | "true" | "yes") => LeanMode::On,
                Some("off" | "0" | "false" | "no") => LeanMode::Off,
                Some(_) => return Err(PgStacError::Config("COGZILLA_PGSTAC_LEAN should be auto, on or off".into())),
            },
        })
    }

    /// Read the process environment. `Ok(None)` means no database is configured.
    pub fn from_env() -> Result<Option<Self>, PgStacError> {
        Self::from_lookup(&|k| std::env::var(k).ok())
    }

    fn from_lookup(get: &dyn Fn(&str) -> Option<String>) -> Result<Option<Self>, PgStacError> {
        if let Some(url) = env_nonempty(get, "DATABASE_URL") {
            return Self::build(&url, get).map(Some);
        }
        let host = env_nonempty(get, "PGHOST");
        let dbname = env_nonempty(get, "PGDATABASE");
        if host.is_none() && dbname.is_none() {
            return Ok(None);
        }
        let mut cfg = Self::build("", get)?;
        cfg.pg.host(host.as_deref().unwrap_or("localhost"));
        if let Some(port) = env_nonempty(get, "PGPORT") {
            let port: u16 = port
                .parse()
                .map_err(|_| PgStacError::Config("PGPORT is not a port number".into()))?;
            cfg.pg.port(port);
        }
        if let Some(user) = env_nonempty(get, "PGUSER") {
            cfg.pg.user(&user);
        }
        if let Some(password) = get("PGPASSWORD").filter(|v| !v.is_empty()) {
            cfg.pg.password(password);
        }
        if let Some(db) = dbname {
            cfg.pg.dbname(&db);
        }
        Ok(Some(cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn nothing_configured_means_off() {
        assert!(PgStacConfig::from_lookup(&lookup(&[])).unwrap().is_none());
        assert!(PgStacConfig::from_lookup(&lookup(&[("PGUSER", "u")])).unwrap().is_none());
    }

    #[test]
    fn libpq_variables_build_a_config() {
        let get = lookup(&[
            ("PGHOST", "db.internal"),
            ("PGPORT", "5439"),
            ("PGUSER", "reader"),
            ("PGDATABASE", "postgis"),
            ("DB_MAX_CONN_SIZE", "4"),
        ]);
        let cfg = PgStacConfig::from_lookup(&get).unwrap().unwrap();
        assert_eq!(cfg.pg.get_ports(), &[5439]);
        assert_eq!(cfg.pg.get_user(), Some("reader"));
        assert_eq!(cfg.pg.get_dbname(), Some("postgis"));
        assert_eq!(
            cfg.pg.get_options(),
            Some("-c search_path=pgstac,public -c max_parallel_workers_per_gather=0")
        );
        assert_eq!(cfg.max_connections, 4);
        assert_eq!(cfg.verify, TlsVerify::None);
    }

    #[test]
    fn database_url_wins_and_verifying_modes_are_handled() {
        let get = lookup(&[
            ("DATABASE_URL", "postgresql://u:p@h:6000/d?sslmode=verify-full&connect_timeout=3"),
            ("PGHOST", "ignored"),
        ]);
        let cfg = PgStacConfig::from_lookup(&get).unwrap().unwrap();
        assert_eq!(cfg.pg.get_ports(), &[6000]);
        assert_eq!(cfg.pg.get_dbname(), Some("d"));
        assert_eq!(cfg.verify, TlsVerify::Full);
        assert_eq!(cfg.pg.get_ssl_mode(), tokio_postgres::config::SslMode::Require);
        assert_eq!(cfg.pg.get_connect_timeout(), Some(&Duration::from_secs(3)));
    }

    #[test]
    fn sslmode_is_extracted_from_either_form() {
        assert_eq!(
            extract_sslmode("postgresql://h/d?sslmode=require"),
            ("postgresql://h/d".to_string(), Some("require".to_string()))
        );
        assert_eq!(
            extract_sslmode("host=h sslmode=verify-ca dbname=d"),
            ("host=h dbname=d".to_string(), Some("verify-ca".to_string()))
        );
        assert_eq!(extract_sslmode("host=h"), ("host=h".to_string(), None));
    }

    #[test]
    fn bad_values_are_refused_without_echoing_secrets() {
        let err = PgStacConfig::build("postgresql://u:hunter2@[bad/d", &lookup(&[])).unwrap_err();
        assert!(!err.to_string().contains("hunter2"));
        assert!(PgStacConfig::build("host=h sslmode=bogus", &lookup(&[])).is_err());
        let get = lookup(&[("PGSTAC_SEARCH_PATH", "pgstac; DROP")]);
        assert!(PgStacConfig::build("host=h", &get).is_err());
    }

    #[test]
    fn debug_output_has_no_password() {
        let cfg = PgStacConfig::build("postgresql://u:hunter2@h/d", &lookup(&[])).unwrap();
        assert!(!format!("{cfg:?}").contains("hunter2"));
    }
}
