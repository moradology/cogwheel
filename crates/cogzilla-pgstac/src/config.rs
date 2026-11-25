//! Connection settings, read the way titiler-pgstac 2+ reads them.
//!
//! `DATABASE_URL` wins when set. Otherwise the libpq variables `PGHOST`, `PGPORT`,
//! `PGUSER`, `PGPASSWORD` and `PGDATABASE` are used, and pgSTAC support stays off
//! unless at least `PGHOST` or `PGDATABASE` is present.

use std::time::Duration;

use tokio_postgres::config::SslMode;

use crate::PgStacError;
use crate::tls::TlsVerify;

/// Everything needed to open the pool.
#[derive(Clone)]
pub struct PgStacConfig {
    pub(crate) pg: tokio_postgres::Config,
    pub(crate) verify: TlsVerify,
    /// PEM file of trusted roots for the verifying modes; the public web roots when unset.
    pub(crate) root_cert: Option<String>,
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

fn invalid_connection_string() -> PgStacError {
    // Never quotes the string or any piece of it: any piece may be part of the password.
    PgStacError::Config("invalid database connection string".into())
}

/// Parameters tokio-postgres reads itself. `sslmode` and `sslrootcert` are handled here.
const PASSED_ON: &[&str] = &[
    "user",
    "password",
    "dbname",
    "options",
    "application_name",
    "sslnegotiation",
    "host",
    "hostaddr",
    "port",
    "connect_timeout",
    "tcp_user_timeout",
    "keepalives",
    "keepalives_idle",
    "keepalives_interval",
    "keepalives_retries",
    "target_session_attrs",
    "channel_binding",
    "load_balance_hosts",
];

/// libpq parameters that nothing here acts on. Naming them in the error is safe because
/// the name comes from this list, not from the connection string.
const NOT_SUPPORTED: &[&str] = &[
    "sslcert",
    "sslkey",
    "sslpassword",
    "sslcertmode",
    "sslcrl",
    "sslcrldir",
    "sslsni",
    "sslcompression",
    "sslkeylogfile",
    "ssl_min_protocol_version",
    "ssl_max_protocol_version",
    "requiressl",
    "requirepeer",
    "require_auth",
    "gssencmode",
    "krbsrvname",
    "gsslib",
    "gssdelegation",
    "passfile",
    "service",
    "client_encoding",
    "fallback_application_name",
    "keepalives_count",
    "replication",
    "min_protocol_version",
    "max_protocol_version",
];

/// The TLS settings of a connection string, and the string without them.
#[derive(Debug, Default, PartialEq)]
struct ConnParts {
    /// What is left for tokio-postgres, which rejects the verifying modes and `sslrootcert`.
    conn: String,
    sslmode: Option<String>,
    sslrootcert: Option<String>,
}

impl ConnParts {
    /// Take one decoded parameter. `Ok(true)` means it stays in the string.
    fn sort(&mut self, key: &str, value: &str) -> Result<bool, PgStacError> {
        // As in libpq, an empty value is the same as leaving the parameter out.
        let given = || Some(value.trim().to_string()).filter(|v| !v.is_empty());
        match key {
            "sslmode" => self.sslmode = given(),
            "sslrootcert" => self.sslrootcert = given(),
            _ if PASSED_ON.contains(&key) => return Ok(true),
            _ => {
                return Err(PgStacError::Config(match NOT_SUPPORTED.iter().find(|k| **k == key) {
                    Some(known) => format!("connection parameter '{known}' is not supported"),
                    None => "unknown parameter in the database connection string".into(),
                }));
            }
        }
        Ok(false)
    }
}

fn percent_decode(s: &str) -> Result<String, PgStacError> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        match (bytes[i], bytes.get(i + 1).copied().and_then(hex), bytes.get(i + 2).copied().and_then(hex)) {
            (b'%', Some(hi), Some(lo)) => {
                out.push((hi * 16 + lo) as u8);
                i += 3;
            }
            (b, _, _) => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| invalid_connection_string())
}

/// Split the TLS settings off a `postgresql://` URL. Only the query is looked at, and
/// only parameter names are compared, so a value or a password that happens to contain
/// `sslmode=` changes nothing.
fn split_url(conn: &str) -> Result<ConnParts, PgStacError> {
    // tokio-postgres ends the credentials at the first `@`, and the query starts at the
    // first `?` after that; the same reading is used here so both agree on what is what.
    let after_credentials = conn.find('@').map_or(0, |i| i + 1);
    let Some(question) = conn[after_credentials..].find('?').map(|i| after_credentials + i) else {
        return Ok(ConnParts {
            conn: conn.to_string(),
            ..ConnParts::default()
        });
    };
    let mut parts = ConnParts::default();
    let mut kept: Vec<&str> = Vec::new();
    for pair in conn[question + 1..].split('&').filter(|p| !p.is_empty()) {
        let (key, value) = pair.split_once('=').ok_or_else(invalid_connection_string)?;
        if parts.sort(&percent_decode(key)?, &percent_decode(value)?)? {
            kept.push(pair);
        }
    }
    parts.conn = conn[..question].to_string();
    if !kept.is_empty() {
        parts.conn.push('?');
        parts.conn.push_str(&kept.join("&"));
    }
    Ok(parts)
}

/// Split the TLS settings off a libpq `keyword = value` string. Values may be quoted with
/// `'`, and a backslash takes the next character literally, quoted or not.
fn split_keyword_value(conn: &str) -> Result<ConnParts, PgStacError> {
    let mut parts = ConnParts::default();
    let mut chars = conn.chars().peekable();
    loop {
        while chars.next_if(|c| c.is_whitespace()).is_some() {}
        if chars.peek().is_none() {
            return Ok(parts);
        }
        let mut key = String::new();
        while let Some(c) = chars.next_if(|c| !c.is_whitespace() && *c != '=') {
            key.push(c);
        }
        while chars.next_if(|c| c.is_whitespace()).is_some() {}
        if key.is_empty() || chars.next() != Some('=') {
            return Err(invalid_connection_string());
        }
        while chars.next_if(|c| c.is_whitespace()).is_some() {}
        let quoted = chars.next_if_eq(&'\'').is_some();
        let mut value = String::new();
        loop {
            match chars.peek().copied() {
                Some('\'') if quoted => {
                    chars.next();
                    break;
                }
                Some(c) if !quoted && c.is_whitespace() => break,
                None if quoted => return Err(invalid_connection_string()),
                None => break,
                Some('\\') => {
                    chars.next();
                    value.push(chars.next().ok_or_else(invalid_connection_string)?);
                }
                Some(c) => {
                    chars.next();
                    value.push(c);
                }
            }
        }
        if parts.sort(&key, &value)? {
            if !parts.conn.is_empty() {
                parts.conn.push(' ');
            }
            // Written back quoted, which holds any value.
            parts.conn.push_str(&key);
            parts.conn.push_str("='");
            parts.conn.push_str(&value.replace('\\', "\\\\").replace('\'', "\\'"));
            parts.conn.push('\'');
        }
    }
}

fn split_connection_string(conn: &str) -> Result<ConnParts, PgStacError> {
    if conn.starts_with("postgresql://") || conn.starts_with("postgres://") {
        split_url(conn)
    } else {
        split_keyword_value(conn)
    }
}

/// What to ask tokio-postgres for and how far to check the certificate, with libpq's
/// meanings. tokio-postgres knows `disable`, `prefer` and `require`; the checking is
/// done on our side. `require` checks the chain once a root certificate is configured,
/// and only `verify-full` checks the host name.
fn tls_plan(
    mode: Option<&str>,
    root_cert: Option<String>,
) -> Result<(SslMode, TlsVerify, Option<String>), PgStacError> {
    // libpq's `sslrootcert=system` trusts the system's roots and insists on `verify-full`.
    // The public web roots stand in for the system's here.
    let system = root_cert.as_deref() == Some("system");
    let mode = mode.map(str::to_ascii_lowercase);
    let mode = match mode.as_deref() {
        Some(mode) => mode,
        None if system => "verify-full",
        None => "prefer",
    };
    if system && mode != "verify-full" {
        return Err(PgStacError::Config("sslrootcert=system needs sslmode=verify-full".into()));
    }
    let root_cert = root_cert.filter(|_| !system);
    let (ssl_mode, verify) = match mode {
        "disable" => (SslMode::Disable, TlsVerify::None),
        "allow" | "prefer" => (SslMode::Prefer, TlsVerify::None),
        "require" if root_cert.is_none() => (SslMode::Require, TlsVerify::None),
        "require" | "verify-ca" => (SslMode::Require, TlsVerify::Ca),
        "verify-full" => (SslMode::Require, TlsVerify::Full),
        // The value is not quoted back: a badly quoted string can put anything here.
        _ => {
            return Err(PgStacError::Config(
                "unsupported sslmode: use disable, allow, prefer, require, verify-ca or verify-full".into(),
            ));
        }
    };
    Ok((ssl_mode, verify, root_cert))
}

impl PgStacConfig {
    /// Build from a `postgresql://` URL or a libpq key/value string.
    pub fn from_connection_string(conn: &str) -> Result<Self, PgStacError> {
        Self::build(conn, &|k| std::env::var(k).ok())
    }

    fn build(conn: &str, get: &dyn Fn(&str) -> Option<String>) -> Result<Self, PgStacError> {
        let parts = split_connection_string(conn)?;
        // The parse error can echo the string, password included, so it is not passed on.
        let mut pg: tokio_postgres::Config = parts.conn.parse().map_err(|_| invalid_connection_string())?;
        let mode = parts.sslmode.or_else(|| env_nonempty(get, "PGSSLMODE"));
        let root_cert = parts.sslrootcert.or_else(|| env_nonempty(get, "PGSSLROOTCERT"));
        let (ssl_mode, verify, root_cert) = tls_plan(mode.as_deref(), root_cert)?;
        pg.ssl_mode(ssl_mode);
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
        // They go after any options the connection string carries, which are kept.
        let ours = format!(
            "-c search_path={} -c max_parallel_workers_per_gather=0",
            schemas.replace(' ', "")
        );
        let options = match pg.get_options().map(str::trim).filter(|o| !o.is_empty()) {
            Some(theirs) => format!("{theirs} {ours}"),
            None => ours,
        };
        pg.options(options);
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
            root_cert,
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
        assert_eq!(cfg.pg.get_ssl_mode(), SslMode::Require);
        assert_eq!(cfg.pg.get_connect_timeout(), Some(&Duration::from_secs(3)));
    }

    fn built(conn: &str, env: &[(&str, &str)]) -> PgStacConfig {
        PgStacConfig::build(conn, &lookup(env)).unwrap()
    }

    fn refusal(conn: &str, env: &[(&str, &str)]) -> String {
        PgStacConfig::build(conn, &lookup(env)).unwrap_err().to_string()
    }

    #[test]
    fn sslmode_is_extracted_from_either_form() {
        let parts = split_connection_string("postgresql://h/d?sslmode=require").unwrap();
        assert_eq!((parts.conn.as_str(), parts.sslmode.as_deref()), ("postgresql://h/d", Some("require")));
        let parts = split_connection_string("host=h sslmode=verify-ca dbname=d").unwrap();
        assert_eq!((parts.conn.as_str(), parts.sslmode.as_deref()), ("host='h' dbname='d'", Some("verify-ca")));
        let parts = split_connection_string("host=h").unwrap();
        assert_eq!((parts.conn.as_str(), parts.sslmode), ("host='h'", None));
        // The parameters around it survive in a URL, percent-encoding untouched.
        let parts = split_connection_string("postgres://u:p@h/d?connect_timeout=3&sslmode=disable&user=a%20b").unwrap();
        assert_eq!(parts.conn, "postgres://u:p@h/d?connect_timeout=3&user=a%20b");
    }

    #[test]
    fn each_sslmode_checks_what_libpq_checks() {
        let cases = [
            ("disable", false, SslMode::Disable, TlsVerify::None),
            ("disable", true, SslMode::Disable, TlsVerify::None),
            ("allow", false, SslMode::Prefer, TlsVerify::None),
            ("prefer", true, SslMode::Prefer, TlsVerify::None),
            ("require", false, SslMode::Require, TlsVerify::None),
            ("require", true, SslMode::Require, TlsVerify::Ca),
            ("verify-ca", false, SslMode::Require, TlsVerify::Ca),
            ("verify-ca", true, SslMode::Require, TlsVerify::Ca),
            ("verify-full", false, SslMode::Require, TlsVerify::Full),
            ("verify-full", true, SslMode::Require, TlsVerify::Full),
        ];
        for (mode, with_root, ssl_mode, verify) in cases {
            let env: &[(&str, &str)] = if with_root { &[("PGSSLROOTCERT", "/certs/root.pem")] } else { &[] };
            let cfg = built(&format!("host=h sslmode={mode}"), env);
            assert_eq!((cfg.pg.get_ssl_mode(), cfg.verify), (ssl_mode, verify), "{mode} {with_root}");
            assert_eq!(cfg.root_cert.is_some(), with_root);
        }
        // A root named in the connection string counts as configured too.
        assert_eq!(built("host=h sslmode=require sslrootcert=/r.pem", &[]).verify, TlsVerify::Ca);
        assert_eq!(built("postgresql://h/d?sslmode=require&sslrootcert=/r.pem", &[]).verify, TlsVerify::Ca);
    }

    #[test]
    fn sslmode_inside_a_value_is_not_a_setting() {
        let strict = [("PGSSLMODE", "verify-full")];
        for conn in [
            "host=h application_name=sslmode=disable",
            "host=h application_name='x sslmode=disable'",
            "host=h password='a sslmode=disable b'",
            "postgresql://h/d?application_name=sslmode=disable",
            "postgresql://h/d?application_name=x%26sslmode%3Ddisable",
            "postgresql://u:sslmode=disable@h/d",
            "postgresql://u:p?sslmode=disable&x@h/d",
        ] {
            let cfg = built(conn, &strict);
            assert_eq!(cfg.verify, TlsVerify::Full, "{conn}");
            assert_eq!(cfg.pg.get_ssl_mode(), SslMode::Require, "{conn}");
        }
        let cfg = built("host=h application_name=sslmode=disable", &strict);
        assert_eq!(cfg.pg.get_application_name(), Some("sslmode=disable"));
        let cfg = built("host=h password='a sslmode=disable b' user=u", &strict);
        assert_eq!(cfg.pg.get_password(), Some(&b"a sslmode=disable b"[..]));
        assert_eq!(cfg.pg.get_user(), Some("u"));
        let cfg = built("postgresql://u:p?sslmode=disable&x@h/d", &strict);
        assert_eq!(cfg.pg.get_password(), Some(&b"p?sslmode=disable&x"[..]));
    }

    #[test]
    fn quoting_and_escaping_follow_libpq() {
        let cfg = built(r"host = h sslmode = 'require' password='it\'s \\ here' dbname=a\ b", &[]);
        assert_eq!(cfg.pg.get_ssl_mode(), SslMode::Require);
        assert_eq!(cfg.pg.get_password(), Some(&br"it's \ here"[..]));
        assert_eq!(cfg.pg.get_dbname(), Some("a b"));
        assert_eq!(built("host=h sslmode='verify-full'", &[]).verify, TlsVerify::Full);
        // A URL's names and values are percent-decoded before they are read.
        let cfg = built("postgresql://h/d?ssl%6Dode=verify%2Dfull&sslrootcert=%2Fcerts%2Fmy%20root.pem", &[]);
        assert_eq!(cfg.verify, TlsVerify::Full);
        assert_eq!(cfg.root_cert.as_deref(), Some("/certs/my root.pem"));
        // The last of a repeated parameter wins, and an empty one counts as not given.
        assert_eq!(built("host=h sslmode=disable sslmode=verify-full", &[]).verify, TlsVerify::Full);
        let cfg = built("host=h sslmode='' sslrootcert=''", &[("PGSSLMODE", "verify-ca"), ("PGSSLROOTCERT", "/env.pem")]);
        assert_eq!((cfg.verify, cfg.root_cert.as_deref()), (TlsVerify::Ca, Some("/env.pem")));
    }

    #[test]
    fn sslrootcert_is_read_from_the_connection_string() {
        let cfg = built("postgresql://u:p@h/d?sslmode=verify-full&sslrootcert=/certs/root.pem", &[]);
        assert_eq!(cfg.verify, TlsVerify::Full);
        assert_eq!(cfg.root_cert.as_deref(), Some("/certs/root.pem"));
        assert_eq!(cfg.pg.get_dbname(), Some("d"));
        let cfg = built("host=h sslmode=verify-ca sslrootcert='/certs/my root.pem'", &[]);
        assert_eq!(cfg.root_cert.as_deref(), Some("/certs/my root.pem"));
        // The connection string wins over the environment, which is the fallback.
        let env = [("PGSSLROOTCERT", "/env.pem")];
        assert_eq!(built("host=h sslrootcert=/conn.pem", &env).root_cert.as_deref(), Some("/conn.pem"));
        assert_eq!(built("host=h", &env).root_cert.as_deref(), Some("/env.pem"));
        assert_eq!(built("host=h", &[]).root_cert, None);
    }

    #[test]
    fn system_roots_insist_on_full_verification() {
        let cfg = built("host=h sslrootcert=system", &[]);
        assert_eq!((cfg.verify, cfg.root_cert), (TlsVerify::Full, None));
        let cfg = built("host=h sslmode=verify-full", &[("PGSSLROOTCERT", "system")]);
        assert_eq!((cfg.verify, cfg.root_cert), (TlsVerify::Full, None));
        assert!(refusal("host=h sslmode=require sslrootcert=system", &[]).contains("verify-full"));
    }

    #[test]
    fn options_in_the_connection_string_are_kept() {
        let ours = "-c search_path=pgstac,public -c max_parallel_workers_per_gather=0";
        let cfg = built("host=h options='-c statement_timeout=5000'", &[]);
        assert_eq!(cfg.pg.get_options(), Some(format!("-c statement_timeout=5000 {ours}").as_str()));
        let cfg = built("postgresql://h/d?options=-c%20statement_timeout%3D5000", &[]);
        assert_eq!(cfg.pg.get_options(), Some(format!("-c statement_timeout=5000 {ours}").as_str()));
        assert_eq!(built("host=h", &[]).pg.get_options(), Some(ours));
        assert_eq!(built("host=h options=' '", &[]).pg.get_options(), Some(ours));
    }

    #[test]
    fn unsupported_parameters_are_named_without_their_values() {
        for conn in [
            "host=h sslcert=/secret/client.crt sslkey=/secret/client.key",
            "postgresql://h/d?sslkey=%2Fsecret%2Fclient.key",
            "host=h gssencmode=/secret/x",
        ] {
            let message = refusal(conn, &[]);
            assert!(message.contains("is not supported"), "{message}");
            assert!(message.contains("'sslcert'") || message.contains("'sslkey'") || message.contains("'gssencmode'"));
            assert!(!message.contains("secret"), "{message}");
        }
        // A name that is not a libpq parameter may be a stray piece of something else.
        let message = refusal("host=h password=hunter2 part2=more", &[]);
        assert!(!message.contains("part2") && !message.contains("more") && !message.contains("hunter2"));
        // tokio-postgres reads this one itself.
        let cfg = built("host=h channel_binding=require", &[]);
        assert_eq!(cfg.pg.get_channel_binding(), tokio_postgres::config::ChannelBinding::Require);
    }

    #[test]
    fn configuration_errors_carry_no_piece_of_the_password() {
        for conn in [
            "host=h password=hunter2 sslmode=hunter3",
            "host=h password='hunter2",
            "host=h password=hunter2 hunter3",
            "host=h password=hunter2 sslmode=hunter3 x",
            "postgresql://u:hunter2@h/d?sslmode=hunter3",
            "postgresql://u:hunter2@h/d?hunter3",
            "postgresql://u:hunter2@h/d?hunter3=x",
            "postgresql://u:hunter2@h:port/d",
        ] {
            let message = refusal(conn, &[]);
            assert!(!message.contains("hunter"), "{conn}: {message}");
        }
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
