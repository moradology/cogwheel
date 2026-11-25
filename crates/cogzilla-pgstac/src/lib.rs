//! A small client for a pgSTAC database.
//!
//! The compatibility path runs the same SQL titiler-pgstac runs (`search_query` to
//! register a search, `geojsonsearch` to list the items that cover a geometry), so the
//! items are pgSTAC's own answer. The lean path ([`PgStac::items_for_pixels`]) lists the
//! items a picture needs with plain queries and falls back to the compatibility path for
//! anything it cannot answer. Both need read access to the `pgstac` schema and install
//! nothing in the database.

mod config;
mod coverage;
mod lean;
mod tls;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod, Runtime};
use parking_lot::Mutex;
use serde_json::Value;
use thiserror::Error;

pub use config::{LeanMode, PgStacConfig};
pub use coverage::SampleGrid;
pub use tls::TlsVerify;

/// Which lookup lists the items of a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup {
    /// As configured (`COGZILLA_PGSTAC_LEAN`).
    Configured,
    /// The lean lookup, on any pgSTAC version.
    Lean,
    /// pgSTAC's `geojsonsearch`.
    Reference,
}

#[derive(Debug, Error)]
pub enum PgStacError {
    #[error("database configuration: {0}")]
    Config(String),
    #[error("database unavailable: {0}")]
    Unavailable(String),
    #[error("database error: {0}")]
    Db(#[from] tokio_postgres::Error),
    #[error("pgSTAC is in read-only mode, so searches cannot be registered")]
    ReadOnly,
    #[error("unexpected pgSTAC result: {0}")]
    Shape(String),
}

impl PgStacError {
    /// The message Postgres itself raised, when the failure came from the server (a bad
    /// filter, an unknown queryable). These describe the request, not the deployment.
    pub fn server_message(&self) -> Option<&str> {
        match self {
            Self::Db(e) => e.as_db_error().map(|d| d.message()),
            _ => None,
        }
    }
}

/// Order object keys the way Postgres `jsonb` does: shorter keys first, then bytewise.
/// A list built from a `jsonb` object's keys has to keep this order to hash the same.
pub fn jsonb_key_order(keys: &mut [String]) {
    keys.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.as_bytes().cmp(b.as_bytes())));
}

/// One row of `pgstac.searches`.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchRecord {
    pub hash: String,
    pub search: Value,
    /// pgSTAC's compiled SQL filter. Stored only by some pgSTAC versions.
    pub where_clause: Option<String>,
    /// pgSTAC's compiled `ORDER BY`. Stored only by some pgSTAC versions.
    pub orderby: Option<String>,
    /// UTC, formatted as titiler-pgstac prints it (`2026-11-24T05:34:04.949443Z`).
    pub lastused: Option<String>,
    pub usecount: Option<i64>,
    pub metadata: Value,
}

/// The limits `geojsonsearch` takes, with titiler-pgstac's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SearchLimits {
    pub scan_limit: i32,
    pub items_limit: i32,
    pub time_limit_secs: i32,
    pub exitwhenfull: bool,
    pub skipcovered: bool,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            scan_limit: 10_000,
            items_limit: 100,
            time_limit_secs: 5,
            exitwhenfull: true,
            skipcovered: true,
        }
    }
}

/// An item as the mosaic path needs it: who it is and where its assets live.
#[derive(Debug, Clone, PartialEq)]
pub struct MosaicItem {
    pub id: String,
    pub collection: Option<String>,
    pub bbox: Option<[f64; 4]>,
    /// Asset key to the asset object exactly as pgSTAC returned it.
    pub assets: BTreeMap<String, Value>,
    /// The item's `self` link, which its relative asset hrefs are relative to.
    pub self_href: Option<String>,
}

/// Whether an href says where it is by itself: a URL, an absolute path, a GDAL `/vsi` path.
pub(crate) fn href_is_absolute(href: &str) -> bool {
    href.contains("://") || href.starts_with("/vsi")
}

impl MosaicItem {
    fn from_feature(feature: &Value) -> Option<Self> {
        let id = feature.get("id")?.as_str()?.to_string();
        let collection = feature
            .get("collection")
            .and_then(Value::as_str)
            .map(str::to_string);
        let bbox = feature.get("bbox").and_then(Value::as_array).and_then(|a| {
            // A 3D bbox is [minx, miny, minz, maxx, maxy, maxz].
            let idx: [usize; 4] = match a.len() {
                4 => [0, 1, 2, 3],
                6 => [0, 1, 3, 4],
                _ => return None,
            };
            Some([
                a[idx[0]].as_f64()?,
                a[idx[1]].as_f64()?,
                a[idx[2]].as_f64()?,
                a[idx[3]].as_f64()?,
            ])
        });
        let assets = feature
            .get("assets")
            .and_then(Value::as_object)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        let self_href = feature
            .get("links")
            .and_then(Value::as_array)
            .and_then(|links| links.iter().find(|l| l.get("rel").and_then(Value::as_str) == Some("self")))
            .and_then(|l| l.get("href")?.as_str())
            .filter(|h| href_is_absolute(h))
            .map(str::to_string);
        Some(Self {
            id,
            collection,
            bbox,
            assets,
            self_href,
        })
    }

    /// The `href` of one asset, if the item has it. A relative href is resolved against the
    /// item's `self` link; with no such link there is nowhere it is relative to, and the
    /// item is taken not to have the asset rather than read from wherever the server runs.
    pub fn asset_href(&self, key: &str) -> Option<std::borrow::Cow<'_, str>> {
        let href = self.assets.get(key)?.get("href")?.as_str()?;
        if href_is_absolute(href) {
            return Some(href.into());
        }
        let Some(base) = self.self_href.as_deref() else {
            // A path from the root is a file of this machine when the item says no more.
            return href.starts_with('/').then(|| href.into());
        };
        let base = base.split(['?', '#']).next().unwrap_or(base);
        if href.starts_with('/') {
            // From the root of wherever the item is.
            let after_scheme = base.find("://")? + 3;
            let origin = base[after_scheme..].find('/').map_or(base, |at| &base[..after_scheme + at]);
            return Some(format!("{origin}{href}").into());
        }
        let folder = &base[..=base.rfind('/')?];
        Some(format!("{folder}{}", href.trim_start_matches("./")).into())
    }

    /// The item in the shape titiler-pgstac's `/assets` routes return.
    pub fn to_json(&self) -> Value {
        let mut out = serde_json::Map::new();
        out.insert("id".into(), Value::String(self.id.clone()));
        if let Some(b) = self.bbox {
            out.insert("bbox".into(), serde_json::json!(b));
        }
        out.insert(
            "assets".into(),
            Value::Object(self.assets.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        );
        if let Some(c) = &self.collection {
            out.insert("collection".into(), Value::String(c.clone()));
        }
        Value::Object(out)
    }
}

const SEARCH_COLUMNS: &str = "hash, search, _where, orderby, \
     to_char(lastused AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') AS lastused, \
     usecount, metadata";

fn search_from_row(row: &tokio_postgres::Row) -> Result<SearchRecord, PgStacError> {
    let get = |e: tokio_postgres::Error| PgStacError::Shape(e.to_string());
    Ok(SearchRecord {
        hash: row.try_get("hash").map_err(get)?,
        search: row.try_get("search").map_err(get)?,
        where_clause: row.try_get("_where").map_err(get)?,
        orderby: row.try_get("orderby").map_err(get)?,
        lastused: row.try_get("lastused").map_err(get)?,
        usecount: row.try_get("usecount").map_err(get)?,
        metadata: row
            .try_get::<_, Option<Value>>("metadata")
            .map_err(get)?
            .unwrap_or_else(|| Value::Object(Default::default())),
    })
}

/// What a list of searches is sorted by: one of the two counters pgSTAC keeps, or a key of
/// the metadata a search was registered with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchField {
    LastUsed,
    UseCount,
    Metadata(String),
}

/// One term of the order of a list of searches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchOrder {
    pub field: SearchField,
    pub descending: bool,
}

/// A bounded, time-limited memo, standing in for titiler-pgstac's `TTLCache` so repeated
/// requests for a tile do not go back to Postgres.
struct TtlCache<V> {
    ttl: Duration,
    max_entries: usize,
    entries: Mutex<std::collections::HashMap<String, (Instant, V)>>,
}

impl<V: Clone> TtlCache<V> {
    fn from_env() -> Option<Self> {
        let env = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string());
        let disabled = env("TITILER_PGSTAC_CACHE_DISABLE")
            .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"));
        if disabled {
            return None;
        }
        let ttl = env("TITILER_PGSTAC_CACHE_TTL")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(300);
        let max_entries = env("TITILER_PGSTAC_CACHE_MAXSIZE")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        (ttl > 0 && max_entries > 0).then(|| Self {
            ttl: Duration::from_secs(ttl),
            max_entries,
            entries: Mutex::new(Default::default()),
        })
    }

    fn get(&self, key: &str) -> Option<V> {
        let mut map = self.entries.lock();
        match map.get(key) {
            Some((at, value)) if at.elapsed() < self.ttl => Some(value.clone()),
            Some(_) => {
                map.remove(key);
                None
            }
            None => None,
        }
    }

    fn put(&self, key: String, value: V) {
        let mut map = self.entries.lock();
        if map.len() >= self.max_entries {
            let ttl = self.ttl;
            map.retain(|_, (at, _)| at.elapsed() < ttl);
            if map.len() >= self.max_entries {
                map.clear();
            }
        }
        map.insert(key, (Instant::now(), value));
    }
}

/// A pooled connection to one pgSTAC database.
pub struct PgStac {
    pool: Pool,
    items: Option<TtlCache<Arc<Vec<MosaicItem>>>>,
    searches: Option<TtlCache<Arc<SearchRecord>>>,
    collections: Option<TtlCache<Arc<Value>>>,
    item_docs: Option<TtlCache<Arc<Value>>>,
    plans: Option<TtlCache<Arc<lean::Plan>>>,
    lean_mode: LeanMode,
    /// Whether the installed pgSTAC is a version the lean lookup was checked against.
    lean_checked: tokio::sync::OnceCell<bool>,
    lean_warned: std::sync::atomic::AtomicBool,
}

impl PgStac {
    /// Build the pool. No connection is opened until the first query, so a database
    /// that is down at start-up does not stop the server from serving other routes.
    pub fn new(config: PgStacConfig) -> Result<Self, PgStacError> {
        let connector = tls::make_connector(config.verify, config.root_cert.as_deref())?;
        let manager = Manager::from_config(
            config.pg.clone(),
            connector,
            ManagerConfig {
                recycling_method: RecyclingMethod::Fast,
            },
        );
        let pool = Pool::builder(manager)
            .max_size(config.max_connections)
            .runtime(Runtime::Tokio1)
            .wait_timeout(Some(config.acquire_timeout))
            .create_timeout(Some(config.acquire_timeout))
            .build()
            .map_err(|e| PgStacError::Config(e.to_string()))?;
        Ok(Self {
            pool,
            items: TtlCache::from_env(),
            searches: TtlCache::from_env(),
            collections: TtlCache::from_env(),
            item_docs: TtlCache::from_env(),
            plans: TtlCache::from_env(),
            lean_mode: config.lean,
            lean_checked: tokio::sync::OnceCell::new(),
            lean_warned: std::sync::atomic::AtomicBool::new(false),
        })
    }

    async fn client(&self) -> Result<deadpool_postgres::Object, PgStacError> {
        self.pool.get().await.map_err(|e| match e {
            deadpool_postgres::PoolError::Backend(db) => PgStacError::Db(db),
            other => PgStacError::Unavailable(other.to_string()),
        })
    }

    /// The installed pgSTAC version, which doubles as a connectivity check.
    pub async fn version(&self) -> Result<String, PgStacError> {
        let client = self.client().await?;
        let row = client.query_one("SELECT get_version()", &[]).await?;
        row.try_get(0).map_err(|e| PgStacError::Shape(e.to_string()))
    }

    /// Register a search (or return the existing row for an identical one), as
    /// titiler-pgstac's `POST /searches/register` does.
    pub async fn register_search(
        &self,
        search: &Value,
        metadata: &Value,
    ) -> Result<SearchRecord, PgStacError> {
        let client = self.client().await?;
        // Older pgSTAC versions have no readonly(); they are never read-only.
        if let Ok(row) = client.query_one("SELECT readonly()", &[]).await
            && row.try_get::<_, bool>(0).unwrap_or(false)
        {
            return Err(PgStacError::ReadOnly);
        }
        let sql = format!(
            "SELECT {SEARCH_COLUMNS} FROM search_query($1::jsonb, _metadata => $2::jsonb)"
        );
        let stmt = client.prepare_cached(&sql).await?;
        let row = client.query_one(&stmt, &[search, metadata]).await?;
        search_from_row(&row)
    }

    /// [`Self::register_search`] behind the time-limited cache, for searches that are
    /// derived from the request on every call (a collection's mosaic). `key` must
    /// identify the search and metadata.
    pub async fn register_search_cached(
        &self,
        key: &str,
        search: &Value,
        metadata: &Value,
    ) -> Result<Arc<SearchRecord>, PgStacError> {
        let key = format!("register|{key}");
        if let Some(hit) = self.searches.as_ref().and_then(|c| c.get(&key)) {
            return Ok(hit);
        }
        let rec = Arc::new(self.register_search(search, metadata).await?);
        if let Some(cache) = &self.searches {
            cache.put(key, rec.clone());
        }
        Ok(rec)
    }

    /// [`Self::get_collection`] behind the time-limited cache. Only collections that exist
    /// are remembered.
    pub async fn get_collection_cached(&self, id: &str) -> Result<Option<Arc<Value>>, PgStacError> {
        if let Some(hit) = self.collections.as_ref().and_then(|c| c.get(id)) {
            return Ok(Some(hit));
        }
        let Some(found) = self.get_collection(id).await? else {
            return Ok(None);
        };
        let found = Arc::new(found);
        if let Some(cache) = &self.collections {
            cache.put(id.to_string(), found.clone());
        }
        Ok(Some(found))
    }

    /// One item's full JSON, as pgSTAC's `get_item` returns it, or `None` when the collection
    /// has no item with that id.
    pub async fn get_item(&self, collection: &str, id: &str) -> Result<Option<Value>, PgStacError> {
        let client = self.client().await?;
        let stmt = client.prepare_cached("SELECT get_item($1::text, $2::text)").await?;
        let row = client.query_one(&stmt, &[&id, &collection]).await?;
        let value: Option<Value> = row.try_get(0).map_err(|e| PgStacError::Shape(e.to_string()))?;
        Ok(value.filter(|v| !v.is_null()))
    }

    /// [`Self::get_item`] behind the time-limited cache. Only items that exist are remembered.
    pub async fn get_item_cached(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<Option<Arc<Value>>, PgStacError> {
        // A NUL cannot occur in either id, so the key is unambiguous.
        let key = format!("{collection}\0{id}");
        if let Some(hit) = self.item_docs.as_ref().and_then(|c| c.get(&key)) {
            return Ok(Some(hit));
        }
        let Some(found) = self.get_item(collection, id).await? else {
            return Ok(None);
        };
        let found = Arc::new(found);
        if let Some(cache) = &self.item_docs {
            cache.put(key, found.clone());
        }
        Ok(Some(found))
    }

    /// A collection's JSON, or `None` when pgSTAC has no collection with that id.
    pub async fn get_collection(&self, id: &str) -> Result<Option<Value>, PgStacError> {
        let client = self.client().await?;
        let stmt = client.prepare_cached("SELECT get_collection($1::text)").await?;
        let row = client.query_one(&stmt, &[&id]).await?;
        let value: Option<Value> = row.try_get(0).map_err(|e| PgStacError::Shape(e.to_string()))?;
        Ok(value.filter(|v| !v.is_null()))
    }

    /// Look a registered search up by its hash.
    pub async fn get_search(&self, hash: &str) -> Result<Option<SearchRecord>, PgStacError> {
        let client = self.client().await?;
        let sql = format!("SELECT {SEARCH_COLUMNS} FROM searches WHERE hash = $1");
        let stmt = client.prepare_cached(&sql).await?;
        match client.query_opt(&stmt, &[&hash]).await? {
            Some(row) => search_from_row(&row).map(Some),
            None => Ok(None),
        }
    }

    /// [`Self::get_search`] behind the time-limited cache, for per-tile use. Only searches
    /// that exist are remembered, so one registered a moment ago is found at once.
    pub async fn get_search_cached(
        &self,
        hash: &str,
    ) -> Result<Option<Arc<SearchRecord>>, PgStacError> {
        if let Some(hit) = self.searches.as_ref().and_then(|c| c.get(hash)) {
            return Ok(Some(hit));
        }
        let Some(found) = self.get_search(hash).await? else {
            return Ok(None);
        };
        let found = Arc::new(found);
        if let Some(cache) = &self.searches {
            cache.put(hash.to_string(), found.clone());
        }
        Ok(Some(found))
    }

    /// Page through the searches registered as mosaics. Returns the page and the total.
    pub async fn list_searches(
        &self,
        limit: i64,
        offset: i64,
        order: &[SearchOrder],
        metadata: &[(String, String)],
    ) -> Result<(Vec<SearchRecord>, i64), PgStacError> {
        let client = self.client().await?;
        // Nothing of the request is written into the SQL: the metadata asked for and the
        // metadata keys sorted by are parameters, and the rest is fixed text. As in
        // titiler-pgstac a metadata value is compared as text, whatever its JSON type.
        let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = Vec::new();
        let mut filter = String::from("WHERE metadata->>'type' = 'mosaic'");
        for (key, value) in metadata {
            params.push(key);
            params.push(value);
            filter.push_str(&format!(" AND metadata->>(${}::text) = ${}::text", params.len() - 1, params.len()));
        }
        // The statements differ with the request, so they are not kept: a client cannot
        // grow the connection's statement cache by varying what it sorts and filters by.
        let matched: i64 = client
            .query_one(&format!("SELECT count(*) FROM searches {filter}"), &params)
            .await?
            .try_get(0)
            .map_err(|e| PgStacError::Shape(e.to_string()))?;
        let mut terms = Vec::with_capacity(order.len());
        for term in order {
            let column = match &term.field {
                SearchField::LastUsed => "lastused".to_string(),
                SearchField::UseCount => "usecount".to_string(),
                SearchField::Metadata(key) => {
                    params.push(key);
                    format!("metadata->>(${}::text)", params.len())
                }
            };
            terms.push(format!("{column} {}", if term.descending { "DESC" } else { "ASC" }));
        }
        let order_sql = if terms.is_empty() { String::new() } else { format!("ORDER BY {}", terms.join(", ")) };
        let sql = format!(
            "SELECT {SEARCH_COLUMNS} FROM searches {filter} {order_sql} LIMIT ${} OFFSET ${}",
            params.len() + 1,
            params.len() + 2
        );
        params.push(&limit);
        params.push(&offset);
        let rows = client.query(&sql, &params).await?;
        let searches = rows.iter().map(search_from_row).collect::<Result<Vec<_>, _>>()?;
        Ok((searches, matched))
    }

    /// The items of a search that cover `geometry` (GeoJSON, WGS84), in the search's
    /// order: pgSTAC's `geojsonsearch`, called as titiler-pgstac calls it.
    pub async fn items_for_geometry(
        &self,
        geometry: &Value,
        search_hash: &str,
        limits: SearchLimits,
    ) -> Result<Arc<Vec<MosaicItem>>, PgStacError> {
        let key = self.items.as_ref().map(|_| {
            format!(
                "{search_hash}|{}|{}|{}|{}|{}|{geometry}",
                limits.scan_limit,
                limits.items_limit,
                limits.time_limit_secs,
                limits.exitwhenfull,
                limits.skipcovered
            )
        });
        if let (Some(cache), Some(key)) = (&self.items, &key)
            && let Some(hit) = cache.get(key)
        {
            return Ok(hit);
        }
        let client = self.client().await?;
        let stmt = client
            .prepare_cached(
                "SELECT geojsonsearch($1::jsonb, $2::text, $3::jsonb, $4::int, $5::int, \
                 $6::text::interval, $7::bool, $8::bool)",
            )
            .await?;
        let fields = serde_json::json!({ "include": ["assets", "id", "bbox", "collection"] });
        let time_limit = format!("{} seconds", limits.time_limit_secs);
        let row = client
            .query_one(
                &stmt,
                &[
                    geometry,
                    &search_hash,
                    &fields,
                    &limits.scan_limit,
                    &limits.items_limit,
                    &time_limit,
                    &limits.exitwhenfull,
                    &limits.skipcovered,
                ],
            )
            .await?;
        let collection: Value = row.try_get(0).map_err(|e| PgStacError::Shape(e.to_string()))?;
        let items: Vec<MosaicItem> = collection
            .get("features")
            .and_then(Value::as_array)
            .map(|fs| fs.iter().filter_map(MosaicItem::from_feature).collect())
            .unwrap_or_default();
        let items = Arc::new(items);
        if let (Some(cache), Some(key)) = (&self.items, key) {
            cache.put(key, items.clone());
        }
        Ok(items)
    }

    /// The items a picture of `geometry` is made from: an id and, for each of `assets`,
    /// an href. `grid` is where the picture samples the geometry; an item is listed when
    /// its footprint holds a sample no earlier item's did. That leaves out the items
    /// `geojsonsearch` keeps for a sliver between samples and is otherwise its list.
    ///
    /// The answer is `geojsonsearch`'s own when the lean lookup is off, when `grid` is
    /// `None` for an area, and whenever the lean lookup stands aside or fails.
    pub async fn items_for_pixels(
        &self,
        geometry: &Value,
        search_hash: &str,
        limits: SearchLimits,
        assets: &[String],
        grid: Option<SampleGrid>,
        lookup: Lookup,
    ) -> Result<Arc<Vec<MosaicItem>>, PgStacError> {
        let lean = match lookup {
            Lookup::Lean => true,
            Lookup::Reference => false,
            Lookup::Configured => match self.lean_mode {
                LeanMode::On => true,
                LeanMode::Off => false,
                LeanMode::Auto => self.lean_checked().await,
            },
        };
        if !lean {
            return self.items_for_geometry(geometry, search_hash, limits).await;
        }
        let key = self.items.as_ref().map(|_| {
            format!(
                "lean|{search_hash}|{}|{}|{}|{}|{}|{grid:?}|{assets:?}|{geometry}",
                limits.scan_limit,
                limits.items_limit,
                limits.time_limit_secs,
                limits.exitwhenfull,
                limits.skipcovered
            )
        });
        if let (Some(cache), Some(key)) = (&self.items, &key)
            && let Some(hit) = cache.get(key)
        {
            return Ok(hit);
        }
        let items = match self.lean_items(geometry, search_hash, limits, assets, grid).await {
            Ok(Some(items)) => Arc::new(items),
            Ok(None) => self.items_for_geometry(geometry, search_hash, limits).await?,
            // No connection to be had: the reference path would only wait for one again.
            Err(e @ PgStacError::Unavailable(_)) => return Err(e),
            Err(e) => {
                use std::sync::atomic::Ordering;
                if self.lean_warned.swap(true, Ordering::Relaxed) {
                    tracing::debug!(error = %e, "lean pgSTAC lookup failed; answering with geojsonsearch");
                } else {
                    tracing::warn!(
                        error = %e,
                        "lean pgSTAC lookup failed; answering with geojsonsearch (later failures are logged at debug)"
                    );
                }
                self.items_for_geometry(geometry, search_hash, limits).await?
            }
        };
        // Whichever path answered, the answer is kept under this request's key, so one the
        // lean lookup cannot answer does not walk again on every repeat.
        if let (Some(cache), Some(key)) = (&self.items, key) {
            cache.put(key, items.clone());
        }
        Ok(items)
    }

    /// Whether the installed pgSTAC is a version the lean lookup was checked against. A
    /// version that cannot be read is not one, and the reference path serves the request.
    async fn lean_checked(&self) -> bool {
        if let Some(known) = self.lean_checked.get() {
            return *known;
        }
        let (checked, settled) = match self.version().await {
            Ok(version) => (lean_checked_version(&version), true),
            // The database answered and there is no version to read: that will not change.
            Err(PgStacError::Shape(_)) => (false, true),
            Err(PgStacError::Db(e)) if e.as_db_error().is_some() => (false, true),
            // The database could not be reached: ask again next time.
            Err(_) => (false, false),
        };
        if settled {
            let _ = self.lean_checked.set(checked);
        }
        checked
    }

    /// `None` when the lean lookup stands aside.
    async fn lean_items(
        &self,
        geometry: &Value,
        search_hash: &str,
        limits: SearchLimits,
        assets: &[String],
        grid: Option<SampleGrid>,
    ) -> Result<Option<Vec<MosaicItem>>, PgStacError> {
        // An unknown search is the reference path's error to raise.
        let Some(search) = self.get_search_cached(search_hash).await? else {
            return Ok(None);
        };
        let mut client = self.client().await?;
        let plan = match self.plans.as_ref().and_then(|c| c.get(search_hash)) {
            Some(plan) => plan,
            None => {
                let plan = Arc::new(lean::Plan::build(&client, &search).await?);
                if let Some(cache) = &self.plans {
                    cache.put(search_hash.to_string(), plan.clone());
                }
                plan
            }
        };
        match plan.items(&mut client, geometry, limits, assets, grid).await? {
            lean::Outcome::Items(items) => Ok(Some(items)),
            lean::Outcome::Aside(why) => {
                tracing::debug!(?why, "lean pgSTAC lookup stood aside");
                Ok(None)
            }
        }
    }
}

/// The lean lookup reads pgSTAC's tables directly, so it is on by default only for the
/// release lines it has been run against `geojsonsearch` on: 0.10 (a catalog of 46,476
/// items, and the HTTP tests) and 0.9 (the HTTP tests, in CI on 0.9.11).
fn lean_checked_version(version: &str) -> bool {
    version.starts_with("0.10.") || version.starts_with("0.9.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_become_items() {
        let f = serde_json::json!({
            "id": "S2B_31TGJ_20241231_0_L2A",
            "collection": "sentinel-2-l2a",
            "bbox": [6.8, 44.1, 7.9, 45.1],
            "assets": {
                "visual": { "href": "https://example.test/TCI.tif", "type": "image/tiff" },
                "thumbnail": { "href": "https://example.test/t.jpg" }
            }
        });
        let item = MosaicItem::from_feature(&f).unwrap();
        assert_eq!(item.id, "S2B_31TGJ_20241231_0_L2A");
        assert_eq!(item.bbox, Some([6.8, 44.1, 7.9, 45.1]));
        assert_eq!(item.asset_href("visual").as_deref(), Some("https://example.test/TCI.tif"));
        assert_eq!(item.asset_href("missing"), None);
        // A relative href is relative to the item's own address, and to nothing without one.
        let mut relative = item.clone();
        relative.assets.insert("visual".into(), serde_json::json!({ "href": "./data/TCI.tif" }));
        assert_eq!(relative.asset_href("visual"), None);
        relative.self_href = Some("https://example.test/c/items/i.json?x=1".into());
        assert_eq!(relative.asset_href("visual").as_deref(), Some("https://example.test/c/items/data/TCI.tif"));
        relative.assets.insert("visual".into(), serde_json::json!({ "href": "/data/TCI.tif" }));
        assert_eq!(relative.asset_href("visual").as_deref(), Some("https://example.test/data/TCI.tif"));
        let back = item.to_json();
        assert_eq!(back["collection"], "sentinel-2-l2a");
        assert_eq!(back["assets"]["visual"]["type"], "image/tiff");
    }

    #[test]
    fn three_dimensional_bbox_keeps_the_horizontal_extent() {
        let f = serde_json::json!({ "id": "a", "bbox": [1.0, 2.0, 0.0, 3.0, 4.0, 100.0] });
        assert_eq!(MosaicItem::from_feature(&f).unwrap().bbox, Some([1.0, 2.0, 3.0, 4.0]));
        let f = serde_json::json!({ "id": "a", "bbox": [1.0, 2.0] });
        assert_eq!(MosaicItem::from_feature(&f).unwrap().bbox, None);
        assert!(MosaicItem::from_feature(&serde_json::json!({ "bbox": [] })).is_none());
    }

    #[test]
    fn item_cache_expires_and_stays_bounded() {
        let cache = TtlCache {
            ttl: Duration::from_millis(30),
            max_entries: 2,
            entries: Mutex::new(Default::default()),
        };
        let v: Arc<Vec<MosaicItem>> = Arc::new(Vec::new());
        cache.put("a".into(), v.clone());
        cache.put("b".into(), v.clone());
        assert!(cache.get("a").is_some());
        cache.put("c".into(), v.clone());
        assert!(cache.entries.lock().len() <= 2);
        std::thread::sleep(Duration::from_millis(40));
        assert!(cache.get("c").is_none());
    }

    #[test]
    fn keys_sort_like_jsonb() {
        let mut keys: Vec<String> =
            ["visual", "red", "aot-jp2", "nir", "blue", "aot"].iter().map(|s| s.to_string()).collect();
        jsonb_key_order(&mut keys);
        assert_eq!(keys, ["aot", "nir", "red", "blue", "visual", "aot-jp2"]);
    }

    #[test]
    fn lean_lookup_is_on_for_the_lines_it_was_run_on() {
        for version in ["0.9.0", "0.9.11", "0.10.2"] {
            assert!(lean_checked_version(version), "{version}");
        }
        // Not 0.8, not what comes after 0.10, and not a version that merely starts alike.
        for version in ["0.8.6", "0.11.0", "0.90.1", "0.100.0", "1.0.0", "", "unreleased"] {
            assert!(!lean_checked_version(version), "{version}");
        }
    }

    #[test]
    fn default_limits_match_titiler_pgstac() {
        let l = SearchLimits::default();
        assert_eq!((l.scan_limit, l.items_limit, l.time_limit_secs), (10_000, 100, 5));
        assert!(l.exitwhenfull && l.skipcovered);
    }
}
