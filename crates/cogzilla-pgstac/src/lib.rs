//! A small client for a pgSTAC database.
//!
//! This is the compatibility path: it runs the same SQL titiler-pgstac runs
//! (`search_query` to register a search, `geojsonsearch` to list the items that cover a
//! geometry), so the items a tile is built from are pgSTAC's own answer. It needs read
//! access to the `pgstac` schema and installs nothing in the database.

mod config;
mod tls;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod, Runtime};
use parking_lot::Mutex;
use serde_json::Value;
use thiserror::Error;

pub use config::PgStacConfig;
pub use tls::TlsVerify;

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
        Some(Self {
            id,
            collection,
            bbox,
            assets,
        })
    }

    /// The `href` of one asset, if the item has it.
    pub fn asset_href(&self, key: &str) -> Option<&str> {
        self.assets.get(key)?.get("href")?.as_str()
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

/// How a search listing is ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchOrder {
    LastUsed { descending: bool },
    UseCount { descending: bool },
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
}

impl PgStac {
    /// Build the pool. No connection is opened until the first query, so a database
    /// that is down at start-up does not stop the server from serving other routes.
    pub fn new(config: PgStacConfig) -> Result<Self, PgStacError> {
        let connector = tls::make_connector(config.verify)?;
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
        order: Option<SearchOrder>,
    ) -> Result<(Vec<SearchRecord>, i64), PgStacError> {
        let client = self.client().await?;
        // The ORDER BY text is chosen from fixed strings, never from the request.
        let order_sql = match order {
            Some(SearchOrder::LastUsed { descending: true }) => "ORDER BY lastused DESC",
            Some(SearchOrder::LastUsed { descending: false }) => "ORDER BY lastused ASC",
            Some(SearchOrder::UseCount { descending: true }) => "ORDER BY usecount DESC",
            Some(SearchOrder::UseCount { descending: false }) => "ORDER BY usecount ASC",
            None => "",
        };
        let sql = format!(
            "SELECT {SEARCH_COLUMNS} FROM searches WHERE metadata->>'type' = 'mosaic' \
             {order_sql} LIMIT $1 OFFSET $2"
        );
        let stmt = client.prepare_cached(&sql).await?;
        let rows = client.query(&stmt, &[&limit, &offset]).await?;
        let searches = rows.iter().map(search_from_row).collect::<Result<Vec<_>, _>>()?;
        let count = client
            .prepare_cached("SELECT count(*) FROM searches WHERE metadata->>'type' = 'mosaic'")
            .await?;
        let matched: i64 = client
            .query_one(&count, &[])
            .await?
            .try_get(0)
            .map_err(|e| PgStacError::Shape(e.to_string()))?;
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
        assert_eq!(item.asset_href("visual"), Some("https://example.test/TCI.tif"));
        assert_eq!(item.asset_href("missing"), None);
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
    fn default_limits_match_titiler_pgstac() {
        let l = SearchLimits::default();
        assert_eq!((l.scan_limit, l.items_limit, l.time_limit_secs), (10_000, 100, 5));
        assert!(l.exitwhenfull && l.skipcovered);
    }
}
