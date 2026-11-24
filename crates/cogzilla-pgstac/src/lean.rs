//! The lean item lookup: what `geometrysearch` does, without the plpgsql.
//!
//! pgSTAC lists the items of a tile by walking the search's time chunks in sort order,
//! one query per chunk, hydrating each row it keeps into a temporary table and deciding
//! what to keep with a growing `ST_Union`. This module walks the same chunks with the
//! same `WHERE` and `ORDER BY`, both taken from pgSTAC, but asks each row only for its
//! id, footprint and the hrefs of the assets wanted, and decides what to keep on a grid
//! of sample points ([`crate::coverage`]). It reads only, and installs nothing.
//!
//! It is never the only path: whatever it cannot answer, `geojsonsearch` answers.

use std::time::{Duration, Instant, SystemTime};

use serde_json::Value;
use tokio_postgres::types::{ToSql, Type};

use crate::coverage::{Coverage, SampleGrid, polygons_from_geojson, polygons_from_wkb};
use crate::{MosaicItem, PgStacError, SearchLimits, SearchRecord};

/// Rows asked for by the first query of a walk. Most tiles are full well within them.
const FIRST_PAGE: i64 = 512;

/// The hrefs of the assets named in `$3`, in that order; null where an item has none.
const HREFS: &str = "ARRAY(SELECT content->'assets'->u.k->>'href' \
                     FROM unnest($3::text[]) WITH ORDINALITY AS u(k, n) ORDER BY u.n)";

/// What a search compiles to, and the time chunks its items are walked in.
pub(crate) struct Plan {
    sql: String,
    /// pgSTAC's chunks that hold any item at all, in walk order. `None` when the search
    /// does not sort by time: one query, no bounds.
    chunks: Option<Vec<(SystemTime, SystemTime)>>,
}

/// Why the lean lookup stood aside for a request. Not an error: the reference path answers.
#[derive(Debug)]
pub(crate) enum Aside {
    /// The request's area holds no sample point, or the request came without a grid.
    Area,
    /// A footprint this does not read (a curve, a surface).
    Footprint,
    /// An item kept has no href for an asset asked for.
    Asset,
}

pub(crate) enum Outcome {
    Items(Vec<MosaicItem>),
    Aside(Aside),
}

impl Plan {
    /// Compile `search` with pgSTAC's own functions. `search_query` is what
    /// `geometrysearch` calls on every tile; it also notes the search as used.
    pub(crate) async fn build(
        client: &tokio_postgres::Client,
        search: &SearchRecord,
    ) -> Result<Self, PgStacError> {
        let row = client
            .query_one(
                "SELECT s._where, s.orderby FROM search_query($1::jsonb, false, $2::jsonb) s",
                &[&search.search, &search.metadata],
            )
            .await?;
        let shape = |e: tokio_postgres::Error| PgStacError::Shape(e.to_string());
        let where_clause: Option<String> = row.try_get(0).map_err(shape)?;
        let orderby: Option<String> = row.try_get(1).map_err(shape)?;
        let where_clause = where_clause.filter(|w| !w.trim().is_empty()).unwrap_or_else(|| " TRUE ".into());
        let orderby = orderby
            .filter(|o| !o.trim().is_empty())
            .ok_or_else(|| PgStacError::Shape("search_query returned no ORDER BY".into()))?;
        // `partition_queries` chooses by the same prefix test.
        let lead = orderby.trim_start().to_ascii_lowercase();
        let direction = match () {
            _ if lead.starts_with("datetime d") => Some("DESC"),
            _ if lead.starts_with("datetime a") => Some("ASC"),
            _ => None,
        };
        let chunks = match direction {
            Some(direction) => {
                // `chunker` cuts time at every partition boundary, which leaves chunks no
                // partition has rows in (the month after the newest, always). The check for
                // a row is one index probe per chunk and does not run the search's filter.
                let rows = client
                    .query(
                        &format!(
                            "SELECT c.s, c.e FROM chunker($1::text) c \
                             WHERE EXISTS (SELECT 1 FROM items WHERE datetime >= c.s AND datetime < c.e) \
                             ORDER BY 1 {direction}"
                        ),
                        &[&where_clause],
                    )
                    .await?;
                let mut chunks = Vec::with_capacity(rows.len());
                for row in rows {
                    chunks.push((row.try_get(0).map_err(shape)?, row.try_get(1).map_err(shape)?));
                }
                Some(chunks)
            }
            None => None,
        };
        // The filter is put together as `geometrysearch` and `partition_queries` do,
        // parenthesis for parenthesis. Without chunks the two times are not bounds, and are
        // compared with each other only so that both statements take the same parameters.
        let bounds = if chunks.is_some() { "datetime >= $5 AND datetime < $6 AND " } else { "$5 <= $6 AND " };
        // An item's document is large and stored out of line, so reading an href out of it
        // costs more than everything else a row needs. The first query of a walk reads
        // hrefs ($7), because most walks end in it; later ones leave them for the few rows kept.
        let sql = format!(
            "SELECT id, collection, st_asbinary(geometry), CASE WHEN $7 THEN {HREFS} END, tableoid::regclass::text \
             FROM items WHERE {bounds}({where_clause} AND st_intersects(geometry, st_geomfromgeojson($1::text))) \
             ORDER BY {orderby} LIMIT $2 OFFSET $4"
        );
        Ok(Self { sql, chunks })
    }

    /// The items of the search that add to `geometry`, in the search's order.
    pub(crate) async fn items(
        &self,
        client: &tokio_postgres::Client,
        geometry: &Value,
        limits: SearchLimits,
        assets: &[String],
        grid: Option<SampleGrid>,
    ) -> Result<Outcome, PgStacError> {
        let started = Instant::now();
        let time_limit = Duration::from_secs(limits.time_limit_secs.max(0) as u64);
        // As in `geometrysearch`: no coverage arithmetic for a point or a line, and
        // skipping covered items implies stopping when full.
        let area = polygons_from_geojson(geometry);
        let skip_covered = limits.skipcovered && area.is_some();
        let exit_when_full = (limits.exitwhenfull || skip_covered) && area.is_some();
        let mut cover = match &area {
            Some(area) if exit_when_full => match grid.and_then(|grid| Coverage::new(area, grid)) {
                Some(cover) => Some(cover),
                None => return Ok(Outcome::Aside(Aside::Area)),
            },
            _ => None,
        };
        let geometry_text = geometry.to_string();
        let asset_keys: Vec<&str> = assets.iter().map(String::as_str).collect();
        let scan_limit = limits.scan_limit as i64;
        let items_limit = limits.items_limit as i64;

        let unbounded = [(SystemTime::UNIX_EPOCH, SystemTime::UNIX_EPOCH)];
        let chunks: &[(SystemTime, SystemTime)] = self.chunks.as_deref().unwrap_or(&unbounded);
        let mut kept: Vec<MosaicItem> = Vec::new();
        // Items kept without their hrefs: where they are in `kept`, and the table they came from.
        let mut owed: Vec<(usize, String)> = Vec::new();
        // pgSTAC's counters, which start at one.
        let (mut counter, mut scanned) = (1i64, 1i64);
        let (mut queries, mut next, mut span) = (0u32, 0usize, 1usize);
        let mut in_queries = Duration::ZERO;
        // pgSTAC opens a cursor per chunk and stops reading when it has enough. Here the
        // chunks are read in runs of one, two, four and so on, each run one query over its
        // whole stretch of time: the rows come in the same order, a tile that fills early
        // costs one small query, and one that never fills costs a handful, not one per chunk.
        'walk: while next < chunks.len() {
            let run = &chunks[next..(next + span).min(chunks.len())];
            next += run.len();
            span *= 2;
            // As pgSTAC limits each chunk's query: the whole scan limit for the first, what
            // the counter leaves of it after.
            let remaining = if queries == 0 { scan_limit } else { scan_limit - scanned };
            if remaining <= 0 {
                break;
            }
            let start = run.iter().map(|c| c.0).min().unwrap_or(SystemTime::UNIX_EPOCH);
            let end = run.iter().map(|c| c.1).max().unwrap_or(SystemTime::UNIX_EPOCH);
            // Only the first query is paged: later runs are read whole, up to the scan limit.
            let (mut offset, mut page) = (0i64, if queries == 0 { FIRST_PAGE.min(remaining) } else { remaining });
            loop {
                let with_hrefs = queries == 0;
                let params: [(&(dyn ToSql + Sync), Type); 7] = [
                    (&geometry_text, Type::TEXT),
                    (&page, Type::INT8),
                    (&asset_keys, Type::TEXT_ARRAY),
                    (&offset, Type::INT8),
                    (&start, Type::TIMESTAMPTZ),
                    (&end, Type::TIMESTAMPTZ),
                    (&with_hrefs, Type::BOOL),
                ];
                let asked = Instant::now();
                let rows = client.query_typed(&self.sql, &params).await?;
                in_queries += asked.elapsed();
                queries += 1;
                let fetched = rows.len() as i64;
                for row in &rows {
                    let shape = |e: tokio_postgres::Error| PgStacError::Shape(e.to_string());
                    if let Some(cover) = &mut cover {
                        let footprint: &[u8] = row.try_get(2).map_err(shape)?;
                        let Some(polygons) = polygons_from_wkb(footprint) else {
                            return Ok(Outcome::Aside(Aside::Footprint));
                        };
                        if cover.add(&polygons) == 0 && skip_covered {
                            scanned += 1;
                            continue;
                        }
                    }
                    let mut item = MosaicItem {
                        id: row.try_get(0).map_err(shape)?,
                        collection: Some(row.try_get(1).map_err(shape)?),
                        bbox: None,
                        assets: Default::default(),
                    };
                    match row.try_get::<_, Option<Vec<Option<String>>>>(3).map_err(shape)? {
                        Some(hrefs) => {
                            if !set_hrefs(&mut item, assets, hrefs) {
                                return Ok(Outcome::Aside(Aside::Asset));
                            }
                        }
                        None if assets.is_empty() => {}
                        None => owed.push((kept.len(), row.try_get(4).map_err(shape)?)),
                    }
                    kept.push(item);
                    if counter >= items_limit
                        || scanned > scan_limit
                        || started.elapsed() > time_limit
                        || cover.as_ref().is_some_and(Coverage::is_full)
                    {
                        break 'walk;
                    }
                    counter += 1;
                    scanned += 1;
                }
                offset += fetched;
                // The run is done when a page comes back short, or the scan limit is read.
                if fetched < page || offset >= remaining {
                    break;
                }
                page = remaining - offset;
            }
        }
        if !owed.is_empty() {
            let asked = Instant::now();
            let found = owed_hrefs(client, &mut kept, &owed, assets).await?;
            in_queries += asked.elapsed();
            queries += 1;
            if !found {
                return Ok(Outcome::Aside(Aside::Asset));
            }
        }
        tracing::debug!(
            queries,
            scanned = scanned - 1,
            kept = kept.len(),
            ms = started.elapsed().as_secs_f64() * 1e3,
            query_ms = in_queries.as_secs_f64() * 1e3,
            "lean pgSTAC lookup"
        );
        Ok(Outcome::Items(kept))
    }
}

/// Give `item` the hrefs of `assets`. `false` when it lacks one.
fn set_hrefs(item: &mut MosaicItem, assets: &[String], hrefs: Vec<Option<String>>) -> bool {
    if hrefs.len() != assets.len() {
        return false;
    }
    for (key, href) in assets.iter().zip(hrefs) {
        let Some(href) = href else {
            return false;
        };
        item.assets.insert(key.clone(), serde_json::json!({ "href": href }));
    }
    true
}

/// Read the hrefs of the items in `owed`, each from the table its row came from: one
/// statement, a primary-key lookup per table. `false` when an item is gone or lacks an asset.
async fn owed_hrefs(
    client: &tokio_postgres::Client,
    kept: &mut [MosaicItem],
    owed: &[(usize, String)],
    assets: &[String],
) -> Result<bool, PgStacError> {
    let asset_keys: Vec<&str> = assets.iter().map(String::as_str).collect();
    // Table name to the ids wanted from it. The names are Postgres's own rendering of a
    // table (`tableoid::regclass`), quoted where they need to be.
    let mut tables: Vec<(&str, Vec<String>)> = Vec::new();
    for (at, table) in owed {
        let id = kept[*at].id.clone();
        match tables.iter_mut().find(|(name, _)| name == table) {
            Some((_, ids)) => ids.push(id),
            None => tables.push((table.as_str(), vec![id])),
        }
    }
    let mut sql = String::new();
    let mut params: Vec<(&(dyn ToSql + Sync), Type)> = vec![(&asset_keys, Type::TEXT_ARRAY)];
    for (n, (table, ids)) in tables.iter().enumerate() {
        if n > 0 {
            sql.push_str(" UNION ALL ");
        }
        // `$3` in HREFS is the asset list; here it is the first parameter.
        let hrefs = HREFS.replace("$3", "$1");
        sql.push_str(&format!("SELECT {n}::int4, id, {hrefs} FROM {table} WHERE id = ANY(${})", n + 2));
        params.push((ids, Type::TEXT_ARRAY));
    }
    let rows = client.query_typed(&sql, &params).await?;
    let shape = |e: tokio_postgres::Error| PgStacError::Shape(e.to_string());
    let mut filled = 0;
    for row in &rows {
        let table: i32 = row.try_get(0).map_err(shape)?;
        let id: &str = row.try_get(1).map_err(shape)?;
        let hrefs: Vec<Option<String>> = row.try_get(2).map_err(shape)?;
        let Some((name, _)) = tables.get(table as usize) else {
            continue;
        };
        let Some((at, _)) = owed.iter().find(|(at, from)| from == name && kept[*at].id == id) else {
            continue;
        };
        if !set_hrefs(&mut kept[*at], assets, hrefs) {
            return Ok(false);
        }
        filled += 1;
    }
    Ok(filled == owed.len())
}
