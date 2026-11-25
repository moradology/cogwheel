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
use tokio_postgres::types::{Timestamp, ToSql, Type};

use crate::coverage::{Coverage, SampleGrid, polygons_from_geojson, polygons_from_wkb};
use crate::{MosaicItem, PgStacError, SearchLimits, SearchRecord};

/// Rows asked for by the first query of a walk. Most tiles are full well within them.
const FIRST_PAGE: i64 = 512;

/// The most rows read from a cursor at once. A walk holds one page in memory at a time and
/// looks at the clock between pages, whatever `scan_limit` the request came with.
const MAX_PAGE: i32 = 4096;

/// A bound of a chunk. pgSTAC's are infinite for a collection that is not partitioned by time.
type Bound = Timestamp<SystemTime>;

/// The hrefs of the assets named in `$3`, in that order; null where an item has none.
const HREFS: &str = "ARRAY(SELECT content->'assets'->u.k->>'href' \
                     FROM unnest($3::text[]) WITH ORDINALITY AS u(k, n) ORDER BY u.n)";

/// What a search compiles to, and the time chunks its items are walked in.
pub(crate) struct Plan {
    sql: String,
    /// pgSTAC's chunks in walk order: start, end, and whether any partition had a row in
    /// it when the plan was made. `None` when the search does not sort by time: one
    /// query, no bounds.
    chunks: Option<Vec<(Bound, Bound, bool)>>,
    /// The walk runs from the newest chunk to the oldest.
    descending: bool,
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
                // partition has rows in (the month after the newest, always). Those are not
                // worth a query of their own, so each is marked and read along with a
                // neighbour. The check for a row is one index probe per chunk and does not
                // run the search's filter.
                let rows = client
                    .query(
                        &format!(
                            "SELECT c.s, c.e, EXISTS (SELECT 1 FROM items WHERE datetime >= c.s AND datetime < c.e) \
                             FROM chunker($1::text) c ORDER BY 1 {direction}"
                        ),
                        &[&where_clause],
                    )
                    .await?;
                let mut chunks = Vec::with_capacity(rows.len());
                for row in rows {
                    chunks.push((
                        row.try_get(0).map_err(shape)?,
                        row.try_get(1).map_err(shape)?,
                        row.try_get(2).map_err(shape)?,
                    ));
                }
                Some(chunks)
            }
            None => None,
        };
        // The filter is put together as `geometrysearch` and `partition_queries` do,
        // parenthesis for parenthesis. Without chunks the two times are not bounds, and are
        // compared with each other only so that both statements take the same parameters.
        let bounds = if chunks.is_some() {
            "datetime >= $5 AND datetime < $6 AND "
        } else {
            "$5::timestamptz <= $6::timestamptz AND "
        };
        // An item's document is large and stored out of line, so reading an href out of it
        // costs more than everything else a row needs. The first query of a walk reads
        // hrefs ($7), because most walks end in it; later ones leave them for the few rows kept.
        let sql = format!(
            "SELECT id, collection, st_asbinary(geometry), CASE WHEN $7 THEN {HREFS} END, tableoid::regclass::text \
             FROM items WHERE {bounds}({where_clause} AND st_intersects(geometry, st_geomfromgeojson($1::text))) \
             ORDER BY {orderby} LIMIT $2 OFFSET $4"
        );
        Ok(Self { sql, chunks, descending: direction == Some("DESC") })
    }

    /// The items of the search that add to `geometry`, in the search's order.
    pub(crate) async fn items(
        &self,
        client: &mut tokio_postgres::Client,
        geometry: &Value,
        limits: SearchLimits,
        assets: &[String],
        grid: Option<SampleGrid>,
    ) -> Result<Outcome, PgStacError> {
        // As in `geometrysearch`: no coverage arithmetic for a point or a line, and
        // skipping covered items implies stopping when full.
        let area = polygons_from_geojson(geometry);
        let skip_covered = limits.skipcovered && area.is_some();
        let exit_when_full = (limits.exitwhenfull || skip_covered) && area.is_some();
        let cover = match &area {
            Some(area) if exit_when_full => match grid.and_then(|grid| Coverage::new(area, grid)) {
                Some(cover) => Some(cover),
                None => return Ok(Outcome::Aside(Aside::Area)),
            },
            _ => None,
        };
        let mut walk = Walk {
            assets,
            cover,
            skip_covered,
            items_limit: limits.items_limit as i64,
            scan_limit: limits.scan_limit as i64,
            time_limit: Duration::from_secs(limits.time_limit_secs.max(0) as u64),
            started: Instant::now(),
            kept: Vec::new(),
            owed: Vec::new(),
            counter: 1,
            scanned: 1,
            queries: 0,
            in_queries: Duration::ZERO,
        };
        let geometry_text = geometry.to_string();
        let asset_keys: Vec<&str> = assets.iter().map(String::as_str).collect();

        // pgSTAC opens a cursor per chunk and stops reading when it has enough. Here the
        // chunks are read in runs of one, two, four and so on, each run one query over its
        // whole stretch of time: the rows come in the same order, a tile that fills early
        // costs one small query, and one that never fills costs a handful, not one per chunk.
        let mut runs = Runs { plan: self, next: 0, span: 1 };
        let mut step = Step::On;
        if let Some((start, end)) = runs.next()
            && walk.scan_limit > 0
        {
            // The first query stands alone, with no transaction around it, and asks for a
            // page with hrefs: most walks end here.
            let (page, offset, with_hrefs) = (FIRST_PAGE.min(walk.scan_limit), 0i64, true);
            let params: [(&(dyn ToSql + Sync), Type); 7] = [
                (&geometry_text, Type::TEXT),
                (&page, Type::INT8),
                (&asset_keys, Type::TEXT_ARRAY),
                (&offset, Type::INT8),
                (start, Type::TIMESTAMPTZ),
                (end, Type::TIMESTAMPTZ),
                (&with_hrefs, Type::BOOL),
            ];
            let asked = Instant::now();
            let rows = client.query_typed(&self.sql, &params).await?;
            walk.in_queries += asked.elapsed();
            walk.queries += 1;
            step = walk.take(&rows)?;
            let more_in_run = rows.len() as i64 == page && page < walk.scan_limit;
            if matches!(step, Step::On) && (more_in_run || runs.more()) {
                // The rest is read as pgSTAC reads it, through cursors: a page at a time,
                // nothing run twice, and no more held in memory than a page.
                let tx = client.transaction().await?;
                let query = (geometry_text.as_str(), asset_keys.as_slice());
                if more_in_run {
                    let rest = walk.scan_limit - page;
                    step = self.read(&tx, &mut walk, query, (start, end), page, rest).await?;
                }
                while matches!(step, Step::On) {
                    // As pgSTAC limits each chunk's query: what the counter leaves of the scan limit.
                    let remaining = walk.scan_limit - walk.scanned;
                    let Some(run) = runs.next().filter(|_| remaining > 0) else {
                        break;
                    };
                    step = self.read(&tx, &mut walk, query, run, 0, remaining).await?;
                }
                tx.commit().await?;
            }
        }
        if let Step::Aside(why) = step {
            return Ok(Outcome::Aside(why));
        }
        if !walk.owed.is_empty() {
            let asked = Instant::now();
            let found = owed_hrefs(client, &mut walk.kept, &walk.owed, assets).await?;
            walk.in_queries += asked.elapsed();
            walk.queries += 1;
            if !found {
                return Ok(Outcome::Aside(Aside::Asset));
            }
        }
        tracing::debug!(
            queries = walk.queries,
            scanned = walk.scanned - 1,
            kept = walk.kept.len(),
            ms = walk.started.elapsed().as_secs_f64() * 1e3,
            query_ms = walk.in_queries.as_secs_f64() * 1e3,
            "lean pgSTAC lookup"
        );
        Ok(Outcome::Items(walk.kept))
    }

    /// Read one run through a cursor, from `offset`, up to `limit` rows.
    async fn read(
        &self,
        tx: &tokio_postgres::Transaction<'_>,
        walk: &mut Walk<'_>,
        (geometry, asset_keys): (&str, &[&str]),
        (start, end): (&Bound, &Bound),
        offset: i64,
        limit: i64,
    ) -> Result<Step, PgStacError> {
        let mut asked = Instant::now();
        // Prepared afresh each time, so Postgres plans it for these bounds and prunes to
        // their partitions instead of reusing a plan over all of them.
        let portal = tx
            .bind(self.sql.as_str(), &[&geometry, &limit, &asset_keys, &offset, start, end, &false])
            .await?;
        walk.queries += 1;
        loop {
            let rows = tx.query_portal(&portal, MAX_PAGE).await?;
            walk.in_queries += asked.elapsed();
            match walk.take(&rows)? {
                Step::On => {}
                other => return Ok(other),
            }
            if rows.len() < MAX_PAGE as usize {
                return Ok(Step::On);
            }
            // pgSTAC looks at the clock only when it keeps a row. Looking between pages as
            // well bounds a walk that keeps nothing.
            if walk.started.elapsed() > walk.time_limit {
                return Ok(Step::Done);
            }
            asked = Instant::now();
        }
    }
}

/// The runs of a plan's chunks: one chunk with rows, then two, then four, each with the
/// empty chunks around it. An empty chunk costs a run nothing and is still read, so a row
/// that has arrived in one since the plan was made is not missed.
struct Runs<'a> {
    plan: &'a Plan,
    next: usize,
    span: usize,
}

impl<'a> Runs<'a> {
    fn chunks(&self) -> &'a [(Bound, Bound, bool)] {
        const UNBOUNDED: &[(Bound, Bound, bool)] =
            &[(Timestamp::Value(SystemTime::UNIX_EPOCH), Timestamp::Value(SystemTime::UNIX_EPOCH), true)];
        self.plan.chunks.as_deref().unwrap_or(UNBOUNDED)
    }

    fn more(&self) -> bool {
        self.next < self.chunks().len()
    }

    /// The time bounds of the next run.
    fn next(&mut self) -> Option<(&'a Bound, &'a Bound)> {
        let chunks = self.chunks();
        let from = self.next;
        let mut with_rows = 0;
        while self.next < chunks.len() && (with_rows < self.span || !chunks[self.next].2) {
            with_rows += usize::from(chunks[self.next].2);
            self.next += 1;
        }
        self.span *= 2;
        // Chunks follow one another in time, so a run's bounds are its two ends'.
        let (first, last) = (chunks.get(from)?, &chunks[self.next - 1]);
        Some(if self.plan.descending { (&last.0, &first.1) } else { (&first.0, &last.1) })
    }
}

/// What the rows read so far did to a walk.
enum Step {
    /// Read on.
    On,
    /// The walk is over: full, or at a limit.
    Done,
    Aside(Aside),
}

/// The state of `geometrysearch`'s loop.
struct Walk<'a> {
    assets: &'a [String],
    cover: Option<Coverage>,
    skip_covered: bool,
    items_limit: i64,
    scan_limit: i64,
    time_limit: Duration,
    started: Instant,
    kept: Vec<MosaicItem>,
    /// Items kept without their hrefs: where they are in `kept`, and the table they came from.
    owed: Vec<(usize, String)>,
    /// pgSTAC's counters, which start at one.
    counter: i64,
    scanned: i64,
    queries: u32,
    in_queries: Duration,
}

impl Walk<'_> {
    /// The body of `geometrysearch`'s loop, for each of `rows` in turn.
    fn take(&mut self, rows: &[tokio_postgres::Row]) -> Result<Step, PgStacError> {
        let shape = |e: tokio_postgres::Error| PgStacError::Shape(e.to_string());
        for row in rows {
            if let Some(cover) = &mut self.cover {
                let footprint: &[u8] = row.try_get(2).map_err(shape)?;
                let Some(polygons) = polygons_from_wkb(footprint) else {
                    return Ok(Step::Aside(Aside::Footprint));
                };
                if cover.add(&polygons) == 0 && self.skip_covered {
                    self.scanned += 1;
                    continue;
                }
            }
            let mut item = MosaicItem {
                id: row.try_get(0).map_err(shape)?,
                collection: Some(row.try_get(1).map_err(shape)?),
                bbox: None,
                assets: Default::default(),
                self_href: None,
            };
            match row.try_get::<_, Option<Vec<Option<String>>>>(3).map_err(shape)? {
                Some(hrefs) => {
                    if !set_hrefs(&mut item, self.assets, hrefs) {
                        return Ok(Step::Aside(Aside::Asset));
                    }
                }
                None if self.assets.is_empty() => {}
                None => self.owed.push((self.kept.len(), row.try_get(4).map_err(shape)?)),
            }
            self.kept.push(item);
            if self.counter >= self.items_limit
                || self.scanned > self.scan_limit
                || self.started.elapsed() > self.time_limit
                || self.cover.as_ref().is_some_and(Coverage::is_full)
            {
                return Ok(Step::Done);
            }
            self.counter += 1;
            self.scanned += 1;
        }
        Ok(Step::On)
    }
}

/// Give `item` the hrefs of `assets`. `false` when it lacks one.
fn set_hrefs(item: &mut MosaicItem, assets: &[String], hrefs: Vec<Option<String>>) -> bool {
    if hrefs.len() != assets.len() {
        return false;
    }
    for (key, href) in assets.iter().zip(hrefs) {
        // A relative href needs the item's `self` link, which is not read here: the
        // reference path, which returns whole items, answers for it.
        let Some(href) = href.filter(|h| crate::href_is_absolute(h)) else {
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
