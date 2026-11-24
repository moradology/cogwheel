# Plan: complete TiTiler parity, with pgSTAC built in

Written 2026-11-23 against cogzilla `ea0d248` (v0.5.2), TiTiler 2.3.0, titiler-pgstac 3.2.2 and pgSTAC 0.10.2.

## Progress

- **2026-11-24, E1 part one.** The `cogzilla-pgstac` crate and the core `/searches` routes (register, list, info, TileJSON, map, tiles, tile assets) are in, on the compatibility path. `benches/pgstac_diff_oracle.py` compares them with a live titiler-pgstac: 51 of 51 checks agree, including identical search ids and identical item lists per tile. Still to do in E1: point, bbox, feature, statistics, WMTS and `/tiles` listings on `/searches`, then the `/collections/{id}` and item families. Mosaic options that need workstreams B and D (pixel selection, per-asset options, mixed grids) answer `400` until then.
- **2026-11-24, E1 part two.** The `/collections/{id}` mosaic routes (info, TileJSON, map, tiles, tile assets), the tileset list and description (`/tiles`, `/tiles/{tms}`) and the item lists for a box or a point (`bbox/.../assets`, `point/.../assets`) are in for both families. A collection request registers the same search, under the same hash and metadata (including `defaults` from the collection's renders), as titiler-pgstac does, and `info` and `register` carry the per-layer links titiler-pgstac builds from `defaults`. The oracle is at 101 of 101. Still to do in E1: point values, `bbox` and `feature` images, statistics and WMTS for both families, CQL2 text filters on collection routes, and the `/collections/{id}/items/{id}` family. Point values could not be compared on the reference data: titiler-pgstac answers `500` there because an item under the point has a JPEG 2000 asset.
- **2026-11-24, E1 part three.** The `/collections/{id}/items/{id}` family runs on the `/stac` handlers with the item read from pgSTAC. An item layer in front of them supplies what titiler-pgstac does differently: required `assets`, the asset list, `404` for unknown items and for tiles outside the item, JSON errors, format from the pixels, and bounds and zooms from the item's projection metadata by rio-tiler's rule. The oracle is at 128 of 133 with 5 known differences, all inherited from `/stac` (preview sizing, statistics sample size and band description) or deliberate (a point outside the item is `404`, not `500`). Still to do in E1: point values, `bbox` and `feature` images, statistics and WMTS on the mosaic families, CQL2 text filters on collection routes, item `renders` with content, and WMTS for items.
- **2026-11-24, E1 part four.** `WMTSCapabilities.xml` for searches, collections and items: one layer per render in `defaults` plus one from the query string, limits from the mosaic's or item's bounds and zooms, `tile_format`, `use_epsg` and `zooms`. The oracle is at 144 of 149 with the same 5 known differences. Still to do in E1: point values, `bbox` and `feature` images and statistics on the mosaic families, CQL2 text filters on collection routes, and item `renders` with content.
- **2026-11-24, E1 part five.** `point/{lon},{lat}` on searches and collections returns the values of every item under the point, read eight items at a time. The oracle is at 155 of 160 with the same 5 known differences. Without `items_limit` the reference search reaches items whose assets are JPEG 2000; titiler-pgstac answers `500` there and cogzilla an error too, so the comparison uses limits of one to three. Still to do in E1: `bbox` and `feature` images and statistics on the mosaic families, `expression` on mosaic points, CQL2 text filters on collection routes, and item `renders` with content.
- **2026-11-24, E1 part six.** `bbox` and `feature` images on searches and collections: items are drawn in the search's order onto the first item's grid, each filling what the ones before left empty, with rio-tiler's sizing rules. The oracle is at 177 of 183 with 6 known differences. Still to do in E1: statistics on the mosaic families, `expression` on mosaic points, CQL2 text filters on collection routes, and item `renders` with content.
- **2026-11-24, E1 part ten.** Per-asset options (`assets=name|bidx=...`, `assets=name|expression=...`) and `asset_as_band` on the mosaic routes, with TiTiler 2's rules: a top-level expression is over `b1..bn` and never over asset names, and a bare `bidx` is passed over. `point` and `statistics` take every combination; tiles and images take what the renderer can draw (one asset with `bidx` or one expression, several with `bidx=1`) and refuse the rest. The oracle is at 264 of 271 with 7 known differences. That closes the E1 list; what remains on these routes is under "Not there yet" in `docs/api.md`.
- **2026-11-24, E1 loose ends.** A route audit against titiler-pgstac's own OpenAPI document: of its 72 operations, the 63 under `/searches` and `/collections` all answer here and are now all listed in `/openapi.json` (16 were served but not listed). The mosaic `feature` routes take a GeoJSON Feature only, as TiTiler does: a bare geometry, a FeatureCollection or any other object is a `422`. The three operations that do not answer are `/algorithms`, `/algorithms/{algorithmId}` and `/conformance`, which belong to workstream C. The oracle is at 268 of 275 with the same 7 known differences.
- **2026-11-24, E2.** The lean item lookup (`crates/cogzilla-pgstac/src/lean.rs`, `coverage.rs`) lists the items of tiles, `bbox` and `feature` images and points on pgSTAC 0.10 by default; `COGZILLA_PGSTAC_LEAN` and `item_lookup` choose otherwise, and `geojsonsearch` answers whatever it cannot. Measured with `benches/pgstac_lean_check.py` over 279 tiles and three searches: for the whole collection, newest first, median lookup 1.3 ms against 17.5 ms at z 8 and above and 29 ms against 1.2 s at z 5 to 7; oldest first, 1.2 ms against 21 ms and 40 ms against 2.3 s; with a cloud-cover filter, 5.2 ms against 68 ms and 67 ms against 1.4 s. A warm z 12 tile request is 5 ms where it was 27 ms. 236 lists identical; the other 43 are `geojsonsearch`'s list in its order with 903 items left out (median first-covered area 0.00003 of a pixel, largest 0.5), plus items `geojsonsearch` never reached on the 13 tiles where it stopped at a limit. The oracle is at 281 of 288 with the same 7 known differences. What E2 did not settle is listed under it below.
- **Found while doing it: partial band reads.** rio-tiler's `tile(indexes=(1,))` on a three-band raster differs from `tile(indexes=(1,2,3))[0]` in 43% of pixels. With every band asked for, GDAL warps once, straight onto the output grid; with only some, it reads the warped raster block by block (512 wide) through its own overview and resizes that. The statistics reader follows both roads and matches exactly; the tile renderer has one road, so `visual|bidx=1` tiles are a known difference.
- **Found while doing it: one band of an 8-bit raster.** In the shared renderer a single band of an 8-bit RGB raster does not come out as the same channel of the RGB picture. On a `/searches` tile, `visual|bidx=1` is one level lower than the red channel of `visual` in 17% of the pixels. On `/cog/tiles` with `bidx=1` it is one level lower in 80% and two lower in 16%, with or without `rescale=0,255`. Not looked into further; it belongs to workstream C.
- **2026-11-24, E1 part nine.** CQL2 text filters on collection routes, the default `filter-lang` in titiler-pgstac. A converter written for this (`http/cql2_text.rs`) gives the JSON of the `cql2` library titiler-pgstac uses, checked against 464 expressions run through that library: 335 it accepts come out identical, 5 it accepts by accident are rejected, and the rest are rejected by both. The oracle is at 232 of 238 with the same 6 known differences. The library itself was not taken as a dependency: it brings a JSON-schema validator, a SQL parser and `serde_json`'s `preserve_order`, which would change the key order of every response. Still to do in E1: `asset_as_band` and per-asset options.
- **2026-11-24, E1 part eight.** `POST statistics` on search and collection mosaics, for a Feature or a FeatureCollection. The route has its own nearest-neighbour reader that follows GDAL's warp step by step: the default WGS84 grid of the first raster (edge points every fifty pixels), bounds grown to whole pixels, GDAL's choice of overview, its straight-line approximation of each output row, and its exact second look at pixels just outside the source. With that, and GDAL's all-touched rule for the polygon and its coverage, all 22 statistics checks agree with titiler-pgstac to the last digit it computes in double precision. The oracle is at 213 of 219 with the same 6 known differences. `feature` images now search by the feature's bounding box, as titiler-pgstac does. Still to do in E1: CQL2 text filters on collection routes, `asset_as_band` and per-asset options.
- **Found while doing it: overview choice.** For a WGS84 box at reduced size the shared bbox renderer picks its overview from the output's resolution in metres along x, which away from the equator is finer than the level GDAL reads (it compares pixel counts on the warped grid, with a 1.2 margin). Images come out slightly sharper than TiTiler's; statistics from them would differ by tens of percent in `std` and the extremes, which is why the statistics route reads its own pixels. The same reader could serve `/cog` and `/stac` statistics and bbox images.
- **2026-11-24, E1 part seven.** `expression` on point routes (mosaics and items), as band math over `b1..bn` in double precision, and item `renders` and `renders/{id}`, with the renders as WMTS layers. The oracle is at 191 of 197 with the same 6 known differences. Still to do in E1: `POST statistics` on the mosaic families and CQL2 text filters on collection routes.
- **Found while doing it: nodata on bbox images.** The shared bbox renderer (`render_bbox_impl`) treats a pixel as nodata when any band equals the nodata value; TiTiler, and cogzilla's own tile path, need every band to. On a Sentinel-2 true-colour file at full resolution that empties 1.5% of the pixels (dark water and shadow). It shows on `/cog/bbox` and `/stac/bbox` as well. This belongs to C.
- **Found while doing it: TileJSON zooms.** `build_tilejson` used to round the native resolution up and use the file's actual overviews, which gave 9 to 14 for a 10 m Sentinel-2 tile at 44°N against TiTiler's 7 to 13. Fixed on `main` in v0.5.29 (`CrsModel::web_mercator_zoom_range` now follows rio-tiler: GDAL's suggested warp size, then the closest matrix); checked there against the real 10 m, 20 m and 60 m files. The item routes call the same function on the item's projection grid.
- **Found while doing it: preview and statistics sizing.** Still open at v0.5.29. On the same file, `/stac/preview?max_size=256` draws the raster into about 240 of the 256 pixels and `statistics?max_size=128` samples 115 by 115 where TiTiler samples 128 by 128; `/stac/statistics` also describes merged bands as `b1` where TiTiler says `visual_b1`. This belongs to C.
- **Found while doing it: resampling.** TiTiler resamples with nearest neighbour by default and cogzilla with bilinear, which accounts for much of the picture difference the baseline reported. The new routes default to nearest, and the oracle's 16 tile pairs then have a median PSNR of 37.5 dB against titiler-pgstac (30.1 to 43.4). `/cog` and `/stac` still default to bilinear; that belongs to C.1.
- **Found while doing it: `resampling` against `reproject`.** For a source that has to be reprojected, which is most of them, TiTiler takes the method from `reproject` and `resampling=bilinear` changes nothing. On `/cog` and `/stac`, cogzilla applies `resampling` either way and rejects `reproject`; the new routes follow TiTiler. The rest belongs to C.2.

## Goal

1. **Complete parity with TiTiler's HTTP API.** Every route, parameter and built-in option TiTiler ships answers the same way from cogzilla.
2. **pgSTAC is part of cogzilla.** The titiler-pgstac API is served by the default binary, not an add-on.
3. **Faster than both**, including the part of a mosaic tile that is spent in Postgres.

Left out for now:

- **Projection support.** The eleven other tile matrix sets, `dst_crs`, `crs`, `coord_crs` beyond `EPSG:4326`/`EPSG:3857`, and source coordinate systems or datums cogzilla does not already place.
- **Things only a Python process can do.** User-supplied Python algorithms, readers and dependencies. The target is TiTiler's built-in set.
- **Non-TIFF sources** (JPEG 2000, NetCDF, Zarr, VRT). TiTiler reads these through GDAL; cogzilla reads GeoTIFF. This is a scoping decision of this plan, not something that was asked for, and it is the first thing to revisit if "complete" is meant to include inputs.

## What "done" means

Parity is defined by a differential oracle, not by a checklist kept by hand.

- The oracle is generated from TiTiler's own OpenAPI document: every route, every query parameter, every enum value becomes a request sent to both servers.
- A check passes when the status code, content type, JSON shape and numbers (within tolerance), and image (valid-pixel mask and PSNR thresholds per resampling method) agree.
- Parity is reached when the only failures are on an explicit projection allowlist.

Today the oracle has 39 hand-written checks covering `/cog`, `/stac`, `/colorMaps` and `/tileMatrixSets`. The generated set will be several hundred.

## Where cogzilla stands

Measured by sending each of TiTiler 2.3.0's 52 core routes (`/cog`, `/stac`, `/algorithms`, `/colorMaps`, `/tileMatrixSets`) to cogzilla at `ea0d248`. 44 exist and 8 do not; the routes TiTiler adds through optional extensions and its application shell were probed separately.

**Routes present:** every `/cog` and `/stac` data route (info, statistics, tiles, TileJSON, map, point, preview, bbox, feature), `/colorMaps`, `/tileMatrixSets`.

**Routes missing:**

| Family | Missing |
| :--- | :--- |
| `/cog` | `/viewer`, `/WMTSCapabilities.xml`, `/validate`, `/stac`, WMS |
| `/stac` | `/viewer`, `/WMTSCapabilities.xml`, `/renders`, `/renders/{id}` |
| `/mosaicjson` | everything except `info`, `tilejson.json` and `tiles`: point, bbox, feature, statistics, the three `assets` routes, map, WMTS, validate |
| Global | `/algorithms`, `/algorithms/{id}`, `/conformance`, `/api`, `/api.html` |
| pgSTAC | all 72 routes of titiler-pgstac (`/searches`, `/collections/{id}`, `/collections/{id}/items/{id}`) |

The `/mosaicjson` row is inferred from titiler-pgstac's mosaic routes, which use the same factory; TiTiler's MosaicJSON app was not available to probe.

**Options rejected today** (from `docs/api.md`, each currently a `400`):

- Resampling: `average`, `mode`, `gauss`, `rms`, and the warp-only `max`, `min`, `med`, `q1`, `q3`, `sum`; `lanczos` and `cubic_spline` are served as cubic; `reproject` is rejected outright.
- Algorithms: `contours`, `normalizedIndex`, `cast`, `ceil`, `floor`, `min`, `max`, `median`, `mean`, `std`, `var`, `sum`, `grayscale`, `bitonal`.
- Expressions: multiple `;`-separated outputs, comparisons, `where(...)`.
- Statistics: `histogram_bins`, `histogram_range`, `categorical`, `c`, `p`, `expression`, `algorithm`, `bidx`, and FeatureCollection bodies.
- Output: GeoTIFF, JPEG 2000, `pngraw`.
- Parameters: `padding`, `width`/`height` as query parameters, `max_size` on bbox routes, `minzoom`, `maxzoom`, `tile_format`, `cover_scale`, colormap `orientation`, `asset_as_band`, and the WMTS `use_epsg` and `zooms`.
- Mosaics: only `pixel_selection=first`; one asset per scene.

**Behaviour that differs:**

- Invalid input returns `400` with plain text; TiTiler returns `422` with a FastAPI error body.
- A tile outside the dataset returns a transparent `200`; TiTiler returns `404`.
- `/vsicurl/` and other GDAL virtual paths are not recognised.
- Multi-asset rendering needs identical grids and at most three assets.

## Baseline: where a pgSTAC tile's time goes

Measured with `benches/pgstac_baseline.py` against the reference stack in `benches/pgstac/`: 46,476 Sentinel-2 L2A items over the western Alps (2020 to 2024) in 60 monthly partitions, stock titiler-pgstac with one worker, cogzilla v0.5.2, an Apple M5 on a home network. One run, 130 tiles for the SQL columns and 45 for the raster columns, on a laptop that was not idle.

Search: the whole collection, newest first.

| z | items per tile (median) | `geojsonsearch` ms | lean query ms | cogzilla warm raster ms | SQL share of the tile |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 100 | 1,144 | 5.0 | | |
| 6 | 72 | 4,870 | 150 | | |
| 7 | 24 | 1,229 | 5.5 | | |
| 8 | 8 | 97.1 | 1.33 | 11.1 | 91% |
| 9 | 4 | 48.0 | 0.94 | 8.5 | 85% |
| 10 | 2 | 26.3 | 0.51 | 7.6 | 83% |
| 11 | 1 | 15.9 | 0.42 | 4.7 | 78% |
| 12 | 1 | 17.2 | 0.47 | 4.3 | 80% |
| 13 | 1 | 15.9 | 0.45 | 5.7 | 76% |
| 14 | 1 | 15.2 | 0.46 | | |

All values are medians. `geojsonsearch` is called exactly as titiler-pgstac calls it. "Lean query" is the E2 prototype's time in Postgres and on the wire; "SQL share" is `geojsonsearch` / (`geojsonsearch` + cogzilla warm raster). A second search with a cloud-cover filter gave the same picture.

What it shows:

1. **Postgres is most of the tile.** Across the 45 rendered tiles, `geojsonsearch` is 73 to 91% (median 81%) of what a cogzilla pgSTAC tile would cost warm: 24.4 ms in total at the median. With the lean query in its place the same tile is 6.1 ms. For scale, stock titiler-pgstac's repeat request, with its item list cached, has a median of 413 ms.
2. **The lean query is about 40 times faster at z ≥ 8.** Median 0.55 ms against 18.9 ms over 112 tiles, with the identical item list on 103 of them. Where the lists differ, the extra items add at most 2.6e-7 of the tile's area.
3. **Low zooms are where pgSTAC breaks down.** At z 5 to 7 `geojsonsearch` has a median of 1.2 s and a worst case of 10.8 s; three of 18 tiles ran past its own 5 s limit. The lean queries take 5 to 170 ms there, but the prototype's exact geometry unions push its total as high as 5.2 s, which is the cost the bitmask in E2 is meant to remove.
4. **About half the items pgSTAC returns at low zoom add nothing measurable.** Requiring each item to add a millionth of the tile's area cuts the 18 low-zoom tiles from 1,063 items to 553. On the eight tiles where pgSTAC stopped at its 100-item limit, the tolerant loop reached further into the catalog instead.
5. **There is no cold-tile win yet.** First requests are dominated by the network and are noisy here, but on the z8 tiles with five to nine scenes cogzilla's one-at-a-time mosaic took 0.7 to 2.7 s against titiler-pgstac's 0.8 to 2.4 s. That is workstream D.
6. **Both servers fail the same way on a bad scene.** Three tiles include a source file that is not a valid TIFF, and both servers answered `500`.
7. **The pictures are not yet the same.** The 45 tile pairs agree on which pixels are valid (median 100%, minimum 99.88%), but the median PSNR is 29.4 dB (22.3 to 34.9). Not investigated.

To reproduce:

```bash
docker compose -f benches/pgstac/docker-compose.yml up -d
benches/.venv/bin/python benches/pgstac_load.py --max-items 80000 --datetime "2020-01-01T00:00:00Z/2024-12-31T23:59:59Z"
benches/.venv/bin/python benches/pgstac_baseline.py --zooms-sql 5-14 --tiles-per-zoom 8 --raster-tiles-per-zoom 4 --sql-iters 3
```

## Workstreams

### A. The oracle

Generate the request matrix from TiTiler's OpenAPI and from titiler-pgstac's. Run it in CI against pinned TiTiler and titiler-pgstac containers (`benches/pgstac/docker-compose.yml` already pins the pgSTAC pair). Report failures grouped by route and parameter so the remaining work is always a number.

*Exit:* the failing set is published per commit, and nothing merges that grows it.

### B. Render core: planes before pixels

Today decode, resample, band math, colour and encode are fused into one pass over a shared coordinate plan, and mosaics composite finished RGBA. That design is why most of the rejected options are rejected.

Add an intermediate stage: each asset is resampled on its own grid into `f32` planes plus a validity mask at tile resolution. Band math, pixel selection, algorithms, statistics and encoding then run on planes. The fused single-pass path stays as the fast path for requests that do not need planes.

This one change unlocks: every resampling method, every algorithm, multi-output and conditional expressions, statistics over expressions, assets on different grids, more than three assets, `asset_as_band`, all nine `pixel_selection` methods, and GeoTIFF/`pngraw`/NumPy outputs that carry data rather than colour.

*Exit:* a planes-path tile matches the fused path bit for bit on requests both can serve, and costs no more than 2× the fused path warm.

### C. API completeness for `/cog` and `/stac`

1. **No kernel work:** `/algorithms`, `/conformance`, `/api`, viewers, WMTS capabilities, `/stac/renders`, `/cog/validate`, `/cog/stac`, `422` error bodies, `404` outside bounds, `/vsicurl/` paths, `width`/`height`/`max_size`/`minzoom`/`maxzoom`/`tile_format`/`orientation`.
2. **On top of B:** resampling methods, the fourteen algorithms, the expression language, statistics options and FeatureCollection bodies, `padding`, `cover_scale`, GeoTIFF and `pngraw` output.
3. **Needs a decision when reached:** JPEG 2000 output (needs an encoder; none is pure Rust) and WMS.

*Exit:* zero oracle failures on `/cog`, `/stac`, `/algorithms`, `/colorMaps` outside the projection allowlist.

### D. Mosaic engine and the MosaicJSON family

Replace the sequential loop in `engine.rs` (`render_mosaic_tile`) with a planner:

- An async, ordered **item source** yields items with footprint, datetime and assets. MosaicJSON, the in-memory grid catalog and pgSTAC all implement it.
- Scenes are fetched in a **speculative parallel window** and composited in order, so wall time is the slowest scene rather than the sum.
- Compositing happens on planes (B), which gives every pixel-selection method.
- Errors are handled as TiTiler handles them: a scene that does not reach the tile is skipped, and any other failure fails the tile (the baseline hit this on three tiles, and both servers returned `500`). The one deliberate exception is a scene in a coordinate system cogzilla cannot place, which is skipped and counted so that the projection gap does not take whole mosaics down.
- Early exit uses real pixel coverage, with footprints used only to decide what to fetch.

Then add the missing `/mosaicjson` routes on top of it.

*Exit:* zero oracle failures on `/mosaicjson`; a cold N-scene tile takes no longer than the slowest single scene plus 20%.

### E. pgSTAC

**E1. Compatibility path.** New crate `cogzilla-pgstac` (tokio-postgres, rustls, a small pool; `PG*` and `DATABASE_URL` as titiler-pgstac 2+ reads them; `search_path=pgstac,public`). All 72 routes. SQL is exactly titiler-pgstac's: `search_query` to register, `geojsonsearch` per tile with the same arguments and defaults, `pgstac.search` for single items. The item routes reuse the existing `/stac` path with the item JSON coming from the database. Compiled into the default binary and inert until a database is configured.

*Exit:* zero oracle failures on the pgSTAC routes; item lists identical to titiler-pgstac's by construction.

**E2. Lean SQL.** Take `_where` and `orderby` from `search_query` once per search (it returns both; verified on 0.10.2). Per tile, do what pgSTAC does but without the plpgsql: walk the partitions in sort order, one plain query each, returning only `id`, footprint and the requested asset hrefs, and stop when the tile is covered. Coverage is decided in cogzilla by rasterising footprints onto a tile-sized bitmask. No temp table, no per-tile writes, no hydration, no growing `ST_Union`, and no GEOS dependency. Read-only and nothing installed in the database. `geojsonsearch` remains the fallback for pgSTAC versions without an adapter, and the oracle compares item lists against it on every run.

The baseline prototype does the partition walk in Python and already answers in about a millisecond where `geojsonsearch` takes 16 to 100. A single query over the partitioned `items` table is the wrong shape: it measured 7.5 ms server-side for one z12 tile, because every one of the 60 partitions is planned and probed.

The bitmask is the untested part. Exact geometry unions are what make low zooms slow in both pgSTAC and the prototype, and the bitmask is expected to remove that cost, but nothing here measures it yet.

*Exit:* item lists are a subset of `geojsonsearch`'s wherever it did not hit its item limit, and every dropped item adds less than a tenth of a pixel; median query time per tile under 2 ms at z ≥ 8 and under 50 ms at z 5 to 7 on the baseline catalog.

*Where it stands (2026-11-24):*

- Subset: met on all 279 tiles measured. Every list was a part of `geojsonsearch`'s in the same order, except on the 13 tiles where `geojsonsearch` stopped at its item or time limit.
- "Less than a tenth of a pixel": not what a grid of sample points can promise, and not met as written. 901 of 903 dropped items first-covered less than a tenth of a pixel; two covered 0.29 and 0.50 of one, in strips long enough to add up and too thin to hold a sample. What holds for every dropped item is that it covers no output pixel's centre that an earlier item did not, which is what decides the picture; the criterion should say that.
- Time: 1.3 ms at z ≥ 8 and 29 ms at z 5 to 7 for a search with no property filter. A search with one (`eo:cloud_cover` below 10) is 5.2 ms and 67 ms: Postgres reads each candidate row's whole document to test the filter, here as in `geojsonsearch`, and that is the floor without an index on the property or the mirror in E3.
- The time is Postgres's: a low-zoom lookup that scans to the 10,000-row limit took 27 to 49 ms, of which 20 to 33 ms was six or seven queries (planning one over sixteen partitions measured 9 ms) and the rest reading 10,000 rows and their footprints. Things tried and kept: skipping chunks no partition has rows in, reading chunks in runs of one, two, four and so on, reading hrefs only for rows kept after the first query, no parallel workers. Reading partitions directly would remove the planning cost and tie the code to pgSTAC's table layout; not done.
- The bitmask was the untested part and is not the cost: one to two microseconds per row, parsing included, once most of the tile is covered.
- Checked on pgSTAC 0.10.2 only, which is why `auto` turns it on for 0.10.x alone. 0.9 is the version most deployments run and is the next thing to check; the unreleased layout that splits `items` into columns will need its own queries.
- `usecount` in `pgstac.searches` no longer counts tiles on this path (the search is compiled once per cache period); `lastused` still moves.
- `statistics` and the `/assets` routes still use `geojsonsearch`.

**E3. Footprint index.** Mirror `id`, `datetime`, footprint and hrefs per collection in memory. A search becomes a bitmap and an ordering over that mirror, and a tile lookup becomes a grid query. Collections above a size cap stay on E2. The index can be written to a file so a Lambda deployment needs no database connection.

*Exit:* median lookup under 100 µs; a measured, documented staleness bound. How to detect changes cheaply is unresolved: pgSTAC has no LISTEN/NOTIFY, and `partition_stats.last_updated` is the candidate to test first.

### F. Beyond parity

- Persist COG headers on disk so a cold scene costs one round trip instead of two (`store.rs` fetches the header on every cold open today).
- ETags for mosaic tiles derived from the search, the item list and the render parameters.
- Low-zoom tiles: the baseline shows these are where pgSTAC is slowest, and no amount of raster speed fixes a tile that needs a hundred scenes.

## Order

| Milestone | Contains | Depends on |
| :--- | :--- | :--- |
| M1 | A, and C.1 | nothing |
| M2 | B | A |
| M3 | C.2 | B |
| M4 | D | B |
| M5 | E1 | D for mosaic routes; the item routes only need C.1 |
| M6 | E2 | E1 |
| M7 | E3 and F | E2 |

M3 and M4 can run in parallel once B lands. E1's item routes can start alongside M2.

## Risks

- **pgSTAC is changing underneath E2.** 0.10 moved base items into their own table, and unreleased `main` splits `items` into columns and adds a Rust read engine of its own. E2 needs a per-version adapter and must never be the only path.
- **B is a rewrite of the hot path.** The fused pass is where the benchmark numbers come from. Keeping it as a fast path, with a bit-for-bit check against the planes path, is what stops parity work from costing the speed.
- **Semantic drift in E2.** pgSTAC decides coverage with exact floating-point comparisons on GEOS output, so it keeps items whose contribution is rounding noise (the baseline saw gains near 1e-16 of the tile). A bitmask will not reproduce those decisions, by design. E1 stays available for anyone who needs the identical list, and the oracle has to report the difference rather than hide it.
- **Picture differences on real mosaics.** The baseline's median PSNR of 29.4 dB has no explanation yet. Resampling defaults and overview choice are the first suspects, and the oracle in workstream A is where it gets pinned down.
- **Index memory in E3.** Archives of tens of millions of items will not fit; the cap and the E2 fallback are part of the design, not an afterthought.
- **Scope of "complete".** If inputs beyond GeoTIFF or projection support come back into scope, B and D are unaffected but the reader and projection crates take on work this plan does not cost.
