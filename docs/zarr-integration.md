# Zarr integration (`crates/cogzilla-zarr` + `/zarr/*`)

Zarr v2 and v3 support is implemented in `crates/cogzilla-zarr` and exposed under `/zarr/*` in `cogzilla-server`.

## Architecture

There is no parallel render pipeline. Decoded Zarr chunks enter `cogzilla-kernel` through the same `BandSource::new(...)` seam that COGs use, and polygon statistics reuse `zonal.rs` (`PolygonMask`) verbatim.

- **Metadata & hierarchy (`hierarchy.rs`, `meta.rs`, `chunk_key.rs`):** Unified `ArrayMeta` across Zarr v2 (`.zarray` / `.zattrs` / `.zmetadata`) and v3 (`zarr.json`, including inline `consolidated_metadata`). Single-GET consolidated open with fallback to unconsolidated group walking.
- **Codecs & sharding (`codecs.rs`, `blosc.rs`, `sharding.rs`):** Reverse-order codec pipelines supporting `bytes`, `transpose`, Fortran order (`order_f`), `gzip`/`zlib` (`libdeflater`), `zstd` (`zstd-sys`), pure-Rust `lz4`, `shuffle`/`bitshuffle`, pure-Rust Blosc 1.x (`blosclz`, `lz4`, `lz4hc`, `zstd`, `zlib`), and v3 `sharding_indexed` (ranged index + inner-chunk reads with CRC32C verification and `ShardIndexCache`).
- **Georeferencing & selection (`crs.rs`, `geo.rs`, `select.rs`, `sel.rs`, `multiscales.rs`):** CF `grid_mapping` / GeoZarr `proj:code` / WKT `AUTHORITY` CRS resolution (default EPSG:4326 when absent), 1D regular-axis affine synthesis with explicit irregularity refusal, `0..360` longitude normalization, `variable`/`variables`/`group`, `sel=dim=value` (with `nearest::` / `exact::` and CF calendar time decoding), and OME-Zarr / GeoZarr `multiscales` overview selection.
- **Safety caps & caching (`caps.rs`, `http/zarr.rs`):** Pre-fetch chunk-count and decoded-byte budgets (`COGZILLA_ZARR_MAX_CHUNKS_PER_REQ`, `COGZILLA_ZARR_MAX_DECODED_BYTES_PER_REQ`, `COGZILLA_ZARR_MAX_SHARD_INDEX_BYTES`), bounded `ZarrHierarchyCache` with local `stat` / remote `HEAD` TTL revalidation, validator-folded ETags, shared `TileCache` warm-tile caching, and Prometheus `cogzilla_zarr_*` metrics.

## Endpoints

All 21 TiTiler `/zarr/*` endpoints are implemented and documented in OpenAPI (`/openapi.json`, `/docs`) and [`docs/api.md`](api.md):

- Metadata & introspection: `/zarr/info`, `/zarr/info.geojson`, `/zarr/bounds`, `/zarr/validate`, `/zarr/dataset/`, `/zarr/dataset/keys`, `/zarr/dataset/dict`, `/zarr/dataset/dimensions`, `/zarr/dataset/coordinates/{name}`
- Tiles & maps: `/zarr/tiles`, `/zarr/tiles/{tms}`, `/zarr/tiles/{tms}/{z}/{x}/{y}[@scale][.{ext}]`, `/zarr/{tms}/tilejson.json`, `/zarr/{tms}/map.html`, `/zarr/{tms}/WMTSCapabilities.xml`
- Extraction & stats: `/zarr/point/{lon},{lat}`, `/zarr/preview[.{ext}]`, `/zarr/bbox/{minx},{miny},{maxx},{maxy}[/{w}x{h}][.{ext}]`, `GET /zarr/statistics`, `POST /zarr/statistics`, `POST /zarr/feature[/{w}x{h}][.{ext}]`

## Verification

- Deterministic fixtures: `benches/generate_zarr_fixtures.py` (`benches/fixtures/zarr`)
- Unit & integration tests: `crates/cogzilla-zarr` + `crates/cogzilla-server/tests/zarr_api.rs` (including byte-identical tile parity and exact polygon statistics parity against the GeoTIFF twin)
- Live parity oracle: `benches/zarr_parity_oracle.py`
- Benchmark harness: `benches/zarr_bench.py`
