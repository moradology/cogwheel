# Zarr integration plan

`/zarr/*` is the last route family TiTiler serves and cogzilla does not. Comparing the two OpenAPI
documents, cogzilla documents 89 operations against TiTiler's 90, 70 are shared, and across all 70
there is **no** query parameter TiTiler documents and cogzilla does not. Every one of the 20
operations that is only TiTiler's is `/zarr/*`. Closing it completes the drop-in claim.

This branch, `zarr-impl`, is the integration target: work lands here in the phases below, and the
branch merges into `main` once Phase 3 is green against the reference server.

## The shape of the work

cogzilla's render core is already format-agnostic, which is what makes this a few thousand lines
rather than a rewrite. Everything downstream of a decoded block reaches the kernel through:

```rust
BandSource::new(ifd_w, ifd_h, tile_w, tile_h, tx0, ty0, grid_w, grid_h,
                blocks: &[Option<&DecodedBand>])
```

Pure geometry plus decoded blocks, with no TIFF type in the signature. A Zarr array maps onto it
directly: array shape to `ifd_w/h`, chunk shape to `tile_w/h`, the chunk window to
`tx0/ty0/grid_w/grid_h`, and a chunk that was never written to `None`, which the kernel already
treats as all-nodata. Resampling, statistics, colour maps, expressions, terrain and the exact
coverage-weighted zonal work are therefore reused as they stand.

The reader must plug in at the same seam the COG reader uses in `engine.rs`:

```rust
let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;   // -> Vec<Option<DecodedTile>>
let src = BandSource::new(.., &block_refs);
```

**Rule for this branch:** no parallel render pipeline. A Zarr request reaches the same kernel entry
points a COG request does, or the change does not land. One definition per behaviour is what keeps
this small, and it is what stops `/zarr` drifting from `/cog` the way a copied pipeline would.

`cogzilla-io` is reused as it stands too: S3, GCS, Azure and HTTP with the full credential chain,
concurrency limiting, caching and eviction. A Zarr chunk is one more object to ask it for. The one
difference from COG reading is shape of traffic — many small whole-object GETs rather than a few
ranged reads into one object — which is a tuning question for Phase 4, not a new transport.

## Phases

### Phase 0 — fixtures, before any of it

CI generates fixtures with `benches/generate_fixtures.py` and caches them on that script's hash,
and the integration tests skip themselves when `benches/fixtures` is absent. There is no Zarr
equivalent, so until `benches/generate_zarr_fixtures.py` exists every other phase has nowhere to
put an integration test. It is first for that reason, and it includes a GeoTIFF twin written from
the same samples on the same grid, for the assertion in Phase 2 that the whole plan rests on.

### Phase 1 — metadata and chunk addressing (this commit)

`crates/cogzilla-zarr`: v2 `.zarray`/`.zattrs` and v3 `zarr.json` parsed into one `ArrayMeta`;
dtype mapping including the wide types that OO1 settled must stay exact; fill values including
`null`, `NaN` and the infinities; the codec chain captured by name and configuration; chunk key
encodings (v2 separator, v3 `default` and `v2`), including the zero-dimension cases; chunk grid
arithmetic. Unit tested, no I/O.

### Phase 2 — chunk reader and codecs

- Consolidated metadata (`.zmetadata`, v3 consolidated) so opening a dataset is one GET, not N.
- Group and array walking; `/zarr/dataset/*` introspection comes free from this.
- Codec chain execution. Already linked and reusable: `gzip`/`zlib` (libdeflater), `zstd`
  (zstd-sys). New and small: `lz4`, `shuffle`/`bitshuffle`, `transpose`, `bytes` endianness.
  New and the long pole: `blosc`, which is a container wrapping blosclz/lz4/zstd/zlib plus a
  shuffle filter — parsing the container and dispatching to decoders we already have covers
  blosc-zstd and blosc-lz4, which is nearly all real data. Linking c-blosc is acceptable if that
  proves faster: the claim is zero-GDAL, and zstd-sys, libwebp-sys and libdeflater are already C.
- v3 `sharding_indexed`: read the shard index, then the inner chunk. This is the one codec that
  changes addressing rather than bytes.
- A missing chunk is a hole, not an error, when the array has a fill value — it maps to the
  kernel's existing sparse-block path and short-circuits to an empty tile for free.

### Phase 3 — georeferencing, dimensions and routes

- **CRS.** cogzilla has no WKT parser today; `cogzilla-proj` is driven by `CrsModel::from_epsg`.
  Read CF `grid_mapping` attributes, which are structured JSON that maps onto the existing
  `CrsModel` variants, and fall back to the `AUTHORITY` code in a `spatial_ref` WKT string.
  Default to EPSG:4326 when the store says nothing, as TiTiler does.
- **Affine** from 1-D coordinate arrays, with an explicit regularity check. Irregular and
  curvilinear grids are refused by name, the way RPC-only files are, not approximated.
- 0..360 longitude normalisation, reusing what R2 settled for COGs.
- `variable`, `variables`, `group`; the `sel=dim=value` DSL including `method::value`; CF time
  decoding for `decode_times` and `show_times`; 2D and 3D only.
- Routes: the 21 TiTiler paths, of which five (`/zarr/dataset`, `/dataset/keys`, `/dataset/dict`,
  `/dataset/dimensions`, `/dataset/coordinates/{name}`) are metadata JSON and nearly free, and the
  rest are twins of existing `/cog` routes. `http.rs` already parameterises URL building by
  `route_prefix`. Of the 36 query parameters, 29 already exist; only the seven above are new.

### Phase 4 — making it fast, which is the point

A plain Zarr array has no overview pyramid, so a low-zoom tile means fetching and decimating many
chunks. TiTiler has the same problem, so parity does not depend on solving it, but this is where
the Rust advantage is largest: the incumbent is xarray plus zarr-python plus obstore with a GDAL
warp for reprojection, and the work is dominated by many small concurrent object reads, which is
exactly what `cogzilla-io` is built for.

- Honour GeoZarr `multiscales` where a dataset ships them.
- Parallel chunk fetch with the existing limiter; decimated decode where only every nth sample is
  needed; the existing chunk cache and eviction.
- A `cogmark` scenario against `titiler.xarray` on the same store, in the same shape as the COG
  comparison, so the claim is measured rather than asserted.

## Cross-cutting, and not optional

These three are not a phase. They cut across the reader and the routes, and the branch does not
merge to `main` without them.

### Bound the fan-out before anything is fetched

Serving COGs, one tile is a handful of ranged reads into one object. Serving Zarr, **one tile can
be thousands of GETs against someone else's bucket**, and at low zoom over a finely chunked array
the fan-out is bounded only by the chunk count, which the client picks by choosing a URL. No
existing cap covers it: `COGZILLA_MAX_UPSTREAM_FETCHES` limits concurrency rather than the total,
and the `{w}x{h}` cap bounds the output image rather than the input. So the covering chunk count
and the decoded byte count are checked from metadata, before the first fetch, and a request over
the cap is refused with the computed number in the message.

The decode side needs the same care: `MAX_DECODED_TILE_BYTES` guards TIFF tiles today, and a blosc
header or a zstd frame can claim any size it likes. A chunk that decodes to more than its own
declared shape is a corrupt or hostile store whatever its header says.

### Freshness, because Zarr stores change

Every caching assumption in this codebase comes from COGs, which are effectively immutable. Zarr
stores are routinely appended to — a new time step an hour is ordinary — and rewritten in place.
Cache the consolidated metadata under `COGZILLA_CACHE_TTL_SECS` and prefer cheap revalidation to
blind expiry; fold the metadata validator into the ETag, or a client revalidating after an append
gets a 304 for data that has moved on. One store is one open URI, however many chunks it holds.

### One table of failure to HTTP status

cogzilla's statuses are better than TiTiler's because they were decided rather than inherited.
Zarr brings a dozen new failure modes at once — unknown codec, irregular grid, missing variable,
`sel` out of range, unmodellable CRS, missing chunk with no fill value, corrupt shard index — and
they are settled in one table, used by every route. Client fault is 4xx, origin fault is 5xx, and
the message names the thing that failed.

## Deliberately out of scope

NetCDF and HDF5 (a hard binary format, and the real weight behind `titiler.xarray`), kerchunk and
virtual Zarr, icechunk, writes, complex and variable-length dtypes, curvilinear grids.

Zarr assets reached through `/stac` and `/mosaicjson` are a separate decision, not an oversight:
the asset layer resolves to COGs today, a Zarr asset needs a `variable` that STAC has no settled
way to express, and TiTiler keeps the two apart as well. Deferred, and recorded as deferred.

## Risks

- **Spec churn.** The GeoZarr conventions (multiscales, geo-proj, spatial) were released at the
  Zarr Summit in October 2025 at Proposal maturity, with OGC Architecture Board review targeted for
  summer 2026 and not yet confirmed. Build to CF first and the conventions second, and keep the
  convention reading in one module so it can move.
- **Blosc.** The decision between a Rust container implementation and linking c-blosc should be
  made on a measurement in Phase 2, not up front.
- **No pyramid.** Without multiscales, a global array at z=0 is an expensive tile however fast the
  decoder is. Phase 4 should state plainly what the cost is rather than hide it.

## The one test that decides whether this plan is right

A tile rendered from a Zarr array must come out **byte-identical** to the same tile rendered from a
GeoTIFF written from the same samples on the same grid. That is the whole thesis of the branch — if
the kernel really is format-agnostic, the two paths cannot disagree — and it is reachable early,
right after the chunk reader. If it cannot be made to pass, the premise is wrong and the plan needs
rethinking before the route work, not after it.

## Done means

- The 21 routes answer, and the parity probe in `benches/results/parity_sweeps` shows no
  unexplained difference against `titiler.application` on the same store.
- `/zarr` leaves the parked list in `docs/titiler-divergences.md`.
- No render-path regression on the existing COG routes: the statistics and tile benchmarks are at
  or below the previous build, as every other change on this project is held to.
