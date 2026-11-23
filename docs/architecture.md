# How `cogzilla` works

## Why no GDAL?

GDAL reads hundreds of formats in thousands of projections, and it is very good at that. A tile server for COGs needs a small corner of it: one container format, a handful of codecs, and the projections satellite imagery actually ships in. `cogzilla` implements only that corner, which lets each stage be specialized: a TIFF parser that opens a file in about a microsecond, closed-form projection math, and decode, resample, band math, and encode fused into a single SIMD pass.

The cost is scope. If your data is not a GeoTIFF, or is in a projection outside the supported list, use TiTiler.

## The four stages

Rust, explicit SIMD (ARM64 NEON, and x86_64 AVX2/FMA chosen at runtime), and native codecs (`libdeflate`, `zstd`, `libwebp`, `zune-jpeg`). There are four stages.

**Reading.** A Classic/BigTIFF and GeoKey parser opens a local file through `mmap` in one to two microseconds (0.9–2.0 µs across runs, hundreds of times faster than `GDALOpenEx`). A remote file opens with one speculative 128 KiB range request. Later reads coalesce adjacent byte ranges, S3 requests are SigV4-signed and gated by a throttle-only AIMD concurrency limiter, and decoded blocks go into a sharded LRU cache (with an optional persistent CRC32-checked tier-2 disk cache for compressed blocks) that collapses concurrent requests for the same block into one fetch.

**Decoding.** DEFLATE, ZSTD, LZW, JPEG and WebP compression, with TIFF predictors 1, 2 and 3 written in NEON and AVX2 and bit-exact against GDAL (`float32` predictor 3 runs at 3.00 GB/s on ZSTD on the M5). Multi-block cold misses decode in parallel across worker threads, with a bounded free-list arena pool recycling standard `f32` block allocations. All eight GDAL numeric types from `uint8` to `float64` are supported, with `uint8` and `uint16` stored in native compact slabs (1–2 bytes/pixel for 2–4× higher cache capacity). Blocks that are entirely nodata are flagged at decode, so empty tiles return a pre-encoded transparent image without rendering (this path has no benchmark of its own). WebMercator-aligned JPEG/WebP COGs with identity RGB styling bypass decode, resample, and re-encode completely via zero-decode pass-through (including TIFF tag 347 `JPEGTables` splicing).

**Projecting.** Closed-form `f64` math for `EPSG:3857`, `EPSG:4326`, WGS84 and NAD83 UTM zones, Albers, Lambert Conformal Conic, and Polar Stereographic, each checked against PROJ to a millimetre. A file in any other coordinate system, or on a datum that is not within about two metres of WGS84, is refused rather than drawn out of place; the list is in [api.md](api.md#coordinate-systems). Tiles at `z <= 8` use adaptive error-bounded quadtree subdivision (`< 0.05 px` tolerance); tiles at `z >= 9` interpolate a 17×17 grid. Zooms 0 to 22 are served.

**Rendering.** One pass per scanline does the resampling (nearest, bilinear or cubic), multi-COG compositing, early-exit dynamic and spatially indexed mosaics (`/mosaic/tiles` and `/mosaic/{name}/tiles`, backed by the pluggable `SceneCatalog` trait), scanline-vectorized band math, rescaling and colormap lookup, then encodes PNG (including 1-byte-per-pixel indexed-palette `PLTE`+`tRNS` PNGs with SIMD `bpp=1` Sub filtering for single-band colormapped tiles), JPEG, WebP or raw `float32`. Band math is either an arbitrary `expression=` compiled to a scanline-vectorized bytecode VM or a built-in preset (`ndvi`, `ndwi`, `nbr`, `savi`, `evi`, `db`, `sar_rgb`). All 211 `rio-tiler` colormaps are included.

## Workspace crates

- `crates/cogzilla-tiff`: Classic/BigTIFF and GeoKey parser with a compact IFD index, for all eight numeric types, tiled or stripped.
- `crates/cogzilla-codec`: DEFLATE, ZSTD, LZW, JPEG and WebP block decoding, NEON/AVX2 predictors 1/2/3, and per-block nodata statistics.
- `crates/cogzilla-proj`: closed-form projections (`EPSG:3857`, `4326`, UTM, Albers, LCC, Polar Stereographic) and the per-tile reprojection planners.
- `crates/cogzilla-kernel`: fused SIMD resampling (nearest, bilinear, cubic), rescaling, band math, colormaps, and the PNG, JPEG, WebP and raw `float32` encoders.
- `crates/cogzilla-io`: `mmap` and ranged HTTP/S3 backends (SigV4, retries, timeouts) and the singleflight block cache.
- `crates/cogzilla-server`: the `hyper` HTTP/1.1 and HTTP/2 server and CLI.
- `benches/cogmark`: the `BM-01` to `BM-11` suite and GDAL parity oracle.
