<p align="center">
  <img src="assets/cogzilla.gif" alt="cogzilla" width="100%" />
</p>

# `cogzilla`

An ultra-fast, zero-GDAL Cloud Optimized GeoTIFF (COG) tile server built in Rust + explicit SIMD (ARM64 NEON & x86_64 AVX2/FMA) + native C codecs (`libdeflate`, `zstd`, `libwebp`), purpose-built for satellite imagery (**Landsat Collection 2 `uint16`**, **OPERA RTC-S1 / Sentinel-1 `float32` SAR**, **Sentinel-2**, DEMs).

## Highlights

- **Zero GDAL / Zero `libproj`**: Custom Classic/BigTIFF + GeoKey parser (`0.92–1.3 µs` local `mmap` open — **~300–440×** faster than `GDALOpenEx`) and closed-form `f64` projections (`EPSG:3857`, `EPSG:4326`, all 120 WGS84/NAD83 UTM zones, CONUS/CA Albers, Lambert Conformal Conic, Polar Stereographic `3413`/`3031`) with automatic `Exact2D` low-zoom (`z <= 8`) and `17×17` `Grid2D` (`z >= 9`) multi-zoom tiling (`z=0..22`).
- **SIMD Codecs & Predictors**: Thread-local `libdeflate`, `ZSTD_DCtx`, custom 64-bit MSB-first TIFF LZW, and ARM64 NEON / x86_64 AVX2 Predictor 1/2/3 unshufflers (**0-ULP bit-exact** against GDAL):
  - `float32` Predictor 3 (single-band & hardware `vld3q_u8` 3-band chunky RGB): **`3.00 GB/s`** on `ZSTD + P3`.
  - `uint16` Predictor 2 (Landsat C2 / Sentinel-2): NEON 8-lane Kogge-Stone prefix-sum widened straight into 64-byte-aligned `f32` slabs at block decode.
  - Register-fused `BlockStats` (`all_nodata`, `no_nodata`) enabling `< 200 ns` static empty-tile responses on tilted satellite swath corners and automatic opaque RGB encoding (`CleanFastPath`).
- **Monomorphized Fused Kernels & Band Math**: Single scanline pass fusing bilinear/nearest resampling, multi-COG compositing (e.g. Landsat 3-file `SR_B4 + SR_B3 + SR_B2` or SAR 2-file `VV + VH`), in-register spectral indices (`ndvi`, `ndwi`, `nbr`, `savi`, `evi`, fast IEEE-754 SAR `db` `10*log10`, dual-pol `sar_rgb`), SIMD PNG Sub filtering + whole-buffer `libdeflate`, `libwebp`, and raw `f32` binary output.
- **Local `mmap` + Coalesced Remote S3/HTTP + Singleflight Cache**: Speculative 128 KiB single-RTT remote header open, byte-range coalescing, AWS SigV4 signing, and sharded lock-free singleflight block caching.

## `COGMark` Benchmarks (vs. GDAL / `rio-tiler` / `TiTiler`)

Measured on Apple M5 (10 cores, ARM64 NEON, `rustc 1.95.0 --release`):

| ID | Workload | `cogzilla` | GDAL / TiTiler | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| **`BM-01`** | `ColdHeader` (5-IFD COG open) | **1.3 µs** | 406 µs | **312×** |
| **`BM-02`** | `UnshufflePred3` (`ZSTD + P3`, 1 MiB) | **350 µs** (3.00 GB/s) | 835 µs (1.26 GB/s) | **2.4×** |
| **`BM-03`** | `NativeMercator` PNG (warm) | **634 µs** | 17,587 µs | **27.7×** |
| **`BM-03`** | `NativeMercator` Raw `f32` (warm) | **268 µs** | 12,377 µs | **46.2×** |
| **`BM-04`** | `Reproj4326` PNG (warm) | **607 µs** | 25,766 µs | **42.4×** |
| **`BM-05a`** | `Satellite UTM10N 3-Band RGB` PNG (warm / cold) | **1.22 ms** / 6.65 ms | 47.7 ms / 47.8 ms | **39.2×** / **7.2×** |
| **`BM-05d`** | `Satellite Fused NDVI` PNG (warm) | **917 µs** | 14,208 µs | **15.5×** |
| **`BM-06`** | `SparseCoastline` (50% empty, avg/tile) | **420 µs** | 9,180 µs | **21.9×** |
| **`BM-07`** | `PanBurst32` HTTP Mercator PNG (32 concurrent) | **6,863 tiles/s** | 181 tiles/s | **37.8×** |
| **`BM-07`** | `PanBurst32` HTTP Satellite RGB PNG (32 concurrent) | **3,791 tiles/s** | 171 tiles/s | **22.2×** |
| **`BM-08`** | `RemoteRangeRTT` (20 ms RTT cold open + tile) | **48.3 ms** (3 reqs) | 147.0 ms (4 reqs) | **3.0×** |
| **`BM-09`** | `Landsat C2 3-File uint16 RGB` PNG (warm) | **1.63 ms** | 25.4 ms | **15.6×** |
| **`BM-10`** | `OPERA RTC-S1 Dual-Pol SAR-RGB` PNG (warm) | **1.35 ms** | 20.7 ms | **15.3×** |
| **`BM-11`** | `MultiZoom` Landsat C2 (`z=5 / 8 / 10 / 12 / 14 / 16`) | **7.3 / 7.0 / 1.8 / 1.6 / 1.2 / 1.2 ms** | — | `99.86–100%` mask parity |

## Quickstart

```bash
# Build release server
cargo build --release -p cogzilla-server

# Serve pre-registered single COGs and multi-COG satellite scenes
./target/release/cogzilla-server \
  --bind 127.0.0.1:8080 \
  --cache-mb 1024 \
  --dataset dem=/path/to/dem.tif \
  --scene landsat=/path/to/SR_B4.TIF,/path/to/SR_B3.TIF,/path/to/SR_B2.TIF \
  --scene sar=/path/to/OPERA_VV.tif,/path/to/OPERA_VH.tif
```

### Drop-In TiTiler API Surface (`/cog/*`, `/stac/*`, `/colorMaps`, `/tileMatrixSets`)

`cogzilla-server` implements TiTiler 2.4's non-Postgres routers so MapLibre, Leaflet, OpenLayers, and QGIS clients can switch by changing a single base URL (verified across **39/39 live HTTP differential oracle checks** against TiTiler in `benches/titiler_diff_oracle.py`):

- **`/cog/*` (`TilerFactory`)**:
  - **Tiles (`@1x` = 256px, `@2x` = 512px)**: `GET /cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@{scale}x][.{png|webp|jpg|f32}]` (`WebMercatorQuad` & `WorldCRS84Quad`), supporting `?unscale=true`, `?nodata=`, `?return_mask=false`, `?color_formula=`, `?colormap={json}`, `?buffer=`, and all 8 GDAL numeric dtypes (`uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `float32`, `float64`).
  - **Arbitrary Band Math**: `?expression=(b4-b3)/(b4+b3)` (compiled to a zero-alloc register bytecode VM) plus fast presets (`ndvi`, `ndwi`, `nbr`, `savi`, `evi`, `db`, `sar_rgb`).
  - **Metadata, Point, Stats & Crops**: `GET /cog/info`, `GET /cog/info.geojson`, `GET /cog/bounds`, `GET /cog/{tms}/tilejson.json`, `GET /cog/tiles`, `GET /cog/tiles/{tms}`, `GET /cog/{tms}/map.html`, `GET /cog/point/{lon},{lat}` (`?coord_crs=`), `GET|POST /cog/statistics` (GeoJSON polygon clipping), `GET /cog/preview[/{w}x{h}][.{ext}]`, `GET /cog/bbox/{minx},{miny},{maxx},{maxy}[/{w}x{h}][.{ext}]`, `POST /cog/feature[/{w}x{h}][.{ext}]`.
- **`/stac/*` (`MultiBaseTilerFactory(reader=STACReader)`)**:
  - `GET /stac/assets`, `GET /stac/info`, `GET /stac/info.geojson`, `GET /stac/asset_statistics`, `GET|POST /stac/statistics`, `GET /stac/tiles`, `GET /stac/tiles/{tms}`, `GET /stac/tiles/{tms}/{z}/{x}/{y}[@{scale}x][.{ext}]`, `GET /stac/{tms}/tilejson.json`, `GET /stac/{tms}/map.html`, `GET /stac/point/{lon},{lat}`, `GET /stac/preview[/{w}x{h}][.{ext}]`, `GET /stac/bbox/...`, `POST /stac/feature[/{w}x{h}][.{ext}]`.
  - Resolves and caches STAC Item JSON (local or HTTP/S), opens multi-asset COGs concurrently (`?assets=red&assets=green&assets=blue`), and supports both `?expression=(b1-b2)/(b1+b2)` and asset-named `?expression=(red-green)/(red+green)`.
  - **21×–82× faster than TiTiler** on STAC tiles (`2.1 ms` vs `48.6 ms`), TileJSON (`63.5×`), info (`45.3×`), and 3-asset point queries (`82.4×`, `0.0000000` error).
- **`/colorMaps` (`ColorMapFactory`)**: `GET /colorMaps`, `GET /colorMaps/{colorMapId}` (256/256 exact `rio-tiler` LUT JSON and `?f=png` / `?format=png` 256×20 colorbar PNG, `PSNR=99.0 dB`).
- **`/tileMatrixSets` (`TMSFactory`)**: `GET /tileMatrixSets`, `GET /tileMatrixSets/{tileMatrixSetId}` (`WebMercatorQuad` 25-level & `WorldCRS84Quad` OGC TileMatrixSet 2.0 JSON).

### Example HTTP Tile Requests

```text
# Pre-registered Landsat C2 3-file uint16 RGB PNG
GET /tiles/landsat/12/652/1583.png?rescale=7500,22500

# Pre-registered OPERA RTC-S1 dual-pol SAR false-color RGB ([VV_dB, VH_dB, VV_dB - VH_dB])
GET /tiles/sar/12/652/1583.png?expr=sar_rgb

# Dynamic multi-COG S3/HTTP URL compositing (Landsat NDVI as WebP)
GET /cog/tiles/12/652/1583.webp?url=s3://bucket/B5.tif&url=s3://bucket/B4.tif&expr=ndvi&rescale=-0.2,0.8&colormap=ndvi&quality=82

# Single-pol SAR linear power -> dB as raw float32 for WebGL / Deck.gl
GET /cog/tiles/12/652/1583.f32?url=s3://bucket/VV.tif&expr=db
```

## Reproducing `COGMark`

```bash
# 1. Generate deterministic COG fixtures (requires uv)
cd benches && uv venv .venv && uv pip install --python .venv/bin/python rasterio rio-tiler titiler.core uvicorn numpy pillow httpx
.venv/bin/python generate_fixtures.py
.venv/bin/python generate_landsat_sar_fixtures.py
cd ..

# 2. Run cogzilla benchmarks + parity dumps
cargo run --release -p cogmark -- --dump-parity benches/results/parity

# 3. Run GDAL / TiTiler baseline & pixel parity gate
benches/.venv/bin/python benches/gdal_oracle.py
```

## Workspace Crates

- `crates/cogzilla-tiff` — Classic/BigTIFF + GeoKey parser & compact IFD index (`float32` + `uint16`).
- `crates/cogzilla-codec` — Thread-local `libdeflate`, `zstd`, 64-bit LZW + NEON/AVX2 Predictor 1/2/3 & `BlockStats`.
- `crates/cogzilla-proj` — Closed-form `EPSG:3857`, `4326`, UTM, Albers, LCC, Polar Stereographic + `Exact2D`/`Grid2D` planners.
- `crates/cogzilla-kernel` — Monomorphized fused SIMD resample, rescale, spectral band math, fast PNG/WebP/Raw `f32` encoders.
- `crates/cogzilla-io` — `MmapBackend`, coalesced HTTP/S3 `RangeBackend` (SigV4), sharded singleflight block cache.
- `crates/cogzilla-server` — Zero-alloc `hyper` HTTP/1.1 + HTTP/2 server & CLI.
- `benches/cogmark` — `BM-01`..`BM-11` benchmark suite & GDAL parity oracle.
