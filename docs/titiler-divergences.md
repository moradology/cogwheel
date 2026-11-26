# Intentional Divergences from TiTiler 2.4.1 / GDAL

`cogzilla` targets drop-in compatibility with TiTiler 2.4.1 wherever parity benefits clients. However, whenever TiTiler, `rio-tiler`, GDAL, or FastAPI makes a correctness, performance, or ergonomics compromise, `cogzilla` intentionally chooses the better behavior rather than replicating the compromise.

This document is the living record of every intentional divergence, **why** it exists, **what it means for clients**, and how to opt out when applicable.

---

## 1. Accuracy, Correctness & Performance Improvements Over TiTiler / GDAL

### 1.1 Exact Analytical Coverage-Weighted & Ellipsoidal Area-Weighted Zonal Statistics (`POST /*/statistics`, PR #165)

- **TiTiler / `rio-tiler`:** Uses a binary pixel mask by default (overcounting thin slivers up to 5× and small/diagonal shapes), or an approximate `cover_scale=10` 10× supersampled rasterization (100× larger temporary boolean grid in RAM, still approximate). Pixel means are unweighted in the raster's CRS plane, which over-weights northern pixels by up to **53%** on a `40°N–60°N` polygon in `EPSG:4326`.
- **`cogzilla`:** Computes **exact analytical pixel coverage fractions** in `[0, 1]` directly on the COG's native pixel grid (zero resampling, Green's-theorem edge traversal in `O(V + perimeter_cells)` + linear scanline fill), multiplied by each pixel's **WGS84 ellipsoidal ground area** (`w = cov × area_px`):
  - `count`: exact sum of pixel coverage fractions for polygons (pixel units, snapped to integer within `1e-6`), exact clipped length in pixels for `LineString`/`MultiLineString` (e.g. `141.42` for a 100-px diagonal), or point count (`1.0` per point) for `Point`/`MultiPoint`.
  - `sum`: coverage-weighted sum of valid pixel values.
  - `area`: exact ellipsoidal ground area in `m²` covered by valid data for polygons; `null` for `Point`/`MultiPoint`/`LineString`/`MultiLineString` (points and lines have no area).
  - `length`: exact WGS84 geodesic length in metres for `LineString`/`MultiLineString` features (omitted for polygons/points).
  - `mean`, `std`, `median`, `percentiles`, `histogram`: weighted by `w = cov × area_px` for polygons (or geodesic segment length for lines) when `?area_weighted=true` (default), or `w = cov` when `?area_weighted=false`. Zero-coverage boundary neighbors (`cov ≤ 1e-12`) are excluded from `valid_pixels`, `masked_pixels`, `min`, `max`, `unique`, `majority`, and `minority`.
  - `?cover_scale=` and `?all_touched=` are accepted and ignored (since exact coverage is already strictly more accurate and faster).
- **Client significance:** Strictly more accurate (max error `≤ 8.55e-9` px on analytic shapes vs up to 5× error on binary masks), zero supersampling RAM blowup, and **2.3×–7.9× faster** than in-process `rio-tiler`/GDAL even over HTTP:

| Workload (native grid, `bidx=1`, `max_size=0`) | `cogzilla` HTTP (exact + ellipsoidal) | `rio-tiler`/GDAL mask (in-process) | `rio-tiler`/GDAL `10×` cov (in-process) |
|---|---:|---:|---:|
| Sliver `0.2×100` (true `20.00 px`) | **0.19 ms** (`20.00`) | 0.28 ms (`100.00`) | 0.35 ms (`100.00`) |
| `100×100` triangle (true `5,000.00 px`) | **0.29 ms** (`5,000.00`) | 0.54 ms (`5,149.00`) | 1.62 ms (`5,005.00`) |
| `1M px` square (true `1,000,000.00 px`) | **24.62 ms** (`1,000,000.00`) | 85.99 ms (`1,000,000.00`) | 195.08 ms (`1,000,000.00`) |
| `10k`-vertex circle (true `282,743.34 px`) | **12.18 ms** (`282,743.32`) | 28.15 ms (`283,876.00`) | 95.19 ms (`282,743.40`) |

- **Opt-out:** Pass `?area_weighted=false` to disable ellipsoidal ground-area weighting and compare directly against CRS-plane coverage-weighted statistics (for small UTM/equal-area polygons, `area_weighted=true` and `false` are automatically identical via a `< 1e-6` Jacobian short-circuit).

### 1.2 Exact Affine Handling for Rotated GeoTIFFs (`F4`)

- **TiTiler / GDAL:** On rotated geotransforms, tile reads disagree with the file's own affine transform (measured `z11` mean error `2,618` on a `50,000` value range; `z9` mask `86%` agreement).
- **`cogzilla`:** Evaluates the full 6-parameter affine transform directly (`z11` **100%** of pixels identical to ground truth; `z9` mask **100%** agreement).
- **Client significance:** Rotated satellite scenes and drone orthomosaics render in the exact right geographic location without warp artifacts.

### 1.3 Ground-Resolution Overview Selection for Polar & Antimeridian Tiles (`H1b`, `H2b`, PR #161)

- **TiTiler / GDAL:** Oversamples polar (`EPSG:32661` UPS) and antimeridian rasters by reading full resolution even when a coarse overview already exceeds the tile's ground resolution (e.g. `ups.tif` at `977 m/px` for WebMercator `z3/6/0` needing `1,688–3,676 m/px`).
- **`cogzilla`:** Selects the overview matching the tile's actual required ground resolution (`ups.tif` `z3/6/0` reads the `2×` overview in **5 ms** vs TiTiler's **11 ms** full-res read).
- **Client significance:** Less I/O, lower latency, less aliasing, and no loss of accuracy.

### 1.4 Polar & LAEA Rasters in Foreign TileMatrixSets (Sweep #12)

- **TiTiler / GDAL:** Reprojecting polar and Lambert Azimuthal Equal-Area (`EPSG:3035`, `3031`, `3413`, UPS) files into foreign grids (`WorldCRS84Quad`, `WebMercatorQuad`, `CanadianNAD83_LCC`, `NZTM2000Quad`, UPS) matches the source file on `93–98%` of pixels.
- **`cogzilla`:** Matches the source file on **~100%** of pixels.
- **Client significance:** Higher geometric fidelity across non-Mercator tile grids.

### 1.5 Area-Exact `resampling=average` & `mode` (`L1`, PR #167)

- **TiTiler / `cogzilla` (prior):** `cogzilla` now integrates over the exact destination pixel footprint `[x0, x1] × [y0, y1]` weighted by exact overlap area:
  - `1:1` overview reads sample the exact single pixel (`0.0` RMS, zero blur).
  - Exact `2×` / `4×` / `8×` reductions equal the exact full-resolution `k×k` block mean / mode to `< 1e-6` even on COGs without overviews.
  - Non-integer factors weight every overlapped source pixel by its exact overlap area.
- **Client significance:** Physically area-true downsampled previews and tiles.

### 1.6 Physical Ground-Metre Terrain Hillshade & Slope (`N2`)

- **TiTiler:** Computes hillshade/slope gradients in pixel units rather than physical ground metres, which distorts shading away from the equator.
- **`cogzilla`:** Computes Horn 3×3 gradients in true ground metres (accounting for latitude `cos(lat)` scaling on geographic/Mercator grids and CRS linear units).
- **Client significance:** Physically accurate hillshade and slope across all latitudes.

### 1.7 Legacy GeoJSON `crs` Member & Safe `400` on Projected Coordinates Without `coord_crs` (`L2`, PR #167)

- **TiTiler:** Sending projected GeoJSON coordinates (e.g. UTM metres) to `POST /cog/statistics` or `POST /cog/feature` without `?coord_crs=` triggers a `500 Internal Server Error` attempting to allocate hundreds of terabytes.
- **`cogzilla`:**
  1. Honours legacy GeoJSON 2008 `"crs"` members (`type: "name"` or `type: "EPSG"`) on Geometry, Feature, and FeatureCollection when `?coord_crs=` is omitted.
  2. If neither `?coord_crs=` nor `"crs"` is provided and coordinates fall outside WGS84 lon/lat bounds (`[-360, 360] × [-90, 90]`), returns **`400 Bad Request`** with an explicit hint to pass `?coord_crs=EPSG:<code>`.
- **Client significance:** Never OOMs, supports legacy CRS-tagged GeoJSON out of the box, and gives immediate actionable feedback when `?coord_crs=` was forgotten.

### 1.8 Proxy Headers & Extra Endpoint Flexibility (Sweeps #10–#11)

- **`X-Forwarded-Proto`, `X-Forwarded-Host`, `X-Forwarded-Prefix`:** Honoured when constructing TileJSON and WMTS URLs (TiTiler ignores them).
- **`POST /*/feature` body types:** Accepts `FeatureCollection` and bare `Geometry` in addition to `Feature` (TiTiler returns `422`).
- **`reproject=gauss`:** Supported as a resampling extension (TiTiler returns `422`).

---

## 2. Deliberate API & Error-Handling Divergences

| Area | TiTiler 2.4.1 | `cogzilla` | Why |
|---|---|---|---|
| **Uniform semantic HTTP statuses** (all routes, all 13 TMSs) | Out-of-matrix `x`/`y` returns `404`; out-of-footprint non-Mercator tiles can return `200` empty | **`400`** when tile address is invalid for the grid (`x`/`y`/`z` out of matrix, unknown TMS, unsupported CRS); **`404`** with JSON detail naming tile & dataset bounds when valid address lies outside dataset footprint (exact densified-outline test, independent of TMS, including `/point`, `/bbox`, `/feature`); **`204`** empty body when inside advertised mosaic/STAC bounds with no covering asset; **`200`** only when inside footprint (including all-nodata pixels) | Makes status codes a semantic fact about the request rather than a quirk of the TileMatrixSet or framework. |
| **Mismatched `rescale` count** (`N4`, PR #161) | Silently accepts e.g. 2 rescale ranges for 3 bands | **`400 Bad Request`** naming the range count and band count (`0`, `1`, and `N` remain valid) | Silent leniency hides bugs on ambiguous input; explicit error is safer for clients. |
| **Mosaic `/point` outside coverage** (`D1`/`M4`, PR #162) | `204 No Content` (empty body) | **`404 Not Found`** with JSON `{"detail": "No asset covers point (lon, lat)"}` (empty mosaic *tiles* stay `204`) | Point callers are scripts that benefit from a diagnostic body; `204` fits map tiles only. |
| **Unsupported CRS on `/tiles`** (`M1`, PR #167) | Returns `404` or GDAL error depending on route | **`400 Bad Request`** (`unsupported CRS`) uniformly across `/info`, `/bounds`, `/preview`, `/point`, and `/tiles` | Debugging blank map tiles immediately reveals the unsupported CRS instead of claiming "outside image bounds". |
| **Validation error body shape** (`E3`) | FastAPI/Pydantic nested `422` dict schema | Clear, concise error messages (JSON `detail`) | Avoids coupling to FastAPI's internal validation format. |
| **Band keys in `/statistics`** (`P4`/`M6`) | `b1`, `b2`, … | Keeps `b1`, `b2`, … (ported scripts index by them) | Preserves drop-in compatibility for existing TiTiler scripts. |

---

## 3. Deliberately Out of Scope (Parked)

- **JPEG 2000 (`B7`):** Requires C++ codec dependencies (`OpenJPEG`) and is rare in Cloud Optimized GeoTIFFs. *(Pure-Rust LERC / LERC_DEFLATE / LERC_ZSTD **is** supported via PR #164 with zero C/C++ dependencies, verified bit-exact on 18 GDAL-written files.)*
- **`/zarr/*` routes (`C6`):** Outside the scope of a COG / STAC / MosaicJSON server.
- **Brotli on `.npy` (Sweep #11):** `gzip` (`Accept-Encoding: gzip` and `*`) is supported across JSON, XML, HTML, `.npy`, and `.tif`, covering all standard HTTP clients without adding a Brotli encoder dependency.
