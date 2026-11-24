# API

`cogzilla-server` implements the routers of TiTiler 2.4 that do not need Postgres, so MapLibre, Leaflet, OpenLayers and QGIS clients can switch by changing the base URL. The `/cog`, `/stac`, `/colorMaps` and `/tileMatrixSets` routes below are covered by 39 differential HTTP checks against a live TiTiler (`benches/titiler_diff_oracle.py`), all passing when last run at v0.1.0. The mosaic routes and `algorithm=` are not covered yet.

**`/cog/*`** (TiTiler's `TilerFactory`)

- Tiles: `GET /cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@{scale}x][.{png|webp|jpg|npy|f32}]`, for `WebMercatorQuad` and `WorldCRS84Quad`. `@1x` is 256 px and `@2x` is 512 px.
- Tile parameters: `rescale`, `colormap_name`, `colormap` (JSON discrete or interval), `expression`, `algorithm` (`terrarium`, `terrainrgb`, `hillshade`, `slope`, `aspect`, with `azimuth`, `altitude`, `z_factor`), `resampling`, `reproject`, `nodata`, `unscale`, `return_mask`, `color_formula`, `buffer`.
- Metadata: `GET /cog/info`, `/cog/info.geojson`, `/cog/bounds`, `/cog/validate`, `/cog/{tms}/tilejson.json`, `/cog/{tms}/WMTSCapabilities.xml`, `/cog/tiles`, `/cog/tiles/{tms}`, `/cog/{tms}/map.html`.
- Point and statistics: `GET /cog/point/{lon},{lat}` (with `coord_crs`), `GET|POST /cog/statistics` (POST clips to a GeoJSON `Feature`, `FeatureCollection`, or geometry).
- Crops: `GET /cog/preview[/{w}x{h}][.{ext}]`, `GET /cog/bbox/{minx},{miny},{maxx},{maxy}[/{w}x{h}][.{ext}]`, `POST /cog/feature[/{w}x{h}][.{ext}]` (also accepts `?width=`/`?height=` query params).

**`/stac/*`** (TiTiler's `MultiBaseTilerFactory` with `STACReader`)

- `GET /stac/assets`, `/stac/bounds`, `/stac/info`, `/stac/info.geojson`, `/stac/asset_statistics`, `/stac/tiles`, `/stac/tiles/{tms}`, `/stac/tiles/{tms}/{z}/{x}/{y}[@{scale}x][.{ext}]`, `/stac/{tms}/tilejson.json`, `/stac/{tms}/WMTSCapabilities.xml`, `/stac/{tms}/map.html`, `/stac/point/{lon},{lat}`, `/stac/preview[/{w}x{h}][.{ext}]`, `/stac/bbox/...`; `GET|POST /stac/statistics`; `POST /stac/feature[/{w}x{h}][.{ext}]`.
- The STAC item can be a local path, an HTTP(S) URL or an `s3://` URI. Its JSON is cached, and assets named with `?assets=red&assets=green&assets=blue` are opened concurrently.
- Expressions can use band positions, `(b1-b2)/(b1+b2)`, or asset names, `(red-green)/(red+green)` or `{asset}_b{k}`.
- On the M5 at v0.1.0, with local fixtures, a warm cache and one request at a time, these routes ran 4–84× faster than TiTiler. The large multiples are tiles (2.1 ms against 52.3 ms, 25×), info (50×), TileJSON (70×) and three-asset point queries (84×); asset listing, statistics and previews were 4–11×.

**`/algorithms`**, **`/colorMaps`**, and **`/tileMatrixSets`**

- `GET /algorithms`, `GET /algorithms/{algorithmId}`: TiTiler `AlgorithmFactory` metadata for all 18 supported algorithms (`hillshade`, `slope`, `aspect`, `terrarium`, `terrainrgb`, `normalizedIndex`, `cast`, `ceil`, `floor`, `min`, `max`, `median`, `mean`, `std`, `var`, `sum`, `grayscale`, `bitonal`).
- `GET /colorMaps`, `GET /colorMaps/{colorMapId}`: the lookup tables match `rio-tiler`'s exactly, and `?f=png` returns a 256×20 colorbar.
- `GET /tileMatrixSets`, `GET /tileMatrixSets/{tileMatrixSetId}`: `WebMercatorQuad` (25 levels) and `WorldCRS84Quad` as OGC TileMatrixSet 2.0 JSON.

## Differences from TiTiler

- **Unsupported or invalid options return `422` JSON, never a silently different picture.** Both invalid parameter values and unsupported TiTiler options return HTTP `422` with FastAPI's `{"detail": [{"type": "value_error", "loc": ["query"], "msg": "..."}]}` JSON schema. Query parameters that neither server defines (cache-busters, tokens) are ignored, as in TiTiler.
- `resampling` and `reproject` accept `nearest`, `bilinear` and `cubic`. `lanczos` and `cubic_spline` are served as cubic. The other GDAL methods (`average`, `mode`, `gauss`, `rms`, `max`, `min`, `med`, `q1`, `q3`, `sum`) are rejected.
- `algorithm` accepts `hillshade`, `slope`, `aspect`, `terrarium`, `terrainrgb`, `normalizedIndex`, `cast`, `ceil`, `floor`, `min`, `max`, `median`, `mean`, `std`, `var`, `sum`, `grayscale` and `bitonal`. Only TiTiler's `contours` is rejected. `aspect` is a `cogzilla` addition.
- Tile matrix sets are `WebMercatorQuad` and `WorldCRS84Quad`; TiTiler's other eleven are rejected.
- Rejected parameters: `padding` (other than `0`), `dst_crs`, `crs`, and `cover_scale`.
- Statistics support `bidx`/`indexes`, custom percentiles (`p`), `histogram_bins`, `histogram_range`, `categorical`, `c`, and (on `/cog/statistics`) `;`-separated `expression`; `algorithm` is rejected on statistics routes. Point queries support `bidx`/`indexes` and `;`-separated `expression`. `POST /statistics` and `POST /feature` both support a single Feature, geometry, or `FeatureCollection` (up to 64 features).
- `coord_crs` accepts `EPSG:4326`/`CRS84`, `EPSG:3857`, and all 3,260+ supported EPSG codes on point queries and `/bbox`. `max_size` is supported on format-only `/bbox` routes (and rejected when explicit `{w}x{h}` dimensions are given). `POST /feature` also supports `?shape_mask=true|false`.
- `tilesize` accepts 64 to 2048.
- Output formats are `png`, `pngraw`, `jpg`, `webp`, `tif`/`tiff` (GeoTIFF), `npy` and raw `f32`. JPEG 2000 is not available.
- `expression` supports arithmetic, comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`), `where(cond, a, b)`, and functions (`sqrt`, `log10`, `ln`, `log2`, `db`, `exp`, `abs`, `ceil`, `floor`, `round`, `min`, `max`, `clip`), plus `;`-separated multi-band outputs on `/point`, `/cog/statistics`, and 3-band RGB tile/crop routes.
- A tile outside the dataset's bounds returns `404` on `/cog/tiles` and `/stac/tiles` (matching TiTiler); `/mosaic` returns a transparent `200` tile.
- Remote sources are `http(s)://`, `/vsicurl/http(s)://`, `s3://`, `/vsis3/`, `gs://`, `/vsigs/`, `az://`, `abfs[s]://`, and `/vsiaz/`. Other GDAL virtual filesystems such as `/vsizip/` or `/vsimem/` are not recognized.
- Mosaics: `GET /mosaicjson/{info,{tms}/tilejson.json,tiles/...}?url=...` serves quadkey-indexed MosaicJSON documents; `GET /mosaic/tiles/{tms}/{z}/{x}/{y}?url=a&url=b`, `GET /mosaic/{name}/tiles/{tms}/{z}/{x}/{y}`, and `POST /mosaic/register` serve front-to-back dynamic or spatially indexed mosaics (including live STAC-API `/search`) with early exit once 100% of pixels are filled.
- Shorthands that TiTiler does not have: `GET /cog/tiles/{z}/{x}/{y}` assumes `WebMercatorQuad`, `expr=` is an alias for `expression=` (both accept the preset names), and `/tiles/{name}/...` serves datasets registered at startup.

## Coordinate systems

A source file is served only if `cogzilla` can place it correctly. Anything else returns `400` with the reason, rather than a picture in the wrong location.

- **By EPSG code**: `3857` and 3,260 projected codes from PROJ's EPSG database. That is every code that uses a supported method on a supported datum, including WGS84, NAD83 and ETRS89 UTM, Australian MGA, SIRGAS UTM, NZTM2000, SWEREF99, NAD83 state plane (metres and feet), Canada Atlas Lambert, the Alaska, British Columbia, Australian and CONUS Albers grids, ETRS89 LAEA Europe (`3035`), EASE-Grid 2.0 North and South, UPS and the NSIDC polar grids.
- **Geographic, in degrees**: WGS84 (`4326`), NAD83 (`4269`), ETRS89 (`4258`), GDA94, GDA2020 and 267 other geographic codes on a supported datum.
- **User-defined** (parameters stored in the file, no EPSG code): Transverse Mercator, Lambert Conformal Conic with one or two standard parallels, Albers Equal Area, polar stereographic with either a latitude of true scale or a scale factor at the pole, Lambert azimuthal equal-area, and sinusoidal (including the MODIS sphere). Units may be metres, feet or US survey feet.
- **Datums**: no datum shift is applied, so a coordinate system is accepted only when its ellipsoid is WGS84 or GRS80 and PROJ's own transformation to WGS84 moves points by less than 2 m anywhere in its area of use. NAD27, OSGB36 and other older datums are refused. The two equal-area world projections are also accepted on a sphere, whose latitude and longitude are taken as WGS84's, as PROJ and GDAL do.
- **Not supported**: Mercator variants other than `3857`, equirectangular, cylindrical equal-area (EASE-Grid 2.0 global, `6933`), oblique projections (Swiss, Dutch and Czech grids), and anything needing a datum shift (British National Grid, NAD27).

The EPSG table is generated from PROJ's database by `benches/gen_epsg_table.py`, and every entry is checked against PROJ to one millimetre at five points in its area of use (`crates/cogzilla-proj/tests/epsg_table.rs`). A further 496 reference points cover wide-area and user-defined cases (`tests/proj_reference.rs`), and `benches/projection_oracle.py` checks 64 coordinate systems end to end through real GeoTIFFs.

## Example requests

```text
# Pre-registered Landsat C2 three-file uint16 RGB
GET /tiles/landsat/12/652/1583.png?rescale=7500,22500

# Pre-registered OPERA RTC-S1 dual-pol SAR false color ([VV_dB, VH_dB, VV_dB - VH_dB])
GET /tiles/sar/12/652/1583.png?expr=sar_rgb

# Landsat NDVI composited from two S3 COGs, as WebP
GET /cog/tiles/WebMercatorQuad/12/652/1583.webp?url=s3://bucket/B5.tif&url=s3://bucket/B4.tif&expr=ndvi&rescale=-0.2,0.8&colormap_name=rdylgn&quality=82

# Single-pol SAR linear power to dB, as raw float32 for WebGL or deck.gl
GET /cog/tiles/WebMercatorQuad/12/652/1583.f32?url=s3://bucket/VV.tif&expr=db
```

## Live demos on public data

The quickest way in is the landing page at `http://localhost:8080/`, which has these demos as buttons, a field for your own COG URL, and per-tile timings. It loads MapLibre and an OpenStreetMap basemap from the internet.

The same maps as direct links, with a server on `localhost:8080` (public AWS S3 COGs, no credentials needed):

- **Sentinel-2 true color** (northeast Colorado):
  ```text
  http://localhost:8080/cog/WebMercatorQuad/map.html?url=https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/T/EE/2024/9/S2A_13TEE_20240929_0_L2A/TCI.tif
  ```
- **Mt. Rainier on-the-fly DEM hillshade** (Copernicus 30m DEM, cubic Horn 3×3):
  ```text
  http://localhost:8080/cog/WebMercatorQuad/map.html?url=https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N46_00_W122_00_DEM/Copernicus_DSM_COG_10_N46_00_W122_00_DEM.tif&algorithm=hillshade&azimuth=315&altitude=45&z_factor=1.5&resampling=cubic
  ```
- **Mt. Rainier slope (`inferno`, 0–60°)** and **aspect (`hsv`, 0–360°)**:
  ```text
  http://localhost:8080/cog/WebMercatorQuad/map.html?url=https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N46_00_W122_00_DEM/Copernicus_DSM_COG_10_N46_00_W122_00_DEM.tif&algorithm=slope&rescale=0,60&colormap_name=inferno&resampling=cubic
  http://localhost:8080/cog/WebMercatorQuad/map.html?url=https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N46_00_W122_00_DEM/Copernicus_DSM_COG_10_N46_00_W122_00_DEM.tif&algorithm=aspect&colormap_name=hsv&resampling=cubic
  ```
- **3D Terrain tiles** (drop into MapLibre GL JS `raster-dem` or Deck.gl `TerrainLayer` with `terrarium` or `terrainrgb`):
  ```text
  GET http://localhost:8080/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N46_00_W122_00_DEM/Copernicus_DSM_COG_10_N46_00_W122_00_DEM.tif&algorithm=terrarium
  GET http://localhost:8080/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N46_00_W122_00_DEM/Copernicus_DSM_COG_10_N46_00_W122_00_DEM.tif&algorithm=terrainrgb
  ```

Or request tiles from any COG reachable over HTTP, S3 (`s3://`), GCS (`gs://`), or Azure Blob (`az://`, `abfs://`):

```text
GET http://localhost:8080/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=s3://your-bucket/your.tif
```
