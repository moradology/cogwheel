# API

`cogzilla-server` implements the routers of TiTiler 2.4 that do not need Postgres, so MapLibre, Leaflet, OpenLayers and QGIS clients can switch by changing the base URL. The `/cog`, `/stac`, `/colorMaps`, and `/tileMatrixSets` routes below are covered by 39 differential HTTP checks against a live TiTiler (`benches/titiler_diff_oracle.py`), all passing on `v0.7.0`. The titiler-pgstac routes, which do need Postgres, are described under [pgSTAC](#pgstac).

**`/cog/*`** (TiTiler's `TilerFactory`)

- Tiles: `GET /cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@{scale}x][.{png|pngraw|webp|jpg|tif|tiff|npy|f32}]`, across all 13 `morecantile` OGC TileMatrixSets (`WebMercatorQuad`, `WorldCRS84Quad`, `WGS1984Quad`, `WorldMercatorWGS84Quad`, `CanadianNAD83_LCC`, `EuropeanETRS89_LAEAQuad`, `NZTM2000Quad`, `UPSArcticWGS84Quad`, `UPSAntarcticWGS84Quad`, `UTM31WGS84Quad`, `GNOSISGlobalGrid`, `CDB1GlobalGrid`, `LINZAntarticaMapTilegrid`). `@1x` is 256 px and `@2x` is 512 px (`@1x`..`@4x` supported).
- Tile parameters: `rescale`, `colormap_name`, `colormap` (JSON discrete or interval, plus automatic TIFF Tag 320 embedded palette support), `expression`, `algorithm` (all 19 TiTiler algorithms + `aspect`, with `algorithm_params` JSON), `resampling` & `reproject` (all 15 GDAL methods), `nodata`, `unscale`, `return_mask`, `color_formula`, `buffer` (fractional float).
- Metadata & OGC: `GET /conformance`, `/cog/info`, `/cog/info.geojson`, `/cog/bounds`, `/cog/validate`, `/cog/{tms}/tilejson.json`, `/cog/{tms}/WMTSCapabilities.xml`, `/cog/tiles`, `/cog/tiles/{tms}`, `/cog/{tms}/map.html`, `/cog/map`.
- Point and statistics: `GET /cog/point/{lon},{lat}` (with `coord_crs`), `GET|POST /cog/statistics` (POST clips to a GeoJSON `Feature`, `FeatureCollection`, or geometry).
- Crops: `GET /cog/preview[/{w}x{h}][.{ext}]`, `GET /cog/bbox/{minx},{miny},{maxx},{maxy}[/{w}x{h}][.{ext}]`, `POST /cog/feature[/{w}x{h}][.{ext}]` (supports `?width=`, `?height=`, `?max_size=0` for full native resolution, `?coord_crs=`, `?dst_crs=`, `?crs=`).

**`/stac/*`** (TiTiler's `MultiBaseTilerFactory` with `STACReader`)

- `GET /stac/assets`, `/stac/bounds`, `/stac/info`, `/stac/info.geojson`, `/stac/asset_statistics`, `/stac/tiles`, `/stac/tiles/{tms}`, `/stac/tiles/{tms}/{z}/{x}/{y}[@{scale}x][.{ext}]`, `/stac/{tms}/tilejson.json`, `/stac/{tms}/WMTSCapabilities.xml`, `/stac/{tms}/map.html`, `/stac/map`, `/stac/point/{lon},{lat}`, `/stac/preview[/{w}x{h}][.{ext}]`, `/stac/bbox/...`; `GET|POST /stac/statistics`; `POST /stac/feature[/{w}x{h}][.{ext}]`.
- Supports per-asset `?asset_bidx=name|1,2,3`, mixed-resolution & mixed-CRS multi-asset compositing (e.g. Sentinel-2 10 m + 20 m bands), and `X-Assets` response headers.
- The STAC item can be a local path, an HTTP(S) URL or an `s3://` URI. Its JSON is cached, and assets named with `?assets=red&assets=green&assets=blue` are opened concurrently.
- Expressions can use band positions, `(b1-b2)/(b1+b2)`, or asset names, `(red-green)/(red+green)` or `{asset}_b{k}`.

**`/mosaicjson/*`**, **`/algorithms`**, **`/colorMaps`**, and **`/tileMatrixSets`**

- `GET /mosaicjson/{info,info.geojson,bounds,assets,statistics,point/{lon},{lat},bbox/{bbox}[/{w}x{h}][.{ext}],tiles/{tms}/{z}/{x}/{y}[@scale][.{ext}],{tms}/tilejson.json,{tms}/WMTSCapabilities.xml,{tms}/map.html}` and `POST /mosaicjson/{validate,statistics,feature[/{w}x{h}][.{ext}]}`.
- `GET /algorithms`, `GET /algorithms/{algorithmId}`: TiTiler `AlgorithmFactory` metadata for all 19 supported algorithms (`hillshade`, `slope`, `aspect`, `contours`, `terrarium`, `terrainrgb`, `normalizedIndex`, `cast`, `ceil`, `floor`, `min`, `max`, `median`, `mean`, `std`, `var`, `sum`, `grayscale`, `bitonal`).
- `GET /colorMaps`, `GET /colorMaps/{colorMapId}`: all 211 lookup tables match `rio-tiler` bit-for-bit, and `?f=png` returns a 256×20 colorbar.
- `GET /tileMatrixSets`, `GET /tileMatrixSets/{tileMatrixSetId}`: all 13 `morecantile` OGC TileMatrixSet 2.0 JSON definitions.

## pgSTAC

With a [pgSTAC database configured](configuration.md#pgstac-database), `cogzilla` serves the search routes of titiler-pgstac 3.2. The items behind each tile come from pgSTAC's own `geojsonsearch`, called with the arguments titiler-pgstac passes, so both servers build a tile from the same items in the same order. `benches/pgstac_diff_oracle.py` sends 51 requests to both servers on one database and compares status codes, JSON bodies, item lists and rendered tiles; all 51 agreed when last run, at v0.5.19 against titiler-pgstac 3.2.2 and pgSTAC 0.10.2. Agreement on tiles means the same valid pixels and a PSNR of at least 25 dB; the 16 tile pairs had a median of 37.5 dB (30.1 to 43.4).

- `POST /searches/register` with a STAC search body (`collections`, `ids`, `bbox`, `intersects`, `datetime`, `query`, `filter`, `sortby`) and optional `metadata`. The returned `id` is pgSTAC's hash, the same one titiler-pgstac returns for the same body.
- `GET /searches/` and `/searches/list` (with `limit`, `offset`, and `sortby` on `lastused` or `usecount`), `GET /searches/{id}/info`.
- `GET /searches/{id}/{tms}/tilejson.json` and `/searches/{id}/{tms}/map.html`.
- `GET /searches/{id}/tiles/{tms}/{z}/{x}/{y}[.{png|jpg|webp|npy}]` and `GET /searches/{id}/tiles/{tms}/{z}/{x}/{y}/assets`.
- Tile parameters: `assets` (required, one to three per item), `expression`, `rescale`, `colormap_name`, `resampling`, `reproject`, `color_formula`, `return_mask`, `unscale` (single asset), `tilesize` (256 or 512), and the search limits `scan_limit`, `items_limit`, `time_limit`, `exitwhenfull` and `skipcovered`.

Responses follow titiler-pgstac: errors are JSON `{"detail": ...}`, a missing `assets` is a `422`, a tile no item covers is a `204`, and a tile with no extension is a JPEG when every pixel is valid and a PNG otherwise. Resampling follows TiTiler on these routes: nearest neighbour by default, with `resampling` applied to sources already in Web Mercator and `reproject` to sources that have to be reprojected.

Not there yet, each answered with a `400` or `404` rather than a different picture:

- The rest of the search routes (`point`, `bbox`, `feature`, `statistics`, WMTS, `/tiles` listings) and the `/collections/{id}` and `/collections/{id}/items/{id}` families.
- `pixel_selection` other than `first`, per-asset options (`assets=name|bidx=...`), `asset_as_band`, `buffer`, `nodata`, a JSON `colormap`, more than three assets, and assets of one item on different grids.
- Tile matrix sets other than `WebMercatorQuad`.

## Notes on TiTiler Compatibility

- **FastAPI-compatible error responses**: invalid parameter values and unsupported options return HTTP `422` (or `400`/`404`) with FastAPI's JSON `{"detail": ...}` schema. Query parameters that neither server defines (cache-busters, tokens) are ignored, as in TiTiler.
- **Resampling**: `resampling` and `reproject` implement all 15 GDAL methods (`nearest`, `bilinear`, `cubic`, `cubic_spline`, `lanczos`, `gauss`, `average`, `mode`, `rms`, `min`, `max`, `med`, `q1`, `q3`, `sum`).
- **GeoTIFF (`.tif`/`.tiff`) output**: styled renders (`rescale`, `colormap_name`, `colormap`, `color_formula`, `hillshade`, `terrarium`, `terrainrgb`, 8-bit RGB) emit 8-bit RGB/RGBA GeoTIFFs; unstyled scientific/float/DEM/expression renders emit native IEEE Float32 GeoTIFFs (`SampleFormat=3`, `GDAL_NODATA="nan"`).
- **Output formats**: `png`, `pngraw`, `jpg`/`jpeg`, `webp`, `tif`/`tiff` (GeoTIFF), `npy` (`application/x-binary`), and raw `f32`. JPEG 2000 is not available.
- **Response headers**: `Server-Timing: total;dur=<ms>`, `ETag` + `304 Not Modified`, and `X-Assets` on `/stac/*`.
- **Statistics & points**: statistics support `bidx`/`indexes`, `asset_bidx`, custom percentiles (`p`), `histogram_bins`, `histogram_range`, `categorical`, `c`, and `;`-separated `expression`; `algorithm` is rejected on statistics routes. Point queries support `bidx`/`indexes`, `asset_bidx`, and `;`-separated `expression`. `POST /statistics` and `POST /feature` both support a single Feature, geometry, or `FeatureCollection` (up to 64 features).
- **CRS & dimensions**: `coord_crs`, `dst_crs`, and `crs` accept `EPSG:4326`/`CRS84`, `EPSG:3857`, `EPSG:3395`, and all 3,260+ supported EPSG codes. `max_size=0` renders at full native resolution (up to 4096 px); `tilesize` accepts 64 to 2048.
- **Expressions**: `expression` supports arithmetic, comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`), `where(cond, a, b)`, and functions (`sqrt`, `log10`, `ln`, `log2`, `db`, `exp`, `abs`, `ceil`, `floor`, `round`, `min`, `max`, `clip`), plus `;`-separated multi-band outputs on `/point`, `/cog/statistics`, and 3-band RGB tile/crop routes.
- **Out-of-bounds tiles**: returns `404` on `/cog/tiles` and `/stac/tiles` (matching TiTiler); `/mosaic` returns a transparent `200` tile.
- **Remote sources & auth**: `http(s)://`, `/vsicurl/http(s)://`, `s3://`, `/vsis3/`, `gs://`, `/vsigs/`, `az://`, `abfs[s]://`, and `/vsiaz/`, with `GDAL_HTTP_BEARER`, `GDAL_HTTP_USERPWD`, `GDAL_HTTP_HEADERS`, and `GDAL_HTTP_HEADER_FILE` (gated on `COGZILLA_ALLOWED_URL_PREFIXES`).
- **Cogzilla extensions**: `GET /cog/tiles/{z}/{x}/{y}` assumes `WebMercatorQuad`, `expr=` aliases `expression=` (including presets), `/tiles/{name}/...` serves pre-registered datasets, and `/mosaic/{tiles,register,{name}/tiles}` serves dynamic/spatially indexed mosaics (including live STAC-API `/search`).

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
