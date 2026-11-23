# API

`cogzilla-server` implements the routers of TiTiler 2.4 that do not need Postgres, so MapLibre, Leaflet, OpenLayers and QGIS clients can switch by changing the base URL. The `/cog`, `/stac`, `/colorMaps` and `/tileMatrixSets` routes below are covered by 39 differential HTTP checks against a live TiTiler (`benches/titiler_diff_oracle.py`), all passing when last run at v0.1.0. The mosaic routes and `algorithm=` are not covered yet.

**`/cog/*`** (TiTiler's `TilerFactory`)

- Tiles: `GET /cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@{scale}x][.{png|webp|jpg|f32}]`, for `WebMercatorQuad` and `WorldCRS84Quad`. `@1x` is 256 px and `@2x` is 512 px.
- Tile parameters: `rescale`, `colormap_name`, `colormap` (JSON), `expression`, `algorithm` (`terrarium`, `terrainrgb`, `hillshade`, `slope`, `aspect`, with `azimuth`, `altitude`, `z_factor`), `resampling`, `nodata`, `unscale`, `return_mask`, `color_formula`, `buffer`.
- Metadata: `GET /cog/info`, `/cog/info.geojson`, `/cog/bounds`, `/cog/{tms}/tilejson.json`, `/cog/tiles`, `/cog/tiles/{tms}`, `/cog/{tms}/map.html`.
- Point and statistics: `GET /cog/point/{lon},{lat}` (with `coord_crs`), `GET|POST /cog/statistics` (POST clips to a GeoJSON polygon).
- Crops: `GET /cog/preview[/{w}x{h}][.{ext}]`, `GET /cog/bbox/{minx},{miny},{maxx},{maxy}[/{w}x{h}][.{ext}]`, `POST /cog/feature[/{w}x{h}][.{ext}]`.

**`/stac/*`** (TiTiler's `MultiBaseTilerFactory` with `STACReader`)

- `GET /stac/assets`, `/stac/info`, `/stac/info.geojson`, `/stac/asset_statistics`, `/stac/tiles`, `/stac/tiles/{tms}`, `/stac/tiles/{tms}/{z}/{x}/{y}[@{scale}x][.{ext}]`, `/stac/{tms}/tilejson.json`, `/stac/{tms}/map.html`, `/stac/point/{lon},{lat}`, `/stac/preview[/{w}x{h}][.{ext}]`, `/stac/bbox/...`; `GET|POST /stac/statistics`; `POST /stac/feature[/{w}x{h}][.{ext}]`.
- The STAC item can be a local path, an HTTP(S) URL or an `s3://` URI. Its JSON is cached, and assets named with `?assets=red&assets=green&assets=blue` are opened concurrently.
- Expressions can use band positions, `(b1-b2)/(b1+b2)`, or asset names, `(red-green)/(red+green)`.
- On the M5 at v0.1.0, with local fixtures, a warm cache and one request at a time, these routes ran 4–84× faster than TiTiler. The large multiples are tiles (2.1 ms against 52.3 ms, 25×), info (50×), TileJSON (70×) and three-asset point queries (84×); asset listing, statistics and previews were 4–11×.

**`/colorMaps`** and **`/tileMatrixSets`**

- `GET /colorMaps`, `GET /colorMaps/{colorMapId}`: the lookup tables match `rio-tiler`'s exactly, and `?f=png` returns a 256×20 colorbar.
- `GET /tileMatrixSets`, `GET /tileMatrixSets/{tileMatrixSetId}`: `WebMercatorQuad` (25 levels) and `WorldCRS84Quad` as OGC TileMatrixSet 2.0 JSON.

## Differences from TiTiler

- **Unsupported options return `400`, never a silently different picture.** TiTiler answers an invalid value with `422`; `cogzilla` answers `400` with a plain-text reason, both for invalid values and for the TiTiler options listed below that it does not implement. Query parameters that neither server defines (cache-busters, tokens) are ignored, as in TiTiler.
- `resampling` accepts `nearest`, `bilinear` and `cubic`. `lanczos` and `cubic_spline` are served as cubic. The other GDAL methods (`average`, `mode`, `gauss`, `rms`, `max`, `min`, `med`, `q1`, `q3`, `sum`) are rejected.
- `algorithm` accepts `hillshade`, `slope`, `aspect`, `terrarium` and `terrainrgb`. TiTiler's `contours`, `normalizedIndex`, `cast`, `ceil`, `floor`, `min`, `max`, `median`, `mean`, `std`, `var`, `sum`, `grayscale` and `bitonal` are rejected, and there are no `/algorithms` routes. `aspect` is a `cogzilla` addition.
- Tile matrix sets are `WebMercatorQuad` and `WorldCRS84Quad`; TiTiler's other eleven are rejected.
- Rejected parameters: `padding` (other than `0`), `reproject`, `dst_crs`, `crs`, `width` and `height` as query parameters (use the `{w}x{h}` path form), `minzoom`, `maxzoom`, `tile_format`, `histogram_bins`, `histogram_range`, `categorical`, `c`, `p`, `cover_scale`, and `orientation` on colormap images.
- Statistics always report the 2nd and 98th percentiles and a 10-bin histogram over every band; `expression`, `algorithm` and `bidx` are rejected there and on point queries. `POST /statistics` takes a single Feature or geometry, not a FeatureCollection.
- `coord_crs` accepts `EPSG:4326` and `EPSG:3857`, on point queries and `/cog/bbox`. `max_size` is rejected on bbox routes.
- `tilesize` accepts 64 to 2048.
- Output formats are `png`, `jpg`, `webp`, `npy` and raw `f32`. GeoTIFF, JPEG 2000 and `pngraw` are not available.
- `expression` supports arithmetic and functions such as `sqrt` and `log`, but not multiple `;`-separated outputs, comparisons or `where(...)`.
- A tile outside the dataset's bounds returns a transparent tile with `200`; TiTiler returns `404`.
- Remote sources are `http(s)://`, `s3://`, `/vsis3/`, `gs://`, `/vsigs/`, `az://`, `abfs[s]://`, and `/vsiaz/`. Other GDAL virtual filesystems such as `/vsicurl/` are not recognized.
- Mosaics: `GET /mosaicjson/{info,{tms}/tilejson.json,tiles/...}?url=...` serves quadkey-indexed MosaicJSON documents; `GET /mosaic/tiles/{tms}/{z}/{x}/{y}?url=a&url=b`, `GET /mosaic/{name}/tiles/{tms}/{z}/{x}/{y}`, and `POST /mosaic/register` serve front-to-back dynamic or spatially indexed mosaics (including live STAC-API `/search`) with early exit once 100% of pixels are filled.
- Shorthands that TiTiler does not have: `GET /cog/tiles/{z}/{x}/{y}` assumes `WebMercatorQuad`, `expr=` is an alias for `expression=` (both accept the preset names), and `/tiles/{name}/...` serves datasets registered at startup.

## Coordinate systems

A source file is served only if `cogzilla` can place it correctly. Anything else returns `400` with the reason, rather than a picture in the wrong location.

- **By EPSG code**: `3857`, WGS84 UTM (`32601`–`32660`, `32701`–`32760`), NAD83 UTM (`26901`–`26923`), CONUS Albers (`5070`), California Albers (`3310`), Lambert-93 (`2154`), and polar stereographic `3413` and `3031`.
- **Geographic, in degrees**: WGS84 (`4326`), NAD83 (`4269`), ETRS89 (`4258`), GDA94, GDA2020 and the other datums listed below.
- **User-defined** (parameters stored in the file, no EPSG code): Transverse Mercator, Lambert Conformal Conic with one or two standard parallels, Albers Equal Area, and polar stereographic with either a latitude of true scale or a scale factor at the pole. Units may be metres, feet or US survey feet.
- **Datums**: no datum shift is applied, so only WGS84 and datums that agree with it to about two metres are accepted: NAD83 and its realizations, ETRS89, GDA94, GDA2020, NZGD2000, SIRGAS, JGD2000, JGD2011, Hartebeesthoek94, RGF93 and SWEREF99. NAD27, OSGB36 and other older datums are refused.
- **Not supported**: any other EPSG code (including ETRS89 UTM, MGA, state plane and national grids, even where the projection method is one of the above), and the Lambert azimuthal equal-area, Mercator variants other than `3857`, sinusoidal, equirectangular, cylindrical equal-area and oblique projections.

The projection math is checked against PROJ to one millimetre at 323 reference points (`crates/cogzilla-proj/tests/proj_reference.rs`), and `benches/projection_oracle.py` checks 55 coordinate systems end to end through real GeoTIFFs.

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
