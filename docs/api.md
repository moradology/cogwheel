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

With a [pgSTAC database configured](configuration.md#pgstac-database), `cogzilla` serves the search, collection and item routes of titiler-pgstac 3.2. The items behind a tile come from pgSTAC's own `geojsonsearch`, called with the arguments titiler-pgstac passes, or from the [lean lookup](#the-lean-item-lookup), which lists the same items in the same order without the items a picture cannot show. `benches/pgstac_diff_oracle.py` sends 389 requests to both servers on one database and compares status codes, JSON bodies, item lists, WMTS documents, statistics and rendered images. When last run, at v0.5.29 against titiler-pgstac 3.2.2 and pgSTAC 0.10.2, 370 agreed and 8 were known differences, listed below. Agreement on images means the same valid pixels and a PSNR of at least 25 dB; none of the agreeing pairs is identical now that pixels are projected exactly, and they were 25.8 dB and up.

- `POST /searches/register` with a STAC search body (`collections`, `ids`, `bbox`, `intersects`, `datetime`, `query`, `filter`, `sortby`) and optional `metadata`. The returned `id` is pgSTAC's hash, the same one titiler-pgstac returns for the same body.
- `GET /searches/` and `/searches/list` (with `limit`, `offset`, and `sortby` on `lastused` or `usecount`), `GET /searches/{id}/info`.
- `GET /searches/{id}/{tms}/tilejson.json` and `/searches/{id}/{tms}/map.html`.
- `GET /searches/{id}/tiles/{tms}/{z}/{x}/{y}[.{png|jpg|webp|npy}]` and `GET /searches/{id}/tiles/{tms}/{z}/{x}/{y}/assets`.
- `GET /searches/{id}/tiles` and `/searches/{id}/tiles/{tms}`: the OGC tileset list and tileset description, with tile limits per zoom computed from the mosaic's bounds.
- `GET /searches/{id}/WMTSCapabilities.xml`: the RESTful WMTS description, for QGIS, ArcGIS and other desktop clients. Each layer in the mosaic's `defaults` is a WMTS layer, and so is the request's own query string when it has `assets`. `tile_format` (png, jpg, jpeg, webp, npy), `use_epsg` and `zooms=WebMercatorQuad::min,max` are taken; with nothing that can make a tile the answer is a `400`.
- `GET /searches/{id}/point/{lon},{lat}?assets=...`: the pixel values of every item under the point, in the search's order, each as `name` (`collection/item`), `values`, `band_names` and `band_descriptions`. `coord_crs` may be `epsg:4326` or `epsg:3857`, `items_limit` and the other search limits apply (up to 100 items by default), and a point no item covers is a `204`. `expression` does band math over `b1..bn`, the stacked bands of the assets in the order given, in double precision; several expressions are separated by `;`. As in titiler-pgstac, asset names are not accepted there.
- `GET /searches/{id}/bbox/{minx},{miny},{maxx},{maxy}.{format}` and `.../{width}x{height}.{format}`, and `POST /searches/{id}/feature[.{format}]` and `/feature/{width}x{height}.{format}` with a GeoJSON feature: one image of the mosaic over a box in WGS84, clipped to the polygon for `feature`. The size follows rio-tiler: `{width}x{height}`, or `width` and `height`, or one of them with the other from the box's shape, or `max_size` as an upper limit, or else the first item's own resolution, that of its finest asset. An image may be at most 4096 pixels on a side. A JPEG has no mask for the polygon: without `rescale` the whole box is written as it was read, and with it everything outside the polygon is black, both as in TiTiler. See [Images](#images) for how they are read.
- `POST /searches/{id}/statistics?assets=...` with a GeoJSON Feature or FeatureCollection: statistics of the mosaic under each feature, added to its properties as `statistics` with the items read as `used_assets`. The numbers are rio-tiler's: every pixel the polygon touches counts, weighted by the share of it under the polygon (`cover_scale`, 10 by default), and `count`, `sum`, `mean`, `std`, the median and the percentiles use those weights. `max_size`, or `width` and `height`, set the size the feature is read at; without them it is read at the first item's own resolution, up to 8,388,608 pixels. `expression` does band math over `b1..bn` as on `point`. `categorical`, `c`, `p`, `histogram_bins` and `histogram_range` are TiTiler's. The pixels are the ones GDAL's nearest-neighbour warp reads: the same grid, the same overview and the same source pixel for each output pixel, so the results equal titiler-pgstac's except for its single-precision sums. A feature with no item under it is a `204`.
- `GET /searches/{id}/bbox/{minx},{miny},{maxx},{maxy}/assets` and `GET /searches/{id}/point/{lon},{lat}/assets`: the items that intersect a box or cover a point, in WGS84, with the same search limits as a tile.
- `GET /collections/{collection_id}/...`: every search route above except `register` and the listing, for the mosaic of one collection. The search is the collection, narrowed by `ids`, `bbox`, `datetime`, `sortby`, `query` and `filter` from the query string. A `filter` is CQL2 text unless `filter-lang=cql2-json` says otherwise, and text is turned into the CQL2 JSON titiler-pgstac's converter gives for it (numbers as floats, timestamps in UTC, WKT as GeoJSON), so both servers register the same search. A filter that does not parse, or is not a predicate, is a `422`. It is registered in pgSTAC under the hash titiler-pgstac derives for the same request, with the same metadata: bounds from the `bbox` parameter or the collection's extent, the asset names from `item_assets`, and the collection's [renders](https://github.com/stac-extensions/render) as `defaults`.
- A mosaic whose metadata has `defaults` gets one extra TileJSON and map link per layer in `info` (and one TileJSON link per layer in the `register` response), with the layer's parameters as the query string. A layer without `assets` gets no link.
- `assets` takes TiTiler's per-asset options after a `|`: `assets=visual|bidx=3,2,1` picks bands of that asset and `assets=red|expression=b1*2` does band math over them. `asset_as_band=true` names each band after its asset and is only for single-band assets. A top-level `expression` is over `b1..bn`, the bands that are left once the options are applied; asset names in it are a `400`, as in TiTiler 2, and a bare `bidx` is passed over. `point` and `statistics` take every combination. Tiles and `bbox`/`feature` images take `bidx` with one or three bands or one expression on a single asset, and `bidx=1` on each of several; other combinations are a `400` there.
- Tile parameters: `assets` (required, up to sixteen per item, each with TiTiler's `|bidx=` and `|expression=` options), `asset_as_band`, `expression` (several results separated by `;`), `rescale`, `colormap_name`, `resampling`, `reproject`, `color_formula`, `return_mask`, `unscale` (single asset), `tilesize` (256 or 512), `pixel_selection` (all nine methods), and the search limits `scan_limit`, `items_limit`, `time_limit`, `exitwhenfull` and `skipcovered`.

Responses follow titiler-pgstac: errors are JSON `{"detail": ...}`, a missing `assets` is a `422`, a tile no item covers is a `204`, and a tile with no extension is a JPEG when every pixel is valid and a PNG otherwise. Resampling follows TiTiler on these routes: nearest neighbour by default, with `resampling` applied to sources already in Web Mercator and `reproject` to sources that have to be reprojected.

`/openapi.json` lists all 63 operations titiler-pgstac 3.2 has under `/searches` and `/collections`. Of the routes titiler-pgstac serves outside those two prefixes, `/algorithms`, `/algorithms/{algorithmId}` and `/conformance` are not on this branch.

### The lean item lookup

pgSTAC's `geojsonsearch` is most of what a mosaic tile costs: a plpgsql loop that writes each item it keeps to a temporary table and decides what to keep with a growing `ST_Union`. On pgSTAC 0.9 and 0.10 the tile, `bbox`, `feature` and `point` routes list their items another way. They take the search's `WHERE` and `ORDER BY` from pgSTAC, walk the same time chunks in the same order with plain queries that return an id, a footprint and the hrefs asked for (one query for most tiles, and cursors read a page at a time for the ones that go on), and decide what to keep on a grid of sample points: five a side for each output pixel (three or one for larger pictures), one of them on the pixel's centre. An item is kept when its footprint holds a sample no earlier item's did, and the walk ends when every sample is held.

On the reference catalog (46,476 Sentinel-2 items in 60 monthly partitions, pgSTAC 0.10.2), over 279 tiles and three searches (the collection newest first, oldest first, and with `eo:cloud_cover` below 10), timed through the `/assets` route: at zoom 8 and above the lookup has a median of 1.3 to 1.4 ms where `geojsonsearch` has 18 to 24 ms, and at zooms 5 to 7 it has 30 to 57 ms against 1.0 to 1.4 s. A warm tile request at zoom 12 falls from 29 ms to 5 ms. The lists were identical on 236 tiles. On the other 43 the lean list was `geojsonsearch`'s in the same order with items left out, 955 in all, each of which was the first to cover a median of 0.00002 of a pixel and at most half a pixel, in strips too thin to hold a sample; on 12 of those tiles `geojsonsearch` had stopped at its 100-item limit and the lean list went on. `benches/pgstac_lean_check.py` repeats the measurement. On pgSTAC 0.9 the evidence is narrower: the HTTP tests, which compare the two lookups on items made for the purpose in both partition layouts, run against 0.9.11 in CI; no catalog of that size has been compared there.

- The list is `geojsonsearch`'s, in its order, less the items it keeps for an area that falls between sample points. Where `geojsonsearch` stops at its item or time limit before the tile is covered, the lean lookup goes on to items it never reached.
- It reads only. It calls pgSTAC's `search_query` once per search each cache period, not once per tile, so `usecount` no longer counts tiles; `lastused` still moves while a search is in use. The list of time chunks is compiled with the search, once per cache period (`TITILER_PGSTAC_CACHE_TTL`, 300 seconds). Items that arrive in between are read at once, since every chunk is still walked, empty or not; a new partition outside the chunks, which is to say more than a month past the newest, waits for the next compile.
- Whatever it cannot answer goes to `geojsonsearch`: an item kept that lacks an asset asked for (so the error names the item's assets, as before), a footprint that is not made of polygons, lines and points, a request with no sample point in it, a picture larger than 4096 pixels a side or sized by the data's own resolution, and any failure of its own queries.
- The `/assets` routes and `statistics` keep `geojsonsearch`'s list: the first because they are what titiler-pgstac answers, the second because every raster is read on a grid of its own and the numbers are compared digit for digit.
- `item_lookup=lean` or `item_lookup=geojsonsearch` chooses for one request, on any route that lists items. On an `/assets` route `item_lookup=lean` returns what a picture would be given (ids, and the hrefs of any `assets` named), sampled as a 256-pixel tile (`tilesize` changes that) or as a `width` by `height` or `max_size` image of the box; `benches/pgstac_lean_check.py` compares the two lists this way. [`COGZILLA_PGSTAC_LEAN`](configuration.md#pgstac-database) sets the default.

### Pixel selection

On tiles, on `bbox` and `feature` images and on mosaic `statistics`, `pixel_selection` takes `first` (the default), `highest`, `lowest`, `mean`, `median`, `stdev` and `count`, `lastbandlow` and `lastbandhigh`, with rio-tiler's arithmetic: each value of each band is decided on its own over the scenes that have one there; `mean` and `median` are cast back to the source's type, which for whole numbers drops the fraction; `stdev` is the population figure; `count` is one band, valid everywhere; `lastbandlow` and `lastbandhigh` take a scene's values where its last band is below or above the one kept, and leave that band out, so they need at least two. On pictures, `rescale`, `colormap_name` and `color_formula` then apply to the result, so `stdev` and `count` want a `rescale` of their own. Statistics are those of the result, worked out in double precision, and list every item that was read; with `count` they are of one band, described as the first.

Only `first` stops when the picture is full. The others read every item the search lists (on tiles up to eight at a time), and by default that list is still pgSTAC's: with `skipcovered` and `exitwhenfull` left on, it holds only the items that add to the tile, so a method has more than one value to choose from only where items overlap without covering one another. Pass `skipcovered=false&exitwhenfull=false` and an `items_limit` to have every item under the tile taken into account, as with titiler-pgstac. `median` holds all the scenes at once and is refused with a `400` beyond 256 MB of them (about a thousand single-band scenes of a 256 pixel tile); the others keep a running figure and have no such limit.

The mean or median of an expression is kept as a fraction; rio-tiler casts it to a whole number when the expression itself gave whole numbers (`b1+b2` on integer data). The rasters these routes were checked on carry no scale or offset, so `unscale` together with a pixel selection is untested beyond doing no harm.

### Tiles from planes

The renderer draws a tile in one pass over the bands of one raster, or over the first band of up to three rasters that lie on one grid. A tile that asks for more is made from planes instead: every asset of every item is read on its own grid onto the tile's, its `bidx` or `expression` applied, the assets stacked in the order given, the request's `expression` worked out over the stack in double precision, and the items put together by the pixel selection. That covers assets on different grids (Sentinel-2's 10 m and 20 m bands together), more than three assets, two assets without band math, options on any asset, expressions with several results, and the last-band methods. Nothing has to be asked for: the route takes this road when the request or the rasters need it.

The bands that come out are encoded as TiTiler encodes them. One band goes through `rescale` and the colour map; three are red, green and blue, each with its own `rescale` when three are given; two come out as red and green with the mask for blue, which is what GDAL makes of two bands and a mask, and are refused as JPEG; four or more are refused with TiTiler's own message (TiTiler does send four as a CMYK JPEG; that is refused here too), after the first item and before any other is read. With no format asked for, the mask decides between JPEG and PNG, as on other tiles.

What a request can make the server hold is bounded: an expression is at most 4096 bytes with 16 results and 64 levels of nesting, on these tiles and on `point` and `statistics`; an asset gives a tile at most 32 bands and an item's assets 32 together; and `statistics` are worked out over at most four times the pixel limit in bands times pixels, beyond which `max_size` is asked for.

Warm, the oracle's 21 such tiles took a median of 8 ms. Cold from S3 over a home connection, a tile of six items and three assets each took 2.5 to 5.7 s over six tiles, against 1.5 to 5.0 s for titiler-pgstac on the same tiles: slower in five of the six, by 10 to 70%. Most of that time is the line, not the reading: one such tile was 48 range requests and 58 MB, because a 20 m band is read at full resolution in blocks of 1024 pixels, and it arrived at about 10 MB a second. One raster on its own, cold, took a median of 317 ms here against 866 ms over twelve different granules.

### How pictures are read

Tiles and images with nearest-neighbour resampling, which is the default, are read from the overview GDAL's warp would take for the size asked, each output pixel taking the source pixel its centre falls in. Where that centre falls is the projection's own answer. GDAL draws the positions along a row between a few exact ones, to within an eighth of a source pixel, and so takes the neighbouring pixel for about one output pixel in twenty (4.7% over the tiles and images measured). Here a row is drawn as a curve through three exact positions, not a line through two, checked to within 1/1024 of a pixel, and every position that close to a source pixel's edge is worked out exactly. A row that jumps or leaves the projection where it is checked is not drawn across but worked out column by column. That gives the same picture as projecting every pixel (none of 2.5 million differed) and takes no longer than GDAL's approximation did here: 3 to 6 ms a tile either way, where projecting every pixel takes 9 to 32. No picture is therefore titiler-pgstac's to the last bit any more: against it, PNG tiles of single rasters agree in 94 to 97% of pixels, and the oracle's pictures stand between 26 and 70 dB, 39 dB at the middle. With GDAL's approximation 83 of 101 were identical. Statistics still follow GDAL's approximation, so that their numbers are TiTiler's.

GDAL is followed where it decides what the picture is, and not where it is only inconsistent:

- One band of several is read from the same pixels as all of them. GDAL takes another road for a partial selection and returns other pixels.
- The edge of a `feature`'s polygon is where pixel centres fall inside it; GDAL cuts in the source raster's pixels. 99.0 to 99.6% of pixels agree.
- Four bands are refused as JPEG; TiTiler sends them as CMYK.
- The projection along a row is not approximated to an eighth of a pixel as GDAL does: each pixel is the one the projection gives.
- Still as GDAL has it, though it need not be: an overview up to a fifth coarser than asked is preferred to the next finer one.

Warm, a tile costs what it cost through the renderer: 1.2 ms for one band (1.1), 2.2 ms for `visual` (2.4), 4.7 ms for three assets over six items (4.5), where titiler-pgstac takes 33 to 530 ms. The per-pixel work of a mosaic's rasters is done side by side on the pool for heavy work; one raster of a tile is done in place.

The renderer still draws: the other resampling methods, `algorithm`, raw arrays, rasters with an alpha band, tile sizes other than 256 and 512, and band math only its own engine reads (`ndvi`, `^`, `max(a, b)`).

### Images

A `bbox` or `feature` image is made from planes as well, whatever it asks for, as long as `resampling` and `reproject` are left at nearest and there is no `algorithm`. Each raster is read the way GDAL's nearest-neighbour warp reads it (the reader `statistics` uses): at the size its own resolution gives for what was asked, from the overview GDAL would take, each pixel from the source pixel its centre falls in, and then brought to the picture's size. That is what rio-tiler does, and the pixels are TiTiler's: of the oracle's 39 PNG images of mosaics, 31 are identical to titiler-pgstac's to the last bit, six differ by a level here and there (band math and colour formulas, 51 dB and up), one is the bilinear case the shared renderer draws (29 dB), and one is two bands under a polygon (28 dB), where the mask is the third band and its edge is not GDAL's cutline: a pixel is inside when its centre is, which agrees for 99.0 to 99.6% of pixels.

Warm, such an image takes about what the shared renderer took and far less than titiler-pgstac: 3 ms against 24 ms at 256 pixels, 18 against 646 at 1024, 51 against 1190 at the data's own 1802 by 901, and 376 ms against 5.4 s for 4096 by 2048.

An image's planes can be large, so more is bounded than on a tile: the first item is read alone and the number read together afterwards follows from what it took; an item's assets may be read into at most 64 million values (bands times pixels) between them; and a pixel selection that would work in more than 512 MB asks for `max_size`.

### Item routes

`/collections/{collection_id}/items/{item_id}/...` is the [`/stac`](#stac) router over an item read from pgSTAC with `get_item` instead of from `url`: `assets`, `info`, `info.geojson`, `statistics`, `asset_statistics`, `tiles/{tms}/{z}/{x}/{y}`, `{tms}/tilejson.json`, `{tms}/map.html`, `point/{lon},{lat}`, `preview`, `bbox/...` and `POST feature` and `statistics`, plus `tiles`, `tiles/{tms}`, `WMTSCapabilities.xml` and `renders`. On top of what `/stac` does, these routes follow titiler-pgstac:

- `assets` is required on every route that reads pixels, also with `expression`, and has to name an asset of the item; `GET .../assets` lists the assets in rio-tiler's default media types.
- An unknown item is a `404`, a tile that does not touch the item is a `404`, and errors are JSON `{"detail": ...}`.
- The item's bounds and zoom range come from its projection metadata (`proj:shape`, `proj:transform`, `proj:epsg` or `proj:code`) by rio-tiler's rule, or from its bbox when it has none. They are used for the tilesets, the TileJSON and `info.geojson`.
- A tile with no extension is a JPEG when every pixel is valid and a PNG otherwise, and `.jpg` is served as `image/jpg`.
- A tile or a `bbox` image whose assets carry band math (`assets=red|expression=b1*2`), or bands of their own on more than one asset (`assets=visual|bidx=1&assets=red|bidx=1`), is drawn by the code that draws a search's, from this one item, and is TiTiler's picture where that code draws it: what the search routes refuse (`nodata`, `buffer`, `tilesize` other than 256 and 512, `unscale` with several assets, `.npy`, `coord_crs`) is refused here too, where titiler-pgstac draws it. `pixel_selection` and the search limits are passed over, as titiler-pgstac passes them over on an item. `preview` and `POST feature` with either are still a `400`.

Known differences from titiler-pgstac on these routes, inherited from the shared `/stac` code:

- `preview`: the picture does not fill `max_size` or the size asked for the way TiTiler's does, so the two images are offset.
- `statistics` and `asset_statistics`: the sample is smaller than `max_size` (115 by 115 where TiTiler reads 128 by 128), so `max`, `minority` and the other sample-sensitive numbers differ, and `statistics` describes a band as `b1` where TiTiler says `visual_b1`.
- A point outside the item is a `404`; titiler-pgstac answers `500`.
- `renders` and `renders/{id}` list the item's STAC renders the way TiTiler does, including its `url=<Item id=...>` in the links, and each render that names assets is a layer in the item's WMTS document. `point` takes `expression` over `b1..bn` like the mosaic routes.

### Not there yet

Each of these is answered with a `400` or `404` rather than a different picture:

- On mosaic `statistics`: `algorithm`, `nodata`, and `coord_crs` and `dst_crs` other than `epsg:4326`. With `resampling` or `reproject` other than nearest the shared renderer does the reading, and the numbers are then close to TiTiler's rather than equal.
- Functions other than the common numeric ones (`where` in particular) in `point` and `statistics` expressions, and on the `bbox` and `feature` routes `coord_crs` and `dst_crs` other than `epsg:4326`, the `npy` format and images larger than 4096 pixels on a side.
- With `resampling` or `reproject` other than nearest, a `bbox` or `feature` image is drawn by the shared bbox renderer: close to TiTiler's rather than equal, and at the data's own resolution a pixel is left empty when any band is zero in a source whose nodata is zero, where TiTiler does so only when every band is.
- In WMTS documents, the twelve tile matrix sets other than `WebMercatorQuad`, and the `tif`, `tiff`, `jp2` and `pngraw` tile formats.
- In CQL2 text filters: WKT coordinates written with an exponent, and the handful of malformed geometries and calls titiler-pgstac's converter lets through as function calls (pgSTAC then rejects them with a `500`; here they are a `422`).
- `coord_crs` other than `epsg:4326` on the `bbox` and `point` routes, and `f=html` on the tileset routes.
- In the tileset list, the twelve tile matrix sets other than `WebMercatorQuad` that titiler-pgstac also lists.
- A tile or image of one or two bands out of a three-band raster reads other pixels than TiTiler, on purpose: GDAL warps a partial band selection in blocks through its overview grid, so that a band on its own is not the band it is in the whole raster (43% of pixels differ on the reference data). Here a band is the same band however it is asked for. `statistics` still follows GDAL there, to give TiTiler's numbers. Bad band indexes and `asset_as_band` on a multi-band asset are a `400` where TiTiler answers `500`.
- On tiles and images made from planes (see above): `algorithm`, raw array formats, and four bands as JPEG, which TiTiler sends as CMYK. Any `pixel_selection` other than `first` with a terrain encoding or a raw array format.
- `buffer`, `nodata` and a JSON `colormap`.
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
