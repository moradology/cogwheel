# Running and configuring `cogzilla`

## Building and running

```bash
cargo build --release -p cogzilla-server

./target/release/cogzilla-server \
  --bind 127.0.0.1:8080 \
  --cache-mb 1024 \
  --dataset dem=/path/to/dem.tif \
  --scene landsat=/path/to/SR_B4.TIF,/path/to/SR_B3.TIF,/path/to/SR_B2.TIF \
  --scene sar=/path/to/OPERA_VV.tif,/path/to/OPERA_VH.tif
```

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--bind` | `127.0.0.1:8080` | Listen address. The Docker image fixes this to `0.0.0.0:8080`; remap the port with `-p`. |
| `--cache-mb` | `512` | Decoded-block cache size in MiB (tier 1, RAM). Size it to your working set (see [the cache note](benchmarks.md#what-these-numbers-do-and-dont-say)). |
| `--tile-cache-mb` | `0` | Optional encoded-tile LRU micro-cache in MiB (`0` keeps request-level singleflight deduplication enabled with no retention). |
| `--disk-cache-dir` | | Directory for persistent compressed-block cache (tier 2, local disk/NVMe). |
| `--disk-cache-mb` | `0` | Maximum tier-2 disk cache size in MiB (`0` disables tier 2). |
| `--dataset name=path` | | Pre-register a local COG, served at `/tiles/{name}/{z}/{x}/{y}`. Repeatable. |
| `--scene name=a.tif,b.tif[,c.tif]` | | Pre-register a multi-file scene (for example Landsat RGB or SAR VV+VH). Repeatable. |
| `--mosaic name=/path/to/catalog.json` | | Pre-register a spatial mosaic catalog (MosaicJSON, STAC `FeatureCollection`, or JSON array), served at `/mosaic/{name}/tiles/{tms}/{z}/{x}/{y}`. Repeatable. |

`--dataset` and `--scene` are optional: the `/cog` and `/stac` routes open any `url=` on demand.

## Configuration

S3 and HTTP behavior is set through the same environment variables GDAL and the AWS SDKs use, so an existing TiTiler deployment's settings mostly carry over.

| Variable | Purpose |
| :--- | :--- |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | Static or temporary credentials. |
| `AWS_PROFILE`, `AWS_SHARED_CREDENTIALS_FILE` | Credentials from a shared credentials file. |
| (none) | ECS task roles and EC2 instance roles (IMDSv2) are picked up automatically. Set `AWS_EC2_METADATA_DISABLED=true` to skip the instance lookup. |
| `AWS_NO_SIGN_REQUEST=YES` | Send unsigned requests, for public buckets. |
| `AWS_REQUEST_PAYER=requester` | Access requester-pays buckets. |
| `AWS_REGION`, `AWS_DEFAULT_REGION` | Default region. A bucket in another region is discovered from S3's redirect and remembered. |
| `AWS_ENDPOINT_URL`, `AWS_S3_ENDPOINT`, `AWS_HTTPS`, `AWS_VIRTUAL_HOSTING=FALSE` | S3-compatible stores (MinIO, Ceph, and so on) and path-style addressing. |
| `GDAL_HTTP_MAX_RETRY` (or `AWS_MAX_ATTEMPTS`) | Retries for transient failures. Default 3. |
| `GDAL_HTTP_CONNECTTIMEOUT`, `GDAL_HTTP_TIMEOUT` | Connect timeout (default 10 s) and per-read timeout (default 30 s). Since 0.1.2. |
| `COGZILLA_MAX_UPSTREAM_FETCHES` | Maximum concurrent S3/GCS/Azure/HTTP range requests (default 64). Backs off automatically on `429`/`503`/timeouts (AIMD, floor 4) and recovers on clean responses. |
| `COGZILLA_ALLOWED_URL_PREFIXES` | Comma-separated prefixes that a `url=` (or STAC item, MosaicJSON document, mosaic scene, or `stac_api_url`) must start with, for example `https://copernicus-dem-30m.s3.amazonaws.com/,s3://my-bucket/public/`. Anything else, including local paths, gets `403`. Unset by default, which allows every location the process can reach: **set it on any server exposed to the public internet.** Startup `--dataset` and `--scene` files are exempt. Redirects from an allowed host are still followed. |
| `COGZILLA_MAX_OPEN_URIS` | Maximum cached opened-COG headers in memory (default 4,096, LRU-evicted). |
| `COGZILLA_CACHE_CONTROL_MAX_AGE` | `Cache-Control: public, max-age=<N>` seconds on tile responses (default `3600`; `0` emits `no-cache`). All tile responses also emit strong `ETag` and honor `If-None-Match` (`304 Not Modified`). |
| `GS_OAUTH2_ACCESS_TOKEN`, `GS_NO_SIGN_REQUEST`, `GS_USER_PROJECT`, `GS_ENDPOINT_URL` | Google Cloud Storage (`gs://`, `/vsigs/`) auth (falls back to GCE/GKE/Cloud Run metadata server) and requester-pays project. |
| `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_SAS_TOKEN`, `AZURE_STORAGE_ACCESS_KEY`, `AZURE_STORAGE_ACCESS_TOKEN`, `AZURE_NO_SIGN_REQUEST` | Azure Blob (`az://`, `abfs[s]://`, `/vsiaz/`) SAS token, Bearer token, or SharedKey HMAC-SHA256 signing. |
| `RUST_LOG` | Log level, for example `info` (the default) or `debug`. |

### pgSTAC database

The `/searches` and `/collections` routes need a [pgSTAC](https://github.com/stac-utils/pgstac) database. They are off, and answer `503`, until one is configured with the same variables titiler-pgstac 2 and later read. No connection is opened until the first request that needs one, so a database that is down does not stop the server from starting or from serving `/cog` and `/stac`.

| Variable | Purpose |
| :--- | :--- |
| `DATABASE_URL` | A `postgresql://user:password@host:port/dbname` URL. Takes precedence over the variables below. |
| `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` | The libpq variables. pgSTAC support turns on when `PGHOST` or `PGDATABASE` is set. |
| `DATABASE_URL` parameters | The URL's query, or a libpq `keyword=value` string, may carry `sslmode`, `sslrootcert`, `options`, `application_name`, `connect_timeout`, `channel_binding`, `target_session_attrs` and the keepalive settings. `options` are kept and the search path is added after them. Client certificates (`sslcert`, `sslkey`) and the other libpq parameters are not supported and are refused by name. |
| `PGSSLMODE` (or `sslmode=` in the URL) | `disable`, `prefer` (the default), `require`, `verify-ca` and `verify-full`, with libpq's meanings. `prefer` and `require` encrypt without checking the server's certificate, except that `require` checks the certificate's chain once a root certificate file is configured. `verify-ca` checks the chain; `verify-full` checks the chain and that the certificate is for the host being connected to. |
| `PGSSLROOTCERT` (or `sslrootcert=` in the URL) | PEM file of certificate authorities the chain is checked against. Without it `verify-ca` and `verify-full` use the public web roots, which a managed database's private authority is usually not in. `system` means the public web roots and needs `verify-full`. |
| `DB_MAX_CONN_SIZE` | Largest number of open connections. Default 10. |
| `PGSTAC_SEARCH_PATH` | Schemas searched for pgSTAC's functions and tables. Default `pgstac,public`. |
| `TITILER_PGSTAC_CACHE_TTL`, `TITILER_PGSTAC_CACHE_MAXSIZE`, `TITILER_PGSTAC_CACHE_DISABLE` | The per-process cache of item lists and search lookups, with titiler-pgstac's names and defaults (300 seconds, 512 entries, enabled). |
| `COGZILLA_PGSTAC_LEAN` | How the items of a tile or image are listed. `auto` (the default) uses the [lean lookup](api.md#the-lean-item-lookup) on the pgSTAC release lines it has been run on (0.9 and 0.10) and pgSTAC's `geojsonsearch` on any other; `on` uses the lean lookup on every version; `off` always uses `geojsonsearch`. A request can choose for itself with `item_lookup=lean` or `item_lookup=geojsonsearch`. |

The database role needs to read the `pgstac` schema and execute its functions. `POST /searches/register` and the `/collections/{id}` mosaic routes store a row in `pgstac.searches`, and pgSTAC's own search function updates that table's use counters on every tile, so a role without write access to it needs pgSTAC's read-only mode switched on. The lean lookup reads the `items` table directly and calls that search function once per search every cache period, not once per tile. Connections are opened with `max_parallel_workers_per_gather=0`: every query here is a short index lookup, and starting parallel workers costs more than the query.

Operational endpoints: `GET /` for the landing page with demo maps, `GET /healthz` for liveness, `GET /stats` for JSON counters, `GET /metrics` for Prometheus exposition, `POST /mosaic/register` for dynamic catalog / STAC-API search registration, and `GET /docs` with `GET /openapi.json` for the route list.
