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
| `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`, `AWS_CONTAINER_CREDENTIALS_FULL_URI`, `AWS_CONTAINER_AUTHORIZATION_TOKEN`, `AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE` | ECS task roles and EKS Pod Identity container credentials (reads `_FILE` when the plain token variable is unset). |
| `AWS_WEB_IDENTITY_TOKEN_FILE`, `AWS_ROLE_ARN`, `AWS_ROLE_SESSION_NAME`, `AWS_STS_REGIONAL_ENDPOINTS` | EKS IRSA web-identity credentials via STS `AssumeRoleWithWebIdentity`, cached to `Expiration` ahead of EC2 IMDSv2. |
| (none) | EC2 instance roles (IMDSv2) are picked up automatically when earlier providers are unset. Set `AWS_EC2_METADATA_DISABLED=true` to skip the instance lookup. |
| `AWS_NO_SIGN_REQUEST=YES` | Send unsigned requests, for public buckets. |
| `AWS_REQUEST_PAYER=requester` | Access requester-pays buckets. |
| `AWS_REGION`, `AWS_DEFAULT_REGION` | Default region. A bucket in another region is discovered from S3's redirect and remembered. |
| `AWS_ENDPOINT_URL`, `AWS_S3_ENDPOINT`, `AWS_HTTPS`, `AWS_VIRTUAL_HOSTING=FALSE` | S3-compatible stores (MinIO, Ceph, and so on) and path-style addressing. |
| `GDAL_HTTP_MAX_RETRY` (or `AWS_MAX_ATTEMPTS`) | Retries for transient failures. Default 3. |
| `GDAL_HTTP_CONNECTTIMEOUT`, `GDAL_HTTP_TIMEOUT` | Connect timeout (default 10 s) and per-read timeout (default 30 s). Since 0.1.2. |
| `COGZILLA_MAX_UPSTREAM_FETCHES` | Maximum concurrent S3/GCS/Azure/HTTP range requests (default 64). Backs off automatically on `429`/`503`/timeouts (AIMD, floor 4) and recovers on clean responses. |
| `COGZILLA_ALLOWED_URL_PREFIXES` | Comma-separated prefixes that a `url=` (or STAC item, MosaicJSON document, mosaic scene, or `stac_api_url`) must start with, for example `https://copernicus-dem-30m.s3.amazonaws.com/,s3://my-bucket/public/`. Anything else, including local paths, gets `403`. `file:///path` and `file://localhost/path` are normalized to `/path` before matching. Unset by default, which allows every location the process can reach: **set it on any server exposed to the public internet.** Startup `--dataset` and `--scene` files are exempt. HTTP redirects are checked against the allowlist on every hop, so a redirect from an allowed host to a disallowed target is refused with `403`. |
| `COGZILLA_ALLOWLIST_RESOLVE_SYMLINKS` | Set to `1` or `true` to canonicalize local paths (resolving symlinks) before checking `COGZILLA_ALLOWED_URL_PREFIXES`, refusing symlinks inside an allowed directory that point outside it (default off). |
| `COGZILLA_CACHE_TTL_SECS` | Remote source cache revalidation TTL in seconds (default `5`; `0` revalidates on every request). Within the TTL, warm reads make zero upstream requests; once elapsed, origins with `ETag`/`Last-Modified` are conditionally revalidated (`If-None-Match`/`If-Modified-Since` → `304`) and validator-less origins expire. Local files are always validated via `stat(len, mtime, inode)` on open. |
| `COGZILLA_MAX_STATISTICS_FEATURES` | Maximum features accepted in a `POST /*/statistics` `FeatureCollection` (default `10000`; `0` disables the count cap, while the 16 MiB body cap and per-feature window/block limits still apply). Features are evaluated in parallel. |
| `COGZILLA_BODY_TIMEOUT_SECS` | Maximum time in seconds to read a `POST` request body before aborting with `408 Request Timeout` (default `30`; `0` disables the timeout). Bodies exceeding 16 MiB are rejected with `413 Content Too Large`. |
| `COGZILLA_MAX_OPEN_URIS` | Maximum cached opened-COG headers in memory (default 4,096, LRU-evicted). |
| `COGZILLA_CACHE_CONTROL_MAX_AGE` | `Cache-Control: public, max-age=<N>` seconds on tile responses and on what a `GET` of a pgSTAC route finds, the list of searches apart (default `3600`; `0` emits `no-cache`). All tile responses also emit strong `ETag` and honor `If-None-Match` (`304 Not Modified`). |
| `GS_OAUTH2_ACCESS_TOKEN`, `GOOGLE_APPLICATION_CREDENTIALS`, `GS_ACCESS_KEY_ID`, `GS_SECRET_ACCESS_KEY`, `GS_NO_SIGN_REQUEST`, `GS_USER_PROJECT`, `GS_ENDPOINT_URL` | Google Cloud Storage (`gs://`, `/vsigs/`) auth: static OAuth2 token, service-account JSON key exchange (RS256 JWT cached to expiry), S3-compatible HMAC keys (SigV4), or GCE/GKE/Cloud Run metadata server fallback; plus requester-pays project. |
| `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_SAS_TOKEN`, `AZURE_STORAGE_ACCESS_KEY`, `AZURE_STORAGE_ACCESS_TOKEN`, `AZURE_CLIENT_ID`, `AZURE_NO_SIGN_REQUEST` | Azure Blob (`az://`, `abfs[s]://`, `/vsiaz/`) auth: portal connection string, SAS token, Bearer token, SharedKey HMAC-SHA256 signing, or Azure Managed Identity (IMDS / `MSI_ENDPOINT` / `IDENTITY_ENDPOINT`, cached to `expires_on`). Set `AZURE_IMDS_DISABLED=true` to skip the IMDS lookup. |
| `COGZILLA_GPU_POLICY` | GPU dispatch policy: `auto` (default, startup-calibrated crossover with silent CPU fallback if no hardware Metal/Vulkan adapter is found), `always` (prefer GPU when supported), or `never` (disable GPU completely). |
| `COGZILLA_GPU_INDEX` | Explicit 0-based hardware GPU adapter index (default `0`). |
| `COGZILLA_GPU_COALESCE_US`, `COGZILLA_GPU_RING_DEPTH`, `COGZILLA_GPU_MAX_QUEUED` | `GpuCoalescer` tuning: micro-batch coalesce window in µs, continuous ring-buffer depth (default 2 unified / 3 discrete), and max queued GPU jobs (default `4096`). |
| `COGZILLA_AIMD`, `COGZILLA_AIMD_TARGET_MS`, `COGZILLA_AIMD_MIN_LIMIT`, `COGZILLA_AIMD_MAX_LIMIT` | HTTP AIMD admission control (`429 Too Many Requests` overload shedding): set `COGZILLA_AIMD=off` to disable; target RTT default `150` ms, max concurrency ceiling default `4096`. Control-plane routes (`/healthz`, `/readyz`, `/metrics`, `/stats`, `/admin/clear-cache`, `/searches`, `/searches/list`, `/searches/register`) and `OPTIONS` bypass AIMD. |
| `COGZILLA_SHUTDOWN_TIMEOUT_SECS` | Maximum seconds to wait for in-flight HTTP connections to drain on `SIGINT`/`SIGTERM` before forcing exit (default `15`). |
| `RUST_LOG` | Log level, for example `info` (the default) or `cogzilla_server=debug` for per-request structured access logs (`request_id`, `method`, `path`, `status`, `dur_ms`). `5xx` failures are always logged at `warn`, and every response carries `X-Request-Id` (echoing a valid caller header or generating one). |

### pgSTAC database

The `/searches` and `/collections` routes need a [pgSTAC](https://github.com/stac-utils/pgstac) database. They are off, and answer `503`, until one is configured with the same variables titiler-pgstac 2 and later read. No connection is opened until the first request that needs one, so a database that is down does not stop the server from starting or from serving `/cog` and `/stac`.

| Variable | Purpose |
| :--- | :--- |
| `DATABASE_URL` | A `postgresql://user:password@host:port/dbname` URL. Takes precedence over the variables below. |
| `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` | The libpq variables. pgSTAC support turns on when `PGHOST` or `PGDATABASE` is set. |
| `DATABASE_URL` parameters | The URL's query, or a libpq `keyword=value` string, may carry `sslmode`, `sslrootcert`, `options`, `application_name`, `connect_timeout`, `channel_binding`, `target_session_attrs` and the keepalive settings. `options` are kept and the search path is added after them. Client certificates (`sslcert`, `sslkey`) and the other libpq parameters are not supported and are refused by name. |
| `PGSSLMODE` (or `sslmode=` in the URL) | `disable`, `prefer` (the default), `require`, `verify-ca` and `verify-full`, with libpq's meanings. `prefer` and `require` encrypt without checking the server's certificate, except that `require` checks the certificate's chain once a root certificate file is configured. `verify-ca` checks the chain; `verify-full` checks the chain and that the certificate is for the host being connected to. |
| `PGSSLROOTCERT` (or `sslrootcert=` in the URL) | PEM file of certificate authorities the chain is checked against. Without it `verify-full` uses the public web roots, which a managed database's private authority is usually not in, and `verify-ca` is refused: with no host name checked, those roots would vouch for any certificate holder. `system` means the public web roots and needs `verify-full`. |
| `DB_MAX_CONN_SIZE` | Largest number of open connections. Default 10. |
| `PGSTAC_SEARCH_PATH` | Schemas searched for pgSTAC's functions and tables. Default `pgstac,public`. |
| `TITILER_PGSTAC_CACHE_TTL`, `TITILER_PGSTAC_CACHE_MAXSIZE`, `TITILER_PGSTAC_CACHE_DISABLE` | The per-process cache of item lists and search lookups, with titiler-pgstac's names and defaults (300 seconds, 512 entries, enabled). |
| `COGZILLA_PGSTAC_LEAN` | How the items of a tile or image are listed. `auto` (the default) uses the [lean lookup](api.md#the-lean-item-lookup) on the pgSTAC release lines it has been run on (0.9 and 0.10) and pgSTAC's `geojsonsearch` on any other; `on` uses the lean lookup on every version; `off` always uses `geojsonsearch`. A request can choose for itself with `item_lookup=lean` or `item_lookup=geojsonsearch`. |

The database role needs to read the `pgstac` schema and execute its functions. `POST /searches/register` and the `/collections/{id}` mosaic routes store a row in `pgstac.searches`, and pgSTAC's own search function updates that table's use counters on every tile, so a role without write access to it needs pgSTAC's read-only mode switched on. The lean lookup reads the `items` table directly and calls that search function once per search every cache period, not once per tile. Connections are opened with `max_parallel_workers_per_gather=0`: every query here is a short index lookup, and starting parallel workers costs more than the query.

Operational endpoints: `GET /` for the landing page with demo maps, `GET /healthz` for liveness, `GET /readyz` for readiness (verifies pgSTAC connectivity when configured, returning `503` if unreachable), `GET /stats` for JSON counters (including GPU and pgSTAC pool/cache telemetry), `GET /metrics` for Prometheus exposition, `POST /admin/clear-cache` to synchronously flush open-URI, block, tile, GPU slab, STAC, MosaicJSON, and pgSTAC TTL caches, `POST /mosaic/register` for dynamic catalog / STAC-API search registration, and `GET /docs` with `GET /openapi.json` for the route list.
