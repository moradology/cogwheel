# Two-tier block cache: behavior

> **Status: Implemented.** The two-tier block cache (`--disk-cache-dir`, `--disk-cache-mb`), LRU RAM eviction, compact `u8`/`u16` slabs, and remote source revalidation (`COGZILLA_CACHE_TTL_SECS`) are implemented on `main` (see [configuration.md](configuration.md)). The original v0.1.2 design proposal is preserved below for historical context.

## Summary

Remote blocks would live in two places: decoded pixels in RAM (as now) and the original compressed
bytes on local disk (new). A block evicted from RAM is still on disk, so the next request decodes it
again locally instead of going back to S3. A block is only fully gone when the disk tier evicts it
too.

The disk tier is filled when a block is fetched, not when it is evicted from RAM. Eviction from RAM
writes nothing.

## Today

- One cache, in process memory, shared by all threads. `--cache-mb` caps it (default 512).
- It holds decoded blocks. Every band is stored as `float32` whatever the source type, so a 1024 px
  block costs 4 MiB per band.
- Concurrent requests for the same missing block collapse into one fetch; the others wait for it.
- When an insert pushes the cache over its cap, one thread removes entries until a quarter of the cap
  has been freed. In v0.1.2 entries are removed in hash-map iteration order, with no recency
  tracking, so a block used a millisecond ago is as likely to go as one untouched for an hour. On
  `main` this is now least-recently-used: each insert and hit stamps the entry, and eviction removes
  the oldest stamps first.
- A remote miss fetches compressed bytes by range request (adjacent ranges within 32 KiB are merged,
  up to 4 MiB per request), decodes them, and discards the compressed bytes.
- Local files are memory-mapped. Their compressed bytes come from the OS page cache and are never
  fetched over the network.
- Nothing checks whether a remote file has changed. An opened file's header and blocks are served
  from cache until evicted or the process restarts.
- The cache is lost on restart.

## Proposed

### Tiers

| | Tier 1: RAM | Tier 2: disk |
|---|---|---|
| Holds | Decoded blocks | Compressed byte ranges, exactly as fetched |
| Size per 1024 px block | 4 MiB per band today | Under 1 MiB typically |
| Cap | `--cache-mb` | `--disk-cache-mb` (0 or unset disables the tier) |
| Shared across threads | Yes | Yes |
| Survives restart | No | Yes |
| Applies to | Remote and local sources | Remote sources only |

### What happens on a request

For each block a tile needs:

1. **RAM hit.** Use the decoded block. No I/O, no decode.
2. **RAM miss, disk hit.** Read the compressed bytes from disk, decode, insert into RAM.
3. **RAM miss, disk miss.** Fetch from the source by range request, write the compressed bytes to
   disk, decode, insert into RAM.

Singleflight covers steps 2 and 3 as it covers fetches today: one request does the work and the rest
wait for its result.

The response does not wait for the disk write in step 3. The write is queued; if it fails or is
dropped, the only consequence is a disk miss later.

### Eviction

- **From RAM:** drop the decoded block. No write, because the compressed copy is already on disk (or
  was never eligible, for local files).
- **From disk:** delete the bytes. The next request for that block goes to the source.
- A block can be on disk without being in RAM. The reverse also happens: in RAM while its disk copy
  has been evicted or its write was dropped.

Both tiers evict by recency. RAM already does on `main` (see "Today").

### Restart

RAM starts empty. The disk tier is reused, so the first request for a previously seen block is a
disk hit. Entries that fail validation on first use are deleted and refetched.

### Source changes

A disk entry records the source's `ETag` (or `Last-Modified` and length) from the response that
produced it. When a file is opened after a restart, its header is fetched as usual and the validator
compared; on a mismatch every disk entry for that file is discarded. This is stricter than today's
in-memory behavior, which never revalidates, because a persistent cache can outlive a change by days.

### Failures

| Event | Behavior |
|---|---|
| Disk full or write error | Skip the write, count it, serve the request normally |
| Read error or checksum mismatch | Delete the entry, treat as a disk miss, fetch from source |
| Disk tier directory missing or unwritable at start | Log once, run with the tier disabled |
| Source fetch fails | Unchanged: retry per `GDAL_HTTP_MAX_RETRY`, then return an error |

The disk tier never makes a request fail that would have succeeded without it.

### Observability

`/stats` gains, alongside the existing counters: disk hits, disk misses, disk bytes used, disk
evictions, and write and read errors. Existing `cache_hits` and `cache_misses` keep meaning tier 1.

## Expected effect

Estimates from the v0.1.2 S3 benchmark, not measurements.

| Request | CPU per 1,000 tiles | Latency driver |
|---|---|---|
| RAM hit | 1.7 CPU-s (measured) | Resample and encode |
| Disk hit | Between 1.7 and 13.2 CPU-s | Local read and decompression |
| Source fetch | 13.2 CPU-s (measured) | Network round trips and transfer |

- Workloads larger than RAM stop falling back to S3 on every eviction. In the benchmark that
  fallback cut two workloads from about 3,000 requests per second to 47 and 13.5.
- Restarts and deploys no longer start cold.
- Repeat reads stop incurring S3 request charges.
- No change for tiles never requested before, or for peak throughput when everything is in RAM.

## Not in this proposal

- **Storing decoded blocks on disk.** They are several times larger than the compressed bytes and
  would have to be written at eviction time, when the server is already under memory pressure.
- **Sharing the disk tier between processes or hosts.** One process owns its directory.

## Zarr metadata caching & freshness (#387)

Unlike COGs (which are effectively immutable), Zarr stores are routinely appended to or rewritten in place:

- **One open-URI slot per store:** `COGZILLA_MAX_OPEN_URIS` (default `4,096`) bounds cached Zarr hierarchies by store prefix, not by chunk count (a 50,000-chunk store occupies one slot).
- **Local stores:** Validated on every open via `stat(len, mtime)` of the root metadata file (`zarr.json`, `.zmetadata`, `.zgroup`, or `.zarray`), so overwrites and appends are visible immediately and deleted stores return `404`.
- **Remote HTTP stores:** Cached under `COGZILLA_CACHE_TTL_SECS` (default `5` s). Within the TTL, warm reads issue zero metadata GETs; once elapsed, the server revalidates via `HEAD` (`ETag` / `Last-Modified` / `Content-Length`) and refreshes the hierarchy only when the validator changes.
- **Validator-backed ETags & `Cache-Control`:** `/zarr/*` raster ETags fold in the store's current metadata validator, so a client revalidating with `If-None-Match` after an append receives `200 OK` with fresh bytes rather than a stale `304`. `/zarr/*` inherits `COGZILLA_CACHE_CONTROL_MAX_AGE` (default `3600` s) unless overridden via `COGZILLA_CACHE_CONTROL_MAX_AGE_ZARR` (e.g. `60` s for actively appended stores).
- **Manual flush:** `POST /admin/clear-cache` synchronously clears the Zarr hierarchy cache alongside the COG, STAC, MosaicJSON, and pgSTAC caches.
- **Caching rendered tiles.** That is a CDN's job.

## Related, separate change

Caching RAM blocks in their source type instead of `float32` would hold about four times as much
`uint8` imagery and twice as much `uint16` in the same memory. It is independent of the disk tier
and smaller in scope.

## Open questions

- Read path: plain positioned reads on a blocking thread pool, or `mmap`. Reads are simpler to bound
  and to handle errors for; `mmap` faults would stall async worker threads.
- Granularity: one disk entry per block, or per merged range as fetched.
- How much of the measured 13.2 CPU-s is decompression. That decides how close a disk hit gets to a
  RAM hit, and should be measured before building this.
