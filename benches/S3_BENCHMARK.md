# cogzilla vs. TiTiler on real S3 COGs

Run of 2026-11-22, cogzilla `7b6bfee`, TiTiler 2.4.1 / rio-tiler 9.4.14 / GDAL (rasterio 1.5.2 wheel).
Benchmark machine: Apple M5 (10 cores), 16 GB, macOS, on a home network.

## Bottom line

| Dimension | Result | Confidence |
|---|---|---|
| **CPU per uncached tile** | cogzilla uses **4.3x less CPU** (13 vs 56 CPU-s per 1,000 tiles, equal mix; 2.1x to 4.8x per workload, excluding the tiny Copernicus file) | High. CPU time does not depend on the network. |
| **CPU per repeated tile** | cogzilla uses **~30x less CPU** (1.7 vs 55 CPU-s per 1,000) when its block cache holds the source blocks | High, but only applies when the cache is big enough (see Memory). |
| **Latency, uncached** | **No consistent difference from this machine.** Both are limited by pulling 1-3 MB per tile over a ~140 Mbit/s link | Network-bound. Says little about in-region behaviour. |
| **Latency and throughput, repeated tiles** | cogzilla: 1-3 ms p50, 2,700-7,900 req/s at c=8. TiTiler: 200-750 ms p50, 7-30 req/s, because it goes back to S3 every time | cogzilla CPU-bound; TiTiler network-bound here. In-region the gap shrinks to roughly the CPU ratio. |
| **Memory** | Idle: 4 MiB vs 1.1 GiB (10 workers). Under load cogzilla's footprint is its cache size plus ~200 MiB | Medium. macOS RSS under-reports (see caveats). |
| **Compute cost at 1,000 tiles/s** | Uncached: 11 vs 47 `c7g.xlarge` ($1,164 vs $4,975/month). Fully cached: 2 vs 47 | Ratio solid; absolute numbers rest on a stated core-speed assumption. |

The honest headline is **about 4x cheaper compute for tiles the server has not seen before**, rising to
**20-60x for tiles whose source blocks are already in cogzilla's cache**. The home-network latency numbers
do not show a cold-tile latency win.

## What was run

Five workloads against public, anonymous-access buckets. No credentials, no requester-pays.

| workload | source | request | zooms | tiles |
|---|---|---|--:|--:|
| `s2_b08_viridis` | Sentinel-2 L2A B08, uint16, DEFLATE predictor 2, 1024 px blocks, UTM 13N, `sentinel-cogs` (us-west-2) | `/cog/tiles/.../{z}/{x}/{y}.png?rescale=0,5000&colormap_name=viridis` | 9-14 | 89 |
| `s2_tci_jpg` | Sentinel-2 L2A TCI, 3-band uint8 pixel-interleaved, DEFLATE p2 | `/cog/tiles/....jpg` | 9-14 | 89 |
| `s2_stac_rgb` | Earth Search STAC item `S2A_13TEE_20240929_0_L2A`, assets red+green+blue (three COGs) | `/stac/tiles/....png?assets=red&assets=green&assets=blue&rescale=0,3000` | 9-14 | 89 |
| `dem_3dep_lzw` | USGS 3DEP 1/3 arc-second, float32, LZW predictor 3, 512 px blocks, EPSG:4269, `prd-tnm` (us-west-2) | `.png?rescale=1500,4300&colormap_name=terrain` | 9-14 | 80 |
| `dem_cop30_bilinear` | Copernicus GLO-30, float32, DEFLATE predictor 3, EPSG:4326, `copernicus-dem-30m` (eu-central-1) | `.png?rescale=1500,4300&colormap_name=terrain&resampling=bilinear` | 9-13 | 64 |

Tiles are a seeded random sample (up to 16 per zoom) inside each footprint. Both servers get byte-identical
query strings, with `s3://` URLs.

**cogzilla**: `target/release/cogzilla-server --bind 127.0.0.1:18811 --cache-mb 512` (default cache; a second
hot run used `--cache-mb 4096`). One process, default Tokio runtime.

**TiTiler**: `uvicorn app --workers 10 --loop uvloop --http httptools --no-access-log`, `TilerFactory` at `/cog`
and `MultiBaseTilerFactory(STACReader)` at `/stac`, one worker per core, with:

```
GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR   GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES
GDAL_HTTP_MULTIPLEX=YES                  GDAL_HTTP_VERSION=2
GDAL_INGESTED_BYTES_AT_OPEN=32768        GDAL_CACHEMAX=200
GDAL_BAND_BLOCK_CACHE=HASHSET            CPL_VSIL_CURL_CACHE_SIZE=200000000
CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.TIF,.tiff
VSI_CACHE=TRUE  VSI_CACHE_SIZE=5000000   PROJ_NETWORK=OFF
```

Both: `AWS_NO_SIGN_REQUEST=YES AWS_REGION=us-west-2`.

Three regimes, because the two servers cache very differently:

- **cold**: fresh process, one request alone, then a quarter of the tiles once each at concurrency 8.
- **miss**: same process after warm-up (connections, COG headers, every worker), then the other three
  quarters of the tiles, never requested before, once each at concurrency 8. This is what an origin behind
  a CDN sees and is the basis for the cost model.
- **hot**: the same tiles requested repeatedly with `oha` for 8 s at concurrency 1, 8 and 32.

Each cold/miss measurement was repeated twice with server order alternated; tables show the median.

### Output parity

Same request to both, PNG/JPEG decoded and compared. The outputs are the same picture but not
pixel-identical: 49-54 dB median PSNR on the DEMs, 32-34 dB on Sentinel-2 (nearest-neighbour sampling of
high-frequency imagery differs by a fraction of a source pixel). Valid-data masks agree on 99.6-100% of
pixels. Bytes fetched from S3 per tile are close for both servers, so they are reading the same overview
levels.

## Time

### Cold and miss (network-bound)

Latency in ms, concurrency 8.

| workload | server | cold: first request | cold p50 | cold p90 | miss p50 | miss p90 | miss p99 | miss tiles/s |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| `s2_b08_viridis` | cogzilla | 430 | 341 | 469 | 162 | 537 | 1,247 | 27.1 |
| | TiTiler | 613 | 238 | 1,352 | 326 | 801 | 1,437 | 15.7 |
| `s2_tci_jpg` | cogzilla | 475 | 358 | 1,315 | 309 | 908 | 1,699 | 17.1 |
| | TiTiler | 545 | 453 | 1,044 | 298 | 1,296 | 2,496 | 13.4 |
| `s2_stac_rgb` | cogzilla | 745 | 600 | 1,856 | 514 | 1,414 | 3,801 | 9.8 |
| | TiTiler | 1,061 | 1,072 | 2,201 | 1,091 | 2,818 | 4,062 | 4.6 |
| `dem_3dep_lzw` | cogzilla | 1,239 | 957 | 2,430 | 436 | 862 | 1,941 | 14.8 |
| | TiTiler | 747 | 418 | 1,134 | 435 | 1,075 | 1,451 | 13.8 |
| `dem_cop30_bilinear` | cogzilla | 1,562 | 1,163 | 3,703 | 6 | 1,314 | 2,542 | 13.6 |
| | TiTiler | 1,356 | 637 | 2,178 | 134 | 1,084 | 1,549 | 17.5 |

Read this table as "roughly equal, noisy". Each tile needs 0.8-3 MB of compressed blocks from S3 (the
sources use 1024 px blocks, so a 256 px tile pulls one to four of them), the link carries about 17 MB/s,
and round trip to us-west-2 is ~50 ms (~150 ms to eu-central-1). That caps either server at roughly 6-20
tiles/s regardless of how fast it decodes. cogzilla is ahead on the STAC composite (about 2x, it fetches
the three assets concurrently) and on the first request in three of five workloads; TiTiler is ahead on
the 3DEP and Copernicus cold passes. Two repetitions is not enough to call any of those differences.

One cogzilla request returned HTTP 500 during a Copernicus miss pass (1 of its 822 cold/miss requests; the
response body was not captured, the script now records it). TiTiler had no errors.

### Hot (repeated tiles)

`oha`, 8 s per step. "cores busy" is CPU-seconds divided by wall time, and tells you which rows are
CPU-bound.

| workload | c | server | req/s | p50 ms | p90 ms | p99 ms | cores busy |
|---|--:|---|--:|--:|--:|--:|--:|
| `s2_b08_viridis` | 1 | cogzilla | 892 | 1.1 | 1.4 | 1.5 | 0.93 |
| | 1 | TiTiler | 2.4 | 437 | 860 | 1,193 | 0.12 |
| | 8 | cogzilla | 4,297 | 1.7 | 2.7 | 3.7 | 6.73 |
| | 8 | TiTiler | 26.9 | 213 | 410 | 2,389 | 0.88 |
| | 32 | cogzilla | 4,726 | 6.5 | 10.0 | 14.5 | 8.22 |
| | 32 | TiTiler | 33.5 | 554 | 1,996 | 5,337 | 1.14 |
| `s2_tci_jpg` | 8 | cogzilla | 3,275 | 2.2 | 3.4 | 5.0 | 6.85 |
| | 8 | TiTiler | 20.5 | 308 | 713 | 1,964 | 0.54 |
| `s2_stac_rgb` | 8 | cogzilla | 2,662 | 2.9 | 4.0 | 5.5 | 7.54 |
| | 8 | TiTiler | 7.1 | 751 | 2,572 | 3,887 | 0.86 |
| `dem_3dep_lzw` | 8 | cogzilla | 6,067 | 1.2 | 1.8 | 2.4 | 6.79 |
| | 8 | TiTiler | 15.7 | 358 | 1,025 | 2,345 | 1.10 |
| `dem_cop30_bilinear` | 8 | cogzilla | 7,859 | 0.9 | 1.6 | 2.1 | 6.79 |
| | 8 | TiTiler | 29.5 | 24 | 967 | 2,507 | 0.78 |
| | 32 | cogzilla | 8,575 | 3.5 | 5.5 | 8.8 | 8.27 |
| | 32 | TiTiler | 219 | 39 | 91 | 2,579 | 6.56 |

cogzilla rows are from the `--cache-mb 4096` run. All steps returned 200 only. Full table for every
concurrency level is in `results/s3/<run>/report.md`.

Two things explain the 100-400x throughput gaps, and neither is "Rust is 100x faster than GDAL":

1. **TiTiler keeps COG headers between requests but not pixel data.** rio-tiler opens and closes the
   dataset per request, and GDAL's multi-range tile reads bypass its global curl cache. Verified with
   `CPL_DEBUG`: a repeated identical tile re-downloads the same byte ranges. So TiTiler's "hot" rows are
   still S3-bound from this machine (0.5-1.5 cores busy out of 10).
2. **The one row where TiTiler is CPU-bound is the fair hot comparison.** The Copernicus file is small
   and its single-range reads do land in GDAL's cache; at c=32 TiTiler reaches 219 req/s on 6.6 cores
   (33 req/s per core) against cogzilla's 8,575 on 8.3 cores (1,037 per core). That is about **31x per
   core**, consistent with the CPU-per-tile table below.

**With the default 512 MiB cache, cogzilla's hot numbers collapsed on three workloads.** The five tile
sets together need about 1.8 GiB of decoded blocks. In the first run (default cache), `s2_b08_viridis`
and `dem_cop30_bilinear` ran as above, but `s2_tci_jpg` and `s2_stac_rgb` thrashed and fell back to S3:
47 and 13.5 req/s at c=8 with p50 of 73 and 358 ms, still 2x TiTiler but nowhere near the cached figures.
`dem_3dep_lzw` thrashed at c=1 (1.5 req/s) and recovered at c=8. The hot-path win is conditional on cache
size relative to the working set.

## CPU per tile

CPU-seconds (user+system, whole process tree) per 1,000 tiles.

| regime | workload | cogzilla | TiTiler | ratio | cogzilla tiles/s per core | TiTiler tiles/s per core |
|---|---|--:|--:|--:|--:|--:|
| miss | `s2_b08_viridis` | 8.6 | 36.3 | 4.2x | 117 | 27.5 |
| miss | `s2_tci_jpg` | 14.4 | 30.8 | 2.1x | 70 | 32.4 |
| miss | `s2_stac_rgb` | 25.4 | 116 | 4.6x | 39 | 8.6 |
| miss | `dem_3dep_lzw` | 13.7 | 66.0 | 4.8x | 73 | 15.2 |
| miss | `dem_cop30_bilinear` | 3.8 | 31.1 | 8.3x | 267 | 32.2 |
| miss | **equal mix** | **13.2** | **56.0** | **4.3x** | **76** | **17.9** |
| hot (c=8) | `s2_b08_viridis` | 1.6 | 32.8 | 21x | 637 | 30.5 |
| hot (c=8) | `s2_tci_jpg` | 2.1 | 26.5 | 13x | 477 | 37.7 |
| hot (c=8) | `s2_stac_rgb` | 2.8 | 120 | 42x | 352 | 8.3 |
| hot (c=8) | `dem_3dep_lzw` | 1.1 | 70.0 | 62x | 890 | 14.3 |
| hot (c=8) | `dem_cop30_bilinear` | 0.9 | 26.6 | 31x | 1,155 | 37.6 |
| hot (c=8) | **equal mix** | **1.7** | **55.3** | **32x** | **588** | **18.1** |

"tiles/s per core" is 1,000 / CPU-s-per-1k: the most one fully busy core can sustain.

- TiTiler costs the same per tile in both regimes (it does the same work either way). cogzilla's miss
  cost is dominated by TLS, inflate/LZW and the PNG encode; its hot cost is resample plus encode only.
- The Copernicus miss row flatters cogzilla: the file is only 16 full-resolution blocks, so the warm-up
  set had already pulled most of it (0.2 GETs per tile, p50 6 ms). Without it the miss mix is 15.5 vs
  62.3, a ratio of 4.0x.
- Cold-phase CPU (fresh process) is higher for both and worse for TiTiler (62-162 CPU-s per 1k) because
  ten workers each pay import, PROJ and TLS start-up. That is start-up cost, not steady state, and is not
  used below.

## Memory

RSS summed over the process tree (TiTiler: master + 10 workers), sampled every 250 ms.

| | cogzilla, 512 MiB cache | cogzilla, 4096 MiB cache | TiTiler, 10 workers |
|---|--:|--:|--:|
| idle after start | 4 MiB | 4 MiB | 1,112 MiB |
| peak, cold/miss phase | 682 MiB | - | 1,689 MiB |
| peak, hot phase | 780 MiB | 1,052 MiB | 1,809 MiB |
| after load, idle 5 s | 244 MiB | 60 MiB | 1,200 MiB |

- TiTiler: about 111 MiB per worker idle, 181 MiB per worker at peak. It scales with worker count, not
  with data: four workers on a 4-vCPU box would be roughly 0.45-0.75 GiB.
- cogzilla: effectively zero baseline; the footprint is the decoded-block cache. **Do not trust the
  cogzilla RSS figures in the last two rows.** In the 4096 MiB run `/stats` reported 1.8 GiB of cached
  blocks while macOS reported 1.05 GiB peak RSS and 60 MiB afterwards, because macOS compresses idle
  anonymous pages and stops counting them. Budget `--cache-mb` plus roughly 200-300 MiB (the 512 MiB run
  peaked at 780 MiB).
- The cache holds decoded blocks: 2-4 MiB per 1024 px block against under 1 MiB compressed. About 410
  distinct tiles across five datasets needed 1.8 GiB. cogzilla trades memory for the hot-path CPU win;
  at equal memory to a small TiTiler deployment it caches a modest working set.

## Scaling and cost

Assumptions, all adjustable as flags to `s3_bench_report.py`:

- Instance: `c7g.xlarge`, 4 vCPU (Graviton3), 8 GiB, **$0.145/hour** on-demand in us-west-2, i.e.
  0.145 x 730 = **$105.85/month**. Price quoted from memory; check it before relying on the dollar figures.
- **Core factor 0.5**: one c7g vCPU is assumed to do half the work of one M5 core. This is a guess, applied
  equally to both servers, so it moves the dollar figures but not the ratio.
- Target average CPU utilisation 60%.
- Equal mix of the five workloads. Compute only: no load balancer, egress, CDN or S3 request charges.

```
capacity per instance = vCPUs x utilisation x core_factor / CPU-seconds-per-tile
  cogzilla, miss: 4 x 0.6 x 0.5 / 0.01317 =  91 tiles/s
  TiTiler,  miss: 4 x 0.6 x 0.5 / 0.05598 =  21 tiles/s
  cogzilla, hot:  4 x 0.6 x 0.5 / 0.00170 = 706 tiles/s
  TiTiler,  hot:  4 x 0.6 x 0.5 / 0.05527 =  22 tiles/s
instances = ceil(load / capacity);  monthly = instances x $105.85
```

| load | regime | cogzilla | TiTiler | instance ratio |
|---|---|--:|--:|--:|
| 100 tiles/s | uncached (miss) | 2 = $212 | 5 = $529 | 2.5x |
| 1,000 tiles/s | uncached (miss) | 11 = $1,164 | 47 = $4,975 | 4.3x |
| 5,000 tiles/s | uncached (miss) | 55 = $5,822 | 234 = $24,769 | 4.3x |
| 100 tiles/s | all cached (hot) | 1 = $106 | 5 = $529 | 5x |
| 1,000 tiles/s | all cached (hot) | 2 = $212 | 47 = $4,975 | 23.5x |
| 5,000 tiles/s | all cached (hot) | 8 = $847 | 231 = $24,451 | 29x |

A real deployment sits between the two rows: a CDN absorbs exact repeats, but neighbouring tiles, other
zooms over the same blocks, and the same tile with different `rescale`/`colormap` all hit cogzilla's block
cache and never hit TiTiler's. The uncached row is the conservative planning number.

Not modelled, and worth knowing:

- **Memory does not bind for either** on this instance at these numbers, but the "all cached" row
  assumes each cogzilla instance has enough cache for its working set, which 8 GiB may or may not cover.
- **S3 GET charges can exceed compute.** Both servers issue about 1-1.6 GETs per uncached single-COG tile
  (3-4 for the three-asset STAC tile). On a bucket you pay for, 1,000 uncached tiles/s x 1.4 GETs x
  2.63 M s/month x $0.0004 per 1,000 is about **$1,470/month**, more than cogzilla's compute at that load.
  cogzilla's block cache removes those GETs for repeated blocks; TiTiler pays them every time.
- At small loads the minimum instance count for availability dominates and the ratio compresses (see
  100 tiles/s).

## Caveats

- **Home network, not in-region.** Every cold, miss and TiTiler-hot latency and throughput figure is
  bound by ~50 ms round trips and ~17 MB/s of bandwidth. In-region, with ~10-30 ms first-byte times and
  multi-Gbit links, uncached latency would be roughly CPU time plus one or two S3 round trips: on the
  order of 20-60 ms for cogzilla and 60-150 ms for TiTiler on these workloads. That is an extrapolation,
  not a measurement. The CPU-per-tile figures and everything derived from them are network-independent.
- **The benchmark could not show an uncached latency or throughput advantage**, only a CPU one. Whether
  cogzilla's lower CPU turns into higher uncached throughput per box needs an in-region run.
- **Core mix.** The M5 has performance and efficiency cores; CPU-seconds at c=32 include slower cores
  (cogzilla's per-tile CPU rises 10-60% from c=1 to c=32). The cost model uses c=8.
- **Load generator on the same machine.** `oha` shares the 10 cores, so cogzilla's peak req/s is
  understated; TiTiler was never CPU-limited except on the Copernicus c=32 step.
- **macOS RSS** excludes compressed memory (see Memory). TiTiler's figures may be slightly understated
  too. Summed RSS double-counts shared library pages across workers; unique-set-size needs root on macOS
  and was not collected.
- **Small samples.** 64-89 tiles per workload, two cold/miss repetitions, 8 s hot steps, one scene per
  dataset. p99 from 60 requests is one or two requests.
- **TiTiler configuration** follows its published tuning advice with one worker per core. It was not
  given a tile or block cache because it does not ship one; a deployment with a CDN or an external cache
  in front narrows the hot-path gap. HTTP/2 multiplexing depends on the libcurl bundled in the rasterio
  wheel and was not verified.
- **cogzilla returned one HTTP 500** in 822 uncached requests. It did not reproduce on replay. The likely
  cause is a single S3 range request failing at the connection level, which cogzilla passes through
  without a retry; fault injection confirmed that gap and also showed a stalled S3 response hangs the
  request with no timeout. Neither is fixed.
- GDAL's per-tile S3 request counts were taken from its debug log in-process on every fourth tile with
  only headers warm; cogzilla's come from `/stats` during the miss pass with a partly warm block cache.
  They are comparable only roughly.

## Reproducing

```bash
cargo build --release -p cogzilla-server
cd benches && uv venv .venv && uv pip install --python .venv/bin/python \
    rasterio rio-tiler titiler.core uvicorn uvloop httptools httpx psutil mercantile pillow numpy && cd ..
brew install oha

benches/.venv/bin/python benches/s3_bench.py                      # ~25 min, ~4 GB of S3 downloads
benches/.venv/bin/python benches/s3_bench.py --phases hot --servers cogzilla --cache-mb 4096 --run-id hot4096
benches/.venv/bin/python benches/s3_bench_report.py <run-id> --cogzilla-hot-from hot4096
```

Raw output for this write-up: `benches/results/s3/20261122-224501/` and
`benches/results/s3/20261122-hot-cache4096/` (JSON per phase, RSS traces, server logs, `report.md`).
