# Benchmarks

The headline numbers from the README, with where each came from and what it does and does not show. Full method and per-workload tables are in [benches/S3_BENCHMARK.md](../benches/S3_BENCHMARK.md) (real COGs on S3) and [benches/COGMARK.md](../benches/COGMARK.md) (local suite).

## Real COGs on S3

Measured against real COGs on S3 (Sentinel-2, USGS 3DEP, Copernicus DEM), with TiTiler tuned per its own guidance and given one worker per core. This is one run of `cogzilla` v0.1.0 on an Apple M5 over a home network, on 2026-11-22:

| Real S3 COGs, equal mix of five workloads | `cogzilla` | TiTiler | Gain |
| :--- | ---: | ---: | ---: |
| CPU per 1,000 tiles, never seen before | **13.2 CPU-s** | 56.0 CPU-s | **4.3×** |
| CPU per 1,000 tiles, source blocks cached | **1.7 CPU-s** | 55.3 CPU-s | **32×** |
| Tiles per second per busy core, cached | **588** | 18 | **32×** |
| Memory at idle | **4 MiB** | 1,112 MiB (10 workers) | **278×** |
| Instances for 1,000 uncached tiles/s (modeled, `c7g.xlarge`) | **11** | 47 | **4.3×** |

## Local fixtures, warm cache

The ceiling, with the network taken out entirely (local files, warm cache), on two machines:

| Local fixtures, warm cache | Apple M5, 10 cores | Linux x86-64, 4 vCPU |
| :--- | ---: | ---: |
| HTTP throughput, Mercator PNG, 32 concurrent | 13,195 vs 180 tiles/s: **73.4×** | 3,363 vs 85 tiles/s: **39.8×** |
| HTTP throughput, satellite RGB PNG, 32 concurrent | 5,728 vs 142 tiles/s: **40.2×** | 1,369 vs 62 tiles/s: **22.1×** |
| Single Mercator PNG tile | 303 µs vs 16.6 ms: **54.7×** | 632 µs vs 21.4 ms: **33.9×** |
| Landsat three-file RGB PNG tile | 1.36 ms vs 52.3 ms: **38.6×** | 2.62 ms vs 62.6 ms: **23.9×** |

Each cell is `cogzilla` against TiTiler on the same machine. The two columns differ in more than hardware:

- **Linux** is a single run on a GitHub-hosted shared runner (4 vCPU AMD EPYC 9V45) at commit `56a6500`, with the same kind of build as the release binaries and both servers measured back to back.
- **M5** is a single run at commit `7282f7c` with a `target-cpu=native` build, TiTiler given 10 workers, and both servers measured back to back. The laptop was not idle during the run, so expect a few percent of movement.

## What these numbers do and don't say

- **The 4.3× is the number to plan with.** It is CPU for a tile the server has not seen, which is what an origin behind a CDN mostly serves. It ranged from 2.1× to 4.8× on four workloads (the fifth showed 8.3×, but its file was small enough to be partly cached) and does not depend on the network.
- **The local multiples depend on the machine.** Warm PNG tiles are 39–75× on the M5 and 24–43× on the 4-vCPU Linux runner; HTTP throughput is 40–73× and 22–40×. The Linux figures are one run on a hosted shared runner (`generic` release build); GitHub assigns a different CPU model from run to run, and TiTiler's own time for the same tile has varied between 21 ms and 31 ms across runs. Quote the Linux range if you need one number for a typical cloud instance.
- **Not every path is that far ahead.** On the Linux x86-64 run (`45302863059`), lossy WebP output is **10.9×** faster warm (**7.0×** cold) vs GDAL, and first-touch (cold) local PNG tiles are 8.4–21.5× faster (low-zoom multi-file scenes are 23–32× faster with adaptive quadtree projection). One raw decode path remains slightly slower than GDAL on x86-64 (LZW with predictor 1 at `0.9×`; DEFLATE with predictor 2 is 1.1–1.2× and LZW with predictor 3 is 1.4×). Those are decode-only measurements, and a warm tile does no decoding.
- **The S3 results are M5-only and from v0.1.0.** The CPU, memory and cost figures in the first table have not been repeated on x86, and have not been re-run since the rendering optimizations in 0.2.0.
- **The larger multiples (13–62× on S3) need the cache to fit.** `cogzilla` keeps decoded blocks in memory and TiTiler keeps none, which is where the big multiples come from. With the default 512 MiB cache, three of the five S3 workloads outgrew it and fell back to S3. Size `--cache-mb` to your working set and budget about 250 MiB on top.
- **No uncached latency win has been measured.** The S3 run was made from a home network, where both servers were limited by bandwidth and finished roughly level. Lower CPU should become lower latency in-region, but that run has not been done.
- **The cost row is a model, not a bill.** It assumes a Graviton vCPU does half the work of an M5 core, 60% target utilization, and compute only. The ratio is measured; the instance counts are not.
- **Same picture, not the same bytes.** Valid-data masks agree with TiTiler on 99.6–100% of pixels; pixel values differ slightly on high-frequency imagery (32–34 dB PSNR on Sentinel-2, 49–54 dB on DEMs).
- **One request in 822 failed.** A single uncached tile returned HTTP 500 during the S3 run, most likely an S3 connection failure that was not retried. Retries and timeouts for that case were added in 0.1.2; the benchmark has not been re-run since.

Method, per-workload tables and caveats are in [benches/S3_BENCHMARK.md](../benches/S3_BENCHMARK.md); the full local suite is in [benches/COGMARK.md](../benches/COGMARK.md).

## Terrain rendering throughput

Serving on-the-fly hillshade tiles (cubic resampling, Horn 3×3, PNG encode per request, encoded-tile cache off) across a whole Copernicus 30 m DEM tile, in one `oha` run against a local build from 2026-11-23 (323 distinct tiles, `z=9..12`, 32 connections, 10,000 requests, Apple M5, load generator on the same machine, source blocks already in the block cache so no S3 requests were made):

| Tiles per second | Median | p99 | Errors |
| ---: | ---: | ---: | ---: |
| **3,372** | 9.3 ms | 19.5 ms | 0 |

This is `cogzilla` alone: TiTiler was not run on this workload, so there is no multiple to quote.

## Live TiTiler 2.4 differential oracle (endpoint latency & parity)

Single-request warm loopback HTTP measurements from [benches/titiler_diff_oracle.py](../benches/titiler_diff_oracle.py) against a live TiTiler 2.4 instance on Apple M5 at commit `2bc38e1` (**39/39 checks passed**, local fixtures, no network):

| Endpoint | `cogzilla` | TiTiler 2.4 | Gain | Parity |
| :--- | ---: | ---: | ---: | :--- |
| `GET /cog/tiles/WebMercatorQuad/...` (3-band UTM RGB PNG) | **1.17 ms** | 50.30 ms | **43.0×** | `mask=100%`, `PSNR=54.0 dB` |
| `GET /cog/tiles/...?expression=(b1-b2)/(b1+b2)` | **0.86 ms** | 32.51 ms | **38.0×** | `mask=100%`, `PSNR=53.5 dB` |
| `GET /stac/tiles/...?expression=(b1-b2)/(b1+b2)` (3-asset) | **0.96 ms** | 44.64 ms | **46.5×** | `mask=100%`, `PSNR=40.0 dB` |
| `GET /cog/tiles/...?color_formula=Gamma RGB 1.5` | **1.41 ms** | 49.11 ms | **34.7×** | `mask=100%`, `PSNR=49.8 dB` |
| `GET /cog/tiles/...?return_mask=false` | **1.26 ms** | 46.86 ms | **37.2×** | `mask=100%`, `PSNR=54.0 dB` |
| `GET /cog/tiles/WorldCRS84Quad/...` | **1.84 ms** | 37.12 ms | **20.2×** | `mask=100%`, `PSNR=55.1 dB` |
| `GET /cog/preview` / `/cog/bbox` / `POST /cog/feature` | **1.80–2.12 ms** | 21.1–28.0 ms | **11.7–14.5×** | `PSNR=44.2–52.5 dB` |
| `GET /cog/statistics?max_size=512` | **2.95 ms** | 54.00 ms | **18.3×** | `mean_err=0.00000` |
| `POST /cog/statistics` (GeoJSON polygon clip) | **1.49 ms** | 111.95 ms | **75.0×** | `mean_err=0.00003` |
| `GET /stac/point/{lon},{lat}` (3-asset) | **0.24 ms** | 18.00 ms | **75.6×** | `max_val_err=0.0000000` |

## Sources

- [docs/gpu-benchmarks.md](gpu-benchmarks.md): GPU acceleration & sustained realistic map-server soak benchmarks (RTX 3090 Ti Vulkan & Apple M5 Metal — staggered pan/zoom, 32-client tile flood, live tiles + `2048²` analytics contention, concurrency scaling, and `GpuCoalescer` telemetry).
- [benches/S3_BENCHMARK.md](../benches/S3_BENCHMARK.md): real COGs on S3 against a tuned TiTiler, covering time, memory and cost, with caveats and reproduction steps. This is the source of the S3 table above.
- [benches/COGMARK.md](../benches/COGMARK.md): the local `BM-01` to `BM-11` suite against GDAL, `rio-tiler` and TiTiler on synthetic fixtures, on an Apple M5 and on a Linux x86-64 runner, with reproduction steps. This is the source of the local table, and it is the best case: no network and a warm cache. The Linux numbers come from the manually triggered [COGMark workflow](../.github/workflows/bench.yml).
