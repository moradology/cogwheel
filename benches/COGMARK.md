# `COGMark`: local benchmarks vs. GDAL / `rio-tiler` / TiTiler

Local synthetic fixtures, no network. "Warm" means the source blocks are already decoded in `cogzilla`'s cache, so warm rows are the best case; for remote data and uncached tiles see [S3_BENCHMARK.md](S3_BENCHMARK.md).

Results are from two machines, at slightly different versions of `cogzilla` and with different TiTiler worker counts, so the two columns are not a like-for-like comparison of the hardware. Within a column, ratios compare `cogzilla` with GDAL, `rio-tiler` or TiTiler on the same machine; compare ratios, not absolute times.

| | Apple M5 | Linux x86-64 |
| :--- | :--- | :--- |
| Hardware | 10 cores, ARM64 NEON | GitHub-hosted runner, 4 vCPU AMD EPYC 9V45 (shared), AVX2 |
| Build | `rustc 1.95.0`, `target-cpu=native`, macOS 27.2 | `rustc 1.100.0`, default target CPU (as the release binaries) |
| Baseline | TiTiler 2.4.1, rio-tiler 9.4.14, rasterio 1.5.2 (GDAL 3.12.2), 10 workers | TiTiler 2.4.1, rio-tiler 9.4.14, rasterio 1.5.2 (GDAL 3.12.2), 4 workers |
| `cogzilla` version | commit `7282f7c` (2026-11-23, after v0.2.0) | commit `56a6500` (2026-11-23, shortly before v0.2.0) |
| Samples | one run, machine not idle | one run (`generic`) |

Three things to keep in mind when quoting these:

- **The Linux column is a single run on a shared runner.** GitHub assigns a different CPU from run to run: recent runs landed on an EPYC 9V74, a 7763 and a 9V45, and the TiTiler baseline alone moved between 21 ms and 31 ms for the same tile. Ratios within a run are meaningful; differences between runs smaller than about 1.5× are not.
- **The M5 column is also a single run, on a laptop that was doing other things.** Both sides were measured back to back, so the ratios hold, but expect a few percent of movement between runs. Sub-microsecond rows move more: the header open measured 1.2 µs in an earlier run and 2.0 µs in this one.
- **"Warm" is the best case.** A warm tile does no I/O and no decoding. The cold rows and the S3 benchmark are closer to what an origin serves.

## Summary

| Workload | M5 (`7282f7c`) | Linux x86-64 (`56a6500`) |
| :--- | ---: | ---: |
| HTTP Mercator PNG, 32 concurrent | 73.4× | 39.8× |
| HTTP satellite RGB PNG, 32 concurrent | 40.2× | 22.1× |
| Mercator PNG tile, warm | 54.7× | 33.9× |
| Reproject to `EPSG:4326` PNG, warm | 72.2× | 35.2× |
| Satellite RGB PNG, warm | 55.8× | 25.4× |
| Landsat three-file RGB PNG, warm | 38.6× | 23.9× |
| OPERA SAR RGB PNG, warm | 39.9× | 24.9× |
| Sparse coastline, average per tile | 46.3× | 28.7× |
| Remote cold open and tile, 20 ms RTT | 3.2× | 1.9× |
| COG header open | 195× | 844× |

Where `cogzilla` is not far ahead, or behind:

- **On the M5, uncompressed predictor-1 decode is slower than GDAL** (`0.7×`), and ZSTD and LZW with predictor 1 are close to a tie (`1.1×`). The remaining points are from the Linux run.
- **One raw decode path is slower than GDAL:** LZW with predictor 1 (`0.8×`). DEFLATE with predictor 2 is close to a tie (`1.1×`), and LZW with predictor 3 is `1.4×`. These are decode-only measurements.
- **Remote cold open** is 1.9×: with a 20 ms round trip, the gain is one fewer request, not faster pixels.
- **Lossy WebP output** is 4.6× warm and 3.9× cold, because the encoder dominates.
- **High-zoom tiles from the three-file Landsat scene** are 9.1–10.4× (`z=14`, `z=16`), against 22–32× at `z=5` to `z=12`.
- **First-touch (cold) local tiles** are 8.4–21.5×, roughly half the warm multiples.

## Apple M5

A single run at commit `7282f7c` on 2026-11-23, with `cogzilla` and the baseline measured back to back. The machine was not idle: load average was 2 to 3 of 10 cores before and after, from other processes. Rows with no baseline have no GDAL or TiTiler equivalent in the suite.

| ID | Workload | cogzilla | GDAL / TiTiler | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| `BM-01` | ColdHeader | 2.0 µs | 399.0 µs | 195.4× |
| `BM-02-NONE_P1` | UnshufflePred3 (NONE_P1) | 59.6 µs | 44.3 µs | 0.7× |
| `BM-02-DEFLATE_P1` | UnshufflePred3 (DEFLATE_P1) | 1.71 ms | 3.18 ms | 1.9× |
| `BM-02-DEFLATE_P2` | UnshufflePred3 (DEFLATE_P2) | 1.93 ms | 3.66 ms | 1.9× |
| `BM-02-DEFLATE_P3` | UnshufflePred3 (DEFLATE_P3) | 1.28 ms | 2.93 ms | 2.3× |
| `BM-02-ZSTD_P1` | UnshufflePred3 (ZSTD_P1) | 599.4 µs | 684.1 µs | 1.1× |
| `BM-02-ZSTD_P3` | UnshufflePred3 (ZSTD_P3) | 343.3 µs | 835.3 µs | 2.4× |
| `BM-02-LZW_P1` | UnshufflePred3 (LZW_P1) | 1.49 ms | 1.59 ms | 1.1× |
| `BM-02-LZW_P3` | UnshufflePred3 (LZW_P3) | 1.25 ms | 1.99 ms | 1.6× |
| `BM-03-PNG-cold` | NativeMercator PNG (cold) | 695.8 µs | 16.43 ms | 23.6× |
| `BM-03-PNG-warm` | NativeMercator PNG (warm) | 302.6 µs | 16.57 ms | 54.7× |
| `BM-03-WebP_Lossy82-cold` | NativeMercator WebP_Lossy82 (cold) | 2.47 ms | 9.47 ms | 3.8× |
| `BM-03-WebP_Lossy82-warm` | NativeMercator WebP_Lossy82 (warm) | 1.95 ms | 9.39 ms | 4.8× |
| `BM-03-WebP_Lossless-cold` | NativeMercator WebP_Lossless (cold) | 11.80 ms | — | — |
| `BM-03-WebP_Lossless-warm` | NativeMercator WebP_Lossless (warm) | 11.19 ms | — | — |
| `BM-03-RawF32-cold` | NativeMercator RawF32 (cold) | 556.8 µs | 6.91 ms | 12.4× |
| `BM-03-RawF32-warm` | NativeMercator RawF32 (warm) | 183.0 µs | 6.95 ms | 38.0× |
| `BM-04-cold` | Reproj4326 PNG (cold) | 1.64 ms | 22.52 ms | 13.8× |
| `BM-04-warm` | Reproj4326 PNG (warm) | 310.2 µs | 22.39 ms | 72.2× |
| `BM-05a-SatRGB-PNG-cold` | Satellite UTM10N RGB PNG (cold) | 2.60 ms | 46.75 ms | 18.0× |
| `BM-05a-SatRGB-PNG-warm` | Satellite UTM10N RGB PNG (warm) | 837.4 µs | 46.75 ms | 55.8× |
| `BM-05b-UTM-LZW-cold` | UTM10N LZW+Pred3 PNG (cold) | 1.65 ms | 24.48 ms | 14.9× |
| `BM-05b-UTM-LZW-warm` | UTM10N LZW+Pred3 PNG (warm) | 327.2 µs | 24.49 ms | 74.8× |
| `BM-05c-Albers-cold` | Albers 5070 PNG (cold) | 754.4 µs | 21.89 ms | 29.0× |
| `BM-05c-Albers-warm` | Albers 5070 PNG (warm) | 312.6 µs | 21.88 ms | 70.0× |
| `BM-05d-FusedNDVI-cold` | Satellite Fused NDVI PNG (cold) | 2.20 ms | — | — |
| `BM-05d-FusedNDVI-warm` | Satellite Fused NDVI PNG (warm) | 492.2 µs | — | — |
| `BM-06` | SparseCoastline (avg/tile) | 196.7 µs | 9.10 ms | 46.3× |
| `BM-07-HTTP-Mercator` | PanBurst32 HTTP Mercator PNG | 13,195 tiles/s | 180 tiles/s | 73.4× |
| `BM-07-HTTP-SatRGB` | PanBurst32 HTTP Satellite UTM10N RGB PNG | 5,728 tiles/s | 142 tiles/s | 40.2× |
| `BM-08` | RemoteRangeRTT (20ms RTT) | 46.33 ms | 149.27 ms | 3.2× |
| `BM-09-LandsatC2-cold` | Landsat C2 3-File uint16 RGB PNG (cold) | 3.68 ms | — | — |
| `BM-09-LandsatC2-warm` | Landsat C2 3-File uint16 RGB PNG (warm) | 1.36 ms | 52.34 ms | 38.6× |
| `BM-10-OperaSAR-RGB-cold` | OPERA RTC-S1 Dual-Pol SAR-RGB PNG (cold) | 3.32 ms | — | — |
| `BM-10-OperaSAR-RGB-warm` | OPERA RTC-S1 Dual-Pol SAR-RGB PNG (warm) | 1.17 ms | 46.90 ms | 39.9× |
| `BM-11-z5` | MultiZoom Landsat C2 RGB z=5 | 391.0 µs | 22.30 ms | 57.0× |
| `BM-11-z8` | MultiZoom Landsat C2 RGB z=8 | 519.8 µs | 19.80 ms | 38.1× |
| `BM-11-z10` | MultiZoom Landsat C2 RGB z=10 | 1.36 ms | 40.72 ms | 30.0× |
| `BM-11-z12` | MultiZoom Landsat C2 RGB z=12 | 1.29 ms | 54.20 ms | 41.9× |
| `BM-11-z14` | MultiZoom Landsat C2 RGB z=14 | 809.2 µs | 18.85 ms | 23.3× |
| `BM-11-z16` | MultiZoom Landsat C2 RGB z=16 | 757.8 µs | 11.87 ms | 15.7× |

## Linux x86-64

Full output of [run 45207296995](https://github.com/moradology/cogzilla/actions/runs/45207296995) (`generic` build, matching the release binaries), a single run at commit `56a6500`. Rows with no baseline have no GDAL or TiTiler equivalent in the suite.

| ID | Workload | cogzilla | GDAL / TiTiler | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| `BM-01` | ColdHeader | 0.9 µs | 769.0 µs | 844.1× |
| `BM-02-NONE_P1` | UnshufflePred3 (NONE_P1) | 31.0 µs | 58.6 µs | 1.9× |
| `BM-02-DEFLATE_P1` | UnshufflePred3 (DEFLATE_P1) | 2.16 ms | 2.18 ms | 1.0× |
| `BM-02-DEFLATE_P2` | UnshufflePred3 (DEFLATE_P2) | 2.51 ms | 2.67 ms | 1.1× |
| `BM-02-DEFLATE_P3` | UnshufflePred3 (DEFLATE_P3) | 1.47 ms | 2.44 ms | 1.7× |
| `BM-02-ZSTD_P1` | UnshufflePred3 (ZSTD_P1) | 555.8 µs | 606.8 µs | 1.1× |
| `BM-02-ZSTD_P3` | UnshufflePred3 (ZSTD_P3) | 365.2 µs | 1.29 ms | 3.5× |
| `BM-02-LZW_P1` | UnshufflePred3 (LZW_P1) | 2.18 ms | 1.77 ms | 0.8× |
| `BM-02-LZW_P3` | UnshufflePred3 (LZW_P3) | 1.80 ms | 2.60 ms | 1.4× |
| `BM-03-PNG-cold` | NativeMercator PNG (cold) | 1.28 ms | 21.43 ms | 16.7× |
| `BM-03-PNG-warm` | NativeMercator PNG (warm) | 631.5 µs | 21.38 ms | 33.9× |
| `BM-03-WebP_Lossy82-cold` | NativeMercator WebP_Lossy82 (cold) | 3.83 ms | 14.76 ms | 3.9× |
| `BM-03-WebP_Lossy82-warm` | NativeMercator WebP_Lossy82 (warm) | 3.18 ms | 14.79 ms | 4.6× |
| `BM-03-WebP_Lossless-cold` | NativeMercator WebP_Lossless (cold) | 16.65 ms | — | — |
| `BM-03-WebP_Lossless-warm` | NativeMercator WebP_Lossless (warm) | 15.82 ms | — | — |
| `BM-03-RawF32-cold` | NativeMercator RawF32 (cold) | 964.1 µs | 10.68 ms | 11.1× |
| `BM-03-RawF32-warm` | NativeMercator RawF32 (warm) | 370.7 µs | 10.69 ms | 28.8× |
| `BM-04-cold` | Reproj4326 PNG (cold) | 2.26 ms | 22.06 ms | 9.8× |
| `BM-04-warm` | Reproj4326 PNG (warm) | 635.2 µs | 22.34 ms | 35.2× |
| `BM-05a-SatRGB-PNG-cold` | Satellite UTM10N RGB PNG (cold) | 4.87 ms | 40.84 ms | 8.4× |
| `BM-05a-SatRGB-PNG-warm` | Satellite UTM10N RGB PNG (warm) | 1.62 ms | 41.31 ms | 25.4× |
| `BM-05b-UTM-LZW-cold` | UTM10N LZW+Pred3 PNG (cold) | 3.64 ms | 30.54 ms | 8.4× |
| `BM-05b-UTM-LZW-warm` | UTM10N LZW+Pred3 PNG (warm) | 729.1 µs | 30.49 ms | 41.8× |
| `BM-05c-Albers-cold` | Albers 5070 PNG (cold) | 1.37 ms | 29.41 ms | 21.5× |
| `BM-05c-Albers-warm` | Albers 5070 PNG (warm) | 677.7 µs | 29.35 ms | 43.3× |
| `BM-05d-FusedNDVI-cold` | Satellite Fused NDVI PNG (cold) | 4.42 ms | — | — |
| `BM-05d-FusedNDVI-warm` | Satellite Fused NDVI PNG (warm) | 1.12 ms | — | — |
| `BM-06` | SparseCoastline (avg/tile) | 412.9 µs | 11.85 ms | 28.7× |
| `BM-07-HTTP-Mercator` | PanBurst32 HTTP Mercator PNG | 3,363 tiles/s | 85 tiles/s | 39.8× |
| `BM-07-HTTP-SatRGB` | PanBurst32 HTTP Satellite UTM10N RGB PNG | 1,369 tiles/s | 62 tiles/s | 22.1× |
| `BM-08` | RemoteRangeRTT (20ms RTT) | 45.70 ms | 87.36 ms | 1.9× |
| `BM-09-LandsatC2-cold` | Landsat C2 3-File uint16 RGB PNG (cold) | 6.95 ms | — | — |
| `BM-09-LandsatC2-warm` | Landsat C2 3-File uint16 RGB PNG (warm) | 2.62 ms | 62.64 ms | 23.9× |
| `BM-10-OperaSAR-RGB-cold` | OPERA RTC-S1 Dual-Pol SAR-RGB PNG (cold) | 5.94 ms | — | — |
| `BM-10-OperaSAR-RGB-warm` | OPERA RTC-S1 Dual-Pol SAR-RGB PNG (warm) | 2.25 ms | 55.90 ms | 24.9× |
| `BM-11-z5` | MultiZoom Landsat C2 RGB z=5 | 1.07 ms | 34.06 ms | 32.0× |
| `BM-11-z8` | MultiZoom Landsat C2 RGB z=8 | 1.28 ms | 29.47 ms | 23.1× |
| `BM-11-z10` | MultiZoom Landsat C2 RGB z=10 | 2.54 ms | 54.76 ms | 21.6× |
| `BM-11-z12` | MultiZoom Landsat C2 RGB z=12 | 2.58 ms | 64.60 ms | 25.0× |
| `BM-11-z14` | MultiZoom Landsat C2 RGB z=14 | 1.90 ms | 19.87 ms | 10.4× |
| `BM-11-z16` | MultiZoom Landsat C2 RGB z=16 | 1.79 ms | 16.23 ms | 9.1× |

## Reproducing

Run from the repository root.

```bash
# 1. Generate deterministic COG fixtures (requires uv)
cd benches && uv venv .venv && uv pip install --python .venv/bin/python rasterio rio-tiler titiler.core uvicorn numpy pillow httpx
.venv/bin/python generate_fixtures.py
.venv/bin/python generate_landsat_sar_fixtures.py
cd ..

# 2. Run cogzilla benchmarks + parity dumps
cargo run --release -p cogmark -- --dump-parity benches/results/parity

# 3. Run GDAL / TiTiler baseline & pixel parity gate
benches/.venv/bin/python benches/gdal_oracle.py
```

To run the suite on a Linux x86-64 runner instead, trigger the COGMark workflow (`.github/workflows/bench.yml`) from the Actions tab; it writes the comparison table to the run summary and uploads the raw JSON. After a local run, `python benches/compare_results.py` prints the same table.

The S3 comparison has its own scripts and instructions in [S3_BENCHMARK.md](S3_BENCHMARK.md#reproducing) (about 25 minutes and 4 GB of downloads from public buckets).
