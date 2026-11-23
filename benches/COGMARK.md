# `COGMark`: local benchmarks vs. GDAL / `rio-tiler` / TiTiler

Local synthetic fixtures, no network. "Warm" means the source blocks are already decoded in `cogzilla`'s cache, so warm rows are the best case; for remote data and uncached tiles see [S3_BENCHMARK.md](S3_BENCHMARK.md).

Results are from two machines. Ratios compare `cogzilla` with GDAL, `rio-tiler` or TiTiler run back to back on the same machine, so compare ratios across machines, not absolute times.

| | Apple M5 | Linux x86-64 |
| :--- | :--- | :--- |
| Hardware | 10 cores, ARM64 NEON | GitHub-hosted runner, 4 vCPU AMD EPYC 9V74 (shared), AVX2 |
| Build | `rustc 1.95.0`, `target-cpu=native` | `rustc 1.100.0`, default target CPU (as the release binaries) |
| Baseline | not recorded | TiTiler 2.4.1, rio-tiler 9.4.14, rasterio 1.5.2 (GDAL 3.12.2), 4 workers |
| Samples | not recorded | one run (`generic`), commit `a9eab25` |

## Summary

| Workload | M5 | Linux x86-64 |
| :--- | ---: | ---: |
| HTTP Mercator PNG, 32 concurrent | 37.8× | 20.8× |
| HTTP satellite RGB PNG, 32 concurrent | 22.2× | 9.9× |
| Mercator PNG tile, warm | 27.7× | 13.8× |
| Reproject to `EPSG:4326` PNG, warm | 42.4× | 14.6× |
| Satellite RGB PNG, warm | 39.2× | 16.8× |
| Landsat three-file RGB PNG, warm | 15.6× | 21.2× |
| OPERA SAR RGB PNG, warm | 15.3× | 17.7× |
| Sparse coastline, average per tile | 21.9× | 13.0× |
| Remote cold open and tile, 20 ms RTT | 3.0× | 1.9× |
| COG header open | 312× | 816× |

Where `cogzilla` is not far ahead, or behind, on the Linux run:

- **One raw decode path is slightly slower than GDAL:** LZW with predictor 1 (`0.8×`, or `0.9×` with `target-cpu=native`). DEFLATE with predictor 1 is a tie (`1.0×`); DEFLATE with predictor 2 (`1.2×`) and LZW with predictor 3 (`1.3×`) are now ahead of GDAL.
- **Lossy WebP output** is 3.6× warm, because the encoder dominates.
- **Low-zoom tiles from the three-file Landsat scene** are about 2–2.5× (`z=5`, `z=8`), and 6.7–8.8× at `z=14` and `z=16`, against 16–21× at `z=10` and `z=12`.
- **First-touch (cold) local tiles** are 3.2–9.2×.

## Apple M5

| ID | Workload | `cogzilla` | GDAL / TiTiler | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| **`BM-01`** | `ColdHeader` (5-IFD COG open) | **1.3 µs** | 406 µs | **312×** |
| **`BM-02`** | `UnshufflePred3` (`ZSTD + P3`, 1 MiB) | **350 µs** (3.00 GB/s) | 835 µs (1.26 GB/s) | **2.4×** |
| **`BM-03`** | `NativeMercator` PNG (warm) | **634 µs** | 17,587 µs | **27.7×** |
| **`BM-03`** | `NativeMercator` Raw `f32` (warm) | **268 µs** | 12,377 µs | **46.2×** |
| **`BM-04`** | `Reproj4326` PNG (warm) | **607 µs** | 25,766 µs | **42.4×** |
| **`BM-05a`** | `Satellite UTM10N 3-Band RGB` PNG (warm / cold) | **1.22 ms** / 6.65 ms | 47.7 ms / 47.8 ms | **39.2×** / **7.2×** |
| **`BM-05d`** | `Satellite Fused NDVI` PNG (warm) | **917 µs** | 14,208 µs | **15.5×** |
| **`BM-06`** | `SparseCoastline` (50% empty, avg/tile) | **420 µs** | 9,180 µs | **21.9×** |
| **`BM-07`** | `PanBurst32` HTTP Mercator PNG (32 concurrent) | **6,863 tiles/s** | 181 tiles/s | **37.8×** |
| **`BM-07`** | `PanBurst32` HTTP Satellite RGB PNG (32 concurrent) | **3,791 tiles/s** | 171 tiles/s | **22.2×** |
| **`BM-08`** | `RemoteRangeRTT` (20 ms RTT cold open + tile) | **48.3 ms** (3 reqs) | 147.0 ms (4 reqs) | **3.0×** |
| **`BM-09`** | `Landsat C2 3-File uint16 RGB` PNG (warm) | **1.63 ms** | 25.4 ms | **15.6×** |
| **`BM-10`** | `OPERA RTC-S1 Dual-Pol SAR-RGB` PNG (warm) | **1.35 ms** | 20.7 ms | **15.3×** |
| **`BM-11`** | `MultiZoom` Landsat C2 (`z=5 / 8 / 10 / 12 / 14 / 16`) | **7.3 / 7.0 / 1.8 / 1.6 / 1.2 / 1.2 ms** | — | `99.86–100%` mask parity |

## Linux x86-64

Full output of [run 45181195849](https://github.com/moradology/cogzilla/actions/runs/45181195849) (`generic`, matching release binaries; see also [run 45181201440](https://github.com/moradology/cogzilla/actions/runs/45181201440) for `target-cpu=native`). Rows with no baseline have no GDAL or TiTiler equivalent in the suite.

| ID | Workload | cogzilla | GDAL / TiTiler | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| `BM-01` | ColdHeader | 1.2 µs | 981.6 µs | 815.9× |
| `BM-02-NONE_P1` | UnshufflePred3 (NONE_P1) | 65.3 µs | 143.3 µs | 2.2× |
| `BM-02-DEFLATE_P1` | UnshufflePred3 (DEFLATE_P1) | 2.49 ms | 2.55 ms | 1.0× |
| `BM-02-DEFLATE_P2` | UnshufflePred3 (DEFLATE_P2) | 2.88 ms | 3.32 ms | 1.2× |
| `BM-02-DEFLATE_P3` | UnshufflePred3 (DEFLATE_P3) | 1.69 ms | 2.73 ms | 1.6× |
| `BM-02-ZSTD_P1` | UnshufflePred3 (ZSTD_P1) | 869.5 µs | 1.09 ms | 1.2× |
| `BM-02-ZSTD_P3` | UnshufflePred3 (ZSTD_P3) | 489.5 µs | 1.54 ms | 3.1× |
| `BM-02-LZW_P1` | UnshufflePred3 (LZW_P1) | 3.03 ms | 2.31 ms | 0.8× |
| `BM-02-LZW_P3` | UnshufflePred3 (LZW_P3) | 2.46 ms | 3.13 ms | 1.3× |
| `BM-03-PNG-cold` | NativeMercator PNG (cold) | 3.58 ms | 24.46 ms | 6.8× |
| `BM-03-PNG-warm` | NativeMercator PNG (warm) | 1.78 ms | 24.42 ms | 13.8× |
| `BM-03-WebP_Lossy82-cold` | NativeMercator WebP_Lossy82 (cold) | 6.57 ms | 17.15 ms | 2.6× |
| `BM-03-WebP_Lossy82-warm` | NativeMercator WebP_Lossy82 (warm) | 4.76 ms | 17.14 ms | 3.6× |
| `BM-03-WebP_Lossless-cold` | NativeMercator WebP_Lossless (cold) | 19.96 ms | — | — |
| `BM-03-WebP_Lossless-warm` | NativeMercator WebP_Lossless (warm) | 18.11 ms | — | — |
| `BM-03-RawF32-cold` | NativeMercator RawF32 (cold) | 2.46 ms | 12.08 ms | 4.9× |
| `BM-03-RawF32-warm` | NativeMercator RawF32 (warm) | 691.0 µs | 12.08 ms | 17.5× |
| `BM-04-cold` | Reproj4326 PNG (cold) | 7.72 ms | 25.06 ms | 3.2× |
| `BM-04-warm` | Reproj4326 PNG (warm) | 1.71 ms | 25.05 ms | 14.6× |
| `BM-05a-SatRGB-PNG-cold` | Satellite UTM10N RGB PNG (cold) | 10.48 ms | 43.83 ms | 4.2× |
| `BM-05a-SatRGB-PNG-warm` | Satellite UTM10N RGB PNG (warm) | 2.61 ms | 43.81 ms | 16.8× |
| `BM-05b-UTM-LZW-cold` | UTM10N LZW+Pred3 PNG (cold) | 10.39 ms | 35.50 ms | 3.4× |
| `BM-05b-UTM-LZW-warm` | UTM10N LZW+Pred3 PNG (warm) | 1.77 ms | 35.50 ms | 20.0× |
| `BM-05c-Albers-cold` | Albers 5070 PNG (cold) | 3.73 ms | 34.24 ms | 9.2× |
| `BM-05c-Albers-warm` | Albers 5070 PNG (warm) | 1.70 ms | 34.10 ms | 20.1× |
| `BM-05d-FusedNDVI-cold` | Satellite Fused NDVI PNG (cold) | 9.96 ms | — | — |
| `BM-05d-FusedNDVI-warm` | Satellite Fused NDVI PNG (warm) | 2.09 ms | — | — |
| `BM-06` | SparseCoastline (avg/tile) | 1.06 ms | 13.76 ms | 13.0× |
| `BM-07-HTTP-Mercator` | PanBurst32 HTTP Mercator PNG | 1,502 tiles/s | 72 tiles/s | 20.8× |
| `BM-07-HTTP-SatRGB` | PanBurst32 HTTP Satellite UTM10N RGB PNG | 812 tiles/s | 82 tiles/s | 9.9× |
| `BM-08` | RemoteRangeRTT (20ms RTT) | 48.51 ms | 90.40 ms | 1.9× |
| `BM-09-LandsatC2-cold` | Landsat C2 3-File uint16 RGB PNG (cold) | 20.57 ms | — | — |
| `BM-09-LandsatC2-warm` | Landsat C2 3-File uint16 RGB PNG (warm) | 3.39 ms | 71.91 ms | 21.2× |
| `BM-10-OperaSAR-RGB-cold` | OPERA RTC-S1 Dual-Pol SAR-RGB PNG (cold) | 18.86 ms | — | — |
| `BM-10-OperaSAR-RGB-warm` | OPERA RTC-S1 Dual-Pol SAR-RGB PNG (warm) | 3.55 ms | 62.67 ms | 17.7× |
| `BM-11-z5` | MultiZoom Landsat C2 RGB z=5 | 16.31 ms | 40.49 ms | 2.5× |
| `BM-11-z8` | MultiZoom Landsat C2 RGB z=8 | 16.49 ms | 35.07 ms | 2.1× |
| `BM-11-z10` | MultiZoom Landsat C2 RGB z=10 | 3.80 ms | 62.24 ms | 16.4× |
| `BM-11-z12` | MultiZoom Landsat C2 RGB z=12 | 3.40 ms | 72.22 ms | 21.2× |
| `BM-11-z14` | MultiZoom Landsat C2 RGB z=14 | 2.61 ms | 22.96 ms | 8.8× |
| `BM-11-z16` | MultiZoom Landsat C2 RGB z=16 | 2.50 ms | 16.66 ms | 6.7× |

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
