# `COGMark`: local benchmarks vs. GDAL / `rio-tiler` / TiTiler

Measured on Apple M5 (10 cores, ARM64 NEON, `rustc 1.95.0 --release`) against local synthetic fixtures. "Warm" means the source blocks are already decoded in `cogzilla`'s cache, so these rows are the best case; for remote data and uncached tiles see [S3_BENCHMARK.md](S3_BENCHMARK.md).

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

The S3 comparison has its own scripts and instructions in [S3_BENCHMARK.md](S3_BENCHMARK.md#reproducing) (about 25 minutes and 4 GB of downloads from public buckets).
