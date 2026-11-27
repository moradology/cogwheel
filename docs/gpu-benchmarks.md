# GPU Acceleration & Sustained Workload Benchmarks

`cogzilla-server` builds with `wgpu` GPU acceleration enabled by default (`default = ["gpu"]`, adding **+1.94 MiB** to the stripped release binary: `10.13 MiB` vs `8.19 MiB` CPU-only, with zero bundled C/CUDA libraries). At startup, `GpuBackend` auto-detects hardware Vulkan or Metal adapters, rejects software renderers (`llvmpipe`/SwiftShader), calibrates its `DispatchPolicy` against the adapter's measured dispatch floor, and falls back silently to CPU if no hardware GPU is present or if `COGZILLA_GPU=never` is set. `cogzilla-lambda` stays strictly CPU-only (`default-features = false`, zero GPU crates).

For the full 49-PR implementation history, see [gpu-roadmap.md](gpu-roadmap.md).

## Architecture & Continuous Batching

All tile, preview, bbox, crop, feature, and `/cog/statistics` / `/stac/statistics` GPU renders flow through `GpuCoalescer` ([coalesce.rs](../crates/cogzilla-gpu/src/coalesce.rs)) using a depth-2 double-buffered ring:

- **Continuous micro-batching (`GpuCoalescer`)**: Heterogeneously packs concurrent `Colormap`, `Terrain` (Horn hillshade/slope/aspect/Terrarium/TerrainRGB), `Expr`, `Rgb`, `SarRgb`, `ExprRgb`, `MultiBandF32`, and `ExprRgbF32` jobs into shared `CommandEncoder` submits, eliminating the per-tile dispatch floor (~84 µs on Vulkan RTX 3090 Ti, ~1.5 ms on Metal M5).
- **1-pass fused outputs**: `CoalescedJob::WithJpeg` chains 4:2:0 YCbCr + 8×8 FDCT + quantization directly onto any RGBA8 job in the same `CommandEncoder` (zero RGBA8 readback or re-upload); `CoalescedJob::WithStats` chains the 3-stage `stats.wgsl` reducer directly onto any `RawF32` job (~12 KB readback instead of 16–64 MB).
- **Coordinate compression**: `BboxCoordSource` avoids 128 MB `Exact2D` coordinate uploads via 1D `Separable` (32 KB at `4096²`), 17×17 `Grid2D` for curved CRSs like UTM/Albers/LCC (2.3 KB), and zero-new-binding 1-bit `SeparableMasked` / `Grid2DMasked` polygon masks (2 MB at `4096²`).
- **Standalone single-submit reducers**: Large multi-megabyte operations — `GpuWeightedStatsReducer` (exact analytical coverage & ellipsoidal area-weighted polygon stats + fused `weighted_expr.wgsl` VM), `GpuMosaicCompositor` (multi-scene pixel selection + fused RGB → GPU JPEG with persistent `scene_buf`), and fallback `GpuJpegEncoder` — batch all bands/expressions of a request into 1 GPU submit behind their own mutexes.

---

## Sustained Realistic Map-Server Soak (`e2e_realistic_workload_bench`)

Measured at commit `f01a5a6` (`gpu-impl`) using [e2e_realistic_workload_bench.rs](../crates/cogzilla-server/examples/e2e_realistic_workload_bench.rs) against the real `TileEngine` with a tiled `2048×2048` 3-band F32 GeoTIFF COG on disk, **encoded-tile cache disabled** (so every request executes a full render + encode), and a 726-tile `z=9..12` Web Mercator viewport pool.

Traffic mix in Scenarios 1–3: **40%** `512×512` RGB Lanczos3 JPEG (`q=85`), **25%** `256×256` DEM Horn Hillshade Lanczos3 PNG, **20%** `512×512` NDVI Expression Cubic + Viridis PNG, **15%** `256×256` Single-band Lanczos3 + Viridis PNG.

### NVIDIA GeForce RTX 3090 Ti (`thales`, Vulkan, discrete, `floor = 83.9 µs`)

#### Sustained Mixed Map Traffic (`8s` each)
| Workload | CPU-Only | GPU Continuous Batching | Throughput Speedup |
| :--- | :--- | :--- | ---: |
| **Pan & Zoom** (16 clients, 0–6 ms jitter) | 155 req/s (p50=21.4, p95=244.8, p99=275.8 ms) | **2,484 req/s** (p50=**2.2**, p95=**4.5**, p99=**10.5 ms**) | **16.1×** |
| **Mixed Tile Flood** (32 clients, 0 ms think) | 232 req/s (p50=29.5, p95=352.1, p99=356.2 ms) | **3,490 req/s** (p50=**8.9**, p95=**11.9**, p99=**19.9 ms**) | **15.1×** |

#### Live Tiles + Heavy Analytics Contention (12 live tile clients + 2 heavy `2048²` clients, `8s`)
| Stream | CPU-Only | GPU Continuous Batching | Speedup |
| :--- | :--- | :--- | ---: |
| **Live Tiles** (12 clients, 2 ms pacing) | 94 req/s (p50=20.1, p95=238.3, p99=254.2 ms) | **1,170 req/s** (p50=**6.0**, p95=**14.8**, p99=**19.6 ms**) | **12.4×** |
| **Heavy Analytics** (`2048²` bbox/stats + `1024²` zonal) | 1 req/s (p50=2,782.7, p95=3,564.3 ms) | **106 req/s** (p50=**13.0**, p95=**41.6**, p99=**49.1 ms**) | **127.0×** |

#### Concurrency Scaling (`512×512` RGB Lanczos3 JPEG, `4s` each)
| Concurrency | CPU-Only | GPU Continuous Batching | Throughput Speedup |
| :--- | :--- | :--- | ---: |
| **1 client** | 5 req/s (p50=201.9, p99=208.9 ms) | **1,149 req/s** (p50=**0.7**, p95=**1.9**, p99=**3.3 ms**) | **228.8×** |
| **4 clients** | 20 req/s (p50=203.6, p99=212.9 ms) | **1,711 req/s** (p50=**2.1**, p95=**4.1**, p99=**6.1 ms**) | **87.4×** |
| **16 clients** | 72 req/s (p50=209.0, p99=287.1 ms) | **2,036 req/s** (p50=**7.4**, p95=**10.7**, p99=**19.1 ms**) | **28.3×** |
| **32 clients** | 103 req/s (p50=343.1, p99=365.7 ms) | **2,091 req/s** (p50=**14.7**, p95=**17.5**, p99=**39.5 ms**) | **20.3×** |

- **Coalescer & Cache Telemetry**: **85,722** coalesced jobs across **17,572** batches (avg batch = `4.88`, max = `32`); slab cache hit rate **99.7%** single-band / **83.9%** multi-band.

---

### Apple M5 (Metal, unified memory, `floor = 1542.1 µs`)

#### Sustained Mixed Map Traffic (`8s` each)
| Workload | CPU-Only | GPU Continuous Batching | Throughput Speedup |
| :--- | :--- | :--- | ---: |
| **Pan & Zoom** (16 clients, 0–6 ms jitter) | 140 req/s (p50=16.2, p95=196.1, p99=228.4 ms) | **940 req/s** (p50=**12.6**, p95=**21.4**, p99=**24.9 ms**) | **6.7×** |
| **Mixed Tile Flood** (32 clients, 0 ms think) | 136 req/s (p50=16.0, p95=204.3, p99=267.0 ms) | **1,426 req/s** (p50=**22.4**, p95=**36.3**, p99=**45.0 ms**) | **10.5×** |

#### Live Tiles + Heavy Analytics Contention (12 live tile clients + 2 heavy `2048²` clients, `8s`)
| Stream | CPU-Only | GPU Continuous Batching | Speedup |
| :--- | :--- | :--- | ---: |
| **Live Tiles** (12 clients, 2 ms pacing) | 89 req/s (p50=22.4, p95=260.7, p99=321.5 ms) | **302 req/s** (p50=**36.9**, p95=**60.1**, p99=**73.8 ms**) | **3.4×** |
| **Heavy Analytics** (`2048²` bbox/stats + `1024²` zonal) | 1 req/s (p50=2,426.4, p95=3,830.3 ms) | **60 req/s** (p50=**31.1**, p95=**53.5**, p99=**62.5 ms**) | **66.3×** |

#### Concurrency Scaling (`512×512` RGB Lanczos3 JPEG, `4s` each)
| Concurrency | CPU-Only | GPU Continuous Batching | Throughput Speedup |
| :--- | :--- | :--- | ---: |
| **1 client** | 12 req/s (p50=86.3, p99=98.1 ms) | **218 req/s** (p50=**4.5**, p95=**7.0**, p99=**8.1 ms**) | **18.5×** |
| **4 clients** | 38 req/s (p50=107.0, p99=122.3 ms) | **316 req/s** (p50=**13.2**, p95=**16.3**, p99=**19.0 ms**) | **8.3×** |
| **16 clients** | 48 req/s (p50=200.7, p99=332.5 ms) | **606 req/s** (p50=**25.0**, p95=**38.4**, p99=**54.6 ms**) | **12.5×** |
| **32 clients** | 49 req/s (p50=199.9, p99=273.2 ms) | **654 req/s** (p50=**48.7**, p95=**68.1**, p99=**100.2 ms**) | **13.2×** |

- **Coalescer & Cache Telemetry**: **27,562** coalesced jobs across **4,586** batches (avg batch = `6.01`, max = `31`); slab cache hit rate **99.2%** single-band / **84.9%** multi-band.

---

## Targeted Single-Endpoint Highlights

| Endpoint / Workload | RTX 3090 Ti (Vulkan) | Apple M5 (Metal) |
| :--- | ---: | ---: |
| `4096²` `/cog/preview` single-band stats (`Separable`) | **0.59 ms** (**1,476.7×** vs CPU) | — |
| `4096²` UTM EPSG:32610 → WGS84 single stats (`Grid2D`) | **1.33 ms** (**702.5×** vs CPU) | **8.93 ms** (**42.5×** vs CPU) |
| `4096²` `/cog/bbox` RGB Lanczos3 JPEG (`Separable`) | **32.26 ms** (**405.3×** vs CPU) | **56.53 ms** (**85.7×** vs CPU) |
| `2048²` 4-band F32 `/cog/statistics` (coalesced 1-submit) | **1.37 ms** (**424.5×** vs CPU) | **5.28 ms** (**47.7×** vs CPU) |
| `4096²` polygon-masked single stats (1-bit `SeparableMasked`) | **4.36 ms** (**85.5×** vs CPU) | **7.64 ms** (**23.6×** vs CPU) |
| `2048²` exact ellipsoidal area-weighted 4-band zonal stats (`zonal.rs`) | **36.29 ms** (**9.4×** vs CPU) | **17.66 ms** (**5.8×** vs CPU) |
| `2048²` exact weighted 3× expr (`NDVI;NDWI;EVI`) zonal stats (1-pass GPU VM) | **37.98 ms** (**7.8×** vs CPU) | **21.19 ms** (**4.6×** vs CPU) |
| `2048² × 12` styled Median mosaic → JPEG (1-pass fused GPU JPEG) | **31.12 ms** (**7.7×** vs CPU) | **33.12 ms** (**4.4×** vs CPU) |
| `2048² × 8 × RGB` Median mosaic → JPEG (1-pass fused RGB → GPU JPEG) | **103.84 ms** (**3.6×** vs CPU) | **125.71 ms** (**5.4×** vs CPU) |

---

## Chaos, Torture & Hard-Limit Testing

Adversarial stress testing via [gpu_torture_test.rs](../crates/cogzilla-server/examples/gpu_torture_test.rs), [gpu_breaking_point_test.rs](../crates/cogzilla-server/examples/gpu_breaking_point_test.rs), and [gpu_torture_round3.rs](../crates/cogzilla-server/examples/gpu_torture_round3.rs) surfaced **5 real production failure modes** (fixed in PRs #345 and #348) and established hard operational limits on both Metal (M5) and Vulkan (RTX 3090 Ti):

### Failure Modes Caught & Fixed
1. **Coalescer staging-buffer crash on concurrent `4096²` bursts** *(fixed in PR #348)*: `16` concurrent `4096²` RawRgba jobs (`1 GiB`) exceeded `wgpu`'s `512 MiB` `max_buffer_size` inside `cogzilla-gpu-coalescer`, panicking the background worker thread permanently. Fixed via `CoalescedJob::estimated_staging_and_slot_bytes()`, per-job rejection to CPU, and 1-slot `deferred` batch splitting capped at `min(max_buffer_size/2, 256 MiB)` (all `16/16` now complete in `1.00–1.19 s` with the coalescer alive).
2. **Orphaned-job queue storm on client disconnect** *(fixed in PR #345)*: Dropping ~32,000 futures mid-flight at 1–3 ms timeouts previously left dead jobs draining through `GpuCoalescer`'s unbounded channel for 5+ seconds. Fixed with `GpuCoalescer::submit_cancellable` + RAII `CancelOnDrop` (discards cancelled requests before GPU encoding) and bounded `COGZILLA_GPU_MAX_QUEUED=4096`.
3. **256–2,048 client Tokio async-worker starvation** *(fixed in PRs #345, #348)*: Previously, crossing `max_in_flight × 32` shed tiles to synchronous CPU rendering directly on Tokio's async worker threads, pinning `p95` at `5,000+ ms`. Raising `policy.max_in_flight = 128` (`4,096` coalesced async GPU jobs) eliminated runtime starvation up to 2,048+ concurrent clients.
4. **Mosaic `in_flight` unit mismatch** *(fixed in PR #345)*: `try_composite_mosaic_*` passed raw `in_flight` instead of `in_flight / 32`, locking Mosaic out of the GPU whenever ≥4 tiles were active.
5. **Noisy-neighbor startup floor calibration** *(fixed in PR #345)*: Switched `GpuProfile::calibrate` from median-of-5 to p20-of-7 so a transient GPU spike at boot no longer permanently inflates `dispatch_floor_us`.

### Verified Torture Results (Post-Fix)

| Adversarial Test | RTX 3090 Ti (Vulkan) | Apple M5 (Metal) |
| :--- | :--- | :--- |
| **Degenerate geometry (`1×1`, `1×4096`, `4096×1`, `333×777`), `[0,0]` rescale, all-NaN, 768 MiB mosaic** | Survived (clean fallback) | Survived (clean fallback) |
| **Mid-flight cancel storm (64 tasks, 1–3 ms timeouts, 4s)** | Survived (`2,295` ok, `34,830` dropped cleanly) | Survived (`957` ok, `35,339` dropped cleanly) |
| **Slab-cache thrashing (12–24 distinct COGs, ~1.15 GiB, 5s)** | `6,153–13,199` tiles, `201–436` evictions, 0 OOM | `3,959–4,305` tiles, `505–961` evictions, 0 OOM |
| **256-client flood (5s)** | **4,655 req/s** (`p50=53.7`, `p95=63.1`, `max=104.9 ms`) | **1,560 req/s** (`p50=177.7`, `p95=197.8`, `max=218.4 ms`) |
| **512-client flood (5s)** | **4,109 req/s** (`p50=124.6`, `p95=147.3`, `max=168.3 ms`) | **1,606 req/s** (`p50=344.0`, `p95=377.5`, `max=399.7 ms`) |
| **2,048-client flood (5s)** | **6,144 req/s** (`p50=318.4`, `p95=392.2`, `max=727.9 ms`) | **3,351 req/s** (`p50=609.6`, `p95=659.9`, `max=681.4 ms`) |
| **16× concurrent `4096²` RawRgba (`1 GiB` burst)** | **16/16 ok, 0 err** (`1,000 ms`, coalescer alive) | **16/16 ok, 0 err** (`1,193 ms`, coalescer alive) |
| **Cross-subsystem chaos (64 tile + 4× `4096²` + 4× zonal + 4× mosaic RGB JPEG, 8s)** | `1,518` tiles, `78` `4096²`, `131` zonals, `74` GPU mosaics (`0` fallbacks) | `1,098` tiles, `62` `4096²`, `49` zonals, `35` GPU mosaics (`0` fallbacks) |
| **Adversarial stats (all-NaN, 1-px, const, `-0.0`, `1e-40`, `±1e30`, disjoint hist)** | Survived | Survived |
| **327-op `CustomExprRgb`, deep nesting, 3× `GpuBackend` init/drop** | Survived (0 diffs) | Survived (0 diffs, `100.5 ms` lifecycle) |
| **VRAM delta across 83,473 GPU jobs** | `19,116 → 19,125 MiB` (**+9 MiB / 0 leak**) | Unified memory, 0 leak |

### Known Hard Limits
- **Single-buffer `512 MiB` adapter limit**: Default `wgpu` Metal/Vulkan `max_buffer_size` and `max_storage_buffer_binding_size` are `512 MiB`. Single jobs requiring `> 256 MiB` staging or `> 512 MiB` storage cleanly fall back to CPU.
- **Extreme concurrency (`> 4,096` concurrent uncached jobs)**: Built-in HTTP **AIMD concurrency limiting** (`AimdLimiter`, enabled by default) automatically adapts the concurrency cap to the Little's Law knee (`limit* ≈ throughput × target_rtt`) and sheds excess traffic with HTTP `429 Too Many Requests` (`Retry-After: 1`) in `< 1 ms`, preventing Tokio worker starvation on both CPU-only and GPU servers.
- **`+Inf` expression overflow**: Expressions overflowing to `±Inf` (e.g. `1e38 * 1e38`) are masked as transparent (`!is_finite_f32`) on GPU, whereas CPU `render_expr` only masks `is_nan()` (clamping `+Inf` to opaque `255`).

---

## Adaptive AIMD Concurrency Limiting (`429 Too Many Requests`)

Live HTTP overload benchmarks via [e2e_aimd_overload_bench.rs](../crates/cogzilla-server/examples/e2e_aimd_overload_bench.rs) (`serve_listener` on `127.0.0.1:0`, `512×512` RGB Lanczos3 JPEG across `10,291` unique tiles, zero tile cache, target RTT `100 ms`, concurrent `/healthz` probes every `100 ms`). Heavy non-Nearest CPU RGB renders (`≥ 256²`) run on `tokio::task::spawn_blocking` so Tokio's async I/O workers remain free to answer `/healthz` and shed `429`s even when all CPU cores are saturated.

| Host & Backend | Clients | AIMD | `200 OK` Goodput | `200 OK` p50 | `200 OK` p99 | `429` Shed (p50) | `/healthz` p99 | Converged `limit` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple M5 CPU-only (10 cores)** | 128 | OFF | 63.7 req/s | 1,949.1 ms | 2,362.2 ms | 0 | 0.5 ms | — |
| **Apple M5 CPU-only (10 cores)** | 128 | **ON** | 56.4 req/s | **180.1 ms (10.8×)** | **728.6 ms (3.2×)** | 37,863 (**0.6 ms**) | **6.9 ms** | `10` |
| **Apple M5 GPU (Metal)** | 256 | OFF | 824.5 req/s | 307.4 ms | 356.2 ms | 0 | 5.9 ms | — |
| **Apple M5 GPU (Metal)** | 256 | **ON** | **794.0 req/s** | **117.5 ms (2.6×)** | **222.1 ms (1.6×)** | 50,924 (**0.7 ms**) | **4.6 ms** | `91` |
| **`thales` CPU-only (48T)** | 128 | OFF | 116.4 req/s | 935.0 ms | 2,601.8 ms | 0 | 2.2 ms | — |
| **`thales` CPU-only (48T)** | 128 | **ON** | 105.1 req/s | **316.9 ms (3.0×)** | **987.2 ms (2.6×)** | 31,539 (**0.3 ms**) | **0.6 ms** | `32` |
| **`thales` RTX 3090 Ti (Vulkan)** | 256 | OFF | 3,495.7 req/s | 73.1 ms | 257.5 ms | 0 | 0.6 ms | — |
| **`thales` RTX 3090 Ti (Vulkan)** | 256 | **ON** | **3,807.2 req/s (+8.9%)** | **63.4 ms** | **97.2 ms (2.6×)** | 2,580 (**0.2 ms**) | **0.5 ms** | `538` |

> **Configuration & Telemetry**: Enabled by default (`COGZILLA_AIMD=off` disables; `COGZILLA_AIMD_TARGET_MS=150`, `COGZILLA_AIMD_MIN_LIMIT`, `COGZILLA_AIMD_MAX_LIMIT=4096`). `/healthz`, `/metrics`, and `OPTIONS` bypass admission control; `4xx` responses call `permit.ignore_rtt()`. Exported in `/metrics` as `cogzilla_aimd_limit`, `cogzilla_aimd_in_flight`, and `cogzilla_aimd_rejected_429_total`.

> **Why AIMD instead of hybrid GPU+CPU work-stealing**: Empirical hill-climbing (`K = 0, 1, 2, 4, 8` CPU helpers stealing from the GPU queue) showed `K = 0` is the global optimum on both hosts — on Apple M5 unified memory, `K = 8` lost **12.8%** throughput to memory-bandwidth contention; on RTX 3090 Ti, `K = 8` blew `p99` up **4.3×** (`42.1 → 180.5 ms`) via CPU stragglers. Pure GPU for heavy work + pure CPU for sub-crossover work + AIMD `429` admission control achieves optimal throughput and tail latency on both CPU-only and GPU deployments.

---

## What These Numbers Do and Don't Say

- **Compute & encode ceiling, not S3 latency**: The soak runs against a local tiled GeoTIFF with warmed decoded blocks in `CogStore`. On cold S3 reads, network I/O and TIFF decompression (DEFLATE/LZW/Zstd) remain on CPU and will dominate single-request latency until blocks enter the cache.
- **Encoded-tile cache was disabled**: Production deployments with `--tile-cache-mb` serve repeat tiles directly from memory without touching CPU or GPU; these numbers measure cache-miss rendering capacity.
- **Tail latency isolation under mixed load**: With `GpuCoalescer` depth-2 pipelining, running heavy `2048²` bbox/stats/zonal jobs concurrently keeps live tile p99 at **19.6 ms** on RTX 3090 Ti (vs `254.2 ms` CPU) and **73.8 ms** on M5 (vs `321.5 ms` CPU).
- **When CPU stays faster**: Very small cold Bilinear single-band tiles on high-dispatch-floor GPUs (like Metal's ~1.5 ms floor) and stacks exceeding the adapter's `max_storage_buffer_binding_size` / `max_buffer_size` are automatically routed to CPU by `DispatchPolicy` and buffer-size guards.

## Reproducing

```bash
# Run the sustained 4-scenario soak benchmark (CPU vs GPU)
cargo run --release -p cogzilla-server --example e2e_realistic_workload_bench

# Pin a specific GPU on multi-GPU Linux hosts
COGZILLA_GPU_INDEX=0 cargo run --release -p cogzilla-server --example e2e_realistic_workload_bench

# Disable GPU at runtime on any server binary
COGZILLA_GPU=never cogzilla-server --port 8080
```
