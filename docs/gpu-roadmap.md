# `cogzilla` GPU Optimization Roadmap (`gpu-impl`)

This document tracks GPU acceleration experiments and PRs targeting the `gpu-impl` feature branch (draft PR against `main`).

## Target Hardware

| Host | CPU | GPU | API / Memory Model |
| :--- | :--- | :--- | :--- |
| **Apple M5** (local) | Apple M5 (10 cores, NEON) | Apple M5 GPU | Metal / `wgpu` (Unified Memory, zero-copy shared buffers) |
| **`thales`** (remote) | AMD Ryzen Threadripper 3960X (24C/48T) | 2× NVIDIA GeForce RTX 3090 Ti (24 GiB each) | Vulkan 1.4 (`wgpu`), CUDA (`libcuda.so.1`) (Discrete PCIe) |

## Guiding Principles

1. **Optional feature-gated crates/backends**: Default CPU builds must remain zero-dependency on GPU SDKs and suffer zero regression in binary size or single-tile CPU latency.
2. **Parity first**: Every GPU kernel must match CPU output within strict tolerances (bit-exact where integer/deterministic, high PSNR / 100% nodata-mask agreement for float resampling/expressions).
3. **Measure both hosts**: Every PR targeting `gpu-impl` must include before/after benchmarks on both Apple M5 and `thales` (RTX 3090 Ti), explicitly noting crossover points (tile size, batch size, resampling mode) where GPU beats CPU.
4. **Adaptive dispatch**: Route work to GPU only when above the measured crossover point; keep fast CPU paths for tiny warm 256×256 single tiles where dispatch overhead would lose.

## Candidate Optimization Backlog

| # | Area | Target Workload | Status | PR |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `cogzilla-gpu` crate foundation | `wgpu` device/queue init, adapter selection (Metal + Vulkan RTX 3090 Ti), buffer pool, Cubic & Lanczos3 compute kernels + benchmark harness | Merged | #210 |
| 2 | Adaptive `DispatchPolicy` + fused colormap | Startup-calibrated cost model (`COGZILLA_GPU_POLICY`), queue backpressure, fused Bilinear/Cubic/Lanczos3 → rescale → RGBA8 / 4×-smaller packed `IndexedU8` | Merged | #213 |
| 3 | DEM / terrain compute shaders | Single-pass 18×18 shared-memory Horn 3×3 (`Hillshade`, `Slope`, `Aspect`) + `Terrarium` / `TerrainRGB` (Bilinear/Cubic/Lanczos3) | Merged | #214 |
| 4 | GPU-resident slab cache + micro-batching | Thread-safe byte-bounded LRU `GpuSlabCache` + 3D `GpuTileBatcher` (`dispatch_workgroups(wg_x, wg_y, K)`) amortizing submit floor | Merged | #216 |
| 5 | Fused `expr_vm` + `color_formula` + colormap | Multi-band WGSL bytecode VM (zero shader recompile) + fused 3-band RGB `Gamma`/`Sigmoidal`/CIE LCH `Saturation` | Merged | #217 |
| 6 | Multi-COG / STAC / Mosaic compositing | `GpuSceneStack` + `GpuMosaicCompositor` (`First`/`Highest`/`Lowest`/`Mean`/`Median`/`Stdev`/`Count`) with fused `f32`/`RGBA8`/`IndexedU8` | Merged | #219 |
| 7 | `cogzilla-server` `--features gpu` integration | Opt-in `GpuBackend` in `TileEngine` (adaptive `DispatchPolicy`, `GpuSlabCache`, fused colormap + DEM/terrain PNG/WebP/JPEG) | Merged | #221 |
| 8 | GPU TIFF Predictor 2/3 unshuffle | Bit-exact zero-scratch single-pass WGSL Predictor 2 & 3 unshuffle + nodata normalization (66.8 GB/s VRAM-only on RTX 3090 Ti) | Merged | #223 |
| 9 | Multi-block stitching & large-crop GPU API | Multi-block `BandSource` stitching (all dtypes: `F32`/`U16`/`U8`/sparse/scale) + arbitrary-size `/preview`/`/bbox` crop PNG acceleration (up to 87.6× at 4096²) | Merged | #225 |
| 10 | Curved 17×17 `Grid2D` & `Exact2D` reprojection | In-shader 17×17 bilinear warp grid + `Exact2D` unlocking GPU acceleration for all 3,260+ projected CRS (UTM/Albers/LCC/Polar; up to 436× on RTX 3090 Ti) | Merged | #228 |
| 11 | Curved `Grid2D`/`Exact2D` DEM & terrain shaders | 17×17 warp-grid + seamless 1-px border extrapolation in `terrain.wgsl` + `TileEngine::try_render_dem` (up to 137× on RTX 3090 Ti, 58.8× on M5) | Merged | #231 |
| 12 | `TileEngine` `CustomExpr` & 3-band `Rgb` GPU wiring | Zero-recompile `ExprProgram` → `GpuInstr` translator + `try_render_expr`/`try_render_rgb` in `TileEngine::render_tile` (up to 36.3× on RTX 3090 Ti, 13.3× on M5 end-to-end PNG) | Merged | #235 |
| 13 | TiTiler `/cog/preview` & `/cog/bbox` GPU wiring | Wired `BboxCoordSource` (`Separable` + `Exact2D`) into `render_bbox_impl` for `Single`, `Dem`, `Terrain`, `CustomExpr`, and `Rgb` crops (up to **84.4×** on RTX 3090 Ti, **33.6×** on M5 end-to-end PNG) | Merged | #238 |
| 14 | Curved `Grid2D`/`Exact2D` ExprVM & 3-band RGB | In-shader 17×17 warp-grid + `Exact2D` in `expr.wgsl`/`GpuExprRenderer`, unlocking GPU acceleration for projected UTM/Albers/LCC & `/cog/bbox` `CustomExpr` + `Rgb` (up to **31.7×** on RTX 3090 Ti, **17.2×** on M5 end-to-end PNG) | Merged | #240 |
| 15 | `GpuMultiBandCache` VRAM LRU & fused `ColorFormula` | Byte-bounded VRAM LRU cache for `GpuMultiBandSlab` (zero CPU stitch/upload on warm `Expr`/`Rgb` tiles & crops) + fused `Gamma`/`Sigmoidal`/CIE LCH `Saturation` wiring in TiTiler (up to **107.4×** on RTX 3090 Ti, **23.3×** on M5 end-to-end PNG) | Merged | #244 |
| 16 | `GpuMosaicCompositor` server wiring | Wired `try_composite_mosaic_f32` & fused `try_composite_mosaic_styled` into `http::render_mosaic_tile_pixel_selection` (`Median`, `Stdev`, etc.) gated by `DispatchPolicy` (up to **2.46×** M5 / **2.25×** RTX 3090 Ti end-to-end including cold 64 MiB upload + PNG) | Merged | #245 |
| 17 | Scientific `RawF32` / `.npy` GPU crops | Added `resample_f32`, `dem_horn_f32`, and `expr_eval_f32` pipelines across `Separable1D`/`Grid2D`/`Exact2D` and wired into `titiler::render_bbox_impl` (backing `.npy`, Float32 `.tif`, `.f32`, and mosaic scene fetches; up to **84.7×** RTX 3090 Ti, **31.5×** M5) | Merged | #249 |
| 18 | Built-in `Spectral1/2/3` GPU indices | Zero-parse `SpectralExpr` → `GpuInstr` compiler (`Db`, `NormalizedDiff`, `Savi`, `Ratio`, `Diff`, `Evi`) wired into `TileEngine::render_tile` and `titiler::render_bbox_impl` (PNG/WebP/JPEG + `.npy`/`.f32`, reusing `GpuMultiBandCache`; up to **189.6×** RTX 3090 Ti, **30.8×** M5) | Merged | #252 |
| 19 | `TileEngine::render_tiles_batch` 3D micro-batching | Groups viewport bursts sharing a COG slab into 1 single 3D GPU dispatch (`GpuTileBatcher`) + 4×-smaller `IndexedU8` readback (32-tile Lanczos3 PNG: **95.8×** / **5,639 tiles/s** RTX 3090 Ti, **21.8×** / **3,252 tiles/s** M5) | Merged | #256 |
| 20 | Fused Dual-Pol SAR RGB + `ColorFormula` GPU pipeline | Added `sar_rgb_color_formula_rgba8` WGSL kernel (`[VV_dB, VH_dB, VV_dB - VH_dB]` + Gamma/Sigmoidal/LCH Saturation) across `Separable1D`/`Grid2D`/`Exact2D` + `GpuMultiBandCache`, wired into `TileEngine` and TiTiler (up to **37.0×** RTX 3090 Ti, **17.8×** M5 end-to-end PNG) | Merged | #258 |
| 21 | Heterogeneous command-buffer micro-batcher (`GpuCoalescer`) | Non-blocking async command-buffer coalescer packing arbitrary concurrent `TileEngine`/TiTiler requests (any COG, slab, colormap, DEM Horn, Terrarium, RawF32, `Separable1D`/`Grid2D`/`Exact2D`) into 1 `CommandEncoder` → 1 `queue.submit()` → 1 `map_async()` → 1 `device.poll(Wait)` (32 mixed requests across 4 COGs × 4 styles in 1.0–1.2 batches: **7,423 tiles/s / 7.02–9.35×** vs multi-core CPU on RTX 3090 Ti, **3,034–3,781 tiles/s / 3.82–4.65×** on M5) | Merged | (this PR) |
| 22 | Hardware JPEG decode/encode | Explore Apple Media Engine / `nvJPEG` for JPEG COGs and `.jpg` tile output | Planned | — |
