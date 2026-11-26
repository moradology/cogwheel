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
| 7 | `cogzilla-server` `--features gpu` integration | Opt-in `GpuBackend` in `TileEngine` (adaptive `DispatchPolicy`, `GpuSlabCache`, fused colormap + DEM/terrain PNG/WebP/JPEG) | Merged | (this PR) |
| 8 | GPU TIFF Predictor 2/3 unshuffle | Parallel byte-unshuffle + prefix-sum in compute shaders (and batched DEFLATE/ZSTD decode exploration on CUDA) | Planned | — |
| 9 | Hardware JPEG decode/encode | Explore Apple Media Engine / `nvJPEG` for JPEG COGs and `.jpg` tile output | Planned | — |
