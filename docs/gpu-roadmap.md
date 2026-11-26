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
| 1 | `cogzilla-gpu` crate foundation | `wgpu` device/queue init, adapter selection (Metal + Vulkan RTX 3090 Ti), buffer pool, benchmark harness | Planned | — |
| 2 | Heavy resamplers & large crops | `cubic`, `cubic_spline`, `lanczos`, `gauss` on `@2x`/`@4x` tiles and `/preview`, `/bbox`, `/feature` (512–4096 px) | Planned | — |
| 3 | DEM / terrain compute shaders | `hillshade`, `slope`, `aspect`, `contours`, `terrarium`, `terrainrgb` (3×3 Horn stencil + resampling fused) | Planned | — |
| 4 | GPU-resident block cache + micro-batching | Persistent VRAM/unified slab atlas + batched 16–64 tile command buffers for pan bursts & pyramid seeding | Planned | — |
| 5 | Fused `expr_vm` + `color_formula` + colormap | Compile `ExprProgram` IR to WGSL compute shaders; fuse multi-band math, gamma/sigmoidal, and LUT lookup | Planned | — |
| 6 | Multi-COG / STAC / Mosaic compositing | GPU compositing & pixel selection (`highest`, `lowest`, `mean`, `median`, `stdev`) over stacked scene planes | Planned | — |
| 7 | GPU TIFF Predictor 2/3 unshuffle | Parallel byte-unshuffle + prefix-sum in compute shaders (and batched DEFLATE/ZSTD decode exploration on CUDA) | Planned | — |
| 8 | Hardware JPEG decode/encode | Explore Apple Media Engine / `nvJPEG` for JPEG COGs and `.jpg` tile output | Planned | — |
