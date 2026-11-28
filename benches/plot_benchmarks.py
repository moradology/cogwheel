#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "numpy>=1.26"]
# ///
"""Generate reproducible multi-line benchmark scaling charts for docs/benchmarks.md
and docs/gpu-benchmarks.md.

Run with zero global installs:
    uv run benches/plot_benchmarks.py
"""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "axes.titlecolor": "#f0f6fc",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linewidth": 0.8,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
})

C_RTX = "#58a6ff"    # Blue: RTX 3090 Ti (Vulkan)
C_M5G = "#3fb950"    # Green: Apple M5 GPU (Metal)
C_TR  = "#f0883e"    # Orange: Threadripper 3960X CPU (24C/48T)
C_M5C = "#d2a8ff"    # Purple: Apple M5 CPU (10C)
C_TT  = "#f85149"    # Red: TiTiler 2.4
C_MUTED = "#8b949e"


def render_concurrency_overload() -> None:
    """Graph 1: Concurrency (1 -> 2,048 clients) & AIMD overload p99 tail latency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=200)
    fig.suptitle(
        "Concurrency & Overload Scaling — 512×512 RGB Lanczos3 JPEG (Encoded-Tile Cache Off)",
        fontsize=13.5, fontweight="bold", y=0.98, color="#f0f6fc",
    )

    c_scale = np.array([1, 4, 16, 32])
    rtx_rps = np.array([1149, 1711, 2036, 2091])
    m5g_rps = np.array([218, 316, 606, 654])
    tr_rps  = np.array([5, 20, 72, 103])
    m5c_rps = np.array([12, 38, 48, 49])

    c_flood = np.array([256, 512, 2048])
    rtx_flood = np.array([4655, 4109, 6144])
    m5g_flood = np.array([1560, 1606, 3351])

    tr_128 = 116.4
    m5c_128 = 63.7

    ax1.plot(c_scale, rtx_rps, "-o", color=C_RTX, lw=2.5, ms=6.5, label="RTX 3090 Ti (Vulkan, coalesced)")
    ax1.plot([32, *c_flood], [2091, *rtx_flood], "--o", color=C_RTX, lw=1.8, ms=5, alpha=0.75)

    ax1.plot(c_scale, m5g_rps, "-s", color=C_M5G, lw=2.5, ms=6.5, label="Apple M5 GPU (Metal, coalesced)")
    ax1.plot([32, *c_flood], [654, *m5g_flood], "--s", color=C_M5G, lw=1.8, ms=5, alpha=0.75)

    ax1.plot([*c_scale, 128], [*tr_rps, tr_128], "-^", color=C_TR, lw=2.3, ms=6.5,
             label="Threadripper 3960X CPU-only (24C/48T)")
    ax1.plot([*c_scale, 128], [*m5c_rps, m5c_128], "-D", color=C_M5C, lw=2.3, ms=6,
             label="Apple M5 CPU-only (10 cores)")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks([1, 4, 16, 32, 128, 256, 512, 2048])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_ylim(3, 12000)
    ax1.set_xlabel("Concurrent Clients (log₂)")
    ax1.set_ylabel("Throughput (req/s, log₁₀)")
    ax1.set_title("A. Goodput: Single-Core (c=1) → Multi-Core → GPU Flood")
    ax1.grid(True, which="both", alpha=0.4)

    ax1.annotate(
        "c = 1 (single core)\n3960X: 5 req/s  |  M5: 12 req/s\n3090 Ti: 1,149 req/s (229×)",
        xy=(1, 1149), xytext=(2.2, 2700),
        arrowprops=dict(arrowstyle="->", color=C_RTX, lw=1.2),
        fontsize=8.5, bbox=dict(boxstyle="round,pad=0.35", fc="#0d1117", ec=C_RTX, alpha=0.9),
    )
    ax1.annotate(
        "Dashed = 256–2,048c\nmixed-tile flood (0 err)",
        xy=(512, 4109), xytext=(110, 6200),
        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.0),
        fontsize=8, color="#c9d1d9",
    )
    ax1.legend(loc="lower right", fontsize=8.3, framealpha=0.92)

    hosts = [
        "Apple M5\nCPU (10C)\n128 clients",
        "Threadripper\n3960X (24C/48T)\n128 clients",
        "Apple M5\nGPU (Metal)\n256 clients",
        "RTX 3090 Ti\nGPU (Vulkan)\n256 clients",
    ]
    p99_off = [2362.2, 2601.8, 356.2, 257.5]
    p99_on  = [728.6,  987.2,  222.1, 97.2]
    p50_on  = [180.1,  316.9,  117.5, 63.4]

    x = np.arange(len(hosts))
    w = 0.28
    ax2.bar(x - w, p99_off, width=w, color="#f85149", alpha=0.85, label="AIMD OFF — 200 OK p99")
    ax2.bar(x,     p99_on,  width=w, color="#58a6ff", alpha=0.92, label="AIMD ON — 200 OK p99")
    ax2.bar(x + w, p50_on,  width=w, color="#3fb950", alpha=0.92, label="AIMD ON — 200 OK p50")

    ax2.set_yscale("log")
    ax2.set_ylim(20, 6000)
    ax2.set_xticks(x)
    ax2.set_xticklabels(hosts, fontsize=8.8)
    ax2.set_ylabel("Latency (ms, log₁₀)")
    ax2.set_title("B. Overload Protection: AIMD ON vs. OFF (429 Shed in <1 ms)")
    ax2.grid(True, axis="y", which="both", alpha=0.4)

    for i in range(len(hosts)):
        gain = p99_off[i] / p99_on[i]
        ax2.text(x[i], p99_on[i] * 1.15, f"{p99_on[i]:.0f}ms\n({gain:.1f}×)",
                 ha="center", va="bottom", fontsize=8, color="#79c0ff", fontweight="bold")
        ax2.text(x[i] - w, p99_off[i] * 1.08, f"{p99_off[i]:.0f}ms",
                 ha="center", va="bottom", fontsize=7.8, color="#ffa198")
        ax2.text(x[i] + w, p50_on[i] * 1.10, f"{p50_on[i]:.0f}ms",
                 ha="center", va="bottom", fontsize=7.8, color="#7ee787")

    ax2.text(
        0.5, 0.04,
        "AIMD 429 shed latency: 0.2–0.7 ms  |  /healthz p99 under overload: 0.5–6.9 ms",
        transform=ax2.transAxes, ha="center", fontsize=8.3, color="#c9d1d9",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#30363d"),
    )
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ASSETS / "bench-concurrency-overload.png", dpi=200)
    plt.close(fig)


def render_workload_contention() -> None:
    """Graph 2: Heavy single-request latency (1-core CPU vs GPU) & live-tile contention isolation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=200)
    fig.suptitle(
        "Workload Complexity (1-Core CPU vs. GPU) & Live-Tile Contention Isolation",
        fontsize=13.5, fontweight="bold", y=0.98, color="#f0f6fc",
    )

    workloads = [
        "4096² UTM→WGS84\nGrid2D Stats",
        "2048² 4-Band\n/cog/statistics",
        "4096² Polygon\nMasked Stats",
        "4096² RGB\nLanczos3 JPEG",
        "2048² Exact\n4-Band Zonal",
        "2048²×12 Median\nMosaic → JPEG",
    ]
    rtx_ms = np.array([1.33, 1.37, 4.36, 32.26, 36.29, 31.12])
    m5g_ms = np.array([8.93, 5.28, 7.64, 56.53, 17.66, 33.12])
    tr_cpu_ms = np.array([1.33 * 702.5, 1.37 * 424.5, 4.36 * 85.5, 32.26 * 405.3, 36.29 * 9.4, 31.12 * 7.7])
    m5_cpu_ms = np.array([8.93 * 42.5,  5.28 * 47.7,  7.64 * 23.6, 56.53 * 85.7,  17.66 * 5.8, 33.12 * 4.4])

    x = np.arange(len(workloads))
    ax1.plot(x, tr_cpu_ms, "-^", color=C_TR, lw=2.2, ms=7, label="Threadripper 3960X CPU (1 core)")
    ax1.plot(x, m5_cpu_ms, "-D", color=C_M5C, lw=2.2, ms=6.5, label="Apple M5 CPU (1 core)")
    ax1.plot(x, m5g_ms,    "-s", color=C_M5G, lw=2.5, ms=7, label="Apple M5 GPU (Metal, floor ≈ 1.5 ms)")
    ax1.plot(x, rtx_ms,    "-o", color=C_RTX, lw=2.5, ms=7, label="RTX 3090 Ti (Vulkan, floor ≈ 84 µs)")

    for yref in (10, 100, 1000):
        ax1.axhline(yref, color="#30363d", ls="--", lw=1)

    ax1.set_yscale("log")
    ax1.set_ylim(0.5, 35000)
    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads, fontsize=8.3)
    ax1.set_ylabel("Single-Request Latency (ms, log₁₀)")
    ax1.set_title("A. Heavy Endpoints: 1-Core CPU vs. Metal vs. Vulkan")
    ax1.grid(True, axis="y", which="both", alpha=0.35)

    for i, sp in enumerate(["703×", "425×", "86×", "405×", "9.4×", "7.7×"]):
        ax1.annotate(
            sp, xy=(x[i], rtx_ms[i]), xytext=(0, -16), textcoords="offset points",
            ha="center", fontsize=8.2, fontweight="bold", color="#79c0ff",
        )
    ax1.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    configs = [
        "Apple M5\nCPU-only (10C)",
        "Threadripper 3960X\nCPU-only (24C/48T)",
        "Apple M5\nGPU (Metal)",
        "RTX 3090 Ti\nGPU (Vulkan)",
    ]
    tile_p99  = [321.5, 254.2, 73.8, 19.6]
    heavy_p50 = [2426.4, 2782.7, 31.1, 13.0]
    tile_rps  = [89, 94, 302, 1170]
    heavy_rps = [1, 1, 60, 106]

    x2 = np.arange(len(configs))
    w = 0.34
    ax2.bar(x2 - w / 2, tile_p99,  width=w, color=C_RTX, alpha=0.9, label="Live Map Tiles (12c) — p99 latency")
    ax2.bar(x2 + w / 2, heavy_p50, width=w, color=C_TR,  alpha=0.9, label="Heavy 2048² Analytics (2c) — p50 latency")

    ax2.set_yscale("log")
    ax2.set_ylim(5, 8000)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(configs, fontsize=8.5)
    ax2.set_ylabel("Latency (ms, log₁₀)")
    ax2.set_title("B. Noisy-Neighbor Isolation (12 Live Tile + 2× 2048² Clients)")
    ax2.grid(True, axis="y", which="both", alpha=0.35)

    for i in range(len(configs)):
        ax2.text(x2[i] - w / 2, tile_p99[i] * 1.12, f"{tile_p99[i]:.1f}ms\n({tile_rps[i]} r/s)",
                 ha="center", va="bottom", fontsize=7.8, color="#79c0ff", fontweight="bold")
        ax2.text(x2[i] + w / 2, heavy_p50[i] * 1.12, f"{heavy_p50[i]:.0f}ms\n({heavy_rps[i]} r/s)",
                 ha="center", va="bottom", fontsize=7.8, color="#ffa657", fontweight="bold")

    ax2.legend(loc="upper right", fontsize=8.3, framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ASSETS / "bench-workload-contention.png", dpi=200)
    plt.close(fig)


def render_s3_cache_scaling() -> None:
    """Graph 3: Real S3 COGs compute cost vs decoded block-cache hit rate (cogzilla vs TiTiler)."""
    fig, ax1 = plt.subplots(figsize=(11.8, 6.2), dpi=200)
    fig.suptitle(
        "Real S3 COGs: Compute Cost vs. Block-Cache Hit Rate (Apple M5, 10 Cores)",
        fontsize=13.5, fontweight="bold", y=0.98, color="#f0f6fc",
    )

    hit = np.linspace(0.0, 1.0, 21)
    pct = hit * 100.0

    def blend(miss: float, hot: float) -> np.ndarray:
        return (1.0 - hit) * miss + hit * hot

    tt_mix  = blend(56.0, 55.3)
    tt_stac = blend(116.0, 120.0)

    cz_mix  = blend(13.2, 1.7)
    cz_stac = blend(25.4, 2.8)
    cz_tci  = blend(14.4, 2.1)
    cz_3dep = blend(13.7, 1.1)
    cz_s2   = blend(8.6,  1.6)

    ax1.plot(pct, tt_stac, ":",  color="#ffa198", lw=1.6, alpha=0.85, label="TiTiler — 3-asset STAC RGB (116 → 120 CPU-s)")
    ax1.plot(pct, tt_mix,  "--", color=C_TT,      lw=2.8,             label="TiTiler — Equal Mix (56.0 → 55.3 CPU-s)")

    ax1.plot(pct, cz_stac, "-s", color=C_TR,      lw=1.9, ms=4.5, markevery=4, label="cogzilla — 3-asset STAC RGB (25.4 → 2.8 CPU-s)")
    ax1.plot(pct, cz_tci,  "-^", color=C_M5C,     lw=1.8, ms=4.5, markevery=4, label="cogzilla — Sentinel-2 TCI JPEG (14.4 → 2.1 CPU-s)")
    ax1.plot(pct, cz_3dep, "-D", color="#39d353", lw=1.8, ms=4.2, markevery=4, label="cogzilla — USGS 3DEP LZW DEM (13.7 → 1.1 CPU-s)")
    ax1.plot(pct, cz_s2,   "-v", color="#79c0ff", lw=1.8, ms=4.5, markevery=4, label="cogzilla — Sentinel-2 B08 Viridis (8.6 → 1.6 CPU-s)")
    ax1.plot(pct, cz_mix,  "-o", color=C_RTX,     lw=3.0, ms=6.0, markevery=4, label="cogzilla — Equal Mix (13.2 → 1.7 CPU-s)")

    ax1.fill_between(pct, cz_mix, tt_mix, color=C_RTX, alpha=0.10)

    ax1.set_yscale("log")
    ax1.set_ylim(0.7, 200)
    ax1.set_xlim(-2, 102)
    ax1.set_xlabel("Decoded Block-Cache Hit Rate (%)   [0% = Cold CDN Origin  →  100% = Warm Working Set]")
    ax1.set_ylabel("CPU-Seconds per 1,000 Tiles (log₁₀, lower is better)")
    ax1.grid(True, which="both", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(1000.0 / 0.7, 1000.0 / 200.0)
    ax2.set_ylabel("Max Throughput per 1 Busy Core (tiles/s, higher is better)", color="#8b949e")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.0f}"))

    ax1.annotate(
        "4.3× less CPU (0% cache hit / cold CDN origin)\n11 vs 47 c7g.xlarge for 1,000 tiles/s",
        xy=(0, 13.2), xytext=(12, 28),
        arrowprops=dict(arrowstyle="->", color=C_RTX, lw=1.4),
        fontsize=8.8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#0d1117", ec=C_RTX, alpha=0.92),
    )
    ax1.annotate(
        "32× less CPU (588 vs 18 tiles/s per core)\n(100% block-cache hit)",
        xy=(100, 1.7), xytext=(52, 0.9),
        arrowprops=dict(arrowstyle="->", color=C_M5G, lw=1.4),
        fontsize=8.8, fontweight="bold", color="#7ee787",
        bbox=dict(boxstyle="round,pad=0.35", fc="#0d1117", ec=C_M5G, alpha=0.92),
    )

    ax1.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
        fontsize=8.3, framealpha=0.95,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(ASSETS / "bench-s3-cache-scaling.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    render_concurrency_overload()
    render_workload_contention()
    render_s3_cache_scaling()
    print(f"Wrote 3 benchmark charts to {ASSETS}")
