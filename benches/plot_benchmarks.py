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
    """Graph 1: Apples-to-apples 512x512 RGB Lanczos3 JPEG concurrency scaling (cogzilla vs
    TiTiler 2.4) & live HTTP AIMD overload tail latency (3x medians at commit 3787b88)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=200)
    fig.suptitle(
        "Concurrency & Overload Scaling — 512×512 RGB Lanczos3 JPEG (3× Medians, Tile Cache Off)",
        fontsize=13.5, fontweight="bold", y=0.98, color="#f0f6fc",
    )

    # Panel A: 512x512 RGB Lanczos3 JPEG (exact same 2048x2048 3-band F32 COG, 726-tile pool)
    c_cz = np.array([1, 4, 16, 32, 128, 256])
    tr_cz_rps = np.array([349, 1381, 5299, 7908, 7934, 8034])
    m5_cz_rps = np.array([675, 2063, 3321, 3288, 3657, 3712])

    c_tt_tr = np.array([1, 4, 16, 32, 48])
    tt_tr_rps = np.array([12.9, 44.8, 155.6, 239.9, 269.9])

    c_tt_m5 = np.array([1, 4, 16, 32])
    tt_m5_rps = np.array([26.2, 72.7, 99.5, 103.8])

    ax1.plot(c_cz, tr_cz_rps, "-o", color=C_RTX, lw=2.6, ms=6.5,
             label="cogzilla — Threadripper 3960X (24C/48T) + RTX 3090 Ti")
    ax1.plot(c_cz, m5_cz_rps, "-s", color=C_M5G, lw=2.5, ms=6.5,
             label="cogzilla — Apple M5 (10C CPU + Metal GPU)")
    ax1.plot(c_tt_tr, tt_tr_rps, "--^", color=C_TR, lw=2.2, ms=6.5,
             label="TiTiler 2.4 — Threadripper 3960X (48 workers, 7.7 GiB RSS)")
    ax1.plot(c_tt_m5, tt_m5_rps, "--D", color=C_TT, lw=2.2, ms=6.0,
             label="TiTiler 2.4 — Apple M5 (10 workers, 1.5 GiB RSS)")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks([1, 4, 16, 32, 48, 128, 256])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_ylim(8, 18000)
    ax1.set_xlabel("Concurrent Clients (log₂)")
    ax1.set_ylabel("Throughput (req/s, log₁₀)")
    ax1.set_title("A. Same-Workload Scaling: cogzilla vs. TiTiler 2.4 (512² Lanczos3)")
    ax1.grid(True, which="both", alpha=0.4)

    ax1.annotate(
        "c = 32: 7,908 vs. 240 req/s (33×)\nM5 c = 32: 3,288 vs. 104 req/s (32×)",
        xy=(32, 7908), xytext=(2.5, 9200),
        arrowprops=dict(arrowstyle="->", color=C_RTX, lw=1.2),
        fontsize=8.4, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#0d1117", ec=C_RTX, alpha=0.92),
    )
    ax1.annotate(
        "c = 1 (1 core)\nM5: 675 vs. 26 r/s (26×)\n3960X: 349 vs. 13 r/s (27×)",
        xy=(1, 675), xytext=(1.6, 260),
        arrowprops=dict(arrowstyle="->", color=C_M5G, lw=1.1),
        fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec=C_M5G, alpha=0.9),
    )
    ax1.legend(loc="lower right", fontsize=8.0, framealpha=0.92)

    # Panel B: Live HTTP AIMD overload protection (3x medians at 3787b88)
    hosts = [
        "Apple M5\nCPU (10C)\n128 clients",
        "Apple M5\nGPU auto\n256 clients",
        "Threadripper\n3960X (48T)\n128 clients",
        "RTX 3090 Ti\n+ 3960X\n256 clients",
    ]
    p99_off = [290.3, 448.9, 33.4, 69.9]
    p99_on  = [97.5,  181.0, 33.8, 56.8]
    p50_on  = [5.5,   10.5,  17.4, 26.9]

    x = np.arange(len(hosts))
    w = 0.28
    ax2.bar(x - w, p99_off, width=w, color="#f85149", alpha=0.85, label="AIMD OFF — 200 OK p99")
    ax2.bar(x,     p99_on,  width=w, color="#58a6ff", alpha=0.92, label="AIMD ON — 200 OK p99")
    ax2.bar(x + w, p50_on,  width=w, color="#3fb950", alpha=0.92, label="AIMD ON — 200 OK p50")

    ax2.set_yscale("log")
    ax2.set_ylim(2, 1200)
    ax2.set_xticks(x)
    ax2.set_xticklabels(hosts, fontsize=8.8)
    ax2.set_ylabel("Latency (ms, log₁₀)")
    ax2.set_title("B. Overload Protection: AIMD ON vs. OFF (target = 100 ms)")
    ax2.grid(True, axis="y", which="both", alpha=0.4)

    for i in range(len(hosts)):
        gain = p99_off[i] / p99_on[i]
        lbl = f"{p99_on[i]:.0f}ms\n({gain:.1f}×)" if gain >= 1.15 else f"{p99_on[i]:.0f}ms"
        ax2.text(x[i], p99_on[i] * 1.14, lbl,
                 ha="center", va="bottom", fontsize=8, color="#79c0ff", fontweight="bold")
        ax2.text(x[i] - w, p99_off[i] * 1.08, f"{p99_off[i]:.0f}ms",
                 ha="center", va="bottom", fontsize=7.8, color="#ffa198")
        ax2.text(x[i] + w, p50_on[i] * 1.10, f"{p50_on[i]:.1f}ms",
                 ha="center", va="bottom", fontsize=7.8, color="#7ee787")

    ax2.text(
        0.5, 0.04,
        "AIMD 429 shed: 0.3–3.4 ms  |  Goodput: 3,508–3,619 r/s (M5) · 7,675–7,834 r/s (3960X)",
        transform=ax2.transAxes, ha="center", fontsize=8.1, color="#c9d1d9",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#30363d"),
    )
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ASSETS / "bench-concurrency-overload.png", dpi=200)
    plt.close(fig)


def render_workload_contention() -> None:
    """Graph 2: Core/worker scaling (256x256 RGB Bilinear PNG, cogzilla vs TiTiler 2.4) &
    live-tile + heavy 2048x2048 contention isolation (3x medians at 3787b88)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=200)
    fig.suptitle(
        "Core / Worker Scaling (256×256 RGB PNG) & Heavy 2048² Contention Isolation (3× Medians)",
        fontsize=13.5, fontweight="bold", y=0.98, color="#f0f6fc",
    )

    # Panel A: HTTP 256x256 RGB Bilinear PNG across core / worker counts (tile_cache=0)
    w_tr = np.array([1, 2, 4, 8, 16, 24, 32, 48])
    cz_tr_256 = np.array([547.4, 1044.9, 1939.2, 3421.4, 5551.8, 6985.4, 7784.4, 7622.1])

    w_tt_tr = np.array([1, 2, 4, 8, 16, 24, 48])
    tt_tr_256 = np.array([45.9, 69.9, 206.5, 267.4, 407.4, 544.5, 951.8])

    w_m5 = np.array([1, 2, 4, 8, 10])
    cz_m5_256 = np.array([1563.0, 2999.7, 4080.3, 5450.3, 7298.0])
    tt_m5_256 = np.array([118.8, 236.4, 245.5, 373.0, 418.6])

    ax1.plot(w_tr, cz_tr_256, "-o", color=C_RTX, lw=2.5, ms=6.5,
             label="cogzilla — Threadripper 3960X (185–428 MiB RSS)")
    ax1.plot(w_m5, cz_m5_256, "-s", color=C_M5G, lw=2.5, ms=6.5,
             label="cogzilla — Apple M5 10C (168–357 MiB RSS)")
    ax1.plot(w_tt_tr, tt_tr_256, "--^", color=C_TR, lw=2.2, ms=6.5,
             label="TiTiler 2.4 — Threadripper 3960X (154–7,021 MiB RSS)")
    ax1.plot(w_m5, tt_m5_256, "--D", color=C_TT, lw=2.2, ms=6.0,
             label="TiTiler 2.4 — Apple M5 (174–1,543 MiB RSS)")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks([1, 2, 4, 8, 10, 16, 24, 32, 48])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_ylim(30, 18000)
    ax1.set_xlabel("Concurrency / TiTiler Worker Processes (log₂)")
    ax1.set_ylabel("Throughput (req/s, log₁₀)")
    ax1.set_title("A. HTTP 256×256 RGB Bilinear PNG Scaling (Tile Cache Off)")
    ax1.grid(True, which="both", alpha=0.4)

    ax1.annotate(
        "M5 10C: 7,298 vs. 419 req/s (17.4×)\n3960X 48w: 7,622 vs. 952 req/s (8.0×)\n16.4× less RAM (428 MiB vs. 7.0 GiB)",
        xy=(10, 7298), xytext=(1.5, 9200),
        arrowprops=dict(arrowstyle="->", color=C_M5G, lw=1.2),
        fontsize=8.2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#0d1117", ec=C_M5G, alpha=0.92),
    )
    ax1.legend(loc="lower right", fontsize=7.9, framealpha=0.92)

    # Panel B: Noisy-neighbor isolation (12 live tile + 2 heavy 2048^2 clients, 3x medians at 3787b88)
    configs = [
        "Apple M5\nCPU-only (10C)",
        "Apple M5\nGPU (Metal)",
        "Threadripper 3960X\nCPU-only (24C/48T)",
        "RTX 3090 Ti\nGPU (Vulkan)",
    ]
    tile_p99  = [7.4,   28.5,  9.0,   9.2]
    heavy_p50 = [120.2, 34.6,  237.7, 46.4]
    tile_rps  = [1958,  1355,  1828,  2183]
    heavy_rps = [11,    56,    7,     45]

    x2 = np.arange(len(configs))
    w = 0.34
    ax2.bar(x2 - w / 2, tile_p99,  width=w, color=C_RTX, alpha=0.9, label="Live Map Tiles (12c) — p99 latency")
    ax2.bar(x2 + w / 2, heavy_p50, width=w, color=C_TR,  alpha=0.9, label="Heavy 2048² Analytics (2c) — p50 latency")

    ax2.set_yscale("log")
    ax2.set_ylim(2, 1200)
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

    ax2.text(
        0.5, 0.04,
        "Heavy 2048² GPU speedup: 5.2× on Apple M5 Metal (56 vs 11 r/s)  |  6.7× on RTX 3090 Ti (45 vs 7 r/s)",
        transform=ax2.transAxes, ha="center", fontsize=8.0, color="#c9d1d9",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#30363d"),
    )
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
