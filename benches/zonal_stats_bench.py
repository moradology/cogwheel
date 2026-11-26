#!/usr/bin/env python3
"""Zonal statistics latency benchmark for POST /cog/statistics.

Measures the 1M-pixel aligned square, 10k-vertex circle, and small rotated quad
on ``satellite_utm10n_rgb_zstd_p3.tif`` (bidx=1, max_size=0, coord_crs=EPSG:32610)
to guard against regressions in ``zonal.rs``.

Usage:
    # Auto-spawn ./target/release/cogzilla-server:
    benches/.venv/bin/python benches/zonal_stats_bench.py

    # Or benchmark an already-running server:
    benches/.venv/bin/python benches/zonal_stats_bench.py --url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import rasterio


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = REPO_ROOT / "benches" / "fixtures" / "satellite_utm10n_rgb_zstd_p3.tif"
DEFAULT_BIN = REPO_ROOT / "target" / "release" / "cogzilla-server"


def build_geometries(fixture: Path) -> list[tuple[str, dict]]:
    with rasterio.open(fixture) as ds:
        t = ds.transform

    def utm(pts: list[tuple[float, float]]) -> list[list[float]]:
        return [list(t * p) for p in pts]

    def feat(ring: list[list[float]]) -> dict:
        return {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
        }

    big_1m = feat(utm([(200, 200), (1200, 200), (1200, 1200), (200, 1200)]))
    angles = np.linspace(0.0, 2.0 * np.pi, 10_000, endpoint=False)
    circle_10k = feat(
        utm([(1024.0 + 600.0 * float(np.cos(a)), 1024.0 + 600.0 * float(np.sin(a))) for a in angles])
    )
    small_quad = feat(utm([(600.3, 600.6), (700.1, 620.4), (680.6, 710.2), (590.9, 690.5)]))
    # 1,000 small 20x20-px polygons tiled across the 2048x2048 scene (BB1 FeatureCollection bench)
    fc_1000 = {
        "type": "FeatureCollection",
        "features": [
            feat(
                utm(
                    [
                        (100.0 + (i % 40) * 45.0, 100.0 + (i // 40) * 60.0),
                        (120.0 + (i % 40) * 45.0, 100.0 + (i // 40) * 60.0),
                        (120.0 + (i % 40) * 45.0, 120.0 + (i // 40) * 60.0),
                        (100.0 + (i % 40) * 45.0, 120.0 + (i // 40) * 60.0),
                    ]
                )
            )
            for i in range(1000)
        ],
    }
    return [("1M", big_1m), ("circle", circle_10k), ("small", small_quad), ("fc1000", fc_1000)]


def wait_ready(client: httpx.Client, base_url: str, timeout_s: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            r = client.get(f"{base_url}/healthz", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server at {base_url} did not become ready within {timeout_s}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark POST /cog/statistics zonal performance.")
    ap.add_argument("--url", default=None, help="Existing server base URL (spawns release binary if omitted)")
    ap.add_argument("--bin", default=str(DEFAULT_BIN), help="Path to cogzilla-server binary")
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="Path to satellite_utm10n_rgb_zstd_p3.tif")
    ap.add_argument("--iters", type=int, default=40, help="Measured iterations per polygon")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup iterations per polygon")
    ap.add_argument("--port", type=int, default=18995, help="Port when auto-spawning server")
    ap.add_argument("--max-1m-ms", type=float, default=None, help="Optional median ceiling (ms) for 1M square")
    ap.add_argument("--max-circle-ms", type=float, default=None, help="Optional median ceiling (ms) for 10k circle")
    args = ap.parse_args()

    fixture = Path(args.fixture).resolve()
    if not fixture.exists():
        print(f"ERROR: fixture not found: {fixture}", file=sys.stderr)
        return 2

    geoms = build_geometries(fixture)
    proc = None
    base_url = args.url
    try:
        if base_url is None:
            bind = f"127.0.0.1:{args.port}"
            base_url = f"http://{bind}"
            proc = subprocess.Popen(
                [args.bin, "--bind", bind],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
            )
        client = httpx.Client(timeout=30.0)
        wait_ready(client, base_url)
        endpoint = f"{base_url}/cog/statistics?url={fixture}&bidx=1&max_size=0&coord_crs=EPSG:32610"

        medians: dict[str, float] = {}
        parts: list[str] = []
        for name, body in geoms:
            iters = min(args.iters, 10) if name == "fc1000" else args.iters
            warmup = min(args.warmup, 2) if name == "fc1000" else args.warmup
            for _ in range(warmup):
                r = client.post(endpoint, json=body)
                r.raise_for_status()
            times_ms: list[float] = []
            for _ in range(iters):
                t0 = time.perf_counter()
                r = client.post(endpoint, json=body)
                r.raise_for_status()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            med = float(np.median(times_ms))
            medians[name] = med
            parts.append(f"{name} min {min(times_ms):.1f} med {med:.1f} ms")
        print(" | ".join(parts))

        rc = 0
        if args.max_1m_ms is not None and medians["1M"] > args.max_1m_ms:
            print(f"FAIL: 1M median {medians['1M']:.1f} ms > {args.max_1m_ms:.1f} ms", file=sys.stderr)
            rc = 1
        if args.max_circle_ms is not None and medians["circle"] > args.max_circle_ms:
            print(f"FAIL: circle median {medians['circle']:.1f} ms > {args.max_circle_ms:.1f} ms", file=sys.stderr)
            rc = 1
        return rc
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
