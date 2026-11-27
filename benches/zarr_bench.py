#!/usr/bin/env python3
"""Benchmark cogzilla /zarr/* (cold tile, warm tile, 3D time-step, full-res stats, zonal stats)
and compare against /cog/* on the GeoTIFF twin (plus optional titiler.xarray URL) (#376)."""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path
import json

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "zarr"


def http_get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.status, resp.read()


def http_post(url: str, payload: bytes) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=payload, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read()


def bench_case(name: str, fn, iters: int = 30, warmup: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    p50 = statistics.median(times_ms)
    p95 = times_ms[int(len(times_ms) * 0.95)]
    mean = statistics.mean(times_ms)
    rps = 1000.0 / mean if mean > 0 else 0.0
    print(f"{name:42s} | p50={p50:6.2f} ms | p95={p95:6.2f} ms | mean={mean:6.2f} ms | {rps:7.1f} req/s")
    return {"name": name, "p50_ms": p50, "p95_ms": p95, "mean_ms": mean, "rps": rps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    zarr_twin = urllib.parse.quote(str((FIXTURES_DIR / "geotiff_twin.zarr").resolve()), safe="")
    tif_twin = urllib.parse.quote(str((FIXTURES_DIR / "geotiff_twin.tif").resolve()), safe="")
    time_3d = urllib.parse.quote(str((FIXTURES_DIR / "time_3d_noleap.zarr").resolve()), safe="")
    sharded = urllib.parse.quote(str((FIXTURES_DIR / "v3_sharded.zarr").resolve()), safe="")

    poly = json.dumps(
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-122.4, 37.6], [-122.0, 37.6], [-122.0, 37.9], [-122.4, 37.9], [-122.4, 37.6]
                ]],
            },
            "properties": {},
        }
    ).encode()

    print(f"Benchmarking {base} ({args.iters} iters per case):")
    print("-" * 100)

    def clear_cache():
        req = urllib.request.Request(f"{base}/admin/clear-cache", method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=10):
            pass

    z_tile = f"{base}/zarr/tiles/WebMercatorQuad/10/163/395.png?url={zarr_twin}&rescale=100,900&colormap_name=viridis&resampling=bilinear"
    c_tile = f"{base}/cog/tiles/WebMercatorQuad/10/163/395.png?url={tif_twin}&rescale=100,900&colormap_name=viridis&resampling=bilinear"
    t_tile = f"{base}/zarr/tiles/WebMercatorQuad/9/81/197.png?url={time_3d}&sel=time=2000-03-01&rescale=100,900"
    s_tile = f"{base}/zarr/tiles/WebMercatorQuad/9/81/197.png?url={sharded}&rescale=100,900"

    bench_case(
        "zarr cold tile (clear-cache + render)",
        lambda: (clear_cache(), http_get(z_tile)),
        iters=min(args.iters, 20),
        warmup=2,
    )
    bench_case("zarr warm tile (tile cache hit)", lambda: http_get(z_tile), iters=args.iters)
    bench_case("cog  warm tile (geotiff_twin.tif)", lambda: http_get(c_tile), iters=args.iters)
    bench_case("zarr 3D time-step tile (sel=2000-03-01)", lambda: http_get(t_tile), iters=args.iters)
    bench_case("zarr v3 sharded tile (cold+ranged)", lambda: (clear_cache(), http_get(s_tile)), iters=min(args.iters, 20), warmup=2)
    bench_case("zarr GET /statistics (full-res 512x512)", lambda: http_get(f"{base}/zarr/statistics?url={zarr_twin}"), iters=min(args.iters, 20))
    bench_case("cog  GET /statistics (full-res 512x512)", lambda: http_get(f"{base}/cog/statistics?url={tif_twin}&max_size=0"), iters=min(args.iters, 20))
    bench_case("zarr POST /statistics (exact zonal)", lambda: http_post(f"{base}/zarr/statistics?url={zarr_twin}", poly), iters=min(args.iters, 20))
    bench_case("cog  POST /statistics (exact zonal)", lambda: http_post(f"{base}/cog/statistics?url={tif_twin}&max_size=0", poly), iters=min(args.iters, 20))


if __name__ == "__main__":
    main()
