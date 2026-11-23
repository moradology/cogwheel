#!/usr/bin/env python3
"""
Projection oracle: cogzilla vs. PROJ on real GeoTIFFs.

Writes a small tiled GeoTIFF in each CRS below (band 1 = column, band 2 = row),
asks a running cogzilla-server for the pixel at lon/lat positions that PROJ says
fall on known pixel centres, and reports how far off the answer is in metres.

Each CRS must end up in one of two states: answered correctly, or refused with
an error. A wrong answer served with a 200 is the failure this script exists
to catch.

Requires a release build: cargo build --release -p cogzilla-server

    python benches/projection_oracle.py [--keep DIR]
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "cogzilla-server"
PORT = 18811
SIZE = 512

# (label, CRS definition, lon, lat of raster centre, pixel size in CRS units)
CASES = [
    # --- hard-coded EPSG codes -------------------------------------------------
    ("EPSG:3857 Web Mercator", "EPSG:3857", -122.3, 47.6, 10),
    ("EPSG:4326 WGS84", "EPSG:4326", 12.5, 41.9, 0.0001),
    ("EPSG:4269 NAD83 geographic", "EPSG:4269", -100.0, 40.0, 0.0001),
    ("EPSG:32610 UTM 10N", "EPSG:32610", -122.3, 47.6, 10),
    ("EPSG:32633 UTM 33N", "EPSG:32633", 15.0, 60.0, 10),
    ("EPSG:32633 UTM 33N at zone edge", "EPSG:32633", 18.4, 0.5, 10),
    ("EPSG:32633 UTM 33N at 83N", "EPSG:32633", 15.0, 83.0, 10),
    ("EPSG:32755 UTM 55S", "EPSG:32755", 147.0, -42.0, 10),
    ("EPSG:32601 UTM 1N near antimeridian", "EPSG:32601", -179.95, 52.0, 10),
    ("EPSG:32660 UTM 60N near antimeridian", "EPSG:32660", 179.95, 52.0, 10),
    ("EPSG:26915 NAD83 UTM 15N", "EPSG:26915", -93.0, 45.0, 10),
    ("EPSG:5070 CONUS Albers", "EPSG:5070", -96.0, 39.0, 30),
    ("EPSG:5070 CONUS Albers, Maine", "EPSG:5070", -68.5, 46.5, 30),
    ("EPSG:3310 California Albers", "EPSG:3310", -120.0, 37.0, 30),
    ("EPSG:2154 Lambert-93", "EPSG:2154", 2.35, 48.85, 10),
    ("EPSG:3413 NSIDC north polar", "EPSG:3413", -45.0, 75.0, 100),
    ("EPSG:3413 north polar, far side", "EPSG:3413", 135.0, 80.0, 100),
    ("EPSG:3031 Antarctic polar", "EPSG:3031", 166.7, -77.8, 100),
    # --- common codes that are not hard-coded -----------------------------------
    ("EPSG:25832 ETRS89 UTM 32N", "EPSG:25832", 9.0, 50.0, 10),
    ("EPSG:28355 GDA94 MGA 55", "EPSG:28355", 147.0, -37.0, 10),
    ("EPSG:7855 GDA2020 MGA 55", "EPSG:7855", 147.0, -37.0, 10),
    ("EPSG:31983 SIRGAS 2000 UTM 23S", "EPSG:31983", -45.0, -23.0, 10),
    ("EPSG:2193 NZTM2000", "EPSG:2193", 174.8, -41.3, 10),
    ("EPSG:3006 SWEREF99 TM", "EPSG:3006", 18.0, 59.3, 10),
    ("EPSG:3978 Canada Atlas Lambert", "EPSG:3978", -95.0, 55.0, 30),
    ("EPSG:3338 Alaska Albers", "EPSG:3338", -150.0, 64.0, 30),
    ("EPSG:3338 Alaska Albers, Aleutians", "EPSG:3338", 175.0, 52.0, 30),
    ("EPSG:3577 Australian Albers", "EPSG:3577", 134.0, -25.0, 30),
    ("EPSG:3005 BC Albers", "EPSG:3005", -125.0, 54.0, 30),
    ("EPSG:3035 ETRS89 LAEA Europe", "EPSG:3035", 10.0, 50.0, 30),
    ("EPSG:3395 World Mercator", "EPSG:3395", 10.0, 50.0, 30),
    ("EPSG:3995 Arctic polar stereographic", "EPSG:3995", 20.0, 78.0, 100),
    ("EPSG:32661 UPS North", "EPSG:32661", 20.0, 86.0, 100),
    ("EPSG:6933 EASE-Grid 2.0 global", "EPSG:6933", 10.0, 50.0, 100),
    ("EPSG:27700 British National Grid", "EPSG:27700", -1.5, 53.0, 10),
    ("EPSG:26710 NAD27 UTM 10N", "EPSG:26710", -122.3, 47.6, 10),
    ("EPSG:4267 NAD27 geographic", "EPSG:4267", -100.0, 40.0, 0.0001),
    ("EPSG:2227 California zone 3 (US feet)", "EPSG:2227", -122.0, 37.5, 30),
    ("EPSG:2263 New York Long Island (US feet)", "EPSG:2263", -73.9, 40.7, 30),
    ("EPSG:26986 Massachusetts mainland (m)", "EPSG:26986", -71.5, 42.3, 10),
    ("EPSG:2056 Swiss LV95", "EPSG:2056", 8.2, 46.8, 10),
    ("EPSG:28992 Dutch RD New", "EPSG:28992", 5.3, 52.1, 10),
    # --- user-defined CRSs (written as GeoKey parameters, not a code) -----------
    ("custom TM, lat_0=49 k=0.9996", "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996 +x_0=400000 +y_0=-100000 +ellps=GRS80 +units=m", -1.5, 53.0, 10),
    ("custom LCC 2SP (Canada-like)", "+proj=lcc +lat_1=49 +lat_2=77 +lat_0=49 +lon_0=-95 +x_0=0 +y_0=0 +ellps=GRS80 +units=m", -95.0, 55.0, 30),
    ("custom LCC 1SP, k_0=0.9998", "+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=2.3 +k_0=0.9998 +x_0=600000 +y_0=200000 +ellps=GRS80 +units=m", 2.35, 48.85, 10),
    ("custom LCC 2SP in US feet", "+proj=lcc +lat_1=38.4 +lat_2=37.1 +lat_0=36.5 +lon_0=-120.5 +x_0=2000000 +y_0=500000 +ellps=GRS80 +units=us-ft", -122.0, 37.5, 30),
    ("custom Albers, southern", "+proj=aea +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=132 +x_0=0 +y_0=0 +ellps=GRS80 +units=m", 134.0, -25.0, 30),
    ("custom polar stereo, lon_0=-45", "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +ellps=WGS84 +units=m", -45.0, 75.0, 100),
    ("custom polar stereo, lon_0=30 lat_ts=60", "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=30 +x_0=0 +y_0=0 +ellps=WGS84 +units=m", 100.0, 80.0, 100),
    ("custom polar stereo south, lon_0=140", "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=140 +x_0=0 +y_0=0 +ellps=WGS84 +units=m", 166.7, -77.8, 100),
    ("custom polar stereo, k_0 at pole", "+proj=stere +lat_0=90 +lon_0=0 +k_0=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84 +units=m", 20.0, 86.0, 100),
    ("custom LAEA on WGS84", "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=WGS84 +units=m", 10.0, 50.0, 30),
    ("custom Mercator on WGS84", "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m", 10.0, 50.0, 30),
    ("MODIS sinusoidal", "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m", 10.0, 50.0, 30),
    ("custom TM on Clarke 1866 (NAD27)", "+proj=tmerc +lat_0=0 +lon_0=-123 +k=0.9996 +x_0=500000 +y_0=0 +datum=NAD27 +units=m", -122.3, 47.6, 10),
    ("custom equirectangular", "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m", 10.0, 50.0, 30),
]


def write_case(path: Path, crs_def: str, lon: float, lat: float, res: float) -> CRS:
    crs = CRS.from_user_input(crs_def)
    xs, ys = transform("EPSG:4326", crs, [lon], [lat])
    cx, cy = xs[0], ys[0]
    if not (math.isfinite(cx) and math.isfinite(cy)):
        raise ValueError("PROJ could not place the raster")
    tr = from_origin(cx - res * SIZE / 2, cy + res * SIZE / 2, res, res)
    cols, rows = np.meshgrid(np.arange(SIZE, dtype=np.float32), np.arange(SIZE, dtype=np.float32))
    with rasterio.open(
        path, "w", driver="GTiff", height=SIZE, width=SIZE, count=2, dtype="float32",
        crs=crs, transform=tr, tiled=True, blockxsize=256, blockysize=256, compress="deflate",
    ) as dst:
        dst.write(cols, 1)
        dst.write(rows, 2)
    return crs


def main() -> int:
    keep = None
    if "--keep" in sys.argv:
        keep = Path(sys.argv[sys.argv.index("--keep") + 1])
        keep.mkdir(parents=True, exist_ok=True)
    tmp = keep or Path(tempfile.mkdtemp(prefix="cogzilla-proj-"))

    env = dict(os.environ)
    env.pop("COGZILLA_ALLOWED_URL_PREFIXES", None)
    srv = subprocess.Popen(
        [str(BIN), "--bind", f"127.0.0.1:{PORT}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )
    base = f"http://127.0.0.1:{PORT}"
    client = httpx.Client(timeout=30)
    for _ in range(50):
        try:
            if client.get(f"{base}/healthz").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)

    rng = np.random.default_rng(7)
    wrong = 0
    print(f"{'CRS':44s} {'result':9s} detail")
    try:
        for i, (label, crs_def, lon, lat, res) in enumerate(CASES):
            path = tmp / f"case_{i:02d}.tif"
            try:
                crs = write_case(path, crs_def, lon, lat, res)
            except Exception as e:  # noqa: BLE001
                print(f"{label:44s} {'SKIP':9s} could not write: {e}")
                continue
            with rasterio.open(path) as src:
                tr = src.transform
                metres_per_unit = src.crs.linear_units_factor[1] if src.crs.is_projected else 111_320.0
            pc = rng.uniform(8, SIZE - 8, 40).astype(int)
            pr = rng.uniform(8, SIZE - 8, 40).astype(int)
            xs = [tr.c + (c + 0.5) * tr.a for c in pc]
            ys = [tr.f + (r + 0.5) * tr.e for r in pr]
            lons, lats = transform(crs, "EPSG:4326", xs, ys)

            status, worst, misses, msg = None, 0.0, 0, ""
            for c, r, lo, la in zip(pc, pr, lons, lats):
                resp = client.get(f"{base}/cog/point/{lo!r},{la!r}", params={"url": str(path)})
                status = resp.status_code
                if status != 200:
                    msg = resp.text[:110]
                    break
                vals = resp.json().get("values") or [None, None]
                if vals[0] is None or vals[1] is None:
                    misses += 1
                    continue
                worst = max(worst, math.hypot(vals[0] - c, vals[1] - r))
            px_m = res * metres_per_unit
            if status == 404:
                # The CRS was accepted but the raster was placed somewhere else.
                wrong += 1
                print(f"{label:44s} {'WRONG':9s} raster misplaced: {msg}")
            elif status != 200:
                print(f"{label:44s} {'REFUSED':9s} {status} {msg}")
            elif misses or worst > 0.5:
                wrong += 1
                detail = f"up to {worst * px_m:,.0f} m off" if worst > 0.5 else ""
                if misses:
                    detail += f" {misses}/40 points returned no data"
                print(f"{label:44s} {'WRONG':9s} {detail.strip()}")
            else:
                print(f"{label:44s} {'ok':9s} within one {px_m:g} m pixel")
    finally:
        srv.terminate()
    print(f"\n{wrong} CRS(s) answered incorrectly with a 200.")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
