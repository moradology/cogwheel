#!/usr/bin/env python3
"""
Live HTTP Differential Oracle: TiTiler vs. cogwheel-server.

Spins up BOTH `cogwheel-server` (--release) and `TiTiler` (uvicorn) on loopback,
sends identical HTTP requests across the full `/cog/*` API surface, verifies
JSON schema/numerical parity and PNG visual PSNR/mask agreement, and measures
per-endpoint warm latency speedups.
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
import rasterio
import rasterio.shutil as rio_shutil
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def ensure_dtype_fixtures() -> dict[str, tuple[str, str]]:
    """Generate 1024x1024 DEFLATE COG fixtures for ALL 8 standard GDAL numeric dtypes."""
    out_dir = FIXTURES / "dtypes"
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.linspace(0.0, 4.0 * np.pi, 1024, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 4.0 * np.pi, 1024, dtype=np.float64)[None, :]
    norm = 0.5 + 0.45 * np.sin(x * 0.9) * np.cos(y * 0.7)  # [0.05, 0.95]
    transform = from_bounds(-13650000.0, 4500000.0, -13550000.0, 4600000.0, 1024, 1024)
    crs = CRS.from_epsg(3857)

    specs = [
        ("uint8", (norm * 240 + 5).astype(np.uint8), 2, "0,255"),
        ("int8", (norm * 200 - 100).astype(np.int8), 2, "-100,100"),
        ("uint16", (norm * 50000 + 1000).astype(np.uint16), 2, "0,60000"),
        ("int16", (norm * 20000 - 10000).astype(np.int16), 2, "-10000,10000"),
        ("uint32", (norm * 100000 + 500).astype(np.uint32), 2, "0,110000"),
        ("int32", (norm * 100000 - 50000).astype(np.int32), 2, "-50000,50000"),
        ("float32", (norm * 1000.0).astype(np.float32), 3, "0,1000"),
        ("float64", (norm * 1000.0).astype(np.float64), 3, "0,1000"),
    ]
    paths = {}
    for dtype_name, arr, pred, rescale in specs:
        p = out_dir / f"dtype_{dtype_name}.tif"
        if not p.exists():
            with rasterio.open(
                p,
                "w",
                driver="GTiff",
                height=1024,
                width=1024,
                count=1,
                dtype=dtype_name,
                crs=crs,
                transform=transform,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="DEFLATE",
                predictor=pred,
            ) as dst:
                dst.write(arr, 1)
        paths[dtype_name] = (str(p.resolve()), rescale)
    return paths

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "benches" / "fixtures"
RESULTS = ROOT / "benches" / "results"
COGWHEEL_BIN = ROOT / "target" / "release" / "cogwheel-server"

PORT_CW = 18801
PORT_TT = 18802


def wait_ready(url: str, timeout_s: float = 10.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(url, timeout=0.5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"Server at {url} did not become ready in {timeout_s}s")


def time_request(client: httpx.Client, method: str, url: str, json_body=None, iters: int = 15):
    # Warmup
    for _ in range(2):
        r = client.request(method, url, json=json_body)
        assert r.status_code == 200, f"{method} {url} -> {r.status_code}: {r.text[:200]}"
    samples = []
    last = None
    for _ in range(iters):
        t0 = time.perf_counter()
        last = client.request(method, url, json=json_body)
        samples.append((time.perf_counter() - t0) * 1e3)
        assert last.status_code == 200
    samples.sort()
    return last, samples[len(samples) // 2]


def compare_pngs(cw_bytes: bytes, tt_bytes: bytes) -> str:
    im_cw = np.array(Image.open(io.BytesIO(cw_bytes)).convert("RGBA"), dtype=np.float32)
    im_tt = np.array(Image.open(io.BytesIO(tt_bytes)).convert("RGBA"), dtype=np.float32)
    if im_cw.shape != im_tt.shape:
        return f"SHAPE DIFF cw={im_cw.shape} vs tt={im_tt.shape}"
    a_cw = im_cw[:, :, 3] > 0
    a_tt = im_tt[:, :, 3] > 0
    mask_pct = float(np.mean(a_cw == a_tt)) * 100.0
    both = a_cw & a_tt
    if np.any(both):
        diff = im_cw[:, :, :3][both] - im_tt[:, :, :3][both]
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff * diff))
        psnr = 99.0 if mse < 1e-6 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    else:
        mae, psnr = 0.0, 99.0
    return f"shape={im_cw.shape[:2]} mask={mask_pct:.2f}% MAE={mae:.2f} PSNR={psnr:.1f}dB"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not COGWHEEL_BIN.exists():
        print("Building cogwheel-server --release...")
        subprocess.run(["cargo", "build", "--release", "-p", "cogwheel-server"], cwd=ROOT, check=True)

    # Write TiTiler app mounted at /cog, /stac, /colorMaps, /tileMatrixSets
    tt_app = RESULTS / "_titiler_oracle_app.py"
    tt_app.write_text(
        "from fastapi import FastAPI\n"
        "from rio_tiler.io import STACReader\n"
        "from titiler.core.factory import TilerFactory, MultiBaseTilerFactory, ColorMapFactory, TMSFactory\n"
        "app = FastAPI()\n"
        "cog = TilerFactory()\n"
        "stac = MultiBaseTilerFactory(reader=STACReader)\n"
        "cmaps = ColorMapFactory()\n"
        "tms = TMSFactory()\n"
        "app.include_router(cog.router, prefix='/cog')\n"
        "app.include_router(stac.router, prefix='/stac')\n"
        "app.include_router(cmaps.router)\n"
        "app.include_router(tms.router)\n"
    )

    cw_proc = subprocess.Popen(
        [str(COGWHEEL_BIN), "--bind", f"127.0.0.1:{PORT_CW}", "--cache-mb", "512"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tt_env = os.environ.copy()
    tt_env.update({
        "GDAL_CACHEMAX": "512",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "TRUE",
    })
    tt_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "_titiler_oracle_app:app",
            "--app-dir",
            str(RESULTS),
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT_TT),
            "--workers",
            "1",
            "--log-level",
            "error",
        ],
        env=tt_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_ready(f"http://127.0.0.1:{PORT_CW}/healthz")
        wait_ready(f"http://127.0.0.1:{PORT_TT}/openapi.json")

        sat = str((FIXTURES / "satellite_utm10n_rgb_zstd_p3.tif").resolve())
        merc = str((FIXTURES / "mercator_4k_zstd_p3.tif").resolve())
        cw_base = f"http://127.0.0.1:{PORT_CW}"
        tt_base = f"http://127.0.0.1:{PORT_TT}"

        poly_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-122.80, 37.60],
                    [-122.50, 37.60],
                    [-122.50, 37.90],
                    [-122.80, 37.90],
                    [-122.80, 37.60],
                ]],
            },
            "properties": {},
        }

        cases = [
            ("GET /cog/info", "GET", f"/cog/info?url={sat}", None, "json_info"),
            ("GET /cog/info.geojson", "GET", f"/cog/info.geojson?url={sat}", None, "json_geojson"),
            ("GET /cog/WebMercatorQuad/tilejson.json", "GET", f"/cog/WebMercatorQuad/tilejson.json?url={sat}", None, "json_tj"),
            ("GET /cog/tiles/WebMercatorQuad/12/652/1583.png (Sat RGB)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&resampling=bilinear", None, "png"),
            ("GET /cog/tiles/.../12/652/1583.jpg (JPEG output)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.jpg?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&resampling=bilinear", None, "png"),
            ("GET /cog/tiles/...?return_mask=false (Opaque RGB)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&return_mask=false", None, "png"),
            ("GET /cog/tiles/...?color_formula=Gamma RGB 1.5", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&color_formula=Gamma%20RGB%201.5", None, "png"),
            ("GET /cog/tiles/WorldCRS84Quad/10/325/297.png (CRS84)", "GET", f"/cog/tiles/WorldCRS84Quad/10/325/297.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "png"),
            ("GET /cog/tiles/...?expression=(b1-b2)/(b1+b2)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&expression=(b1-b2)/(b1%2Bb2)&rescale=-0.5,0.5&colormap_name=viridis&resampling=bilinear", None, "png"),
            ("GET /cog/tiles/WebMercatorQuad/12/657/1582.png (Mercator)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&rescale=0,1000&colormap_name=viridis&resampling=bilinear", None, "png"),
            ("GET /cog/point/-122.65,37.773", "GET", f"/cog/point/-122.65,37.773?url={sat}", None, "json_point"),
            ("GET /cog/point (coord_crs=EPSG:3857)", "GET", f"/cog/point/-13653336.0,4547451.0?url={sat}&coord_crs=EPSG:3857", None, "json_point"),
            ("GET /cog/statistics?max_size=512", "GET", f"/cog/statistics?url={sat}&max_size=512", None, "json_stats"),
            ("POST /cog/statistics (GeoJSON Polygon clip)", "POST", f"/cog/statistics?url={sat}&max_size=512", poly_feature, "json_post_stats"),
            ("GET /cog/preview/256x256.png", "GET", f"/cog/preview/256x256.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "png"),
            ("GET /cog/bbox/.../256x256.png", "GET", f"/cog/bbox/-122.8,37.6,-122.5,37.9/256x256.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "png"),
            ("POST /cog/feature/256x256.png", "POST", f"/cog/feature/256x256.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", poly_feature, "png"),
        ]

        # Add ALL 8 GDAL numeric data types (uint8, int8, uint16, int16, uint32, int32, float32, float64)
        dtype_paths = ensure_dtype_fixtures()
        for dt_name, (dt_path, dt_rescale) in dtype_paths.items():
            cases.append((
                f"DTYPE [{dt_name:7s}] /cog/tiles/WebMercatorQuad/12/657/1582.png",
                "GET",
                f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={dt_path}&rescale={dt_rescale}&colormap_name=viridis&resampling=bilinear",
                None,
                "png",
            ))

        # Add /colorMaps, /tileMatrixSets, and /stac/* differential parity checks
        stac_item = str((FIXTURES / "landsat_c2_stac.json").resolve())
        cases.extend([
            ("GET /colorMaps", "GET", "/colorMaps", None, "json_cmaps"),
            ("GET /colorMaps/viridis (256-entry exact LUT)", "GET", "/colorMaps/viridis", None, "json_cmap_lut"),
            ("GET /colorMaps/viridis?f=png", "GET", "/colorMaps/viridis?f=png", None, "png"),
            ("GET /tileMatrixSets", "GET", "/tileMatrixSets", None, "json_tms_list"),
            ("GET /tileMatrixSets/WebMercatorQuad", "GET", "/tileMatrixSets/WebMercatorQuad", None, "json_tms_one"),
            ("GET /stac/assets", "GET", f"/stac/assets?url={stac_item}", None, "json_stac_assets"),
            ("GET /stac/info?assets=red&assets=green", "GET", f"/stac/info?url={stac_item}&assets=red&assets=green", None, "json_stac_info"),
            ("GET /stac/tiles/.../12/652/1583.png (RGB 3-asset)", "GET", f"/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_item}&assets=red&assets=green&assets=blue&rescale=7000,20000&resampling=bilinear", None, "png"),
            ("GET /stac/tiles/...?expression=(b1-b2)/(b1+b2)", "GET", f"/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_item}&assets=red&assets=green&expression=(b1-b2)/(b1%2Bb2)&rescale=-0.5,0.5&colormap_name=rdylgn&resampling=bilinear", None, "png"),
            ("GET /stac/WebMercatorQuad/tilejson.json", "GET", f"/stac/WebMercatorQuad/tilejson.json?url={stac_item}&assets=red&assets=green&assets=blue", None, "json_tj"),
            ("GET /stac/point/-122.65,37.773 (3-asset)", "GET", f"/stac/point/-122.65,37.773?url={stac_item}&assets=red&assets=green&assets=blue", None, "json_point"),
            ("GET /stac/asset_statistics?assets=red&assets=green", "GET", f"/stac/asset_statistics?url={stac_item}&assets=red&assets=green&max_size=256", None, "json_stac_asset_stats"),
            ("GET /stac/statistics?assets=red&assets=green", "GET", f"/stac/statistics?url={stac_item}&assets=red&assets=green&max_size=256", None, "json_stats"),
            ("GET /stac/preview/256x256.png (3-asset RGB)", "GET", f"/stac/preview/256x256.png?url={stac_item}&assets=red&assets=green&assets=blue&rescale=7000,20000", None, "png"),
        ])

        print("\n=== Live HTTP Differential Oracle: cogwheel vs. TiTiler ===")
        print(f"{'Endpoint':54s} | {'cogwheel':>9s} | {'TiTiler':>9s} | {'Speedup':>8s} | Parity vs TiTiler")
        print("-" * 135)

        summary = []
        with httpx.Client(timeout=30.0) as client:
            for label, method, path, body, kind in cases:
                r_cw, ms_cw = time_request(client, method, cw_base + path, body, iters=12)
                r_tt, ms_tt = time_request(client, method, tt_base + path, body, iters=12)
                speedup = ms_tt / ms_cw

                if kind == "png":
                    parity = compare_pngs(r_cw.content, r_tt.content)
                elif kind == "json_info":
                    j_cw, j_tt = r_cw.json(), r_tt.json()
                    b_err = max(abs(a - b) for a, b in zip(j_cw["bounds"], j_tt["bounds"]))
                    parity = f"dtype={j_cw['dtype']}=={j_tt['dtype']} count={j_cw['count']}=={j_tt['count']} bounds_err={b_err:.5f} deg"
                elif kind == "json_geojson":
                    j_cw, j_tt = r_cw.json(), r_tt.json()
                    parity = f"type={j_cw['type']}=={j_tt['type']} geom={j_cw['geometry']['type']}=={j_tt['geometry']['type']}"
                elif kind == "json_tj":
                    j_cw, j_tt = r_cw.json(), r_tt.json()
                    b_err = max(abs(a - b) for a, b in zip(j_cw["bounds"], j_tt["bounds"]))
                    parity = f"tilejson={j_cw['tilejson']}=={j_tt['tilejson']} zoom=({j_cw['minzoom']},{j_cw['maxzoom']}) vs ({j_tt['minzoom']},{j_tt['maxzoom']}) bounds_err={b_err:.5f}"
                elif kind == "json_point":
                    j_cw, j_tt = r_cw.json(), r_tt.json()
                    v_err = max(abs(a - b) for a, b in zip(j_cw["values"], j_tt["values"]))
                    parity = f"max_val_err={v_err:.7f} bands={j_cw['band_names']}"
                elif kind == "json_stats":
                    j_cw, j_tt = r_cw.json(), r_tt.json()
                    m_err = abs(j_cw["b1"]["mean"] - j_tt["b1"]["mean"])
                    s_err = abs(j_cw["b1"]["std"] - j_tt["b1"]["std"])
                    parity = f"b1.mean_err={m_err:.5f} b1.std_err={s_err:.5f}"
                elif kind == "json_post_stats":
                    j_cw, j_tt = r_cw.json()["properties"]["statistics"], r_tt.json()["properties"]["statistics"]
                    m_err = abs(j_cw["b1"]["mean"] - j_tt["b1"]["mean"])
                    s_err = abs(j_cw["b1"]["std"] - j_tt["b1"]["std"])
                    parity = f"poly b1.mean_err={m_err:.5f} b1.std_err={s_err:.5f}"
                elif kind == "json_cmaps":
                    j_cw = {c["id"] for c in r_cw.json()["colormaps"]}
                    j_tt = {c["id"] for c in r_tt.json()["colormaps"]}
                    overlap = len(j_cw & j_tt)
                    parity = f"cogwheel={len(j_cw)} titiler={len(j_tt)} overlap={overlap}/{len(j_cw)}"
                elif kind == "json_cmap_lut":
                    j_cw, j_tt = r_cw.json(), r_tt.json()
                    exact = sum(1 for k in j_cw if j_cw[k] == j_tt.get(k))
                    parity = f"exact_entries={exact}/256"
                elif kind == "json_tms_list":
                    ids_cw = {t["id"] for t in r_cw.json()["tileMatrixSets"]}
                    ids_tt = {t["id"] for t in r_tt.json()["tileMatrixSets"]}
                    parity = f"ids={sorted(ids_cw)} subset_of_tt={ids_cw.issubset(ids_tt)}"
                elif kind == "json_tms_one":
                    j_cw, j_tt = r_cw.json(), r_tt.json()
                    parity = f"id={j_cw['id']}=={j_tt['id']} matrices={len(j_cw['tileMatrices'])}=={len(j_tt['tileMatrices'])}"
                elif kind == "json_stac_assets":
                    j_cw, j_tt = set(r_cw.json()), set(r_tt.json())
                    parity = f"assets={sorted(j_cw)} exact_match={j_cw == j_tt}"
                elif kind == "json_stac_info":
                    j_cw, j_tt = r_cw.json(), r_tt.json()
                    parity = f"keys={sorted(j_cw.keys())}=={sorted(j_tt.keys())}"
                elif kind == "json_stac_asset_stats":
                    j_cw, j_tt = r_cw.json(), r_tt.json()
                    m_err = abs(j_cw["red"]["b1"]["mean"] - j_tt["red"]["b1"]["mean"])
                    parity = f"keys={sorted(j_cw.keys())}=={sorted(j_tt.keys())} red.b1.mean_err={m_err:.2f}"
                else:
                    parity = "ok"

                print(f"{label:54s} | {ms_cw:6.2f} ms | {ms_tt:6.2f} ms | {speedup:7.1f}x | {parity}")
                summary.append({
                    "endpoint": label,
                    "cogwheel_ms": ms_cw,
                    "titiler_ms": ms_tt,
                    "speedup": speedup,
                    "parity": parity,
                })

        (RESULTS / "titiler_diff_oracle.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSaved differential oracle report to {RESULTS / 'titiler_diff_oracle.json'}")
    finally:
        cw_proc.terminate()
        tt_proc.terminate()
        cw_proc.wait(timeout=5)
        tt_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
