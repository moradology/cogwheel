#!/usr/bin/env python3
"""
Live HTTP Differential Oracle: TiTiler 2.4 vs. cogzilla-server.

Spins up BOTH `cogzilla-server` (--release) and stock `TiTiler` (uvicorn) on loopback,
sends identical HTTP requests across the full `/cog/*`, `/stac/*`, `/algorithms/*`,
`/colorMaps/*`, and `/tileMatrixSets/*` API surface, asserts status codes, headers,
JSON schema/numerical parity, XML well-formedness, GeoTIFF/NPY structure, and
visual PNG/JPEG/WebP mask agreement & PSNR, and reports warm latency speedups.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "benches" / "fixtures"
RESULTS = ROOT / "benches" / "results"
COGZILLA_BIN = ROOT / "target" / "release" / "cogzilla-server"

PORT_CW = 18801
PORT_TT = 18802


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


def wait_ready(url: str, timeout_s: float = 15.0):
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


def time_request(client: httpx.Client, method: str, url: str, json_body=None, iters: int = 5):
    for _ in range(1):
        r = client.request(method, url, json=json_body)
        if r.status_code != 200:
            return r, 0.0
    samples = []
    last = None
    for _ in range(iters):
        t0 = time.perf_counter()
        last = client.request(method, url, json=json_body)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return last, samples[len(samples) // 2]


def compare_images(
    cw_bytes: bytes,
    tt_bytes: bytes,
    min_mask_pct: float = 99.0,
    min_psnr: float = 25.0,
) -> tuple[bool, str]:
    im_cw = np.array(Image.open(io.BytesIO(cw_bytes)).convert("RGBA"), dtype=np.float32)
    im_tt = np.array(Image.open(io.BytesIO(tt_bytes)).convert("RGBA"), dtype=np.float32)
    if im_cw.shape != im_tt.shape:
        return False, f"SHAPE DIFF cw={im_cw.shape} vs tt={im_tt.shape}"
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
    ok = mask_pct >= min_mask_pct and psnr >= min_psnr
    return ok, f"shape={im_cw.shape[:2]} mask={mask_pct:.2f}% MAE={mae:.2f} PSNR={psnr:.1f}dB"


def compare_geotiffs(cw_bytes: bytes, tt_bytes: bytes) -> tuple[bool, str]:
    with MemoryFile(cw_bytes) as mcw, mcw.open() as dcw, MemoryFile(tt_bytes) as mtt, mtt.open() as dtt:
        if (dcw.height, dcw.width) != (dtt.height, dtt.width):
            return False, f"shape cw={(dcw.height, dcw.width)} vs tt={(dtt.height, dtt.width)}"
        cw_epsg = dcw.crs.to_epsg() if dcw.crs else None
        tt_epsg = dtt.crs.to_epsg() if dtt.crs else None
        if cw_epsg != tt_epsg:
            return False, f"epsg cw={cw_epsg} vs tt={tt_epsg}"
        b_err = max(abs(a - b) for a, b in zip(dcw.bounds, dtt.bounds))
        scale = max(1.0, abs(dtt.bounds.right - dtt.bounds.left))
        if b_err / scale > 1e-3:
            return False, f"bounds_err={b_err:.4f} (rel={b_err/scale:.4f})"
        return True, f"shape={(dcw.height, dcw.width)} epsg={cw_epsg} bands={dcw.count}/{dtt.count} dtype={dcw.dtypes[0]}/{dtt.dtypes[0]}"


def compare_npy(cw_bytes: bytes, tt_bytes: bytes) -> tuple[bool, str]:
    a_cw = np.load(io.BytesIO(cw_bytes))
    a_tt = np.load(io.BytesIO(tt_bytes))
    if a_cw.shape[-2:] != a_tt.shape[-2:]:
        return False, f"spatial shape cw={a_cw.shape} vs tt={a_tt.shape}"
    return True, f"cw={a_cw.shape}:{a_cw.dtype} tt={a_tt.shape}:{a_tt.dtype}"


def compare_wmts_xml(cw_bytes: bytes, tt_bytes: bytes) -> tuple[bool, str]:
    rcw = ET.fromstring(cw_bytes)
    rtt = ET.fromstring(tt_bytes)
    ns = {"wmts": "http://www.opengis.net/wmts/1.0", "ows": "http://www.opengis.net/ows/1.1"}
    cw_tms = [e.text for e in rcw.findall(".//wmts:TileMatrixSetLink/wmts:TileMatrixSet", ns)]
    tt_tms = [e.text for e in rtt.findall(".//wmts:TileMatrixSetLink/wmts:TileMatrixSet", ns)]
    if cw_tms != tt_tms:
        return False, f"TMS link cw={cw_tms} vs tt={tt_tms}"
    cw_mats = len(rcw.findall(".//wmts:TileMatrix", ns))
    tt_mats = len(rtt.findall(".//wmts:TileMatrix", ns))
    return True, f"tms={cw_tms} matrices=cw:{cw_mats}/tt:{tt_mats}"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not COGZILLA_BIN.exists():
        print("Building cogzilla-server --release...")
        subprocess.run(["cargo", "build", "--release", "-p", "cogzilla-server"], cwd=ROOT, check=True)

    tt_app = RESULTS / "_titiler_oracle_app.py"
    tt_app.write_text(
        "from fastapi import FastAPI\n"
        "from rio_tiler.errors import PointOutsideBounds\n"
        "from rio_tiler.io import STACReader\n"
        "from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers\n"
        "from titiler.core.factory import (\n"
        "    TilerFactory, MultiBaseTilerFactory, ColorMapFactory, TMSFactory, AlgorithmFactory\n"
        ")\n"
        "app = FastAPI()\n"
        "add_exception_handlers(app, {**DEFAULT_STATUS_CODES, PointOutsideBounds: 404})\n"
        "cog = TilerFactory(router_prefix='/cog', add_ogc_maps=True)\n"
        "stac = MultiBaseTilerFactory(reader=STACReader, router_prefix='/stac', add_ogc_maps=True)\n"
        "cmaps = ColorMapFactory()\n"
        "tms = TMSFactory()\n"
        "algs = AlgorithmFactory()\n"
        "app.include_router(cog.router, prefix='/cog')\n"
        "app.include_router(stac.router, prefix='/stac')\n"
        "app.include_router(cmaps.router)\n"
        "app.include_router(tms.router)\n"
        "app.include_router(algs.router)\n"
    )

    cw_proc = subprocess.Popen(
        [str(COGZILLA_BIN), "--bind", f"127.0.0.1:{PORT_CW}", "--cache-mb", "512"],
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
        stac_item = str((FIXTURES / "landsat_c2_stac.json").resolve())
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
        tri_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-122.78, 37.62],
                    [-122.52, 37.62],
                    [-122.78, 37.88],
                    [-122.78, 37.62],
                ]],
            },
            "properties": {},
        }
        gc_overlap = {
            "type": "Feature",
            "geometry": {
                "type": "GeometryCollection",
                "geometries": [
                    {
                        "type": "Polygon",
                        "coordinates": [[
                            [-122.78, 37.62], [-122.60, 37.62],
                            [-122.60, 37.80], [-122.78, 37.80], [-122.78, 37.62],
                        ]],
                    },
                    {
                        "type": "Polygon",
                        "coordinates": [[
                            [-122.70, 37.70], [-122.52, 37.70],
                            [-122.52, 37.88], [-122.70, 37.88], [-122.70, 37.70],
                        ]],
                    },
                ],
            },
            "properties": {},
        }
        fc_two = {
            "type": "FeatureCollection",
            "features": [poly_feature, tri_feature],
        }
        discrete_cmap = quote(json.dumps({"0": [0, 0, 0, 0], "100": [255, 0, 0, 255], "200": [0, 255, 0, 255]}))
        interval_cmap = quote(json.dumps([[[0, 200], [0, 0, 255, 255]], [[200, 600], [0, 255, 0, 255]], [[600, 1200], [255, 0, 0, 255]]]))
        hs_params = quote(json.dumps({"buffer": 0}))
        dtype_paths = ensure_dtype_fixtures()
        u8_path = dtype_paths["uint8"][0]

        cases = [
            # --- /cog metadata & discovery ---
            ("GET /cog/info", "GET", f"/cog/info?url={sat}", None, "json_info"),
            ("GET /cog/info.geojson", "GET", f"/cog/info.geojson?url={sat}", None, "json_geojson"),
            ("GET /cog/WebMercatorQuad/tilejson.json", "GET", f"/cog/WebMercatorQuad/tilejson.json?url={sat}", None, "json_tj"),
            ("GET /cog/WorldCRS84Quad/tilejson.json", "GET", f"/cog/WorldCRS84Quad/tilejson.json?url={sat}", None, "json_tj"),
            ("GET /cog/WebMercatorQuad/tilejson.json (zooms+scale)", "GET", f"/cog/WebMercatorQuad/tilejson.json?url={sat}&minzoom=10&maxzoom=14&tile_scale=2", None, "json_tj"),

            # --- /cog/tiles formats, scales, TMS, resampling, expressions, algorithms ---
            ("GET /cog/tiles/WebMercatorQuad/12/652/1583.png (Sat RGB)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&reproject=bilinear", None, "png"),
            ("GET /cog/tiles/.../12/652/1583.png?tilesize=512", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&tilesize=512&bidx=1&bidx=2&bidx=3&rescale=0,0.35&reproject=bilinear", None, "png"),
            ("GET /cog/tiles/.../12/652/1583.jpg (JPEG output)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.jpg?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&reproject=bilinear", None, "png"),
            ("GET /cog/tiles/.../12/652/1583.webp (WebP output)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.webp?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&reproject=bilinear", None, "png"),
            ("GET /cog/tiles/.../12/652/1583.tif (GeoTIFF output)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.tif?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "geotiff"),
            ("GET /cog/tiles/.../12/652/1583.npy (NPY output)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={sat}&bidx=1", None, "npy"),
            ("GET /cog/tiles/...?buffer=8 (Buffered 272px)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&buffer=8", None, "png"),
            ("GET /cog/tiles/...?padding=2", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&padding=2", None, "png"),
            ("GET /cog/tiles/...?return_mask=false (Opaque RGB)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&return_mask=false", None, "png"),
            ("GET /cog/tiles/...?color_formula=Gamma RGB 1.5", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&color_formula=Gamma%20RGB%201.5", None, "png"),
            ("GET /cog/tiles/...?color_formula=Sigmoidal RGB 15 0.35", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&color_formula=Sigmoidal%20RGB%2015%200.35", None, "png"),
            ("GET /cog/tiles/WorldCRS84Quad/10/325/297.png (CRS84)", "GET", f"/cog/tiles/WorldCRS84Quad/10/325/297.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "png"),
            ("GET /cog/tiles/...?expression=(b1-b2)/(b1+b2)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&expression=(b1-b2)/(b1%2Bb2)&rescale=-0.5,0.5&colormap_name=viridis&reproject=bilinear", None, "png"),
            ("GET /cog/tiles/...?expression=b3;b2;b1 (3-band expr)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&expression=b3;b2;b1&rescale=0,0.35", None, "png"),
            ("GET /cog/tiles/WebMercatorQuad/12/657/1582.png (Mercator)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&rescale=0,1000&colormap_name=viridis&resampling=bilinear", None, "png"),
            ("GET /cog/tiles/...?resampling=nearest (Mercator)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&rescale=0,1000&colormap_name=viridis&resampling=nearest", None, "png"),
            ("GET /cog/tiles/...?resampling=cubic (Mercator)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&rescale=0,1000&colormap_name=viridis&resampling=cubic", None, "png"),
            ("GET /cog/tiles/...?resampling=lanczos (Mercator)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&rescale=0,1000&colormap_name=viridis&resampling=lanczos", None, "png"),
            ("GET /cog/tiles/...?resampling=average (Mercator)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&rescale=0,1000&colormap_name=viridis&resampling=average", None, "png"),
            ("GET /cog/tiles/...?colormap=interval JSON", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&colormap={interval_cmap}", None, "png"),
            ("GET /cog/tiles/...?colormap=discrete JSON (uint8)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={u8_path}&colormap={discrete_cmap}", None, "png"),
            ("GET /cog/tiles/...?algorithm=hillshade (buffer=0)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&algorithm=hillshade&algorithm_params={hs_params}", None, "png_mask"),
            ("GET /cog/tiles/...?algorithm=normalizedIndex", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&bidx=2&algorithm=normalizedIndex&rescale=-1,1&colormap_name=viridis&reproject=bilinear", None, "png"),
            ("GET /cog/tiles/...?algorithm=terrarium", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&algorithm=terrarium", None, "png"),
            ("GET /cog/tiles/...?algorithm=terrainrgb", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={merc}&algorithm=terrainrgb", None, "png"),
            ("GET /cog/tiles/...?algorithm=ceil (uint8)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={u8_path}&algorithm=ceil&rescale=0,255", None, "png"),
            ("GET /cog/tiles/...?algorithm=floor (uint8)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={u8_path}&algorithm=floor&rescale=0,255", None, "png"),

            # --- /cog point, statistics, preview, bbox, feature, map ---
            ("GET /cog/point/-122.65,37.773", "GET", f"/cog/point/-122.65,37.773?url={sat}", None, "json_point"),
            ("GET /cog/point (coord_crs=EPSG:3857)", "GET", f"/cog/point/-13653336.0,4547451.0?url={sat}&coord_crs=EPSG:3857", None, "json_point"),
            ("GET /cog/point (coord_crs=epsg:900913)", "GET", f"/cog/point/-13653336.0,4547451.0?url={sat}&coord_crs=epsg:900913", None, "json_point"),
            ("GET /cog/point?bidx=3&bidx=1", "GET", f"/cog/point/-122.65,37.773?url={sat}&bidx=3&bidx=1", None, "json_point"),
            ("GET /cog/point?expression=(b1-b2)/(b1+b2)", "GET", f"/cog/point/-122.65,37.773?url={sat}&expression=(b1-b2)/(b1%2Bb2)", None, "json_point"),
            ("GET /cog/statistics?max_size=512", "GET", f"/cog/statistics?url={sat}&max_size=512", None, "json_stats"),
            ("GET /cog/statistics?p=10&p=90&histogram_bins=10", "GET", f"/cog/statistics?url={sat}&max_size=256&p=10&p=90&histogram_bins=10", None, "json_stats_ext"),
            ("GET /cog/statistics?bidx=3&bidx=1", "GET", f"/cog/statistics?url={sat}&max_size=256&bidx=3&bidx=1", None, "json_stats_keys"),
            ("POST /cog/statistics (GeoJSON Polygon clip)", "POST", f"/cog/statistics?url={sat}&max_size=512", poly_feature, "json_post_stats"),
            ("POST /cog/statistics (Triangle clip)", "POST", f"/cog/statistics?url={sat}&max_size=256", tri_feature, "json_post_stats"),
            ("POST /cog/statistics (GeometryCollection overlap)", "POST", f"/cog/statistics?url={sat}&max_size=256", gc_overlap, "json_post_stats"),
            ("POST /cog/statistics (FeatureCollection)", "POST", f"/cog/statistics?url={sat}&max_size=256", fc_two, "json_post_stats_fc"),
            ("GET /cog/preview/256x256.png", "GET", f"/cog/preview/256x256.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "png"),
            ("GET /cog/preview/256x256.tif", "GET", f"/cog/preview/256x256.tif?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "geotiff"),
            ("GET /cog/preview?max_size=128", "GET", f"/cog/preview.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&max_size=128", None, "png"),
            ("GET /cog/bbox/.../256x256.png", "GET", f"/cog/bbox/-122.8,37.6,-122.5,37.9/256x256.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "png"),
            ("GET /cog/bbox/.../256x256.tif", "GET", f"/cog/bbox/-122.8,37.6,-122.5,37.9/256x256.tif?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "geotiff"),
            ("GET /cog/bbox (coord_crs+dst_crs=EPSG:3857)", "GET", f"/cog/bbox/-13670000,4525000,-13635000,4565000/256x256.png?url={sat}&coord_crs=EPSG:3857&dst_crs=EPSG:3857&bidx=1&bidx=2&bidx=3&rescale=0,0.35", None, "png"),
            ("POST /cog/feature/256x256.png (Polygon)", "POST", f"/cog/feature/256x256.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", poly_feature, "png"),
            ("POST /cog/feature/256x256.png (Triangle)", "POST", f"/cog/feature/256x256.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", tri_feature, "png"),
            ("POST /cog/feature/256x256.png (GeomColl overlap)", "POST", f"/cog/feature/256x256.png?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35", gc_overlap, "png"),
            ("GET /cog/map (OGC API Maps)", "GET", f"/cog/map?url={sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&width=128&height=128", None, "png"),

            # --- Error status parity ---
            ("ERR /cog/tiles OOB -> 404", "GET", f"/cog/tiles/WebMercatorQuad/12/0/0.png?url={sat}", None, "status_exact"),
            ("ERR /cog/point OOB -> 404", "GET", f"/cog/point/0,0?url={sat}", None, "status_exact"),
            ("ERR /cog/info missing url -> 422", "GET", "/cog/info", None, "status_4xx"),
            ("ERR /cog/point bare coord_crs=3857 -> 422", "GET", f"/cog/point/0,0?url={sat}&coord_crs=3857", None, "status_4xx"),
            ("ERR /cog/point ESRI:3857 -> 422", "GET", f"/cog/point/0,0?url={sat}&coord_crs=ESRI:3857", None, "status_4xx"),
            ("ERR /stac/info missing assets -> 400/422", "GET", f"/stac/info?url={stac_item}", None, "status_4xx"),
            ("ERR /colorMaps/nonexistent -> 404/422", "GET", "/colorMaps/nonexistent_cmap_xyz", None, "status_4xx"),
            ("ERR /algorithms/nonexistent -> 404/422", "GET", "/algorithms/nonexistent_alg_xyz", None, "status_4xx"),
        ]

        # ALL 8 GDAL numeric data types
        for dt_name, (dt_path, dt_rescale) in dtype_paths.items():
            cases.append((
                f"DTYPE [{dt_name:7s}] /cog/tiles/WebMercatorQuad/12/657/1582.png",
                "GET",
                f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={dt_path}&rescale={dt_rescale}&colormap_name=viridis&resampling=bilinear",
                None,
                "png",
            ))

        # /algorithms, /colorMaps, /tileMatrixSets, and /stac/*
        cases.extend([
            ("GET /algorithms", "GET", "/algorithms", None, "json_algs"),
            ("GET /algorithms/hillshade", "GET", "/algorithms/hillshade", None, "json_alg_one"),
            ("GET /algorithms/terrarium", "GET", "/algorithms/terrarium", None, "json_alg_one"),
            ("GET /colorMaps", "GET", "/colorMaps", None, "json_cmaps"),
            ("GET /colorMaps/viridis (256-entry exact LUT)", "GET", "/colorMaps/viridis", None, "json_cmap_lut"),
            ("GET /colorMaps/viridis?f=png", "GET", "/colorMaps/viridis?f=png", None, "png"),
            ("GET /tileMatrixSets", "GET", "/tileMatrixSets", None, "json_tms_list"),
            ("GET /tileMatrixSets/WebMercatorQuad", "GET", "/tileMatrixSets/WebMercatorQuad", None, "json_tms_one"),
            ("GET /stac/assets", "GET", f"/stac/assets?url={stac_item}", None, "json_stac_assets"),
            ("GET /stac/info?assets=red&assets=green", "GET", f"/stac/info?url={stac_item}&assets=red&assets=green", None, "json_stac_info"),
            ("GET /stac/info.geojson?assets=red", "GET", f"/stac/info.geojson?url={stac_item}&assets=red", None, "json_geojson"),
            ("GET /stac/tiles/.../12/652/1583.png (RGB 3-asset)", "GET", f"/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_item}&assets=red&assets=green&assets=blue&rescale=7000,20000&reproject=bilinear", None, "png"),
            ("GET /stac/tiles/.../12/652/1583.png?tilesize=512", "GET", f"/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_item}&tilesize=512&assets=red&assets=green&assets=blue&rescale=7000,20000&reproject=bilinear", None, "png"),
            ("GET /stac/tiles/.../12/652/1583.tif (STAC GeoTIFF)", "GET", f"/stac/tiles/WebMercatorQuad/12/652/1583.tif?url={stac_item}&assets=red&assets=green&assets=blue&rescale=7000,20000", None, "geotiff"),
            ("GET /stac/tiles/...?expression=(b1-b2)/(b1+b2)", "GET", f"/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_item}&assets=red&assets=green&expression=(b1-b2)/(b1%2Bb2)&rescale=-0.5,0.5&colormap_name=rdylgn&reproject=bilinear", None, "png"),
            ("GET /stac/WebMercatorQuad/tilejson.json", "GET", f"/stac/WebMercatorQuad/tilejson.json?url={stac_item}&assets=red&assets=green&assets=blue", None, "json_tj"),
            ("GET /stac/point/-122.65,37.773 (3-asset)", "GET", f"/stac/point/-122.65,37.773?url={stac_item}&assets=red&assets=green&assets=blue", None, "json_point"),
            ("GET /stac/asset_statistics?assets=red&assets=green", "GET", f"/stac/asset_statistics?url={stac_item}&assets=red&assets=green&max_size=256", None, "json_stac_asset_stats"),
            ("GET /stac/statistics?assets=red&assets=green", "GET", f"/stac/statistics?url={stac_item}&assets=red&assets=green&max_size=256", None, "json_stats"),
            ("POST /stac/statistics (Polygon)", "POST", f"/stac/statistics?url={stac_item}&assets=red&assets=green&max_size=256", poly_feature, "json_post_stats"),
            ("GET /stac/preview/256x256.png (3-asset RGB)", "GET", f"/stac/preview/256x256.png?url={stac_item}&assets=red&assets=green&assets=blue&rescale=7000,20000", None, "png"),
            ("GET /stac/bbox/.../256x256.png (3-asset RGB)", "GET", f"/stac/bbox/-122.8,37.6,-122.5,37.9/256x256.png?url={stac_item}&assets=red&assets=green&assets=blue&rescale=7000,20000", None, "png"),
            ("POST /stac/feature/256x256.png (Triangle)", "POST", f"/stac/feature/256x256.png?url={stac_item}&assets=red&assets=green&assets=blue&rescale=7000,20000", tri_feature, "png"),
            ("GET /stac/map (OGC API Maps)", "GET", f"/stac/map?url={stac_item}&assets=red&assets=green&assets=blue&rescale=7000,20000&width=128&height=128", None, "png"),

            # --- Content-Bbox / Content-Crs header parity & 1-band greyscale PNG/JPEG mode ---
            ("HDR /cog/tiles WebMercatorQuad", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&rescale=0,0.35", None, "geo_headers"),
            ("HDR /cog/tiles WorldCRS84Quad", "GET", f"/cog/tiles/WorldCRS84Quad/10/325/297.png?url={sat}&bidx=1&rescale=0,0.35", None, "geo_headers"),
            ("HDR /cog/preview native CRS", "GET", f"/cog/preview/64x64.png?url={sat}&bidx=1&rescale=0,0.35", None, "geo_headers"),
            ("HDR /cog/bbox default 4326", "GET", f"/cog/bbox/-122.8,37.6,-122.5,37.9/64x64.png?url={sat}&bidx=1&rescale=0,0.35", None, "geo_headers"),
            ("HDR /cog/bbox dst_crs=EPSG:3857", "GET", f"/cog/bbox/-122.8,37.6,-122.5,37.9/64x64.png?url={sat}&dst_crs=EPSG:3857&bidx=1&rescale=0,0.35", None, "geo_headers"),
            ("HDR POST /cog/feature", "POST", f"/cog/feature/64x64.png?url={sat}&bidx=1&rescale=0,0.35", poly_feature, "geo_headers"),
            ("HDR /stac/tiles WebMercatorQuad", "GET", f"/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_item}&assets=red&rescale=7000,20000", None, "geo_headers"),
            ("HDR /stac/preview native CRS", "GET", f"/stac/preview/64x64.png?url={stac_item}&assets=red&rescale=7000,20000", None, "geo_headers"),
            ("MODE 1-band /cog/tiles PNG -> L/LA", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat}&bidx=1&rescale=0,0.35", None, "gray_png"),
            ("MODE 1-band /cog/preview PNG -> L/LA", "GET", f"/cog/preview/64x64.png?url={sat}&bidx=1&rescale=0,0.35", None, "gray_png"),
            ("MODE 1-band /cog/preview return_mask=false -> L", "GET", f"/cog/preview/64x64.png?url={sat}&bidx=1&rescale=0,0.35&return_mask=false", None, "gray_png_l"),
            ("MODE 1-band /cog/tiles JPEG -> L", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.jpg?url={sat}&bidx=1&rescale=0,0.35", None, "gray_jpg"),
            ("MODE 1-band /cog/preview JPEG -> L", "GET", f"/cog/preview/64x64.jpg?url={sat}&bidx=1&rescale=0,0.35", None, "gray_jpg"),
        ])

        print(f"\n=== Live HTTP Differential Oracle: cogzilla vs. TiTiler ({len(cases)} checks) ===")
        print(f"{'St':4s} | {'Endpoint':54s} | {'cogzilla':>9s} | {'TiTiler':>9s} | {'Speedup':>8s} | Parity vs TiTiler")
        print("-" * 145)

        summary = []
        failures = []
        with httpx.Client(timeout=30.0) as client:
            for label, method, path, body, kind in cases:
                if kind in ("status_exact", "status_4xx"):
                    r_cw = client.request(method, cw_base + path, json=body)
                    r_tt = client.request(method, tt_base + path, json=body)
                    ms_cw, ms_tt, speedup = 0.0, 0.0, 0.0
                    if kind == "status_exact":
                        ok = r_cw.status_code == r_tt.status_code
                    else:
                        ok = 400 <= r_cw.status_code < 500 and 400 <= r_tt.status_code < 500
                    parity = f"status cw={r_cw.status_code} vs tt={r_tt.status_code}"
                else:
                    r_cw, ms_cw = time_request(client, method, cw_base + path, body, iters=5)
                    r_tt, ms_tt = time_request(client, method, tt_base + path, body, iters=5)
                    speedup = (ms_tt / ms_cw) if ms_cw > 0 else 0.0
                    if r_cw.status_code != 200 or r_tt.status_code != 200:
                        ok = False
                        parity = f"HTTP status cw={r_cw.status_code} ({r_cw.text[:80]!r}) vs tt={r_tt.status_code} ({r_tt.text[:80]!r})"
                    elif kind == "png":
                        ok, parity = compare_images(r_cw.content, r_tt.content)
                    elif kind == "png_mask":
                        ok, parity = compare_images(r_cw.content, r_tt.content, min_mask_pct=99.0, min_psnr=0.0)
                    elif kind == "png_loose":
                        ok, parity = compare_images(r_cw.content, r_tt.content, min_mask_pct=98.0, min_psnr=15.0)
                    elif kind == "geotiff":
                        ok, parity = compare_geotiffs(r_cw.content, r_tt.content)
                    elif kind == "npy":
                        ok, parity = compare_npy(r_cw.content, r_tt.content)
                    elif kind == "wmts_xml":
                        ok, parity = compare_wmts_xml(r_cw.content, r_tt.content)
                    elif kind == "json_bounds":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        b_err = max(abs(a - b) for a, b in zip(j_cw["bounds"], j_tt["bounds"]))
                        scale = max(1.0, abs(j_tt["bounds"][2] - j_tt["bounds"][0]))
                        ok = (b_err / scale) < 1e-3
                        parity = f"bounds_err={b_err:.5f} rel={b_err/scale:.5f}"
                    elif kind == "json_info":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        b_err = max(abs(a - b) for a, b in zip(j_cw["bounds"], j_tt["bounds"]))
                        ok = j_cw["dtype"] == j_tt["dtype"] and j_cw["count"] == j_tt["count"] and b_err < 1e-3
                        parity = f"dtype={j_cw['dtype']}=={j_tt['dtype']} count={j_cw['count']}=={j_tt['count']} bounds_err={b_err:.5f} deg"
                    elif kind == "json_geojson":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        ok = j_cw["type"] == j_tt["type"] and j_cw["geometry"]["type"] == j_tt["geometry"]["type"]
                        parity = f"type={j_cw['type']}=={j_tt['type']} geom={j_cw['geometry']['type']}=={j_tt['geometry']['type']}"
                    elif kind == "json_tj":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        b_err = max(abs(a - b) for a, b in zip(j_cw["bounds"], j_tt["bounds"]))
                        ok = j_cw["tilejson"] == j_tt["tilejson"] and b_err < 1e-3
                        parity = f"tilejson={j_cw['tilejson']}=={j_tt['tilejson']} zoom=({j_cw['minzoom']},{j_cw['maxzoom']}) vs ({j_tt['minzoom']},{j_tt['maxzoom']}) bounds_err={b_err:.5f}"
                    elif kind == "json_point":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        v_err = max(abs(a - b) for a, b in zip(j_cw["values"], j_tt["values"]))
                        ok = len(j_cw["values"]) == len(j_tt["values"]) and v_err < 1e-4
                        parity = f"max_val_err={v_err:.7f} bands={j_cw['band_names']}"
                    elif kind == "json_stats":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        m_err = abs(j_cw["b1"]["mean"] - j_tt["b1"]["mean"])
                        s_err = abs(j_cw["b1"]["std"] - j_tt["b1"]["std"])
                        scale = max(1.0, abs(j_tt["b1"]["mean"]))
                        ok = (m_err / scale) < 1e-2 and (s_err / scale) < 1e-2
                        parity = f"b1.mean_err={m_err:.5f} b1.std_err={s_err:.5f}"
                    elif kind == "json_stats_ext":
                        j_cw, j_tt = r_cw.json()["b1"], r_tt.json()["b1"]
                        p10_err = abs(j_cw["percentile_10"] - j_tt["percentile_10"])
                        p90_err = abs(j_cw["percentile_90"] - j_tt["percentile_90"])
                        bins_ok = len(j_cw["histogram"][0]) == len(j_tt["histogram"][0]) == 10
                        ok = bins_ok and p10_err < 0.05 and p90_err < 0.05
                        parity = f"p10_err={p10_err:.4f} p90_err={p90_err:.4f} bins={len(j_cw['histogram'][0])}"
                    elif kind == "json_stats_keys":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        ok = set(j_cw.keys()) == set(j_tt.keys())
                        parity = f"keys cw={sorted(j_cw.keys())} vs tt={sorted(j_tt.keys())}"
                    elif kind == "json_post_stats":
                        j_cw = r_cw.json()["properties"]["statistics"]
                        j_tt = r_tt.json()["properties"]["statistics"]
                        m_err = abs(j_cw["b1"]["mean"] - j_tt["b1"]["mean"])
                        s_err = abs(j_cw["b1"]["std"] - j_tt["b1"]["std"])
                        scale = max(1.0, abs(j_tt["b1"]["mean"]))
                        ok = (m_err / scale) < 2e-2 and (s_err / scale) < 2e-2
                        parity = f"poly b1.mean_err={m_err:.5f} b1.std_err={s_err:.5f}"
                    elif kind == "json_post_stats_fc":
                        f_cw = r_cw.json()["features"]
                        f_tt = r_tt.json()["features"]
                        ok = len(f_cw) == len(f_tt) == 2
                        m_err = max(
                            abs(a["properties"]["statistics"]["b1"]["mean"] - b["properties"]["statistics"]["b1"]["mean"])
                            for a, b in zip(f_cw, f_tt)
                        )
                        ok = ok and m_err < 0.02
                        parity = f"features={len(f_cw)} max_mean_err={m_err:.5f}"
                    elif kind == "json_algs":
                        a_cw = set(r_cw.json().keys())
                        a_tt = set(r_tt.json().keys())
                        ok = a_tt.issubset(a_cw)
                        parity = f"cw={len(a_cw)} tt={len(a_tt)} missing={sorted(a_tt - a_cw)}"
                    elif kind == "json_alg_one":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        ok = "title" in j_cw and "title" in j_tt
                        parity = f"title cw={j_cw.get('title')!r} tt={j_tt.get('title')!r}"
                    elif kind == "json_cmaps":
                        j_cw = {c["id"] for c in r_cw.json()["colormaps"]}
                        j_tt = {c["id"] for c in r_tt.json()["colormaps"]}
                        overlap = len(j_cw & j_tt)
                        ok = overlap == len(j_tt)
                        parity = f"cogzilla={len(j_cw)} titiler={len(j_tt)} overlap={overlap}/{len(j_cw)}"
                    elif kind == "json_cmap_lut":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        exact = sum(1 for k in j_cw if j_cw[k] == j_tt.get(k))
                        ok = exact == 256
                        parity = f"exact_entries={exact}/256"
                    elif kind == "json_tms_list":
                        ids_cw = {t["id"] for t in r_cw.json()["tileMatrixSets"]}
                        ids_tt = {t["id"] for t in r_tt.json()["tileMatrixSets"]}
                        ok = ids_cw.issubset(ids_tt)
                        parity = f"count={len(ids_cw)} subset_of_tt={ok}"
                    elif kind == "json_tms_one":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        ok = j_cw["id"] == j_tt["id"] and len(j_cw["tileMatrices"]) == len(j_tt["tileMatrices"])
                        parity = f"id={j_cw['id']}=={j_tt['id']} matrices={len(j_cw['tileMatrices'])}=={len(j_tt['tileMatrices'])}"
                    elif kind == "json_stac_assets":
                        j_cw, j_tt = set(r_cw.json()), set(r_tt.json())
                        ok = j_cw == j_tt
                        parity = f"assets={sorted(j_cw)} exact_match={ok}"
                    elif kind == "json_stac_info":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        ok = sorted(j_cw.keys()) == sorted(j_tt.keys())
                        parity = f"keys={sorted(j_cw.keys())}=={sorted(j_tt.keys())}"
                    elif kind == "json_stac_asset_stats":
                        j_cw, j_tt = r_cw.json(), r_tt.json()
                        m_err = abs(j_cw["red"]["b1"]["mean"] - j_tt["red"]["b1"]["mean"])
                        ok = sorted(j_cw.keys()) == sorted(j_tt.keys()) and m_err < 5.0
                        parity = f"keys={sorted(j_cw.keys())}=={sorted(j_tt.keys())} red.b1.mean_err={m_err:.2f}"
                    elif kind == "geo_headers":
                        crs_cw, crs_tt = r_cw.headers.get("content-crs"), r_tt.headers.get("content-crs")
                        bb_cw, bb_tt = r_cw.headers.get("content-bbox"), r_tt.headers.get("content-bbox")
                        crs_ok = bool(crs_cw) and crs_cw == crs_tt
                        bb_ok = False
                        max_rel = 1.0
                        if bb_cw and bb_tt:
                            v_cw = [float(x) for x in bb_cw.split(",")]
                            v_tt = [float(x) for x in bb_tt.split(",")]
                            if len(v_cw) == 4 and len(v_tt) == 4:
                                max_rel = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(v_cw, v_tt))
                                bb_ok = max_rel < 1e-4
                        ok = crs_ok and bb_ok
                        parity = f"crs={crs_cw} vs {crs_tt} bbox_rel_err={max_rel:.2e}"
                    elif kind in ("gray_png", "gray_png_l", "gray_jpg"):
                        im_cw = Image.open(io.BytesIO(r_cw.content))
                        im_tt = Image.open(io.BytesIO(r_tt.content))
                        if kind == "gray_png":
                            ok = im_cw.mode in ("L", "LA") and im_tt.mode in ("L", "LA")
                        else:
                            ok = im_cw.mode == "L" and im_tt.mode == "L"
                        parity = f"mode cw={im_cw.mode} vs tt={im_tt.mode}"
                    else:
                        ok, parity = True, "ok"

                tag = "PASS" if ok else "FAIL"
                if not ok:
                    failures.append((label, parity))
                print(f"{tag:4s} | {label:54s} | {ms_cw:6.2f} ms | {ms_tt:6.2f} ms | {speedup:7.1f}x | {parity}")
                summary.append({
                    "endpoint": label,
                    "ok": ok,
                    "cogzilla_ms": ms_cw,
                    "titiler_ms": ms_tt,
                    "speedup": speedup,
                    "parity": parity,
                })

        (RESULTS / "titiler_diff_oracle.json").write_text(json.dumps(summary, indent=2))
        passed = len(cases) - len(failures)
        print(f"\nOracle summary: {passed}/{len(cases)} passed, {len(failures)} failed")
        if failures:
            print("Failures:")
            for lbl, msg in failures:
                print(f"  - {lbl}: {msg}")
            sys.exit(1)
    finally:
        cw_proc.terminate()
        tt_proc.terminate()
        cw_proc.wait(timeout=5)
        tt_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
