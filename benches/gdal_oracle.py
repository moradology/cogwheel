#!/usr/bin/env python3
"""
GDAL / rio-tiler / TiTiler benchmark harness and pixel-accuracy parity oracle for COGMark.
Runs the exact same BM-01..BM-08 workloads against C++ libgdal (via rasterio/rio-tiler)
and TiTiler HTTP (uvicorn), plus validates cogzilla pixel accuracy against GDAL.
"""

from __future__ import annotations

import asyncio
import http.server
import io
import json
import os
import socketserver
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rio_tiler.io import Reader as RioReader
from rio_tiler.colormap import cmap as rio_cmap

FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = Path(__file__).parent / "results"
PARITY_DIR = RESULTS / "parity"


def summarize_us(samples_s: list[float]) -> tuple[float, float]:
    us = sorted(s * 1e6 for s in samples_s)
    med = us[len(us) // 2]
    p99 = us[min(len(us) - 1, int(len(us) * 0.99))]
    return med, p99


def verify_parity():
    """Compare cogzilla raw f32 tiles against rio-tiler/GDAL WarpedVRT pixel-by-pixel."""
    print("\n=== GDAL Parity Gate ===")
    checks = [
        ("mercator_z12_f32.bin", "mercator_4k_zstd_p3.tif", (657, 1582, 12), [1]),
        ("geo4326_z12_f32.bin", "geo4326_4k_deflate_p3.tif", (657, 1583, 12), [1]),
        ("sat_utm10n_rgb_z12_f32.bin", "satellite_utm10n_rgb_zstd_p3.tif", (652, 1583, 12), [1, 2, 3]),
        ("albers5070_z12_f32.bin", "albers5070_2k_zstd_p3.tif", (662, 1595, 12), [1]),
    ]
    report = {}
    for dump_name, cog_name, (x, y, z), bands in checks:
        dump_path = PARITY_DIR / dump_name
        if not dump_path.exists():
            print(f"  [SKIP] {dump_name} not found")
            continue
        cw = np.fromfile(dump_path, dtype="<f4").reshape(len(bands), 256, 256)
        with RioReader(str(FIXTURES / cog_name)) as src:
            img = src.tile(x, y, z, indexes=bands, resampling_method="bilinear")
            gd = img.data.astype(np.float32)
            # Apply mask -> NaN
            gd_masked = np.where(img.mask[None, :, :] == 0, np.nan, gd)

        # Compare interior pixels (excluding 1px border where GDAL edge-clamp vs NaN can differ slightly)
        cw_in = cw[:, 2:-2, 2:-2]
        gd_in = gd_masked[:, 2:-2, 2:-2]
        both_valid = np.isfinite(cw_in) & np.isfinite(gd_in)
        mask_agree = float(np.mean(np.isfinite(cw_in) == np.isfinite(gd_in))) * 100.0
        if np.any(both_valid):
            diff = np.abs(cw_in[both_valid] - gd_in[both_valid])
            scale = np.maximum(np.abs(gd_in[both_valid]), 1.0)
            max_abs = float(np.max(diff))
            mean_abs = float(np.mean(diff))
            p99_rel = float(np.percentile(diff / scale, 99))
            # Check correlation
            corr = float(np.corrcoef(cw_in[both_valid].ravel(), gd_in[both_valid].ravel())[0, 1])
        else:
            max_abs = mean_abs = p99_rel = 0.0
            corr = 1.0

        print(
            f"  {dump_name:28s}: mask_agree={mask_agree:6.2f}%  mean_abs={mean_abs:.5f}  "
            f"p99_rel={p99_rel:.6f}  corr={corr:.7f}"
        )
        report[dump_name] = {
            "mask_agree_pct": mask_agree,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "p99_rel": p99_rel,
            "corr": corr,
        }

    # Verify PNG decodability with Pillow
    png_path = PARITY_DIR / "mercator_z12.png"
    if png_path.exists():
        im = Image.open(png_path)
        arr = np.array(im)
        print(f"  mercator_z12.png decoded cleanly via PIL: mode={im.mode}, shape={arr.shape}")
        report["mercator_png_valid"] = {"mode": im.mode, "shape": list(arr.shape)}

    return report


def run_gdal_benchmarks() -> list[dict]:
    print("\n=== COGMark Benchmark Suite (GDAL / rio-tiler / TiTiler) ===")
    results = []

    # ------------------------------------------------------------------------
    # BM-01: ColdHeader (rasterio.open -> GDALOpenEx)
    # ------------------------------------------------------------------------
    p_merc = str(FIXTURES / "mercator_4k_zstd_p3.tif")
    for _ in range(5):
        with rasterio.open(p_merc) as s:
            _ = (s.crs, s.transform, s.overviews(1))
    samples = []
    for _ in range(100):
        t0 = time.perf_counter()
        with rasterio.open(p_merc) as s:
            _ = (s.crs, s.transform, s.overviews(1))
        samples.append(time.perf_counter() - t0)
    med, p99 = summarize_us(samples)
    print(f"BM-01 ColdHeader (GDALOpenEx): median={med:.1f} us  p99={p99:.1f} us")
    results.append({"id": "BM-01", "name": "ColdHeader", "median_us": med, "p99_us": p99})

    # ------------------------------------------------------------------------
    # BM-02: UnshufflePred3 (single 512x512 block read via GDAL with GDAL_CACHEMAX=0)
    # ------------------------------------------------------------------------
    codecs = [
        ("NONE_P1", "codec_none_p1.tif"),
        ("DEFLATE_P1", "codec_deflate_p1.tif"),
        ("DEFLATE_P2", "codec_deflate_p2.tif"),
        ("DEFLATE_P3", "codec_deflate_p3.tif"),
        ("ZSTD_P1", "codec_zstd_p1.tif"),
        ("ZSTD_P3", "codec_zstd_p3.tif"),
        ("LZW_P1", "codec_lzw_p1.tif"),
        ("LZW_P3", "codec_lzw_p3.tif"),
    ]
    win0 = rasterio.windows.Window(0, 0, 512, 512)
    win1 = rasterio.windows.Window(512, 0, 512, 512)
    with rasterio.Env(GDAL_CACHEMAX=0):
        for label, fname in codecs:
            with rasterio.open(str(FIXTURES / fname)) as src:
                for i in range(4):
                    _ = src.read(1, window=win0 if (i & 1) == 0 else win1)
                samples = []
                for i in range(60):
                    w = win0 if (i & 1) == 0 else win1
                    t0 = time.perf_counter()
                    _ = src.read(1, window=w)
                    samples.append(time.perf_counter() - t0)
            med, p99 = summarize_us(samples)
            gb_s = (1048576.0 / (med * 1e-6)) / 1e9
            print(f"BM-02 UnshufflePred3 [{label:10s}] (GDAL): median={med:7.1f} us  ({gb_s:.2f} GB/s)")
            results.append({"id": f"BM-02-{label}", "name": f"UnshufflePred3 ({label})", "median_us": med, "p99_us": p99})

    # ------------------------------------------------------------------------
    # BM-03: NativeMercator (rio-tiler + GDAL -> PNG / WebP / RawF32)
    # ------------------------------------------------------------------------
    cm_viridis = rio_cmap.get("viridis")
    for label, img_fmt, extra_kw in [
        ("PNG", "PNG", {}),
        ("WebP_Lossy82", "WEBP", {"quality": 82}),
        ("RawF32", None, {}),
    ]:
        # Cold (GDAL_CACHEMAX=0)
        with rasterio.Env(GDAL_CACHEMAX=0):
            with RioReader(p_merc) as src:
                for _ in range(3):
                    im = src.tile(657, 1582, 12, resampling_method="bilinear")
                    if img_fmt:
                        im.rescale(in_range=((0, 1000),))
                        _ = im.render(img_format=img_fmt, colormap=cm_viridis, **extra_kw)
                    else:
                        _ = im.data.tobytes()
                samples = []
                for _ in range(30):
                    t0 = time.perf_counter()
                    im = src.tile(657, 1582, 12, resampling_method="bilinear")
                    if img_fmt:
                        im.rescale(in_range=((0, 1000),))
                        _ = im.render(img_format=img_fmt, colormap=cm_viridis, **extra_kw)
                    else:
                        _ = im.data.tobytes()
                    samples.append(time.perf_counter() - t0)
        med_c, p99_c = summarize_us(samples)

        # Warm (GDAL_CACHEMAX=512MB)
        with rasterio.Env(GDAL_CACHEMAX=512):
            with RioReader(p_merc) as src:
                for _ in range(5):
                    im = src.tile(657, 1582, 12, resampling_method="bilinear")
                    if img_fmt:
                        im.rescale(in_range=((0, 1000),))
                        _ = im.render(img_format=img_fmt, colormap=cm_viridis, **extra_kw)
                    else:
                        _ = im.data.tobytes()
                samples = []
                for _ in range(60):
                    t0 = time.perf_counter()
                    im = src.tile(657, 1582, 12, resampling_method="bilinear")
                    if img_fmt:
                        im.rescale(in_range=((0, 1000),))
                        _ = im.render(img_format=img_fmt, colormap=cm_viridis, **extra_kw)
                    else:
                        _ = im.data.tobytes()
                    samples.append(time.perf_counter() - t0)
        med_w, p99_w = summarize_us(samples)
        print(f"BM-03 NativeMercator [{label:13s}] (GDAL): cold={med_c:7.1f} us  warm={med_w:7.1f} us")
        results.append({"id": f"BM-03-{label}-cold", "median_us": med_c, "p99_us": p99_c})
        results.append({"id": f"BM-03-{label}-warm", "median_us": med_w, "p99_us": p99_w})

    # ------------------------------------------------------------------------
    # BM-04: Reproj4326 (EPSG:4326 DEFLATE+Pred3 -> PNG)
    # ------------------------------------------------------------------------
    p_4326 = str(FIXTURES / "geo4326_4k_deflate_p3.tif")
    for mode, cache_mb, iters in [("cold", 0, 25), ("warm", 512, 50)]:
        with rasterio.Env(GDAL_CACHEMAX=cache_mb):
            with RioReader(p_4326) as src:
                for _ in range(3):
                    im = src.tile(657, 1583, 12, resampling_method="bilinear")
                    im.rescale(in_range=((0, 1000),))
                    _ = im.render(img_format="PNG", colormap=cm_viridis)
                samples = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    im = src.tile(657, 1583, 12, resampling_method="bilinear")
                    im.rescale(in_range=((0, 1000),))
                    _ = im.render(img_format="PNG", colormap=cm_viridis)
                    samples.append(time.perf_counter() - t0)
        med, p99 = summarize_us(samples)
        print(f"BM-04 Reproj4326 [PNG] (GDAL {mode}): {med:7.1f} us")
        results.append({"id": f"BM-04-{mode}", "median_us": med, "p99_us": p99})

    # ------------------------------------------------------------------------
    # BM-05: ReprojProjected (Satellite 3-band UTM10N RGB, UTM LZW, Albers 5070)
    # ------------------------------------------------------------------------
    proj_cases = [
        ("BM-05a-SatRGB-PNG", "satellite_utm10n_rgb_zstd_p3.tif", (652, 1583, 12), [1, 2, 3], ((0, 0.35), (0, 0.35), (0, 0.35)), None),
        ("BM-05b-UTM-LZW", "utm10n_2k_lzw_p3.tif", (652, 1583, 12), [1], ((0, 0.35),), cm_viridis),
        ("BM-05c-Albers", "albers5070_2k_zstd_p3.tif", (662, 1595, 12), [1], ((0, 1000),), cm_viridis),
    ]
    for cid, fname, (x, y, z), bidx, rsc, cm in proj_cases:
        fpath = str(FIXTURES / fname)
        for mode, cache_mb, iters in [("cold", 0, 20), ("warm", 512, 40)]:
            with rasterio.Env(GDAL_CACHEMAX=cache_mb):
                with RioReader(fpath) as src:
                    for _ in range(3):
                        im = src.tile(x, y, z, indexes=bidx, resampling_method="bilinear")
                        im.rescale(in_range=rsc)
                        _ = im.render(img_format="PNG", colormap=cm)
                    samples = []
                    for _ in range(iters):
                        t0 = time.perf_counter()
                        im = src.tile(x, y, z, indexes=bidx, resampling_method="bilinear")
                        im.rescale(in_range=rsc)
                        _ = im.render(img_format="PNG", colormap=cm)
                        samples.append(time.perf_counter() - t0)
            med, p99 = summarize_us(samples)
            print(f"{cid:20s} (GDAL {mode}): {med:7.1f} us")
            results.append({"id": f"{cid}-{mode}", "median_us": med, "p99_us": p99})

    # ------------------------------------------------------------------------
    # BM-06: SparseCoastline (32 mixed tiles)
    # ------------------------------------------------------------------------
    p_coast = str(FIXTURES / "coastal_sparse_4k_zstd_p3.tif")
    tiles_32 = [(x, y, 12) for y in range(1581, 1585) for x in range(654, 662)]
    with rasterio.Env(GDAL_CACHEMAX=512):
        with RioReader(p_coast) as src:
            for x, y, z in tiles_32:
                im = src.tile(x, y, z, resampling_method="bilinear")
                im.rescale(in_range=((0, 1000),))
                _ = im.render(img_format="PNG", colormap=cm_viridis)
            samples = []
            for _ in range(5):
                t0 = time.perf_counter()
                for x, y, z in tiles_32:
                    im = src.tile(x, y, z, resampling_method="bilinear")
                    im.rescale(in_range=((0, 1000),))
                    _ = im.render(img_format="PNG", colormap=cm_viridis)
                samples.append((time.perf_counter() - t0) / 32.0)
    med, p99 = summarize_us(samples)
    print(f"BM-06 SparseCoastline (GDAL avg/tile): {med:7.1f} us")
    results.append({"id": "BM-06", "median_us": med, "p99_us": p99})

    # ------------------------------------------------------------------------
    # BM-07: PanBurst32 HTTP (TiTiler + uvicorn with 4 workers on loopback)
    # ------------------------------------------------------------------------
    titiler_app_path = RESULTS / "_titiler_app.py"
    titiler_app_path.write_text(
        "from fastapi import FastAPI\n"
        "from titiler.core.factory import TilerFactory\n"
        "app = FastAPI()\n"
        "cog = TilerFactory()\n"
        "app.include_router(cog.router)\n"
    )
    port = 18765
    env = os.environ.copy()
    env.update({
        "GDAL_CACHEMAX": "200",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "536870912",
    })
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "_titiler_app:app",
            "--app-dir",
            str(RESULTS),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "10",
            "--log-level",
            "error",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Wait for TiTiler to come up
        for _ in range(50):
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/openapi.json", timeout=0.5)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)

        async def run_http_burst(urls: list[str], rounds: int):
            limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
            async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
                # Warmup / cold burst
                t0 = time.perf_counter()
                resps = await asyncio.gather(*(client.get(u) for u in urls))
                assert all(r.status_code == 200 for r in resps), [r.status_code for r in resps[:3]]
                cold_ms = (time.perf_counter() - t0) * 1e3

                t1 = time.perf_counter()
                for _ in range(rounds):
                    resps = await asyncio.gather(*(client.get(u) for u in urls))
                    assert all(r.status_code == 200 for r in resps)
                elapsed_s = time.perf_counter() - t1
                qps = (rounds * len(urls)) / elapsed_s
                warm_ms = (elapsed_s * 1e3) / rounds
                return cold_ms, warm_ms, qps

        urls_merc = [
            f"http://127.0.0.1:{port}/tiles/WebMercatorQuad/12/{x}/{y}.png?url={p_merc}&rescale=0,1000&colormap_name=viridis&resampling=bilinear"
            for y in range(1581, 1585)
            for x in range(654, 662)
        ]
        cold_ms, warm_ms, qps = asyncio.run(run_http_burst(urls_merc, rounds=6))
        print(f"BM-07 PanBurst32 HTTP [TiTiler Mercator PNG]: cold_32={cold_ms:.1f} ms  warm_32={warm_ms:.1f} ms  ({qps:.1f} tiles/sec)")
        results.append({
            "id": "BM-07-HTTP-Mercator",
            "median_us": warm_ms * 1000.0,
            "extra": {"cold_32_ms": cold_ms, "warm_32_ms": warm_ms, "qps": qps},
        })

        p_sat = str(FIXTURES / "satellite_utm10n_rgb_zstd_p3.tif")
        urls_sat = [
            f"http://127.0.0.1:{port}/tiles/WebMercatorQuad/13/{x}/{y}.png?url={p_sat}&bidx=1&bidx=2&bidx=3&rescale=0,0.35&resampling=bilinear"
            for y in range(3164, 3168)
            for x in range(1301, 1309)
        ]
        sc_ms, sw_ms, sqps = asyncio.run(run_http_burst(urls_sat, rounds=4))
        print(f"BM-07 PanBurst32 HTTP [TiTiler Sat RGB PNG]:  cold_32={sc_ms:.1f} ms  warm_32={sw_ms:.1f} ms  ({sqps:.1f} tiles/sec)")
        results.append({
            "id": "BM-07-HTTP-SatRGB",
            "median_us": sw_ms * 1000.0,
            "extra": {"cold_32_ms": sc_ms, "warm_32_ms": sw_ms, "qps": sqps},
        })
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    # ------------------------------------------------------------------------
    # BM-08: RemoteRangeRTT (GDAL /vsicurl/ against 20ms-RTT local Range server)
    # ------------------------------------------------------------------------
    file_bytes = Path(p_merc).read_bytes()
    stats_box = {"reqs": 0, "bytes": 0}

    class DelayedRangeHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_HEAD(self):
            stats_box["reqs"] += 1
            time.sleep(0.020)
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(file_bytes)))
            self.end_headers()

        def do_GET(self):
            stats_box["reqs"] += 1
            time.sleep(0.020)
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                a, b = rng[6:].split("-")
                s = int(a)
                e = min(int(b), len(file_bytes) - 1) if b else len(file_bytes) - 1
                chunk = file_bytes[s : e + 1]
                stats_box["bytes"] += len(chunk)
                self.send_response(206)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {s}-{e}/{len(file_bytes)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                stats_box["bytes"] += len(file_bytes)
                self.send_response(200)
                self.send_header("Content-Length", str(len(file_bytes)))
                self.end_headers()
                self.wfile.write(file_bytes)

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    srv = ThreadingHTTPServer(("127.0.0.1", 0), DelayedRangeHandler)
    srv_port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{srv_port}/mercator.tif"
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            VSI_CACHE="FALSE",
        ):
            t0 = time.perf_counter()
            with RioReader(url) as src:
                open_ms = (time.perf_counter() - t0) * 1e3
                open_reqs = stats_box["reqs"]
                t1 = time.perf_counter()
                im = src.tile(657, 1582, 12, resampling_method="bilinear")
                im.rescale(in_range=((0, 1000),))
                _ = im.render(img_format="PNG", colormap=cm_viridis)
                cold_tile_ms = (time.perf_counter() - t1) * 1e3
        total_reqs = stats_box["reqs"]
        total_bytes = stats_box["bytes"]
        print(
            f"BM-08 RemoteRangeRTT (GDAL /vsicurl/, 20ms RTT): open={open_ms:.1f}ms ({open_reqs} RTT)  "
            f"cold_tile={cold_tile_ms:.1f}ms ({total_reqs - open_reqs} tile RTTs, total_reqs={total_reqs}, {total_bytes/1024:.1f} KB)"
        )
        results.append({
            "id": "BM-08",
            "median_us": (open_ms + cold_tile_ms) * 1000.0,
            "extra": {
                "open_ms": open_ms,
                "open_reqs": open_reqs,
                "cold_tile_ms": cold_tile_ms,
                "tile_reqs": total_reqs - open_reqs,
                "total_reqs": total_reqs,
                "total_bytes": total_bytes,
            },
        })
    finally:
        srv.shutdown()

    # ------------------------------------------------------------------------
    # BM-09: Landsat C2 3-File uint16 RGB (warm), BM-10: OPERA RTC-S1 SAR-RGB (warm),
    # BM-11: Multi-Zoom Pyramid (z=5, 8, 10, 12, 14, 16)
    # ------------------------------------------------------------------------
    from rio_tiler.utils import render as rio_render
    if (FIXTURES / "landsat_c2_B4.tif").exists():
        with rasterio.Env(GDAL_CACHEMAX=512):
            with (
                RioReader(str(FIXTURES / "landsat_c2_B4.tif")) as r4,
                RioReader(str(FIXTURES / "landsat_c2_B3.tif")) as r3,
                RioReader(str(FIXTURES / "landsat_c2_B2.tif")) as r2,
            ):
                for _ in range(3):
                    a4 = r4.tile(652, 1583, 12, resampling_method="bilinear")
                    a3 = r3.tile(652, 1583, 12, resampling_method="bilinear")
                    a2 = r2.tile(652, 1583, 12, resampling_method="bilinear")
                    arr = np.stack([a4.data[0], a3.data[0], a2.data[0]], axis=0).astype(np.float32)
                    u8 = np.clip((arr - 7500.0) * (255.0 / 15000.0), 0, 255).astype(np.uint8)
                    _ = rio_render(u8, mask=a4.mask, img_format="PNG")
                samples = []
                for _ in range(25):
                    t0 = time.perf_counter()
                    a4 = r4.tile(652, 1583, 12, resampling_method="bilinear")
                    a3 = r3.tile(652, 1583, 12, resampling_method="bilinear")
                    a2 = r2.tile(652, 1583, 12, resampling_method="bilinear")
                    arr = np.stack([a4.data[0], a3.data[0], a2.data[0]], axis=0).astype(np.float32)
                    u8 = np.clip((arr - 7500.0) * (255.0 / 15000.0), 0, 255).astype(np.uint8)
                    _ = rio_render(u8, mask=a4.mask, img_format="PNG")
                    samples.append(time.perf_counter() - t0)
                med, p99 = summarize_us(samples)
                print(f"BM-09 Landsat C2 3-File uint16 RGB (GDAL warm): median={med:7.1f} us")
                results.append({"id": "BM-09-LandsatC2-warm", "median_us": med, "p99_us": p99})

                for z, x, y in [(5, 5, 12), (8, 40, 98), (10, 163, 395), (12, 652, 1583), (14, 2608, 6332), (16, 10432, 25328)]:
                    samples = []
                    for _ in range(10):
                        t0 = time.perf_counter()
                        a4 = r4.tile(x, y, z, resampling_method="bilinear")
                        a3 = r3.tile(x, y, z, resampling_method="bilinear")
                        a2 = r2.tile(x, y, z, resampling_method="bilinear")
                        arr = np.stack([a4.data[0], a3.data[0], a2.data[0]], axis=0).astype(np.float32)
                        u8 = np.clip((arr - 7500.0) * (255.0 / 15000.0), 0, 255).astype(np.uint8)
                        _ = rio_render(u8, mask=a4.mask, img_format="PNG")
                        samples.append(time.perf_counter() - t0)
                    mz, pz = summarize_us(samples)
                    print(f"BM-11 MultiZoom Landsat C2 z={z:2d} (GDAL warm): median={mz:7.1f} us")
                    results.append({"id": f"BM-11-z{z}", "median_us": mz, "p99_us": pz})

            with (
                RioReader(str(FIXTURES / "opera_rtc_s1_VV.tif")) as rvv,
                RioReader(str(FIXTURES / "opera_rtc_s1_VH.tif")) as rvh,
            ):
                samples = []
                for _ in range(25):
                    t0 = time.perf_counter()
                    a1 = rvv.tile(652, 1583, 12, resampling_method="bilinear")
                    a2 = rvh.tile(652, 1583, 12, resampling_method="bilinear")
                    vv = a1.data[0].astype(np.float32)
                    vh = a2.data[0].astype(np.float32)
                    vv_db = 10.0 * np.log10(np.maximum(vv, 1e-10))
                    vh_db = 10.0 * np.log10(np.maximum(vh, 1e-10))
                    r = np.clip((vv_db + 20.0) * (255.0 / 20.0), 0, 255).astype(np.uint8)
                    g = np.clip((vh_db + 25.0) * (255.0 / 20.0), 0, 255).astype(np.uint8)
                    b = np.clip((vv_db - vh_db) * (255.0 / 15.0), 0, 255).astype(np.uint8)
                    _ = rio_render(np.stack([r, g, b], axis=0), mask=a1.mask, img_format="PNG")
                    samples.append(time.perf_counter() - t0)
                med, p99 = summarize_us(samples)
                print(f"BM-10 OPERA RTC-S1 Dual-Pol SAR-RGB (GDAL warm): median={med:7.1f} us")
                results.append({"id": "BM-10-OperaSAR-RGB-warm", "median_us": med, "p99_us": p99})

    return results


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    parity = verify_parity()
    gdal_results = run_gdal_benchmarks()
    out = {"parity": parity, "gdal": gdal_results}
    (RESULTS / "gdal_results.json").write_text(json.dumps(out, indent=2))
    print(f"Saved GDAL results to {RESULTS / 'gdal_results.json'}")


if __name__ == "__main__":
    main()
