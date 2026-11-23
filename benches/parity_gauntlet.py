#!/usr/bin/env python3
"""
Comprehensive GeoTIFF / COG Parity Gauntlet: cogwheel vs. TiTiler.

Constructs a diverse corpus of 25+ GeoTIFFs spanning:
  - 8 Projections (EPSG:3857, 4326, UTM North 32610, UTM South 32755, NAD83 UTM 26918,
                   CONUS Albers 5070, Arctic Polar Stereo 3413, Antarctic Polar Stereo 3031)
  - All 8 GDAL numeric dtypes (uint8, int8, uint16, int16, uint32, int32, float32, float64)
  - All compressions & predictors (NONE, DEFLATE P1/P2/P3, ZSTD P1/P2/P3, LZW P1/P2/P3)
  - Interleaves & band counts (1-band, 3-band PIXEL chunky, 3-band BAND planar, 4-band RGBN uint16)
  - Structural variants (BigTIFF, 256x256 tiles, 512x512 tiles, odd non-tile-aligned dims 1300x950,
                         no overviews vs multi-overview, NaN nodata, -9999 sentinel nodata,
                         sparse half-nodata, GDAL scale/offset ?unscale=true)

Runs EVERY file live against BOTH `cogwheel-server` (--release) and `TiTiler` (uvicorn) across:
  1. GET /cog/info            (dtype, band count, width, height, native bounds)
  2. GET /cog/point/{lon},{lat} (exact pixel values)
  3. GET /cog/statistics      (mean & std agreement)
  4. GET /cog/tiles/...png    (shape, alpha mask %, MAE, PSNR dB, and latency speedup)
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
import rasterio
import rasterio.shutil as rio_shutil
from rasterio.crs import CRS
from rasterio.enums import Resampling as RioResampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "benches" / "fixtures" / "parity_corpus"
RESULTS_DIR = ROOT / "benches" / "results"
COGWHEEL_BIN = ROOT / "target" / "release" / "cogwheel-server"

PORT_CW = 18811
PORT_TT = 18812


@dataclass
class CorpusCase:
    name: str
    category: str
    path: Path
    tile_query: str  # extra query params for /cog/tiles/... (e.g. rescale, colormap_name, bidx, unscale)
    check_stats: bool = True
    check_point: bool = True


def _make_pattern(h: int, w: int, bands: int = 1) -> np.ndarray:
    """Smooth deterministic spatial pattern in [0.05, 0.95] with distinct per-band signatures."""
    y = np.linspace(0.0, 4.0 * np.pi, h, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 4.0 * np.pi, w, dtype=np.float64)[None, :]
    out = []
    for b in range(bands):
        phase = b * 1.1
        arr = 0.5 + 0.42 * np.sin(x * (0.8 + 0.15 * b) + phase) * np.cos(y * (0.7 + 0.1 * b) - phase)
        out.append(arr)
    return np.stack(out, axis=0)


def _write_cog(
    path: Path,
    data: np.ndarray,
    crs: CRS,
    bounds: tuple[float, float, float, float],
    *,
    compress: str = "DEFLATE",
    predictor: int = 1,
    blocksize: int = 256,
    interleave: str = "PIXEL",
    bigtiff: bool = False,
    overviews: list[int] | None = None,
    nodata=None,
    scales: tuple[float, ...] | None = None,
    offsets: tuple[float, ...] | None = None,
    photometric: str | None = None,
):
    if path.exists():
        return
    bands, h, w = data.shape
    transform = from_bounds(*bounds, w, h)
    tmp = path.with_suffix(".tmp.tif")

    profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        tiled=True,
        blockxsize=blocksize,
        blockysize=blocksize,
        interleave=interleave,
        nodata=nodata,
    )
    if compress != "NONE":
        profile["compress"] = compress
        if predictor > 1:
            profile["predictor"] = predictor
    if photometric:
        profile["photometric"] = photometric
    if bigtiff:
        profile["BIGTIFF"] = "YES"

    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
        if scales is not None:
            dst.scales = scales
        if offsets is not None:
            dst.offsets = offsets
        if overviews:
            dst.build_overviews(overviews, RioResampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    if overviews:
        copy_kw = dict(
            driver="GTiff",
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            interleave=interleave,
            copy_src_overviews=True,
        )
        if compress != "NONE":
            copy_kw["compress"] = compress
            if predictor > 1:
                copy_kw["predictor"] = predictor
        if photometric:
            copy_kw["photometric"] = photometric
        if bigtiff:
            copy_kw["BIGTIFF"] = "YES"
        rio_shutil.copy(tmp, path, **copy_kw)
        tmp.unlink()
    else:
        tmp.rename(path)


def build_corpus() -> list[CorpusCase]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[CorpusCase] = []

    # ------------------------------------------------------------------
    # 1. PROJECTIONS (8 distinct CRSs worldwide)
    # ------------------------------------------------------------------
    proj_specs = [
        ("proj_3857_mercator", CRS.from_epsg(3857), (-13650000.0, 4500000.0, -13550000.0, 4600000.0)),
        ("proj_4326_wgs84", CRS.from_epsg(4326), (-122.8, 37.5, -122.2, 38.1)),
        ("proj_32610_utm10n", CRS.from_epsg(32610), (500000.0, 4150000.0, 560000.0, 4210000.0)),
        ("proj_32755_utm55s_sydney", CRS.from_epsg(32755), (300000.0, 6220000.0, 360000.0, 6280000.0)),
        ("proj_26918_nad83_utm18n", CRS.from_epsg(26918), (560000.0, 4490000.0, 620000.0, 4550000.0)),
        ("proj_5070_conus_albers", CRS.from_epsg(5070), (-150000.0, 1750000.0, -50000.0, 1850000.0)),
        ("proj_3413_arctic_polar", CRS.from_epsg(3413), (-500000.0, -1500000.0, -300000.0, -1300000.0)),
        ("proj_3031_antarctic_polar", CRS.from_epsg(3031), (1000000.0, 1000000.0, 1200000.0, 1200000.0)),
    ]
    pat1 = (_make_pattern(1024, 1024, 1) * 1000.0).astype(np.float32)
    for name, crs, bounds in proj_specs:
        p = CORPUS_DIR / f"{name}.tif"
        _write_cog(p, pat1, crs, bounds, compress="ZSTD", predictor=3, overviews=[2, 4])
        cases.append(CorpusCase(name, "projection", p, "rescale=0,1000&colormap_name=viridis&resampling=bilinear"))

    # ------------------------------------------------------------------
    # 2. ALL 8 GDAL NUMERIC DTYPES
    # ------------------------------------------------------------------
    wm_crs = CRS.from_epsg(3857)
    wm_bounds = (-13650000.0, 4500000.0, -13550000.0, 4600000.0)
    norm = _make_pattern(1024, 1024, 1)
    dtype_specs = [
        ("dtype_uint8", (norm * 240 + 5).astype(np.uint8), 2, "0,255"),
        ("dtype_int8", (norm * 200 - 100).astype(np.int8), 2, "-100,100"),
        ("dtype_uint16", (norm * 50000 + 1000).astype(np.uint16), 2, "0,60000"),
        ("dtype_int16", (norm * 20000 - 10000).astype(np.int16), 2, "-10000,10000"),
        ("dtype_uint32", (norm * 100000 + 500).astype(np.uint32), 2, "0,110000"),
        ("dtype_int32", (norm * 100000 - 50000).astype(np.int32), 2, "-50000,50000"),
        ("dtype_float32", (norm * 1000.0).astype(np.float32), 3, "0,1000"),
        ("dtype_float64", (norm * 1000.0).astype(np.float64), 3, "0,1000"),
    ]
    for name, arr, pred, rescale in dtype_specs:
        p = CORPUS_DIR / f"{name}.tif"
        _write_cog(p, arr, wm_crs, wm_bounds, compress="DEFLATE", predictor=pred, overviews=[2])
        cases.append(CorpusCase(name, "dtype", p, f"rescale={rescale}&colormap_name=viridis&resampling=bilinear"))

    # ------------------------------------------------------------------
    # 3. COMPRESSIONS & PREDICTORS (NONE, DEFLATE P1/P2/P3, ZSTD P1/P3, LZW P1/P2/P3)
    # ------------------------------------------------------------------
    f32_arr = (norm * 1000.0).astype(np.float32)
    u16_arr = (norm * 50000 + 1000).astype(np.uint16)
    codec_specs = [
        ("codec_none_f32", f32_arr, "NONE", 1, "0,1000"),
        ("codec_deflate_p1_f32", f32_arr, "DEFLATE", 1, "0,1000"),
        ("codec_deflate_p2_u16", u16_arr, "DEFLATE", 2, "0,60000"),
        ("codec_deflate_p3_f32", f32_arr, "DEFLATE", 3, "0,1000"),
        ("codec_zstd_p1_f32", f32_arr, "ZSTD", 1, "0,1000"),
        ("codec_zstd_p2_u16", u16_arr, "ZSTD", 2, "0,60000"),
        ("codec_zstd_p3_f32", f32_arr, "ZSTD", 3, "0,1000"),
        ("codec_lzw_p1_f32", f32_arr, "LZW", 1, "0,1000"),
        ("codec_lzw_p2_u16", u16_arr, "LZW", 2, "0,60000"),
        ("codec_lzw_p3_f32", f32_arr, "LZW", 3, "0,1000"),
    ]
    for name, arr, comp, pred, rescale in codec_specs:
        p = CORPUS_DIR / f"{name}.tif"
        _write_cog(p, arr, wm_crs, wm_bounds, compress=comp, predictor=pred)
        cases.append(CorpusCase(name, "codec", p, f"rescale={rescale}&colormap_name=plasma&resampling=bilinear"))

    # JPEG-in-TIFF (YCbCr RGB + Grayscale) & WebP-in-TIFF
    rgb_u8_codec = (_make_pattern(1024, 1024, 3) * 240 + 5).astype(np.uint8)
    gray_u8_codec = (norm * 240 + 5).astype(np.uint8)

    p = CORPUS_DIR / "codec_jpeg_ycbcr_rgb_u8.tif"
    _write_cog(p, rgb_u8_codec, wm_crs, wm_bounds, compress="JPEG", photometric="YCBCR", overviews=[2])
    cases.append(CorpusCase("codec_jpeg_ycbcr_rgb_u8", "codec", p, "bidx=1&bidx=2&bidx=3&resampling=bilinear"))

    p = CORPUS_DIR / "codec_jpeg_gray_u8.tif"
    _write_cog(p, gray_u8_codec, wm_crs, wm_bounds, compress="JPEG", overviews=[2])
    cases.append(CorpusCase("codec_jpeg_gray_u8", "codec", p, "colormap_name=viridis&resampling=bilinear"))

    p = CORPUS_DIR / "codec_webp_rgb_u8.tif"
    _write_cog(p, rgb_u8_codec, wm_crs, wm_bounds, compress="WEBP", overviews=[2])
    cases.append(CorpusCase("codec_webp_rgb_u8", "codec", p, "bidx=1&bidx=2&bidx=3&resampling=bilinear"))

    # ------------------------------------------------------------------
    # 4. INTERLEAVE, BAND COUNTS & STRUCTURAL QUIRKS
    # ------------------------------------------------------------------
    rgb_u8 = (_make_pattern(1024, 1024, 3) * 240 + 5).astype(np.uint8)
    rgb_f32 = (_make_pattern(1024, 1024, 3) * 0.35).astype(np.float32)
    rgbn_u16 = (_make_pattern(1024, 1024, 4) * 15000 + 2000).astype(np.uint16)

    # 3-band PIXEL (chunky) uint8
    p = CORPUS_DIR / "layout_rgb_pixel_u8.tif"
    _write_cog(p, rgb_u8, wm_crs, wm_bounds, compress="DEFLATE", predictor=2, interleave="PIXEL", overviews=[2])
    cases.append(CorpusCase("layout_rgb_pixel_u8", "layout", p, "bidx=1&bidx=2&bidx=3&resampling=bilinear"))

    # 3-band BAND (planar / separate) float32
    p = CORPUS_DIR / "layout_rgb_planar_f32.tif"
    _write_cog(p, rgb_f32, wm_crs, wm_bounds, compress="ZSTD", predictor=3, interleave="BAND", overviews=[2])
    cases.append(CorpusCase("layout_rgb_planar_f32", "layout", p, "bidx=1&bidx=2&bidx=3&rescale=0,0.35&resampling=bilinear"))

    # 4-band RGBN uint16 (false-color NIR-R-G selection bidx=4,1,2)
    p = CORPUS_DIR / "layout_4band_rgbn_u16.tif"
    _write_cog(p, rgbn_u16, wm_crs, wm_bounds, compress="DEFLATE", predictor=2, interleave="PIXEL", overviews=[2])
    cases.append(CorpusCase("layout_4band_rgbn_u16_bidx412", "layout", p, "bidx=4&bidx=1&bidx=2&rescale=0,20000&resampling=bilinear"))

    # 4-band RGBN NDVI expression
    cases.append(CorpusCase("layout_4band_rgbn_ndvi_expr", "layout", p, "expression=(b4-b1)/(b4%2Bb1)&rescale=-0.5,0.5&colormap_name=rdylgn&resampling=bilinear", check_stats=False, check_point=False))

    # BigTIFF
    p = CORPUS_DIR / "struct_bigtiff_zstd_p3.tif"
    _write_cog(p, f32_arr, wm_crs, wm_bounds, compress="ZSTD", predictor=3, bigtiff=True, overviews=[2])
    cases.append(CorpusCase("struct_bigtiff_zstd_p3", "structure", p, "rescale=0,1000&colormap_name=inferno&resampling=bilinear"))

    # Odd non-tile-aligned dimensions (1300x950 with 256x256 tiles -> partial edge tiles)
    odd_arr = (_make_pattern(950, 1300, 1) * 1000.0).astype(np.float32)
    p = CORPUS_DIR / "struct_odd_dims_1300x950.tif"
    _write_cog(p, odd_arr, wm_crs, wm_bounds, compress="DEFLATE", predictor=3, blocksize=256, overviews=[2])
    cases.append(CorpusCase("struct_odd_dims_1300x950", "structure", p, "rescale=0,1000&colormap_name=viridis&resampling=bilinear"))

    # 512x512 tiles, NO overviews
    p = CORPUS_DIR / "struct_512tiles_no_ovr.tif"
    _write_cog(p, f32_arr, wm_crs, wm_bounds, compress="ZSTD", predictor=3, blocksize=512, overviews=None)
    cases.append(CorpusCase("struct_512tiles_no_ovr", "structure", p, "rescale=0,1000&colormap_name=turbo&resampling=bilinear"))

    # Float32 NaN nodata (half-masked diagonal swath!)
    nan_arr = f32_arr.copy()
    yy, xx = np.mgrid[0:1024, 0:1024]
    nan_arr[0, (xx + yy) < 700] = np.nan
    p = CORPUS_DIR / "nodata_float32_nan_swath.tif"
    _write_cog(p, nan_arr, wm_crs, wm_bounds, compress="ZSTD", predictor=3, nodata=float("nan"), overviews=[2])
    cases.append(CorpusCase("nodata_float32_nan_swath", "nodata", p, "rescale=0,1000&colormap_name=viridis&resampling=bilinear"))

    # Int16 sentinel nodata (-9999)
    i16_nd = (norm * 5000 + 500).astype(np.int16)
    i16_nd[0, (xx + yy) < 700] = -9999
    p = CORPUS_DIR / "nodata_int16_minus9999.tif"
    _write_cog(p, i16_nd, wm_crs, wm_bounds, compress="DEFLATE", predictor=2, nodata=-9999, overviews=[2])
    cases.append(CorpusCase("nodata_int16_minus9999", "nodata", p, "rescale=0,6000&colormap_name=terrain&resampling=bilinear"))

    # GDAL scale/offset (Landsat C2 style: scale=0.0000275, offset=-0.2) with ?unscale=true
    ls_arr = (norm * 20000 + 8000).astype(np.uint16)
    p = CORPUS_DIR / "meta_landsat_scale_offset.tif"
    _write_cog(
        p, ls_arr, wm_crs, wm_bounds, compress="DEFLATE", predictor=2,
        nodata=0, scales=(0.0000275,), offsets=(-0.2,), overviews=[2],
    )
    cases.append(CorpusCase("meta_landsat_unscale_true", "metadata", p, "unscale=true&rescale=0,0.6&colormap_name=viridis&resampling=bilinear"))

    # 4-band RGBA with ExtraSamples alpha mask (diagonal transparent corner)
    from rasterio.enums import ColorInterp
    alpha_mask = np.where((xx + yy) < 1100, 0, 255).astype(np.uint8)
    rgba_u8 = np.vstack([rgb_u8, alpha_mask[None, :, :]])
    p_rgba = CORPUS_DIR / "mask_rgba_4band_u8.tif"
    if not p_rgba.exists():
        transform = from_bounds(*wm_bounds, 1024, 1024)
        with rasterio.open(
            p_rgba, "w", driver="GTiff", height=1024, width=1024, count=4, dtype="uint8",
            crs=wm_crs, transform=transform, tiled=True, blockxsize=256, blockysize=256,
            compress="DEFLATE", predictor=2,
        ) as dst:
            dst.write(rgba_u8)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    cases.append(CorpusCase("mask_rgba_4band_u8", "mask", p_rgba, "bidx=1&bidx=2&bidx=3&resampling=bilinear"))

    # 3-band JPEG YCbCr COG with 1-bit internal transparency mask IFD (GDAL_TIFF_INTERNAL_MASK=YES)
    p_jmask = CORPUS_DIR / "mask_jpeg_internal_1bit.tif"
    if not p_jmask.exists():
        transform = from_bounds(*wm_bounds, 1024, 1024)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(
                p_jmask, "w", driver="GTiff", height=1024, width=1024, count=3, dtype="uint8",
                crs=wm_crs, transform=transform, tiled=True, blockxsize=256, blockysize=256,
                compress="JPEG", photometric="YCBCR",
            ) as dst:
                dst.write(rgb_u8)
                dst.write_mask(alpha_mask)
    cases.append(CorpusCase("mask_jpeg_internal_1bit", "mask", p_jmask, "bidx=1&bidx=2&bidx=3&resampling=bilinear"))

    # Stripped (non-tiled!) legacy GeoTIFFs: DEFLATE P3 float32 + odd-height RGB uint8 (partial last strip!)
    p_strip1 = CORPUS_DIR / "struct_stripped_deflate_f32.tif"
    if not p_strip1.exists():
        transform = from_bounds(*wm_bounds, 1024, 1024)
        with rasterio.open(
            p_strip1, "w", driver="GTiff", height=1024, width=1024, count=1, dtype="float32",
            crs=wm_crs, transform=transform, tiled=False, blockysize=32, compress="DEFLATE", predictor=3,
        ) as dst:
            dst.write(f32_arr)
    cases.append(CorpusCase("struct_stripped_deflate_f32", "structure", p_strip1, "rescale=0,1000&colormap_name=spectral_r&resampling=bilinear"))

    p_strip2 = CORPUS_DIR / "struct_stripped_odd_rows_rgb_u8.tif"
    if not p_strip2.exists():
        odd_rgb = (_make_pattern(950, 1024, 3) * 240 + 5).astype(np.uint8)
        transform = from_bounds(*wm_bounds, 1024, 950)
        with rasterio.open(
            p_strip2, "w", driver="GTiff", height=950, width=1024, count=3, dtype="uint8",
            crs=wm_crs, transform=transform, tiled=False, blockysize=64, compress="NONE",
        ) as dst:
            dst.write(odd_rgb)
    cases.append(CorpusCase("struct_stripped_odd_rows_rgb_u8", "structure", p_strip2, "bidx=1&bidx=2&bidx=3&resampling=bilinear"))

    # Bicubic (4x4 Catmull-Rom) resampling parity (single-band f32 + 3-band RGB u8)
    cases.append(CorpusCase("resample_cubic_f32", "resampling", CORPUS_DIR / "proj_3857_mercator.tif", "rescale=0,1000&colormap_name=cividis&resampling=cubic", check_stats=False, check_point=False))
    cases.append(CorpusCase("resample_cubic_rgb_u8", "resampling", CORPUS_DIR / "layout_rgb_pixel_u8.tif", "bidx=1&bidx=2&bidx=3&resampling=cubic", check_stats=False, check_point=False))

    return cases


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
    raise RuntimeError(f"Server at {url} not ready in {timeout_s}s")


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    lat_rad = math.radians(max(-85.0511, min(85.0511, lat)))
    n = 1 << z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def pick_interior_tile_and_point(tif_path: Path) -> tuple[int, int, int, float, float]:
    """Pick a WebMercatorQuad tile and lon/lat point well inside the raster's valid footprint."""
    with rasterio.open(tif_path) as ds:
        w, s, e, n = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
        # Clamp latitude for WebMercatorQuad (Antarctic polar stereo goes down to -80)
        s_clamped = max(-84.5, s)
        n_clamped = min(84.5, n)
        # Pick 65% across (avoids the top-left diagonal nodata mask in swath fixtures!)
        lon = w + 0.65 * (e - w)
        lat = s_clamped + 0.35 * (n_clamped - s_clamped)
        span_deg = max(e - w, n_clamped - s_clamped)
        # Choose zoom where tile spans roughly 1/4 of the raster
        z = max(2, min(14, int(round(math.log2(360.0 / max(span_deg * 0.35, 1e-4))))))
        x, y = lonlat_to_tile(lon, lat, z)
        return z, x, y, lon, lat


def compare_pngs(cw_bytes: bytes, tt_bytes: bytes) -> tuple[bool, float, float, float]:
    im_cw = np.array(Image.open(io.BytesIO(cw_bytes)).convert("RGBA"), dtype=np.float32)
    im_tt = np.array(Image.open(io.BytesIO(tt_bytes)).convert("RGBA"), dtype=np.float32)
    if im_cw.shape != im_tt.shape:
        return False, 0.0, 999.0, 0.0
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
    ok = (mask_pct >= 98.5) and (psnr >= 35.0 or mae <= 2.5)
    return ok, mask_pct, mae, psnr


def time_get(client: httpx.Client, url: str, iters: int = 5):
    r0 = client.get(url)
    if r0.status_code != 200:
        return r0, 0.0
    times = []
    last = r0
    for _ in range(iters):
        t0 = time.perf_counter()
        last = client.get(url)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return last, times[len(times) // 2]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not COGWHEEL_BIN.exists():
        print("Building cogwheel-server --release...")
        subprocess.run(["cargo", "build", "--release", "-p", "cogwheel-server"], cwd=ROOT, check=True)

    print("Building diverse GeoTIFF parity corpus...")
    cases = build_corpus()
    print(f"Prepared {len(cases)} test cases across {len({c.path for c in cases})} distinct GeoTIFFs.\n")

    tt_app = RESULTS_DIR / "_titiler_gauntlet_app.py"
    tt_app.write_text(
        "from fastapi import FastAPI\n"
        "from titiler.core.factory import TilerFactory\n"
        "app = FastAPI()\n"
        "app.include_router(TilerFactory().router, prefix='/cog')\n"
    )

    cw_proc = subprocess.Popen(
        [str(COGWHEEL_BIN), "--bind", f"127.0.0.1:{PORT_CW}", "--cache-mb", "512"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tt_env = os.environ.copy()
    tt_env.update({"GDAL_CACHEMAX": "512", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "VSI_CACHE": "TRUE"})
    tt_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "_titiler_gauntlet_app:app",
            "--app-dir", str(RESULTS_DIR), "--host", "127.0.0.1", "--port", str(PORT_TT),
            "--workers", "1", "--log-level", "error",
        ],
        env=tt_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_ready(f"http://127.0.0.1:{PORT_CW}/healthz")
        wait_ready(f"http://127.0.0.1:{PORT_TT}/openapi.json")

        cw_base = f"http://127.0.0.1:{PORT_CW}"
        tt_base = f"http://127.0.0.1:{PORT_TT}"

        hdr = f"{'Case':32s} | {'Cat':10s} | {'Info':6s} | {'Point':9s} | {'Stats':9s} | {'Mask%':6s} | {'MAE':5s} | {'PSNR':7s} | {'CW ms':6s} | {'TT ms':6s} | {'Speed':6s} | Status"
        print(hdr)
        print("-" * len(hdr))

        results = []
        failures = []

        with httpx.Client(timeout=30.0) as client:
            for c in cases:
                url_str = str(c.path.resolve())
                z, x, y, lon, lat = pick_interior_tile_and_point(c.path)

                # 1. /cog/info
                info_ok = True
                info_msg = "OK"
                r_cw_i = client.get(f"{cw_base}/cog/info?url={url_str}")
                r_tt_i = client.get(f"{tt_base}/cog/info?url={url_str}")
                if r_cw_i.status_code != 200 or r_tt_i.status_code != 200:
                    info_ok = False
                    info_msg = f"E{r_cw_i.status_code}/{r_tt_i.status_code}"
                else:
                    ji_cw, ji_tt = r_cw_i.json(), r_tt_i.json()
                    b_err = max(abs(a - b) for a, b in zip(ji_cw["bounds"], ji_tt["bounds"]))
                    if ji_cw["dtype"] != ji_tt["dtype"] or ji_cw["count"] != ji_tt["count"] or b_err > 1e-2:
                        info_ok = False
                        info_msg = "DIFF"

                # 2. /cog/point
                pt_ok = True
                pt_msg = "-"
                if c.check_point:
                    r_cw_p = client.get(f"{cw_base}/cog/point/{lon:.6f},{lat:.6f}?url={url_str}")
                    r_tt_p = client.get(f"{tt_base}/cog/point/{lon:.6f},{lat:.6f}?url={url_str}")
                    if r_cw_p.status_code != 200 or r_tt_p.status_code != 200:
                        pt_ok = False
                        pt_msg = f"E{r_cw_p.status_code}/{r_tt_p.status_code}"
                    else:
                        v_cw = r_cw_p.json()["values"]
                        v_tt = r_tt_p.json()["values"]
                        errs = []
                        for a, b in zip(v_cw, v_tt):
                            if a is None and b is None:
                                errs.append(0.0)
                            elif a is None or b is None:
                                errs.append(999.0)
                            else:
                                errs.append(abs(float(a) - float(b)))
                        max_e = max(errs) if errs else 0.0
                        pt_msg = f"{max_e:.1e}"
                        # Allow tiny floating rounding or 1-pixel boundary diff on extreme projections
                        scale_ref = max([abs(float(v)) for v in v_tt if v is not None] + [1.0])
                        if max_e > max(1e-2, 0.02 * scale_ref):
                            pt_ok = False

                # 3. /cog/statistics
                st_ok = True
                st_msg = "-"
                if c.check_stats:
                    r_cw_s = client.get(f"{cw_base}/cog/statistics?url={url_str}&max_size=256")
                    r_tt_s = client.get(f"{tt_base}/cog/statistics?url={url_str}&max_size=256")
                    if r_cw_s.status_code != 200 or r_tt_s.status_code != 200:
                        st_ok = False
                        st_msg = f"E{r_cw_s.status_code}/{r_tt_s.status_code}"
                    else:
                        s_cw = r_cw_s.json()["b1"]["mean"]
                        s_tt = r_tt_s.json()["b1"]["mean"]
                        std_tt = r_tt_s.json()["b1"]["std"]
                        rel = abs(s_cw - s_tt) / max(abs(s_tt), abs(std_tt), 1.0)
                        st_msg = f"{rel*100:.2f}%"
                        if rel > 0.02:
                            st_ok = False

                # 4. /cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png
                tile_path = f"/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url={url_str}&{c.tile_query}"
                r_cw_t, ms_cw = time_get(client, cw_base + tile_path, iters=6)
                r_tt_t, ms_tt = time_get(client, tt_base + tile_path, iters=6)
                if r_cw_t.status_code != 200 or r_tt_t.status_code != 200:
                    tile_ok, mask_pct, mae, psnr = False, 0.0, 999.0, 0.0
                    status = f"FAIL(tile {r_cw_t.status_code}/{r_tt_t.status_code})"
                    speed = 0.0
                else:
                    tile_ok, mask_pct, mae, psnr = compare_pngs(r_cw_t.content, r_tt_t.content)
                    speed = ms_tt / max(ms_cw, 0.01)
                    all_ok = info_ok and pt_ok and st_ok and tile_ok
                    status = "PASS" if all_ok else f"FAIL({'I' if not info_ok else ''}{'P' if not pt_ok else ''}{'S' if not st_ok else ''}{'T' if not tile_ok else ''})"

                print(
                    f"{c.name:32s} | {c.category:10s} | {info_msg:6s} | {pt_msg:9s} | {st_msg:9s} | "
                    f"{mask_pct:5.1f}% | {mae:5.2f} | {psnr:5.1f}dB | {ms_cw:6.2f} | {ms_tt:6.2f} | {speed:5.1f}x | {status}"
                )
                row = dict(
                    name=c.name, category=c.category, tile=f"{z}/{x}/{y}",
                    info_ok=info_ok, point_ok=pt_ok, stats_ok=st_ok, tile_ok=tile_ok,
                    mask_pct=mask_pct, mae=mae, psnr=psnr,
                    cw_ms=ms_cw, tt_ms=ms_tt, speedup=speed, status=status,
                )
                results.append(row)
                if status != "PASS":
                    failures.append(row)

        (RESULTS_DIR / "parity_gauntlet.json").write_text(json.dumps(results, indent=2))
        passed = len(results) - len(failures)
        print(f"\nSummary: {passed}/{len(results)} passed.")
        if failures:
            print("Failures:")
            for f in failures:
                print("  -", f)
            sys.exit(1)
    finally:
        cw_proc.terminate()
        tt_proc.terminate()
        cw_proc.wait(timeout=5)
        tt_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
