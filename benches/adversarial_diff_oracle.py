#!/usr/bin/env python3
"""
Adversarial Differential Oracle: proves cogzilla is EQUAL or INTENTIONALLY BETTER
than stock TiTiler 2.4.1 across ~240 hard edge cases and synthetic torture COGs.

Every check is classified into one of two strict contracts:
  - PARITY: cogzilla MUST match TiTiler 2.4.1 within tight tolerances (status, shape,
    dtype, pixel PSNR/MAE, stats, point values, GeoTIFF georeferencing).
  - INTENTIONALLY_BETTER: cogzilla MUST satisfy its documented improved contract
    (docs/titiler-divergences.md) while confirming TiTiler's known limitation.

Supports:
  --cpu-only           Run cogzilla with COGZILLA_GPU_POLICY=never (primary CPU path)
  --check-cpu-vs-gpu   Also compare CPU-only cogzilla vs default GPU-enabled cogzilla
  --only SUBSTR        Filter checks by substring
  --json-out PATH      Write machine-readable JSON report

Exits 0 iff 100% of checks pass (zero unexplained discrepancies).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import warnings
import zlib
from pathlib import Path
from urllib.parse import quote

import httpx
import mercantile
import numpy as np
from PIL import Image
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS as RasterioCRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "benches" / "fixtures"
CORPUS = FIXTURES / "parity_corpus"
EXTRA = FIXTURES / "oracle_extra"
DTYPES = FIXTURES / "dtypes"
RESULTS = ROOT / "benches" / "results"
COGZILLA_BIN = ROOT / "target" / "release" / "cogzilla-server"

sys.path.insert(0, str(ROOT / "benches"))
from oracle_lib import (
    compare_geotiffs,
    compare_images,
    compare_npy,
    first_json_diff,
    same_place,
    wait_ready,
)


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def q(path: Path) -> str:
    return quote(str(path.resolve()), safe="")


def qe(s: str) -> str:
    return quote(s, safe="")


def center_lonlat_and_tile(path: Path, z: int | None = None) -> tuple[float, float, int, int, int, tuple[float, float, float, float]]:
    with rasterio.open(path) as ds:
        wgs = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
    minx, miny, maxx, maxy = wgs
    lon = 0.5 * (minx + maxx)
    lat = max(-84.0, min(84.0, 0.5 * (miny + maxy)))
    if z is None:
        span = max(maxx - minx, maxy - miny, 1e-4)
        z = max(1, min(14, int(round(math.log2(360.0 / span))) + 1))
    n = 1 << z
    tx = int((lon + 180.0) / 360.0 * n) % n
    lat_rad = math.radians(lat)
    ty = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    ty = max(0, min(n - 1, ty))
    return lon, lat, z, tx, ty, (minx, miny, maxx, maxy)


def build_synthetic_fixtures(synth_dir: Path) -> dict[str, Path]:
    synth_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    tf_sf = from_bounds(-13630000.0, 4530000.0, -13600000.0, 4560000.0, 256, 256)

    # Rotated 15-deg GeoTIFF
    p = synth_dir / "rotated_15deg_3857.tif"
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    base = Affine.translation(-13630000.0, 4560000.0) * Affine.scale(100.0, -100.0) * Affine.rotation(15.0)
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=base, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write((xx * 2.0 + yy * 3.0 + 10.0)[None, ...])
    out["rotated"] = p

    # South-up (positive e) GeoTIFF
    p = synth_dir / "south_up_4326.tif"
    tf_su = Affine(0.01, 0.0, -123.0, 0.0, 0.01, 37.0)
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:4326", transform=tf_su, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write((xx + yy * 2.0 + 5.0)[None, ...])
    out["south_up"] = p

    # Ungeoreferenced RGB u8
    p = synth_dir / "ungeoref_rgb_u8.tif"
    yy8, xx8 = np.mgrid[0:128, 0:128]
    with rasterio.open(p, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8", tiled=True, blockxsize=128, blockysize=128, compress="deflate") as ds:
        ds.write(np.stack([(xx8 * 2).astype(np.uint8), (yy8 * 2).astype(np.uint8), ((xx8 + yy8) % 256).astype(np.uint8)], axis=0))
    out["ungeoref"] = p

    # High-precision int32 (> 2^24)
    p = synth_dir / "highprec_int32.tif"
    yy32, xx32 = np.mgrid[0:256, 0:256].astype(np.int32)
    arr32 = np.int32(100_000_000) + xx32 + yy32 * np.int32(256)
    arr32[0:32, 0:32] = -2147483648
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="int32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-2147483648) as ds:
        ds.write(arr32[None, ...])
    out["highprec_i32"] = p

    # Per-band distinct scale & offset (3-band uint16)
    p = synth_dir / "perband_scale_offset_u16.tif"
    yy16, xx16 = np.mgrid[0:256, 0:256].astype(np.uint16)
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint16", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=0) as ds:
        ds.write(np.stack([(xx16 * 10 + 100).astype(np.uint16), (yy16 * 20 + 500).astype(np.uint16), ((xx16 + yy16) * 5 + 50).astype(np.uint16)], axis=0))
        ds.scales = (0.01, 0.001, 0.1)
        ds.offsets = (-10.0, 5.0, -2.0)
    out["perband_so"] = p

    # Sparse COG (SPARSE_OK=TRUE)
    p = synth_dir / "sparse_i16.tif"
    tf_sp = from_bounds(-13640000.0, 4520000.0, -13600000.0, 4560000.0, 512, 512)
    with rasterio.open(p, "w", driver="GTiff", width=512, height=512, count=1, dtype="int16", crs="EPSG:3857", transform=tf_sp, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999, SPARSE_OK="TRUE") as ds:
        yy_s, xx_s = np.mgrid[0:256, 0:256].astype(np.int16)
        ds.write((xx_s + yy_s + 100).astype(np.int16), 1, window=rasterio.windows.Window(0, 0, 256, 256))
    out["sparse_i16"] = p

    # Categorical uint8
    p = synth_dir / "categorical_u8.tif"
    classes = np.array([0, 1, 2, 3, 5, 10], dtype=np.uint8)
    yy_c, xx_c = np.mgrid[0:256, 0:256]
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="uint8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=0) as ds:
        ds.write(classes[((xx_c // 32) + (yy_c // 32)) % len(classes)][None, ...])
    out["cat_u8"] = p

    # Negative float32 with FLT_MIN nodata
    p = synth_dir / "neg_f32_min_nodata.tif"
    nd_fmin = float(np.finfo(np.float32).min)
    arr_nf = ((xx - 128.0) * 0.5 + (yy - 128.0) * 0.25).astype(np.float32)
    arr_nf[0:64, 0:64] = nd_fmin
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=nd_fmin) as ds:
        ds.write(arr_nf[None, ...])
    out["neg_f32"] = p

    # Tiny 1x1 & 3x3
    p = synth_dir / "tiny_1x1.tif"
    with rasterio.open(p, "w", driver="GTiff", width=1, height=1, count=1, dtype="float32", crs="EPSG:3857", transform=from_bounds(-13630000.0, 4530000.0, -13600000.0, 4560000.0, 1, 1), nodata=-9999.0) as ds:
        ds.write(np.array([[[123.5]]], dtype=np.float32))
    out["tiny_1x1"] = p

    p = synth_dir / "tiny_3x3.tif"
    with rasterio.open(p, "w", driver="GTiff", width=3, height=3, count=1, dtype="float32", crs="EPSG:3857", transform=from_bounds(-13630000.0, 4530000.0, -13600000.0, 4560000.0, 3, 3), nodata=-9999.0) as ds:
        ds.write(np.arange(1, 10, dtype=np.float32).reshape(1, 3, 3) * 10.0)
    out["tiny_3x3"] = p

    # Constant 42.0 & All-nodata
    p = synth_dir / "constant_42.tif"
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write(np.full((1, 256, 256), 42.0, dtype=np.float32))
    out["constant"] = p

    p = synth_dir / "all_nodata.tif"
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write(np.full((1, 256, 256), -9999.0, dtype=np.float32))
    out["all_nodata"] = p

    # Sentinel boundary nodata: int16(-32768), uint8(255), uint16(65535)
    for tag, dt, nd, lo, hi in [
        ("nd_i16_min", "int16", -32768, -1000, 1000),
        ("nd_u8_255", "uint8", 255, 0, 250),
        ("nd_u16_max", "uint16", 65535, 0, 60000),
    ]:
        p = synth_dir / f"{tag}.tif"
        arr = (lo + ((xx_c + yy_c) * (hi - lo)) // 510).astype(dt)
        arr[0:64, 0:64] = nd
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype=dt, crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=nd) as ds:
            ds.write(arr[None, ...])
        out[tag] = p

    # Float32 with +inf / -inf / NaN
    p = synth_dir / "f32_with_infs.tif"
    arr_inf = ((xx + yy) * 0.5).astype(np.float32)
    arr_inf[0:16, 0:16] = np.inf
    arr_inf[16:32, 0:16] = -np.inf
    arr_inf[32:48, 0:16] = np.nan
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=float("nan")) as ds:
        ds.write(arr_inf[None, ...])
    out["f32_infs"] = p

    # Planar RGB + internal 1-bit mask & Planar RGBA
    p = synth_dir / "planar_rgb_1bit_mask.tif"
    rgb = np.stack([xx_c.astype(np.uint8), yy_c.astype(np.uint8), ((xx_c + yy_c) // 2).astype(np.uint8)], axis=0)
    mask = np.full((256, 256), 255, dtype=np.uint8)
    mask[0:96, 0:96] = 0
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", interleave="band") as ds:
            ds.write(rgb)
            ds.write_mask(mask)
    out["planar_1bit"] = p

    p = synth_dir / "planar_rgba.tif"
    rgba = np.concatenate([rgb, mask[None, ...]], axis=0)
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=4, dtype="uint8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", interleave="band", photometric="rgb") as ds:
        ds.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green, rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha)
        ds.write(rgba)
    out["planar_rgba"] = p

    # Antimeridian-crossing EPSG:4326 COG (179..181)
    p = synth_dir / "antimeridian_4326.tif"
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:4326", transform=from_bounds(179.0, -18.0, 181.0, -16.0, 256, 256), tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write((xx + yy * 2.0)[None, ...])
    out["antimeridian"] = p

    # Mixed-dtype / mixed-nodata STAC item
    p_u8, p_f32, p_u16 = synth_dir / "stac_a_u8.tif", synth_dir / "stac_b_f32.tif", synth_dir / "stac_c_u16.tif"
    with rasterio.open(p_u8, "w", driver="GTiff", width=256, height=256, count=1, dtype="uint8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=0) as ds:
        a = ((xx_c // 2) + 10).astype(np.uint8); a[0:32, 0:32] = 0; ds.write(a[None, ...])
    with rasterio.open(p_f32, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        a = (yy * 0.5 + 1.0).astype(np.float32); a[224:256, 224:256] = -9999.0; ds.write(a[None, ...])
    with rasterio.open(p_u16, "w", driver="GTiff", width=256, height=256, count=1, dtype="uint16", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=65535) as ds:
        a = ((xx_c + yy_c) * 10 + 100).astype(np.uint16); a[0:32, 224:256] = 65535; ds.write(a[None, ...])
    p_stac_mix = synth_dir / "mixed_dtype_nodata_stac.json"
    p_stac_mix.write_text(json.dumps({
        "type": "Feature", "stac_version": "1.0.0", "id": "mixed-dtype-nodata",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.44, 37.64], [-122.17, 37.64], [-122.17, 37.86], [-122.44, 37.86], [-122.44, 37.64]]]},
        "bbox": [-122.44, 37.64, -122.17, 37.86],
        "properties": {"datetime": "2024-01-01T00:00:00Z"},
        "assets": {
            "a": {"href": str(p_u8.resolve()), "type": "image/tiff; application=geotiff"},
            "b": {"href": str(p_f32.resolve()), "type": "image/tiff; application=geotiff"},
            "c": {"href": str(p_u16.resolve()), "type": "image/tiff; application=geotiff"},
        },
        "links": [],
    }))
    out["stac_mix"] = p_stac_mix

    # Wave 5: Exotic CRSs (LAEA 3035, LCC 3978, NZTM 2193, EASE 6933, UPS 32661/32761, BNG 27700, NAD83 4269, MODIS Sinusoidal)
    for name, crs, bounds in [
        ("laea_3035", "EPSG:3035", (4000000.0, 2800000.0, 4200000.0, 3000000.0)),
        ("lcc_3978", "EPSG:3978", (-1000000.0, 0.0, -800000.0, 200000.0)),
        ("nztm_2193", "EPSG:2193", (1700000.0, 5400000.0, 1800000.0, 5500000.0)),
        ("ease_6933", "EPSG:6933", (-11800000.0, 4400000.0, -11600000.0, 4600000.0)),
        ("ups_n_32661", "EPSG:32661", (1900000.0, 1000000.0, 2100000.0, 1200000.0)),
        ("ups_s_32761", "EPSG:32761", (1900000.0, 2800000.0, 2100000.0, 3000000.0)),
        ("bng_27700", "EPSG:27700", (400000.0, 200000.0, 500000.0, 300000.0)),
        ("nad83_4269", "EPSG:4269", (-123.0, 37.0, -122.0, 38.0)),
        ("modis_sinu", "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs", (-10000000.0, 4000000.0, -9800000.0, 4200000.0)),
    ]:
        p_crs = synth_dir / f"{name}.tif"
        arr_c = (xx * 1.5 + yy * 2.5 + 10.0).astype(np.float32)
        arr_c[0:24, 0:24] = -9999.0
        with rasterio.open(p_crs, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs=crs, transform=from_bounds(*bounds, 256, 256), tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
            ds.write(arr_c[None, ...])
        out[name] = p_crs

    # GCP-only affine GeoTIFF (no geotransform)
    p_gcp = synth_dir / "gcp_affine_4326.tif"
    gcps = [
        GroundControlPoint(row=0, col=0, x=-123.0, y=38.0, id="1"),
        GroundControlPoint(row=0, col=256, x=-122.0, y=38.0, id="2"),
        GroundControlPoint(row=256, col=0, x=-123.0, y=37.0, id="3"),
        GroundControlPoint(row=256, col=256, x=-122.0, y=37.0, id="4"),
    ]
    with rasterio.open(p_gcp, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", gcps=gcps, crs="EPSG:4326", tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write((xx + yy * 2.0 + 10.0)[None, ...])
    out["gcp_affine"] = p_gcp

    # Multi-overview + unscale + nodata COG
    p_ov = synth_dir / "multi_ovr_unscale_u16.tif"
    yy1k, xx1k = np.mgrid[0:1024, 0:1024].astype(np.uint16)
    arr_ov = ((xx1k * 20 + yy1k * 30 + 1000) % 50000 + 100).astype(np.uint16)
    arr_ov[0:128, 0:128] = 0
    with rasterio.open(p_ov, "w", driver="GTiff", width=1024, height=1024, count=1, dtype="uint16", crs="EPSG:3857", transform=from_bounds(-13650000.0, 4500000.0, -13550000.0, 4600000.0, 1024, 1024), tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=2, nodata=0) as ds:
        ds.write(arr_ov[None, ...])
        ds.scales = (0.0001,)
        ds.offsets = (-0.1,)
        ds.build_overviews([2, 4, 8], Resampling.nearest)
    out["multi_ovr_unscale"] = p_ov

    # Stripped float64 predictor=3 deflate, BigTIFF int32 predictor=2 zstd, int8 nodata=-128 unscale
    p_f64p3 = synth_dir / "stripped_f64_p3.tif"
    arr_f64 = (xx.astype(np.float64) * 1.234567890123 + yy.astype(np.float64) * 9.876543210987 + 1e9)
    arr_f64[0:32, 0:32] = -9999.0
    with rasterio.open(p_f64p3, "w", driver="GTiff", width=256, height=256, count=1, dtype="float64", crs="EPSG:3857", transform=tf_sf, tiled=False, compress="deflate", predictor=3, nodata=-9999.0) as ds:
        ds.write(arr_f64[None, ...])
    out["stripped_f64_p3"] = p_f64p3

    p_big = synth_dir / "bigtiff_i32_p2_zstd.tif"
    arr_big = (xx.astype(np.int32) * 100000 - yy.astype(np.int32) * 50000 + 50000000)
    arr_big[0:32, 0:32] = -2147483648
    with rasterio.open(p_big, "w", driver="GTiff", width=256, height=256, count=1, dtype="int32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="zstd", predictor=2, nodata=-2147483648, BIGTIFF="YES") as ds:
        ds.write(arr_big[None, ...])
    out["bigtiff_i32"] = p_big

    p_i8 = synth_dir / "int8_nd128_unscale.tif"
    arr_i8 = (((xx + yy) % 200) - 100).astype(np.int8)
    arr_i8[0:32, 0:32] = -128
    with rasterio.open(p_i8, "w", driver="GTiff", width=256, height=256, count=1, dtype="int8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-128) as ds:
        ds.write(arr_i8[None, ...])
        ds.scales = (0.1,)
        ds.offsets = (10.0,)
    out["int8_unscale"] = p_i8

    return out


def compare_stats_strict(j_cw: dict, j_tt: dict) -> tuple[bool, str]:
    if set(j_cw.keys()) != set(j_tt.keys()):
        return False, f"KEYS DIFF cw={sorted(j_cw.keys())} vs tt={sorted(j_tt.keys())}"
    diffs = []
    for k in sorted(j_tt.keys()):
        a, b = j_cw[k], j_tt[k]
        for fld in ("min", "max", "mean", "std", "median", "percentile_2", "percentile_98"):
            va, vb = a.get(fld), b.get(fld)
            if va is None and vb is None:
                continue
            if (va is None) != (vb is None):
                diffs.append(f"{k}.{fld}: cw={va} vs tt={vb}")
                continue
            fa, fb = float(va), float(vb)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                if not (math.isnan(fa) and math.isnan(fb)) and fa != fb:
                    diffs.append(f"{k}.{fld}: cw={fa} vs tt={fb}")
                continue
            scale = max(1e-3, abs(fb))
            if abs(fa - fb) / scale > 2e-2 and abs(fa - fb) > 1e-3:
                diffs.append(f"{k}.{fld}: cw={fa:.6g} vs tt={fb:.6g}")
        for fld in b:
            if fld.startswith("percentile_") and fld not in ("percentile_2", "percentile_98"):
                va, vb = a.get(fld), b.get(fld)
                if va is None and vb is None:
                    continue
                if (va is None) != (vb is None):
                    diffs.append(f"{k}.{fld}: cw={va} vs tt={vb}")
                    continue
                fa, fb = float(va), float(vb)
                if abs(fa - fb) / max(1e-3, abs(fb)) > 2e-2 and abs(fa - fb) > 1e-3:
                    diffs.append(f"{k}.{fld}: cw={fa} vs tt={fb}")
        if "values" in b and a.get("values") != b.get("values"):
            diffs.append(f"{k}.values: cw={a.get('values')} vs tt={b.get('values')}")
        if "histogram" in b:
            ha, hb = a.get("histogram"), b.get("histogram")
            if ha is None or hb is None or len(ha) != 2 or len(hb) != 2:
                diffs.append(f"{k}.histogram: missing/malformed")
            else:
                ca_h, ea_h = ha
                cb_h, eb_h = hb
                if len(ca_h) != len(cb_h) or len(ea_h) != len(eb_h):
                    diffs.append(f"{k}.histogram len: cw=({len(ca_h)},{len(ea_h)}) vs tt=({len(cb_h)},{len(eb_h)})")
                elif eb_h:
                    edge_err = max(abs(float(x) - float(y)) for x, y in zip(ea_h, eb_h))
                    edge_scale = max(1.0, max(abs(float(y)) for y in eb_h))
                    if edge_err / edge_scale > 2e-2:
                        diffs.append(f"{k}.histogram edges err={edge_err:.4g}")
        ca, cb = a.get("count"), b.get("count")
        if (ca is None) != (cb is None):
            diffs.append(f"{k}.count: cw={ca} vs tt={cb}")
        elif ca is not None and cb is not None:
            if abs(float(ca) - float(cb)) / max(1.0, float(cb)) > 3e-2:
                diffs.append(f"{k}.count: cw={ca} vs tt={cb}")
    if diffs:
        return False, "; ".join(diffs[:4])
    return True, f"keys={sorted(j_cw.keys())} ok"


def compare_point_strict(j_cw: dict, j_tt: dict) -> tuple[bool, str]:
    vcw, vtt = j_cw.get("values", []), j_tt.get("values", [])
    if len(vcw) != len(vtt):
        return False, f"len cw={len(vcw)} vs tt={len(vtt)}"
    diffs = []
    for i, (a, b) in enumerate(zip(vcw, vtt)):
        if a is None and b is None:
            continue
        if (a is None) != (b is None):
            diffs.append(f"[{i}] cw={a} vs tt={b}")
            continue
        scale = max(1.0, abs(float(b)))
        if abs(float(a) - float(b)) / scale > 1e-3:
            diffs.append(f"[{i}] cw={a} vs tt={b}")
    if diffs:
        return False, "; ".join(diffs)
    return True, f"values={vcw[:4]}"


def start_cogzilla(bin_path: Path, port: int, cpu_only: bool) -> subprocess.Popen:
    env = os.environ.copy()
    if cpu_only:
        env["COGZILLA_GPU_POLICY"] = "never"
    else:
        env.pop("COGZILLA_GPU_POLICY", None)
    return subprocess.Popen(
        [str(bin_path), "--bind", f"127.0.0.1:{port}", "--cache-mb", "256"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def build_mosaic_fixtures(d: Path) -> dict:
    """Build 1-band, 2-band, and 3-band RGB aligned COGs on exact z=12/657/1582 grid."""
    d.mkdir(parents=True, exist_ok=True)
    b = mercantile.xy_bounds(657, 1582, 12)
    tf = from_bounds(b.left, b.bottom, b.right, b.top, 256, 256)

    s1 = np.full((256, 256), 10.0, dtype=np.float32)
    s2 = np.full((256, 256), 50.0, dtype=np.float32)
    s3 = np.full((256, 256), 20.0, dtype=np.float32)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    s1 += xx * 0.1
    s2 += yy * 0.2
    s3 += (xx + yy) * 0.05

    # Staggered nodata holes (-9999.0) covering overlap cardinalities 0, 1, 2, 3:
    s1[0:64, 0:128] = -9999.0
    s2[0:64, 0:64] = -9999.0
    s2[64:128, 0:64] = -9999.0
    s3[0:64, 0:128] = -9999.0

    paths_1b = []
    for idx, arr in enumerate((s1, s2, s3), 1):
        p = d / f"scene_{idx}_1b.tif"
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
            ds.write(arr[None, ...])
        paths_1b.append(str(p.resolve()))

    scores = [30.0, 10.0, 20.0]
    vals = [100.0, 200.0, 300.0]
    paths_2b, s2b_data, s2b_score = [], [], []
    for idx, (val, sc, mask_src) in enumerate(zip(vals, scores, (s1, s2, s3)), 1):
        b1 = np.where(mask_src == -9999.0, -9999.0, val + xx * 0.01).astype(np.float32)
        b2 = np.where(mask_src == -9999.0, -9999.0, sc + yy * 0.01).astype(np.float32)
        p = d / f"scene_{idx}_2b.tif"
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=2, dtype="float32", crs="EPSG:3857", transform=tf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
            ds.write(np.stack([b1, b2], axis=0))
        paths_2b.append(str(p.resolve()))
        s2b_data.append(np.where(b1 == -9999.0, np.nan, b1))
        s2b_score.append(np.where(b2 == -9999.0, np.nan, b2))

    paths_rgb, rgb_scenes = [], []
    rgb_bases = [(40, 120, 200), (180, 60, 100), (100, 220, 50)]
    for idx, ((r0, g0, b0), mask_src) in enumerate(zip(rgb_bases, (s1, s2, s3)), 1):
        r = np.where(mask_src == -9999.0, 0, np.clip(r0 + (xx // 8).astype(np.int32), 1, 255)).astype(np.uint8)
        g = np.where(mask_src == -9999.0, 0, np.clip(g0 + (yy // 8).astype(np.int32), 1, 255)).astype(np.uint8)
        bl = np.where(mask_src == -9999.0, 0, np.clip(b0 + ((xx + yy) // 16).astype(np.int32), 1, 255)).astype(np.uint8)
        arr3 = np.stack([r, g, bl], axis=0)
        p = d / f"scene_{idx}_rgb.tif"
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8", crs="EPSG:3857", transform=tf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=0) as ds:
            ds.write(arr3)
        paths_rgb.append(str(p.resolve()))
        rgb_scenes.append(np.where(arr3 == 0, np.nan, arr3.astype(np.float64)))

    qk = mercantile.quadkey(657, 1582, 12)
    lonlat = mercantile.bounds(657, 1582, 12)
    bounds_wgs = [lonlat.west, lonlat.south, lonlat.east, lonlat.north]

    mj_1b = d / "mosaic_1b.json"
    mj_1b.write_text(json.dumps({"mosaicjson": "0.0.2", "name": "test-1b", "minzoom": 12, "maxzoom": 12, "quadkey_zoom": 12, "bounds": bounds_wgs, "tiles": {qk: paths_1b}}))
    mj_2b = d / "mosaic_2b.json"
    mj_2b.write_text(json.dumps({"mosaicjson": "0.0.2", "name": "test-2b", "minzoom": 12, "maxzoom": 12, "quadkey_zoom": 12, "bounds": bounds_wgs, "tiles": {qk: paths_2b}}))
    mj_rgb = d / "mosaic_rgb.json"
    mj_rgb.write_text(json.dumps({"mosaicjson": "0.0.2", "name": "test-rgb", "minzoom": 12, "maxzoom": 12, "quadkey_zoom": 12, "bounds": bounds_wgs, "tiles": {qk: paths_rgb}}))

    stack = np.stack([np.where(a == -9999.0, np.nan, a) for a in (s1, s2, s3)], axis=0)
    valid_cnt = np.isfinite(stack).sum(axis=0)
    rgb_stack = np.stack(rgb_scenes, axis=0)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        gt: dict[str, np.ndarray] = {}
        gt["first"] = np.where(np.isfinite(stack[0]), stack[0], np.where(np.isfinite(stack[1]), stack[1], stack[2]))
        gt["highest"] = np.nanmax(stack, axis=0)
        gt["lowest"] = np.nanmin(stack, axis=0)
        gt["mean"] = np.nanmean(stack, axis=0)
        gt["median"] = np.nanmedian(stack, axis=0)
        gt["stdev"] = np.nanstd(stack, axis=0, ddof=0)
        gt["count"] = valid_cnt.astype(np.float32)

        d_stack = np.stack(s2b_data, axis=0)
        sc_stack = np.stack(s2b_score, axis=0)
        idx_low = np.argmin(np.where(np.isfinite(sc_stack), sc_stack, np.inf), axis=0)
        gt["lastbandlow"] = np.where(valid_cnt > 0, np.take_along_axis(d_stack, idx_low[None], axis=0)[0], np.nan)
        idx_high = np.argmax(np.where(np.isfinite(sc_stack), sc_stack, -np.inf), axis=0)
        gt["lastbandhigh"] = np.where(valid_cnt > 0, np.take_along_axis(d_stack, idx_high[None], axis=0)[0], np.nan)

        gt_rgb: dict[str, np.ndarray] = {}
        gt_rgb["first"] = np.where(np.isfinite(rgb_stack[0]), rgb_stack[0], np.where(np.isfinite(rgb_stack[1]), rgb_stack[1], rgb_stack[2]))
        gt_rgb["highest"] = np.nanmax(rgb_stack, axis=0)
        gt_rgb["lowest"] = np.nanmin(rgb_stack, axis=0)
        gt_rgb["mean"] = np.nanmean(rgb_stack, axis=0)
        gt_rgb["median"] = np.nanmedian(rgb_stack, axis=0)
        gt_rgb["stdev"] = np.nanstd(rgb_stack, axis=0, ddof=0)

    return {"mj_1b": mj_1b, "mj_2b": mj_2b, "mj_rgb": mj_rgb, "gt": gt, "gt_rgb": gt_rgb, "bounds_wgs": bounds_wgs, "xy_bounds": b, "s1": s1, "s2": s2, "s3": s3}


def run_wave7_checks(
    client: httpx.Client,
    cw_base: str,
    gpu_base: str | None,
    tmp_dir: Path,
    only_substr: str | None,
    failures: list[dict],
    gpu_diffs: list[dict],
) -> int:
    """Wave 7: MosaicJSON (all 9 pixel_selection methods, 3-band RGB, expr, bbox, point, POST stats) + HTTP protocol contracts."""
    def wants(label: str) -> bool:
        return not only_substr or only_substr.lower() in label.lower()

    all_labels = [
        *[f"W7 MOSAIC {m} .npy" for m in ("first", "highest", "lowest", "mean", "median", "stdev", "count", "lastbandlow", "lastbandhigh")],
        *[f"W7 MOSAIC {m} .png" for m in ("first", "highest", "lowest", "mean", "median", "stdev", "count", "lastbandlow", "lastbandhigh")],
        *[f"W7 MOSAIC RGB {m} .npy" for m in ("first", "highest", "lowest", "mean", "median", "stdev")],
        *[f"W7 MOSAIC expr+{m} .npy" for m in ("first", "mean", "highest")],
        "W7 MOSAIC /point multi-asset",
        *[f"W7 MOSAIC /bbox {m}" for m in ("first", "mean", "highest")],
        *[f"W7 MOSAIC POST /statistics {m}" for m in ("first", "mean", "highest")],
        "W7 HTTP ETag + If-None-Match -> 304",
        "W7 HTTP If-None-Match:* -> 304",
        "W7 HTTP HEAD tile -> 200 empty",
        "W7 HTTP OPTIONS -> 204",
        "W7 HTTP trailing slash /cog/info/ -> 200",
        "W7 HTTP X-Forwarded-{Proto,Host,Prefix}",
        "W7 HTTP Vary: Accept on /colorMaps",
    ]
    if not any(wants(lb) for lb in all_labels):
        return 0

    fx = build_mosaic_fixtures(tmp_dir / "wave7_mosaic")
    mj_1b, mj_2b, mj_rgb = fx["mj_1b"], fx["mj_2b"], fx["mj_rgb"]
    gt, gt_rgb, xb = fx["gt"], fx["gt_rgb"], fx["xy_bounds"]
    sat = FIXTURES / "satellite_utm10n_rgb_zstd_p3.tif"
    merc = FIXTURES / "mercator_4k_zstd_p3.tif"

    ran = 0

    def record(label: str, path: str, kind: str, ok: bool, detail: str, method: str = "GET"):
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": label, "method": method, "path": path, "kind": kind, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {label:54s} | {detail}")

    # 1. All 9 pixel_selection methods on 1-band (or 2-band for lastband*) .npy + .png
    for method in ("first", "highest", "lowest", "mean", "median", "stdev", "count", "lastbandlow", "lastbandhigh"):
        mj = mj_2b if method in ("lastbandlow", "lastbandhigh") else mj_1b
        lb_npy = f"W7 MOSAIC {method} .npy"
        if wants(lb_npy):
            path = f"/mosaicjson/tiles/WebMercatorQuad/12/657/1582.npy?url={q(mj)}&pixel_selection={method}"
            rc = client.get(cw_base + path)
            if gpu_base is not None:
                rg = client.get(gpu_base + path)
                if rc.status_code != rg.status_code:
                    gpu_diffs.append({"label": lb_npy, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
                elif rc.status_code == 200:
                    gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                    if not gok:
                        gpu_diffs.append({"label": lb_npy, "detail": gd})
            if rc.status_code != 200:
                record(lb_npy, path, "mosaic_gt", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
            else:
                arr = np.load(io.BytesIO(rc.content))
                low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
                mask = arr[-1] > low
                exp = gt[method]
                shape_ok = arr.shape == (2, 256, 256)
                if method == "count":
                    max_err = float(np.max(np.abs(arr[0].astype(np.float64) - exp.astype(np.float64))))
                    all_valid = bool(mask.all())
                    record(lb_npy, path, "mosaic_gt", shape_ok and all_valid and max_err < 1e-4, f"shape={arr.shape} dtype={arr.dtype} all_valid={all_valid} max_err={max_err:.6f}")
                else:
                    exp_mask = np.isfinite(exp)
                    mask_eq = float((mask == exp_mask).mean()) * 100.0
                    both = mask & exp_mask
                    max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp[both].astype(np.float64)))) if both.any() else 0.0
                    record(lb_npy, path, "mosaic_gt", shape_ok and mask_eq == 100.0 and max_err < 1e-2, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

        lb_png = f"W7 MOSAIC {method} .png"
        if wants(lb_png):
            path = f"/mosaicjson/tiles/WebMercatorQuad/12/657/1582.png?url={q(mj)}&pixel_selection={method}&rescale=0,300&colormap_name=viridis"
            rc = client.get(cw_base + path)
            if rc.status_code != 200:
                record(lb_png, path, "mosaic_png", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
            else:
                try:
                    rgba = np.array(Image.open(io.BytesIO(rc.content)).convert("RGBA"))
                except Exception as exc:
                    record(lb_png, path, "mosaic_png", False, f"invalid PNG: {exc}")
                    continue
                alpha = rgba[..., 3]
                if method == "count":
                    alpha_ok = bool((alpha == 255).all())
                    valid_rgb = rgba[..., :3]
                else:
                    exp_mask = np.isfinite(gt[method])
                    alpha_ok = bool(((alpha > 0) == exp_mask).all())
                    valid_rgb = rgba[exp_mask, :3]
                rgb_std = float(valid_rgb.std()) if valid_rgb.size else 0.0
                ok = rgba.shape == (256, 256, 4) and alpha_ok and rgb_std > 0.5
                detail = f"shape={rgba.shape} alpha_ok={alpha_ok} rgb_std={rgb_std:.1f}"
                if gpu_base is not None:
                    rg = client.get(gpu_base + path)
                    if rg.status_code != 200:
                        gpu_diffs.append({"label": lb_png, "detail": f"status cpu=200 gpu={rg.status_code}"})
                    else:
                        iok, idet = compare_images(rc.content, rg.content)
                        if not iok:
                            gpu_diffs.append({"label": lb_png, "detail": idet})
                        detail = f"{detail} | CPU==GPU {idet}"
                record(lb_png, path, "mosaic_png", ok, detail)

    # 2. 3-band RGB mosaic with pixel_selection
    for method in ("first", "highest", "lowest", "mean", "median", "stdev"):
        lb = f"W7 MOSAIC RGB {method} .npy"
        if not wants(lb):
            continue
        path = f"/mosaicjson/tiles/WebMercatorQuad/12/657/1582.npy?url={q(mj_rgb)}&pixel_selection={method}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            record(lb, path, "mosaic_gt", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = np.load(io.BytesIO(rc.content))
            low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
            mask = arr[-1] > low
            exp = gt_rgb[method]
            exp_mask = np.isfinite(exp[0])
            mask_eq = float((mask == exp_mask).mean()) * 100.0
            both = mask & exp_mask
            shape_ok = arr.shape == (4, 256, 256)
            if arr.dtype == np.uint8:
                # Integer mean/median truncates toward zero in cogzilla; first/highest/lowest are exact.
                ref = np.trunc(exp) if method in ("mean", "median") else exp
                max_err = max(float(np.max(np.abs(arr[bi][both].astype(np.float64) - ref[bi][both]))) for bi in range(3)) if both.any() else 0.0
                tol = 0.0
            else:
                max_err = max(float(np.max(np.abs(arr[bi][both].astype(np.float64) - exp[bi][both]))) for bi in range(3)) if both.any() else 0.0
                tol = 1e-2
            record(lb, path, "mosaic_gt", shape_ok and mask_eq == 100.0 and max_err <= tol, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.4f}")

    # 3. MosaicJSON expression + pixel_selection
    for method in ("first", "mean", "highest"):
        lb = f"W7 MOSAIC expr+{method} .npy"
        if not wants(lb):
            continue
        path = f"/mosaicjson/tiles/WebMercatorQuad/12/657/1582.npy?url={q(mj_1b)}&pixel_selection={method}&expression={qe('b1*2.0+5.0')}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            record(lb, path, "mosaic_gt", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = np.load(io.BytesIO(rc.content))
            low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
            mask = arr[-1] > low
            exp = gt[method] * 2.0 + 5.0
            exp_mask = np.isfinite(exp)
            both = mask & exp_mask
            mask_eq = float((mask == exp_mask).mean()) * 100.0
            max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp[both]))) if both.any() else 0.0
            record(lb, path, "mosaic_gt", arr.shape == (2, 256, 256) and mask_eq == 100.0 and max_err < 1e-2, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    # 4. MosaicJSON /point (multi-asset response at exact EPSG:3857 pixel center [row=128, col=128])
    dx = (xb.right - xb.left) / 256.0
    dy = (xb.top - xb.bottom) / 256.0
    pt_ll = mercantile.lnglat(xb.left + 128.5 * dx, xb.top - 128.5 * dy)
    lb_pt = "W7 MOSAIC /point multi-asset"
    if wants(lb_pt):
        path = f"/mosaicjson/point/{pt_ll.lng:.8f},{pt_ll.lat:.8f}?url={q(mj_1b)}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code or (rc.status_code == 200 and first_json_diff(rc.json(), rg.json()) is not None):
                gpu_diffs.append({"label": lb_pt, "detail": f"cpu={rc.text[:80]} gpu={rg.text[:80]}"})
        if rc.status_code != 200:
            record(lb_pt, path, "mosaic_point", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            assets = rc.json().get("assets", [])
            got_vals = [a["values"][0] for a in assets]
            exp_vals = [float(fx["s1"][128, 128]), float(fx["s2"][128, 128]), float(fx["s3"][128, 128])]
            ok = len(got_vals) == 3 and all(abs(g - e) < 1e-3 for g, e in zip(got_vals, exp_vals))
            record(lb_pt, path, "mosaic_point", ok, f"got={[round(v, 4) for v in got_vals]} exp={[round(v, 4) for v in exp_vals]}")

    # 5. MosaicJSON /bbox
    bbox_s = f"{xb.left:.4f},{xb.bottom:.4f},{xb.right:.4f},{xb.top:.4f}"
    for method in ("first", "mean", "highest"):
        lb = f"W7 MOSAIC /bbox {method}"
        if not wants(lb):
            continue
        path = f"/mosaicjson/bbox/{bbox_s}/256x256.npy?url={q(mj_1b)}&coord_crs=EPSG:3857&dst_crs=EPSG:3857&pixel_selection={method}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            record(lb, path, "mosaic_gt", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = np.load(io.BytesIO(rc.content))
            low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
            mask = arr[-1] > low
            exp = gt[method]
            exp_mask = np.isfinite(exp)
            both = mask & exp_mask
            mask_eq = float((mask == exp_mask).mean()) * 100.0
            max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp[both]))) if both.any() else 0.0
            record(lb, path, "mosaic_gt", arr.shape == (2, 256, 256) and mask_eq == 100.0 and max_err < 1e-2, f"shape={arr.shape} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    # 6. MosaicJSON POST /statistics over west half [0:256, 0:128] covering overlap cardinalities 0, 1, 2, and 3!
    mid_x = 0.5 * (xb.left + xb.right)
    sw = mercantile.lnglat(xb.left, xb.bottom)
    se_mid = mercantile.lnglat(mid_x, xb.bottom)
    ne_mid = mercantile.lnglat(mid_x, xb.top)
    nw = mercantile.lnglat(xb.left, xb.top)
    poly_west = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[sw.lng, sw.lat], [se_mid.lng, se_mid.lat], [ne_mid.lng, ne_mid.lat], [nw.lng, nw.lat], [sw.lng, sw.lat]]]},
    }
    for method in ("first", "mean", "highest"):
        lb = f"W7 MOSAIC POST /statistics {method}"
        if not wants(lb):
            continue
        path = f"/mosaicjson/statistics?url={q(mj_1b)}&pixel_selection={method}"
        rc = client.post(cw_base + path, json=poly_west)
        if gpu_base is not None:
            rg = client.post(gpu_base + path, json=poly_west)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                jd = first_json_diff(rc.json(), rg.json())
                if jd is not None:
                    gpu_diffs.append({"label": lb, "detail": f"CPU vs GPU stats diff: {jd}"})
        if rc.status_code != 200:
            record(lb, path, "mosaic_stats", False, f"HTTP {rc.status_code} {rc.text[:100]!r}", method="POST")
        else:
            st = rc.json()["properties"]["statistics"]["b1"]
            sub = gt[method][:, 0:128]
            valid_sub = sub[np.isfinite(sub)]
            exp_mean = float(valid_sub.mean())
            exp_min = float(valid_sub.min())
            exp_max = float(valid_sub.max())
            exp_count = float(valid_sub.size)
            mean_err = abs(st["mean"] - exp_mean)
            min_err = abs(st["min"] - exp_min)
            max_err = abs(st["max"] - exp_max)
            cnt_rel = abs(st["count"] - exp_count) / exp_count
            ok = mean_err < 0.1 and min_err < 0.2 and max_err < 0.2 and cnt_rel < 0.02
            record(lb, path, "mosaic_stats", ok, f"mean={st['mean']:.3f}/{exp_mean:.3f} min={st['min']:.2f}/{exp_min:.2f} max={st['max']:.2f}/{exp_max:.2f} cnt={st['count']:.0f}/{exp_count:.0f}", method="POST")

    # 7. HTTP protocol contracts (ETag/304, HEAD, OPTIONS, trailing slash, X-Forwarded-*, Vary: Accept)
    tile_u = f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000&colormap_name=viridis"
    etag: str | None = None
    if wants("W7 HTTP ETag + If-None-Match -> 304"):
        etag = client.get(cw_base + tile_u).headers.get("etag")

    def check_x_forwarded() -> tuple[bool, str]:
        r = client.get(
            cw_base + f"/cog/WebMercatorQuad/tilejson.json?url={q(sat)}",
            headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "tiles.example.com", "X-Forwarded-Prefix": "/v1"},
        )
        if r.status_code != 200:
            return False, f"HTTP {r.status_code} {r.text[:80]!r}"
        tiles = r.json().get("tiles") or []
        tile0 = tiles[0] if tiles else ""
        return tile0.startswith("https://tiles.example.com/v1/"), f"tile0={tile0[:60]}"

    proto_specs = [
        ("W7 HTTP ETag + If-None-Match -> 304", "GET", tile_u, lambda: (
            (r := client.get(cw_base + tile_u, headers={"If-None-Match": etag or ""})),
            bool(etag) and r.status_code == 304 and len(r.content) == 0 and r.headers.get("etag") == etag,
            f"etag={etag} status={r.status_code}",
        )[1:]),
        ("W7 HTTP If-None-Match:* -> 304", "GET", tile_u, lambda: (
            (r := client.get(cw_base + tile_u, headers={"If-None-Match": "*"})),
            r.status_code == 304 and len(r.content) == 0,
            f"status={r.status_code}",
        )[1:]),
        ("W7 HTTP HEAD tile -> 200 empty", "HEAD", tile_u, lambda: (
            (r := client.head(cw_base + tile_u)),
            r.status_code == 200 and len(r.content) == 0,
            f"status={r.status_code} len={len(r.content)}",
        )[1:]),
        ("W7 HTTP OPTIONS -> 204", "OPTIONS", f"/cog/info?url={q(sat)}", lambda: (
            (r := client.options(cw_base + f"/cog/info?url={q(sat)}")),
            r.status_code == 204,
            f"status={r.status_code}",
        )[1:]),
        ("W7 HTTP trailing slash /cog/info/ -> 200", "GET", f"/cog/info/?url={q(sat)}", lambda: (
            (r := client.get(cw_base + f"/cog/info/?url={q(sat)}")),
            r.status_code == 200,
            f"status={r.status_code}",
        )[1:]),
        ("W7 HTTP X-Forwarded-{Proto,Host,Prefix}", "GET", f"/cog/WebMercatorQuad/tilejson.json?url={q(sat)}", check_x_forwarded),
        ("W7 HTTP Vary: Accept on /colorMaps", "GET", "/colorMaps", lambda: (
            (r := client.get(cw_base + "/colorMaps")),
            r.status_code == 200 and "accept" in r.headers.get("vary", "").lower(),
            f"status={r.status_code} vary={r.headers.get('vary')!r}",
        )[1:]),
    ]
    for lb, http_method, path, fn in proto_specs:
        if not wants(lb):
            continue
        ok, det = fn()
        record(lb, path, "http_proto", ok, det, method=http_method)

    return ran


def _write_v2_zarr_array(store: Path, name: str, arr: np.ndarray, chunks: tuple[int, ...], attrs: dict, compressor: dict | None = None, fill_value=None):
    ad = store / name
    ad.mkdir(parents=True, exist_ok=True)
    meta = {
        "chunks": list(chunks),
        "compressor": compressor,
        "dtype": arr.dtype.str,
        "fill_value": fill_value,
        "filters": None,
        "order": "C",
        "shape": list(arr.shape),
        "zarr_format": 2,
    }
    (ad / ".zarray").write_text(json.dumps(meta))
    (ad / ".zattrs").write_text(json.dumps(attrs))
    grid = [int(np.ceil(s / c)) for s, c in zip(arr.shape, chunks)]
    for idx in np.ndindex(*grid):
        slices = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, arr.shape))
        sub = arr[slices]
        if sub.shape != chunks:
            pad = np.full(chunks, fill_value if fill_value is not None else 0, dtype=arr.dtype)
            pad[tuple(slice(0, x) for x in sub.shape)] = sub
            sub = pad
        raw = np.ascontiguousarray(sub).tobytes()
        if compressor is None:
            enc = raw
        elif compressor["id"] == "zlib":
            enc = zlib.compress(raw, compressor.get("level", 1))
        elif compressor["id"] == "gzip":
            enc = gzip.compress(raw, compresslevel=compressor.get("level", 1))
        else:
            raise ValueError(compressor)
        key = ".".join(str(i) for i in idx) if idx else "0"
        (ad / key).write_bytes(enc)


def _write_v3_zarr_array(store: Path, name: str, arr: np.ndarray, chunks: tuple[int, ...], attrs: dict, dim_names: list[str], fill_value, use_gzip: bool = True):
    ad = store / name
    ad.mkdir(parents=True, exist_ok=True)
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    if use_gzip:
        codecs.append({"name": "gzip", "configuration": {"level": 1}})
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(arr.shape),
        "data_type": arr.dtype.name,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill_value,
        "codecs": codecs,
        "dimension_names": dim_names,
        "attributes": attrs,
    }
    (ad / "zarr.json").write_text(json.dumps(meta))
    grid = [int(np.ceil(s / c)) for s, c in zip(arr.shape, chunks)]
    for idx in np.ndindex(*grid):
        slices = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, arr.shape))
        sub = np.ascontiguousarray(arr[slices]).tobytes()
        enc = gzip.compress(sub, compresslevel=1) if use_gzip else sub
        cp = ad / "c" / Path(*[str(i) for i in idx])
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(enc)


def build_zarr_fixtures(d: Path) -> dict:
    """Build 4 synthetic Zarr v2/v3 stores covering 3D time slicing, EPSG:3857, south-up v3, and 0..360° lon."""
    d.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(38.0 - 0.5 / 128, 37.0 + 0.5 / 128, 128, dtype=np.float64)
    lon = np.linspace(-123.0 + 0.5 / 128, -122.0 - 0.5 / 128, 128, dtype=np.float64)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    t0 = 20.0 + xx * 0.5 + yy * 0.25
    t1 = 100.0 + xx * 1.0 - yy * 0.25
    data_3d = np.stack([t0, t1], axis=0).astype(np.float32)
    data_3d[:, 0:32, 0:32] = -9999.0

    # 1. Zarr v2 Geographic 3D (time=2, lat=128, lon=128)
    s_v2_geo = d / "v2_geo_3d.zarr"
    s_v2_geo.mkdir()
    (s_v2_geo / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (s_v2_geo / ".zattrs").write_text(json.dumps({"Conventions": "CF-1.8"}))
    _write_v2_zarr_array(s_v2_geo, "time", np.array([0, 1], dtype=np.int64), (2,), {"_ARRAY_DIMENSIONS": ["time"]})
    _write_v2_zarr_array(s_v2_geo, "lat", lat, (128,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude", "units": "degrees_north"})
    _write_v2_zarr_array(s_v2_geo, "lon", lon, (128,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude", "units": "degrees_east"})
    _write_v2_zarr_array(s_v2_geo, "temp", data_3d, (1, 64, 64), {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"], "scale_factor": 0.5, "add_offset": 10.0}, compressor={"id": "zlib", "level": 1}, fill_value=-9999.0)

    # 2. Zarr v2 Projected EPSG:3857 2D (256x256) aligned to z=12/657/1582
    s_v2_proj = d / "v2_proj_3857.zarr"
    s_v2_proj.mkdir()
    (s_v2_proj / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (s_v2_proj / ".zattrs").write_text(json.dumps({"Conventions": "CF-1.8"}))
    b = mercantile.xy_bounds(657, 1582, 12)
    dx, dy = (b.right - b.left) / 256.0, (b.top - b.bottom) / 256.0
    x_c = np.linspace(b.left + 0.5 * dx, b.right - 0.5 * dx, 256, dtype=np.float64)
    y_c = np.linspace(b.top - 0.5 * dy, b.bottom + 0.5 * dy, 256, dtype=np.float64)
    yy2, xx2 = np.mgrid[0:256, 0:256]
    elev = (500 + xx2 * 4 + yy2 * 2).astype(np.int16)
    elev[0:64, 0:64] = -32768
    wkt_3857 = RasterioCRS.from_epsg(3857).to_wkt()
    _write_v2_zarr_array(s_v2_proj, "x", x_c, (256,), {"_ARRAY_DIMENSIONS": ["x"], "standard_name": "projection_x_coordinate", "units": "m"})
    _write_v2_zarr_array(s_v2_proj, "y", y_c, (256,), {"_ARRAY_DIMENSIONS": ["y"], "standard_name": "projection_y_coordinate", "units": "m"})
    _write_v2_zarr_array(s_v2_proj, "spatial_ref", np.array(0, dtype=np.int64), (), {"_ARRAY_DIMENSIONS": [], "crs_wkt": wkt_3857, "spatial_ref": wkt_3857})
    _write_v2_zarr_array(s_v2_proj, "elev", elev, (128, 128), {"_ARRAY_DIMENSIONS": ["y", "x"], "grid_mapping": "spatial_ref", "scale_factor": 0.1, "add_offset": 100.0}, compressor={"id": "gzip", "level": 1}, fill_value=-32768)

    # 3. Zarr v3 Geographic 2D with ASCENDING latitude (south-up!), float64, NaN hole
    s_v3_southup = d / "v3_southup.zarr"
    s_v3_southup.mkdir()
    (s_v3_southup / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"Conventions": "CF-1.8"}}))
    lat_asc = np.linspace(37.0 + 0.5 / 128, 38.0 - 0.5 / 128, 128, dtype=np.float64)
    v3_arr = (lat_asc[:, None] * 10.0 + (lon[None, :] + 130.0)).astype(np.float64)
    v3_arr[0:32, 0:32] = np.nan
    _write_v3_zarr_array(s_v3_southup, "lat", lat_asc, (128,), {"standard_name": "latitude", "units": "degrees_north"}, ["lat"], "NaN", use_gzip=False)
    _write_v3_zarr_array(s_v3_southup, "lon", lon, (128,), {"standard_name": "longitude", "units": "degrees_east"}, ["lon"], "NaN", use_gzip=False)
    _write_v3_zarr_array(s_v3_southup, "sst", v3_arr, (64, 64), {"units": "K"}, ["lat", "lon"], "NaN", use_gzip=True)

    # 4. Zarr v2 0..360° longitude grid
    s_v2_360 = d / "v2_lon360.zarr"
    s_v2_360.mkdir()
    (s_v2_360 / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (s_v2_360 / ".zattrs").write_text(json.dumps({"Conventions": "CF-1.8"}))
    _write_v2_zarr_array(s_v2_360, "lat", lat, (128,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude"})
    _write_v2_zarr_array(s_v2_360, "lon", lon + 360.0, (128,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude"})
    _write_v2_zarr_array(s_v2_360, "var360", t0.astype(np.float32), (64, 64), {"_ARRAY_DIMENSIONS": ["lat", "lon"]}, compressor={"id": "zlib", "level": 1}, fill_value=-9999.0)

    return {"v2_geo": s_v2_geo, "v2_proj": s_v2_proj, "v3_southup": s_v3_southup, "v2_360": s_v2_360, "data_3d": data_3d, "elev": elev, "v3_arr": v3_arr, "lat": lat, "lon": lon, "lat_asc": lat_asc}


def run_wave8_checks(
    client: httpx.Client,
    cw_base: str,
    gpu_base: str | None,
    tmp_dir: Path,
    only_substr: str | None,
    failures: list[dict],
    gpu_diffs: list[dict],
) -> int:
    """Wave 8: Zarr v2 & v3 adversarial oracle (tiles .npy/.png, return_mask=false, sel=, unscale, south-up v3, 0..360° lon, preview, bbox, GET/POST stats, ETag/HEAD)."""
    def wants(label: str) -> bool:
        return not only_substr or only_substr.lower() in label.lower()

    all_labels = [
        "W8 ZARR v2 /dataset/keys",
        "W8 ZARR v2 /info",
        "W8 ZARR v3 /validate",
        "W8 ZARR v2 proj tile .npy unscale=False",
        "W8 ZARR v2 proj tile .npy unscale=True",
        "W8 ZARR v2 proj tile .npy return_mask=false",
        "W8 ZARR v2 proj tile .png",
        *[f"W8 ZARR v2 3D /point sel=time={t} unscale={u}" for t in (0, 1) for u in (False, True)],
        "W8 ZARR v2 3D /preview/128x128.npy sel=time=1",
        "W8 ZARR v2 3D /bbox 128x128.npy sel=time=0",
        "W8 ZARR v2 3D GET /statistics sel=time=0",
        "W8 ZARR v2 3D POST /statistics west half",
        "W8 ZARR v3 south-up /point NE",
        "W8 ZARR v3 south-up /bbox 128x128.npy",
        "W8 ZARR v2 0..360° lon /point (-122.5, 37.5)",
        "W8 ZARR tile ETag + If-None-Match -> 304",
        "W8 ZARR tile HEAD -> 200 empty",
        "W8 ZARR tile cache isolates .npy vs .png",
    ]
    if not any(wants(lb) for lb in all_labels):
        return 0

    fx = build_zarr_fixtures(tmp_dir / "wave8_zarr")
    ran = 0

    def rec(label: str, path: str, kind: str, ok: bool, detail: str, method: str = "GET"):
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": label, "method": method, "path": path, "kind": kind, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {label:54s} | {detail}")

    def try_npy(label: str, path: str, raw: bytes):
        try:
            return np.load(io.BytesIO(raw))
        except Exception as exc:
            rec(label, path, "zarr_npy", False, f"invalid .npy (len={len(raw)}, magic={raw[:8]!r}): {exc}")
            return None

    # 1. Metadata & introspection
    for lb, path, check_fn in [
        ("W8 ZARR v2 /dataset/keys", f"/zarr/dataset/keys?url={q(fx['v2_geo'])}", lambda r: r.status_code == 200 and "temp" in r.json()),
        ("W8 ZARR v2 /info", f"/zarr/info?url={q(fx['v2_geo'])}&variable=temp", lambda r: r.status_code == 200 and r.json().get("width") == 128 and r.json().get("height") == 128),
        ("W8 ZARR v3 /validate", f"/zarr/validate?url={q(fx['v3_southup'])}&variable=sst", lambda r: r.status_code == 200 and r.json().get("bounds") == [-123.0, 37.0, -122.0, 38.0]),
    ]:
        if wants(lb):
            r = client.get(cw_base + path)
            ok = check_fn(r) if r.status_code == 200 else False
            rec(lb, path, "zarr_meta", ok, f"status={r.status_code} {r.text[:80]}")

    # 2. Zarr v2 Projected EPSG:3857 exact tile z=12/657/1582 (.npy unscale=False/True, return_mask=false, .png)
    for unscale in (False, True):
        lb = f"W8 ZARR v2 proj tile .npy unscale={unscale}"
        if not wants(lb):
            continue
        path = f"/zarr/tiles/WebMercatorQuad/12/657/1582.npy?url={q(fx['v2_proj'])}&variable=elev&unscale={'true' if unscale else 'false'}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            rec(lb, path, "zarr_npy", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = try_npy(lb, path, rc.content)
            if arr is not None:
                low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
                mask = arr[-1] > low
                exp_mask = fx["elev"] != -32768
                exp_val = fx["elev"].astype(np.float64) * 0.1 + 100.0 if unscale else fx["elev"].astype(np.float64)
                mask_eq = float((mask == exp_mask).mean()) * 100.0
                both = mask & exp_mask
                max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp_val[both]))) if both.any() else 0.0
                rec(lb, path, "zarr_npy", arr.shape == (2, 256, 256) and mask_eq == 100.0 and max_err < 1e-2, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    lb_nm = "W8 ZARR v2 proj tile .npy return_mask=false"
    if wants(lb_nm):
        path = f"/zarr/tiles/WebMercatorQuad/12/657/1582.npy?url={q(fx['v2_proj'])}&variable=elev&return_mask=false"
        rc = client.get(cw_base + path)
        if rc.status_code != 200:
            rec(lb_nm, path, "zarr_npy", False, f"HTTP {rc.status_code}")
        else:
            arr = try_npy(lb_nm, path, rc.content)
            if arr is not None:
                valid = fx["elev"] != -32768
                max_err = float(np.max(np.abs(arr[0][valid].astype(np.float64) - fx["elev"][valid].astype(np.float64))))
                zeroed = bool((arr[0][~valid] == 0.0).all())
                rec(lb_nm, path, "zarr_npy", arr.shape == (1, 256, 256) and zeroed and max_err < 1e-3, f"shape={arr.shape} zeroed={zeroed} max_err={max_err:.5f}")

    lb_png = "W8 ZARR v2 proj tile .png"
    if wants(lb_png):
        path = f"/zarr/tiles/WebMercatorQuad/12/657/1582.png?url={q(fx['v2_proj'])}&variable=elev&rescale=500,2000&colormap_name=viridis"
        rc = client.get(cw_base + path)
        if rc.status_code != 200:
            rec(lb_png, path, "zarr_png", False, f"HTTP {rc.status_code}")
        else:
            rgba = np.array(Image.open(io.BytesIO(rc.content)).convert("RGBA"))
            exp_mask = fx["elev"] != -32768
            alpha_ok = bool(((rgba[..., 3] > 0) == exp_mask).all())
            rgb_std = float(rgba[exp_mask, :3].std())
            detail = f"shape={rgba.shape} alpha_ok={alpha_ok} rgb_std={rgb_std:.1f}"
            if gpu_base is not None:
                rg = client.get(gpu_base + path)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb_png, "detail": f"status cpu=200 gpu={rg.status_code}"})
                else:
                    iok, idet = compare_images(rc.content, rg.content)
                    if not iok:
                        gpu_diffs.append({"label": lb_png, "detail": idet})
                    detail = f"{detail} | CPU==GPU {idet}"
            rec(lb_png, path, "zarr_png", rgba.shape == (256, 256, 4) and alpha_ok and rgb_std > 1.0, detail)

    # 3. Zarr v2 Geographic 3D: sel=time=0/1 & unscale=False/True on /point
    lat_p, lon_p = float(fx["lat"][64]), float(fx["lon"][64])
    for t_idx in (0, 1):
        for unscale in (False, True):
            lb = f"W8 ZARR v2 3D /point sel=time={t_idx} unscale={unscale}"
            if not wants(lb):
                continue
            path = f"/zarr/point/{lon_p:.8f},{lat_p:.8f}?url={q(fx['v2_geo'])}&variable=temp&sel={qe(f'time={t_idx}')}&unscale={'true' if unscale else 'false'}"
            rc = client.get(cw_base + path)
            if gpu_base is not None:
                rg = client.get(gpu_base + path)
                if rc.status_code != rg.status_code or (rc.status_code == 200 and first_json_diff(rc.json(), rg.json()) is not None):
                    gpu_diffs.append({"label": lb, "detail": f"cpu={rc.text[:80]} gpu={rg.text[:80]}"})
            if rc.status_code != 200:
                rec(lb, path, "zarr_point", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
            else:
                got = rc.json()["values"][0]
                raw_v = float(fx["data_3d"][t_idx, 64, 64])
                exp = raw_v * 0.5 + 10.0 if unscale else raw_v
                rec(lb, path, "zarr_point", abs(got - exp) < 1e-3, f"got={got:.4f} exp={exp:.4f}")

    # 4. Zarr v2 Geographic 3D /preview & /bbox
    for lb, path, exp_arr in [
        ("W8 ZARR v2 3D /preview/128x128.npy sel=time=1", f"/zarr/preview/128x128.npy?url={q(fx['v2_geo'])}&variable=temp&sel={qe('time=1')}", fx["data_3d"][1]),
        ("W8 ZARR v2 3D /bbox 128x128.npy sel=time=0", f"/zarr/bbox/-123,37,-122,38/128x128.npy?url={q(fx['v2_geo'])}&variable=temp&sel={qe('time=0')}", fx["data_3d"][0]),
    ]:
        if not wants(lb):
            continue
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            rec(lb, path, "zarr_npy", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = try_npy(lb, path, rc.content)
            if arr is not None:
                low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
                mask = arr[-1] > low
                exp_mask = exp_arr != -9999.0
                both = mask & exp_mask
                mask_eq = float((mask == exp_mask).mean()) * 100.0
                max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp_arr[both]))) if both.any() else 0.0
                rec(lb, path, "zarr_npy", arr.shape == (2, 128, 128) and mask_eq == 100.0 and max_err < 1e-3, f"shape={arr.shape} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    # 5. Zarr v2 GET & POST /statistics
    stats_path = f"/zarr/statistics?url={q(fx['v2_geo'])}&variable=temp&sel={qe('time=0')}"
    if wants("W8 ZARR v2 3D GET /statistics sel=time=0"):
        rc = client.get(cw_base + stats_path)
        if rc.status_code != 200:
            rec("W8 ZARR v2 3D GET /statistics sel=time=0", stats_path, "zarr_stats", False, f"HTTP {rc.status_code}")
        else:
            st = next(iter(rc.json().values()))
            valid = fx["data_3d"][0][fx["data_3d"][0] != -9999.0]
            rec("W8 ZARR v2 3D GET /statistics sel=time=0", stats_path, "zarr_stats", abs(st["mean"] - float(valid.mean())) < 1e-2 and int(st["count"]) == valid.size, f"mean={st['mean']:.4f}/{valid.mean():.4f} cnt={st['count']}/{valid.size}")

    if wants("W8 ZARR v2 3D POST /statistics west half"):
        poly_west = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-123.0, 37.0], [-122.5, 37.0], [-122.5, 38.0], [-123.0, 38.0], [-123.0, 37.0]]]}}
        rc = client.post(cw_base + stats_path, json=poly_west)
        if gpu_base is not None:
            rg = client.post(gpu_base + stats_path, json=poly_west)
            if rc.status_code != rg.status_code or (rc.status_code == 200 and first_json_diff(rc.json(), rg.json()) is not None):
                gpu_diffs.append({"label": "W8 ZARR v2 3D POST /statistics west half", "detail": "CPU vs GPU stats diff"})
        if rc.status_code != 200:
            rec("W8 ZARR v2 3D POST /statistics west half", stats_path, "zarr_stats", False, f"HTTP {rc.status_code}", method="POST")
        else:
            st = next(iter(rc.json()["properties"]["statistics"].values()))
            sub = fx["data_3d"][0][:, 0:64]
            valid = sub[sub != -9999.0]
            rec("W8 ZARR v2 3D POST /statistics west half", stats_path, "zarr_stats", abs(st["mean"] - float(valid.mean())) < 0.2 and int(st["count"]) == valid.size, f"mean={st['mean']:.4f}/{valid.mean():.4f} cnt={st['count']:.0f}/{valid.size}", method="POST")

    # 6. Zarr v3 ASCENDING latitude (south-up!) /point + /bbox
    if wants("W8 ZARR v3 south-up /point NE"):
        lat_ne, lon_ne = float(fx["lat_asc"][100]), float(fx["lon"][100])
        exp_ne = float(fx["v3_arr"][100, 100])
        path = f"/zarr/point/{lon_ne:.8f},{lat_ne:.8f}?url={q(fx['v3_southup'])}&variable=sst"
        rc = client.get(cw_base + path)
        if rc.status_code != 200:
            rec("W8 ZARR v3 south-up /point NE", path, "zarr_point", False, f"HTTP {rc.status_code}")
        else:
            got = rc.json()["values"][0]
            rec("W8 ZARR v3 south-up /point NE", path, "zarr_point", abs(got - exp_ne) < 1e-3, f"got={got:.4f} exp={exp_ne:.4f}")

    if wants("W8 ZARR v3 south-up /bbox 128x128.npy"):
        path = f"/zarr/bbox/-123,37,-122,38/128x128.npy?url={q(fx['v3_southup'])}&variable=sst"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": "W8 ZARR v3 south-up /bbox 128x128.npy", "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": "W8 ZARR v3 south-up /bbox 128x128.npy", "detail": gd})
        if rc.status_code != 200:
            rec("W8 ZARR v3 south-up /bbox 128x128.npy", path, "zarr_npy", False, f"HTTP {rc.status_code}")
        else:
            arr = try_npy("W8 ZARR v3 south-up /bbox 128x128.npy", path, rc.content)
            if arr is not None:
                low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
                mask = arr[-1] > low
                exp = fx["v3_arr"][::-1, :]
                exp_mask = np.isfinite(exp)
                both = mask & exp_mask
                mask_eq = float((mask == exp_mask).mean()) * 100.0
                max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp[both]))) if both.any() else 0.0
                rec("W8 ZARR v3 south-up /bbox 128x128.npy", path, "zarr_npy", arr.shape == (2, 128, 128) and mask_eq == 100.0 and max_err < 1e-3, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    # 7. Zarr v2 0..360° longitude /point
    if wants("W8 ZARR v2 0..360° lon /point (-122.5, 37.5)"):
        path = f"/zarr/point/{lon_p:.8f},{lat_p:.8f}?url={q(fx['v2_360'])}&variable=var360"
        rc = client.get(cw_base + path)
        if rc.status_code != 200:
            rec("W8 ZARR v2 0..360° lon /point (-122.5, 37.5)", path, "zarr_point", False, f"HTTP {rc.status_code}")
        else:
            got = rc.json()["values"][0]
            exp = float(fx["data_3d"][0, 64, 64])
            rec("W8 ZARR v2 0..360° lon /point (-122.5, 37.5)", path, "zarr_point", abs(got - exp) < 1e-3, f"got={got:.4f} exp={exp:.4f}")

    # 8. Zarr tile ETag/304 & HEAD
    tu = f"/zarr/tiles/WebMercatorQuad/12/657/1582.png?url={q(fx['v2_proj'])}&variable=elev&rescale=500,2000&colormap_name=viridis"
    if wants("W8 ZARR tile ETag + If-None-Match -> 304"):
        etag = client.get(cw_base + tu).headers.get("etag")
        r304 = client.get(cw_base + tu, headers={"If-None-Match": etag or ""})
        rec("W8 ZARR tile ETag + If-None-Match -> 304", tu, "zarr_http", bool(etag) and r304.status_code == 304 and len(r304.content) == 0, f"etag={etag} status={r304.status_code}")
    if wants("W8 ZARR tile HEAD -> 200 empty"):
        rh = client.head(cw_base + tu)
        rec("W8 ZARR tile HEAD -> 200 empty", tu, "zarr_http", rh.status_code == 200 and len(rh.content) == 0, f"status={rh.status_code} len={len(rh.content)}", method="HEAD")

    # 9. Regression: identical query string with .npy then .png must NOT collide in tile_cache
    if wants("W8 ZARR tile cache isolates .npy vs .png"):
        qs = f"url={q(fx['v2_proj'])}&variable=elev"
        p_npy = f"/zarr/tiles/WebMercatorQuad/12/657/1582.npy?{qs}"
        p_png = f"/zarr/tiles/WebMercatorQuad/12/657/1582.png?{qs}"
        r_npy = client.get(cw_base + p_npy)
        r_png = client.get(cw_base + p_png)
        ok_cache = (
            r_npy.status_code == 200
            and r_png.status_code == 200
            and r_npy.content.startswith(b"\x93NUMPY")
            and r_png.content.startswith(b"\x89PNG\r\n\x1a\n")
            and r_npy.headers.get("etag") != r_png.headers.get("etag")
        )
        rec("W8 ZARR tile cache isolates .npy vs .png", p_png, "zarr_http", ok_cache, f"npy_magic={r_npy.content[:6]!r} png_magic={r_png.content[:4]!r}")

    return ran


def main() -> int:
    parser = argparse.ArgumentParser(description="Adversarial differential oracle proving cogzilla is equal or intentionally better than TiTiler 2.4.1.")
    parser.add_argument("--cpu-only", action="store_true", help="Run cogzilla with COGZILLA_GPU_POLICY=never")
    parser.add_argument("--check-cpu-vs-gpu", action="store_true", help="Also verify CPU-only cogzilla matches GPU-enabled cogzilla")
    parser.add_argument("--cogzilla-bin", default=str(COGZILLA_BIN))
    parser.add_argument("--only", default="", help="Substring filter on check label")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    bin_path = Path(args.cogzilla_bin)
    if not bin_path.exists():
        raise SystemExit(f"cogzilla binary not found at {bin_path}; run `cargo build --release --bin cogzilla-server --locked`")

    with tempfile.TemporaryDirectory(prefix="cogzilla_adv_oracle_") as tmp:
        tmp_dir = Path(tmp)
        s = build_synthetic_fixtures(tmp_dir / "synth")
        tt_app = tmp_dir / "_tt_app.py"
        tt_app.write_text(
            "from fastapi import FastAPI\n"
            "from rio_tiler.errors import PointOutsideBounds\n"
            "from rio_tiler.io import STACReader\n"
            "from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers\n"
            "from titiler.core.factory import TilerFactory, MultiBaseTilerFactory, ColorMapFactory, TMSFactory, AlgorithmFactory\n"
            "app = FastAPI()\n"
            "add_exception_handlers(app, {**DEFAULT_STATUS_CODES, PointOutsideBounds: 404})\n"
            "cog = TilerFactory(router_prefix='/cog', add_ogc_maps=True)\n"
            "stac = MultiBaseTilerFactory(reader=STACReader, router_prefix='/stac', add_ogc_maps=True)\n"
            "app.include_router(cog.router, prefix='/cog')\n"
            "app.include_router(stac.router, prefix='/stac')\n"
            "app.include_router(ColorMapFactory().router)\n"
            "app.include_router(TMSFactory().router)\n"
            "app.include_router(AlgorithmFactory().router)\n"
        )

        port_cw = pick_free_port()
        port_tt = pick_free_port()
        procs: list[subprocess.Popen] = []

        try:
            # When --check-cpu-vs-gpu is enabled, always run primary cw_proc as CPU-only
            # and gpu_proc with GPU enabled so we genuinely compare CPU vs GPU.
            primary_cpu_only = args.cpu_only or args.check_cpu_vs_gpu
            cw_proc = start_cogzilla(bin_path, port_cw, cpu_only=primary_cpu_only)
            procs.append(cw_proc)

            port_gpu = None
            if args.check_cpu_vs_gpu:
                port_gpu = pick_free_port()
                gpu_proc = start_cogzilla(bin_path, port_gpu, cpu_only=False)
                procs.append(gpu_proc)

            tt_env = os.environ.copy()
            tt_env.update({"GDAL_CACHEMAX": "256", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "VSI_CACHE": "TRUE"})
            tt_proc = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn", "_tt_app:app",
                    "--app-dir", str(tmp_dir), "--host", "127.0.0.1", "--port", str(port_tt),
                    "--workers", "1", "--log-level", "error",
                ],
                env=tt_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs.append(tt_proc)

            wait_ready(f"http://127.0.0.1:{port_cw}/healthz")
            wait_ready(f"http://127.0.0.1:{port_tt}/openapi.json")
            if port_gpu is not None:
                wait_ready(f"http://127.0.0.1:{port_gpu}/healthz")
            for p in procs:
                assert p.poll() is None, "child server exited early"

            cw_base = f"http://127.0.0.1:{port_cw}"
            tt_base = f"http://127.0.0.1:{port_tt}"
            gpu_base = f"http://127.0.0.1:{port_gpu}" if port_gpu else None

            sat = FIXTURES / "satellite_utm10n_rgb_zstd_p3.tif"
            merc = FIXTURES / "mercator_4k_zstd_p3.tif"
            geo4326 = FIXTURES / "geo4326_4k_deflate_p3.tif"
            scale_off = CORPUS / "meta_landsat_scale_offset.tif"
            nd_f32 = CORPUS / "nodata_float32_nan_swath.tif"
            nd_i16 = CORPUS / "nodata_int16_minus9999.tif"
            mask_1bit = CORPUS / "mask_jpeg_internal_1bit.tif"
            mask_rgba = CORPUS / "mask_rgba_4band_u8.tif"
            rgbn_u16 = CORPUS / "layout_4band_rgbn_u16.tif"
            no_ovr = CORPUS / "struct_512tiles_no_ovr.tif"
            polar3413 = CORPUS / "proj_3413_arctic_polar.tif"
            palette = EXTRA / "palette_u8.tif"
            geo360 = EXTRA / "geo_0_360.tif"
            landsat_stac = FIXTURES / "landsat_c2_stac.json"

            # (label, method, path, body, kind)
            cases: list[tuple[str, str, str, dict | None, str]] = []

            # --- 1. Unscale across endpoints ---
            lon, lat, z, tx, ty, b = center_lonlat_and_tile(scale_off)
            poly_so = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
                [b[0] + 0.25 * (b[2] - b[0]), b[1] + 0.25 * (b[3] - b[1])],
                [b[0] + 0.75 * (b[2] - b[0]), b[1] + 0.25 * (b[3] - b[1])],
                [b[0] + 0.75 * (b[2] - b[0]), b[1] + 0.75 * (b[3] - b[1])],
                [b[0] + 0.25 * (b[2] - b[0]), b[1] + 0.75 * (b[3] - b[1])],
                [b[0] + 0.25 * (b[2] - b[0]), b[1] + 0.25 * (b[3] - b[1])],
            ]]}, "properties": {}}
            cases += [
                ("PARITY UNSCALE point true", "GET", f"/cog/point/{lon:.5f},{lat:.5f}?url={q(scale_off)}&unscale=true", None, "point"),
                ("PARITY UNSCALE tile .npy true", "GET", f"/cog/tiles/WebMercatorQuad/{z}/{tx}/{ty}.npy?url={q(scale_off)}&unscale=true", None, "npy"),
                ("PARITY UNSCALE tile .tif true", "GET", f"/cog/tiles/WebMercatorQuad/{z}/{tx}/{ty}.tif?url={q(scale_off)}&unscale=true", None, "geotiff"),
                ("PARITY UNSCALE stats GET true", "GET", f"/cog/statistics?url={q(scale_off)}&unscale=true&max_size=256", None, "stats"),
                ("PARITY UNSCALE stats POST max_size=256", "POST", f"/cog/statistics?url={q(scale_off)}&unscale=true&max_size=256", poly_so, "post_stats"),
                ("PARITY UNSCALE feature .npy true", "POST", f"/cog/feature/64x64.npy?url={q(scale_off)}&unscale=true", poly_so, "npy"),
            ]

            # --- 2. All 8 GDAL dtypes (.npy, .tif, return_mask=false, stats, point) ---
            for dt_name in ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"):
                p = DTYPES / f"dtype_{dt_name}.tif"
                lon, lat, _, _, _, _ = center_lonlat_and_tile(p, z=12)
                cases += [
                    (f"PARITY DTYPE [{dt_name}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(p)}", None, "npy"),
                    (f"PARITY DTYPE [{dt_name}] .npy no-mask", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(p)}&return_mask=false", None, "npy"),
                    (f"PARITY DTYPE [{dt_name}] .tif", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.tif?url={q(p)}", None, "geotiff"),
                    (f"PARITY DTYPE [{dt_name}] stats", "GET", f"/cog/statistics?url={q(p)}&max_size=128", None, "stats"),
                    (f"PARITY DTYPE [{dt_name}] point", "GET", f"/cog/point/{lon:.5f},{lat:.5f}?url={q(p)}", None, "point"),
                ]

            # --- 3. Synthetic torture COGs ---
            cases += [
                ("PARITY SYNTH rotated point", "GET", f"/cog/point/-122.30,37.75?url={q(s['rotated'])}", None, "point"),
                ("PARITY SYNTH south_up .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/82/197.npy?url={q(s['south_up'])}", None, "npy"),
                ("PARITY SYNTH south_up point", "GET", f"/cog/point/-121.705,38.205?url={q(s['south_up'])}", None, "point"),
                ("PARITY SYNTH highprec_i32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['highprec_i32'])}", None, "npy"),
                ("PARITY SYNTH highprec_i32 point", "GET", f"/cog/point/-122.30,37.75?url={q(s['highprec_i32'])}", None, "point"),
                ("PARITY SYNTH highprec_i32 stats", "GET", f"/cog/statistics?url={q(s['highprec_i32'])}&max_size=256", None, "stats"),
                ("PARITY SYNTH perband_so unscale .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['perband_so'])}&unscale=true", None, "npy"),
                ("PARITY SYNTH perband_so expr unscale", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['perband_so'])}&unscale=true&expression={qe('b1+b2+b3')}", None, "npy"),
                ("PARITY SYNTH sparse default .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['sparse_i16'])}", None, "npy"),
                ("PARITY SYNTH sparse nodata=0 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['sparse_i16'])}&nodata=0", None, "npy"),
                ("PARITY SYNTH sparse stats nodata=0", "GET", f"/cog/statistics?url={q(s['sparse_i16'])}&nodata=0&max_size=256", None, "stats"),
                ("PARITY SYNTH categorical stats", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&categorical=true&max_size=256", None, "stats"),
                ("PARITY SYNTH categorical subset", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&categorical=true&c=1&c=5&max_size=256", None, "stats"),
                ("PARITY SYNTH custom hist edges", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&histogram_bins=0,2,6,12&max_size=256", None, "stats"),
                ("PARITY SYNTH custom percentiles", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&p=10&p=50&p=90&max_size=256", None, "stats"),
                ("PARITY SYNTH neg_f32 ceil .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['neg_f32'])}&algorithm=ceil", None, "npy"),
                ("PARITY SYNTH neg_f32 floor .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['neg_f32'])}&algorithm=floor", None, "npy"),
                ("PARITY SYNTH neg_f32 cast default .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['neg_f32'])}&algorithm=cast", None, "npy"),
                ("PARITY SYNTH tiny_1x1 cubic .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['tiny_1x1'])}&resampling=cubic", None, "npy"),
                ("PARITY SYNTH tiny_3x3 lanczos .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['tiny_3x3'])}&resampling=lanczos", None, "npy"),
                ("PARITY SYNTH constant_42 stats", "GET", f"/cog/statistics?url={q(s['constant'])}", None, "stats"),
                ("PARITY SYNTH constant_42 rescale=42,42", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['constant'])}&rescale=42,42&colormap_name=viridis", None, "png"),
                ("PARITY SYNTH all_nodata .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['all_nodata'])}&rescale=0,100&colormap_name=viridis", None, "png"),
                ("PARITY SYNTH all_nodata .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['all_nodata'])}", None, "npy"),
                ("PARITY SYNTH all_nodata stats", "GET", f"/cog/statistics?url={q(s['all_nodata'])}", None, "stats"),
                ("PARITY SYNTH nd_i16_min .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['nd_i16_min'])}", None, "npy"),
                ("PARITY SYNTH nd_u8_255 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['nd_u8_255'])}", None, "npy"),
                ("PARITY SYNTH nd_u16_max .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['nd_u16_max'])}", None, "npy"),
                ("PARITY SYNTH f32_infs .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['f32_infs'])}", None, "npy"),
                ("PARITY SYNTH f32_infs stats", "GET", f"/cog/statistics?url={q(s['f32_infs'])}", None, "stats"),
                ("PARITY SYNTH planar_1bit .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['planar_1bit'])}", None, "png"),
                ("PARITY SYNTH planar_rgba .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['planar_rgba'])}", None, "png"),
                ("PARITY SYNTH antimeridian east .npy", "GET", f"/cog/tiles/WebMercatorQuad/6/63/35.npy?url={q(s['antimeridian'])}", None, "npy"),
            ]

            # --- 4. Nodata-boundary resampling & /statistics resampling ---
            for rm in ("bilinear", "cubic", "cubic_spline", "lanczos", "average", "gauss"):
                cases.append((f"PARITY NDBND f32_swath [{rm}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(nd_f32)}&resampling={rm}", None, "npy"))
            for rm in ("bilinear", "cubic", "average"):
                cases.append((f"PARITY STATRESAMP i16_nd [{rm}]", "GET", f"/cog/statistics?url={q(nd_i16)}&resampling={rm}&max_size=128", None, "stats"))

            # --- 5. Non-Mercator & Polar TMS, 16-bit PNGs, POST /statistics & /feature, OGC /map, STAC ---
            donut_poly = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
                [[-122.80, 37.60], [-122.50, 37.60], [-122.50, 37.90], [-122.80, 37.90], [-122.80, 37.60]],
                [[-122.70, 37.70], [-122.60, 37.70], [-122.60, 37.80], [-122.70, 37.80], [-122.70, 37.70]],
            ]}, "properties": {}}
            multi_poly = {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [
                [[[-122.80, 37.60], [-122.70, 37.60], [-122.70, 37.70], [-122.80, 37.70], [-122.80, 37.60]]],
                [[[-122.60, 37.80], [-122.50, 37.80], [-122.50, 37.90], [-122.60, 37.90], [-122.60, 37.80]]],
            ]}, "properties": {}}
            with rasterio.open(no_ovr) as ds:
                b2 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            poly_no_ovr = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
                [b2[0] + 0.1 * (b2[2] - b2[0]), b2[1] + 0.1 * (b2[3] - b2[1])],
                [b2[0] + 0.9 * (b2[2] - b2[0]), b2[1] + 0.1 * (b2[3] - b2[1])],
                [b2[0] + 0.9 * (b2[2] - b2[0]), b2[1] + 0.9 * (b2[3] - b2[1])],
                [b2[0] + 0.1 * (b2[2] - b2[0]), b2[1] + 0.9 * (b2[3] - b2[1])],
                [b2[0] + 0.1 * (b2[2] - b2[0]), b2[1] + 0.1 * (b2[3] - b2[1])],
            ]]}, "properties": {}}
            cases += [
                ("PARITY TMS WorldCRS84Quad .npy", "GET", f"/cog/tiles/WorldCRS84Quad/6/20/18.npy?url={q(geo4326)}", None, "npy"),
                ("PARITY TMS UPSArctic (2,1,2) .npy", "GET", f"/cog/tiles/UPSArcticWGS84Quad/2/1/2.npy?url={q(polar3413)}", None, "npy"),
                ("PARITY U16PNG 1-band .png", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(DTYPES / 'dtype_uint16.tif')}", None, "png"),
                ("PARITY U16PNG 3-band RGB .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.png?url={q(rgbn_u16)}&bidx=1&bidx=2&bidx=3", None, "png"),
                ("PARITY POSTSTATS MultiPolygon max_size=256", "POST", f"/cog/statistics?url={q(sat)}&max_size=256", multi_poly, "post_stats"),
                ("PARITY POSTSTATS donut max_size=256", "POST", f"/cog/statistics?url={q(sat)}&max_size=256", donut_poly, "post_stats"),
                ("PARITY POSTSTATS no-overview max_size=128", "POST", f"/cog/statistics?url={q(no_ovr)}&max_size=128", poly_no_ovr, "post_stats"),
                ("PARITY FEAT donut .png", "POST", f"/cog/feature/64x64.png?url={q(sat)}&rescale=0,0.35", donut_poly, "png"),
                ("PARITY OGC map bbox-crs=3857 .png", "GET", f"/cog/map?url={q(sat)}&bbox=-13665000,4530000,-13635000,4560000&bbox-crs=http://www.opengis.net/def/crs/EPSG/0/3857&width=64&height=64&rescale=0,0.35&f=png", None, "png"),
                ("PARITY BUFFER=3 hillshade .png", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&algorithm=hillshade&buffer=3", None, "png"),
                ("PARITY BUFFER=3 slope .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(merc)}&algorithm=slope&buffer=3", None, "npy"),
                ("PARITY EXPR+ALG grayscale .png", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&expression=b1*2;b2*2;b3*2&algorithm=grayscale&rescale=0,0.7", None, "png"),
                ("PARITY NODATA i16 override=0 stats", "GET", f"/cog/statistics?url={q(nd_i16)}&nodata=0&max_size=128", None, "stats"),
                ("PARITY STACMIX u8+f32+u16 .npy", "GET", f"/stac/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['stac_mix'])}&assets=a&assets=b&assets=c", None, "npy"),
                ("PARITY STACMIX expr a+b .npy", "GET", f"/stac/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['stac_mix'])}&assets=a&assets=b&expression={qe('b1+b2')}", None, "npy"),
                ("PARITY STAC asset_bidx .npy", "GET", f"/stac/tiles/WebMercatorQuad/12/652/1583.npy?url={q(landsat_stac)}&assets=red&assets=green&asset_bidx=red|1&asset_bidx=green|1", None, "npy"),
            ]

            # --- 5b. Wave 5: Exotic CRSs, GCP-only, multi-overview+unscale, exotic codecs ---
            for name in ("laea_3035", "lcc_3978", "nztm_2193", "ease_6933", "ups_n_32661", "ups_s_32761", "bng_27700", "nad83_4269", "modis_sinu"):
                p = s[name]
                lon, lat, z, tx, ty, _ = center_lonlat_and_tile(p)
                cases += [
                    (f"PARITY CRS [{name}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/{z}/{tx}/{ty}.npy?url={q(p)}", None, "npy"),
                    (f"PARITY CRS [{name}] point", "GET", f"/cog/point/{lon:.5f},{lat:.5f}?url={q(p)}", None, "point"),
                ]
            cases += [
                ("PARITY GCP affine .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/81/197.npy?url={q(s['gcp_affine'])}", None, "npy"),
                ("PARITY GCP affine point", "GET", f"/cog/point/-122.5,37.5?url={q(s['gcp_affine'])}", None, "point"),
                ("PARITY OVR+UNSCALE z9 cubic .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/81/197.npy?url={q(s['multi_ovr_unscale'])}&unscale=true&resampling=cubic", None, "npy"),
                ("PARITY OVR+UNSCALE z9 average .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/81/197.npy?url={q(s['multi_ovr_unscale'])}&unscale=true&resampling=average", None, "npy"),
                ("PARITY CODEC stripped_f64_p3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['stripped_f64_p3'])}", None, "npy"),
                ("PARITY CODEC bigtiff_i32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['bigtiff_i32'])}", None, "npy"),
                ("PARITY CODEC int8_unscale .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['int8_unscale'])}&unscale=true", None, "npy"),
            ]

            # --- 5c. Wave 6: Colormap LUTs, Content-Crs/Content-Bbox headers, expr VM, custom colormaps ---
            for cm in ("viridis", "plasma", "inferno", "terrain", "coolwarm", "turbo", "gray", "twilight"):
                cases.append((f"PARITY CMAP LUT [{cm}]", "GET", f"/colorMaps/{cm}", None, "json_exact"))
            cases += [
                ("PARITY HDR tile WebMercatorQuad", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000", None, "headers"),
                ("PARITY HDR bbox dst_crs=3857", "GET", f"/cog/bbox/-122.8,37.6,-122.5,37.9/64x64.png?url={q(sat)}&dst_crs=epsg:3857&rescale=0,0.35", None, "headers"),
                ("PARITY HDR preview native", "GET", f"/cog/preview/64x64.png?url={q(sat)}&rescale=0,0.35", None, "headers"),
                ("PARITY HDR feature 4326", "POST", f"/cog/feature/64x64.png?url={q(sat)}&rescale=0,0.35", donut_poly, "headers"),
                ("PARITY EXPR trig+log+pow .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('sin(b1)+log1p(b2)+(b3+0.1)**-0.5')}", None, "npy"),
                ("PARITY EXPR where+and int32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('where((b1>0.1)&(b2<0.3),1,0)')}", None, "npy"),
                ("PARITY EXPR where+or f32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('where((b1>0.25)|(b3>0.25),b1,b2)')}", None, "npy"),
                ("PARITY CMAP neg float interval .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['neg_f32'])}&colormap={qe(json.dumps([[[-100,0],[0,0,255,255]],[[0,100],[255,0,0,255]]]))}", None, "png"),
                ("PARITY CMAP semi-transparent alpha .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['cat_u8'])}&colormap={qe(json.dumps({'1':[255,0,0,128],'2':[0,255,0,0],'3':[0,0,255,255],'5':[255,255,0,200],'10':[255,0,255,255]}))}", None, "png"),
            ]

            # --- 6. INTENTIONALLY BETTER contracts (docs/titiler-divergences.md) ---
            cases += [
                ("BETTER inverted rescale (cw inverts, tt clips 255)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=1000,0&colormap_name=viridis", None, "better_inverted_rescale"),
                ("BETTER resampling=min (cw 200, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000&colormap_name=viridis&resampling=min", None, "better_cw200_tt422"),
                ("BETTER resampling=med (cw 200, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000&colormap_name=viridis&resampling=med", None, "better_cw200_tt422"),
                ("BETTER color_formula 1-band gray (cw 200, tt 500)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000&color_formula={qe('Gamma RGB 1.8')}", None, "better_cw200_tt500"),
                ("BETTER 0..360 tile west (cw 200, tt 404)", "GET", f"/cog/tiles/WebMercatorQuad/2/0/1.png?url={q(geo360)}&rescale=0,255&colormap_name=viridis", None, "better_cw200_tt404"),
                ("BETTER 0..360 point -122.5,37.5 (cw 200, tt 404)", "GET", f"/cog/point/-122.5,37.5?url={q(geo360)}", None, "better_cw200_tt404"),
                ("BETTER antimeridian west tile (cw 200, tt 404)", "GET", f"/cog/tiles/WebMercatorQuad/6/0/35.npy?url={q(s['antimeridian'])}", None, "better_cw200_tt404"),
                ("BETTER polar TMS OOB (cw 404, tt blank 200)", "GET", f"/cog/tiles/UPSArcticWGS84Quad/2/1/1.npy?url={q(polar3413)}", None, "better_cw404_tt200"),
                ("BETTER ungeoref preview (cw 200, tt 500)", "GET", f"/cog/preview/64x64.png?url={q(s['ungeoref'])}", None, "better_cw200_tt500"),
                ("BETTER out-of-range bidx=99 (cw 400, tt 500)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&bidx=99", None, "better_cw4xx_tt500"),
                ("BETTER palette .npy raw indices (cw 2ch, tt 4ch)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.npy?url={q(palette)}", None, "better_palette_npy"),
                ("BETTER 2-band Float32 .png (cw 400, tt 200)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&bidx=1&bidx=2&rescale=0,0.35", None, "better_cw400_tt200"),
                ("BETTER 2-band Uint16 .png (cw 400)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.png?url={q(rgbn_u16)}&bidx=1&bidx=2", None, "better_cw400"),
                ("BETTER mask_1bit return_mask=false (valid match, collar zeroed)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.png?url={q(mask_1bit)}&return_mask=false", None, "better_mask_zeroed"),
                ("BETTER mask_rgba return_mask=false (valid match, collar zeroed)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.png?url={q(mask_rgba)}&return_mask=false", None, "better_mask_zeroed"),
                ("BETTER slope seamless 256x256 (cw 256, tt 250)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(merc)}&algorithm=slope", None, "better_slope_256"),
                ("BETTER cast dtype=int16 (cw int16, tt hardcoded u8)", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['neg_f32'])}&algorithm=cast&algorithm_params={qe(json.dumps({'dtype':'int16'}))}", None, "better_cast_i16"),
            ]

            if args.only:
                cases = [c for c in cases if args.only.lower() in c[0].lower()]

            mode_str = "CPU-ONLY" if primary_cpu_only else "DEFAULT"
            print(f"\n=== Adversarial Differential Oracle ({mode_str}): {len(cases)} TiTiler-diff checks + Wave 7 Mosaic/HTTP checks ===")
            failures = []
            gpu_diffs = []
            with httpx.Client(timeout=30.0) as client:
                for label, method, path, body, kind in cases:
                    r_cw = client.request(method, cw_base + path, json=body)
                    try:
                        r_tt = client.request(method, tt_base + path, json=body)
                    except httpx.HTTPError:
                        with httpx.Client(timeout=30.0) as c2:
                            r_tt = c2.request(method, tt_base + path, json=body)

                    is_unmasked = "return_mask=false" in path
                    is_smooth = "resampling=" in path and "nearest" not in path

                    if gpu_base is not None:
                        r_gpu = client.request(method, gpu_base + path, json=body)
                        if r_cw.status_code != r_gpu.status_code:
                            gpu_diffs.append({"label": label, "detail": f"status cpu={r_cw.status_code} gpu={r_gpu.status_code}"})
                        elif r_cw.status_code == 200:
                            ct = r_cw.headers.get("content-type", "")
                            if kind == "headers":
                                hok, hd = same_place(r_cw, r_gpu)
                                iok, idet = compare_images(r_cw.content, r_gpu.content)
                                gok, gd = (hok and iok), (hd if not hok else idet)
                            elif kind == "json_exact":
                                d = first_json_diff(r_cw.json(), r_gpu.json())
                                gok, gd = (d is None), (d or "exact JSON match")
                            elif "image/png" in ct or "image/jpeg" in ct or "image/webp" in ct:
                                gok, gd = compare_images(r_cw.content, r_gpu.content)
                            elif "application/x-binary" in ct or ".npy" in path:
                                gok, gd = compare_npy(r_cw.content, r_gpu.content, check_values=True, masked=not is_unmasked, smooth=is_smooth)
                            elif "image/tiff" in ct:
                                gok, gd = compare_geotiffs(r_cw.content, r_gpu.content, compare_pixels=True)
                            elif kind == "stats":
                                gok, gd = compare_stats_strict(r_cw.json(), r_gpu.json())
                            elif kind == "post_stats":
                                gok, gd = compare_stats_strict(r_cw.json()["properties"]["statistics"], r_gpu.json()["properties"]["statistics"])
                            elif kind == "point" or "/point/" in path:
                                gok, gd = compare_point_strict(r_cw.json(), r_gpu.json())
                            else:
                                gok, gd = (r_cw.content == r_gpu.content), "bytes diff"
                            if not gok:
                                gpu_diffs.append({"label": label, "detail": gd})

                    if kind == "better_inverted_rescale":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGB"))
                            at = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGB"))
                            cw_spatial_std = max(float(ac[..., c].std()) for c in range(3))
                            tt_spatial_std = max(float(at[..., c].std()) for c in range(3))
                            ok = cw_spatial_std > 1.0 and tt_spatial_std < 1e-6
                            detail = f"cw spatial_std={cw_spatial_std:.1f} vs tt spatial_std={tt_spatial_std:.1f}"
                    elif kind in ("better_cw200_tt422", "better_cw200_tt500", "better_cw200_tt404"):
                        exp_tt = int(kind.split("_tt")[1])
                        payload_ok = False
                        if r_cw.status_code == 200:
                            ct = r_cw.headers.get("content-type", "")
                            if "image/" in ct:
                                im = Image.open(io.BytesIO(r_cw.content))
                                payload_ok = im.size[0] > 0 and im.size[1] > 0
                            elif ".npy" in path:
                                arr = np.load(io.BytesIO(r_cw.content))
                                payload_ok = arr.size > 0
                            elif "json" in ct:
                                payload_ok = len(r_cw.json().get("values", [])) > 0
                        ok = r_cw.status_code == 200 and payload_ok and r_tt.status_code == exp_tt
                        detail = f"cw={r_cw.status_code}(valid={payload_ok}) tt={r_tt.status_code}"
                    elif kind == "better_cw404_tt200":
                        ok = r_cw.status_code == 404 and r_tt.status_code == 200
                        detail = f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_cw4xx_tt500":
                        ok = (400 <= r_cw.status_code < 500) and r_tt.status_code == 500
                        detail = f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_cw400_tt200":
                        ok = r_cw.status_code == 400 and r_tt.status_code == 200
                        detail = f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_cw400":
                        ok = r_cw.status_code == 400
                        detail = f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_palette_npy":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            at = np.load(io.BytesIO(r_tt.content))
                            uniq = set(np.unique(ac[0]).tolist())
                            ok = ac.shape == (2, 256, 256) and ac.dtype == np.uint8 and at.shape == (4, 256, 256) and len(uniq) >= 2
                            detail = f"cw={ac.shape}:{ac.dtype} uniq={sorted(uniq)[:6]} tt={at.shape}:{at.dtype}"
                    elif kind == "better_mask_zeroed":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGB"))
                            at = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGB"))
                            valid = ac.sum(axis=-1) > 0
                            zeroed = ~valid
                            mae = float(np.abs(ac[valid].astype(np.float32) - at[valid].astype(np.float32)).mean()) if valid.any() else 999.0
                            ok = ac.shape == at.shape and int(valid.sum()) > 1000 and int(zeroed.sum()) > 1000 and mae < 1.0
                            detail = f"valid_px={int(valid.sum())} zeroed_px={int(zeroed.sum())} valid_MAE={mae:.2f}"
                    elif kind == "better_slope_256":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            at = np.load(io.BytesIO(r_tt.content))
                            diff = float(np.nanmax(np.abs(ac[0, 3:253, 3:253] - at[0])))
                            border_valid = bool((ac[1, 0, :] > 0).all() and (ac[1, -1, :] > 0).all() and (ac[1, :, 0] > 0).all() and (ac[1, :, -1] > 0).all())
                            ok = ac.shape == (2, 256, 256) and at.shape == (2, 250, 250) and diff < 1e-2 and border_valid
                            detail = f"cw={ac.shape} tt={at.shape} inner_max_diff={diff:.4f} border_valid={border_valid}"
                    elif kind == "better_cast_i16":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            at = np.load(io.BytesIO(r_tt.content))
                            valid = ac[1] > np.iinfo(np.int16).min
                            ok = ac.dtype == np.int16 and at.dtype == np.uint8 and bool(valid.any()) and int(ac[0][valid].min()) < 0
                            detail = f"cw={ac.dtype} min={int(ac[0][valid].min())} tt={at.dtype}"
                    elif r_cw.status_code != 200 or r_tt.status_code != 200:
                        ok = False
                        detail = f"HTTP cw={r_cw.status_code} ({r_cw.text[:100]!r}) vs tt={r_tt.status_code} ({r_tt.text[:100]!r})"
                    elif kind == "png":
                        ok, detail = compare_images(r_cw.content, r_tt.content)
                    elif kind == "npy":
                        ok, detail = compare_npy(r_cw.content, r_tt.content, check_values=True, masked=not is_unmasked, smooth=is_smooth)
                    elif kind == "geotiff":
                        ok, detail = compare_geotiffs(r_cw.content, r_tt.content, compare_pixels=True)
                    elif kind == "json_exact":
                        d = first_json_diff(r_cw.json(), r_tt.json())
                        ok, detail = (d is None), (d or "exact JSON match")
                    elif kind == "headers":
                        if not (r_cw.headers.get("content-crs") and r_cw.headers.get("content-bbox") and r_tt.headers.get("content-crs") and r_tt.headers.get("content-bbox")):
                            ok, detail = False, f"missing geo headers: cw=({r_cw.headers.get('content-crs')},{r_cw.headers.get('content-bbox')}) tt=({r_tt.headers.get('content-crs')},{r_tt.headers.get('content-bbox')})"
                        else:
                            ok, detail = same_place(r_cw, r_tt)
                    elif kind == "stats":
                        ok, detail = compare_stats_strict(r_cw.json(), r_tt.json())
                    elif kind == "post_stats":
                        ok, detail = compare_stats_strict(r_cw.json()["properties"]["statistics"], r_tt.json()["properties"]["statistics"])
                    elif kind == "point":
                        ok, detail = compare_point_strict(r_cw.json(), r_tt.json())
                    else:
                        raise ValueError(f"Unknown check kind: {kind!r}")

                    tag = "PASS" if ok else "FAIL"
                    if not ok:
                        failures.append({"label": label, "method": method, "path": path, "kind": kind, "detail": detail})
                    print(f"{tag} | {label:54s} | {detail}")

                w7_ran = run_wave7_checks(client, cw_base, gpu_base, tmp_dir, args.only, failures, gpu_diffs)
                w8_ran = run_wave8_checks(client, cw_base, gpu_base, tmp_dir, args.only, failures, gpu_diffs)

            total_checks = len(cases) + w7_ran + w8_ran

            if gpu_diffs:
                print("\nCPU-vs-GPU differences:")
                for gd in gpu_diffs:
                    print(f"  GPU_DIFF | {gd['label']:54s} | {gd['detail']}")

            if args.json_out:
                Path(args.json_out).write_text(json.dumps({
                    "mode": mode_str,
                    "total": total_checks,
                    "failed": len(failures),
                    "gpu_diffs": gpu_diffs,
                    "failures": failures,
                }, indent=2))
            print(f"\nSummary ({mode_str}): {total_checks - len(failures)}/{total_checks} passed, {len(failures)} failed" + (f", {len(gpu_diffs)} CPU-vs-GPU diffs" if args.check_cpu_vs_gpu else ""))
            return 1 if (failures or gpu_diffs) else 0
        finally:
            for p in procs:
                try:
                    if p.poll() is None:
                        p.terminate()
                except OSError:
                    pass
            for p in procs:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        p.kill()
                        p.wait(timeout=5)
                    except OSError:
                        pass


if __name__ == "__main__":
    sys.exit(main())
