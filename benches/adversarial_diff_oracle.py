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
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
from PIL import Image
import rasterio
from rasterio.control import GroundControlPoint
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
            print(f"\n=== Adversarial Differential Oracle ({mode_str}): {len(cases)} checks ===")
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

            if gpu_diffs:
                print("\nCPU-vs-GPU differences:")
                for gd in gpu_diffs:
                    print(f"  GPU_DIFF | {gd['label']:54s} | {gd['detail']}")

            if args.json_out:
                Path(args.json_out).write_text(json.dumps({
                    "mode": mode_str,
                    "total": len(cases),
                    "failed": len(failures),
                    "gpu_diffs": gpu_diffs,
                    "failures": failures,
                }, indent=2))
            print(f"\nSummary ({mode_str}): {len(cases) - len(failures)}/{len(cases)} passed, {len(failures)} failed" + (f", {len(gpu_diffs)} CPU-vs-GPU diffs" if args.check_cpu_vs_gpu else ""))
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
