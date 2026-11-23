#!/usr/bin/env python3
"""Generate realistic USGS Landsat Collection 2 (uint16, 3 separate band COGs)
and NASA/ASF OPERA RTC-S1 SAR (float32 linear power, separate VV & VH COGs) fixtures."""

from pathlib import Path
import numpy as np
import rasterio
import rasterio.shutil as rio_shutil
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_field(h: int, w: int, seed: int) -> np.ndarray:
    y = np.linspace(0.0, 5.0 * np.pi, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 5.0 * np.pi, w, dtype=np.float32)[None, :]
    base = (
        0.5
        + 0.25 * np.sin(x * 0.8) * np.cos(y * 0.6)
        + 0.12 * np.sin(x * 2.7 + y * 1.9)
        + 0.05 * np.cos(x * 8.3 - y * 6.1)
    )
    rng = np.random.default_rng(seed)
    return np.clip(base + rng.normal(0.0, 0.02, size=(h, w)).astype(np.float32), 0.01, 0.99)


def write_single_band_cog(
    path: Path,
    arr: np.ndarray,
    dtype: str,
    crs: CRS,
    transform,
    compress: str,
    predictor: int,
    nodata,
    overviews: list[int],
):
    h, w = arr.shape
    tmp = path.with_suffix(".tmp.tif")
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": compress,
        "predictor": predictor,
        "nodata": nodata,
    }
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.build_overviews(overviews, Resampling.average)

    if path.exists():
        path.unlink()
    rio_shutil.copy(
        tmp,
        path,
        driver="GTiff",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress=compress,
        predictor=predictor,
        copy_src_overviews=True,
    )
    tmp.unlink()
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.2f} MB) [{dtype} {compress} pred={predictor}]")


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    h, w = 2048, 2048
    # UTM 10N over SF Bay (30m Landsat / OPERA RTC-S1 grid: 61.44 km x 61.44 km)
    crs = CRS.from_epsg(32610)
    transform = from_bounds(500000.0, 4150000.0, 561440.0, 4211440.0, w, h)

    # Tilted orbital swath mask (~25% nodata triangles at corners)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx - w * 0.5) * np.cos(0.21) + (yy - h * 0.5) * np.sin(0.21)
    v = -(xx - w * 0.5) * np.sin(0.21) + (yy - h * 0.5) * np.cos(0.21)
    swath = (np.abs(u) <= w * 0.42) & (np.abs(v) <= h * 0.46)

    # 1. USGS Landsat Collection 2 Level-2: 3 separate uint16 DEFLATE+Pred2 COGs (nodata=0)
    # Typical L2 Surface Reflectance DN range: 7273 (0.0) .. ~25000 (~0.48 reflectance)
    for band_name, seed in [("B4", 501), ("B3", 502), ("B2", 503)]:
        f = make_field(h, w, seed)
        dn = np.clip((7500.0 + f * 15000.0), 1.0, 65535.0).astype(np.uint16)
        dn[~swath] = 0
        write_single_band_cog(
            FIXTURES_DIR / f"landsat_c2_{band_name}.tif",
            dn,
            dtype="uint16",
            crs=crs,
            transform=transform,
            compress="DEFLATE",
            predictor=2,
            nodata=0,
            overviews=[2, 4, 8, 16],
        )

    # 2. NASA/ASF OPERA RTC-S1: separate float32 linear gamma-naught power VV & VH COGs
    # Typical VV linear power: 0.01 (-20 dB) .. 0.8 (-1 dB); VH: 0.003 (-25 dB) .. 0.25 (-6 dB)
    vv_db = -18.0 + make_field(h, w, 601) * 16.0
    vh_db = -24.0 + make_field(h, w, 602) * 15.0
    vv_lin = np.power(10.0, vv_db / 10.0).astype(np.float32)
    vh_lin = np.power(10.0, vh_db / 10.0).astype(np.float32)
    vv_lin[~swath] = np.nan
    vh_lin[~swath] = np.nan

    for pol, arr in [("VV", vv_lin), ("VH", vh_lin)]:
        write_single_band_cog(
            FIXTURES_DIR / f"opera_rtc_s1_{pol}.tif",
            arr,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress="DEFLATE",
            predictor=3,
            nodata=float("nan"),
            overviews=[2, 4, 8, 16],
        )

    print("Landsat C2 & OPERA RTC-S1 fixtures generated.")


if __name__ == "__main__":
    main()
