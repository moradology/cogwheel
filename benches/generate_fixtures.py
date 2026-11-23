#!/usr/bin/env python3
"""Generate deterministic float32 COG fixtures for the COGMark benchmark suite."""

from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_synthetic_terrain(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Smooth multi-frequency terrain/reflectance field with deterministic fine texture."""
    y = np.linspace(0.0, 4.0 * np.pi, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 4.0 * np.pi, w, dtype=np.float32)[None, :]
    base = (
        500.0
        + 250.0 * np.sin(x * 0.7) * np.cos(y * 0.5)
        + 120.0 * np.sin(x * 2.3 + y * 1.7)
        + 40.0 * np.cos(x * 7.1 - y * 5.3)
    )
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 3.5, size=(h, w)).astype(np.float32)
    return (base + noise).astype(np.float32)


def write_cog(
    path: Path,
    data: np.ndarray,
    crs: CRS,
    transform,
    compress: str = "ZSTD",
    predictor: int = 3,
    blocksize: int = 512,
    nodata: float | None = float("nan"),
    interleave: str = "PIXEL",
    overviews: list[int] | None = None,
    sparse_ok: bool = False,
):
    if data.ndim == 2:
        data = data[None, :, :]
    count, height, width = data.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
        "interleave": interleave,
        "BIGTIFF": "IF_SAFER",
    }
    if compress and compress.upper() != "NONE":
        profile["compress"] = compress.upper()
        if predictor in (1, 2, 3):
            profile["predictor"] = predictor
    if nodata is not None:
        profile["nodata"] = nodata
    if sparse_ok:
        profile["SPARSE_OK"] = "TRUE"

    tmp_path = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
        if overviews:
            dst.build_overviews(overviews, Resampling.average)

    # Re-write via GDAL CreateCopy with COPY_SRC_OVERVIEWS=YES to enforce COG IFD layout
    import rasterio.shutil as rio_shutil
    copy_opts = {
        "driver": "GTiff",
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
        "interleave": interleave,
        "copy_src_overviews": True,
        "BIGTIFF": "IF_SAFER",
    }
    if compress and compress.upper() != "NONE":
        copy_opts["compress"] = compress.upper()
        if predictor in (1, 2, 3):
            copy_opts["predictor"] = predictor
    if sparse_ok:
        copy_opts["SPARSE_OK"] = "TRUE"
    if path.exists():
        path.unlink()
    rio_shutil.copy(tmp_path, path, **copy_opts)
    tmp_path.unlink()
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.2f} MB) [{compress} pred={predictor} {crs}]")


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. BM-02 codec matrix: 1024x1024 f32 single-band in EPSG:3857 (4x 512x512 blocks)
    # San Francisco Bay Area in Web Mercator
    merc_bounds = (-13650000.0, 4500000.0, -13550000.0, 4600000.0)
    data_1k = make_synthetic_terrain(1024, 1024, seed=101)
    t_1k = from_bounds(*merc_bounds, 1024, 1024)

    codec_cases = [
        ("codec_none_p1.tif", "NONE", 1),
        ("codec_deflate_p1.tif", "DEFLATE", 1),
        ("codec_deflate_p2.tif", "DEFLATE", 2),
        ("codec_deflate_p3.tif", "DEFLATE", 3),
        ("codec_zstd_p1.tif", "ZSTD", 1),
        ("codec_zstd_p3.tif", "ZSTD", 3),
        ("codec_lzw_p1.tif", "LZW", 1),
        ("codec_lzw_p3.tif", "LZW", 3),
    ]
    for name, comp, pred in codec_cases:
        write_cog(
            FIXTURES_DIR / name,
            data_1k,
            CRS.from_epsg(3857),
            t_1k,
            compress=comp,
            predictor=pred,
            blocksize=512,
            nodata=None,
            overviews=None,
        )

    # 2. BM-01 & BM-03 & BM-07 & BM-08: 4096x4096 EPSG:3857 ZSTD+Pred3 with 4 overviews
    # Covers z=10..14 cleanly over SF Bay
    data_4k = make_synthetic_terrain(4096, 4096, seed=202)
    t_4k = from_bounds(*merc_bounds, 4096, 4096)
    write_cog(
        FIXTURES_DIR / "mercator_4k_zstd_p3.tif",
        data_4k,
        CRS.from_epsg(3857),
        t_4k,
        compress="ZSTD",
        predictor=3,
        blocksize=512,
        nodata=float("nan"),
        overviews=[2, 4, 8, 16],
    )

    # 3. BM-04: 4096x4096 EPSG:4326 DEFLATE+Pred3 with overviews
    # Covers lon [-123.0, -121.5], lat [37.0, 38.5]
    t_4326 = from_bounds(-123.0, 37.0, -121.5, 38.5, 4096, 4096)
    write_cog(
        FIXTURES_DIR / "geo4326_4k_deflate_p3.tif",
        data_4k,
        CRS.from_epsg(4326),
        t_4326,
        compress="DEFLATE",
        predictor=3,
        blocksize=512,
        nodata=float("nan"),
        overviews=[2, 4, 8, 16],
    )

    # 4. BM-05a: Realistic Satellite Imagery 3-band RGB float32 reflectance in UTM 10N (EPSG:32610)
    # Tilted orbital swath (~25% NaN nodata triangles at corners, matching Sentinel-2 / Landsat)
    h_sat, w_sat = 2048, 2048
    yy, xx = np.mgrid[0:h_sat, 0:w_sat].astype(np.float32)
    # Rotate coordinates by ~12 degrees to simulate descending satellite orbit swath
    u = (xx - w_sat * 0.5) * np.cos(0.21) + (yy - h_sat * 0.5) * np.sin(0.21)
    v = -(xx - w_sat * 0.5) * np.sin(0.21) + (yy - h_sat * 0.5) * np.cos(0.21)
    swath_mask = (np.abs(u) <= w_sat * 0.42) & (np.abs(v) <= h_sat * 0.46)

    b1 = np.clip(0.05 + 0.25 * (make_synthetic_terrain(h_sat, w_sat, seed=301) / 1000.0), 0.0, 1.0)
    b2 = np.clip(0.04 + 0.22 * (make_synthetic_terrain(h_sat, w_sat, seed=302) / 1000.0), 0.0, 1.0)
    b3 = np.clip(0.03 + 0.20 * (make_synthetic_terrain(h_sat, w_sat, seed=303) / 1000.0), 0.0, 1.0)
    sat_rgb = np.stack([b1, b2, b3], axis=0).astype(np.float32)
    sat_rgb[:, ~swath_mask] = np.nan

    # UTM 10N covering SF Bay (~500000..561440 E, 4150000..4211440 N -> 30m Landsat resolution!)
    t_utm = from_bounds(500000.0, 4150000.0, 561440.0, 4211440.0, w_sat, h_sat)
    write_cog(
        FIXTURES_DIR / "satellite_utm10n_rgb_zstd_p3.tif",
        sat_rgb,
        CRS.from_epsg(32610),
        t_utm,
        compress="ZSTD",
        predictor=3,
        blocksize=512,
        nodata=float("nan"),
        interleave="PIXEL",
        overviews=[2, 4, 8],
    )
    # Also single-band LZW+Pred3 UTM 10N (e.g. Sentinel-2 Band 4 / DEM)
    write_cog(
        FIXTURES_DIR / "utm10n_2k_lzw_p3.tif",
        sat_rgb[0],
        CRS.from_epsg(32610),
        t_utm,
        compress="LZW",
        predictor=3,
        blocksize=512,
        nodata=float("nan"),
        overviews=[2, 4, 8],
    )

    # 5. BM-05b: NAD83 CONUS Albers (EPSG:5070) single-band f32
    t_albers = from_bounds(-2300000.0, 1800000.0, -2200000.0, 1900000.0, 2048, 2048)
    data_2k = make_synthetic_terrain(2048, 2048, seed=404)
    write_cog(
        FIXTURES_DIR / "albers5070_2k_zstd_p3.tif",
        data_2k,
        CRS.from_epsg(5070),
        t_albers,
        compress="ZSTD",
        predictor=3,
        blocksize=512,
        nodata=float("nan"),
        overviews=[2, 4, 8],
    )

    # 6. BM-06: Sparse coastline (50% all-NaN ocean blocks + 50% land blocks)
    coastal = data_4k.copy()
    coastal[:, :2048] = np.nan  # Western half is pure ocean NaN
    write_cog(
        FIXTURES_DIR / "coastal_sparse_4k_zstd_p3.tif",
        coastal,
        CRS.from_epsg(3857),
        t_4k,
        compress="ZSTD",
        predictor=3,
        blocksize=512,
        nodata=float("nan"),
        overviews=[2, 4, 8, 16],
        sparse_ok=True,
    )

    # Save manifest of fixtures & suggested test XYZ tiles
    manifest = {
        "fixtures_dir": str(FIXTURES_DIR.resolve()),
        "tiles": {
            "mercator_z12": {"z": 12, "x": 655, "y": 1583},
            "geo4326_z12": {"z": 12, "x": 655, "y": 1583},
            "utm10n_z12": {"z": 12, "x": 655, "y": 1583},
            "albers5070_z11": {"z": 11, "x": 327, "y": 791},
        },
    }
    (FIXTURES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("All fixtures generated successfully.")


if __name__ == "__main__":
    main()
