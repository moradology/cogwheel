#!/usr/bin/env python3
"""Generate deterministic Zarr v2/v3 correctness fixtures (and a GeoTIFF twin) for workspace tests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numcodecs
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import zarr
from zarr.codecs import (
    BloscCodec,
    BytesCodec,
    GzipCodec,
    TransposeCodec,
    ZstdCodec,
)

DEFAULT_OUT_DIR = Path(__file__).parent / "fixtures" / "zarr"

# San Francisco Bay Area in Web Mercator (same region as benches/generate_fixtures.py)
MERC_BOUNDS = (-13650000.0, 4500000.0, -13550000.0, 4600000.0)
WGS84_BOUNDS = (-123.0, 37.0, -121.5, 38.5)
UTM10N_BOUNDS = (500000.0, 4150000.0, 561440.0, 4211440.0)


def make_field(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Deterministic smooth multi-frequency float32 field with fine texture."""
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


def cell_centers(bounds: tuple[float, float, float, float], h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, x) 1-D float64 cell-center coordinate arrays matching rasterio.from_bounds (descending y)."""
    minx, miny, maxx, maxy = bounds
    dx = (maxx - minx) / w
    dy = (maxy - miny) / h
    x = minx + (np.arange(w, dtype=np.float64) + 0.5) * dx
    y = maxy - (np.arange(h, dtype=np.float64) + 0.5) * dy
    return y, x


def clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_v2_coords(
    root: zarr.Group,
    y: np.ndarray,
    x: np.ndarray,
    *,
    y_attrs: dict | None = None,
    x_attrs: dict | None = None,
) -> None:
    arr_y = root.create_array("y", shape=y.shape, chunks=y.shape, dtype="<f8", compressors=None)
    arr_y[:] = y
    arr_y.attrs["_ARRAY_DIMENSIONS"] = ["y"]
    if y_attrs:
        arr_y.attrs.update(y_attrs)

    arr_x = root.create_array("x", shape=x.shape, chunks=x.shape, dtype="<f8", compressors=None)
    arr_x[:] = x
    arr_x.attrs["_ARRAY_DIMENSIONS"] = ["x"]
    if x_attrs:
        arr_x.attrs.update(x_attrs)


def add_v3_coords(
    root: zarr.Group,
    y: np.ndarray,
    x: np.ndarray,
    *,
    y_attrs: dict | None = None,
    x_attrs: dict | None = None,
) -> None:
    arr_y = root.create_array(
        "y",
        shape=y.shape,
        chunks=y.shape,
        dtype="float64",
        serializer=BytesCodec(endian="little"),
        compressors=None,
        dimension_names=["y"],
    )
    arr_y[:] = y
    if y_attrs:
        arr_y.attrs.update(y_attrs)

    arr_x = root.create_array(
        "x",
        shape=x.shape,
        chunks=x.shape,
        dtype="float64",
        serializer=BytesCodec(endian="little"),
        compressors=None,
        dimension_names=["x"],
    )
    arr_x[:] = x
    if x_attrs:
        arr_x.attrs.update(x_attrs)


def gen_v2_basic(out_dir: Path, data: np.ndarray) -> None:
    """v2 stores with and without .zmetadata consolidated metadata."""
    y, x = cell_centers(WGS84_BOUNDS, *data.shape)
    for name, consolidate in [("v2_consolidated.zarr", True), ("v2_unconsolidated.zarr", False)]:
        path = clean_dir(out_dir / name)
        root = zarr.open_group(path, mode="w", zarr_format=2)
        root.attrs["title"] = name
        add_v2_coords(root, y, x)
        arr = root.create_array(
            "temp",
            shape=data.shape,
            chunks=(64, 64),
            dtype="<f4",
            fill_value=float("nan"),
            compressors=numcodecs.Zstd(level=3),
        )
        arr[:] = data
        arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
        arr.attrs["units"] = "K"
        if consolidate:
            zarr.consolidate_metadata(path)


def gen_v3_basic(out_dir: Path, data: np.ndarray) -> None:
    """v3 stores with and without consolidated metadata."""
    y, x = cell_centers(WGS84_BOUNDS, *data.shape)
    for name, consolidate in [("v3_consolidated.zarr", True), ("v3_unconsolidated.zarr", False)]:
        path = clean_dir(out_dir / name)
        root = zarr.open_group(path, mode="w", zarr_format=3)
        root.attrs["title"] = name
        add_v3_coords(root, y, x)
        arr = root.create_array(
            "temp",
            shape=data.shape,
            chunks=(64, 64),
            dtype="float32",
            fill_value=float("nan"),
            serializer=BytesCodec(endian="little"),
            compressors=ZstdCodec(level=3),
            dimension_names=["y", "x"],
        )
        arr[:] = data
        arr.attrs["units"] = "K"
        if consolidate:
            zarr.consolidate_metadata(path)


def gen_v3_sharding_pair(out_dir: Path, data: np.ndarray) -> None:
    """v3 sharded and unsharded stores with identical samples."""
    y, x = cell_centers(WGS84_BOUNDS, *data.shape)

    unsharded_path = clean_dir(out_dir / "v3_unsharded.zarr")
    root_u = zarr.open_group(unsharded_path, mode="w", zarr_format=3)
    add_v3_coords(root_u, y, x)
    arr_u = root_u.create_array(
        "data",
        shape=data.shape,
        chunks=(32, 32),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=3),
        dimension_names=["y", "x"],
    )
    arr_u[:] = data
    zarr.consolidate_metadata(unsharded_path)

    sharded_path = clean_dir(out_dir / "v3_sharded.zarr")
    root_s = zarr.open_group(sharded_path, mode="w", zarr_format=3)
    add_v3_coords(root_s, y, x)
    arr_s = root_s.create_array(
        "data",
        shape=data.shape,
        shards=(64, 64),
        chunks=(32, 32),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=3),
        dimension_names=["y", "x"],
    )
    arr_s[:] = data
    zarr.consolidate_metadata(sharded_path)


def gen_codecs(out_dir: Path, data: np.ndarray) -> None:
    """v2 and v3 codec matrices: blosc-zstd/lz4/blosclz, zstd, gzip, uncompressed, shuffle/bitshuffle, transpose."""
    # v2 codec matrix
    v2_path = clean_dir(out_dir / "codecs_v2.zarr")
    root2 = zarr.open_group(v2_path, mode="w", zarr_format=2)
    v2_cases = [
        ("uncompressed", None, None),
        ("gzip", None, numcodecs.GZip(level=5)),
        ("zstd", None, numcodecs.Zstd(level=3)),
        ("blosc_zstd_shuffle", None, numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)),
        ("blosc_zstd_bitshuffle", None, numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)),
        ("blosc_lz4_shuffle", None, numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)),
        ("blosc_lz4_noshuffle", None, numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE)),
        ("blosc_blosclz_shuffle", None, numcodecs.Blosc(cname="blosclz", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)),
        ("shuffle_zstd", [numcodecs.Shuffle(elementsize=4)], numcodecs.Zstd(level=3)),
    ]
    for name, filters, compressor in v2_cases:
        arr = root2.create_array(
            name,
            shape=data.shape,
            chunks=(64, 64),
            dtype="<f4",
            fill_value=float("nan"),
            filters=filters,
            compressors=compressor,
        )
        arr[:] = data
        arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    zarr.consolidate_metadata(v2_path)

    # v3 codec matrix
    v3_path = clean_dir(out_dir / "codecs_v3.zarr")
    root3 = zarr.open_group(v3_path, mode="w", zarr_format=3)
    v3_cases = [
        ("uncompressed", None, None),
        ("gzip", None, GzipCodec(level=5)),
        ("zstd", None, ZstdCodec(level=3)),
        ("blosc_zstd_shuffle", None, BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")),
        ("blosc_zstd_bitshuffle", None, BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")),
        ("blosc_lz4_shuffle", None, BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")),
        ("blosc_lz4_noshuffle", None, BloscCodec(cname="lz4", clevel=5, shuffle="noshuffle")),
        ("blosc_blosclz_shuffle", None, BloscCodec(cname="blosclz", clevel=5, shuffle="shuffle")),
        ("transpose_f_order", [TransposeCodec(order=(1, 0))], ZstdCodec(level=3)),
    ]
    for name, filters, compressor in v3_cases:
        arr = root3.create_array(
            name,
            shape=data.shape,
            chunks=(64, 64),
            dtype="float32",
            fill_value=float("nan"),
            filters=filters,
            serializer=BytesCodec(endian="little"),
            compressors=compressor,
            dimension_names=["y", "x"],
        )
        arr[:] = data
    zarr.consolidate_metadata(v3_path)


def gen_big_endian(out_dir: Path, data: np.ndarray) -> None:
    """Big-endian v3 (bytes codec) and v2 (>f4) stores."""
    y, x = cell_centers(WGS84_BOUNDS, *data.shape)
    data_i16 = np.round(data).astype(np.int16)

    p3 = clean_dir(out_dir / "big_endian_v3.zarr")
    root3 = zarr.open_group(p3, mode="w", zarr_format=3)
    add_v3_coords(root3, y, x)
    a3_f = root3.create_array(
        "f32_be",
        shape=data.shape,
        chunks=(64, 64),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="big"),
        compressors=None,
        dimension_names=["y", "x"],
    )
    a3_f[:] = data
    a3_i = root3.create_array(
        "i16_be",
        shape=data_i16.shape,
        chunks=(64, 64),
        dtype="int16",
        fill_value=-9999,
        serializer=BytesCodec(endian="big"),
        compressors=ZstdCodec(level=3),
        dimension_names=["y", "x"],
    )
    a3_i[:] = data_i16
    zarr.consolidate_metadata(p3)

    p2 = clean_dir(out_dir / "big_endian_v2.zarr")
    root2 = zarr.open_group(p2, mode="w", zarr_format=2)
    add_v2_coords(root2, y, x)
    a2_f = root2.create_array(
        "f32_be",
        shape=data.shape,
        chunks=(64, 64),
        dtype=">f4",
        fill_value=float("nan"),
        compressors=None,
    )
    a2_f[:] = data
    a2_f.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    zarr.consolidate_metadata(p2)


def gen_time_3d(out_dir: Path) -> None:
    """3-D store with CF time dimension and non-standard 'noleap' calendar."""
    nt, h, w = 6, 64, 64
    y, x = cell_centers(WGS84_BOUNDS, h, w)
    cube = np.stack([make_field(h, w, seed=500 + t) + t * 10.0 for t in range(nt)], axis=0)
    time_vals = np.array([0, 31, 59, 90, 120, 151], dtype=np.int64)

    path = clean_dir(out_dir / "time_3d_noleap.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=3)
    add_v3_coords(root, y, x)
    t_arr = root.create_array(
        "time",
        shape=time_vals.shape,
        chunks=time_vals.shape,
        dtype="int64",
        serializer=BytesCodec(endian="little"),
        compressors=None,
        dimension_names=["time"],
    )
    t_arr[:] = time_vals
    t_arr.attrs["units"] = "days since 2000-01-01"
    t_arr.attrs["calendar"] = "noleap"
    t_arr.attrs["standard_name"] = "time"

    var = root.create_array(
        "tas",
        shape=cube.shape,
        chunks=(2, 32, 32),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=3),
        dimension_names=["time", "y", "x"],
    )
    var[:] = cube
    var.attrs["units"] = "K"
    zarr.consolidate_metadata(path)


def gen_sparse_holes(out_dir: Path) -> None:
    """Sparse 3x3 chunk grid where 2/3 of chunks (6 of 9) are never written."""
    h, w = 192, 192
    y, x = cell_centers(WGS84_BOUNDS, h, w)
    path = clean_dir(out_dir / "sparse_holes.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=3)
    add_v3_coords(root, y, x)
    arr = root.create_array(
        "sparse",
        shape=(h, w),
        chunks=(64, 64),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=3),
        dimension_names=["y", "x"],
    )
    # Write only the 3 diagonal chunks; the remaining 6 chunks are never created on disk.
    for idx in range(3):
        r0, r1 = idx * 64, (idx + 1) * 64
        arr[r0:r1, r0:r1] = make_field(64, 64, seed=600 + idx)
    zarr.consolidate_metadata(path)

    # Sanity check: exactly 3 chunk files exist under sparse/c/
    chunk_files = [p for p in (path / "sparse" / "c").rglob("*") if p.is_file()]
    assert len(chunk_files) == 3, f"expected 3 written chunks, found {len(chunk_files)}"


def gen_lon_0_360_desc_y(out_dir: Path) -> None:
    """0..360 longitude axis with descending Y axis."""
    h, w = 36, 72
    x = np.arange(w, dtype=np.float64) * 5.0 + 2.5  # 2.5 .. 357.5
    y = 87.5 - np.arange(h, dtype=np.float64) * 5.0  # +87.5 .. -87.5 (descending)
    data = make_field(h, w, seed=700)

    path = clean_dir(out_dir / "lon_0_360_desc_y.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=3)
    add_v3_coords(
        root,
        y,
        x,
        y_attrs={"standard_name": "latitude", "units": "degrees_north"},
        x_attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    arr = root.create_array(
        "sst",
        shape=data.shape,
        chunks=(36, 36),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=3),
        dimension_names=["y", "x"],
    )
    arr[:] = data
    zarr.consolidate_metadata(path)


def gen_irregular_coords(out_dir: Path) -> None:
    """Irregular coordinate spacing that the affine check must refuse."""
    h, w = 32, 32
    y = np.linspace(38.5, 37.0, h, dtype=np.float64)
    # Non-uniform spacing along x (quadratically warped)
    t = np.linspace(0.0, 1.0, w, dtype=np.float64)
    x = -123.0 + 1.5 * (0.5 * t + 0.5 * t * t)
    data = make_field(h, w, seed=800)

    path = clean_dir(out_dir / "irregular_coords.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=3)
    add_v3_coords(root, y, x)
    arr = root.create_array(
        "warped",
        shape=data.shape,
        chunks=(32, 32),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=None,
        dimension_names=["y", "x"],
    )
    arr[:] = data
    zarr.consolidate_metadata(path)


def gen_crs_variants(out_dir: Path) -> None:
    """CF grid_mapping projected store, spatial_ref WKT store, and no-CRS store."""
    h, w = 128, 128
    data = make_field(h, w, seed=900)

    # 1. CF grid_mapping projected store (UTM 10N / EPSG:32610)
    p_cf = clean_dir(out_dir / "crs_cf_grid_mapping.zarr")
    root_cf = zarr.open_group(p_cf, mode="w", zarr_format=3)
    y_utm, x_utm = cell_centers(UTM10N_BOUNDS, h, w)
    add_v3_coords(
        root_cf,
        y_utm,
        x_utm,
        y_attrs={"standard_name": "projection_y_coordinate", "units": "m"},
        x_attrs={"standard_name": "projection_x_coordinate", "units": "m"},
    )
    gm = root_cf.create_array(
        "transverse_mercator",
        shape=(),
        chunks=(),
        dtype="int32",
        serializer=BytesCodec(endian="little"),
        compressors=None,
    )
    gm[()] = 0
    gm.attrs.update(
        {
            "grid_mapping_name": "transverse_mercator",
            "latitude_of_projection_origin": 0.0,
            "longitude_of_central_meridian": -123.0,
            "scale_factor_at_central_meridian": 0.9996,
            "false_easting": 500000.0,
            "false_northing": 0.0,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        }
    )
    arr_cf = root_cf.create_array(
        "band",
        shape=data.shape,
        chunks=(64, 64),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=3),
        dimension_names=["y", "x"],
    )
    arr_cf[:] = data
    arr_cf.attrs["grid_mapping"] = "transverse_mercator"
    zarr.consolidate_metadata(p_cf)

    # 2. spatial_ref WKT store (EPSG:3857)
    p_wkt = clean_dir(out_dir / "crs_spatial_ref_wkt.zarr")
    root_wkt = zarr.open_group(p_wkt, mode="w", zarr_format=3)
    y_m, x_m = cell_centers(MERC_BOUNDS, h, w)
    add_v3_coords(root_wkt, y_m, x_m)
    sr = root_wkt.create_array(
        "spatial_ref",
        shape=(),
        chunks=(),
        dtype="int64",
        serializer=BytesCodec(endian="little"),
        compressors=None,
    )
    sr[()] = 0
    wkt_3857 = CRS.from_epsg(3857).to_wkt()
    sr.attrs.update({"crs_wkt": wkt_3857, "spatial_ref": wkt_3857})
    arr_wkt = root_wkt.create_array(
        "band",
        shape=data.shape,
        chunks=(64, 64),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=3),
        dimension_names=["y", "x"],
    )
    arr_wkt[:] = data
    arr_wkt.attrs["grid_mapping"] = "spatial_ref"
    zarr.consolidate_metadata(p_wkt)

    # 3. No CRS at all (defaults to EPSG:4326)
    p_none = clean_dir(out_dir / "crs_none.zarr")
    root_none = zarr.open_group(p_none, mode="w", zarr_format=3)
    y_g, x_g = cell_centers(WGS84_BOUNDS, h, w)
    add_v3_coords(root_none, y_g, x_g)
    arr_none = root_none.create_array(
        "band",
        shape=data.shape,
        chunks=(64, 64),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=3),
        dimension_names=["y", "x"],
    )
    arr_none[:] = data
    zarr.consolidate_metadata(p_none)


def gen_multiscale(out_dir: Path) -> None:
    """Small 2-level GeoZarr/NGFF-style multiscale store (256x256 and 128x128)."""
    data0 = make_field(256, 256, seed=1000)
    data1 = data0.reshape(128, 2, 128, 2).mean(axis=(1, 3)).astype(np.float32)

    path = clean_dir(out_dir / "multiscale.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "elevation",
            "axes": [
                {"name": "y", "type": "space", "unit": "meter"},
                {"name": "x", "type": "space", "unit": "meter"},
            ],
            "datasets": [
                {"path": "0"},
                {"path": "1"},
            ],
        }
    ]

    wkt_3857 = CRS.from_epsg(3857).to_wkt()
    for level_name, arr_data in [("0", data0), ("1", data1)]:
        grp = root.create_group(level_name)
        y, x = cell_centers(MERC_BOUNDS, *arr_data.shape)
        add_v3_coords(grp, y, x)
        sr = grp.create_array(
            "spatial_ref",
            shape=(),
            chunks=(),
            dtype="int64",
            serializer=BytesCodec(endian="little"),
            compressors=None,
        )
        sr[()] = 0
        sr.attrs["spatial_ref"] = wkt_3857
        a = grp.create_array(
            "elevation",
            shape=arr_data.shape,
            chunks=(64, 64),
            dtype="float32",
            fill_value=float("nan"),
            serializer=BytesCodec(endian="little"),
            compressors=ZstdCodec(level=3),
            dimension_names=["y", "x"],
        )
        a[:] = arr_data
        a.attrs["grid_mapping"] = "spatial_ref"
    zarr.consolidate_metadata(path)


def gen_geotiff_twin(out_dir: Path) -> None:
    """512x512 EPSG:3857 Zarr store + GeoTIFF twin from identical samples on the exact same grid."""
    h, w = 512, 512
    data = make_field(h, w, seed=1100)
    # Put a deterministic NaN patch in one corner to exercise nodata parity too.
    data[:16, :16] = np.nan

    # Zarr twin
    zarr_path = clean_dir(out_dir / "geotiff_twin.zarr")
    root = zarr.open_group(zarr_path, mode="w", zarr_format=3)
    y, x = cell_centers(MERC_BOUNDS, h, w)
    add_v3_coords(root, y, x)
    sr = root.create_array(
        "spatial_ref",
        shape=(),
        chunks=(),
        dtype="int64",
        serializer=BytesCodec(endian="little"),
        compressors=None,
    )
    sr[()] = 0
    wkt_3857 = CRS.from_epsg(3857).to_wkt()
    sr.attrs.update({"crs_wkt": wkt_3857, "spatial_ref": wkt_3857})
    arr = root.create_array(
        "data",
        shape=data.shape,
        chunks=(256, 256),
        dtype="float32",
        fill_value=float("nan"),
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=3),
        dimension_names=["y", "x"],
    )
    arr[:] = data
    arr.attrs["grid_mapping"] = "spatial_ref"
    zarr.consolidate_metadata(zarr_path)

    # GeoTIFF twin (same 512x512 grid, same 256x256 tile size, no overviews so native grid matches)
    tif_path = out_dir / "geotiff_twin.tif"
    if tif_path.exists():
        tif_path.unlink()
    transform = from_bounds(*MERC_BOUNDS, w, h)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(3857),
        transform=transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="ZSTD",
        nodata=float("nan"),
    ) as dst:
        dst.write(data, 1)


def canonicalize_outputs(out_dir: Path) -> None:
    """Sort JSON metadata keys and zero RFC 1952 gzip MTIME headers for byte-identical determinism."""
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name in {"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"} or p.suffix == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            p.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        else:
            raw = p.read_bytes()
            if len(raw) >= 10 and raw[0:2] == b"\x1f\x8b":
                # RFC 1952 §2.3.1: MTIME == 0 means no timestamp is available.
                p.write_bytes(raw[:4] + b"\x00\x00\x00\x00" + raw[8:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_128 = make_field(128, 128, seed=101)
    gen_v2_basic(out_dir, base_128)
    gen_v3_basic(out_dir, base_128)
    gen_v3_sharding_pair(out_dir, base_128)
    gen_codecs(out_dir, base_128)
    gen_big_endian(out_dir, base_128)
    gen_time_3d(out_dir)
    gen_sparse_holes(out_dir)
    gen_lon_0_360_desc_y(out_dir)
    gen_irregular_coords(out_dir)
    gen_crs_variants(out_dir)
    gen_multiscale(out_dir)
    gen_geotiff_twin(out_dir)
    canonicalize_outputs(out_dir)

    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"Generated Zarr fixtures in {out_dir} ({total_bytes / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
