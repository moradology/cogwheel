#!/usr/bin/env python3
"""Zarr parity oracle harness (#377): verifies cogzilla /zarr/* against numpy/zarr-python
ground truth derived directly from the fixtures (never trusting either server)."""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_lib import compare_response, zonal_diff

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "zarr"


def req_raw(url: str, method: str = "GET", body: bytes | None = None) -> tuple[int, bytes, dict]:
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read(), dict(resp.headers.items())
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--only", default="", help="Substring filter for case names")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    passed = 0
    failed = 0

    def run_case(name: str, fn):
        nonlocal passed, failed
        if args.only and args.only not in name:
            return
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    def u(rel: str) -> str:
        return urllib.parse.quote(str((FIXTURES_DIR / rel).resolve()), safe="")

    # 1. Numpy ground truth on geotiff_twin.zarr statistics
    def case_twin_stats_numpy():
        arr = np.asarray(zarr.open_group(FIXTURES_DIR / "geotiff_twin.zarr", mode="r")["data"][:])
        valid = arr[np.isfinite(arr)]
        st, body, _ = req_raw(f"{base}/zarr/statistics?url={u('geotiff_twin.zarr')}")
        assert st == 200, f"status={st} body={body}"
        b1 = json.loads(body)["b1"]
        assert int(b1["count"]) == valid.size, f"count {b1['count']} != {valid.size}"
        assert abs(b1["min"] - float(valid.min())) < 1e-4
        assert abs(b1["max"] - float(valid.max())) < 1e-4
        assert abs(b1["mean"] - float(valid.mean())) < 1e-3

    # 2. Byte-identical + decoded PNG tile parity between /zarr/tiles and /cog/tiles on geotiff_twin
    def case_twin_tile_byte_identical():
        z_st, z_png, z_hdrs = req_raw(
            f"{base}/zarr/tiles/WebMercatorQuad/10/163/395.png?url={u('geotiff_twin.zarr')}&rescale=100,900&colormap_name=viridis&resampling=bilinear"
        )
        c_st, c_png, _ = req_raw(
            f"{base}/cog/tiles/WebMercatorQuad/10/163/395.png?url={u('geotiff_twin.tif')}&rescale=100,900&colormap_name=viridis&resampling=bilinear"
        )
        assert z_st == 200 and c_st == 200
        ok, note = compare_response(z_hdrs.get("content-type", "image/png"), z_png, c_png)
        assert ok, f"Zarr PNG tile diverged from COG twin: {note}"
        assert z_png == c_png, f"Zarr PNG tile bytes diverged from COG twin ({note})"

    # 3. Sharded vs unsharded tile parity (decoded array + byte-identical)
    def case_sharded_vs_unsharded_tile():
        s_st, s_npy, s_hdrs = req_raw(f"{base}/zarr/tiles/WebMercatorQuad/9/81/197.npy?url={u('v3_sharded.zarr')}")
        u_st, u_npy, _ = req_raw(f"{base}/zarr/tiles/WebMercatorQuad/9/81/197.npy?url={u('v3_unsharded.zarr')}")
        assert s_st == 200 and u_st == 200
        ok, note = compare_response(s_hdrs.get("content-type", "application/x-binary"), s_npy, u_npy)
        assert ok, f"v3 sharded NPY tile diverged from unsharded: {note}"
        assert s_npy == u_npy, f"v3 sharded NPY bytes diverged from unsharded ({note})"

    # 4. 3-D CF noleap time coordinates + sel round-trip + out-of-range 400
    def case_3d_noleap_sel():
        st, body, _ = req_raw(f"{base}/zarr/dataset/coordinates/time?url={u('time_3d_noleap.zarr')}&decode_times=true")
        assert st == 200
        times = json.loads(body)["data"]
        assert times[0] == "2000-01-01T00:00:00" and times[2] == "2000-03-01T00:00:00"

        # Verify numpy ground truth on slice 2 vs /zarr/statistics?sel=time=2000-03-01
        arr = np.asarray(zarr.open_group(FIXTURES_DIR / "time_3d_noleap.zarr", mode="r")["tas"][2])
        st2, body2, _ = req_raw(f"{base}/zarr/statistics?url={u('time_3d_noleap.zarr')}&sel=time=2000-03-01")
        assert st2 == 200
        b1 = json.loads(body2)["b1"]
        assert abs(b1["mean"] - float(np.nanmean(arr))) < 1e-3

        # Out-of-range date must return 400 with available range
        st_oor, body_oor, _ = req_raw(
            f"{base}/zarr/statistics?url={u('time_3d_noleap.zarr')}&sel=time=nearest::2099-01-01"
        )
        assert st_oor == 400, f"expected 400 for out-of-range sel, got {st_oor}"
        assert "2000-06-01" in body_oor.decode()

    # 5. Sparse holes (2/3 chunks missing)
    def case_sparse_holes():
        arr = np.asarray(zarr.open_group(FIXTURES_DIR / "sparse_holes.zarr", mode="r")["sparse"][:])
        valid = arr[np.isfinite(arr)]
        st, body, _ = req_raw(f"{base}/zarr/statistics?url={u('sparse_holes.zarr')}")
        assert st == 200
        b1 = json.loads(body)["b1"]
        assert int(b1["count"]) == valid.size
        assert abs(b1["mean"] - float(valid.mean())) < 1e-3

    # 6. CRS variants + 0..360 longitudes + irregular grid refusal
    def case_crs_and_refusals():
        st_cf, body_cf, _ = req_raw(f"{base}/zarr/info?url={u('crs_cf_grid_mapping.zarr')}")
        assert st_cf == 200 and "transverse_mercator" in json.loads(body_cf)["crs"]

        st_wkt, body_wkt, _ = req_raw(f"{base}/zarr/info?url={u('crs_spatial_ref_wkt.zarr')}")
        assert st_wkt == 200 and json.loads(body_wkt)["crs"] == "EPSG:3857"

        st_360, body_360, _ = req_raw(f"{base}/zarr/bounds?url={u('lon_0_360_desc_y.zarr')}")
        assert st_360 == 200 and json.loads(body_360)["bounds"] == [-180.0, -90.0, 180.0, 90.0]

        st_irr, body_irr, _ = req_raw(f"{base}/zarr/info?url={u('irregular_coords.zarr')}")
        assert st_irr == 400
        detail = json.loads(body_irr)["detail"]
        assert '"x"' in detail and "index 2" in detail

    # 7. Multiscale store serves low-zoom and high-zoom tiles
    def case_multiscale():
        st, body, _ = req_raw(f"{base}/zarr/tiles/WebMercatorQuad/9/81/197.png?url={u('multiscale.zarr')}&rescale=100,900")
        assert st == 200 and body[:4] == b"\x89PNG"

    # 8. POST /zarr/statistics exact polygon parity vs POST /cog/statistics
    def case_post_zonal_parity():
        poly = json.dumps({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [-122.4, 37.6], [-122.0, 37.6], [-122.0, 37.9], [-122.4, 37.9], [-122.4, 37.6]
            ]]},
            "properties": {},
        }).encode()
        z_st, z_b, _ = req_raw(f"{base}/zarr/statistics?url={u('geotiff_twin.zarr')}", method="POST", body=poly)
        c_st, c_b, _ = req_raw(f"{base}/cog/statistics?url={u('geotiff_twin.tif')}&max_size=0", method="POST", body=poly)
        assert z_st == 200 and c_st == 200
        zb1 = json.loads(z_b)["properties"]["statistics"]["b1"]
        cb1 = json.loads(c_b)["properties"]["statistics"]["b1"]
        # Validate exact shared summary fields via oracle_lib.zonal_diff (plus area tolerance).
        diff = zonal_diff(
            {k: zb1[k] for k in ("count", "min", "max", "mean")},
            {k: cb1[k] for k in ("count", "min", "max", "mean")},
        )
        assert diff is None, f"zonal stats diverged: {diff}"
        assert abs(zb1["area"] - cb1["area"]) < 1e-2

    run_case("zarr_twin_stats_numpy", case_twin_stats_numpy)
    run_case("zarr_twin_tile_byte_identical", case_twin_tile_byte_identical)
    run_case("zarr_sharded_vs_unsharded_tile", case_sharded_vs_unsharded_tile)
    run_case("zarr_3d_noleap_sel", case_3d_noleap_sel)
    run_case("zarr_sparse_holes", case_sparse_holes)
    run_case("zarr_crs_and_refusals", case_crs_and_refusals)
    run_case("zarr_multiscale", case_multiscale)
    run_case("zarr_post_zonal_parity", case_post_zonal_parity)

    print(f"\nZarr parity oracle: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
