#!/usr/bin/env python3
"""Unit tests for benches/oracle_lib.py (#505)."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle_lib as olib

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
except ImportError:
    rasterio = None


class OracleLibTests(unittest.TestCase):
    def test_compare_json_normalizes_base_and_volatile(self):
        a = {"url": "http://127.0.0.1:18823/searches/abc", "lastused": "2026-01-01", "v": 1.0000000001}
        b = {"url": "http://127.0.0.1:18821/searches/abc", "usecount": 42, "v": 1.0}
        ok, _ = olib.compare_json(a, b, base_a="http://127.0.0.1:18823", base_b="http://127.0.0.1:18821")
        self.assertTrue(ok)

        c = {"url": "http://127.0.0.1:18821/searches/xyz", "v": 1.0}
        ok2, note2 = olib.compare_json(a, c, base_a="http://127.0.0.1:18823", base_b="http://127.0.0.1:18821")
        self.assertFalse(ok2)
        self.assertIn("$.url", note2)

    def test_same_place(self):
        ra = SimpleNamespace(headers={"content-crs": "<http://www.opengis.net/def/crs/EPSG/0/4326>", "content-bbox": "0.0,1.0,2.0,3.0"})
        rb = SimpleNamespace(headers={"content-crs": "<http://www.opengis.net/def/crs/EPSG/0/4326>", "content-bbox": "0.0,1.0,2.0000001,3.0"})
        ok, _ = olib.same_place(ra, rb)
        self.assertTrue(ok)

        rc = SimpleNamespace(headers={"content-crs": "<http://www.opengis.net/def/crs/EPSG/0/3857>", "content-bbox": "0.0,1.0,2.0,3.0"})
        ok2, note2 = olib.same_place(ra, rc)
        self.assertFalse(ok2)
        self.assertIn("content-crs", note2)

    def test_stats_and_zonal_diff(self):
        a = {"b1": {"mean": 100.0, "std": 10.0, "count": 500, "histogram": [[1, 2], [0, 1, 2]]}}
        b = {"b1": {"mean": 102.0, "std": 10.1, "count": 999, "histogram": [[9, 9], [0, 1, 2]]}}
        self.assertIsNone(olib.stats_diff(a, b))
        self.assertIsNotNone(olib.zonal_diff(a, b))

        za = {"properties": {"statistics": {"b1": {"mean": 42.0, "count": 100.0}}}}
        zb = {"properties": {"statistics": {"b1": {"mean": 42.0 + 1e-7, "count": 100.0}}}}
        self.assertIsNone(olib.zonal_diff(za, zb))

    def test_compare_wmts_xml_both_modes(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <Capabilities xmlns="http://www.opengis.net/wmts/1.0" xmlns:ows="http://www.opengis.net/ows/1.1">
          <Contents>
            <Layer>
              <ows:Title>demo</ows:Title>
              <ResourceURL template="http://HOST/tiles/{z}/{x}/{y}.png"/>
              <TileMatrixSetLink><TileMatrixSet>WebMercatorQuad</TileMatrixSet></TileMatrixSetLink>
            </Layer>
            <TileMatrixSet>
              <ows:Identifier>WebMercatorQuad</ows:Identifier>
              <TileMatrix><ows:Identifier>0</ows:Identifier></TileMatrix>
            </TileMatrixSet>
          </Contents>
        </Capabilities>"""
        a = xml.replace("HOST", "127.0.0.1:18823").encode()
        b = xml.replace("HOST", "127.0.0.1:18821").encode()
        ok_summary, _ = olib.compare_wmts_xml(a, b)
        self.assertTrue(ok_summary)
        ok_deep, _ = olib.compare_wmts_xml(a, b, base_a="http://127.0.0.1:18823", base_b="http://127.0.0.1:18821")
        self.assertTrue(ok_deep)

    @unittest.skipIf(np is None, "numpy not installed")
    def test_compare_arrays_and_response_dispatch(self):
        arr = np.zeros((2, 8, 8), dtype=np.uint8)
        arr[0] = 42
        arr[1] = 255
        buf = io.BytesIO()
        np.save(buf, arr)
        raw = buf.getvalue()
        ok, _ = olib.compare_response("application/x-binary", raw, raw)
        self.assertTrue(ok)

        arr2 = arr.copy()
        arr2[0] = 99
        buf2 = io.BytesIO()
        np.save(buf2, arr2)
        ok2, _ = olib.compare_response("application/x-binary", raw, buf2.getvalue())
        self.assertFalse(ok2)

    @unittest.skipIf(np is None or Image is None, "numpy or PIL not installed")
    def test_compare_images_and_response_dispatch(self):
        img = Image.fromarray(np.full((16, 16, 4), [10, 20, 30, 255], dtype=np.uint8), mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        ok, note = olib.compare_response("image/png", raw, raw)
        self.assertTrue(ok)
        self.assertIn("PSNR=99.0dB", note)

    @unittest.skipIf(np is None or rasterio is None, "numpy or rasterio not installed")
    def test_compare_geotiffs_both_modes(self):
        arr = np.zeros((2, 16, 16), dtype=np.uint8)
        arr[0] = 77
        arr[1] = 255
        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff",
                height=16,
                width=16,
                count=2,
                dtype="uint8",
                crs=CRS.from_epsg(3857),
                transform=from_bounds(0, 0, 16, 16, 16, 16),
            ) as dst:
                dst.write(arr)
            tif = mem.read()
        ok_meta, _ = olib.compare_geotiffs(tif, tif, compare_pixels=False)
        self.assertTrue(ok_meta)
        ok_px, _ = olib.compare_geotiffs(tif, tif, compare_pixels=True)
        self.assertTrue(ok_px)
        ok_disp, _ = olib.compare_response("image/tiff; application=geotiff", tif, tif)
        self.assertTrue(ok_disp)

    def test_compare_response_json_dispatch(self):
        a = json.dumps({"x": 1, "lastused": "t1"}).encode()
        b = json.dumps({"x": 1, "lastused": "t2"}).encode()
        ok, _ = olib.compare_response("application/json", a, b)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
