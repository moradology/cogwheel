#!/usr/bin/env python3
"""
pgSTAC differential oracle: cogzilla's `/searches` routes against stock titiler-pgstac.

Sends the same requests to both servers, backed by the same pgSTAC database, and
compares status codes, content types, JSON bodies, item lists and rendered tiles.
cogzilla is started by this script; titiler-pgstac and the database come from
benches/pgstac/docker-compose.yml.

A JSON check passes when the bodies are equal once each server's own base URL and the
bookkeeping fields (`lastused`, `usecount`) are taken out. An image check passes when
status, content type and size agree, at least 99% of pixels agree on being valid, and
PSNR over the shared valid pixels is at least 25 dB; the PSNR itself is reported so
closer agreement can be tracked.

Requirements:
  docker compose -f benches/pgstac/docker-compose.yml up -d
  benches/.venv/bin/python benches/pgstac_load.py
  cargo build --release -p cogzilla-server

Usage:
  benches/.venv/bin/python benches/pgstac_diff_oracle.py
  benches/.venv/bin/python benches/pgstac_diff_oracle.py --cogzilla-bin /path/to/cogzilla-server

Exits non-zero when any check fails.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import xml.etree.ElementTree as ET

import httpx
import psycopg
import mercantile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benches" / "results" / "pgstac"
LOADED = ROOT / "benches" / "fixtures" / "pgstac" / "loaded.json"
COGZILLA_BIN = ROOT / "target" / "release" / "cogzilla-server"

PORT_COGZILLA = 18823
TITILER = f"http://127.0.0.1:{os.environ.get('TITILER_PGSTAC_PORT', '18821')}"
COGZILLA = f"http://127.0.0.1:{PORT_COGZILLA}"
VOLATILE = {"lastused", "usecount"}


def dsn_from_env() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    user = os.environ.get("PGUSER", "username")
    password = os.environ.get("PGPASSWORD", "password")
    host = os.environ.get("PGHOST", "127.0.0.1")
    port = os.environ.get("PGPORT", os.environ.get("PGSTAC_PORT", "5439"))
    db = os.environ.get("PGDATABASE", "postgis")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def wait_ready(url: str, timeout_s: float = 20.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"{url} did not become ready in {timeout_s}s")


def normalize(value, base: str):
    """Strip what legitimately differs between the two servers."""
    if isinstance(value, dict):
        return {k: normalize(v, base) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [normalize(v, base) for v in value]
    if isinstance(value, str):
        return value.replace(base, "{base}")
    return value


def first_diff(a, b, path="$"):
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k}: missing in cogzilla"
            if k not in b:
                return f"{path}.{k}: only in cogzilla"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} vs {len(b)} entries"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, float) or isinstance(b, float):
        return None if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) else f"{path}: {a} vs {b}"
    return None if a == b else f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}"


def compare_images(cz: bytes, tt: bytes) -> tuple[bool, str]:
    try:
        a = np.array(Image.open(io.BytesIO(cz)).convert("RGBA"), dtype=np.float32)
        b = np.array(Image.open(io.BytesIO(tt)).convert("RGBA"), dtype=np.float32)
    except Exception as e:  # noqa: BLE001 - an undecodable image is a failed check
        return False, f"undecodable image: {e}"
    if a.shape != b.shape:
        return False, f"size {a.shape[1]}x{a.shape[0]} vs {b.shape[1]}x{b.shape[0]}"
    ma, mb = a[:, :, 3] > 0, b[:, :, 3] > 0
    mask = float((ma == mb).mean()) * 100
    both = ma & mb
    if both.any():
        d = a[:, :, :3][both] - b[:, :, :3][both]
        mse = float((d * d).mean())
        psnr = 99.0 if mse < 1e-9 else 20 * math.log10(255 / math.sqrt(mse))
    else:
        psnr = 99.0
    return mask >= 99.0 and psnr >= 25.0, f"mask {mask:.2f}%, PSNR {psnr:.1f} dB"


def wmts_view(text: str, base: str):
    """A WMTS capabilities document as plain lists, keeping only what concerns WebMercatorQuad:
    cogzilla serves these routes in that one tile matrix set, TiTiler lists thirteen."""
    wm = "{http://www.opengis.net/wmts/1.0}"
    ows = "{http://www.opengis.net/ows/1.1}"

    def plain(el):
        attrs = {k: v.replace(base, "{base}") for k, v in sorted(el.attrib.items())}
        text = (el.text or "").strip().replace(base, "{base}")
        if el.tag.endswith("Corner"):
            # Coordinates are compared as numbers: the two servers project with different code.
            text = [float(v) for v in text.split()]
        return [el.tag, attrs, text, [plain(c) for c in el]]

    root = ET.fromstring(text)
    contents = root.find(f"{wm}Contents")
    layers = [l for l in contents.findall(f"{wm}Layer")
              if l.find(f"{wm}TileMatrixSetLink/{wm}TileMatrixSet").text == "WebMercatorQuad"]
    matrix_sets = [t for t in contents.findall(f"{wm}TileMatrixSet")
                   if t.find(f"{ows}Identifier").text == "WebMercatorQuad"]
    rest = [plain(c) for c in root if c.tag != f"{wm}Contents"]
    return {"root": [root.tag, dict(sorted(root.attrib.items()))], "outside_contents": rest,
            "layers": [plain(l) for l in layers], "tile_matrix_sets": [plain(t) for t in matrix_sets]}


def stats_diff(a, b, path="$"):
    """Statistics come from differently resampled pixels: compare structure exactly and the
    summary numbers loosely (5% of the reference, or 2 units)."""
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            return f"{path}: keys differ: {sorted(set(a) ^ set(b)) if isinstance(a, dict) else type(a).__name__}"
        for k in b:
            if k in ("histogram", "majority", "minority", "unique", "count", "sum",
                     "masked_pixels", "valid_pixels", "valid_percent"):
                continue
            d = stats_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        if not isinstance(a, (int, float)) or abs(a - b) > max(2.0, 0.05 * abs(b)):
            return f"{path}: {a} vs {b}"
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def zonal_diff(a, b, path="$", key=""):
    """Mosaic statistics are read from the same pixels on both sides, so everything has to
    be equal. TiTiler sums coverage-weighted values in single precision: `mean`, `count`,
    `sum` and `std` are compared to one part in a million, other numbers to one in 1e9."""
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            return f"{path}: keys differ: {sorted(set(a) ^ set(b)) if isinstance(a, dict) else type(a).__name__}"
        for k in b:
            d = zonal_diff(a[k], b[k], f"{path}.{k}", k)
            if d:
                return d
        return None
    if isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            return f"{path}: lists differ"
        for i, (x, y) in enumerate(zip(a, b)):
            d = zonal_diff(x, y, f"{path}[{i}]", key)
            if d:
                return d
        return None
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        tol = 1e-6 if key in ("mean", "count", "sum", "std") else 1e-9
        if not isinstance(a, (int, float)) or abs(a - b) > tol * max(1.0, abs(b)):
            return f"{path}: {a!r} vs {b!r}"
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


class Oracle:
    def __init__(self, timeout: float):
        self.client = httpx.Client(timeout=timeout)
        self.rows: list[dict] = []

    def fetch(self, base: str, method: str, path: str, body):
        t0 = time.perf_counter()
        r = self.client.request(method, base + path, json=body)
        return r, (time.perf_counter() - t0) * 1e3

    def check(self, name: str, kind: str, path: str, method: str = "GET", body=None, known: str = ""):
        """Send one request to both servers. `known` names a difference that is understood and
        not fixed yet: the check is then reported as KNOWN instead of failing the run, and as
        a failure again once the two servers agree, so the note cannot go stale."""
        tt, tt_ms = self.fetch(TITILER, method, path, body)
        cz, cz_ms = self.fetch(COGZILLA, method, path, body)
        ok, note = True, ""
        if cz.status_code != tt.status_code:
            ok, note = False, f"status {cz.status_code} vs {tt.status_code}: {cz.text[:80]!r}"
        elif kind == "status":
            note = f"{cz.status_code}"
        elif kind == "json":
            try:
                diff = first_diff(normalize(cz.json(), COGZILLA), normalize(tt.json(), TITILER))
            except ValueError:
                diff = f"not JSON: {cz.text[:80]!r} vs {tt.text[:80]!r}"
            ok, note = diff is None, diff or f"{cz.status_code}, bodies equal"
        elif kind == "tilesets":
            # cogzilla serves these routes in Web Mercator only, so only that entry is compared.
            ours = normalize(cz.json(), COGZILLA)
            theirs = normalize(tt.json(), TITILER)
            theirs["tilesets"] = [t for t in theirs["tilesets"] if "WebMercatorQuad TileMatrixSet" in t["title"]]
            diff = first_diff(ours, theirs)
            ok, note = diff is None, diff or "WebMercatorQuad entry equal"
        elif kind == "wmts":
            if cz.status_code != 200:
                diff = first_diff(normalize(cz.json(), COGZILLA), normalize(tt.json(), TITILER))
                ok, note = diff is None, diff or f"{cz.status_code}, bodies equal"
            elif cz.headers.get("content-type") != tt.headers.get("content-type"):
                ok, note = False, f"content type {cz.headers.get('content-type')} vs {tt.headers.get('content-type')}"
            else:
                a, b = wmts_view(cz.text, COGZILLA), wmts_view(tt.text, TITILER)
                diff = first_diff(a, b)
                ok, note = diff is None, diff or f"{len(a['layers'])} layer(s), WebMercatorQuad parts equal"
        elif kind == "stats":
            diff = stats_diff(cz.json(), tt.json())
            ok, note = diff is None, diff or "same bands and fields, numbers within tolerance"
        elif kind == "zonal":
            if cz.status_code != 200:
                note = f"{cz.status_code}"
            elif cz.headers.get("content-type") != tt.headers.get("content-type"):
                ok, note = False, f"content type {cz.headers.get('content-type')} vs {tt.headers.get('content-type')}"
            else:
                diff = zonal_diff(cz.json(), tt.json())
                ok, note = diff is None, diff or "same features, items and statistics"
        elif kind == "ids":
            a, b = [i["id"] for i in cz.json()], [i["id"] for i in tt.json()]
            ok = a == b
            note = f"{len(a)} items, same order" if ok else f"item lists differ: {len(a)} vs {len(b)}"
        elif kind == "part":
            # The lean lookup's list: titiler-pgstac's, in its order, less what falls between
            # sample points. It may not be empty where theirs is not, nor hold anything theirs lacks.
            a, b = [i["id"] for i in cz.json()], [i["id"] for i in tt.json()]
            rest = iter(b)
            ok = all(i in rest for i in a) and bool(a) == bool(b)
            note = (f"{len(a)} of their {len(b)} items, same order" if ok
                    else f"not a part of their list: {len(a)} vs {len(b)}")
        elif kind == "image":
            ct_cz = cz.headers.get("content-type", "")
            ct_tt = tt.headers.get("content-type", "")
            if cz.status_code != 200:
                note = f"{cz.status_code}"
            elif ct_cz != ct_tt:
                ok, note = False, f"content type {ct_cz} vs {ct_tt}"
            else:
                ok, note = compare_images(cz.content, tt.content)
                note = f"{ct_cz}, {note}"
        status = "PASS" if ok else "FAIL"
        if known:
            status, note = ("KNOWN", f"{note} [{known}]") if not ok else ("FAIL", f"agrees now, drop the note [{known}]")
        self.rows.append({
            "name": name, "kind": kind, "method": method, "path": path, "ok": status == "PASS",
            "status": status, "note": note,
            "cogzilla_ms": round(cz_ms, 1), "titiler_ms": round(tt_ms, 1),
        })
        print(f"  {status}  {name}: {note}", flush=True)
        return cz, tt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cogzilla-bin", default=str(COGZILLA_BIN))
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    if not LOADED.exists():
        print("no loaded.json: run benches/pgstac_load.py first", file=sys.stderr)
        return 1
    loaded = json.loads(LOADED.read_text())
    collection = loaded["collection"]
    w, s, e, n = (float(v) for v in loaded["bbox"].split(","))
    lon, lat = (w + e) / 2, (s + n) / 2

    env = os.environ.copy()
    env["DATABASE_URL"] = dsn_from_env()
    proc = subprocess.Popen(
        [args.cogzilla_bin, "--bind", f"127.0.0.1:{PORT_COGZILLA}", "--cache-mb", "1024"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_ready(f"{COGZILLA}/healthz")
        wait_ready(f"{TITILER}/healthz", 60)
        o = Oracle(args.timeout)

        print("register")
        plain = {"collections": [collection]}
        cz, _ = o.check("register a collection search", "json", "/searches/register", "POST", plain)
        sid = cz.json().get("id", "missing")
        described = {
            "collections": [collection],
            "filter": {"op": "<", "args": [{"property": "eo:cloud_cover"}, 20]},
            "metadata": {"name": "low cloud", "minzoom": 8, "maxzoom": 14, "bounds": [w, s, e, n]},
        }
        cz, _ = o.check("register with filter and metadata", "json", "/searches/register", "POST", described)
        sid_meta = cz.json().get("id", "missing")
        o.check("register with datetime and sortby", "json", "/searches/register", "POST", {
            "collections": [collection], "datetime": "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z",
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        })
        layered = {
            "collections": [collection],
            "metadata": {"name": "layers", "defaults": {
                "rgb": {"assets": ["visual"], "minzoom": 8, "maxzoom": 14},
                "ndvi": {"expression": "(nir-red)/(nir+red)", "rescale": [[-1, 1]], "colormap_name": "viridis"},
                "false_colour": {"assets": ["nir", "red", "green"], "rescale": [[0, 3000], [0, 2500]],
                                 "color_formula": "gamma rgb 1.3", "resampling": "bilinear"},
                "one_band": {"assets": "red", "asset_bidx": ["red|1"], "colormap_name": "viridis", "unscale": True},
            }},
        }
        cz, _ = o.check("register with default layers", "json", "/searches/register", "POST", layered)
        sid_layers = cz.json().get("id", "missing")
        o.check("register rejects bbox with intersects", "status", "/searches/register", "POST", {
            "collections": [collection], "bbox": [0, 0, 1, 1],
            "intersects": {"type": "Point", "coordinates": [0, 0]},
        })
        o.check("register rejects cql2-text", "status", "/searches/register", "POST", {
            "collections": [collection], "filter-lang": "cql2-text",
        })
        o.check("register with an unknown operator", "status", "/searches/register", "POST", {
            "filter": {"op": "nope", "args": [1]},
        })

        print("metadata")
        o.check("info", "json", f"/searches/{sid}/info")
        o.check("info with metadata", "json", f"/searches/{sid_meta}/info")
        o.check("info with default layers", "json", f"/searches/{sid_layers}/info")
        o.check("info for an unknown search", "json", "/searches/deadbeef/info")
        o.check("tilejson", "json", f"/searches/{sid}/WebMercatorQuad/tilejson.json?assets=visual")
        o.check("tilejson passes parameters through", "json",
                f"/searches/{sid}/WebMercatorQuad/tilejson.json?assets=visual&tilesize=256&minzoom=8&maxzoom=14"
                "&tile_format=png&rescale=0,255&expression=(b1%2Bb2)/2")
        o.check("tilejson keeps tilesize in place", "json",
                f"/searches/{sid}/WebMercatorQuad/tilejson.json?tilesize=256&assets=red&assets=nir")
        o.check("tilejson from search metadata", "json",
                f"/searches/{sid_meta}/WebMercatorQuad/tilejson.json?assets=visual")
        o.check("tilejson without assets", "json", f"/searches/{sid}/WebMercatorQuad/tilejson.json")
        o.check("tilejson for an unknown search", "json",
                "/searches/deadbeef/WebMercatorQuad/tilejson.json?assets=visual")
        o.check("tilejson for an unknown tile matrix set", "status",
                f"/searches/{sid}/Nope/tilejson.json?assets=visual")
        o.check("map viewer", "status", f"/searches/{sid}/WebMercatorQuad/map.html?assets=visual")

        print("items")
        tiles = {z: mercantile.tile(lon, lat, z) for z in (8, 9, 10, 11, 12, 13)}
        for z, t in tiles.items():
            o.check(f"items for a z{z} tile", "ids", f"/searches/{sid}/tiles/WebMercatorQuad/{z}/{t.x}/{t.y}/assets")
        t12 = tiles[12]
        base12 = f"/searches/{sid}/tiles/WebMercatorQuad/12/{t12.x}/{t12.y}"
        o.check("items with a filter", "ids",
                f"/searches/{sid_meta}/tiles/WebMercatorQuad/10/{tiles[10].x}/{tiles[10].y}/assets")
        o.check("items with limits", "ids",
                f"/searches/{sid}/tiles/WebMercatorQuad/8/{tiles[8].x}/{tiles[8].y}/assets"
                "?items_limit=3&exitwhenfull=false&skipcovered=false")
        # The lean lookup, which titiler-pgstac does not have: it ignores the parameter and
        # answers with geojsonsearch's list, which the lean list has to be a part of.
        for z, t in tiles.items():
            o.check(f"lean items for a z{z} tile", "part",
                    f"/searches/{sid}/tiles/WebMercatorQuad/{z}/{t.x}/{t.y}/assets?item_lookup=lean&assets=visual")
        o.check("lean items with a filter", "part",
                f"/searches/{sid_meta}/tiles/WebMercatorQuad/10/{tiles[10].x}/{tiles[10].y}/assets?item_lookup=lean")
        o.check("lean items with limits", "ids",
                f"/searches/{sid}/tiles/WebMercatorQuad/8/{tiles[8].x}/{tiles[8].y}/assets"
                "?items_limit=3&exitwhenfull=false&skipcovered=false&item_lookup=lean")
        o.check("lean items outside the data", "json", f"/searches/{sid}/tiles/WebMercatorQuad/10/100/100/assets?item_lookup=lean")
        o.check("items outside the data", "json", f"/searches/{sid}/tiles/WebMercatorQuad/10/100/100/assets")
        o.check("items for an unknown search", "json", "/searches/deadbeef/tiles/WebMercatorQuad/10/100/100/assets")

        print("tiles")
        for z in (9, 10, 11, 12, 13):
            t = tiles[z]
            o.check(f"z{z} png", "image", f"/searches/{sid}/tiles/WebMercatorQuad/{z}/{t.x}/{t.y}.png?assets=visual")
        o.check("format chosen from the pixels", "image", f"{base12}?assets=visual")
        o.check("jpg", "image", f"{base12}.jpg?assets=visual")
        o.check("webp", "image", f"{base12}.webp?assets=visual")
        o.check("tilesize=512", "image", f"{base12}.png?assets=visual&tilesize=512")
        o.check("resampling=bilinear on a reprojected source", "image", f"{base12}.png?assets=visual&resampling=bilinear")
        o.check("reproject=bilinear", "image", f"{base12}.png?assets=visual&reproject=bilinear")
        o.check("three assets as RGB", "image", f"{base12}.png?assets=red&assets=green&assets=blue&rescale=0,3000")
        o.check("one band with a colormap", "image", f"{base12}.png?assets=nir&rescale=0,5000&colormap_name=viridis")
        o.check("expression across two assets", "image",
                f"{base12}.png?assets=nir&assets=red&expression=(b1-b2)/(b1%2Bb2)&rescale=-1,1&colormap_name=rdylgn")
        o.check("filtered search", "image",
                f"/searches/{sid_meta}/tiles/WebMercatorQuad/12/{t12.x}/{t12.y}.png?assets=visual")
        o.check("partial tile with items_limit=1", "image",
                f"/searches/{sid}/tiles/WebMercatorQuad/8/{tiles[8].x}/{tiles[8].y}?assets=visual&items_limit=1")

        print("errors and empties")
        o.check("tile without assets", "json", f"{base12}.png")
        o.check("tile outside the data", "status", f"/searches/{sid}/tiles/WebMercatorQuad/10/100/100.png?assets=visual")
        o.check("tile past the grid edge", "status", f"/searches/{sid}/tiles/WebMercatorQuad/12/9999/1.png?assets=visual")
        o.check("tile at an impossible zoom", "status", f"/searches/{sid}/tiles/WebMercatorQuad/40/1/1?assets=visual")
        o.check("tile for an unknown search", "json", "/searches/deadbeef/tiles/WebMercatorQuad/12/1/1?assets=visual")
        o.check("tile with an unknown asset", "json", f"{base12}?assets=nope")
        o.check("tile for an unknown tile matrix set", "status",
                f"/searches/{sid}/tiles/Nope/12/{t12.x}/{t12.y}?assets=visual")
        o.check("unknown route", "status", f"/searches/{sid}/nope")

        print("tilesets and item lists")
        o.check("tileset list", "tilesets", f"/searches/{sid}/tiles")
        o.check("tileset list keeps the query", "tilesets", f"/searches/{sid_meta}/tiles?assets=visual&rescale=0,255")
        o.check("tileset", "json", f"/searches/{sid}/tiles/WebMercatorQuad")
        o.check("tileset from search metadata", "json", f"/searches/{sid_meta}/tiles/WebMercatorQuad?assets=visual")
        o.check("tileset with a zoom range", "json", f"/searches/{sid_meta}/tiles/WebMercatorQuad?minzoom=5&maxzoom=7")
        o.check("tileset for an unknown tile matrix set", "status", f"/searches/{sid}/tiles/Nope")
        o.check("tileset for an unknown search", "json", "/searches/deadbeef/tiles/WebMercatorQuad")
        box = f"{lon - 0.05:.4f},{lat - 0.05:.4f},{lon + 0.05:.4f},{lat + 0.05:.4f}"
        o.check("items in a bbox", "ids", f"/searches/{sid}/bbox/{box}/assets")
        o.check("items in a bbox with limits", "ids", f"/searches/{sid}/bbox/{box}/assets?items_limit=7&skipcovered=false")
        o.check("items in a bbox, filtered search", "ids", f"/searches/{sid_meta}/bbox/{box}/assets")
        o.check("items in a bbox outside the data", "json", f"/searches/{sid}/bbox/-100,10,-99,11/assets")
        o.check("bbox with three numbers", "json", f"/searches/{sid}/bbox/1,2,3/assets")
        o.check("items at a point", "ids", f"/searches/{sid}/point/{lon:.4f},{lat:.4f}/assets")
        o.check("items at a point with a limit", "ids", f"/searches/{sid}/point/{lon:.4f},{lat:.4f}/assets?items_limit=5")
        o.check("items at a point outside the data", "json", f"/searches/{sid}/point/-100,10/assets")
        o.check("lean items in a bbox", "part", f"/searches/{sid}/bbox/{box}/assets?item_lookup=lean&max_size=256")
        o.check("lean items in a bbox, every one", "ids",
                f"/searches/{sid}/bbox/{box}/assets?items_limit=7&skipcovered=false&exitwhenfull=false&item_lookup=lean")
        o.check("lean items at a point", "ids", f"/searches/{sid}/point/{lon:.4f},{lat:.4f}/assets?item_lookup=lean")
        o.check("lean items at a point with a limit", "ids",
                f"/searches/{sid}/point/{lon:.4f},{lat:.4f}/assets?items_limit=5&item_lookup=lean")
        o.check("items at a point for an unknown search", "json", "/searches/deadbeef/point/1,2/assets")

        print("collection")
        col = f"/collections/{collection}"
        # A second, empty collection that carries STAC renders, to compare the derived defaults.
        renders_id = "cogzilla-oracle-renders"
        with psycopg.connect(env["DATABASE_URL"], autocommit=True) as conn:
            conn.execute("SELECT pgstac.upsert_collection(%s::jsonb)", [json.dumps({
                "id": renders_id, "type": "Collection", "stac_version": "1.0.0",
                "description": "cogzilla oracle: renders", "license": "proprietary", "links": [],
                "extent": {"spatial": {"bbox": [[w, s, e, n]]}, "temporal": {"interval": [[None, None]]}},
                "item_assets": {"visual": {}, "red": {}, "nir": {}},
                "renders": {
                    "rgb": {"title": "True colour", "assets": ["visual"], "minmax_zoom": [8, 14]},
                    "ndvi": {"title": "NDVI", "expression": "(nir-red)/(nir+red)", "rescale": [[-1, 1]],
                             "colormap_name": "viridis", "tilematrixsets": {"WebMercatorQuad": [8, 14]}},
                },
            })])
        o.check("collection info with renders", "json", f"/collections/{renders_id}/info")
        o.check("collection info", "json", f"{col}/info")
        o.check("collection info for an unknown collection", "json", "/collections/nope/info")
        o.check("collection info with ids", "json", f"{col}/info?ids=a,b")
        o.check("collection info with a bbox", "json", f"{col}/info?bbox=6,44,8,46")
        o.check("collection info with a bad bbox", "status", f"{col}/info?bbox=6,44,8")
        o.check("collection info with a datetime", "json",
                f"{col}/info?datetime=2024-06-01T00:00:00Z/2024-08-31T23:59:59Z")
        o.check("collection info with sortby", "json",
                f"{col}/info?sortby=%2Bproperties.eo:cloud_cover,-datetime")
        cloud = "query=%7B%22eo%3Acloud_cover%22%3A%7B%22lt%22%3A20%7D%7D"
        o.check("collection info with a query", "json", f"{col}/info?{cloud}")
        cql = ("filter=%7B%22op%22%3A%22%3C%22%2C%22args%22%3A%5B%7B%22property%22%3A%22eo%3Acloud_cover%22%7D%2C20%5D%7D"
               "&filter-lang=cql2-json")
        o.check("collection info with a CQL2 JSON filter", "json", f"{col}/info?{cql}")
        # CQL2 text is the default language: the search has to get the same JSON and hash.
        for label, text in [
            ("a comparison", "eo:cloud_cover < 20"),
            ("and, or, not", "eo:cloud_cover <= 10.5 AND (platform = 'sentinel-2b' OR NOT s2:mgrs_tile = '32TMR')"),
            ("between, in, like", "eo:cloud_cover BETWEEN 5 AND 40 AND platform IN ('sentinel-2a', 'sentinel-2b') AND s2:mgrs_tile LIKE '32T%'"),
            ("is null and negations", "view:sun_elevation IS NOT NULL AND platform NOT LIKE 'landsat%' AND eo:cloud_cover NOT BETWEEN 90 AND 100"),
            ("arithmetic and a function", "eo:cloud_cover + 1 < 2 * 20 - -1 AND casei(platform) = casei('SENTINEL-2B')"),
            ("a timestamp with an offset", "datetime > TIMESTAMP('2024-12-01T00:00:00+01:00') AND datetime < TIMESTAMP('2025-01-01T00:00:00.500Z')"),
            ("a temporal operator and an interval", "T_DURING(datetime, INTERVAL('2024-12-01T00:00:00Z', '2025-01-01T00:00:00Z'))"),
            ("a point", f"S_INTERSECTS(geometry, POINT({lon} {lat}))"),
            ("polygons", f"S_INTERSECTS(geometry, POLYGON(({lon - 0.1} {lat - 0.1}, {lon + 0.1} {lat - 0.1}, {lon} {lat + 0.1}, {lon - 0.1} {lat - 0.1}))) OR S_WITHIN(geometry, MULTIPOLYGON((({lon - 2} {lat - 2}, {lon + 2} {lat - 2}, {lon + 2} {lat + 2}, {lon - 2} {lat + 2}, {lon - 2} {lat - 2}))))"),
            ("a quoted property", "\"eo:cloud_cover\" < 20 and id <> 'it''s'"),
        ]:
            o.check(f"collection info with CQL2 text: {label}", "json", f"{col}/info?filter={quote(text)}")
        text = quote("eo:cloud_cover < 20 AND s2:mgrs_tile = '32TMR'")
        o.check("collection info with filter-lang=cql2-text", "json", f"{col}/info?filter={text}&filter-lang=cql2-text")
        o.check("items in a tile, CQL2 text filter", "ids",
                f"{col}/tiles/WebMercatorQuad/10/{tiles[10].x}/{tiles[10].y}/assets?filter={text}")
        o.check("items in a tile, CQL2 text with geometry and time", "ids",
                f"{col}/tiles/WebMercatorQuad/10/{tiles[10].x}/{tiles[10].y}/assets?filter="
                + quote(f"S_INTERSECTS(geometry, POINT({lon} {lat})) AND datetime >= TIMESTAMP('2024-12-15T00:00:00Z')"))
        o.check("collection tile, CQL2 text filter", "image",
                f"{col}/tiles/WebMercatorQuad/12/{t12.x}/{t12.y}.png?assets=visual&filter={text}")
        # pgSTAC 0.10 does not take a BBOX operand: both servers pass its error on.
        o.check("CQL2 text with a BBOX operand", "status",
                f"{col}/info?filter=" + quote(f"S_INTERSECTS(geometry, BBOX({lon - 2}, {lat - 2}, {lon + 2}, {lat + 2}))"))
        o.check("CQL2 text that does not parse", "status", f"{col}/info?filter={quote('eo:cloud_cover <')}")
        o.check("CQL2 text that is not a predicate", "status", f"{col}/info?filter={quote('eo:cloud_cover + 1')}")
        o.check("CQL2 text with a bad timestamp", "status", f"{col}/info?filter=" + quote("datetime > TIMESTAMP('yesterday')"))
        o.check("an unknown filter-lang", "status", f"{col}/info?filter={text}&filter-lang=cql2-nope")
        o.check("collection tilejson", "json", f"{col}/WebMercatorQuad/tilejson.json?assets=visual")
        o.check("collection tilejson with a bbox", "json",
                f"{col}/WebMercatorQuad/tilejson.json?assets=visual&bbox=6,44,8,46&minzoom=6&maxzoom=7")
        o.check("collection tilejson for an unknown collection", "json",
                "/collections/nope/WebMercatorQuad/tilejson.json?assets=visual")
        o.check("collection map viewer", "status", f"{col}/WebMercatorQuad/map.html?assets=visual")
        o.check("collection tileset list", "tilesets", f"{col}/tiles?assets=visual")
        o.check("collection tileset", "json", f"{col}/tiles/WebMercatorQuad?assets=visual&minzoom=4&maxzoom=9")
        o.check("collection tileset with a bbox", "json",
                f"{col}/tiles/WebMercatorQuad?bbox=6,44,8,46&minzoom=6&maxzoom=7")
        ctile = f"{col}/tiles/WebMercatorQuad/12/{t12.x}/{t12.y}"
        o.check("collection items for a tile", "ids", f"{ctile}/assets")
        o.check("collection items for a tile with a query", "ids", f"{ctile}/assets?{cloud}")
        o.check("collection items for a tile with a datetime", "ids",
                f"{ctile}/assets?datetime=2024-06-01T00:00:00Z/2024-08-31T23:59:59Z")
        o.check("collection items for a tile, sorted", "ids", f"{ctile}/assets?sortby=%2Bproperties.eo:cloud_cover")
        o.check("collection items in a bbox", "ids", f"{col}/bbox/{box}/assets")
        o.check("collection items at a point", "ids", f"{col}/point/{lon:.4f},{lat:.4f}/assets?items_limit=20")
        o.check("collection tile", "image", f"{ctile}.png?assets=visual")
        o.check("collection tile, format from the pixels", "image", f"{ctile}?assets=visual")
        o.check("collection tile with a query", "image", f"{ctile}.png?assets=visual&{cloud}")
        o.check("collection tile, three assets", "image",
                f"{ctile}.png?assets=red&assets=green&assets=blue&rescale=0,3000")
        o.check("collection tile without assets", "json", f"{ctile}.png")
        o.check("collection tile outside the data", "status",
                f"{col}/tiles/WebMercatorQuad/10/100/100.png?assets=visual")
        o.check("collection tile for an unknown collection", "json",
                f"/collections/nope/tiles/WebMercatorQuad/12/{t12.x}/{t12.y}?assets=visual")
        o.check("collection tile for an unknown tile matrix set", "status",
                f"{col}/tiles/Nope/12/{t12.x}/{t12.y}?assets=visual")
        o.check("collection unknown route", "status", f"{col}/nope")

        print("point values")
        # The newest items under these points have GeoTIFF assets. Without a limit the search
        # reaches items whose assets are JPEG 2000, which titiler-pgstac answers with a 500.
        pt = f"{lon:.4f},{lat:.4f}"
        mx, my = mercantile.xy(lon, lat)
        o.check("point, newest item", "json", f"/searches/{sid}/point/{pt}?assets=visual&items_limit=1")
        o.check("point, three items", "json", f"/searches/{sid}/point/{pt}?assets=visual&items_limit=3")
        o.check("point, two assets", "json", f"/searches/{sid}/point/{pt}?assets=red&assets=nir&items_limit=2")
        o.check("point, filtered search", "json", f"/searches/{sid_meta}/point/{pt}?assets=visual&items_limit=2")
        o.check("point in Web Mercator coordinates", "json",
                f"/searches/{sid}/point/{mx:.2f},{my:.2f}?assets=visual&items_limit=2&coord_crs=epsg:3857")
        o.check("point outside the data", "status", f"/searches/{sid}/point/-100,10?assets=visual")
        o.check("point without assets", "json", f"/searches/{sid}/point/{pt}")
        o.check("point with an unknown asset", "json", f"/searches/{sid}/point/{pt}?assets=nope&items_limit=1")
        o.check("point for an unknown search", "json", f"/searches/deadbeef/point/{pt}?assets=visual")
        o.check("collection point", "json", f"/collections/{collection}/point/{pt}?assets=visual&items_limit=2")
        o.check("point, band math on one asset", "json", f"/searches/{sid}/point/{pt}?assets=visual&expression=b1*2&items_limit=1")
        o.check("point, band math across assets", "json",
                f"/searches/{sid}/point/{pt}?assets=red&assets=nir&expression=(b2-b1)/(b2%2Bb1)&items_limit=2")
        o.check("point, two expressions", "json",
                f"/searches/{sid}/point/{pt}?assets=red&assets=nir&expression=b1%2Bb2;b2-b1&items_limit=1")
        o.check("point, expression with an asset name", "json",
                f"/searches/{sid}/point/{pt}?assets=red&assets=nir&expression=(nir-red)/(nir%2Bred)&items_limit=1")
        o.check("collection point, band math", "json",
                f"/collections/{collection}/point/{pt}?assets=red&assets=nir&expression=b2/b1&items_limit=1")
        o.check("collection point with a datetime", "json",
                f"/collections/{collection}/point/{pt}?assets=red&items_limit=2&datetime=2024-06-01T00:00:00Z/2024-08-31T23:59:59Z")

        print("bbox and feature images")
        part = f"{lon - 0.1:.4f},{lat - 0.05:.4f},{lon + 0.1:.4f},{lat + 0.05:.4f}"
        wide = f"{lon - 0.9:.4f},{lat - 0.5:.4f},{lon + 0.9:.4f},{lat + 0.5:.4f}"
        triangle = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [lon - 0.1, lat - 0.05], [lon + 0.1, lat - 0.05], [lon, lat + 0.05], [lon - 0.1, lat - 0.05]]]}}
        o.check("bbox, max_size", "image", f"/searches/{sid}/bbox/{part}.png?assets=visual&max_size=256")
        o.check("bbox, exact size", "image", f"/searches/{sid}/bbox/{part}/300x200.png?assets=visual")
        o.check("bbox, width only", "image", f"/searches/{sid}/bbox/{part}.png?assets=visual&width=300")
        o.check("bbox, height only", "image", f"/searches/{sid}/bbox/{part}.png?assets=visual&height=90")
        o.check("bbox at the data's own resolution", "image", f"/searches/{sid}/bbox/{part}.png?assets=visual",
                known="shared bbox renderer: a pixel with a zero in any band is nodata; TiTiler needs every band")
        o.check("bbox, jpg", "image", f"/searches/{sid}/bbox/{part}.jpg?assets=visual&max_size=256")
        o.check("bbox, webp", "image", f"/searches/{sid}/bbox/{part}/256x128.webp?assets=visual")
        o.check("bbox over several items", "image", f"/searches/{sid}/bbox/{wide}.png?assets=visual&max_size=512")
        o.check("bbox, three assets", "image",
                f"/searches/{sid}/bbox/{part}.png?assets=red&assets=green&assets=blue&rescale=0,3000&max_size=256")
        o.check("bbox, one band with a colormap", "image",
                f"/searches/{sid}/bbox/{part}.png?assets=nir&rescale=0,5000&colormap_name=viridis&max_size=256")
        o.check("bbox, filtered search", "image", f"/searches/{sid_meta}/bbox/{part}.png?assets=visual&max_size=256")
        o.check("bbox without assets", "json", f"/searches/{sid}/bbox/{part}.png")
        o.check("bbox without a format", "status", f"/searches/{sid}/bbox/{part}?assets=visual")
        o.check("bbox outside the data", "status", f"/searches/{sid}/bbox/-100,10,-99,11.png?assets=visual&max_size=64")
        o.check("bbox with an unknown asset", "json", f"/searches/{sid}/bbox/{part}.png?assets=nope&max_size=64")
        o.check("bbox for an unknown search", "json", f"/searches/deadbeef/bbox/{part}.png?assets=visual")
        o.check("collection bbox", "image", f"/collections/{collection}/bbox/{part}.png?assets=visual&max_size=256")
        o.check("feature, max_size", "image", f"/searches/{sid}/feature.png?assets=visual&max_size=256", "POST", triangle)
        o.check("feature, exact size", "image", f"/searches/{sid}/feature/200x100.png?assets=visual", "POST", triangle)
        o.check("feature, format from the pixels", "image", f"/searches/{sid}/feature?assets=visual&max_size=128", "POST", triangle)
        o.check("feature, format=jpg", "image", f"/searches/{sid}/feature?assets=visual&max_size=128&format=jpg", "POST", triangle)
        o.check("feature without assets", "json", f"/searches/{sid}/feature.png", "POST", triangle)
        o.check("collection feature", "image", f"/collections/{collection}/feature.png?assets=visual&max_size=256", "POST", triangle)
        o.check("feature, a bare geometry", "status", f"/searches/{sid}/feature.png?assets=visual&max_size=64", "POST", triangle["geometry"])
        o.check("feature, an empty object", "status", f"/searches/{sid}/feature/64x64.png?assets=visual", "POST", {})
        o.check("feature, a feature collection", "status", f"/searches/{sid}/feature?assets=visual&max_size=64", "POST",
                {"type": "FeatureCollection", "features": [triangle]})
        o.check("collection feature, an empty object", "status", f"/collections/{collection}/feature.png?assets=visual&max_size=64", "POST", {})

        print("statistics under features")
        five = {"type": "Feature", "properties": {"name": "five sides"}, "geometry": {"type": "Polygon", "coordinates": [[
            [lon - 0.03, lat - 0.03], [lon + 0.03, lat - 0.03], [lon + 0.03, lat + 0.01], [lon, lat + 0.03],
            [lon - 0.03, lat + 0.01], [lon - 0.03, lat - 0.03]]]}}
        holed = {"type": "Feature", "id": "holed", "properties": None, "geometry": {"type": "MultiPolygon", "coordinates": [
            [[[lon - 0.03, lat - 0.03], [lon + 0.03, lat - 0.03], [lon + 0.03, lat + 0.03], [lon - 0.03, lat + 0.03], [lon - 0.03, lat - 0.03]],
             [[lon - 0.01, lat - 0.01], [lon + 0.01, lat - 0.01], [lon + 0.01, lat + 0.01], [lon - 0.01, lat + 0.01], [lon - 0.01, lat - 0.01]]],
            [[[lon + 0.07, lat - 0.03], [lon + 0.09, lat - 0.03], [lon + 0.08, lat - 0.01], [lon + 0.07, lat - 0.03]]]]}}
        large = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [lon - 0.6, lat - 0.4], [lon + 0.6, lat - 0.35], [lon + 0.5, lat + 0.4], [lon - 0.2, lat + 0.3], [lon - 0.6, lat - 0.4]]]}}
        far = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [-150.4, 10.0], [-150.3, 10.0], [-150.3, 10.1], [-150.4, 10.0]]]}}
        st = f"/searches/{sid}/statistics"
        o.check("statistics at the data's own resolution", "zonal", f"{st}?assets=red", "POST", five)
        o.check("statistics, max_size", "zonal", f"{st}?assets=red&max_size=64", "POST", five)
        o.check("statistics, width and height", "zonal", f"{st}?assets=red&width=48&height=32", "POST", five)
        o.check("statistics, height only", "zonal", f"{st}?assets=nir&height=100", "POST", five)
        o.check("statistics, band math over two assets", "zonal",
                f"{st}?assets=red&assets=nir&expression=(b2-b1)/(b2%2Bb1)", "POST", five)
        o.check("statistics, two expressions", "zonal",
                f"{st}?assets=red&assets=nir&expression=b1%2Bb2;b2/b1&max_size=128", "POST", five)
        o.check("statistics, three bands, bin edges, a range and cover_scale", "zonal",
                f"{st}?assets=visual&height=100&histogram_bins=0,100,200,255&histogram_range=0,300&cover_scale=4", "POST", holed)
        o.check("statistics, categories and percentiles", "zonal",
                f"{st}?assets=scl&max_size=128&categorical=true&c=4&c=5&c=6&c=99&p=5&p=50&p=95", "POST", five)
        o.check("statistics, categorical without categories", "zonal", f"{st}?assets=scl&max_size=64&categorical=true", "POST", five)
        o.check("statistics, bin count and range", "zonal",
                f"{st}?assets=red&max_size=64&histogram_bins=4&histogram_range=0,2000", "POST", five)
        o.check("statistics over several items", "zonal", f"{st}?assets=red&max_size=512", "POST", large)
        o.check("statistics, filtered search", "zonal", f"/searches/{sid_meta}/statistics?assets=red&max_size=64", "POST", five)
        o.check("statistics of a feature collection", "zonal", f"{st}?assets=red&max_size=64", "POST",
                {"type": "FeatureCollection", "features": [five, holed]})
        o.check("collection statistics", "zonal", f"/collections/{collection}/statistics?assets=red&max_size=64", "POST", five)
        o.check("collection statistics, filtered", "zonal",
                f"/collections/{collection}/statistics?assets=red&max_size=64&datetime=2024-12-01T00:00:00Z/..", "POST", five)
        o.check("statistics without assets", "json", st, "POST", five)
        o.check("statistics of a bare geometry", "status", f"{st}?assets=red", "POST", five["geometry"])
        o.check("statistics outside the data", "status", f"{st}?assets=red&max_size=64", "POST", far)
        o.check("statistics with an unknown asset", "json", f"{st}?assets=nope&max_size=64", "POST", five)
        o.check("statistics with GET", "status", f"{st}?assets=red")
        o.check("statistics for an unknown search", "json", "/searches/deadbeef/statistics?assets=red", "POST", five)
        o.check("statistics for an unknown collection", "json", "/collections/nope/statistics?assets=red", "POST", five)

        print("per-asset options and asset_as_band")
        opt = lambda v: quote(v, safe="")
        ndvi = opt("(b2-b1)/(b2+b1)")
        for label, q in [
            ("one band of three", f"assets={opt('visual|bidx=1')}"),
            ("bands in another order", f"assets={opt('visual|bidx=3,1')}"),
            ("band math on one asset", f"assets={opt('red|expression=b1*2')}"),
            ("band math with two results", f"assets={opt('visual|expression=b1+b2;b3')}"),
            ("band math on two assets", f"assets={opt('red|expression=b1*2')}&assets={opt('nir|expression=b1/2')}"),
            ("an option on one of two assets", f"assets={opt('visual|bidx=2')}&assets=nir"),
            ("options and an expression over the result", f"assets={opt('visual|bidx=1,2')}&expression={opt('b1+b2')}"),
            ("an option TiTiler does not know", f"assets={opt('visual|nope=1')}"),
            ("the same asset twice", "assets=red&assets=red"),
            ("asset_as_band", "assets=red&assets=nir&asset_as_band=true"),
            ("asset_as_band and an expression", f"assets=red&assets=nir&asset_as_band=true&expression={ndvi}"),
            ("asset_as_band=false", "assets=red&asset_as_band=false"),
            ("asset names in an expression with asset_as_band", f"assets=red&assets=nir&asset_as_band=true&expression={opt('(nir-red)/(nir+red)')}"),
            ("a bare bidx is passed over", "assets=visual&bidx=1"),
        ]:
            o.check(f"point, {label}", "json", f"/searches/{sid}/point/{pt}?{q}&items_limit=1")
        o.check("point, an option without a value", "status", f"/searches/{sid}/point/{pt}?assets={opt('visual|bidx')}&items_limit=1")
        o.check("point, a band index that is not a number", "status", f"/searches/{sid}/point/{pt}?assets={opt('visual|bidx=x')}&items_limit=1")
        o.check("collection point, options", "json", f"/collections/{collection}/point/{pt}?assets={opt('visual|bidx=3,1')}&items_limit=1")
        o.check("tile, bidx=1 on a one-band asset", "image", f"{base12}.png?assets={opt('red|bidx=1')}&rescale=0,3000")
        o.check("tile, band math on one asset", "image", f"{base12}.png?assets={opt('red|expression=b1*2')}&rescale=0,6000")
        o.check("tile, three bands in another order", "image", f"{base12}.png?assets={opt('visual|bidx=3,2,1')}")
        o.check("tile, asset_as_band", "image", f"{base12}.png?assets=red&assets=green&assets=blue&rescale=0,3000&asset_as_band=true")
        o.check("tile, asset_as_band and an expression", "image", f"{base12}.png?assets=red&assets=nir&expression={ndvi}&rescale=-1,1&asset_as_band=true")
        o.check("tile, a bare bidx is passed over", "image", f"{base12}.png?assets=visual&bidx=1")
        o.check("tile, asset names in an expression", "json", f"{base12}.png?assets=red&assets=nir&expression={opt('(nir-red)/(nir+red)')}&rescale=-1,1")
        # Every scene under the tile, so the methods have something to choose between.
        every = "&skipcovered=false&exitwhenfull=false&items_limit=6"
        for method, q in [
            ("first", "assets=visual"),
            ("highest", "assets=visual"),
            ("highest", "assets=red&rescale=0,20000"),
            ("lowest", "assets=red&rescale=0,3000&colormap_name=viridis"),
            ("mean", "assets=red&rescale=0,12000"),
            ("mean", "assets=red&assets=green&assets=blue&asset_as_band=true&rescale=0,12000"),
            # These rasters carry no scale or offset, so this only shows `unscale` does no harm.
            ("mean", "assets=red&unscale=true&rescale=0,12000"),
            ("median", "assets=visual"),
            ("median", f"assets=red&expression={opt('b1/2')}&rescale=0,4000"),
            ("stdev", "assets=red&rescale=0,8000"),
            ("stdev", "assets=visual&rescale=0,120"),
            ("count", "assets=visual&rescale=0,8"),
        ]:
            o.check(f"tile, pixel_selection={method}, {q.split('&')[0][7:]}{' and more' if q.count('&') > 1 else ''}", "image",
                    f"{base12}.png?{q}&pixel_selection={method}{every}")
        o.check("tile, pixel_selection with the search's own limits", "image", f"{base12}.png?assets=visual&pixel_selection=median")
        o.check("tile, pixel_selection and a colour formula", "image",
                f"{base12}.png?assets=visual&pixel_selection=mean&color_formula={opt('gamma rgb 1.8')}{every}")
        o.check("tile, a colour formula", "image", f"{base12}.png?assets=visual&color_formula={opt('gamma rgb 1.8')}")
        o.check("tile, a colour formula, format chosen from the pixels", "image",
                f"{base12}?assets=visual&color_formula={opt('sigmoidal rgb 6 0.4')}")
        o.check("tile, no mask", "image", f"{base12}.png?assets=visual&return_mask=false")
        o.check("tile, a pixel_selection nobody has", "status", f"{base12}.png?assets=visual&pixel_selection=nope")
        # Tiles the renderer cannot draw in one pass, made from planes: each asset read on
        # its own grid, then stacked, put through band math and the pixel selection.
        four = "assets=red&assets=green&assets=blue&assets=nir&asset_as_band=true"
        mixed = "assets=swir16&assets=nir&assets=red&asset_as_band=true&rescale=0,4000"
        ndvi = opt("(b4-b1)/(b4+b1)")
        for name, q in [
            ("two assets", ".png?assets=red&assets=nir&rescale=0,4000"),
            ("two assets as bands, a range each", ".png?assets=red&assets=nir&asset_as_band=true&rescale=0,3000&rescale=0,6000"),
            ("four assets, band math", f".png?{four}&expression={ndvi}&rescale=-1,1"),
            ("four assets, band math and a colour map", f".png?{four}&expression={ndvi}&rescale=-1,1&colormap_name=viridis"),
            ("four assets, three results", f".png?{four}&expression={opt('b4;b1;b2')}&rescale=0,4000"),
            ("two assets, two results", f".png?assets=red&assets=nir&asset_as_band=true&expression={opt('b2;b1')}&rescale=0,4000"),
            ("four assets, pixel_selection=lastbandhigh", f".png?{four}&rescale=0,4000&pixel_selection=lastbandhigh{every}"),
            ("four assets, pixel_selection=lastbandlow", f".png?{four}&rescale=0,4000&pixel_selection=lastbandlow{every}"),
            ("two assets, pixel_selection=lastbandhigh", f".png?assets=red&assets=nir&asset_as_band=true&rescale=0,4000&pixel_selection=lastbandhigh{every}"),
            ("one asset, pixel_selection=lastbandlow", f".png?assets=visual&pixel_selection=lastbandlow{every}"),
            ("assets on two grids", f".png?{mixed}"),
            ("assets on two grids, format chosen from the pixels", f"?{mixed}"),
            ("assets on two grids, webp", f".webp?{mixed}"),
            ("assets on two grids, pixel_selection=mean", f".png?{mixed}&pixel_selection=mean{every}"),
            ("assets on two grids, band math", f".png?assets=swir16&assets=nir&asset_as_band=true&expression={opt('(b2-b1)/(b2+b1)')}&rescale=-1,1"),
            ("assets on two grids, a colour formula", f".png?{mixed}&color_formula={opt('gamma rgb 1.5')}"),
            ("assets on two grids, reproject=bilinear", f".png?{mixed}&reproject=bilinear"),
            ("band math on one asset of several", f".png?assets={opt('red|expression=b1*2')}&assets=nir&assets=green&rescale=0,8000"),
            ("band math over an asset's own", f".png?assets={opt('visual|expression=b1+b2;b3')}&expression={opt('b1/(b2+1)')}&rescale=0,4"),
        ]:
            o.check(f"planes tile, {name}", "image", f"{base12}{q}")
        o.check("planes tile, on a collection", "image", f"/collections/{collection}/tiles/WebMercatorQuad/12/{t12.x}/{t12.y}.png?{mixed}")
        # A tile at the edge of the data, where the mask is not full and decides the format.
        edge = f"/searches/{sid}/tiles/WebMercatorQuad/8/131/88"
        o.check("planes tile, two bands, format chosen from the pixels", "image", f"{edge}?assets=red&assets=nir&rescale=0,4000")
        o.check("planes tile, two grids, format chosen from the pixels", "image", f"{edge}?{mixed}")
        o.check("planes tile, asset_as_band after bidx", "image",
                f"{base12}.png?assets={opt('visual|bidx=2')}&assets=nir&asset_as_band=true&rescale=0,255&rescale=0,4000")
        o.check("planes tile, asset_as_band on three bands", "status", f"{base12}.png?assets=visual&assets=nir&asset_as_band=true",
                known="titiler-pgstac answers 500 for asset_as_band on a multi-band asset; cogzilla answers 400")
        o.check("planes tile, four bands as png", "json", f"{base12}.png?{four}&rescale=0,4000")
        o.check("planes tile, five bands", "json", f"{base12}.png?{four}&assets=swir16&rescale=0,4000")
        o.check("planes tile, two bands as jpeg", "json", f"{base12}?assets=red&assets=nir&asset_as_band=true&rescale=0,4000")
        o.check("planes tile, four bands as jpeg", "status", f"{base12}.jpg?{four}&rescale=0,4000",
                known="TiTiler sends four bands as a CMYK JPEG; cogzilla refuses them as it does for PNG")
        o.check("planes tile, two bands of three", "image", f"{base12}.png?assets={opt('visual|bidx=3,2')}&rescale=0,255")
        for method, q in [
            ("first", f"{four}&expression={ndvi}"),
            ("first", "assets=red&assets=green&assets=blue&assets=nir&assets=swir16"),
            ("lastbandhigh", "assets=red&assets=nir&asset_as_band=true"),
            ("lastbandlow", "assets=visual"),
            ("mean", f"{four}&expression={ndvi}"),
        ]:
            o.check(f"statistics, pixel_selection={method}, {'band math' if 'expression' in q else q.count('assets=')} assets" if q.count('assets=') > 1 else f"statistics, pixel_selection={method}, one asset", "zonal",
                    f"{st}?{q}&max_size=64&pixel_selection={method}{every}", "POST", five)
        o.check("bbox, pixel_selection=lastbandhigh", "status",
                f"/searches/{sid}/bbox/{part}.png?assets=visual&max_size=64&pixel_selection=lastbandhigh{every}",
                known="bbox and feature images do not take the last-band methods, more than three assets or assets on different grids yet")
        for method, q in [
            ("highest", "assets=visual"),
            ("lowest", "assets=red&rescale=0,3000&colormap_name=viridis"),
            ("mean", "assets=red&assets=green&assets=blue&asset_as_band=true&rescale=0,12000"),
            ("median", f"assets=red&expression={opt('b1/2')}&rescale=0,4000"),
            ("stdev", "assets=red&rescale=0,8000"),
            ("count", "assets=visual&rescale=0,8"),
        ]:
            o.check(f"bbox, pixel_selection={method}", "image",
                    f"/searches/{sid}/bbox/{part}.png?{q}&max_size=256&pixel_selection={method}{every}")
        o.check("bbox, pixel_selection and a colour formula", "image",
                f"/searches/{sid}/bbox/{part}.png?assets=visual&max_size=256&pixel_selection=median&color_formula={opt('gamma rgb 1.8')}{every}")
        o.check("feature, pixel_selection=mean", "image",
                f"/searches/{sid}/feature.png?assets=visual&max_size=256&pixel_selection=mean{every}", "POST", triangle)
        o.check("collection bbox, pixel_selection=highest", "image",
                f"/collections/{collection}/bbox/{part}.png?assets=visual&max_size=256&pixel_selection=highest{every}")
        for method, q in [
            ("highest", "assets=red"),
            ("lowest", "assets=red"),
            ("mean", "assets=red"),
            ("mean", "assets=visual"),
            ("mean", f"assets=red&assets=nir&asset_as_band=true&expression={opt('(b2-b1)/(b2+b1)')}"),
            ("median", "assets=red"),
            ("median", "assets=red&assets=nir&asset_as_band=true"),
            ("stdev", "assets=red"),
            ("count", "assets=red"),
            ("count", "assets=visual"),
        ]:
            o.check(f"statistics, pixel_selection={method}, {q.split('&')[0][7:]}{' and more' if q.count('&') else ''}", "zonal",
                    f"{st}?{q}&max_size=64&pixel_selection={method}{every}", "POST", five)
        o.check("statistics, pixel_selection with the search's own limits", "zonal", f"{st}?assets=red&max_size=64&pixel_selection=median", "POST", five)
        o.check("statistics, pixel_selection over many features", "zonal", f"{st}?assets=red&max_size=64&pixel_selection=mean{every}", "POST",
                {"type": "FeatureCollection", "features": [five, holed]})
        o.check("statistics, a pixel_selection nobody has", "status", f"{st}?assets=red&pixel_selection=nope", "POST", five)
        o.check("tile, one band of three", "image", f"{base12}.png?assets={opt('visual|bidx=1')}",
                known="GDAL reads a subset of an RGB raster's bands from other pixels than all of them (rio-tiler tile(indexes=(1,)) differs from tile(indexes=(1,2,3))[0] in 43% of pixels); cogzilla reads the same pixels either way")
        o.check("bbox, bidx=1 on a one-band asset", "image", f"/searches/{sid}/bbox/{part}.png?assets={opt('red|bidx=1')}&rescale=0,3000&max_size=256")
        o.check("bbox, asset_as_band", "image", f"/searches/{sid}/bbox/{part}.png?assets=red&assets=green&assets=blue&rescale=0,3000&asset_as_band=true&max_size=256")
        o.check("statistics, one band of three", "zonal", f"{st}?assets={opt('visual|bidx=2')}&max_size=64", "POST", five)
        o.check("statistics, bands in another order", "zonal", f"{st}?assets={opt('visual|bidx=3,1')}", "POST", five)
        o.check("statistics, band math on one asset", "zonal", f"{st}?assets={opt('red|expression=b1*2')}&max_size=64", "POST", five)
        o.check("statistics, asset_as_band", "zonal", f"{st}?assets=red&assets=nir&asset_as_band=true&max_size=64", "POST", five)
        o.check("statistics, asset_as_band and an expression", "zonal",
                f"{st}?assets=red&assets=nir&asset_as_band=true&expression={ndvi}&max_size=64", "POST", five)
        o.check("statistics, options and an expression over the result", "zonal",
                f"{st}?assets={opt('visual|bidx=1,2')}&assets=nir&expression={opt('b1+b2;b3/2')}&max_size=64", "POST", five)

        print("wmts")
        o.check("search wmts", "wmts", f"/searches/{sid}/WMTSCapabilities.xml?assets=visual")
        o.check("search wmts keeps tile parameters", "wmts",
                f"/searches/{sid}/WMTSCapabilities.xml?assets=visual&rescale=0,255&tilesize=512&tile_scale=2&resampling=bilinear")
        o.check("search wmts, jpg and EPSG codes", "wmts",
                f"/searches/{sid}/WMTSCapabilities.xml?assets=visual&tile_format=jpg&use_epsg=true&minzoom=3&maxzoom=5")
        o.check("search wmts from search metadata", "wmts", f"/searches/{sid_meta}/WMTSCapabilities.xml?assets=visual")
        o.check("search wmts with a zoom override", "wmts",
                f"/searches/{sid}/WMTSCapabilities.xml?assets=visual&zooms=WebMercatorQuad::4,9")
        o.check("search wmts from default layers", "wmts", f"/searches/{sid_layers}/WMTSCapabilities.xml")
        o.check("search wmts, default layers and the query", "wmts",
                f"/searches/{sid_layers}/WMTSCapabilities.xml?assets=visual&tile_format=webp")
        o.check("search wmts with no layer", "wmts", f"/searches/{sid}/WMTSCapabilities.xml")
        o.check("search wmts with an unknown format", "status", f"/searches/{sid}/WMTSCapabilities.xml?assets=visual&tile_format=nope")
        o.check("search wmts for an unknown search", "json", "/searches/deadbeef/WMTSCapabilities.xml?assets=visual")
        o.check("collection wmts", "wmts", f"/collections/{collection}/WMTSCapabilities.xml?assets=visual&rescale=0,255")
        o.check("collection wmts with a bbox", "wmts", f"/collections/{collection}/WMTSCapabilities.xml?assets=visual&bbox=6,44,8,46")
        o.check("collection wmts with renders", "wmts", f"/collections/cogzilla-oracle-renders/WMTSCapabilities.xml")
        with psycopg.connect(env["DATABASE_URL"], autocommit=True) as conn:
            item_id, ib = conn.execute(
                "SELECT id, content->'bbox' FROM pgstac.items WHERE collection = %s"
                " AND (content->'properties'->>'eo:cloud_cover')::float < 5 ORDER BY datetime DESC, id LIMIT 1",
                [collection],
            ).fetchone()
        ilon, ilat = (ib[0] + ib[2]) / 2, (ib[1] + ib[3]) / 2
        it = f"/collections/{collection}/items/{item_id}"
        ti = mercantile.tile(ilon, ilat, 11)
        itile = f"{it}/tiles/WebMercatorQuad/{ti.z}/{ti.x}/{ti.y}"
        ibox = f"{ilon - 0.1:.4f},{ilat - 0.1:.4f},{ilon + 0.1:.4f},{ilat + 0.1:.4f}"
        o.check("item assets", "json", f"{it}/assets")
        o.check("item for an unknown id", "json", f"/collections/{collection}/items/nope/info?assets=visual")
        o.check("item in an unknown collection", "json", f"/collections/nope/items/{item_id}/assets")
        o.check("item info", "json", f"{it}/info?assets=visual")
        o.check("item info, two assets", "json", f"{it}/info?assets=red&assets=nir")
        o.check("item info without assets", "json", f"{it}/info")
        o.check("item info with an unknown asset", "json", f"{it}/info?assets=nope")
        o.check("item info.geojson", "json", f"{it}/info.geojson?assets=visual")
        o.check("item renders", "json", f"{it}/renders")
        o.check("item tileset list", "tilesets", f"{it}/tiles?assets=visual")
        o.check("item tileset", "json", f"{it}/tiles/WebMercatorQuad?assets=visual")
        o.check("item map viewer", "status", f"{it}/WebMercatorQuad/map.html?assets=visual")
        o.check("item tile", "image", f"{itile}.png?assets=visual")
        o.check("item tile, format from the pixels", "image", f"{itile}?assets=visual")
        o.check("item tile, jpg", "image", f"{itile}.jpg?assets=visual")
        full = mercantile.tile(ilon, ilat, 14)
        ifull = f"{it}/tiles/WebMercatorQuad/{full.z}/{full.x}/{full.y}"
        o.check("item tile, fully covered, format from the pixels", "image", f"{ifull}?assets=visual")
        o.check("item tile, fully covered, format=png", "image", f"{ifull}?assets=visual&format=png")
        o.check("item tile, three assets", "image", f"{itile}.png?assets=red&assets=green&assets=blue&rescale=0,3000")
        o.check("item tile, one band with a colormap", "image",
                f"{itile}.png?assets=nir&rescale=0,5000&colormap_name=viridis")
        o.check("item tile, tilesize=512", "image", f"{itile}.png?assets=visual&tilesize=512")
        o.check("item tile without assets", "json", f"{itile}.png")
        o.check("item tile with expression only", "status", f"{itile}.png?expression=(nir-red)/(nir%2Bred)")
        o.check("item tile outside the item", "status", f"{it}/tiles/WebMercatorQuad/5/0/0.png?assets=visual")
        o.check("item preview", "image", f"{it}/preview.png?assets=visual&max_size=256",
                known="shared /stac preview: the picture does not fill max_size")
        o.check("item preview, sized", "image", f"{it}/preview/200x200.png?assets=visual",
                known="shared /stac preview: the picture does not fill max_size")
        o.check("item bbox", "image", f"{it}/bbox/{ibox}/256x256.png?assets=visual")
        o.check("item point", "json", f"{it}/point/{ilon:.4f},{ilat:.4f}?assets=visual")
        o.check("item point, two assets", "json", f"{it}/point/{ilon:.4f},{ilat:.4f}?assets=red&assets=nir")
        o.check("item point outside the item", "status", f"{it}/point/0,0?assets=visual",
                known="titiler-pgstac answers 500 for a point outside the item; cogzilla answers 404")
        o.check("item statistics", "stats", f"{it}/statistics?assets=visual&max_size=128",
                known="shared /stac statistics: the sample is smaller than max_size, and merged bands are described as b1, not visual_b1")
        o.check("item asset statistics", "stats", f"{it}/asset_statistics?assets=red&max_size=128",
                known="shared /stac statistics: the sample is smaller than max_size, and merged bands are described as b1, not visual_b1")
        o.check("item wmts", "wmts", f"{it}/WMTSCapabilities.xml?assets=visual")
        o.check("item wmts, jpg and EPSG codes", "wmts",
                f"{it}/WMTSCapabilities.xml?assets=visual&tile_format=jpg&use_epsg=true&minzoom=9&maxzoom=10")
        o.check("item wmts with no layer", "wmts", f"{it}/WMTSCapabilities.xml")
        o.check("item point, band math", "json",
                f"{it}/point/{ilon:.4f},{ilat:.4f}?assets=red&assets=nir&expression=(b2-b1)/(b2%2Bb1)")
        o.check("item point, two expressions", "json",
                f"{it}/point/{ilon:.4f},{ilat:.4f}?assets=red&assets=nir&expression=b1%2Bb2;b2*2")
        o.check("item point, expression with an asset name", "json",
                f"{it}/point/{ilon:.4f},{ilat:.4f}?assets=red&assets=nir&expression=nir-red")

        # An item that carries STAC renders, with the assets of a real one.
        with psycopg.connect(env["DATABASE_URL"], autocommit=True) as conn:
            src = conn.execute("SELECT pgstac.get_item(%s, %s)", [item_id, collection]).fetchone()[0]
            conn.execute("SELECT pgstac.upsert_item(%s::jsonb)", [json.dumps({
                "type": "Feature", "stac_version": "1.0.0", "id": "cogzilla-oracle-item",
                "collection": renders_id, "stac_extensions": src.get("stac_extensions", []),
                "bbox": src["bbox"], "geometry": src["geometry"], "links": [],
                "properties": {
                    "datetime": "2024-12-31T10:00:00Z", "proj:epsg": src["properties"].get("proj:epsg"),
                    "renders": {
                        "rgb": {"title": "True colour", "assets": ["visual"]},
                        "ndvi": {"title": "NDVI", "assets": ["red", "nir"], "expression": "(nir-red)/(nir+red)",
                                 "rescale": [[-1, 1]], "colormap_name": "viridis"},
                        "one_band": {"assets": ["red"], "asset_bidx": ["red|1"], "rescale": [[0, 3000]],
                                     "minmax_zoom": [8, 12]},
                        "no_assets": {"expression": "b1*2"},
                    },
                },
                "assets": {k: src["assets"][k] for k in ("visual", "red", "nir", "thumbnail")},
            })])
        rit = f"/collections/{renders_id}/items/cogzilla-oracle-item"
        o.check("item renders with content", "json", f"{rit}/renders")
        o.check("item render", "json", f"{rit}/renders/ndvi")
        o.check("item render that cannot make a tile", "json", f"{rit}/renders/no_assets")
        o.check("item render that does not exist", "json", f"{rit}/renders/nope")
        o.check("item wmts from renders", "wmts", f"{rit}/WMTSCapabilities.xml")
        o.check("item wmts from renders and the query", "wmts", f"{rit}/WMTSCapabilities.xml?assets=visual&tile_format=jpg")
        o.check("item unknown route", "status", f"{it}/nope")
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    passed = sum(r["status"] == "PASS" for r in o.rows)
    known = sum(r["status"] == "KNOWN" for r in o.rows)
    failed = len(o.rows) - passed - known
    out = RESULTS / f"oracle-{time.strftime('%Y%m%d-%H%M%S')}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"passed": passed, "known": known, "failed": failed, "total": len(o.rows), "checks": o.rows}, indent=1))
    print(f"\n{passed} of {len(o.rows)} checks agree, {known} known differences, {failed} failures. raw: {out}")
    for r in o.rows:
        if r["status"] != "PASS":
            print(f"  {r['status']}  {r['name']}: {r['note']}\n        {r['method']} {r['path']}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
