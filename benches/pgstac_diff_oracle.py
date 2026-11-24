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

import httpx
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


class Oracle:
    def __init__(self, timeout: float):
        self.client = httpx.Client(timeout=timeout)
        self.rows: list[dict] = []

    def fetch(self, base: str, method: str, path: str, body):
        t0 = time.perf_counter()
        r = self.client.request(method, base + path, json=body)
        return r, (time.perf_counter() - t0) * 1e3

    def check(self, name: str, kind: str, path: str, method: str = "GET", body=None):
        tt, tt_ms = self.fetch(TITILER, method, path, body)
        cz, cz_ms = self.fetch(COGZILLA, method, path, body)
        ok, note = True, ""
        if cz.status_code != tt.status_code:
            ok, note = False, f"status {cz.status_code} vs {tt.status_code}: {cz.text[:80]!r}"
        elif kind == "status":
            note = f"{cz.status_code}"
        elif kind == "json":
            diff = first_diff(normalize(cz.json(), COGZILLA), normalize(tt.json(), TITILER))
            ok, note = diff is None, diff or f"{cz.status_code}, bodies equal"
        elif kind == "tilesets":
            # cogzilla serves these routes in Web Mercator only, so only that entry is compared.
            ours = normalize(cz.json(), COGZILLA)
            theirs = normalize(tt.json(), TITILER)
            theirs["tilesets"] = [t for t in theirs["tilesets"] if "WebMercatorQuad TileMatrixSet" in t["title"]]
            diff = first_diff(ours, theirs)
            ok, note = diff is None, diff or "WebMercatorQuad entry equal"
        elif kind == "ids":
            a, b = [i["id"] for i in cz.json()], [i["id"] for i in tt.json()]
            ok = a == b
            note = f"{len(a)} items, same order" if ok else f"item lists differ: {len(a)} vs {len(b)}"
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
        self.rows.append({
            "name": name, "kind": kind, "method": method, "path": path, "ok": ok, "note": note,
            "cogzilla_ms": round(cz_ms, 1), "titiler_ms": round(tt_ms, 1),
        })
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {note}", flush=True)
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
        o.check("tile with an unknown asset", "status", f"{base12}?assets=nope")
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
        o.check("items at a point for an unknown search", "json", "/searches/deadbeef/point/1,2/assets")

        print("collection")
        col = f"/collections/{collection}"
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
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    passed = sum(r["ok"] for r in o.rows)
    out = RESULTS / f"oracle-{time.strftime('%Y%m%d-%H%M%S')}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": passed, "total": len(o.rows), "checks": o.rows}, indent=1))
    print(f"\n{passed} of {len(o.rows)} checks pass. raw: {out}")
    for r in o.rows:
        if not r["ok"]:
            print(f"  FAIL  {r['name']}: {r['note']}\n        {r['method']} {r['path']}")
    return 0 if passed == len(o.rows) else 1


if __name__ == "__main__":
    sys.exit(main())
