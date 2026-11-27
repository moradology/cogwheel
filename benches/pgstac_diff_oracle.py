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
import re
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
    # A name or a password may hold `/` or `@`, which a URL would read as its own.
    user = quote(os.environ.get("PGUSER", "username"), safe="")
    password = quote(os.environ.get("PGPASSWORD", "password"), safe="")
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


def geotiff_array(body: bytes) -> tuple[bytes, str]:
    """A GeoTIFF as the NumPy array it holds, and what it says of itself."""
    import warnings

    from rasterio.io import MemoryFile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MemoryFile(body) as memory, memory.open() as ds:
            said = (f"{ds.dtypes[0]} x{ds.count} {[c.name for c in ds.colorinterp]} "
                    f"epsg {ds.crs.to_epsg() if ds.crs else None} {[round(v, 6) for v in ds.bounds]}")
            out = io.BytesIO()
            np.save(out, ds.read())
            return out.getvalue(), said


def compare_geotiffs(cz: bytes, tt: bytes, smooth: bool) -> tuple[bool, str]:
    try:
        (a, said_a), (b, said_b) = geotiff_array(cz), geotiff_array(tt)
    except Exception as e:  # noqa: BLE001 - an unreadable GeoTIFF is a failed check
        return False, f"unreadable GeoTIFF: {e}"
    if said_a != said_b:
        return False, f"{said_a} vs {said_b}"
    ok, note = compare_arrays(a, b, smooth)
    return ok, f"GeoTIFF, {note}"


def same_place(cz, tt) -> tuple[bool, str]:
    """`Content-Crs` equal, and `Content-Bbox` equal to a millionth of a unit (the last
    digits of a float worked out another way)."""
    crs_cz, crs_tt = cz.headers.get("content-crs"), tt.headers.get("content-crs")
    if crs_cz != crs_tt:
        return False, f"content-crs {crs_cz} vs {crs_tt}"
    box_cz, box_tt = cz.headers.get("content-bbox"), tt.headers.get("content-bbox")
    if (box_cz is None) != (box_tt is None):
        return False, f"content-bbox {box_cz} vs {box_tt}"
    if box_cz is None:
        return True, "neither placed"
    a, b = [float(v) for v in box_cz.split(",")], [float(v) for v in box_tt.split(",")]
    if len(a) != 4 or any(abs(x - y) > 1e-6 * max(1.0, abs(y)) for x, y in zip(a, b)):
        return False, f"content-bbox {box_cz} vs {box_tt}"
    return True, f"{crs_cz.rsplit('/', 1)[-1]} {box_cz}"


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


def compare_arrays(cz: bytes, tt: bytes, smooth: bool = False, masked: bool = True) -> tuple[bool, str]:
    """NumPy arrays: the same type and shape, the same mask band, and values as close as
    two pictures of them would be (most equal; the rest from the pixel next door).
    Without a mask band (`return_mask=false`) a pixel counts as there where any band of
    the reference's is not zero."""
    try:
        a, b = np.load(io.BytesIO(cz)), np.load(io.BytesIO(tt))
    except Exception as e:  # noqa: BLE001 - an unreadable array is a failed check
        return False, f"unreadable array: {e}"
    if a.dtype != b.dtype or a.shape != b.shape:
        return False, f"{a.dtype} {a.shape} vs {b.dtype} {b.shape}"
    if not masked:
        there = (b != 0).any(axis=0)
        top = np.iinfo(a.dtype).max if a.dtype.kind in "ui" else 1
        a = np.concatenate([a, (there * top).astype(a.dtype)[None]])
        b = np.concatenate([b, (there * top).astype(b.dtype)[None]])
    ma, mb = a[-1] != 0, b[-1] != 0
    mask = float((ma == mb).mean()) * 100
    tops = set(np.unique(a[-1][ma]).tolist()) == set(np.unique(b[-1][mb]).tolist())
    both = ma & mb
    if not both.any():
        return mask >= 99.0 and tops, f"{a.dtype} {a.shape}, mask {mask:.2f}%"
    va, vb = a[:-1][:, both].astype(np.float64), b[:-1][:, both].astype(np.float64)
    if a.dtype.kind == "f":
        # Fractions are not equal from one pixel to its neighbour: the means are compared.
        # Halved first: the largest double, which a division by zero is written as, has no sum.
        ha, hb = (va / 2).mean(), (vb / 2).mean()
        close = ha == hb or abs(ha - hb) <= 0.02 * abs(hb) + 1e-9
        return mask >= 99.0 and tops and close, f"{a.dtype} {a.shape}, mask {mask:.2f}%, mean {va.mean():.5g} vs {vb.mean():.5g}"
    # Band by band: one band that is wrong is not hidden by the others that are right.
    same = min(float((va[i] == vb[i]).mean()) * 100 for i in range(va.shape[0]))
    if smooth:
        # Between pixels two readers do not land on the same number: the worst band's
        # mean difference is held to a tenth of its mean value.
        off = max(float(np.abs(va[i] - vb[i]).mean() / (np.abs(vb[i]).mean() + 1e-9)) for i in range(va.shape[0])) * 100
        return mask >= 99.0 and tops and off <= 10.0, f"{a.dtype} {a.shape}, mask {mask:.2f}%, values {off:.1f}% apart in the worst band"
    return mask >= 99.0 and tops and same >= 75.0, f"{a.dtype} {a.shape}, mask {mask:.2f}%, {same:.1f}% of values equal in the worst band"


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
        elif kind == "page":
            # An HTML page of the same document: the markup is not TiTiler's templates, so what
            # is compared is that both are pages and that every tileset ours links to is one
            # theirs links to.
            def tileset_links(r, base):
                hrefs = re.findall(r'href="([^"]+)"', r.text)
                # A tile's URL template (`{z}`, or `%7Bz%7D` once encoded) is not a tileset.
                links = {h.replace(base, "").replace("&amp;", "&") for h in hrefs if "/tiles/" in h and "{" not in h and "%7B" not in h}
                # A page's link to itself (a tileset's `self`) is left out of the comparison.
                return {h for h in links if h.split("?")[0] != path.split("?")[0]}
            types = (cz.headers.get("content-type", ""), tt.headers.get("content-type", ""))
            missing = tileset_links(cz, COGZILLA) - tileset_links(tt, TITILER)
            ok = all(t.startswith("text/html") for t in types) and not missing
            note = f"{types[0]}" if ok else f"{types} links only ours: {sorted(missing)[:2]}"
        elif kind == "placed":
            ok, note = same_place(cz, tt)
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
        elif kind == "list":
            try:
                a, b = normalize(cz.json(), COGZILLA), normalize(tt.json(), TITILER)
                view = lambda d: ([s["search"]["hash"] for s in d["searches"]], d["links"], d["context"])  # noqa: E731
                if view(a) != view(b):
                    ok, note = False, f"{view(a)} vs {view(b)}"[:300]
            except Exception as e:  # noqa: BLE001 - a body that is not a list page is a failed check
                ok, note = False, f"not a list page: {e}"
        elif kind == "image":
            ct_cz = cz.headers.get("content-type", "")
            ct_tt = tt.headers.get("content-type", "")
            if cz.status_code != 200:
                note = f"{cz.status_code}"
            elif ct_cz != ct_tt:
                ok, note = False, f"content type {ct_cz} vs {ct_tt}"
            elif cz.content[:6] == b"\x93NUMPY":
                ok, note = compare_arrays(cz.content, tt.content, smooth="resampling=" in path, masked="return_mask=false" not in path)
            elif cz.content[:4] == b"II*\x00":
                ok, note = compare_geotiffs(cz.content, tt.content, "resampling=" in path)
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

        print("list")
        # The page itself carries counters that move with every request; its order, its
        # links and what it matched are what is compared.
        for name, qs in [("list", "limit=3"), ("list sorted by a metadata key", "limit=5&sortby=-name,lastused"),
                         ("list sorted by something no search has", "limit=5&sortby=-hash"),
                         ("list of searches with some metadata", "name=low%20cloud&limit=4"),
                         ("list of searches with metadata none has", "limit=3&name=nope&owner=me"),
                         ("list page two", "limit=2&offset=2&sortby=usecount"),
                         ("list of searches with a number in their metadata", "minzoom=3&limit=5"),
                         ("list past the last page", "limit=2&offset=100000")]:
            o.check(name, "list", f"/searches/list?{qs}")

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
        o.check("items with limits of zero, which is none given", "ids",
                f"/searches/{sid}/tiles/WebMercatorQuad/8/{tiles[8].x}/{tiles[8].y}/assets?items_limit=0&scan_limit=0&time_limit=0")
        # pgSTAC gives a scan limit whole to its first chunk of time and one fewer to the rest.
        for limit in (1, 2, 3):
            o.check(f"tile with scan_limit={limit}", "status", f"{base12}.png?assets=visual&scan_limit={limit}")
        o.check("items in a box given in Web Mercator", "ids",
                f"/searches/{sid}/bbox/930000,5700000,990000,5760000/assets?coord_crs=epsg:3857")
        o.check("items at a point given in Web Mercator", "ids",
                f"/searches/{sid}/point/946187,5740000/assets?coord_crs=epsg:3857")
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
        o.check("three results of band math, no range", "image", f"{base12}.png?assets=visual&expression=b3;b2;b1",
                known="titiler-pgstac stretches the int32 result's whole range (every pixel 127); cogzilla draws three results in the raster's range, as its renderer draws an item's")
        o.check("jpg", "image", f"{base12}.jpg?assets=visual")
        o.check("tif", "image", f"{base12}.tif?assets=visual")
        o.check("tif, bilinear", "image", f"{base12}.tif?assets=visual&resampling=bilinear&reproject=bilinear")
        # A GeoTIFF is the array: the raster's own type, every band, the mask as alpha.
        o.check("tif of one band", "image", f"{base12}.tif?assets=red")
        o.check("tiff of one band", "image", f"{base12}.tiff?assets=red")
        o.check("tif of two assets", "image", f"{base12}.tif?assets=red&assets=nir&asset_as_band=true")
        o.check("tif of band math", "image", f"{base12}.tif?assets=red&assets=nir&asset_as_band=true&expression=(b2-b1)/(b2%2Bb1)")
        o.check("tif, rescaled under a colour map", "image", f"{base12}.tif?assets=red&rescale=0,3000&colormap_name=viridis")
        o.check("tif of three assets that are not bytes", "image", f"{base12}.tif?assets=red&assets=green&assets=blue&asset_as_band=true")
        o.check("bbox, tif under an algorithm", "status", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/64x64.tif?assets=red&algorithm=hillshade")
        # An algorithm's parameters on a mosaic. Contours are drawn as titiler-pgstac draws them;
        # hillshade is shaded another way (see the plan), so only that it is answered is checked.
        o.check("contours with their parameters", "image",
                f"{base12}.png?assets=red&algorithm=contours&algorithm_params=%7B%22increment%22%3A500%2C%22thickness%22%3A2%7D")
        o.check("hillshade with its parameters", "status",
                f"{base12}.png?assets=red&algorithm=hillshade&algorithm_params=%7B%22azimuth%22%3A200%2C%22angle_altitude%22%3A20%7D")
        o.check("bbox, hillshade with its parameters", "status",
                f"/searches/{sid}/bbox/8.40,45.70,8.52,45.78/128x128.png?assets=red&algorithm=hillshade&algorithm_params=%7B%22azimuth%22%3A90%7D")
        # An algorithm over a buffered tile, and over a box drawn in metres: the renderer draws both.
        o.check("tile with a buffer, contours", "image", f"{base12}.png?assets=red&algorithm=contours&buffer=2")
        o.check("tile with half a pixel of buffer, terrarium", "image", f"{base12}.png?assets=red&algorithm=terrarium&buffer=0.5")
        o.check("tif of a tile with a buffer, contours", "status", f"{base12}.tif?assets=red&algorithm=contours&buffer=2")
        o.check("bbox in metres, contours", "image",
                f"/searches/{sid}/bbox/930000,5700000,960000,5730000/200x200.png?coord_crs=epsg:3857&assets=red&algorithm=contours")
        o.check("bbox drawn in metres, terrarium", "image",
                f"/searches/{sid}/bbox/8.40,45.70,8.52,45.78/200x200.png?dst_crs=epsg:3857&assets=red&algorithm=terrarium")
        # normalizedIndex over two assets is rio-tiler's band math of the first two bands.
        ndi = "assets=red&assets=nir&algorithm=normalizedIndex"
        o.check("normalizedIndex over two assets", "image", f"{base12}.png?{ndi}&rescale=-1,1")
        o.check("normalizedIndex over two assets as bands, a colour map", "image", f"{base12}.png?{ndi}&asset_as_band=true&rescale=-1,1&colormap_name=rdylgn")
        o.check("normalizedIndex over two assets, highest", "image", f"{base12}.png?{ndi}&rescale=0,1&pixel_selection=highest")
        o.check("normalizedIndex over two assets as an array", "image", f"{base12}.npy?{ndi}")
        o.check("normalizedIndex over two assets as a GeoTIFF", "image", f"{base12}.tif?{ndi}")
        o.check("bbox, normalizedIndex over two assets as an array", "image", f"/searches/{sid}/bbox/8.40,45.70,8.52,45.78/128x128.npy?{ndi}")
        o.check("bbox, normalizedIndex over two assets", "image", f"/searches/{sid}/bbox/8.40,45.70,8.52,45.78/128x128.png?{ndi}&rescale=-1,1")
        # A tile of a size the renderer does not draw, under an algorithm: drawn as its box.
        o.check("contours, 300 pixels", "image", f"{base12}.png?assets=red&algorithm=contours&tilesize=300")
        o.check("terrarium, 128 pixels", "image", f"{base12}.png?assets=red&algorithm=terrarium&tilesize=128")
        o.check("normalizedIndex over two assets, 384 pixels", "image", f"{base12}.png?{ndi}&rescale=-1,1&tilesize=384")
        o.check("algorithm parameters of the wrong type", "status",
                f"{base12}.png?assets=red&algorithm=hillshade&algorithm_params=%7B%22azimuth%22%3A%22x%22%7D")
        # Band math read with a smoothing resampling: whole numbers stay whole, rescaled is bytes.
        band_math = "assets=red&assets=nir&asset_as_band=true&resampling=bilinear&reproject=bilinear&expression="
        o.check("array of whole band math, bilinear", "image", f"{base12}.npy?{band_math}b1*2")
        o.check("array of band math, bilinear, rescaled", "image", f"{base12}.npy?{band_math}(b2-b1)/(b2%2Bb1)&rescale=-1,1")
        o.check("tif with a buffer", "image", f"{base12}.tif?assets=red&buffer=2")
        o.check("array with a buffer, bilinear", "image", f"{base12}.npy?assets=red&buffer=2&resampling=bilinear&reproject=bilinear")
        o.check("picture with half a pixel of buffer, average", "image", f"{base12}.png?assets=visual&buffer=0.5&resampling=average&reproject=average")
        o.check("array of two assets with a buffer, cubic, 512", "image",
                f"{base12}.npy?assets=red&assets=nir&asset_as_band=true&buffer=3&tilesize=512&resampling=cubic&reproject=cubic")
        o.check("tif by format=", "image", f"{base12}?assets=red&format=tif")
        # A method that gathers a pixel's footprint, far below the raster's coarsest overview:
        # read as a box is. (The reference takes seconds for each of these.)
        low7 = f"/searches/{sid}/tiles/WebMercatorQuad/7/66/45"
        o.check("tile far below the overviews, average", "image", f"{low7}.npy?assets=red&resampling=average&reproject=average")
        o.check("tile far below the overviews, rms, three bands", "image", f"{low7}.png?assets=visual&resampling=rms&reproject=rms")
        # A GeometryCollection: of more members and deeper than the shared parser's limits,
        # and one holding a line beside a polygon.
        def collection_square(k):
            x, y = 8.40 + 0.01 * (k % 10), 45.70 + 0.01 * (k // 10)
            return {"type": "Polygon", "coordinates": [[[x, y], [x + 0.004, y], [x + 0.004, y + 0.004], [x, y + 0.004], [x, y]]]}
        many = {"type": "Feature", "properties": {}, "geometry": {
            "type": "GeometryCollection", "geometries": [collection_square(k) for k in range(100)]}}
        deep = collection_square(0)
        for _ in range(12):
            deep = {"type": "GeometryCollection", "geometries": [deep]}
        deep = {"type": "Feature", "properties": {}, "geometry": deep}
        mixed = {"type": "Feature", "properties": {}, "geometry": {"type": "GeometryCollection", "geometries": [
            collection_square(0), {"type": "LineString", "coordinates": [[8.5, 45.7], [8.6, 45.8]]}]}}
        for name, shape in (("of a hundred polygons", many), ("nested twelve deep", deep), ("of a polygon and a line", mixed)):
            o.check(f"statistics, a collection {name}", "status", f"/searches/{sid}/statistics?assets=red&max_size=128", "POST", shape)
            o.check(f"feature, a collection {name}", "status", f"/searches/{sid}/feature/96x96.npy?assets=red", "POST", shape)
        # 16-bit data asked for as a PNG and as it is: a PNG of 16 bits, which the reader of
        # pictures here takes as it takes any other.
        o.check("png of 16 bits, one band", "image", f"{base12}.png?assets=red")
        o.check("png of 16 bits, one band and no mask", "image", f"{base12}.png?assets=red&return_mask=false")
        o.check("png of 16 bits, three bands", "image", f"{base12}.png?assets=red&assets=green&assets=blue&asset_as_band=true")
        o.check("png of 16 bits, two bands and no mask", "image", f"{base12}.png?assets=red&assets=nir&asset_as_band=true&return_mask=false")
        o.check("png of 16 bits, four bands and no mask", "image",
                f"{base12}.png?assets=red&assets=green&assets=blue&assets=nir&asset_as_band=true&return_mask=false")
        o.check("png of 16 bits, four bands and a mask", "status",
                f"{base12}.png?assets=red&assets=green&assets=blue&assets=nir&asset_as_band=true")
        o.check("png of 16 bits, a bbox", "image", f"/searches/{sid}/bbox/8.4,45.7,8.5,45.75/64x64.png?assets=red")
        # Smoothed pixels on a grid in metres: one reader for all four.
        smoothed_shape = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [8.40, 45.70], [8.52, 45.71], [8.50, 45.78], [8.41, 45.76], [8.40, 45.70]]]}}
        o.check("tile, bilinear, a nodata of the request's own", "image", f"{base12}.npy?assets=red&resampling=bilinear&reproject=bilinear&nodata=0")
        o.check("tile, average, a nodata of the request's own, three bands", "image", f"{base12}.png?assets=visual&resampling=average&reproject=average&nodata=0")
        o.check("tile of 300 pixels, bilinear", "image", f"{base12}.npy?assets=red&resampling=bilinear&reproject=bilinear&tilesize=300")
        o.check("bbox in metres, bilinear", "image",
                f"/searches/{sid}/bbox/930000,5700000,960000,5730000/256x256.npy?assets=red&coord_crs=epsg:3857&resampling=bilinear&reproject=bilinear")
        o.check("bbox drawn in metres, average, a nodata of the request's own", "image",
                f"/searches/{sid}/bbox/8.4,45.7,8.5,45.75/128x128.npy?assets=red&dst_crs=epsg:3857&resampling=average&reproject=average&nodata=0")
        o.check("feature drawn in metres, bilinear", "image",
                f"/searches/{sid}/feature/128x128.npy?assets=red&dst_crs=epsg:3857&resampling=bilinear&reproject=bilinear", "POST", smoothed_shape)
        o.check("statistics on a grid in metres, average", "status",
                f"/searches/{sid}/statistics?assets=red&dst_crs=epsg:3857&resampling=average&reproject=average&max_size=128", "POST", smoothed_shape)
        # An item whose footprint meets the window and whose raster adds no pixel to it.
        # (Its values at this zoom are as far from the reference's as the tile engine's are:
        # the kernel's width, a known difference. That it is drawn is what is checked.)
        o.check("tile, bilinear, a nodata, a raster that misses it", "status",
                f"/searches/{sid}/tiles/WebMercatorQuad/8/132/92.npy?assets=red&resampling=bilinear&reproject=bilinear&nodata=0")
        o.check("bbox drawn in metres, bilinear, a raster that misses it", "image",
                f"/searches/{sid}/bbox/7.0,45.0,9.0,46.5/400x300.npy?assets=red&dst_crs=epsg:3857&resampling=bilinear&reproject=bilinear")
        o.check("bbox, bilinear, a nodata of the request's own", "image",
                f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/128x128.npy?assets=red&resampling=bilinear&reproject=bilinear&nodata=400")
        o.check("bbox, average, a nodata of the request's own", "image",
                f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/128x128.npy?assets=red&resampling=nearest&reproject=average&nodata=0")
        # Paths that do not fit: a 422 where a value cannot be read, and a zero side is worked out.
        for name, path in (
            ("point that is not numbers", f"/searches/{sid}/point/abc,1?assets=red"),
            ("point of three numbers", f"/searches/{sid}/point/8.85,45.55,1?assets=red"),
            ("items at a point that is not numbers", f"/searches/{sid}/point/abc,1/assets"),
            ("bbox in a format nobody has", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0.xyz?assets=red"),
            ("bbox of a size nobody has, in a format nobody has", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/64x64.xyz?assets=red"),
            ("bbox of a size that is not numbers", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/axb.png?assets=red"),
            ("tile in a format nobody has", f"{base12}.xyz?assets=red"),
            ("tile at a zoom that is not a number", f"/searches/{sid}/tiles/WebMercatorQuad/abc/1/1.png?assets=red"),
            ("tile at a row that is not a number", f"/searches/{sid}/tiles/WebMercatorQuad/1/1/abc.png?assets=red"),
        ):
            o.check(name, "status", path)
        o.check("bbox, a width of zero is worked out", "image", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/0x64.png?assets=red&rescale=0,3000")
        o.check("bbox, a height of zero is worked out", "image", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/64x0.npy?assets=red")
        o.check("bbox, tif of one band", "image", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/128x128.tif?assets=red")
        o.check("bbox in metres, tif of one band", "image", f"/searches/{sid}/bbox/912000,5700000,990000,5780000/64x64.tif?assets=red&coord_crs=epsg:3857")
        o.check("webp", "image", f"{base12}.webp?assets=visual")
        o.check("tilesize=512", "image", f"{base12}.png?assets=visual&tilesize=512")
        o.check("tilesize=128", "image", f"{base12}.png?assets=visual&tilesize=128")
        # A buffered tile: more pixels on every side, from the items of the tile itself.
        o.check("buffer=10", "image", f"{base12}.png?assets=visual&buffer=10")
        o.check("a range given upside down", "image", f"{base12}.png?assets=red&rescale=3000,0",
                known="on purpose: numpy's clip with its bounds crossed answers the upper one, so TiTiler paints every pixel 255; cogzilla inverts the ramp")
        o.check("buffer=0.5, one band", "image", f"{base12}.png?assets=red&rescale=0,3000&buffer=0.5")
        o.check("buffer=64 at a low zoom", "image", f"/searches/{sid}/tiles/WebMercatorQuad/8/134/91.png?assets=visual&buffer=64")
        o.check("tilesize=300, one band", "image", f"{base12}.png?assets=red&rescale=0,3000&tilesize=300")
        o.check("tilesize=1024, three assets", "image", f"{base12}.png?assets=red&assets=green&assets=blue&rescale=0,3000&tilesize=1024")
        o.check("resampling=bilinear on a reprojected source", "image", f"{base12}.png?assets=visual&resampling=bilinear")
        o.check("reproject=bilinear", "image", f"{base12}.png?assets=visual&reproject=bilinear")
        o.check("three assets as RGB", "image", f"{base12}.png?assets=red&assets=green&assets=blue&rescale=0,3000")
        o.check("one band with a colormap", "image", f"{base12}.png?assets=nir&rescale=0,5000&colormap_name=viridis")
        # A `nodata` of the request's own stands in the place of the raster's: on this tile
        # the newest item has a wide stretch without data, which then counts as zeros.
        edge = f"/searches/{sid}/tiles/WebMercatorQuad/9/265/186.png?items_limit=1&exitwhenfull=false&skipcovered=false"
        o.check("one band, the raster's own nodata", "image", f"{edge}&assets=red&rescale=0,3000")
        o.check("one band, nodata named", "image", f"{edge}&assets=red&rescale=0,3000&nodata=1000")
        o.check("three bands, nodata named", "image", f"{edge}&assets=visual&nodata=40")
        o.check("two items, nodata named band by band", "image", f"{edge.replace('items_limit=1', 'items_limit=2')}&assets=visual&nodata=40")
        o.check("box, nodata named", "image", f"/searches/{sid}/bbox/6.328125,43.580390855607845,7.03125,44.087585028245165/256x256.png?items_limit=1&exitwhenfull=false&skipcovered=false&assets=visual&nodata=40")
        classes = "%7B%224%22%3A%5B0%2C128%2C0%2C255%5D%2C%225%22%3A%22%23c8c800%22%2C%226%22%3A%5B0%2C0%2C255%2C255%5D%2C%228%22%3A%5B200%2C200%2C200%2C255%5D%2C%229%22%3A%5B255%2C255%2C255%2C255%5D%7D"
        # Neighbouring intervals in near colours: the pixels the two servers take from
        # either side of an edge then weigh what they would in any other picture.
        spans = "%5B%5B%5B0%2C500%5D%2C%5B60%2C70%2C80%2C255%5D%5D%2C%5B%5B500%2C1500%5D%2C%5B110%2C120%2C130%2C255%5D%5D%2C%5B%5B1500%2C20000%5D%2C%5B170%2C180%2C190%2C255%5D%5D%5D"
        o.check("one band with classes of colour", "image", f"{base12}.png?assets=scl&colormap={classes}")
        o.check("one band with classes of colour, format from the pixels", "image", f"{base12}?assets=scl&colormap={classes}")
        o.check("one band with intervals of colour", "image", f"{base12}.png?assets=red&colormap={spans}")
        o.check("three bands with classes of colour", "json", f"{base12}.png?assets=visual&colormap={classes}")
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
        o.check("tileset list as a page", "page", f"/searches/{sid}/tiles?assets=visual&f=html")
        o.check("tileset list in another format", "json", f"/searches/{sid}/tiles?f=png")
        o.check("tileset in another format", "json", f"/searches/{sid}/tiles/WebMercatorQuad?assets=visual&f=npy")
        o.check("tileset as a page", "page", f"/searches/{sid}/tiles/WebMercatorQuad?assets=visual&f=html")
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
        cql_json = "%7B%22op%22%3A%22%3C%22%2C%22args%22%3A%5B%7B%22property%22%3A%22eo%3Acloud_cover%22%7D%2C50%5D%7D"
        o.check("collection info with a JSON filter and no filter-lang", "json", f"{col}/info?filter={cql_json}")
        o.check("collection info with a JSON filter that is not JSON", "status", f"{col}/info?filter=%7Bnot+json")
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
        o.check("collection tileset list as a page", "page", f"{col}/tiles?assets=visual&f=html")
        o.check("collection tileset as a page", "page", f"{col}/tiles/WebMercatorQuad?assets=visual&f=html")
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
        # numexpr beyond arithmetic: comparisons, logic, floor division, where, two arguments.
        o.check("point, a comparison and where", "json",
                f"/searches/{sid}/point/{pt}?assets=red&assets=nir&expression=b1%3E%3Db2;where(b1%3Eb2,b1,b2);(b1%3E1)%26~(b2%3C1)&items_limit=1")
        o.check("point, floor division and functions of two", "json",
                f"/searches/{sid}/point/{pt}?assets=red&assets=nir&expression=b1//7;minimum(b1,b2);arctan2(b1,b2);round(b1/7);log1p(b1)&items_limit=1")
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
        # In Web Mercator: drawn in it, given in it, and given in it but drawn in degrees.
        (west, south), (east, north) = [mercantile.xy(lon + dx, lat + dy) for dx, dy in ((-0.1, -0.05), (0.1, 0.05))]
        metres = f"{west:.1f},{south:.1f},{east:.1f},{north:.1f}"
        o.check("bbox drawn in Web Mercator", "image", f"/searches/{sid}/bbox/{part}.png?assets=visual&max_size=256&dst_crs=epsg:3857")
        o.check("bbox given in Web Mercator", "image", f"/searches/{sid}/bbox/{metres}.png?assets=visual&max_size=256&coord_crs=epsg:3857")
        o.check("bbox given in Web Mercator, drawn in degrees", "image",
                f"/searches/{sid}/bbox/{metres}/300x200.png?assets=visual&coord_crs=epsg:3857&dst_crs=epsg:4326")
        o.check("bbox drawn in a system that is not supported", "status", f"/searches/{sid}/bbox/{part}.png?assets=visual&max_size=64&dst_crs=epsg:32632",
                known="projections other than degrees and Web Mercator are left out for now; cogzilla answers 400")
        o.check("bbox, tif", "image", f"/searches/{sid}/bbox/{part}/300x200.tif?assets=visual")
        o.check("bbox, width only", "image", f"/searches/{sid}/bbox/{part}.png?assets=visual&width=300")
        o.check("bbox, height only", "image", f"/searches/{sid}/bbox/{part}.png?assets=visual&height=90")
        o.check("bbox at the data's own resolution", "image", f"/searches/{sid}/bbox/{part}.png?assets=visual")
        o.check("bbox, reproject=bilinear", "image", f"/searches/{sid}/bbox/{part}.png?assets=visual&max_size=256&reproject=bilinear")
        o.check("bbox, jpg", "image", f"/searches/{sid}/bbox/{part}.jpg?assets=visual&max_size=256")
        o.check("bbox, webp", "image", f"/searches/{sid}/bbox/{part}/256x128.webp?assets=visual")
        o.check("bbox over several items", "image", f"/searches/{sid}/bbox/{wide}.png?assets=visual&max_size=512")
        o.check("bbox, three assets", "image",
                f"/searches/{sid}/bbox/{part}.png?assets=red&assets=green&assets=blue&rescale=0,3000&max_size=256")
        o.check("bbox, one band with classes of colour", "image", f"/searches/{sid}/bbox/{part}.png?assets=scl&max_size=256&colormap={classes}")
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
        dot = {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": triangle["geometry"]["coordinates"][0][0]}}
        o.check("feature, a point", "status", f"/searches/{sid}/feature.png?assets=visual&max_size=64", "POST", dot)
        o.check("feature, a point, exact size", "status", f"/searches/{sid}/feature/64x64.png?assets=visual", "POST", dot)
        # A route asked with a method it does not take, and values FastAPI cannot read.
        o.check("info, posted to", "status", f"/searches/{sid}/info", "POST", triangle)
        o.check("tilesets, posted to", "status", f"/searches/{sid}/tiles", "POST", triangle)
        o.check("a path that is no route, posted to", "status", f"/searches/{sid}/nope", "POST", triangle)
        for bad in ("max_size=abc", "dst_crs=nope", "unscale=maybe"):
            o.check(f"statistics with {bad}", "status", f"/searches/{sid}/statistics?assets=red&max_size=64&{bad}", "POST", triangle)
        for bad in ("unscale=maybe", "return_mask=maybe", "nodata=abc"):
            o.check(f"tile with {bad}", "status", f"{base12}.png?assets=red&{bad}")
        o.check("bbox in a system that is not one", "status", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/32x32.png?assets=red&coord_crs=nope")
        # A feature posted in Web Mercator metres is read as one, and drawn in metres.
        def metres(lon: float, lat: float) -> list[float]:
            return [lon * 20037508.342789244 / 180, math.log(math.tan((90 + lat) * math.pi / 360)) * 6378137]
        kite = [[8.41, 45.71], [8.63, 45.69], [8.57, 45.83], [8.47, 45.79], [8.41, 45.71]]
        in_metres = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[metres(*p) for p in kite]]}}
        o.check("feature posted in metres", "image", f"/searches/{sid}/feature/32x32.npy?assets=red&coord_crs=epsg:3857", "POST", in_metres)
        o.check("feature posted in metres, drawn in degrees", "image",
                f"/searches/{sid}/feature/32x32.npy?assets=red&coord_crs=epsg:3857&dst_crs=epsg:4326", "POST", in_metres)
        # Statistics count the pixels of a grid in `dst_crs`, which is `coord_crs` unless given.
        in_degrees = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [kite]}}
        # normalizedIndex is band math of the first two bands, its values and bins in float32.
        o.check("statistics of normalizedIndex", "zonal",
                f"/searches/{sid}/statistics?assets=red&assets=nir&max_size=64&algorithm=normalizedIndex", "POST", in_degrees)
        o.check("statistics of normalizedIndex, a histogram range", "zonal",
                f"/searches/{sid}/statistics?assets=red&assets=nir&asset_as_band=true&max_size=64&algorithm=normalizedIndex&histogram_range=-1,1&histogram_bins=7", "POST", in_degrees)
        # The other algorithms, applied to the mosaic's bands as titiler-pgstac's post_process does.
        rgb = "assets=red&assets=green&assets=blue&asset_as_band=true"
        for alg, assets in (("grayscale", rgb), ("bitonal", "assets=visual"), ("mean", rgb), ("median", "assets=visual"),
                            ("std", rgb), ("var", "assets=visual"), ("min", rgb), ("max", "assets=visual"), ("sum", rgb),
                            ("cast", "assets=red"), ("floor", "assets=visual"), ("terrarium", "assets=red"), ("terrainrgb", "assets=red")):
            o.check(f"statistics of {alg}", "zonal", f"/searches/{sid}/statistics?{assets}&max_size=64&algorithm={alg}", "POST", in_degrees)
        # hillshade and slope look at a pixel's neighbours; titiler-pgstac answers them only
        # without a buffer (with one, its coverage is the size of the uncropped image).
        no_buffer = "&algorithm_params=%7B%22buffer%22%3A0%7D"
        o.check("statistics of hillshade", "zonal", f"/searches/{sid}/statistics?assets=red&max_size=64&algorithm=hillshade{no_buffer}", "POST", in_degrees)
        o.check("statistics of slope, in metres", "zonal",
                f"/searches/{sid}/statistics?assets=red&max_size=64&algorithm=slope{no_buffer}&dst_crs=epsg:3857", "POST", in_degrees)
        o.check("statistics of contours", "zonal", f"/searches/{sid}/statistics?assets=red&max_size=64&algorithm=contours", "POST", in_degrees)
        o.check("statistics of contours of three bands", "json", f"/searches/{sid}/statistics?assets=visual&max_size=64&algorithm=contours", "POST", in_degrees)
        # pydantic's words, but for the link to its documentation of the version it is.
        o.check("statistics, an algorithm parameter out of range", "status",
                f"/searches/{sid}/statistics?assets=red&max_size=64&algorithm=hillshade&algorithm_params=%7B%22azimuth%22%3A400%7D", "POST", in_degrees)
        o.check("statistics of grayscale after band math", "zonal",
                f"/searches/{sid}/statistics?assets=red&assets=nir&asset_as_band=true&expression=b1*2;b2;b1&max_size=64&algorithm=grayscale", "POST", in_degrees)
        o.check("statistics of a mean after a pixel selection", "zonal",
                f"/searches/{sid}/statistics?assets=visual&max_size=64&algorithm=mean&pixel_selection=mean", "POST", in_degrees)
        o.check("statistics of normalizedIndex on one asset", "zonal",
                f"/searches/{sid}/statistics?assets=visual&max_size=64&algorithm=normalizedIndex", "POST", in_degrees)
        o.check("statistics of where", "zonal",
                f"/searches/{sid}/statistics?assets=red&assets=nir&asset_as_band=true&max_size=64&expression=where(b1%3E1000,1,0)", "POST", in_degrees)
        o.check("statistics of logic and minimum", "zonal",
                f"/searches/{sid}/statistics?assets=red&assets=nir&asset_as_band=true&max_size=64&expression=(b1%3E1000)%26(b2%3E2000);minimum(b1,b2)", "POST", in_degrees)
        o.check("statistics of a feature posted in metres", "zonal", f"/searches/{sid}/statistics?assets=red&max_size=64&coord_crs=epsg:3857", "POST", in_metres)
        o.check("statistics on a grid in metres", "zonal", f"/searches/{sid}/statistics?assets=red&max_size=64&dst_crs=epsg:3857", "POST", in_degrees)
        o.check("statistics of band math on a grid in metres", "zonal",
                f"/searches/{sid}/statistics?assets=red&assets=nir&asset_as_band=true&expression=(b2-b1)/(b2%2Bb1)&max_size=48&dst_crs=epsg:3857", "POST", in_degrees)
        o.check("statistics of a feature posted in metres, on a grid in degrees", "zonal",
                f"/searches/{sid}/statistics?assets=red&max_size=64&coord_crs=epsg:3857&dst_crs=epsg:4326", "POST", in_metres)
        o.check("statistics pass a colour map over", "status", f"/searches/{sid}/statistics?assets=red&colormap_name=nope&max_size=64", "POST", triangle)
        # Arrays keep the data's type and carry the mask as a band of their own.
        o.check("feature as an array", "image", f"/searches/{sid}/feature/200x100.npy?assets=visual", "POST", triangle)
        o.check("feature as an array, 16-bit", "image", f"/searches/{sid}/feature/120x80.npy?assets=red", "POST", triangle)
        abox = f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/120x80.npy"
        o.check("box as an array, five bands of three items", "image",
                f"{abox}?assets=red&assets=green&assets=blue&assets=nir&assets=scl&items_limit=3")
        o.check("box as an array, the wider type first or last", "image", f"{abox}?assets=scl&assets=red")
        o.check("box as an array, two bands rescaled", "image", f"{abox}?assets=red&assets=nir&rescale=0,3000&rescale=0,5000")
        o.check("box as an array, five bands rescaled", "image",
                f"{abox}?assets=red&assets=green&assets=blue&assets=nir&assets=scl&rescale=0,3000")
        o.check("box as an array, a root of float32", "image", f"{abox}?assets=red&expression=sqrt(b1)&unscale=true")
        o.check("box as an array, a division by nothing", "image", f"{abox}?assets=red&expression=b1/0")
        # A tile is an array as a box is: the data's type, and the mask as a band.
        o.check("tile as an array", "image", f"{base12}.npy?assets=red")
        o.check("tile as an array, three bands of bytes", "image", f"{base12}.npy?assets=visual")
        o.check("tile as an array, five assets", "image", f"{base12}.npy?assets=red&assets=green&assets=blue&assets=nir&assets=scl")
        o.check("tile as an array, band math", "image", f"{base12}.npy?assets=red&expression=b1*2")
        o.check("tile as an array, a mean", "image", f"{base12}.npy?assets=red&pixel_selection=mean&items_limit=3")
        o.check("tile as an array, bilinear", "image", f"{base12}.npy?assets=red&resampling=bilinear&reproject=bilinear")
        # The methods that gather a pixel's footprint. `resampling=nearest` is only there to
        # mark the comparison as one of smoothed values.
        for method in ("average", "mode", "rms", "max", "min", "sum", "med", "q1", "q3"):
            o.check(f"tile as an array, reproject={method}", "image", f"{base12}.npy?assets=red&resampling=nearest&reproject={method}")
        for method in ("average", "mode", "gauss", "rms"):
            o.check(f"tile, resampling={method}", "image", f"{base12}.png?assets=red&rescale=0,3000&resampling={method}")
        o.check("bbox, reproject=average", "image", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/256x256.npy?assets=red&resampling=nearest&reproject=average")
        o.check("tile at zoom 9, reproject=sum", "image", f"/searches/{sid}/tiles/WebMercatorQuad/9/268/182.npy?assets=red&resampling=nearest&reproject=sum")
        o.check("tile, a resampling nobody has", "status", f"{base12}.png?assets=red&resampling=nope")
        o.check("bbox, a reproject nobody has", "status", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/64x64.png?assets=red&reproject=nope")
        o.check("tile, reproject=gauss", "status", f"{base12}.png?assets=red&reproject=gauss")
        o.check("tile, resampling=max", "status", f"{base12}.png?assets=red&resampling=max")
        o.check("tile as an array, bilinear, three bands", "image", f"{base12}.npy?assets=visual&resampling=bilinear&reproject=bilinear")
        each = "&skipcovered=false&exitwhenfull=false"
        o.check("tile as an array, bilinear, a mean", "image",
                f"{base12}.npy?assets=red&pixel_selection=mean&items_limit=3&resampling=bilinear&reproject=bilinear{each}")
        o.check("tile as an array, bilinear, a rescaled mean", "image",
                f"{base12}.npy?assets=red&pixel_selection=mean&items_limit=3&rescale=0,3000&resampling=bilinear&reproject=bilinear{each}")
        o.check("tile as an array, bilinear, a count", "image",
                f"{base12}.npy?assets=visual&pixel_selection=count&items_limit=3&resampling=bilinear&reproject=bilinear{each}")
        o.check("tile as an array, bilinear, rescaled", "image",
                f"{base12}.npy?assets=red&rescale=0,3000&resampling=bilinear&reproject=bilinear")
        o.check("box as an array, band math in doubles", "image",
                f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/120x80.npy?assets=red&assets=nir&expression=(b2-b1)/(b2%2Bb1)")
        o.check("feature as an array, band math", "image",
                f"/searches/{sid}/feature/120x80.npy?assets=red&assets=nir&expression=(b2-b1)/(b2%2Bb1)", "POST", triangle,
                known="titiler-pgstac loses a feature's polygon when band math is written as an array or a GeoTIFF: "
                      "every pixel of the box is marked as there, where its own PNG of the same request is cut to the polygon; cogzilla cuts all three")
        o.check("feature as an array of a picture's bytes", "image", f"/searches/{sid}/feature/120x80.npy?assets=red&rescale=0,3000", "POST", triangle)
        o.check("feature, format from the pixels", "image", f"/searches/{sid}/feature?assets=visual&max_size=128", "POST", triangle)
        o.check("feature, format=jpg", "image", f"/searches/{sid}/feature?assets=visual&max_size=128&format=jpg", "POST", triangle)
        o.check("feature without assets", "json", f"/searches/{sid}/feature.png", "POST", triangle)
        # A GeometryCollection of polygons is burned as their union, an overlap included.
        cx, cy = triangle["geometry"]["coordinates"][0][0]
        square = [[[cx - 0.05, cy - 0.05], [cx + 0.05, cy - 0.05], [cx + 0.05, cy + 0.05], [cx - 0.05, cy + 0.05], [cx - 0.05, cy - 0.05]]]
        polygons = {"type": "Feature", "properties": {}, "geometry": {"type": "GeometryCollection", "geometries": [
            triangle["geometry"], {"type": "Polygon", "coordinates": square}]}}
        o.check("feature, a collection of polygons", "image", f"/searches/{sid}/feature/200x150.png?assets=visual", "POST", polygons)
        o.check("statistics of a collection of polygons", "zonal", f"/searches/{sid}/statistics?assets=red&max_size=64", "POST", polygons)
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
        o.check("statistics, nodata named", "zonal", f"{st}?assets=red&max_size=128&nodata=1000", "POST", five)
        o.check("statistics, nodata named, three bands and a mean", "zonal",
                f"{st}?assets=visual&max_size=128&nodata=40&pixel_selection=mean&items_limit=3", "POST", holed)
        o.check("statistics, bin count and range", "zonal",
                f"{st}?assets=red&max_size=64&histogram_bins=4&histogram_range=0,2000", "POST", five)
        o.check("statistics over several items", "zonal", f"{st}?assets=red&max_size=512", "POST", large)
        # A mean of band math is cast back to the type numexpr gave it: whole numbers toward
        # zero, comparisons to true where any item's is.
        two = "assets=red&assets=nir&asset_as_band=true&max_size=128&pixel_selection=mean"
        for label, band_math in (("whole", "b1*2;b2%2Bb1"), ("negative", "b1-b2"), ("a comparison", "b1%3E1000"),
                            ("a fraction", "b1/3"), ("where", "where(b1%3E1000,b1,0)")):
            o.check(f"statistics of a mean of band math, {label}", "zonal", f"{st}?{two}&expression={band_math}", "POST", large)
        o.check("statistics, filtered search", "zonal", f"/searches/{sid_meta}/statistics?assets=red&max_size=64", "POST", five)
        o.check("statistics of a feature collection", "zonal", f"{st}?assets=red&max_size=64", "POST",
                {"type": "FeatureCollection", "features": [five, holed]})
        o.check("collection statistics", "zonal", f"/collections/{collection}/statistics?assets=red&max_size=64", "POST", five)
        o.check("collection statistics, filtered", "zonal",
                f"/collections/{collection}/statistics?assets=red&max_size=64&datetime=2024-12-01T00:00:00Z/..", "POST", five)
        # Lines burned through every pixel they cross, points in the pixel they are in, as
        # GDAL burns them; their coverage is the share of a finer grid they touch.
        def shaped(geometry):
            return {"type": "Feature", "properties": {}, "geometry": geometry}
        line = shaped({"type": "LineString", "coordinates": [[8.41, 45.71], [8.45, 45.77], [8.51, 45.73]]})
        lines = shaped({"type": "MultiLineString", "coordinates": [[[8.42, 45.72], [8.50, 45.76]], [[8.43, 45.75], [8.49, 45.71]]]})
        points = shaped({"type": "MultiPoint", "coordinates": [[8.42, 45.72], [8.50, 45.76], [8.46, 45.74]]})
        held = shaped({"type": "GeometryCollection", "geometries": [line["geometry"]]})
        o.check("statistics of a line", "zonal", f"{st}?assets=red", "POST", line)
        o.check("statistics of a line, in metres", "zonal", f"{st}?assets=red&dst_crs=epsg:3857", "POST", line)
        o.check("statistics of two lines, max_size", "zonal", f"{st}?assets=red&max_size=200", "POST", lines)
        o.check("statistics of points", "zonal", f"{st}?assets=red", "POST", points)
        o.check("statistics of a line in a geometry collection", "zonal", f"{st}?assets=red&max_size=200", "POST", held)
        o.check("feature, a line", "image", f"/searches/{sid}/feature.npy?assets=red&max_size=128", "POST", line)
        o.check("feature, a line, in metres", "image", f"/searches/{sid}/feature.npy?assets=red&max_size=128&dst_crs=epsg:3857", "POST", line)
        o.check("feature, a line, picture", "image", f"/searches/{sid}/feature.png?assets=red&rescale=0,3000&max_size=128", "POST", held)
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
        o.check("tile, a colour formula with saturation", "image",
                f"{base12}.png?assets=visual&color_formula={opt('gamma rgb 3.5 saturation 1.7 sigmoidal rgb 15 0.35')}")
        # The routes that describe a mosaic check its tiles' parameters as the tiles do.
        for route in ("WebMercatorQuad/tilejson.json", "WebMercatorQuad/map.html", "WMTSCapabilities.xml"):
            for bad in ("colormap_name=nope", "rescale=abc", "resampling=nope", "nodata=abc", "unscale=maybe", "algorithm=nope", "items_limit=abc"):
                o.check(f"{route} with {bad}", "status", f"/searches/{sid}/{route}?assets=red&{bad}")
            # ... and take what the tiles take, however it is spelled.
            for good in ("rescale=[0,1000]", "rescale=[[0,1000]]", "rescale=0_0,1_000", "tile_format=tiff", "nodata=%205",
                         "tilesize=512.0", "items_limit=%205", "scan_limit=1_000", "padding=1.0", "buffer=%200.5",
                         "minzoom=5.0", "colormap=null", f"color_formula={opt('gamma RGB')}", "tilesize=512."):
                if route.startswith("WMTS") and good == "tile_format=tiff":
                    continue
                o.check(f"{route} with {good}", "status", f"/searches/{sid}/{route}?assets=red&{good}")
        # A tile's size, padding and buffer are more than zero; the WMTS document offers the tiles' formats.
        for bad in ("tilesize=0", "tilesize=-1", "buffer=0", "buffer=-1", "buffer=nan", "padding=0", "padding=-1"):
            o.check(f"tile with {bad}", "status", f"{base12}.png?assets=red&{bad}")
            o.check(f"tilejson with {bad}", "status", f"/searches/{sid}/WebMercatorQuad/tilejson.json?assets=red&{bad}")
            o.check(f"WMTS with {bad}", "status", f"/searches/{sid}/WMTSCapabilities.xml?assets=red&{bad}")
        for twice in ("buffer=0&buffer=1", "padding=0&padding=1", "buffer=1&buffer=0"):
            o.check(f"tile with {twice}", "status", f"{base12}.png?assets=red&{twice}")
            o.check(f"tilejson with {twice}", "status", f"/searches/{sid}/WebMercatorQuad/tilejson.json?assets=red&{twice}")
        # Numbers as Python reads them.
        for spelt in ("tilesize=%20256", "tilesize=2_56", "tilesize=512.0", "padding=100", "buffer=1_0", "buffer=%2B1",
                      "items_limit=5%20", "scan_limit=1_000", "time_limit=%2010", "tilesize=1e2", "padding=1.5", "buffer=abc"):
            o.check(f"tile with {spelt}", "status", f"{base12}.png?assets=red&{spelt}")
        o.check("tile, a size written with space around it", "image", f"{base12}.png?assets=visual&tilesize=%20512")
        # A point checks what it reads; the list of tilesets reads no zooms; `assets=` is a layer.
        for bad in ("unscale=maybe", "resampling=nope", "nodata=abc", "coord_crs=nope"):
            o.check(f"point with {bad}", "status", f"/searches/{sid}/point/{lon},{lat}?assets=red&{bad}")
        o.check("tilesets with a zoom that is not one", "status", f"/searches/{sid}/tiles?minzoom=abc")
        o.check("WMTS with assets and nothing after it", "status", f"/searches/{sid}/WMTSCapabilities.xml?assets=")
        o.check("bbox, a height written 1.0", "status", f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0.png?assets=red&height=1.0&width=8")
        for fmt in ("tif", "tiff", "pngraw"):
            o.check(f"WMTS with tile_format={fmt}", "status", f"/searches/{sid}/WMTSCapabilities.xml?assets=red&tile_format={fmt}")
        for date in ("20241201", "2024-12-01_00:00:00Z", "", "../.."):
            o.check(f"collection tilejson with datetime={date!r}", "status",
                    f"/collections/{collection}/WebMercatorQuad/tilejson.json?assets=red&datetime={date}")
        o.check("collection tilejson with a date that is not one", "status", f"/collections/{collection}/WebMercatorQuad/tilejson.json?assets=red&datetime=nope")
        o.check("tile, a colour map's name where a map goes", "status", f"{base12}.png?assets=red&rescale=0,3000&colormap=viridis")
        o.check("tile, a colour map nobody has", "status", f"{base12}.png?assets=red&rescale=0,3000&colormap_name=nope")
        # One band: the formula is on the band's byte, and the colour map is looked up after.
        o.check("tile, a colour formula under a colour map", "image",
                f"{base12}.png?assets=red&rescale=0,3000&colormap_name=viridis&color_formula={opt('gamma R 1.5')}")
        smooth = "&resampling=bilinear&reproject=bilinear"
        o.check("tile, bilinear, a colour formula under a colour map", "image",
                f"{base12}.png?assets=red&rescale=0,3000&colormap_name=viridis&color_formula={opt('gamma R 1.5')}{smooth}")
        o.check("tile, bilinear, a colour formula on one band", "image",
                f"{base12}.png?assets=red&rescale=0,3000&color_formula={opt('gamma R 1.5')}{smooth}")
        o.check("bbox, bilinear, a colour formula under a colour map", "image",
                f"/searches/{sid}/bbox/8.2,45.5,8.9,46.0/256x256.png?assets=red&rescale=0,3000&colormap_name=viridis"
                f"&color_formula={opt('gamma R 1.5')}{smooth}")
        o.check("tile, a colour formula under intervals", "image",
                f"{base12}.png?assets=red&rescale=0,3000&color_formula={opt('gamma R 1.5')}&colormap="
                + opt("[[[0,100],[255,0,0,255]],[[100,256],[0,0,255,255]]]"))
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
        # The same on images: each asset read on its own grid over the box.
        bb = f"/searches/{sid}/bbox/{part}"
        for name, q in [
            ("two assets", ".png?assets=red&assets=nir&rescale=0,4000&max_size=256"),
            ("four assets, band math", f".png?{four}&expression={ndvi}&rescale=-1,1&max_size=256"),
            ("four assets, pixel_selection=lastbandhigh", f".png?{four}&rescale=0,4000&max_size=256&pixel_selection=lastbandhigh{every}"),
            ("one asset, pixel_selection=lastbandlow", f".png?assets=visual&max_size=256&pixel_selection=lastbandlow{every}"),
            ("assets on two grids, pixel_selection=mean", f".png?{mixed}&max_size=256&pixel_selection=mean{every}"),
            ("assets on two grids, the data's own resolution", f".png?{mixed}"),
            ("assets on two grids, the coarser first", f".png?assets=swir16&assets=red&asset_as_band=true&expression={opt('b1-b2')}&rescale=-2000,2000"),
            ("assets on two grids, exact size", f"/300x200.png?{mixed}"),
            ("band math on one asset of several", f".png?assets={opt('red|expression=b1*2')}&assets=nir&assets=green&rescale=0,8000&max_size=256"),
        ]:
            o.check(f"planes bbox, {name}", "image", f"{bb}{q}")
        o.check("planes bbox, assets on two grids", "image", f"{bb}.png?{mixed}&max_size=256")
        o.check("planes bbox, four bands", "json", f"{bb}.png?{four}&rescale=0,4000&max_size=64")
        o.check("planes feature, assets on two grids", "image", f"/searches/{sid}/feature/300x200.png?{mixed}", "POST", triangle)
        o.check("planes feature, two assets", "image", f"/searches/{sid}/feature.png?assets=red&assets=nir&rescale=0,4000&max_size=256", "POST", triangle)
        o.check("feature, jpg with a rescale", "image",
                f"/searches/{sid}/feature.jpg?assets=red&assets=green&assets=blue&asset_as_band=true&rescale=0,4000&max_size=256", "POST", triangle)
        o.check("planes collection bbox", "image", f"/collections/{collection}/bbox/{part}/300x200.png?{mixed}")
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
                known="on purpose: GDAL reads a subset of a raster's bands from other pixels than it reads all of them (43% of pixels differ); cogzilla reads the same pixels either way")
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
        for route in ("WebMercatorQuad/tilejson.json", "WebMercatorQuad/map.html", "WMTSCapabilities.xml"):
            for bad in ("tilesize=0", "buffer=0", "padding=-1"):
                o.check(f"item {route} with {bad}", "status", f"{it}/{route}?assets=red&{bad}")
        # What an item's routes answer before the item is read: the method, then the path.
        for name, method, path in (
            ("item info, posted to", "POST", f"{it}/info"),
            ("item tile in a format nobody has", "GET", f"{it}/tiles/WebMercatorQuad/10/529/373.xyz?assets=red"),
            ("item tile at a zoom that is not a number", "GET", f"{it}/tiles/WebMercatorQuad/abc/1/1.png?assets=red"),
            ("item tile on a grid nobody has", "GET", f"{it}/tiles/Nope/1/1/1.png?assets=red"),
            ("item tilejson on a grid nobody has", "GET", f"{it}/Nope/tilejson.json?assets=red"),
            ("item point that is not numbers", "GET", f"{it}/point/abc,1?assets=red"),
            ("item preview in a format nobody has", "GET", f"{it}/preview.xyz?assets=red"),
            ("item preview of a size that is not a number", "GET", f"{it}/preview.png?assets=red&max_size=abc"),
            ("item statistics of a width that is not a number", "GET", f"{it}/statistics?assets=red&width=abc"),
        ):
            o.check(name, "status", path, method, triangle if method == "POST" else None)
        # A side of zero in the path follows the other; both, and `max_size` bounds the picture.
        o.check("item preview, a width of zero in the path", "image", f"{it}/preview/0x64.png?assets=visual")
        o.check("item preview, a height of zero in the path", "image", f"{it}/preview/100x0.png?assets=red&rescale=0,3000")
        o.check("item preview, no size in the path and a max_size", "image", f"{it}/preview/0x0.png?assets=visual&max_size=128")
        o.check("item preview, a side in the path outranks max_size", "image", f"{it}/preview/0x64.png?assets=visual&max_size=32")
        o.check("item preview, a width of zero as an array", "image", f"{it}/preview/0x64.npy?assets=red")
        o.check("item preview as an array, three bands of bytes", "image", f"{it}/preview.npy?assets=visual&max_size=128")
        o.check("item preview as an array, two assets", "image", f"{it}/preview.npy?assets=red&assets=nir&max_size=128")
        o.check("item preview as a GeoTIFF, in the data's type", "image", f"{it}/preview.tif?assets=red&max_size=128")
        o.check("item preview as an array, rescaled", "image", f"{it}/preview.npy?assets=red&rescale=0,3000&max_size=128")
        o.check("item preview as an array, under a colour map", "image", f"{it}/preview.npy?assets=red&rescale=0,3000&colormap_name=viridis&max_size=128")
        o.check("item preview as a GeoTIFF, band math", "image", f"{it}/preview.tif?assets=red&assets=nir&asset_as_band=true&expression=b2-b1&max_size=128")
        # Where an item's preview and posted feature lie: the raster's own system, or the one asked for.
        o.check("item preview, where it lies", "placed", f"{it}/preview.png?assets=visual&max_size=128")
        o.check("item preview at a size, where it lies", "placed", f"{it}/preview/100x80.png?assets=visual")
        o.check("item preview as a GeoTIFF, where it lies", "placed", f"{it}/preview.tif?assets=red&max_size=64")
        inside = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [6.5, 43.3], [6.6, 43.3], [6.6, 43.4], [6.5, 43.4], [6.5, 43.3]]]}}
        o.check("item feature, where it lies", "placed", f"{it}/feature/64x64.png?assets=visual", "POST", inside)
        o.check("item feature drawn in metres, where it lies", "placed",
                f"{it}/feature.png?assets=visual&max_size=64&dst_crs=epsg:3857", "POST", inside)
        o.check("item tile as an array, in the data's type", "image", f"{itile}.npy?assets=red")
        o.check("item tile as an array, three bands of bytes", "image", f"{itile}.npy?assets=visual")
        o.check("item tile as an array, two assets, no mask", "image", f"{itile}.npy?assets=red&assets=nir&return_mask=false")
        o.check("item tile as an array, rescaled", "image", f"{itile}.npy?assets=red&rescale=0,3000")
        o.check("item tile as an array, format in the query", "image", f"{itile}?assets=red&format=npy")
        o.check("item tile as a GeoTIFF, in the data's type", "image", f"{itile}.tif?assets=red")
        o.check("item tile as a GeoTIFF, two assets", "image", f"{itile}.tiff?assets=red&assets=nir")
        o.check("item bbox as an array, in the data's type", "image", f"{it}/bbox/{ibox}/64x64.npy?assets=red")
        o.check("item bbox as an array, at the data's own size", "image", f"{it}/bbox/{ibox}.npy?assets=scl&max_size=512")
        o.check("item bbox as a GeoTIFF, in the data's type", "image", f"{it}/bbox/{ibox}/64x64.tif?assets=red")
        o.check("item bbox, a width of zero in the path", "image", f"{it}/bbox/6.3,43.6,6.5,43.8/0x64.png?assets=visual")
        o.check("item bbox, a height of zero in the path", "image", f"{it}/bbox/6.3,43.6,6.5,43.8/64x0.tif?assets=visual")
        o.check("item bbox, no size in the path and a max_size", "image", f"{it}/bbox/6.3,43.6,6.5,43.8/0x0.png?assets=visual&max_size=200")
        o.check("bbox, no size in the path and a max_size", "image", f"/searches/{sid}/bbox/8.4,45.7,8.5,45.75/0x0.png?assets=visual&max_size=100")
        o.check("bbox, a width of zero in the path and a max_size", "image", f"/searches/{sid}/bbox/8.4,45.7,8.5,45.75/0x64.png?assets=visual&max_size=100")
        # A size of zero is a size not given.
        o.check("item preview, a width of zero", "status", f"{it}/preview.png?assets=red&rescale=0,3000&width=0&max_size=64")
        o.check("bbox, a max_size of zero", "status", f"/searches/{sid}/bbox/8.4,45.7,8.5,45.75.png?assets=red&rescale=0,3000&max_size=0")
        o.check("bbox, a height of zero beside a width", "image", f"/searches/{sid}/bbox/8.4,45.7,8.5,45.75.png?assets=red&rescale=0,3000&width=32&height=0")
        o.check("item in an unknown collection", "json", f"/collections/nope/items/{item_id}/assets")
        o.check("item info", "json", f"{it}/info?assets=visual")
        o.check("item info, two assets", "json", f"{it}/info?assets=red&assets=nir")
        o.check("item info without assets", "json", f"{it}/info")
        o.check("item info with an unknown asset", "json", f"{it}/info?assets=nope")
        o.check("item info.geojson", "json", f"{it}/info.geojson?assets=visual")
        o.check("item renders", "json", f"{it}/renders")
        o.check("item tileset list", "tilesets", f"{it}/tiles?assets=visual")
        o.check("item tileset list as a page", "page", f"{it}/tiles?assets=visual&f=html")
        o.check("item tileset list in another format", "json", f"{it}/tiles?f=png")
        o.check("item tileset as a page", "page", f"{it}/tiles/WebMercatorQuad?assets=visual&f=html")
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
        o.check("item preview", "image", f"{it}/preview.png?assets=visual&max_size=256")
        o.check("item preview, sized", "image", f"{it}/preview/200x200.png?assets=visual")
        o.check("item bbox", "image", f"{it}/bbox/{ibox}/256x256.png?assets=visual")
        # Options on the assets themselves: band math on one, bands chosen on several.
        o.check("item tile, band math on one asset", "image", f"{itile}.png?assets=red%7Cexpression=b1*2&rescale=0,6000")
        o.check("item tile, band math on each of two assets", "image",
                f"{itile}.png?assets=red%7Cexpression=b1/2&assets=nir%7Cexpression=b1/4&rescale=0,3000")
        o.check("item tile, band math on one asset beside a plain one", "image",
                f"{itile}.png?assets=red%7Cexpression=b1*2&assets=green&asset_as_band=true&rescale=0,6000")
        o.check("item tile, band math giving two bands of one asset", "image",
                f"{itile}.png?assets=visual%7Cexpression=b1%2Bb2;b3&rescale=0,400")
        o.check("item tile, bands chosen on two assets", "image",
                f"{itile}.png?assets=visual%7Cbidx=1&assets=red%7Cbidx=1&rescale=0,3000")
        o.check("item tile, bands chosen on one of two assets", "image",
                f"{itile}.png?assets=visual%7Cbidx=3,1&assets=red&rescale=0,3000")
        o.check("item bbox, band math on one asset", "image", f"{it}/bbox/{ibox}/256x256.png?assets=red%7Cexpression=b1*2&rescale=0,6000")
        o.check("item bbox, bands chosen on two assets", "image",
                f"{it}/bbox/{ibox}.png?assets=visual%7Cbidx=1&assets=red%7Cbidx=1&rescale=0,3000")
        o.check("item tile off the grid, band math on one asset", "json", f"{it}/tiles/WebMercatorQuad/1/5/5.png?assets=red%7Cexpression=b1*2")
        o.check("item tile, band math on one asset, a search's options passed over", "image",
                f"{itile}.png?assets=red%7Cexpression=b1*2&rescale=0,6000&pixel_selection=count&items_limit=many&scan_limit=-1")
        o.check("item preview, band math on one asset", "image", f"{it}/preview.png?assets=red%7Cexpression=b1*2&rescale=0,6000")
        o.check("item preview, band math on two assets", "image",
                f"{it}/preview.png?assets=red%7Cexpression=b1*2&assets=nir%7Cexpression=b1&expression=(b2-b1)/(b2%2Bb1)&rescale=-1,1")
        o.check("item preview of two assets, every band of each", "image", f"{it}/preview.npy?assets=visual&assets=red&max_size=64")
        o.check("item preview, band math over the stack of two assets", "image",
                f"{it}/preview.npy?assets=visual&assets=red&max_size=64&expression=b1;b4*2")
        o.check("item preview, an asset's band math past its first band beside another", "image",
                f"{it}/preview.npy?assets=visual%7Cexpression=b1%2Bb2;b3&assets=red%7Cexpression=b1*2&max_size=64")
        o.check("item preview, bands chosen on one of two assets", "image", f"{it}/preview.npy?assets=visual%7Cbidx=3,1&assets=red&max_size=64")
        o.check("item preview, an asset asked for twice", "image",
                f"{it}/preview.npy?assets=red%7Cexpression=b1*2&assets=red%7Cexpression=b1*3&max_size=64")
        o.check("item preview, an asset asked for twice, with band math once", "image",
                f"{it}/preview.npy?assets=red&assets=red%7Cexpression=b1*2&max_size=64")
        o.check("item preview, an asset asked for twice, a band each time", "image",
                f"{it}/preview.npy?assets=visual%7Cbidx=1&assets=visual%7Cbidx=2&max_size=64")
        o.check("item point, band math on each of two assets", "json",
                f"{it}/point/{ilon:.4f},{ilat:.4f}?assets=visual%7Cexpression=b1%2Bb2;b3&assets=red%7Cexpression=b1*2&expression=b1/b3")
        o.check("item statistics, band math on one asset's own bands", "stats", f"{it}/statistics?assets=visual%7Cexpression=b1%2Bb2;b3&max_size=64")
        o.check("item statistics, grayscale over an asset's fractional band math", "stats",
                f"{it}/statistics?assets=visual%7Cexpression=b1/2;b1;b2;b3&expression=b2;b3;b4&algorithm=grayscale&max_size=64")
        o.check("item preview, an algorithm over band math", "image", f"{it}/preview.npy?assets=red&expression=b1/2&algorithm=floor&max_size=64")
        o.check("item statistics, normalizedIndex over band math", "stats",
                f"{it}/statistics?assets=red&assets=nir&expression=b1;b2&algorithm=normalizedIndex&max_size=64")
        o.check("item statistics, grayscale over whole-number band math", "stats",
                f"{it}/statistics?assets=visual&expression=b1*2;b2;b3&algorithm=grayscale&max_size=64")
        o.check("item statistics, band math on an asset beside a plain one", "stats",
                f"{it}/statistics?assets=red%7Cexpression=b1*2&assets=nir&expression=(b1-b2)/(b1%2Bb2)&max_size=64")
        o.check("item point", "json", f"{it}/point/{ilon:.4f},{ilat:.4f}?assets=visual")
        o.check("item point, two assets", "json", f"{it}/point/{ilon:.4f},{ilat:.4f}?assets=red&assets=nir")
        o.check("item point, spaces around the numbers", "json", f"{it}/point/{ilon:.4f}%20,%20{ilat:.4f}?assets=red")
        # An item is one image: its routes have no pixel_selection and pass over any value of it.
        o.check("item tile, a pixel_selection", "status", f"{itile}.png?assets=visual&pixel_selection=mean")
        o.check("item tile, a pixel_selection nobody has", "status", f"{itile}.png?assets=visual&pixel_selection=bogus")
        o.check("item point outside the item", "status", f"{it}/point/0,0?assets=visual",
                known="titiler-pgstac answers 500 for a point outside the item; cogzilla answers 404")
        o.check("item statistics", "stats", f"{it}/statistics?assets=visual&max_size=128")
        o.check("item asset statistics", "stats", f"{it}/asset_statistics?assets=red&max_size=128")
        # Band math on an item's statistics: b1..bn over the stacked bands, described in their names.
        o.check("item statistics, band math over two assets", "stats",
                f"{it}/statistics?assets=nir&assets=red&asset_as_band=true&max_size=64&expression=(b1-b2)/(b1%2Bb2);b1")
        o.check("item statistics, band math on one asset", "stats", f"{it}/statistics?assets=visual&max_size=64&expression=b1-b3;where(b1%3Eb2,1,0)")
        o.check("item statistics, band math by an asset's name", "status",
                f"{it}/statistics?assets=nir&assets=red&asset_as_band=true&max_size=64&expression=(nir-red)/(nir%2Bred)")
        # An algorithm's parameters, checked as TiTiler's models check them where one is
        # applied (a 400 in pydantic's words; not JSON is a 422), and passed over elsewhere.
        far_azimuth = "algorithm=hillshade&algorithm_params=%7B%22azimuth%22%3A400%7D"
        for label, url in (
            ("search tile, an azimuth out of range", f"{base12}.png?assets=red&{far_azimuth}"),
            ("search tile, an azimuth as text", f"{base12}.png?assets=red&algorithm=hillshade&algorithm_params=%7B%22azimuth%22%3A%2230%22%7D"),
            ("search tile, algorithm_params not JSON", f"{base12}.png?assets=red&algorithm=hillshade&algorithm_params=notjson"),
            ("search tile, contours thicker than 10", f"{base12}.png?assets=red&algorithm=contours&algorithm_params=%7B%22thickness%22%3A11%7D"),
            ("search info, an azimuth passed over", f"/searches/{sid}/info?assets=red&{far_azimuth}"),
            ("item tile outside it, an azimuth out of range", f"{it}/tiles/WebMercatorQuad/3/0/0.png?assets=red&{far_azimuth}"),
            ("item preview, an azimuth out of range", f"{it}/preview.png?assets=red&max_size=64&{far_azimuth}"),
            ("item point, an algorithm passed over", f"{it}/point/{ilon},{ilat}?assets=red&{far_azimuth}"),
            ("item asset_statistics, an algorithm passed over", f"{it}/asset_statistics?assets=red&max_size=64&{far_azimuth}"),
        ):
            o.check(label, "status", url)
        # What an item's route reads FastAPI validates first, a 422; what it does not read
        # titiler-pgstac passes over, whatever it holds.
        ibox = f"{it}/bbox/{ilon - 0.05},{ilat - 0.05},{ilon + 0.05},{ilat + 0.05}/32x32.png"
        for label, method, url in (
            ("tile, nodata unreadable", "GET", f"{itile}.png?assets=red&nodata=abc"),
            ("tile, return_mask unreadable", "GET", f"{itile}.png?assets=red&return_mask=maybe"),
            ("tile, a size passed over", "GET", f"{itile}.png?assets=red&max_size=abc&dst_crs=nope"),
            ("box, buffer and crs passed over", "GET", f"{ibox}?assets=red&buffer=abc&crs=nope"),
            ("box, a width passed over beside its own size", "GET", f"{ibox}?assets=red&width=abc"),
            ("box, coord_crs unreadable", "GET", f"{ibox}?assets=red&coord_crs=nope"),
            ("preview, coord_crs passed over", "GET", f"{it}/preview.png?assets=red&max_size=64&coord_crs=nope"),
            ("statistics, return_mask passed over", "GET", f"{it}/statistics?assets=red&max_size=64&return_mask=maybe"),
            ("info, everything passed over", "GET", f"{it}/info?assets=red&width=abc&nodata=abc&dst_crs=nope"),
            ("map, coord_crs passed over", "GET", f"{it}/WebMercatorQuad/map.html?assets=red&coord_crs=nope&width=abc"),
            ("point, coord_crs unreadable", "GET", f"{it}/point/{ilon},{ilat}?assets=red&coord_crs=nope"),
        ):
            o.check(f"item {label}", "status", url, method)
        # An item's statistics of a posted feature are worked out as a mosaic of the item's:
        # its pixels covered by fractions, in `dst_crs`, finer with `cover_scale`.
        iwedge = {"type": "Feature", "id": "w", "properties": {"name": "wedge"}, "geometry": {"type": "Polygon", "coordinates": [[
            [ilon - 0.05, ilat - 0.03], [ilon + 0.05, ilat - 0.03], [ilon, ilat + 0.04], [ilon - 0.05, ilat - 0.03]]]}}
        for label, q in (("", "assets=red&max_size=64"), (", in metres", "assets=red&max_size=64&dst_crs=epsg:3857"),
                         (", cover_scale", "assets=red&max_size=64&cover_scale=4"), (", a width and height", "assets=red&width=50&height=30"),
                         (", hillshade", "assets=red&max_size=64&algorithm=hillshade&algorithm_params=%7B%22buffer%22%3A0%7D"),
                         (", band math", "assets=nir&assets=red&asset_as_band=true&max_size=64&expression=(b1-b2)/(b1%2Bb2)"),
                         (", a pixel_selection passed over", "assets=red&max_size=64&pixel_selection=stdev")):
            o.check(f"item statistics of a feature{label}", "zonal", f"{it}/statistics?{q}", "POST", iwedge)
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
