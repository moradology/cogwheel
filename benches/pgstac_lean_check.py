#!/usr/bin/env python3
"""Compare cogzilla's lean item lookup with pgSTAC's geojsonsearch, tile by tile.

For a sample of tiles per zoom this asks one cogzilla server for the items of each tile
twice: through `geojsonsearch` (the `/assets` route as it stands) and through the lean
lookup (`item_lookup=lean`). It reports how long each takes and how the lists differ:

  * whether the lean list is `geojsonsearch`'s, or a part of it in the same order;
  * for every item only `geojsonsearch` lists, how much of the tile that item was the
    first to cover, in pixels of a 256-pixel tile, by exact geometry (shapely);
  * for every item only the lean lookup lists, the same (these appear where
    `geojsonsearch` stopped at its item or time limit and the lean lookup went on).

Needs the reference stack of benches/pgstac/docker-compose.yml and a release build:

    docker compose -f benches/pgstac/docker-compose.yml up -d
    cargo build --release -p cogzilla-server
    benches/.venv/bin/python benches/pgstac_lean_check.py --zooms 5-14 --tiles-per-zoom 12
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx
import mercantile
import psycopg
import shapely
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benches" / "results" / "pgstac"
LOADED = ROOT / "benches" / "fixtures" / "pgstac" / "loaded.json"
PORT = 18824

DEFAULT_COLLECTION = "sentinel-2-l2a"
SEARCH_NAMES = ("all", "clear", "oldest")


def searches(collection: str) -> dict[str, dict]:
    return {
        "all": {"collections": [collection]},
        "clear": {"collections": [collection], "query": {"eo:cloud_cover": {"lt": 10}}},
        "oldest": {"collections": [collection], "sortby": [{"field": "datetime", "direction": "asc"}]},
    }


def dsn_from_env() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    # User and password are escaped so a "/" or "@" in either does not break the URL.
    return "postgresql://{}:{}@{}:{}/{}".format(
        quote(os.environ.get("PGUSER", "username"), safe=""),
        quote(os.environ.get("PGPASSWORD", "password"), safe=""),
        os.environ.get("PGHOST", "127.0.0.1"),
        os.environ.get("PGPORT", os.environ.get("PGSTAC_PORT", "5439")),
        os.environ.get("PGDATABASE", "postgis"),
    )


def parse_zooms(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def sample_tiles(bbox: list[float], z: int, n: int, rng: random.Random) -> list[mercantile.Tile]:
    w, s, e, nn = bbox
    dx, dy = (e - w) * 0.15, (nn - s) * 0.15
    tiles = list(mercantile.tiles(w + dx, s + dy, e - dx, nn - dy, [z]))
    rng.shuffle(tiles)
    return tiles[:n]


def wait_ready(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"{url} did not come up")


def timed_ids(client: httpx.Client, url: str, iters: int) -> tuple[float, list[str], int]:
    """Median milliseconds over `iters` requests, the ids, and the status. Each request
    carries a different scan limit so none is answered from the server's item cache."""
    times, ids, status = [], [], 0
    for i in range(iters):
        t0 = time.perf_counter()
        r = client.get(f"{url}&scan_limit={10000 - i}")
        times.append((time.perf_counter() - t0) * 1e3)
        status = r.status_code
        if status != 200:
            return statistics.median(times), [], status
        ids = [f["id"] for f in r.json()]
    # The first request also pays for whatever the server keeps per search.
    return statistics.median(times[1:] if len(times) > 2 else times), ids, status


def first_cover(conn: psycopg.Connection, tile: mercantile.Tile, ids: list[str]) -> dict[str, float]:
    """For a list in pgSTAC's order: how much of the tile each item is the first to cover,
    in pixels of a 256-pixel tile."""
    if not ids:
        return {}
    b = mercantile.bounds(tile)
    outline = box(b.west, b.south, b.east, b.north)
    with conn.cursor(binary=True) as cur:
        cur.execute("SELECT id, st_asbinary(geometry) FROM pgstac.items WHERE id = ANY(%s)", (ids,))
        shapes = {i: shapely.from_wkb(bytes(g)) for i, g in cur.fetchall()}
    pixel = outline.area / (256 * 256)
    union, before, out = None, 0.0, {}
    for i in ids:
        clipped = outline.intersection(shapes[i])
        union = clipped if union is None else union.union(clipped)
        out[i] = (union.area - before) / pixel
        before = union.area
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zooms", default="5-14")
    ap.add_argument("--tiles-per-zoom", type=int, default=12)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--searches", default="all,clear", help=f"comma-separated, from: {', '.join(SEARCH_NAMES)}")
    ap.add_argument("--asset", default="visual")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cogzilla-bin", default=str(ROOT / "target" / "release" / "cogzilla-server"))
    args = ap.parse_args()

    dsn = dsn_from_env()
    if not LOADED.exists():
        print(f"{LOADED} is missing: load the catalog with benches/pgstac_load.py first", file=sys.stderr)
        return 2
    loaded = json.loads(LOADED.read_text())
    bbox = [float(v) for v in loaded["bbox"].split(",")]
    # The collection the loader put in the database, as it recorded it.
    bodies = searches(loaded.get("collection", DEFAULT_COLLECTION))
    env = dict(os.environ, DATABASE_URL=dsn, COGZILLA_PGSTAC_LEAN="on")
    proc = subprocess.Popen(
        [args.cogzilla_bin, "--bind", f"127.0.0.1:{PORT}"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{PORT}"
    records: list[dict] = []
    try:
        wait_ready(f"{base}/healthz")
        rng = random.Random(args.seed)
        tiles = {z: sample_tiles(bbox, z, args.tiles_per_zoom, rng) for z in parse_zooms(args.zooms)}
        with httpx.Client(timeout=120) as client, psycopg.connect(dsn) as conn:
            version = conn.execute("SELECT pgstac.get_version()").fetchone()[0]
            for name in args.searches.split(","):
                r = client.post(f"{base}/searches/register", json=bodies[name])
                r.raise_for_status()
                sid = r.json()["id"]
                for z, zt in tiles.items():
                    for t in zt:
                        url = f"{base}/searches/{sid}/tiles/WebMercatorQuad/{t.z}/{t.x}/{t.y}/assets?assets={args.asset}"
                        ref_ms, ref_ids, ref_status = timed_ids(client, url, args.iters)
                        lean_ms, lean_ids, lean_status = timed_ids(client, url + "&item_lookup=lean", args.iters)
                        rec = {
                            "search": name, "z": z, "x": t.x, "y": t.y,
                            "ref_ms": ref_ms, "lean_ms": lean_ms,
                            "ref_status": ref_status, "lean_status": lean_status,
                            "ref_items": len(ref_ids), "lean_items": len(lean_ids),
                            "same": ref_ids == lean_ids,
                        }
                        if ref_status == 200 and lean_status == 200 and ref_ids != lean_ids:
                            kept = set(lean_ids)
                            rec["in_order"] = [i for i in ref_ids if i in kept] == [i for i in lean_ids if i in set(ref_ids)]
                            ref_cover = first_cover(conn, t, ref_ids)
                            lean_cover = first_cover(conn, t, lean_ids)
                            rec["only_ref"] = sorted((ref_cover[i] for i in ref_ids if i not in kept), reverse=True)
                            rec["only_lean"] = sorted(
                                (lean_cover[i] for i in lean_ids if i not in set(ref_ids)), reverse=True
                            )
                        records.append(rec)
                    done = [r for r in records if r["search"] == name and r["z"] == z]
                    print(
                        f"{name} z{z}: geojsonsearch {statistics.median(r['ref_ms'] for r in done):.1f} ms, "
                        f"lean {statistics.median(r['lean_ms'] for r in done):.2f} ms, "
                        f"{sum(r['same'] for r in done)}/{len(done)} lists the same",
                        flush=True,
                    )
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    raw = RESULTS / f"lean-{stamp}.json"
    raw.write_text(json.dumps({"pgstac": version, "args": vars(args), "records": records}, indent=1))

    med = lambda vs: statistics.median(vs) if vs else float("nan")
    print()
    print("| search | z | tiles | items (median) | geojsonsearch ms | lean ms | same list | part of it | "
          "largest dropped, px | lean-only items |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in args.searches.split(","):
        for z in tiles:
            zs = [r for r in records if r["search"] == name and r["z"] == z]
            ok = [r for r in zs if r["ref_status"] == 200 and r["lean_status"] == 200]
            part = [r for r in ok if not r["same"] and not r.get("only_lean") and r.get("in_order")]
            dropped = [v for r in ok for v in r.get("only_ref", [])]
            print(
                f"| {name} | {z} | {len(zs)} | {med([r['ref_items'] for r in ok]):.0f} | "
                f"{med([r['ref_ms'] for r in zs]):.1f} | {med([r['lean_ms'] for r in zs]):.2f} | "
                f"{sum(r['same'] for r in ok)} | {len(part)} | "
                f"{max(dropped):.3g} | " if dropped else
                f"| {name} | {z} | {len(zs)} | {med([r['ref_items'] for r in ok]):.0f} | "
                f"{med([r['ref_ms'] for r in zs]):.1f} | {med([r['lean_ms'] for r in zs]):.2f} | "
                f"{sum(r['same'] for r in ok)} | {len(part)} | - | ",
                end="",
            )
            print(f"{sum(len(r.get('only_lean', [])) for r in ok)} |")
    failed = [r for r in records if r["lean_status"] != 200 and r["ref_status"] == 200]
    if failed:
        print(f"\n{len(failed)} tiles the lean lookup failed on and geojsonsearch did not", file=sys.stderr)
    print(f"\nraw: {raw}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
