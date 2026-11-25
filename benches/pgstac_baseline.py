#!/usr/bin/env python3
"""
pgSTAC baseline: where does the time for a titiler-pgstac mosaic tile go?

For a sample of tiles per zoom, against the local reference stack
(benches/pgstac/docker-compose.yml) this measures, per registered search:

  * sql      - `geojsonsearch(...)` called directly with exactly the arguments
               titiler-pgstac sends (tile polygon, search hash, fields, 10000 / 100 /
               5 s limits, exitwhenfull, skipcovered). Median of repeated warm calls.
  * lean     - a prototype of the "lean" path: pgSTAC's own strategy (walk partitions in
               sort order) as plain queries that return only id, footprint and one asset
               href, with pgSTAC's coverage rules (skipcovered / exitwhenfull) applied
               client-side. It is written in Python with exact geometry unions, so its
               time is an upper bound; the point is the id-list check against
               `geojsonsearch` and the order of magnitude.
  * eps      - the same loop with a tolerance instead of exact float equality: how many
               of the items pgSTAC returns add a measurable share of the tile.
  * titiler  - a stock titiler-pgstac tile, first request and repeats. With one worker
               the repeats hit titiler-pgstac's asset-list cache, so they are raster only.
  * cogzilla - cogzilla's existing `/mosaic/tiles` on the same scenes in the same order
               (the asset hrefs `geojsonsearch` returned), first request and repeats.
               This is the raster cost a pgSTAC route in cogzilla would have today.

The headline is the SQL share: sql / (sql + cogzilla warm raster). It says how much of
a cogzilla pgSTAC tile would be spent waiting on Postgres if the SQL were left as is.

Raw records go to benches/results/pgstac/<run-id>/baseline.json, the summary table to
BASELINE.md in the same directory and to stdout.

Requirements:
  docker compose -f benches/pgstac/docker-compose.yml up -d
  benches/.venv/bin/python benches/pgstac_load.py
  cargo build --release -p cogzilla-server
  uv pip install --python benches/.venv/bin/python "pypgstac[psycopg]==0.10.2" \
      shapely httpx mercantile numpy pillow

Usage:
  benches/.venv/bin/python benches/pgstac_baseline.py
  benches/.venv/bin/python benches/pgstac_baseline.py --skip-raster --tiles-per-zoom 40
  benches/.venv/bin/python benches/pgstac_baseline.py --cogzilla-bin /path/to/cogzilla-server

Scene data is read from the public Sentinel-2 COG bucket over plain HTTPS; no
credentials are used.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
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
COGZILLA_BIN = ROOT / "target" / "release" / "cogzilla-server"

PORT_COGZILLA = 18822
TITILER = f"http://127.0.0.1:{os.environ.get('TITILER_PGSTAC_PORT', '18821')}"

# Exactly what titiler-pgstac 3.2 passes to geojsonsearch.
FIELDS = {"include": ["assets", "id", "bbox", "collection"]}
SCAN_LIMIT, ITEMS_LIMIT, TIME_LIMIT = 10000, 100, "5 seconds"


def dsn_from_env() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    # Escaped so a "/" or "@" in either does not break the URL.
    user = quote(os.environ.get("PGUSER", "username"), safe="")
    password = quote(os.environ.get("PGPASSWORD", "password"), safe="")
    host = os.environ.get("PGHOST", "127.0.0.1")
    port = os.environ.get("PGPORT", os.environ.get("PGSTAC_PORT", "5439"))
    db = os.environ.get("PGDATABASE", "postgis")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def parse_zooms(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def med(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def tile_polygon(t: mercantile.Tile) -> dict:
    b = mercantile.bounds(t)
    return {
        "type": "Polygon",
        "coordinates": [[
            [b.west, b.south], [b.west, b.north], [b.east, b.north], [b.east, b.south], [b.west, b.south],
        ]],
    }


def sample_tiles(bbox: list[float], z: int, n: int, rng: random.Random) -> list[mercantile.Tile]:
    # Shrink the bbox so sampled tiles sit inside the loaded data at high zooms.
    w, s, e, nn = bbox
    dx, dy = (e - w) * 0.15, (nn - s) * 0.15
    tiles = list(mercantile.tiles(w + dx, s + dy, e - dx, nn - dy, [z]))
    rng.shuffle(tiles)
    return tiles[:n]


# ----------------------------------------------------------------------------
# SQL: the stock function, and the lean prototype
# ----------------------------------------------------------------------------

def run_geojsonsearch(conn: psycopg.Connection, poly: dict, search_hash: str) -> tuple[float, dict, int]:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT geojsonsearch(%s, %s, %s, %s, %s, %s, %s, %s)::text",
            (json.dumps(poly), search_hash, json.dumps(FIELDS), SCAN_LIMIT, ITEMS_LIMIT, TIME_LIMIT, True, True),
        )
        raw = cur.fetchone()[0]
    fc = json.loads(raw)
    return (time.perf_counter() - t0) * 1e3, fc, len(raw)


def list_partitions(conn: psycopg.Connection, orderby: str) -> list[str]:
    """Leaf partitions in the order the search sorts by, or just `items` if that is not by time."""
    first = orderby.strip().lower()
    if not first.startswith("datetime"):
        return ["items"]
    direction = "ASC" if first.startswith("datetime asc") else "DESC"
    with conn.cursor() as cur:
        cur.execute(f"SELECT partition FROM partition_stats ORDER BY lower(partition_dtrange) {direction}")
        return [r[0] for r in cur.fetchall()] or ["items"]


def run_lean(
    conn: psycopg.Connection, poly: dict, where: str, orderby: str, asset: str, partitions: list[str],
    eps: float = 0.0,
) -> tuple[float, list[str], int, list[float], float]:
    """pgSTAC's strategy without the plpgsql: walk partitions in sort order, one plain query each.

    Each query returns only id, footprint and one asset href. The coverage loop is mirrored
    from geometrysearch. With eps=0 the comparisons are pgSTAC's exact float ones. With eps>0
    an item must add more than eps of the tile's area to be kept, and the tile counts as full
    once all but eps of it is covered. Returns (total ms, ids, rows scanned, share of tile
    gained per kept item, ms spent in the queries alone).
    """
    t0 = time.perf_counter()
    b = poly["coordinates"][0]
    tile = box(b[0][0], b[0][1], b[2][0], b[2][1])
    tile_area = tile.area
    poly_json = json.dumps(poly)
    ids: list[str] = []
    gains: list[float] = []
    scanned = 0
    unioned = None
    prev_area = 0.0
    done = False
    sql_ms = 0.0
    with conn.cursor(binary=True) as cur:
        for part in partitions:
            q0 = time.perf_counter()
            cur.execute(
                "SELECT id, st_asbinary(geometry), content->'assets'->(%s::text)->>'href' "
                f"FROM {part} WHERE {where} "
                "AND st_intersects(geometry, st_setsrid(st_geomfromgeojson(%s::text), 4326)) "
                f"ORDER BY {orderby}",
                (asset, poly_json),
            )
            rows = cur.fetchall()
            sql_ms += (time.perf_counter() - q0) * 1e3
            for item_id, wkb, _href in rows:
                scanned += 1
                if scanned > SCAN_LIMIT:
                    done = True
                    break
                clipped = tile.intersection(shapely.from_wkb(bytes(wkb)))
                unioned = clipped if unioned is None else unioned.union(clipped)
                area = unioned.area
                if area - prev_area <= eps * tile_area:
                    continue
                gains.append((area - prev_area) / tile_area)
                prev_area = area
                ids.append(item_id)
                if len(ids) >= ITEMS_LIMIT or area >= tile_area * (1.0 - eps):
                    done = True
                    break
            if done:
                break
    return (time.perf_counter() - t0) * 1e3, ids, scanned, gains, sql_ms


# ----------------------------------------------------------------------------
# HTTP timing and image comparison
# ----------------------------------------------------------------------------

def timed_get(client: httpx.Client, url: str) -> tuple[float, int, bytes]:
    t0 = time.perf_counter()
    try:
        r = client.get(url)
        return (time.perf_counter() - t0) * 1e3, r.status_code, r.content
    except httpx.HTTPError as e:
        return (time.perf_counter() - t0) * 1e3, 0, str(e).encode()


def compare_pngs(a: bytes, b: bytes) -> dict:
    import numpy as np
    from PIL import Image

    try:
        ia = np.array(Image.open(io.BytesIO(a)).convert("RGBA"), dtype=np.float32)
        ib = np.array(Image.open(io.BytesIO(b)).convert("RGBA"), dtype=np.float32)
    except Exception as e:  # noqa: BLE001 - a broken image is a result, not a crash
        return {"error": str(e)}
    if ia.shape != ib.shape:
        return {"error": f"shape {ia.shape} vs {ib.shape}"}
    ma, mb = ia[:, :, 3] > 0, ib[:, :, 3] > 0
    both = ma & mb
    out = {
        "mask_agree_pct": round(float((ma == mb).mean()) * 100, 2),
        "valid_pct_cogzilla": round(float(ma.mean()) * 100, 2),
        "valid_pct_titiler": round(float(mb.mean()) * 100, 2),
    }
    if both.any():
        d = ia[:, :, :3][both] - ib[:, :, :3][both]
        mse = float((d * d).mean())
        out["mae"] = round(float(abs(d).mean()), 2)
        out["psnr_db"] = 99.0 if mse < 1e-6 else round(20 * math.log10(255 / math.sqrt(mse)), 1)
    return out


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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zooms-sql", default="4-14", help="zooms for the SQL-only measurements")
    ap.add_argument("--zooms-raster", default="8-13", help="zooms that also render tiles")
    ap.add_argument("--tiles-per-zoom", type=int, default=16)
    ap.add_argument("--raster-tiles-per-zoom", type=int, default=6)
    ap.add_argument("--sql-iters", type=int, default=5)
    ap.add_argument("--repeat-iters", type=int, default=3)
    ap.add_argument("--asset", default="visual")
    ap.add_argument("--eps", type=float, default=1e-6, help="coverage tolerance for the eps variant, as a share of tile area")
    ap.add_argument("--cogzilla-label", default=None, help="what the cogzilla binary was built from")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--skip-raster", action="store_true")
    ap.add_argument("--cogzilla-bin", default=str(COGZILLA_BIN))
    ap.add_argument("--cogzilla-cache-mb", type=int, default=2048)
    args = ap.parse_args()

    if not LOADED.exists():
        print("no loaded.json: run benches/pgstac_load.py first", file=sys.stderr)
        return 1
    loaded = json.loads(LOADED.read_text())
    collection = loaded["collection"]
    bbox = [float(v) for v in loaded["bbox"].split(",")]

    searches = {
        "all": {"collections": [collection]},
        "lowcloud": {
            "collections": [collection],
            "filter-lang": "cql2-json",
            "filter": {"op": "<", "args": [{"property": "eo:cloud_cover"}, 20]},
        },
    }

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg.connect(dsn_from_env(), autocommit=True, options="-c search_path=pgstac,public")
    with conn.cursor() as cur:
        cur.execute("SELECT get_version(), count(*) FROM items")
        pgstac_version, n_items = cur.fetchone()
        cur.execute("SELECT count(*) FROM partition_stats")
        n_partitions = cur.fetchone()[0]
    tt_health = httpx.get(f"{TITILER}/healthz", timeout=10).json()

    plans: dict[str, dict] = {}
    for name, body in searches.items():
        r = httpx.post(f"{TITILER}/searches/register", json=body, timeout=30)
        r.raise_for_status()
        search_hash = r.json()["id"]
        with conn.cursor() as cur:
            # The lean path's inputs: pgSTAC's own compiled WHERE clause and ORDER BY.
            cur.execute("SELECT _where, orderby FROM search_query(%s::jsonb)", (json.dumps(body),))
            where, orderby = cur.fetchone()
        plans[name] = {
            "hash": search_hash, "where": where, "orderby": orderby, "body": body,
            "partitions": list_partitions(conn, orderby),
        }
        print(f"search {name}: {search_hash}  WHERE {where.strip()}  ORDER BY {orderby}")

    cz_proc = None
    cz_base = f"http://127.0.0.1:{PORT_COGZILLA}"
    cz_version = None
    if not args.skip_raster:
        cz_proc = subprocess.Popen(
            [args.cogzilla_bin, "--bind", f"127.0.0.1:{PORT_COGZILLA}", "--cache-mb", str(args.cogzilla_cache_mb)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wait_ready(f"{cz_base}/healthz")
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        cz_version = args.cogzilla_label or f"binary {args.cogzilla_bin} (repository HEAD is {head})"

    zooms_sql = parse_zooms(args.zooms_sql)
    zooms_raster = set() if args.skip_raster else set(parse_zooms(args.zooms_raster))
    all_zooms = sorted(set(zooms_sql) | zooms_raster)
    # Each search gets its own tiles, so one search's requests do not warm the next one's caches.
    tiles_by_search = {}
    for k, name in enumerate(plans):
        rng = random.Random(args.seed * 1000 + k)
        tiles_by_search[name] = {z: sample_tiles(bbox, z, args.tiles_per_zoom, rng) for z in all_zooms}

    records: list[dict] = []
    client = httpx.Client(timeout=args.timeout)
    try:
        for name, plan in plans.items():
            for z, tiles in tiles_by_search[name].items():
                for i, t in enumerate(tiles):
                    poly = tile_polygon(t)
                    rec: dict = {"search": name, "z": z, "x": t.x, "y": t.y}
                    do_raster = z in zooms_raster and i < args.raster_tiles_per_zoom
                    tt_url = (
                        f"{TITILER}/searches/{plan['hash']}/tiles/WebMercatorQuad/{z}/{t.x}/{t.y}.png"
                        f"?assets={quote(args.asset)}"
                    )

                    # titiler first, on a tile nothing has asked Postgres about yet.
                    tt_png = None
                    if do_raster and i % 2 == 0:
                        ms, status, tt_png = timed_get(client, tt_url)
                        rec["titiler_first_ms"], rec["titiler_status"] = ms, status

                    sql_ms = []
                    fc, payload = {}, 0
                    for _ in range(args.sql_iters):
                        ms, fc, payload = run_geojsonsearch(conn, poly, plan["hash"])
                        sql_ms.append(ms)
                    feats = fc.get("features") or []
                    ids = [f["id"] for f in feats]
                    rec.update({
                        "sql_first_ms": sql_ms[0],
                        "sql_ms": med(sql_ms[1:]) if len(sql_ms) > 1 else sql_ms[0],
                        "items": len(ids),
                        "sql_payload_bytes": payload,
                    })

                    lean_ms, lean_sql_ms = [], []
                    lean_ids, scanned, gains = [], 0, []
                    for _ in range(args.sql_iters):
                        ms, lean_ids, scanned, gains, q_ms = run_lean(
                            conn, poly, plan["where"], plan["orderby"], args.asset, plan["partitions"]
                        )
                        lean_ms.append(ms)
                        lean_sql_ms.append(q_ms)
                    rec.update({
                        "lean_ms": med(lean_ms[1:]) if len(lean_ms) > 1 else lean_ms[0],
                        "lean_sql_ms": med(lean_sql_ms[1:]) if len(lean_sql_ms) > 1 else lean_sql_ms[0],
                        "lean_scanned": scanned,
                        "lean_items": len(lean_ids),
                        "lean_ids_match": lean_ids == ids,
                    })
                    if lean_ids != ids:
                        only_sql = set(ids) - set(lean_ids)
                        only_lean = set(lean_ids) - set(ids)
                        gain_of = dict(zip(lean_ids, gains))
                        rec["lean_mismatch"] = {
                            "common_prefix": next(
                                (k for k, (a, b) in enumerate(zip(ids, lean_ids)) if a != b),
                                min(len(ids), len(lean_ids)),
                            ),
                            "only_sql": len(only_sql),
                            "only_lean": len(only_lean),
                            "max_gain_only_lean": max((gain_of[i] for i in only_lean), default=0.0),
                            "lean_ids": lean_ids,
                        }
                    # Same loop with a tolerance instead of exact float equality.
                    eps_ms = []
                    eps_ids, eps_scanned = [], 0
                    for _ in range(max(2, args.sql_iters - 2)):
                        ms, eps_ids, eps_scanned, _, _ = run_lean(
                            conn, poly, plan["where"], plan["orderby"], args.asset, plan["partitions"], eps=args.eps
                        )
                        eps_ms.append(ms)
                    rec.update({
                        "eps_ms": med(eps_ms[1:]) if len(eps_ms) > 1 else eps_ms[0],
                        "eps_items": len(eps_ids),
                        "eps_scanned": eps_scanned,
                        "eps_is_subset": set(eps_ids) <= set(ids),
                    })

                    if do_raster and ids:
                        hrefs = [f["assets"][args.asset]["href"] for f in feats if args.asset in f.get("assets", {})]
                        cz_url = (
                            f"{cz_base}/mosaic/tiles/WebMercatorQuad/{z}/{t.x}/{t.y}.png?"
                            + "&".join(f"url={quote(h, safe='')}" for h in hrefs)
                        )
                        ms, status, cz_png = timed_get(client, cz_url)
                        rec["cogzilla_first_ms"], rec["cogzilla_status"] = ms, status
                        if status != 200:
                            rec["cogzilla_error"] = cz_png[:200].decode(errors="replace")
                        if tt_png is None:
                            ms, status, tt_png = timed_get(client, tt_url)
                            rec["titiler_first_ms"], rec["titiler_status"] = ms, status
                        rep = [timed_get(client, tt_url)[0] for _ in range(args.repeat_iters)]
                        rec["titiler_repeat_ms"] = med(rep)
                        rep = [timed_get(client, cz_url)[0] for _ in range(args.repeat_iters + 2)]
                        rec["cogzilla_warm_ms"] = med(rep)
                        if rec.get("titiler_status") == 200 and rec.get("cogzilla_status") == 200:
                            rec["image"] = compare_pngs(cz_png, tt_png)
                            (out_dir / "tiles").mkdir(exist_ok=True)
                            stem = f"{name}_{z}_{t.x}_{t.y}"
                            (out_dir / "tiles" / f"{stem}.titiler.png").write_bytes(tt_png)
                            (out_dir / "tiles" / f"{stem}.cogzilla.png").write_bytes(cz_png)
                    rec["ids"] = ids
                    records.append(rec)
                done = [r for r in records if r["search"] == name and r["z"] == z]
                print(
                    f"  {name} z{z}: {len(done)} tiles, items med {med([r['items'] for r in done]):.0f}, "
                    f"sql {med([r['sql_ms'] for r in done]):.1f} ms, lean query {med([r['lean_sql_ms'] for r in done]):.2f} ms",
                    flush=True,
                )
    finally:
        client.close()
        conn.close()
        if cz_proc is not None:
            cz_proc.terminate()
            cz_proc.wait(timeout=10)

    meta = {
        "run_id": run_id,
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        if platform.system() == "Darwin" else platform.processor(),
        "pgstac": pgstac_version,
        "titiler": tt_health.get("versions", {}),
        "cogzilla": cz_version,
        "cogzilla_bin": args.cogzilla_bin if not args.skip_raster else None,
        "items": n_items,
        "partitions": n_partitions,
        "loaded": loaded,
        "asset": args.asset,
        "args": vars(args),
        "searches": plans,
    }
    (out_dir / "baseline.json").write_text(json.dumps({"meta": meta, "records": records}, indent=1))

    md = summarize(meta, records)
    (out_dir / "BASELINE.md").write_text(md)
    print("\n" + md)
    print(f"raw: {out_dir / 'baseline.json'}")
    return 0


def summarize(meta: dict, records: list[dict]) -> str:
    def f(v: float, nd: int = 1) -> str:
        return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{nd}f}"

    lines = [
        f"# pgSTAC baseline, run {meta['run_id']}",
        "",
        f"- pgSTAC {meta['pgstac']}, {meta['items']:,} items in {meta['partitions']} partitions "
        f"({meta['loaded']['collection']}, bbox {meta['loaded']['bbox']})",
        f"- titiler-pgstac {meta['titiler'].get('titiler.pgstac')} on titiler {meta['titiler'].get('titiler')}, "
        f"GDAL {meta['titiler'].get('gdal')}",
        f"- cogzilla: {meta['cogzilla'] or 'not run'}",
        f"- {meta['cpu']} ({meta['machine']}), asset `{meta['asset']}`",
        "",
    ]
    for name in meta["searches"]:
        rs = [r for r in records if r["search"] == name]
        lines += [
            f"## Search `{name}`",
            "",
            f"`WHERE {meta['searches'][name]['where'].strip()}` `ORDER BY {meta['searches'][name]['orderby']}`",
            "",
            "| z | tiles | items med / max | SQL ms med / p95 | lean query ms med | lean total ms med | "
            "ids match | eps items med / max | titiler first ms | titiler repeat ms | cogzilla first ms | cogzilla warm ms | SQL share |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for z in sorted({r["z"] for r in rs}):
            zs = [r for r in rs if r["z"] == z]
            sql = [r["sql_ms"] for r in zs]
            items = [r["items"] for r in zs]
            match = sum(1 for r in zs if r["lean_ids_match"])
            tf = [r["titiler_first_ms"] for r in zs if r.get("titiler_status") == 200]
            tr = [r["titiler_repeat_ms"] for r in zs if "titiler_repeat_ms" in r and r.get("titiler_status") == 200]
            cf = [r["cogzilla_first_ms"] for r in zs if r.get("cogzilla_status") == 200]
            cw = [r["cogzilla_warm_ms"] for r in zs if "cogzilla_warm_ms" in r and r.get("cogzilla_status") == 200]
            # SQL share is computed per tile on tiles that have both numbers, then medianed.
            shares = [
                r["sql_ms"] / (r["sql_ms"] + r["cogzilla_warm_ms"]) * 100
                for r in zs
                if "cogzilla_warm_ms" in r and r.get("cogzilla_status") == 200
            ]
            lines.append(
                f"| {z} | {len(zs)} | {f(med(items), 0)} / {max(items)} | {f(med(sql))} / {f(pct(sql, 0.95))} | "
                f"{f(med([r['lean_sql_ms'] for r in zs]), 2)} | {f(med([r['lean_ms'] for r in zs]))} | "
                f"{match}/{len(zs)} | "
                f"{f(med([r['eps_items'] for r in zs]), 0)} / {max(r['eps_items'] for r in zs)} | {f(med(tf), 0)} | {f(med(tr), 0)} | {f(med(cf), 0)} | {f(med(cw))} | "
                f"{(f(med(shares), 0) + '%') if shares else '-'} |"
            )
        failed = [r for r in rs if "cogzilla_status" in r and r["cogzilla_status"] != 200]
        tfailed = [r for r in rs if "titiler_status" in r and r["titiler_status"] != 200]
        imgs = [r["image"] for r in rs if "image" in r and "error" not in r["image"]]
        lines.append("")
        if imgs:
            lines.append(
                f"Images ({len(imgs)} tile pairs): mask agreement median "
                f"{f(med([i['mask_agree_pct'] for i in imgs]))}%, "
                f"PSNR median {f(med([i['psnr_db'] for i in imgs if 'psnr_db' in i]))} dB."
            )
        if failed or tfailed:
            lines.append(f"Failures: cogzilla {len(failed)}, titiler {len(tfailed)} (see baseline.json).")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
