#!/usr/bin/env python3
"""
Load a real STAC collection into the local pgSTAC reference database.

Pages a public STAC API `/search` for one collection over a bbox and date range,
caches the raw items on disk (so reloads do not hit the API again), and ingests the
collection and items with pypgstac. The default is a year of Sentinel-2 L2A over the
western Alps from Earth Search: public COGs in UTM, dense revisits, tilted swath
footprints. That is the shape of catalog that makes per-tile search expensive.

Requirements:
  docker compose -f benches/pgstac/docker-compose.yml up -d
  uv pip install --python benches/.venv/bin/python "pypgstac[psycopg]==0.10.2" httpx

Usage:
  benches/.venv/bin/python benches/pgstac_load.py
  benches/.venv/bin/python benches/pgstac_load.py --max-items 50000 --bbox 2,42,14,49
  benches/.venv/bin/python benches/pgstac_load.py --partition-trunc none

Only public, anonymous endpoints are used and nothing about this machine is sent to them.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "benches" / "fixtures" / "pgstac"

DEFAULT_API = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"
DEFAULT_BBOX = "5.0,43.5,12.0,48.0"
DEFAULT_DATETIME = "2024-01-01T00:00:00Z/2024-12-31T23:59:59Z"


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


def normalize_api(api: str) -> str:
    """The API URL with scheme and host lowercased and no trailing slash."""
    u = urlsplit(api.strip())
    return u._replace(scheme=u.scheme.lower(), netloc=u.netloc.lower(), path=u.path.rstrip("/")).geturl()


def cache_paths(api: str, collection: str, bbox: str, dt: str, max_items: int) -> tuple[Path, Path]:
    tag = f"{collection}_{bbox}_{dt}_{max_items}".replace("/", "_").replace(":", "").replace(",", "_")
    # Items from another API must not be taken for this one's. The default API keeps
    # the name without a digest, so caches written before the digest existed still hit.
    if normalize_api(api) != normalize_api(DEFAULT_API):
        tag += "_api-" + hashlib.sha256(normalize_api(api).encode()).hexdigest()[:12]
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{tag}.collection.json", CACHE / f"{tag}.items.ndjson.gz"


def fetch_with_retry(client: httpx.Client, method: str, url: str, **kw) -> httpx.Response:
    delay = 1.0
    for attempt in range(6):
        try:
            r = client.request(method, url, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == 5:
                raise
            print(f"  retry {attempt + 1} after {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 20.0)
    raise RuntimeError("unreachable")


def download(api: str, collection: str, bbox: str, dt: str, max_items: int, page: int) -> tuple[Path, Path]:
    coll_path, items_path = cache_paths(api, collection, bbox, dt, max_items)
    if coll_path.exists() and items_path.exists():
        print(f"using cached items: {items_path}")
        return coll_path, items_path

    headers = {"User-Agent": "stac-bench/1.0", "Accept": "application/geo+json, application/json"}
    with httpx.Client(timeout=120.0, headers=headers, follow_redirects=True) as client:
        coll = fetch_with_retry(client, "GET", f"{api}/collections/{collection}").json()
        coll_path.write_text(json.dumps(coll))

        body = {
            "collections": [collection],
            "bbox": [float(v) for v in bbox.split(",")],
            "datetime": dt,
            "limit": page,
        }
        url, method, n, t0 = f"{api}/search", "POST", 0, time.time()
        tmp = items_path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", compresslevel=5) as out:
            while url and n < max_items:
                r = fetch_with_retry(client, method, url, json=body if method == "POST" else None)
                fc = r.json()
                feats = fc.get("features", [])
                if not feats:
                    break
                for f in feats:
                    if n >= max_items:
                        break
                    out.write(json.dumps(f, separators=(",", ":")))
                    out.write("\n")
                    n += 1
                nxt = next((l for l in fc.get("links", []) if l.get("rel") == "next"), None)
                if not nxt:
                    break
                url = nxt["href"]
                method = nxt.get("method", "GET").upper()
                if method == "POST":
                    body = nxt.get("body", body)
                print(f"  {n} items ({time.time() - t0:.0f}s)", flush=True)
        tmp.rename(items_path)
        print(f"downloaded {n} items -> {items_path}")
    return coll_path, items_path


def iter_items(items_path: Path, collection: str):
    with gzip.open(items_path, "rt") as f:
        for line in f:
            item = json.loads(line)
            item["collection"] = collection
            # API-relative links are meaningless once the item lives in another catalog.
            item["links"] = []
            yield item


def load(dsn: str, coll_path: Path, items_path: Path, partition_trunc: str) -> dict:
    from pypgstac.db import PgstacDB
    from pypgstac.load import Loader, Methods

    coll = json.loads(coll_path.read_text())
    coll["links"] = []
    collection_id = coll["id"]

    with PgstacDB(dsn=dsn) as db:
        version = db.version
        loader = Loader(db=db)
        loader.load_collections(iter([coll]), insert_mode=Methods.upsert)
        if partition_trunc != "none":
            # Must be set before the first item lands; changing it later repartitions.
            db.query_one(
                "UPDATE pgstac.collections SET partition_trunc = %s WHERE id = %s AND "
                "partition_trunc IS DISTINCT FROM %s RETURNING id",
                (partition_trunc, collection_id, partition_trunc),
            )
        t0 = time.time()
        loader.load_items(iter_items(items_path, collection_id), insert_mode=Methods.upsert)
        load_s = time.time() - t0

        n_items = db.query_one("SELECT count(*) FROM pgstac.items WHERE collection = %s", (collection_id,))
        n_parts = db.query_one(
            "SELECT count(*) FROM pgstac.partition_stats WHERE collection = %s", (collection_id,)
        )
        with db.connect() as conn:
            conn.autocommit = True
            conn.execute("ANALYZE pgstac.items")
        size = db.query_one(
            "SELECT pg_size_pretty(sum(pg_total_relation_size(c.oid))) FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'pgstac' AND c.relname LIKE '_items_%%' AND c.relkind = 'r'"
        )
    return {
        "pgstac_version": version,
        "collection": collection_id,
        "items": n_items,
        "leaf_partitions": n_parts,
        "partition_trunc": partition_trunc,
        "items_size": size,
        "load_seconds": round(load_s, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--bbox", default=DEFAULT_BBOX, help="minlon,minlat,maxlon,maxlat")
    ap.add_argument("--datetime", default=DEFAULT_DATETIME)
    ap.add_argument("--max-items", type=int, default=20000)
    ap.add_argument("--page", type=int, default=250)
    ap.add_argument("--partition-trunc", choices=["none", "year", "month"], default="month")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    coll_path, items_path = download(
        args.api, args.collection, args.bbox, args.datetime, args.max_items, args.page
    )
    if args.download_only:
        return 0
    summary = load(dsn_from_env(), coll_path, items_path, args.partition_trunc)
    summary.update({"api": args.api, "bbox": args.bbox, "datetime": args.datetime})
    (CACHE / "loaded.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
