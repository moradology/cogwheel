#!/usr/bin/env python3
"""
S3 head-to-head: cogzilla-server vs. TiTiler on real, public, anonymous-access COGs.

Sends identical tile requests to both servers and records, per workload:

  * cold   - fresh process, empty caches, every tile requested exactly once
  * miss   - warmed-up process (connections, COG headers, workers) asked for tiles it
             has never served. This is what an origin behind a CDN sees, and is the
             basis for the CPU-per-tile cost figures.
  * hot    - the same tiles requested repeatedly, closed-loop load (oha) at several
             concurrency levels. cogzilla serves these from its decoded-block cache;
             TiTiler/GDAL keeps no pixel data between requests and goes back to S3.
  * s3io   - S3 range requests and bytes fetched per tile (cogzilla /stats, GDAL debug log)
  * memory - RSS/USS of the whole server process tree, sampled throughout
  * cpu    - user+system CPU seconds of the process tree per phase

Raw output goes to benches/results/s3/<run-id>/. Summarise with s3_bench_report.py.

Requirements:
  cargo build --release -p cogzilla-server
  cd benches && uv venv .venv && uv pip install --python .venv/bin/python \
      rasterio rio-tiler titiler.core uvicorn uvloop httptools httpx psutil mercantile pillow numpy
  brew install oha            # or: cargo install oha

Usage:
  benches/.venv/bin/python benches/s3_bench.py                 # everything
  benches/.venv/bin/python benches/s3_bench.py --phases cold   # cold + miss only
  benches/.venv/bin/python benches/s3_bench.py --workloads s2_b08_viridis --concurrency 1,8

All buckets used are public and read with unsigned requests (AWS_NO_SIGN_REQUEST=YES),
so no credentials are used and no requester-pays charges can be incurred.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import mercantile
import psutil

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benches" / "results" / "s3"
COGZILLA_BIN = ROOT / "target" / "release" / "cogzilla-server"

PORTS = {"cogzilla": 18811, "titiler": 18812}

S2 = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/T/EE/2024/9/S2A_13TEE_20240929_0_L2A"
S2_ITEM = (
    "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/"
    "S2A_13TEE_20240929_0_L2A"
)
# WGS84 footprints (lon/lat), already inset so sampled tiles carry real data.
S2_BBOX = (-104.90, 39.75, -103.85, 40.55)
DEM_BBOX = (-105.90, 39.10, -105.10, 39.90)

WORKLOADS = {
    # uint16, DEFLATE predictor 2, 1024px blocks, UTM 13N, us-west-2
    "s2_b08_viridis": {
        "desc": "Sentinel-2 L2A B08, uint16 DEFLATE p2, UTM -> colormapped PNG",
        "path": "/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png",
        "query": f"url={S2}/B08.tif&rescale=0,5000&colormap_name=viridis",
        "bbox": S2_BBOX,
        "zooms": range(9, 15),
    },
    # uint8 x3 pixel-interleaved, DEFLATE predictor 2, UTM 13N, us-west-2
    "s2_tci_jpg": {
        "desc": "Sentinel-2 L2A TCI, 3-band uint8 DEFLATE p2, UTM -> JPEG",
        "path": "/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.jpg",
        "query": f"url={S2}/TCI.tif",
        "bbox": S2_BBOX,
        "zooms": range(9, 15),
    },
    # three separate uint16 COGs resolved through a STAC item
    "s2_stac_rgb": {
        "desc": "Sentinel-2 L2A STAC item, assets red+green+blue (3 COGs) -> RGB PNG",
        "path": "/stac/tiles/WebMercatorQuad/{z}/{x}/{y}.png",
        "query": f"url={S2_ITEM}&assets=red&assets=green&assets=blue&rescale=0,3000",
        "bbox": S2_BBOX,
        "zooms": range(9, 15),
    },
    # float32, LZW predictor 3, 512px blocks, NAD83 geographic, us-west-2
    "dem_3dep_lzw": {
        "desc": "USGS 3DEP 1/3 arc-second DEM, float32 LZW p3, EPSG:4269 -> colormapped PNG",
        "path": "/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png",
        "query": "url=s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n40w106/"
        "USGS_13_n40w106.tif&rescale=1500,4300&colormap_name=terrain",
        "bbox": DEM_BBOX,
        "zooms": range(9, 15),
    },
    # float32, DEFLATE predictor 3, EPSG:4326, eu-central-1 (cross-region redirect)
    "dem_cop30_bilinear": {
        "desc": "Copernicus DEM GLO-30, float32 DEFLATE p3, EPSG:4326, bilinear -> colormapped PNG",
        "path": "/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png",
        "query": "url=s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N39_00_W106_00_DEM/"
        "Copernicus_DSM_COG_10_N39_00_W106_00_DEM.tif"
        "&rescale=1500,4300&colormap_name=terrain&resampling=bilinear",
        "bbox": DEM_BBOX,
        "zooms": range(9, 14),
    },
}

# Shared by both servers: anonymous access only.
COMMON_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "AWS_REGION": "us-west-2",
    "AWS_EC2_METADATA_DISABLED": "true",
}

# TiTiler / GDAL tuning, following TiTiler's "performance tuning" guidance.
TITILER_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",
    "GDAL_CACHEMAX": "200",
    "GDAL_BAND_BLOCK_CACHE": "HASHSET",
    "CPL_VSIL_CURL_CACHE_SIZE": "200000000",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
    "PROJ_NETWORK": "OFF",
    "PYTHONWARNINGS": "ignore",
}

TITILER_APP = '''\
from fastapi import FastAPI
from rio_tiler.io import STACReader
from titiler.core.factory import MultiBaseTilerFactory, TilerFactory

app = FastAPI()
app.include_router(TilerFactory().router, prefix="/cog")
app.include_router(MultiBaseTilerFactory(reader=STACReader).router, prefix="/stac")


@app.get("/healthz")
def healthz():
    return {"ok": True}
'''


def tiles_for(wl: dict, per_zoom: int, seed: int) -> list[tuple[int, int, int]]:
    """Deterministic sample of up to `per_zoom` tiles per zoom intersecting the footprint."""
    rng = random.Random(seed)
    out = []
    for z in wl["zooms"]:
        cands = sorted((t.z, t.x, t.y) for t in mercantile.tiles(*wl["bbox"], zooms=[z]))
        rng.shuffle(cands)
        out.extend(sorted(cands[:per_zoom]))
    return out


def urls_for(server: str, wl: dict, tiles) -> list[str]:
    base = f"http://127.0.0.1:{PORTS[server]}"
    return [f"{base}{wl['path'].format(z=z, x=x, y=y)}?{wl['query']}" for z, x, y in tiles]


# --------------------------------------------------------------------------- servers


class Server:
    def __init__(self, name: str, run_dir: Path, titiler_workers: int, cache_mb: int):
        self.name = name
        self.run_dir = run_dir
        self.titiler_workers = titiler_workers
        self.cache_mb = cache_mb
        self.proc: subprocess.Popen | None = None
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log = None

    def command(self) -> tuple[list[str], dict, Path]:
        if self.name == "cogzilla":
            cmd = [
                str(COGZILLA_BIN),
                "--bind", f"127.0.0.1:{PORTS['cogzilla']}",
                "--cache-mb", str(self.cache_mb),
            ]
            return cmd, {**COMMON_ENV, "RUST_LOG": "warn"}, ROOT
        app_dir = self.run_dir
        (app_dir / "_titiler_s3_app.py").write_text(TITILER_APP)
        cmd = [
            sys.executable, "-m", "uvicorn", "_titiler_s3_app:app",
            "--host", "127.0.0.1", "--port", str(PORTS["titiler"]),
            "--workers", str(self.titiler_workers),
            "--loop", "uvloop", "--http", "httptools",
            "--no-access-log", "--log-level", "warning",
        ]
        return cmd, {**COMMON_ENV, **TITILER_ENV}, app_dir

    def start(self):
        cmd, env, cwd = self.command()
        self._log = open(self.run_dir / f"{self.name}.log", "ab")
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, env={**os.environ, **env}, stdout=self._log, stderr=subprocess.STDOUT
        )
        url = f"http://127.0.0.1:{PORTS[self.name]}/healthz"
        t0 = time.time()
        while time.time() - t0 < 60:
            try:
                if httpx.get(url, timeout=1.0).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError(f"{self.name} did not become ready")
        if self.name == "titiler":
            # every worker must be up before we call the fleet "ready"
            while len(self.procs()) < self.titiler_workers + 1 and time.time() - t0 < 60:
                time.sleep(0.2)
            time.sleep(1.5)
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def procs(self) -> list[psutil.Process]:
        try:
            root = psutil.Process(self.proc.pid)
            return [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            return []

    def snapshot(self, full: bool = False) -> dict:
        rss = uss = cpu = 0.0
        n = 0
        for p in self.procs():
            try:
                ct = p.cpu_times()
                cpu += ct.user + ct.system
                rss += p.memory_info().rss
                n += 1
                if full and uss is not None:
                    try:
                        uss += p.memory_full_info().uss
                    except (psutil.AccessDenied, AttributeError):
                        uss = None  # not permitted on this OS without privileges
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        snap = {"t": time.time(), "rss_mb": rss / 2**20, "cpu_s": cpu, "nproc": n}
        if full and uss is not None:
            snap["uss_mb"] = uss / 2**20
        return snap

    def _sample_loop(self):
        while not self._stop.is_set():
            self.samples.append(self.snapshot())
            self._stop.wait(0.25)

    def rss_window(self, t0: float, t1: float) -> dict:
        vals = sorted(s["rss_mb"] for s in self.samples if t0 <= s["t"] <= t1)
        if not vals:
            return {}
        return {
            "rss_peak_mb": round(vals[-1], 1),
            "rss_median_mb": round(vals[len(vals) // 2], 1),
            "rss_min_mb": round(vals[0], 1),
        }

    def stats(self) -> dict | None:
        if self.name != "cogzilla":
            return None
        try:
            return httpx.get(f"http://127.0.0.1:{PORTS['cogzilla']}/stats", timeout=2).json()
        except Exception:
            return None

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if self.proc:
            for p in reversed(self.procs()):
                try:
                    p.terminate()
                except psutil.NoSuchProcess:
                    pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                for p in self.procs():
                    p.kill()
            self.proc = None
        if self._log:
            self._log.close()
        time.sleep(0.5)


# --------------------------------------------------------------------------- phases


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


async def fetch_once(urls: list[str], concurrency: int) -> dict:
    """Request every URL exactly once, `concurrency` at a time, in list order."""
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    statuses: dict[str, int] = {}
    failures: list[dict] = []
    nbytes = 0
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:

        async def one(u):
            nonlocal nbytes
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.get(u)
                    code = str(r.status_code)
                    nbytes += len(r.content)
                    if r.status_code != 200:
                        failures.append({"url": u, "status": r.status_code, "body": r.text[:300]})
                except Exception as e:  # noqa: BLE001
                    code = type(e).__name__
                lat.append((time.perf_counter() - t0) * 1e3)
                statuses[code] = statuses.get(code, 0) + 1

        t0 = time.perf_counter()
        await asyncio.gather(*(one(u) for u in urls))
        wall = time.perf_counter() - t0
    lat.sort()
    return {
        "n": len(urls),
        "concurrency": concurrency,
        "wall_s": round(wall, 3),
        "tiles_per_s": round(len(urls) / wall, 2),
        "statuses": statuses,
        "failures": failures,
        "bytes": nbytes,
        "first_ms": None,
        "p50_ms": round(pct(lat, 0.50), 1),
        "p90_ms": round(pct(lat, 0.90), 1),
        "p99_ms": round(pct(lat, 0.99), 1),
        "max_ms": round(lat[-1], 1),
        "mean_ms": round(sum(lat) / len(lat), 1),
    }


def run_cold_miss(args, run_dir: Path, tiles: dict) -> list[dict]:
    """
    Fresh process per (rep, workload, server). Tiles are split into two disjoint sets:

      cold: first request alone (header fetch + first tile), then the rest of set A once.
      miss: after set A has been replayed to warm connections, COG headers and every worker
            process, set B (never requested before) is fetched once. No tile repeats, so
            neither server can answer from a previous response's pixel data, but nearby
            tiles may share source blocks exactly as they would in a real map session.
    """
    rows = []

    def record(srv, phase, wname, rep, urls, c0, res, st0):
        c1 = srv.snapshot()
        st1 = srv.stats()
        cpu = c1["cpu_s"] - c0["cpu_s"]
        res.update(
            phase=phase, server=srv.name, workload=wname, rep=rep,
            cpu_s=round(cpu, 3), cpu_s_per_1k=round(cpu / len(urls) * 1000, 2),
            cpu_cores_used=round(cpu / (c1["t"] - c0["t"]), 2),
            **srv.rss_window(c0["t"], c1["t"]),
        )
        if st0 and st1:
            res["s3_gets_per_tile"] = round(
                (st1["remote_range_requests"] - st0["remote_range_requests"]) / len(urls), 2)
            res["s3_kb_per_tile"] = round(
                (st1["remote_bytes_fetched"] - st0["remote_bytes_fetched"]) / len(urls) / 1024, 1)
        rows.append(res)
        first = f"first={res['first_ms']:6.0f}ms " if res.get("first_ms") else " " * 17
        print(
            f"[{phase} r{rep}] {wname:20s} {srv.name:9s} {first}"
            f"p50={res['p50_ms']:6.0f} p90={res['p90_ms']:6.0f} p99={res['p99_ms']:6.0f} "
            f"{res['tiles_per_s']:6.1f} t/s cpu/1k={res['cpu_s_per_1k']:6.1f}s {res['statuses']}",
            flush=True,
        )
        (run_dir / "cold_miss.json").write_text(json.dumps(rows, indent=1))

    for rep in range(args.cold_reps):
        for wname in args.workloads:
            order = ["cogzilla", "titiler"] if rep % 2 == 0 else ["titiler", "cogzilla"]
            set_a = tiles[wname][:: args.split]
            set_b = [t for t in tiles[wname] if t not in set(set_a)]
            for sname in order:
                srv = Server(sname, run_dir, args.titiler_workers, args.cache_mb)
                srv.start()
                try:
                    wl = WORKLOADS[wname]
                    ua, ub = urls_for(sname, wl, set_a), urls_for(sname, wl, set_b)
                    # --- cold
                    st0, c0 = srv.stats(), srv.snapshot()
                    t0 = time.perf_counter()
                    r = httpx.get(ua[0], timeout=120)
                    first_ms = (time.perf_counter() - t0) * 1e3
                    res = asyncio.run(fetch_once(ua[1:], args.cold_concurrency))
                    res["first_ms"] = round(first_ms, 1)
                    res["first_status"] = r.status_code
                    record(srv, "cold", wname, rep, ua, c0, res, st0)
                    # --- warm the process (discarded)
                    for _ in range(args.warm_passes):
                        asyncio.run(fetch_once(ua, 16))
                    time.sleep(0.5)
                    # --- miss
                    st0, c0 = srv.stats(), srv.snapshot()
                    res = asyncio.run(fetch_once(ub, args.cold_concurrency))
                    record(srv, "miss", wname, rep, ub, c0, res, st0)
                finally:
                    srv.stop()
    return rows


S3IO_SNIPPET = r"""
import json, logging, re, sys
from rio_tiler.io import Reader, STACReader
spec = json.load(open(sys.argv[1]))
state = {"gets": 0, "bytes": 0}
pat = re.compile(r"Downloading (\d+)-(\d+)")
class H(logging.Handler):
    def emit(self, rec):
        m = pat.search(rec.getMessage())
        if m:
            state["gets"] += 1
            state["bytes"] += int(m.group(2)) - int(m.group(1)) + 1
for name in ("rasterio._err", "rasterio._env"):
    lg = logging.getLogger(name); lg.setLevel(logging.DEBUG); lg.addHandler(H()); lg.propagate = False
out = {}
for w, job in spec.items():
    def render(t):
        x, y, z = t[1], t[2], t[0]
        if job["stac"]:
            with STACReader(job["url"]) as r:
                r.tile(x, y, z, assets=job["assets"])
        else:
            with Reader(job["url"]) as r:
                r.tile(x, y, z, **job["kw"])
    render(job["tiles"][0])            # warm headers; not counted
    state.update(gets=0, bytes=0)
    for t in job["tiles"][1:]:
        render(t)
    n = len(job["tiles"]) - 1
    out[w] = {"tiles": n, "s3_gets_per_tile": round(state["gets"] / n, 2),
              "s3_kb_per_tile": round(state["bytes"] / n / 1024, 1)}
print(json.dumps(out))
"""


def run_s3io(args, run_dir: Path, tiles: dict) -> dict:
    """Count GDAL's S3 range requests per tile (headers warm) by parsing its debug log."""
    from urllib.parse import parse_qs

    spec = {}
    for wname in args.workloads:
        q = parse_qs(WORKLOADS[wname]["query"])
        kw = {}
        if "resampling" in q:
            kw["reproject_method"] = q["resampling"][0]
        spec[wname] = {
            "stac": WORKLOADS[wname]["path"].startswith("/stac"),
            "url": q["url"][0], "assets": q.get("assets"), "kw": kw,
            "tiles": tiles[wname][:: args.s3io_stride],
        }
    (run_dir / "s3io_spec.json").write_text(json.dumps(spec))
    env = {**os.environ, **COMMON_ENV, **TITILER_ENV, "CPL_DEBUG": "ON"}
    out = subprocess.run([sys.executable, "-c", S3IO_SNIPPET, str(run_dir / "s3io_spec.json")],
                         env=env, capture_output=True, text=True, check=True).stdout
    res = json.loads(out.strip().splitlines()[-1])
    for w, r in res.items():
        print(f"[s3io  ] {w:20s} gdal      {r}", flush=True)
    (run_dir / "s3io_gdal.json").write_text(json.dumps(res, indent=1))
    return res


def oha(urls_file: Path, concurrency: int, *, duration_s: float | None = None, n: int | None = None) -> dict:
    cmd = ["oha", "--no-tui", "--output-format", "json", "-c", str(concurrency),
           "--urls-from-file", "-t", "120s"]
    cmd += ["-z", f"{duration_s}s"] if duration_s else ["-n", str(n)]
    cmd.append(str(urls_file))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def run_hot(args, run_dir: Path, tiles: dict) -> tuple[list[dict], list[dict]]:
    """One long-lived process per server; prime caches, then closed-loop load."""
    rows, mem = [], []
    for sname in args.servers:
        srv = Server(sname, run_dir, args.titiler_workers, args.cache_mb)
        srv.start()
        try:
            time.sleep(2)
            idle = srv.snapshot(full=True)
            mem.append({"server": sname, "stage": "idle", **idle})
            for wname in args.workloads:
                urls = urls_for(sname, WORKLOADS[wname], tiles[wname])
                uf = run_dir / f"urls_{sname}_{wname}.txt"
                uf.write_text("\n".join(urls) + "\n")
                # Prime caches. cogzilla keeps decoded blocks; TiTiler/GDAL keeps only headers.
                t0 = time.time()
                prime = oha(uf, 16, n=len(urls) * (4 if sname == 'cogzilla' else 1))
                print(f"[prime ] {wname:20s} {sname:9s} {time.time() - t0:5.1f}s "
                      f"ok={prime['summary']['successRate']:.3f}", flush=True)
                for c in args.concurrency:
                    time.sleep(1.0)
                    s0 = srv.snapshot()
                    res = oha(uf, c, duration_s=args.duration)
                    s1 = srv.snapshot()
                    codes = res.get("statusCodeDistribution", {})
                    ok = codes.get("200", 0)
                    wall = res["summary"]["total"]
                    lp = res["latencyPercentiles"]
                    cpu = s1["cpu_s"] - s0["cpu_s"]
                    row = {
                        "phase": "hot", "server": sname, "workload": wname, "concurrency": c,
                        "requests_ok": ok, "statuses": codes,
                        "errors": res.get("errorDistribution", {}),
                        "wall_s": round(wall, 2),
                        "rps": round(ok / wall, 1),
                        "p50_ms": round(lp["p50"] * 1e3, 2),
                        "p90_ms": round(lp["p90"] * 1e3, 2),
                        "p99_ms": round(lp["p99"] * 1e3, 2),
                        "mean_ms": round(res["summary"]["average"] * 1e3, 2),
                        "bytes_per_req": res["summary"]["sizePerRequest"],
                        "cpu_s": round(cpu, 2),
                        "cpu_cores_used": round(cpu / (s1["t"] - s0["t"]), 2),
                        "cpu_s_per_1k": round(cpu / ok * 1000, 3) if ok else None,
                        "rps_per_core": round(ok / cpu, 1) if cpu > 0 else None,
                        **srv.rss_window(s0["t"], s1["t"]),
                    }
                    rows.append(row)
                    print(
                        f"[hot   ] {wname:20s} {sname:9s} c={c:<3d} {row['rps']:8.1f} rps "
                        f"p50={row['p50_ms']:7.2f} p90={row['p90_ms']:7.2f} p99={row['p99_ms']:7.2f}ms "
                        f"cores={row['cpu_cores_used']:5.2f} cpu/1k={row['cpu_s_per_1k']}s "
                        f"rss={row.get('rss_peak_mb')}MB",
                        flush=True,
                    )
                    (run_dir / "hot.json").write_text(json.dumps(rows, indent=1))
                mem.append({"server": sname, "stage": f"after_{wname}", **srv.snapshot(full=True)})
            time.sleep(5)
            mem.append({"server": sname, "stage": "steady_after_load", **srv.snapshot(full=True)})
            all_rss = [s["rss_mb"] for s in srv.samples]
            mem.append({"server": sname, "stage": "peak_overall", "rss_mb": max(all_rss),
                        "cogzilla_stats": srv.stats()})
            (run_dir / f"rss_trace_{sname}.json").write_text(json.dumps(srv.samples))
            (run_dir / "memory.json").write_text(json.dumps(mem, indent=1))
        finally:
            srv.stop()
    return rows, mem


def run_parity(args, run_dir: Path, tiles: dict) -> list[dict]:
    """Sanity check that both servers render the same picture for the same request."""
    import numpy as np
    from PIL import Image

    rows = []
    servers = {n: Server(n, run_dir, 2, args.cache_mb) for n in PORTS}
    for s in servers.values():
        s.start()
    try:
        for wname in args.workloads:
            wl = WORKLOADS[wname]
            sample = tiles[wname][:: max(1, len(tiles[wname]) // 6)][:6]
            psnrs, mask_agree = [], []
            for t in sample:
                imgs = {}
                for sname in servers:
                    r = httpx.get(urls_for(sname, wl, [t])[0], timeout=120)
                    r.raise_for_status()
                    imgs[sname] = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"), dtype=np.float64)
                a, b = imgs["cogzilla"], imgs["titiler"]
                valid = (a[..., 3] > 0) & (b[..., 3] > 0)
                mask_agree.append(float(((a[..., 3] > 0) == (b[..., 3] > 0)).mean()))
                if valid.any():
                    mse = ((a[..., :3] - b[..., :3])[valid] ** 2).mean()
                    psnrs.append(99.0 if mse == 0 else float(10 * np.log10(255**2 / mse)))
            row = {"workload": wname, "tiles": len(sample),
                   "psnr_db_min": round(min(psnrs), 1) if psnrs else None,
                   "psnr_db_median": round(sorted(psnrs)[len(psnrs) // 2], 1) if psnrs else None,
                   "mask_agreement_min": round(min(mask_agree), 4)}
            rows.append(row)
            print(f"[parity] {row}", flush=True)
    finally:
        for s in servers.values():
            s.stop()
    (run_dir / "parity.json").write_text(json.dumps(rows, indent=1))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", default="parity,s3io,cold,hot")
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--servers", default="cogzilla,titiler", help="servers for the hot phase")
    ap.add_argument("--tiles-per-zoom", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cold-reps", type=int, default=2)
    ap.add_argument("--cold-concurrency", type=int, default=8)
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--split", type=int, default=4, help="every Nth tile goes to the cold set, the rest to the miss set")
    ap.add_argument("--warm-passes", type=int, default=3)
    ap.add_argument("--s3io-stride", type=int, default=4, help="sample every Nth tile for GDAL S3 I/O counting")
    ap.add_argument("--duration", type=float, default=8.0, help="seconds per hot load step")
    ap.add_argument("--titiler-workers", type=int, default=os.cpu_count())
    ap.add_argument("--cache-mb", type=int, default=512, help="cogzilla --cache-mb")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    args.workloads = [w for w in args.workloads.split(",") if w]
    args.servers = [x for x in args.servers.split(",") if x]
    args.concurrency = [int(c) for c in args.concurrency.split(",")]
    phases = args.phases.split(",")

    if not COGZILLA_BIN.exists():
        sys.exit("build first: cargo build --release -p cogzilla-server")
    if "hot" in phases and not shutil.which("oha"):
        sys.exit("oha not found on PATH (brew install oha)")

    run_dir = RESULTS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tiles = {w: tiles_for(WORKLOADS[w], args.tiles_per_zoom, args.seed) for w in args.workloads}

    import rasterio
    import rio_tiler
    import titiler.core
    import uvicorn

    meta = {
        "run_id": args.run_id,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": {
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "cpu_brand": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                        capture_output=True, text=True).stdout.strip()
            if sys.platform == "darwin" else platform.processor(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        },
        "versions": {
            "cogzilla_git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(),
            "titiler.core": titiler.core.__version__, "rio-tiler": rio_tiler.__version__,
            "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
            "uvicorn": uvicorn.__version__,
            "oha": subprocess.run(["oha", "--version"], capture_output=True, text=True).stdout.strip()
            if shutil.which("oha") else None,
        },
        "config": {
            "common_env": COMMON_ENV, "titiler_env": TITILER_ENV,
            "titiler_workers": args.titiler_workers, "cogzilla_cache_mb": args.cache_mb,
            "cogzilla_cmd": Server("cogzilla", run_dir, 0, args.cache_mb).command()[0],
            "titiler_cmd": Server("titiler", run_dir, args.titiler_workers, args.cache_mb).command()[0],
            "cold_concurrency": args.cold_concurrency, "cold_reps": args.cold_reps,
            "split": args.split, "warm_passes": args.warm_passes, "hot_concurrency": args.concurrency, "hot_duration_s": args.duration,
        },
        "workloads": {
            w: {"desc": WORKLOADS[w]["desc"], "path": WORKLOADS[w]["path"],
                "query": WORKLOADS[w]["query"], "n_tiles": len(tiles[w]),
                "zooms": [WORKLOADS[w]["zooms"][0], WORKLOADS[w]["zooms"][-1]],
                "tiles": tiles[w]}
            for w in args.workloads
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"run dir: {run_dir}", flush=True)
    for w in args.workloads:
        print(f"  {w}: {len(tiles[w])} tiles", flush=True)

    if "parity" in phases:
        run_parity(args, run_dir, tiles)
    if "s3io" in phases:
        run_s3io(args, run_dir, tiles)
    if "cold" in phases:
        run_cold_miss(args, run_dir, tiles)
    if "hot" in phases:
        run_hot(args, run_dir, tiles)
    print(f"done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
