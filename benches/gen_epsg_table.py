#!/usr/bin/env python3
"""
Generate cogzilla's EPSG table and its PROJ reference points.

Walks every projected CRS in PROJ's EPSG database and keeps those cogzilla can
place exactly:

  * the projection method is one cogzilla implements (Transverse Mercator,
    Lambert Conformal Conic, Albers, polar stereographic, Lambert azimuthal
    equal-area, sinusoidal), with conventional east/north axes and the
    Greenwich prime meridian;
  * units are metres, feet or US survey feet;
  * the ellipsoid is WGS84 or GRS80 (a sphere is allowed for the two
    equal-area world projections, as PROJ and GDAL treat it); and
  * PROJ's own transformation from WGS84 differs from applying no datum shift
    by less than 2 m across the CRS's area of use.

Writes:
  crates/cogzilla-proj/src/epsg_table.rs            the table
  crates/cogzilla-proj/tests/data/epsg_reference.csv  code,lon,lat,x,y from PROJ

    benches/.venv/bin/python benches/gen_epsg_table.py
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from pathlib import Path

import pyproj
from pyproj import CRS, Transformer
from pyproj.database import query_crs_info
from pyproj.enums import PJType

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "crates/cogzilla-proj/src/epsg_table.rs"
REFERENCE = ROOT / "crates/cogzilla-proj/tests/data/epsg_reference.csv"

DATUM_TOLERANCE_M = 2.0
UNITS = {"m": (0, 1.0), "ft": (1, 0.3048), "us-ft": (2, 1200.0 / 3937.0)}
ALLOWED_KEYS = {
    "proj", "zone", "south", "lat_0", "lon_0", "lat_1", "lat_2", "lat_ts", "k", "k_0", "x_0", "y_0",
    "ellps", "datum", "a", "rf", "b", "R", "units", "no_defs", "type", "towgs84",
}
# Method ids shared with epsg_table.rs.
TM, LCC2, LCC1, AEA, PS_TS, PS_K, LAEA, SINU = range(8)
METHOD_NAMES = ["Tm", "Lcc2sp", "Lcc1sp", "Albers", "PolarTs", "PolarK", "Laea", "Sinusoidal"]


def ellipsoid_ok(crs: CRS, allow_sphere: bool):
    ell = crs.ellipsoid
    a, rf = ell.semi_major_metre, ell.inverse_flattening or 0.0
    if abs(a - 6378137.0) < 0.5 and abs(rf - 298.25722) < 1e-3:
        return a, rf
    if allow_sphere and rf == 0.0 and 6.35e6 < a < 6.40e6:
        return a, 0.0
    return None


def sample_points(crs: CRS):
    aou = crs.area_of_use
    if aou is None:
        return []
    w, s, e, n = aou.west, aou.south, aou.east, aou.north
    if e < w:  # crosses the antimeridian
        e += 360.0
    pts = []
    for fx, fy in ((0.5, 0.5), (0.1, 0.15), (0.9, 0.85), (0.15, 0.9), (0.85, 0.1)):
        lon = w + fx * (e - w)
        lon = lon - 360.0 if lon > 180.0 else lon
        lat = max(-89.9, min(89.9, s + fy * (n - s)))
        pts.append((lon, lat))
    return pts


def classify(code: int):
    """Return (entry, reference rows) or a string reason for leaving the code out."""
    crs = CRS.from_epsg(code)
    d = crs.to_dict()
    extra = set(d) - ALLOWED_KEYS
    if extra:
        return f"proj4 keys {sorted(extra)}"
    if "towgs84" in d and any(float(v) != 0.0 for v in d["towgs84"]):
        return "non-zero towgs84"
    if crs.prime_meridian.longitude != 0.0:
        return "prime meridian"
    unit = UNITS.get(d.get("units", "m"))
    if unit is None:
        return f"units {d.get('units')}"
    unit_id, unit_m = unit
    proj = d.get("proj")
    ell = ellipsoid_ok(crs, allow_sphere=proj in ("laea", "sinu"))
    if ell is None:
        return "ellipsoid"
    a, rf = ell
    x0, y0 = d.get("x_0", 0.0) / unit_m, d.get("y_0", 0.0) / unit_m
    lat0, lon0 = d.get("lat_0", 0.0), d.get("lon_0", 0.0)
    k = d.get("k", d.get("k_0", 1.0))

    if proj == "utm":
        zone = int(d["zone"])
        method, p = TM, [0.0, zone * 6.0 - 183.0, 0.9996, 500000.0, 10_000_000.0 if "south" in d else 0.0]
    elif proj == "tmerc":
        method, p = TM, [lat0, lon0, k, x0, y0]
    elif proj == "lcc":
        lat1 = d.get("lat_1", lat0)
        lat2 = d.get("lat_2", lat1)
        if lat1 != lat2:
            if k != 1.0:
                return "lcc 2sp with scale factor"
            method, p = LCC2, [lat1, lat2, lat0, lon0, x0, y0]
        elif lat1 == lat0:
            method, p = LCC1, [lat0, lon0, k, x0, y0]
        elif k == 1.0:
            method, p = LCC2, [lat1, lat1, lat0, lon0, x0, y0]
        else:
            return "lcc 1sp with a separate origin and scale"
    elif proj == "aea":
        method, p = AEA, [d["lat_1"], d.get("lat_2", d["lat_1"]), lat0, lon0, x0, y0]
    elif proj == "stere":
        if abs(abs(lat0) - 90.0) > 1e-9:
            return "oblique stereographic"
        lat_ts = d.get("lat_ts")
        if lat_ts is not None and abs(abs(lat_ts) - 90.0) > 1e-9:
            if k != 1.0:
                return "polar stereographic with both lat_ts and k"
            if (lat_ts > 0) != (lat0 > 0):
                return "lat_ts in the opposite hemisphere"
            method, p = PS_TS, [lat_ts, lon0, x0, y0]
        else:
            method, p = PS_K, [1.0 if lat0 > 0 else -1.0, k, lon0, x0, y0]
    elif proj == "laea":
        method, p = LAEA, [lat0, lon0, x0, y0]
    elif proj == "sinu":
        method, p = SINU, [lon0, x0, y0]
    else:
        return f"method {proj}"

    pts = sample_points(crs)
    if not pts:
        return "no area of use"
    lons, lats = [q[0] for q in pts], [q[1] for q in pts]
    geo = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    wgs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    gx, gy = geo.transform(lons, lats)
    wx, wy = wgs.transform(lons, lats)
    rows = []
    for lon, lat, x1, y1, x2, y2 in zip(lons, lats, gx, gy, wx, wy):
        if not all(map(math.isfinite, (x1, y1, x2, y2))):
            continue
        if math.hypot(x1 - x2, y1 - y2) * unit_m > DATUM_TOLERANCE_M:
            return "datum differs from WGS84 by more than 2 m"
        rows.append((code, lon, lat, x1, y1))
    if len(rows) < 3:
        return "too few valid sample points"
    return (code, method, unit_id, a, rf, p + [0.0] * (6 - len(p))), rows


def wgs84_like_geographic():
    """Geographic 2D CRS codes and datum codes that agree with WGS84 within tolerance."""
    gcs, datums = set(), set()
    for info in query_crs_info(auth_name="EPSG", pj_types=PJType.GEOGRAPHIC_2D_CRS, allow_deprecated=True):
        code = int(info.code)
        if code > 65535:
            continue
        try:
            crs = CRS.from_epsg(code)
            if ellipsoid_ok(crs, False) is None or crs.prime_meridian.longitude != 0.0:
                continue
            if any("degree" not in (ax.unit_name or "") for ax in crs.axis_info):
                continue
            pts = sample_points(crs)
            if not pts:
                continue
            t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            xs, ys = t.transform([q[0] for q in pts], [q[1] for q in pts])
            worst = max(
                math.hypot((x - lon) * math.cos(math.radians(lat)), y - lat) * 111_320.0
                for (lon, lat), x, y in zip(pts, xs, ys)
            )
            if not worst < DATUM_TOLERANCE_M:
                continue
            gcs.add(code)
            ids = (crs.datum.to_json_dict().get("id") or {})
            if ids.get("authority") == "EPSG" and int(ids.get("code", 0)) <= 65535:
                datums.add(int(ids["code"]))
        except Exception:  # noqa: BLE001
            continue
    return sorted(gcs), sorted(datums)


def fmt(v: float) -> str:
    r = repr(float(v))
    return r if ("." in r or "e" in r or "inf" in r) else r + ".0"


def main():
    entries, reference, skipped = [], [], Counter()
    infos = query_crs_info(auth_name="EPSG", pj_types=PJType.PROJECTED_CRS, allow_deprecated=True)
    seen = set()
    for info in infos:
        code = int(info.code)
        if code > 65535 or code in (3857, 3785) or code in seen:
            continue
        seen.add(code)
        try:
            res = classify(code)
        except Exception as e:  # noqa: BLE001
            res = f"error {type(e).__name__}"
        if isinstance(res, str):
            skipped[res.split(" [")[0] if res.startswith("proj4") else res] += 1
            continue
        entries.append(res[0])
        reference += res[1]
    entries.sort()
    gcs, datums = wgs84_like_geographic()

    lines = [
        f"// Generated by benches/gen_epsg_table.py from PROJ {pyproj.proj_version_str}. Do not edit by hand.",
        "//",
        "// Every entry is checked against PROJ in tests/epsg_table.rs.",
        "",
        "use crate::epsg::{EpsgDef, Method::*};",
        "",
        f"/// {len(entries)} projected EPSG codes, sorted by code.",
        "#[rustfmt::skip]",
        "pub static EPSG_TABLE: &[EpsgDef] = &[",
    ]
    for code, method, unit_id, a, rf, p in entries:
        ps = ", ".join(fmt(v) for v in p)
        lines.append(
            f"    EpsgDef {{ code: {code}, method: {METHOD_NAMES[method]}, unit: {unit_id}, a: {fmt(a)}, rf: {fmt(rf)}, p: [{ps}] }},"
        )
    lines += ["];", ""]
    for name, doc, vals in (
        ("WGS84_LIKE_GCS", "Geographic CRS codes on a datum within 2 m of WGS84.", gcs),
        ("WGS84_LIKE_DATUM", "Geodetic datum codes within 2 m of WGS84.", datums),
    ):
        lines += [f"/// {doc}", "#[rustfmt::skip]", f"pub static {name}: &[u16] = &["]
        for i in range(0, len(vals), 12):
            lines.append("    " + ", ".join(str(v) for v in vals[i : i + 12]) + ",")
        lines += ["];", ""]
    TABLE.write_text("\n".join(lines))
    REFERENCE.write_text(
        "code,lon,lat,x,y\n" + "".join(f"{c},{lo!r},{la!r},{x!r},{y!r}\n" for c, lo, la, x, y in reference)
    )
    by_method = Counter(METHOD_NAMES[e[1]] for e in entries)
    print(f"{len(entries)} codes kept: {dict(by_method)}")
    print(f"{len(gcs)} geographic CRSs, {len(datums)} datums, {len(reference)} reference points")
    print("left out:")
    for reason, n in skipped.most_common(25):
        print(f"  {n:5d}  {reason}")


if __name__ == "__main__":
    main()
