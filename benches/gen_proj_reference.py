#!/usr/bin/env python3
"""
Regenerate crates/cogzilla-proj/tests/data/proj_reference_data.rs from PROJ.

Each case pairs the GeoKeys cogzilla would read from a file with the PROJ
definition of the same CRS, and a set of lon/lat points. PROJ (through
rasterio) supplies the projected coordinates the Rust test compares against.
The source CRS is the projection's own geographic CRS, so the comparison is of
projection math alone and no datum shift enters.

    benches/.venv/bin/python benches/gen_proj_reference.py
"""

import math
from pathlib import Path

from rasterio.warp import transform

OUT = Path(__file__).resolve().parent.parent / "crates/cogzilla-proj/tests/data/proj_reference_data.rs"
GRS80 = "+proj=longlat +ellps=GRS80 +no_defs"
WGS84 = "EPSG:4326"


def grid(lons, lats):
    return [(lo, la) for lo in lons for la in lats]


def epsg(code):
    return f"GeoKeys {{ model_type: Some(1), projected_cs_type: Some({code}), ..Default::default() }}"


def user(ct, unit=None, gcs=None, **keys):
    """A user-defined projected CRS, with the keys GDAL writes for it."""
    fields = ", ".join(f"{k}: Some({v!r})" for k, v in keys.items())
    units = f"proj_linear_units: Some({unit}), " if unit else ""
    datum = (
        "semi_major_axis: Some(6378137.0), inv_flattening: Some(298.257222101), "
        if gcs is None
        else f"geographic_type: Some({gcs}), "
    )
    return (
        "GeoKeys { model_type: Some(1), projected_cs_type: Some(32767), "
        f"proj_coord_trans: Some({ct}), {units}{datum}{fields}, ..Default::default() }}"
    )


# (name, Rust GeoKeys, PROJ target, PROJ source, points)
CASES = [
    ("EPSG:32633 UTM 33N, to 9 degrees past the central meridian", epsg(32633), "EPSG:32633", WGS84,
     grid([6.0, 9.0, 12.0, 13.5, 15.0, 17.9, 18.0, 21.0, 24.0], [0.0, 0.5, 30.0, 45.0, 60.0, 80.0, 84.0])),
    ("EPSG:32755 UTM 55S", epsg(32755), "EPSG:32755", WGS84,
     grid([144.0, 145.5, 147.0, 149.9], [-80.0, -42.0, -10.0, -0.001])),
    ("EPSG:32601 UTM 1N across the antimeridian", epsg(32601), "EPSG:32601", WGS84,
     grid([179.0, 179.99, -180.0, -179.5, -177.0, -174.0], [10.0, 52.0, 70.0])),
    ("EPSG:32660 UTM 60N across the antimeridian", epsg(32660), "EPSG:32660", WGS84,
     grid([174.0, 177.0, 179.99, 180.0, -179.99, -179.0], [10.0, 52.0, 70.0])),
    ("EPSG:26915 NAD83 UTM 15N", epsg(26915), "EPSG:26915", "EPSG:4269",
     grid([-96.0, -93.0, -90.01], [25.0, 45.0, 60.0])),
    ("EPSG:5070 CONUS Albers", epsg(5070), "EPSG:5070", "EPSG:4269",
     grid([-160.0, -124.7, -96.0, -80.0, -66.9], [19.0, 24.5, 37.0, 49.4, 71.0])),
    ("EPSG:3310 California Albers", epsg(3310), "EPSG:3310", "EPSG:4269",
     grid([-124.4, -120.0, -114.1], [32.5, 37.0, 42.0])),
    ("EPSG:2154 Lambert-93", epsg(2154), "EPSG:2154", "EPSG:4171",
     grid([-5.0, 3.0, 9.6], [41.3, 46.5, 51.1])),
    ("EPSG:3413 NSIDC north polar", epsg(3413), "EPSG:3413", WGS84,
     grid([-180.0, -135.0, -45.0, 0.0, 45.0, 135.0, 179.0], [40.0, 60.0, 70.0, 85.0, 89.99])),
    ("EPSG:3031 Antarctic polar", epsg(3031), "EPSG:3031", WGS84,
     grid([-180.0, -90.0, 0.0, 60.0, 166.7], [-40.0, -60.0, -71.0, -85.0, -89.99])),
    ("custom TM with a latitude of origin",
     user(1, nat_origin_lat=49.0, nat_origin_long=-2.0, scale_at_nat_origin=0.9996,
          false_easting=400000.0, false_northing=-100000.0),
     "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996 +x_0=400000 +y_0=-100000 +ellps=GRS80 +units=m", GRS80,
     grid([-8.0, -2.0, 1.8], [49.9, 53.0, 60.8])),
    ("custom LCC 2SP written with false-origin keys",
     user(8, std_parallel_1=49.0, std_parallel_2=77.0, false_origin_lat=49.0, false_origin_long=-95.0,
          false_origin_easting=0.0, false_origin_northing=0.0),
     "+proj=lcc +lat_1=49 +lat_2=77 +lat_0=49 +lon_0=-95 +x_0=0 +y_0=0 +ellps=GRS80 +units=m", GRS80,
     grid([-141.0, -95.0, -52.0], [42.0, 55.0, 83.0])),
    ("custom LCC 1SP with a scale factor",
     user(9, nat_origin_lat=46.8, nat_origin_long=2.3, scale_at_nat_origin=0.9998,
          false_easting=600000.0, false_northing=200000.0),
     "+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=2.3 +k_0=0.9998 +x_0=600000 +y_0=200000 +ellps=GRS80 +units=m", GRS80,
     grid([-4.0, 2.3, 8.0], [42.0, 46.8, 51.0])),
    ("custom LCC 2SP in US survey feet",
     user(8, unit=9003, std_parallel_1=38.4, std_parallel_2=37.1, false_origin_lat=36.5,
          false_origin_long=-120.5, false_origin_easting=6561666.667, false_origin_northing=1640416.667),
     "+proj=lcc +lat_1=38.4 +lat_2=37.1 +lat_0=36.5 +lon_0=-120.5 +x_0=2000000.0001016 +y_0=500000.0001016 "
     "+ellps=GRS80 +units=us-ft", GRS80,
     grid([-123.0, -120.5, -118.0], [36.5, 37.5, 38.5])),
    ("custom Albers in the southern hemisphere",
     user(11, std_parallel_1=-18.0, std_parallel_2=-36.0, nat_origin_lat=0.0, nat_origin_long=132.0,
          false_easting=0.0, false_northing=0.0),
     "+proj=aea +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=132 +x_0=0 +y_0=0 +ellps=GRS80 +units=m", GRS80,
     grid([113.0, 132.0, 153.6], [-43.6, -25.0, -10.0])),
    ("custom Albers whose central meridian is near the antimeridian",
     user(11, std_parallel_1=55.0, std_parallel_2=65.0, nat_origin_lat=50.0, nat_origin_long=-154.0,
          false_easting=400000.0, false_northing=0.0),
     "+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +x_0=400000 +y_0=0 +ellps=GRS80 +units=m", GRS80,
     grid([172.0, 179.9, -179.9, -154.0, -130.0], [51.0, 64.0, 71.0])),
    ("custom polar stereographic with a rotated central meridian",
     user(15, gcs=4326, nat_origin_lat=60.0, straight_vert_pole_long=30.0, scale_at_nat_origin=1.0,
          false_easting=0.0, false_northing=0.0),
     "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=30 +x_0=0 +y_0=0 +ellps=WGS84 +units=m", WGS84,
     grid([-150.0, -45.0, 30.0, 100.0], [50.0, 80.0, 89.0])),
    ("custom south polar stereographic",
     user(15, gcs=4326, nat_origin_lat=-71.0, straight_vert_pole_long=140.0, scale_at_nat_origin=1.0,
          false_easting=0.0, false_northing=0.0),
     "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=140 +x_0=0 +y_0=0 +ellps=WGS84 +units=m", WGS84,
     grid([-150.0, -45.0, 140.0, 166.7], [-50.0, -77.8, -89.0])),
    ("custom polar stereographic with the scale factor at the pole (UPS north)",
     user(15, gcs=4326, nat_origin_lat=90.0, straight_vert_pole_long=0.0, scale_at_nat_origin=0.994,
          false_easting=2000000.0, false_northing=2000000.0),
     "+proj=stere +lat_0=90 +lon_0=0 +k_0=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84 +units=m", WGS84,
     grid([-150.0, -45.0, 20.0, 100.0], [60.0, 84.0, 89.0])),
]


def main():
    out = [
        "// Generated from PROJ by benches/gen_proj_reference.py. Do not edit by hand.",
        "pub struct Case {",
        "    pub name: &'static str,",
        "    pub keys: GeoKeys,",
        "    /// (lon, lat, x, y)",
        "    pub points: &'static [(f64, f64, f64, f64)],",
        "}",
        "",
        "pub fn cases() -> Vec<Case> {",
        "    vec![",
    ]
    total = 0
    for name, keys, dst, src, pts in CASES:
        xs, ys = transform(src, dst, [p[0] for p in pts], [p[1] for p in pts])
        rows = []
        for (lon, lat), x, y in zip(pts, xs, ys):
            assert math.isfinite(x) and math.isfinite(y), (name, lon, lat)
            rows.append(f"                ({lon!r}, {lat!r}, {x!r}, {y!r}),")
            total += 1
        out += [
            "        Case {",
            f'            name: "{name}",',
            f"            keys: {keys},",
            "            points: &[",
            *rows,
            "            ],",
            "        },",
        ]
    out += ["    ]", "}", ""]
    OUT.write_text("\n".join(out))
    print(f"wrote {len(CASES)} cases, {total} points to {OUT}")


if __name__ == "__main__":
    main()
