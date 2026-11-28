//! Unit tests for `cogzilla_proj::crs` and `cogzilla_proj::plan`.

use super::*;

#[test]
fn adaptive_low_zoom_matches_exact_projection_within_0_1_px() {
    let cases: &[(u16, Affine2D, u32, u32, TileCoord)] = &[
        (
            32610,
            Affine2D { a: 30.0, b: 0.0, c: 500_000.0, d: 0.0, e: -30.0, f: 4_250_000.0 },
            4096,
            4096,
            TileCoord { z: 5, x: 5, y: 12 },
        ),
        (
            32610,
            Affine2D { a: 30.0, b: 0.0, c: 500_000.0, d: 0.0, e: -30.0, f: 4_250_000.0 },
            4096,
            4096,
            TileCoord { z: 8, x: 40, y: 98 },
        ),
        (
            5070,
            Affine2D { a: 1000.0, b: 0.0, c: -2_000_000.0, d: 0.0, e: -1000.0, f: 3_000_000.0 },
            4000,
            3000,
            TileCoord { z: 4, x: 3, y: 6 },
        ),
        (
            3413,
            Affine2D { a: 1000.0, b: 0.0, c: -1_000_000.0, d: 0.0, e: -1000.0, f: 1_000_000.0 },
            2000,
            2000,
            TileCoord { z: 3, x: 3, y: 1 },
        ),
    ];

    for &(epsg, p2c, w, h, tile) in cases {
        let gk = GeoKeys {
            projected_cs_type: Some(epsg),
            ..Default::default()
        };
        let crs = CrsModel::from_geo_keys(&gk).unwrap();
        let c2p = p2c.invert().unwrap();
        let plan = CoordPlan::<256>::build(&crs, &c2p, None, w, h, tile)
            .unwrap_or_else(|| panic!("expected non-empty plan for EPSG:{epsg} z={}", tile.z));
        let CoordPlan::Exact2D { u, v, .. } = plan else {
            panic!("expected Exact2D for z <= 8");
        };

        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
        let step_x = (max_x - min_x) / 256.0;
        let step_y = (max_y - min_y) / 256.0;
        let fw = w as f32;
        let fh = h as f32;
        let mut max_err = 0.0f32;
        let mut checked = 0usize;

        for py in 0..256 {
            let my = max_y - (py as f64 + 0.5) * step_y;
            for px in 0..256 {
                let mx = min_x + (px as f64 + 0.5) * step_x;
                let (sx, sy) = crs.from_3857(mx, my);
                if !sx.is_finite() || !sy.is_finite() {
                    continue;
                }
                let (col, row) = c2p.forward(sx, sy);
                let eu = (col - 0.5) as f32;
                let ev = (row - 0.5) as f32;
                if eu >= 0.0 && eu < fw && ev >= 0.0 && ev < fh {
                    let au = u[py * 256 + px];
                    let av = v[py * 256 + px];
                    assert!(
                        au.is_finite() && av.is_finite(),
                        "EPSG:{epsg} z={} pixel ({px},{py}) missed in-bounds ({eu},{ev})",
                        tile.z
                    );
                    let err = (au - eu).abs().max((av - ev).abs());
                    max_err = max_err.max(err);
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no in-bounds pixels checked for EPSG:{epsg}");
        assert!(
            max_err <= 0.1,
            "EPSG:{epsg} z={} max projection error {max_err} px > 0.1 px",
            tile.z
        );
    }
}
fn keys(f: impl FnOnce(&mut GeoKeys)) -> GeoKeys {
    let mut gk = GeoKeys::default();
    f(&mut gk);
    gk
}

#[test]
fn refuses_what_it_cannot_place() {
    let refused: Vec<(&str, GeoKeys)> = vec![
        ("an EPSG code outside the list", keys(|k| k.projected_cs_type = Some(29999))),
        ("NAD27 CONUS Albers (not in table)", keys(|k| k.projected_cs_type = Some(5069))),
        (
            "an unknown code even with parameter keys beside it",
            keys(|k| {
                k.projected_cs_type = Some(29999);
                k.proj_coord_trans = Some(1);
                k.nat_origin_long = Some(-123.0);
            }),
        ),
        (
            "unsupported non-WGS84 geographic datum (NTF Paris)",
            keys(|k| {
                k.model_type = Some(2);
                k.geographic_type = Some(4275);
            }),
        ),
        (
            "a user-defined projection on a WGS84 geographic CRS that is not implemented",
            keys(|k| {
                k.model_type = Some(1);
                k.projected_cs_type = Some(32767);
                k.proj_coord_trans = Some(22); // Polyconic
                k.geographic_type = Some(4326);
            }),
        ),
        (
            "a user-defined projection on the Clarke 1866 ellipsoid",
            keys(|k| {
                k.projected_cs_type = Some(32767);
                k.proj_coord_trans = Some(1);
                k.nat_origin_long = Some(-123.0);
                k.semi_major_axis = Some(6_378_206.4);
                k.inv_flattening = Some(294.978_698_2);
            }),
        ),
        (
            "linear units other than metres or feet",
            keys(|k| {
                k.projected_cs_type = Some(32767);
                k.proj_coord_trans = Some(1);
                k.nat_origin_long = Some(0.0);
                k.proj_linear_units = Some(9036); // kilometres
            }),
        ),
        (
            "Albers with no standard parallels",
            keys(|k| {
                k.projected_cs_type = Some(32767);
                k.proj_coord_trans = Some(11);
                k.nat_origin_long = Some(-96.0);
                k.nat_origin_lat = Some(23.0);
            }),
        ),
        (
            "angular units in grads",
            keys(|k| {
                k.model_type = Some(2);
                k.geographic_type = Some(4326);
                k.geog_angular_units = Some(9105);
            }),
        ),
        (
            "a sphere under a projection that is not defined on one",
            keys(|k| {
                k.projected_cs_type = Some(32767);
                k.proj_coord_trans = Some(1);
                k.nat_origin_long = Some(0.0);
                k.semi_major_axis = Some(6_371_007.181);
                k.semi_minor_axis = Some(6_371_007.181);
            }),
        ),
        (
            "non-Greenwich prime meridian code (Rome)",
            keys(|k| {
                k.model_type = Some(2);
                k.geographic_type = Some(4326);
                k.prime_meridian = Some(8906);
            }),
        ),
        (
            "non-Greenwich prime meridian longitude",
            keys(|k| {
                k.model_type = Some(2);
                k.geographic_type = Some(4326);
                k.prime_meridian = Some(32767);
                k.prime_meridian_long = Some(12.452333);
            }),
        ),
        (
            "EPSG:32610 with conflicting foot linear unit override",
            keys(|k| {
                k.projected_cs_type = Some(32610);
                k.proj_linear_units = Some(9002);
            }),
        ),
        ("no CRS keys at all", GeoKeys::default()),
    ];
    for (what, gk) in refused {
        assert!(CrsModel::from_geo_keys(&gk).is_err(), "{what} should be refused");
    }
}

#[test]
fn accepts_geographic_crs_on_wgs84_like_datums() {
    for gcs in [4326, 4269, 4258, 4283, 7844] {
        let gk = keys(|k| {
            k.model_type = Some(2);
            k.geographic_type = Some(gcs);
        });
        assert!(matches!(CrsModel::from_geo_keys(&gk), Ok(CrsModel::Geographic)), "EPSG:{gcs}");
    }
}

#[test]
fn coord_plan_high_res_low_zoom_does_not_overflow_i32() {
    // 1 cm/px EPSG:3857 COG near mx = -13.6e6 queried at z=0 produces max_u > 3e9 (> i32::MAX).
    let pixel_to_crs = Affine2D {
        a: 0.01,
        b: 0.0,
        c: -13_600_000.0,
        d: 0.0,
        e: -0.01,
        f: 4_500_000.0,
    };
    let crs_to_pixel = pixel_to_crs.invert().unwrap();
    let plan = CoordPlan::<256>::build(
        &CrsModel::Epsg3857,
        &crs_to_pixel,
        None,
        1024,
        1024,
        TileCoord { z: 0, x: 0, y: 0 },
    );
    assert!(plan.is_some(), "high-res COG at z=0 must not overflow i32 and drop tile");
}

#[test]
fn wgs84_bounds_does_not_expand_narrow_mercator_or_geographic_strips() {
    // Narrow EPSG:3857 strip touching (0°, 85.0511°)
    let merc_aff = Affine2D {
        a: 1000.0,
        b: 0.0,
        c: -100_000.0,
        d: 0.0,
        e: -1000.0,
        f: 20_037_508.342_789_244,
    };
    let mb = CrsModel::Epsg3857.wgs84_bounds(&merc_aff, None, 200, 200);
    assert!(mb[0] > -2.0 && mb[2] < 2.0, "unexpected mercator bounds expansion: {mb:?}");

    // Narrow Geographic strip touching (0°, 90°)
    let geo_aff = Affine2D {
        a: 0.1,
        b: 0.0,
        c: -5.0,
        d: 0.0,
        e: -0.1,
        f: 90.0,
    };
    let gb = CrsModel::Geographic.wgs84_bounds(&geo_aff, None, 100, 100);
    assert!(gb[0] >= -5.1 && gb[2] <= 5.1, "unexpected geographic bounds expansion: {gb:?}");
}

#[test]
fn refuses_degenerate_and_non_finite_user_defined_geokeys() {
    let cases = [
        (
            "LCC 2SP symmetric parallels (n == 0)",
            keys(|k| {
                k.projected_cs_type = Some(32767);
                k.proj_coord_trans = Some(8);
                k.std_parallel_1 = Some(30.0);
                k.std_parallel_2 = Some(-30.0);
                k.false_origin_lat = Some(0.0);
                k.false_origin_long = Some(0.0);
            }),
        ),
        (
            "LCC 1SP lat0 == 0",
            keys(|k| {
                k.projected_cs_type = Some(32767);
                k.proj_coord_trans = Some(9);
                k.nat_origin_lat = Some(0.0);
                k.nat_origin_long = Some(0.0);
            }),
        ),
        (
            "TM zero scale factor",
            keys(|k| {
                k.projected_cs_type = Some(32767);
                k.proj_coord_trans = Some(1);
                k.nat_origin_long = Some(0.0);
                k.scale_at_nat_origin = Some(0.0);
            }),
        ),
        (
            "PolarStereo NaN lat_ts",
            keys(|k| {
                k.projected_cs_type = Some(32767);
                k.proj_coord_trans = Some(15);
                k.nat_origin_lat = Some(f64::NAN);
                k.straight_vert_pole_long = Some(0.0);
            }),
        ),
    ];
    for (what, gk) in cases {
        assert!(CrsModel::from_geo_keys(&gk).is_err(), "{what} should be refused");
    }
}

#[test]
fn antimeridian_utm_bounds_and_zoom_range_are_accurate() {
    // UTM Zone 60N (EPSG:32660, central meridian 177°E) 10m raster spanning ~179.5°E to ~179.5°W
    let crs = CrsModel::from_geo_keys(&keys(|k| {
        k.projected_cs_type = Some(32660);
    }))
    .unwrap();
    let (cx, cy) = crs.from_lonlat_deg(180.0, 0.0);
    let aff = Affine2D {
        a: 10.0,
        b: 0.0,
        c: cx - 50_000.0,
        d: 0.0,
        e: -10.0,
        f: cy + 50_000.0,
    };
    let b = crs.wgs84_bounds(&aff, None, 10_000, 10_000);
    assert_eq!(b[0], -180.0);
    assert_eq!(b[2], 180.0);
    let (_minz, maxz) = crs.web_mercator_zoom_range(&aff, 10_000, 10_000, 16.0);
    assert!(maxz >= 13, "10m antimeridian raster should have maxzoom >= 13, got {maxz}");

    // Polar-centered Polar Stereographic (EPSG:3413) 100m raster at North Pole (lat=90°):
    // world_crs84_zoom_range must not collapse to 0 from the 1/cos(lat) longitude singularity.
    let polar = CrsModel::from_geo_keys(&keys(|k| {
        k.projected_cs_type = Some(3413);
    }))
    .unwrap();
    let polar_aff = Affine2D {
        a: 100.0,
        b: 0.0,
        c: -50_000.0,
        d: 0.0,
        e: -100.0,
        f: 50_000.0,
    };
    let (p_minz, p_maxz) = polar.world_crs84_zoom_range(&polar_aff, 1000, 1000, 8.0);
    assert!(p_maxz >= 9 && p_minz <= p_maxz, "100m polar raster should have CRS84 maxzoom >= 9, got ({p_minz}, {p_maxz})");
}

#[test]
fn sentinel2_and_mercator_zoom_range_match_titiler() {
    // Verify zoom_for_res boundary behavior against morecantile.
    const CELL0_MERC: f64 = 156_543.033_928_040_97;
    assert_eq!(zoom_for_res(CELL0_MERC, CELL0_MERC, 24), 0);
    assert_eq!(zoom_for_res(13.834, CELL0_MERC, 24), 13);
    assert_eq!(zoom_for_res(13.834 * 64.0, CELL0_MERC, 24), 7);
    assert_eq!(zoom_for_res(9.554_628_535_647_032, CELL0_MERC, 24), 14);
    assert_eq!(zoom_for_res(0.000_1, CELL0_MERC, 24), 24);

    // Nine ground-truth grids matching rio-tiler 9.4 (7 UTM, 1 geographic, 1 Web Mercator).
    let utm = |epsg: u16| {
        CrsModel::from_geo_keys(&keys(|k| {
            k.projected_cs_type = Some(epsg);
        }))
        .unwrap()
    };
    let zooms = |crs: &CrsModel, width: u32, height: u32, bounds: [f64; 4]| {
        let aff = Affine2D {
            a: (bounds[2] - bounds[0]) / width as f64,
            b: 0.0,
            c: bounds[0],
            d: 0.0,
            e: -(bounds[3] - bounds[1]) / height as f64,
            f: bounds[3],
        };
        crs.web_mercator_zoom_range(&aff, width, height, 16.0)
    };
    let s2 = [699_960.0, 4_790_220.0, 809_760.0, 4_900_020.0];
    assert_eq!(zooms(&utm(32631), 10_980, 10_980, s2), (7, 13));
    assert_eq!(zooms(&utm(32631), 5_490, 5_490, s2), (7, 12));
    assert_eq!(zooms(&utm(32631), 1_830, 1_830, s2), (8, 11));
    assert_eq!(
        zooms(&utm(32632), 10_980, 10_980, [399_960.0, 5_090_220.0, 509_760.0, 5_200_020.0]),
        (7, 13)
    );
    assert_eq!(
        zooms(&utm(32633), 10_980, 10_980, [499_980.0, 6_790_200.0, 609_780.0, 6_900_000.0]),
        (7, 13)
    );
    assert_eq!(
        zooms(&utm(32619), 7_621, 7_791, [300_000.0, 100_000.0, 528_630.0, 333_730.0]),
        (7, 12)
    );
    assert_eq!(
        zooms(&utm(32755), 10_980, 10_980, [300_000.0, 5_690_220.0, 409_800.0, 5_800_020.0]),
        (8, 14)
    );
    assert_eq!(zooms(&CrsModel::Geographic, 3_600, 1_800, [-10.0, 30.0, 26.0, 48.0]), (3, 7));
    assert_eq!(
        zooms(&CrsModel::Epsg3857, 4_096, 2_048, [500_000.0, 5_300_000.0, 622_304.0, 5_361_152.0]),
        (9, 12)
    );
}

#[test]
fn geographic_0_360_wraps_western_hemisphere() {
    // Global 0..360° × -90..90° 1° grid (360x180)
    let p2c = Affine2D {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 0.0,
        e: -1.0,
        f: 90.0,
    };
    let c2p = p2c.invert().unwrap();
    assert!(CrsModel::Geographic.is_geographic_0_360(&p2c, 360));
    let wb = CrsModel::Geographic.wgs84_bounds(&p2c, None, 360, 180);
    assert_eq!(wb, [-180.0, -90.0, 180.0, 90.0]);
    // Western Hemisphere WebMercator tile (z=2, x=0, y=1 covers lon -180..-90, lat 0..66.5)
    let plan = CoordPlan::<256>::build(
        &CrsModel::Geographic,
        &c2p,
        None,
        360,
        180,
        TileCoord { z: 2, x: 0, y: 1 },
    )
    .expect("0..360 geographic grid should intersect Western Hemisphere tile");
    let (c0, c1, _, _) = plan.src_pixel_bounds();
    assert!(c0 >= 178 && c1 <= 272, "expected wrapped cols in [178..272], got {c0}..{c1}");

    // Verify EPSG:3395 World Mercator forward/inverse round-trip accuracy (< 1e-9 deg)
    for (lon, lat) in [(10.0, 45.0), (-122.4194, 37.7749), (140.0, -35.0), (0.0, 0.0)] {
        let (mx, my) = world_mercator_forward(lon, lat);
        let (rlon, rlat) = world_mercator_inverse(mx, my);
        assert!(
            (rlon - lon).abs() < 1e-9 && (rlat - lat).abs() < 1e-9,
            "EPSG:3395 round-trip failed for ({lon}, {lat}) -> ({rlon}, {rlat})"
        );
    }
}

#[test]
fn laea_antipode_tile_does_not_interpolate_ghost_pixels() {
    // EPSG:3035 (Europe LAEA: lat0=52°N, lon0=10°E, FE=4321000, FN=3210000)
    // Antipode is (-170°W, -52°S), inside WebMercator tile z=9/14/342 and z=12/113/2743.
    let crs = CrsModel::from_epsg(3035).unwrap();
    let p2c = Affine2D {
        a: 1000.0,
        b: 0.0,
        c: 2_321_000.0,
        d: 0.0,
        e: -1000.0,
        f: 5_210_000.0,
    };
    let c2p = p2c.invert().unwrap();
    for tile in [
        TileCoord { z: 9, x: 14, y: 342 },
        TileCoord { z: 12, x: 113, y: 2743 },
    ] {
        let plan = CoordPlan::<256>::build(&crs, &c2p, None, 4000, 4000, tile);
        assert!(
            plan.is_none(),
            "antipode tile {tile:?} in South Pacific must not intersect Europe EPSG:3035 COG"
        );
    }
}

#[test]
fn geographic_antimeridian_preserves_native_bounds_and_west_tile_wrapping() {
    // 170..190 × -20..0 (20° × 20° at 1024×1024)
    let p2c_east = Affine2D {
        a: 20.0 / 1024.0,
        b: 0.0,
        c: 170.0,
        d: 0.0,
        e: -20.0 / 1024.0,
        f: 0.0,
    };
    assert_eq!(
        CrsModel::Geographic.wgs84_bounds(&p2c_east, None, 1024, 1024),
        [170.0, -20.0, 190.0, 0.0]
    );
    // Still serves tiles west of antimeridian (z=4/0/8 covers lon -180..-157.5, lat -21.9..0)
    let c2p_east = p2c_east.invert().unwrap();
    assert!(
        CoordPlan::<256>::build(
            &CrsModel::Geographic,
            &c2p_east,
            None,
            1024,
            1024,
            TileCoord { z: 4, x: 0, y: 8 },
        )
        .is_some(),
        "170..190 geographic COG must still render tiles west of the antimeridian"
    );

    // -190..-170 × -20..0 (20° × 20° at 1024×1024)
    let p2c_west = Affine2D {
        a: 20.0 / 1024.0,
        b: 0.0,
        c: -190.0,
        d: 0.0,
        e: -20.0 / 1024.0,
        f: 0.0,
    };
    assert_eq!(
        CrsModel::Geographic.wgs84_bounds(&p2c_west, None, 1024, 1024),
        [-190.0, -20.0, -170.0, 0.0]
    );
}

#[test]
fn polar_stereographic_zoom_range_and_tile_resolution() {
    // Antarctic EPSG:3031 (-1e6..1e6, 1024×1024) enclosing the South Pole
    let ps3031 = CrsModel::from_epsg(3031).unwrap();
    let p2c_3031 = Affine2D {
        a: 2_000_000.0 / 1024.0,
        b: 0.0,
        c: -1_000_000.0,
        d: 0.0,
        e: -2_000_000.0 / 1024.0,
        f: 1_000_000.0,
    };
    assert_eq!(ps3031.web_mercator_zoom_range(&p2c_3031, 1024, 1024, 4.0), (2, 2));
    let (rx_3031, ry_3031) = ps3031.target_resolution_for_tile(TileCoord { z: 1, x: 1, y: 1 }, 256);
    // Tile 1/1/1 is much coarser than the 4x overview (7812.5m), so select_ifd must pick the 4x overview
    // rather than oversampling finer levels.
    assert!(
        rx_3031.min(ry_3031) >= 7_812.5,
        "EPSG:3031 tile 1/1/1 should select 4x overview (>= 7812.5m), got ({rx_3031}, {ry_3031})"
    );

    // Arctic UPS North EPSG:32661 (1.5e6..2.5e6, 1024×1024) enclosing the North Pole
    let ups = CrsModel::from_epsg(32661).unwrap();
    let p2c_ups = Affine2D {
        a: 1_000_000.0 / 1024.0,
        b: 0.0,
        c: 1_500_000.0,
        d: 0.0,
        e: -1_000_000.0 / 1024.0,
        f: 2_500_000.0,
    };
    assert_eq!(ups.web_mercator_zoom_range(&p2c_ups, 1024, 1024, 4.0), (2, 2));
    let (rx_ups, ry_ups) = ups.target_resolution_for_tile(TileCoord { z: 3, x: 6, y: 0 }, 256);
    // UPS tile 3/6/0 (~2500 m/px at center) is coarser than 2x overview (1953.1m) and finer than 4x (3906.2m),
    // so select_ifd picks the 2x overview instead of oversampling full-res (976.6m).
    assert!(
        rx_ups.min(ry_ups) >= 1_953.125 && rx_ups.min(ry_ups) < 3_906.25,
        "UPS tile 3/6/0 should select 2x overview (in [1953.1, 3906.25)), got ({rx_ups}, {ry_ups})"
    );
}

#[test]
fn ease_grid_6933_3410_and_equal_earth_8857_match_proj() {
    // Reference coordinates computed via pyproj / PROJ 9.x
    let cases: &[(u16, &[(f64, f64, f64, f64)])] = &[
        (
            6933,
            &[
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 30.0, 0.0, 3_658_789.324_855),
                (0.0, 60.0, 0.0, 6_351_419.997_326),
                (0.0, 84.0, 0.0, 7_301_649.868_311),
                (179.9, 30.0, 17_357_881.817_136, 3_658_789.324_855),
                (-179.9, -45.0, -17_357_881.817_136, -5_180_102.328_839),
            ],
        ),
        (
            3410,
            &[
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 30.0, 0.0, 3_678_430.200_868),
                (0.0, 60.0, 0.0, 6_371_228.000_000),
                (0.0, 84.0, 0.0, 7_316_558.750_695),
                (179.9, 30.0, 17_324_563.835_940, 3_678_430.200_868),
                (-179.9, -45.0, -17_324_563.835_940, -5_202_085.878_311),
            ],
        ),
        (
            8857,
            &[
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 30.0, 0.0, 3_764_325.426_892),
                (0.0, 60.0, 0.0, 6_924_035.137_553),
                (0.0, 84.0, 0.0, 8_324_478.230_694),
                (179.9, 30.0, 16_128_646.549_557, 3_764_325.426_892),
                (-179.9, -45.0, -14_784_256.707_468, -5_466_867.760_214),
            ],
        ),
        (
            8858,
            &[
                (0.0, 0.0, 8_621_979.531_108, 0.0),
                (0.0, 30.0, 8_068_805.944_748, 3_764_325.426_892),
                (179.9, 30.0, -8_077_771.284_687, 3_764_325.426_892),
            ],
        ),
        (
            8859,
            &[
                (0.0, 0.0, -14_369_965.885_181, 0.0),
                (0.0, 30.0, -13_448_009.907_913, 3_764_325.426_892),
                (179.9, 30.0, 2_680_636.641_644, 3_764_325.426_892),
            ],
        ),
    ];

    for &(code, pts) in cases {
        let crs = CrsModel::from_epsg(code).unwrap_or_else(|e| panic!("EPSG:{code}: {e}"));
        for &(lon, lat, ex, ey) in pts {
            let (gx, gy) = crs.from_lonlat_deg(lon, lat);
            assert!(
                (gx - ex).abs() < 1e-5 && (gy - ey).abs() < 1e-5,
                "EPSG:{code} forward ({lon},{lat}): got ({gx:.6},{gy:.6}), expected ({ex:.6},{ey:.6})"
            );
            let (rlon, rlat) = crs.to_lonlat_deg(gx, gy);
            assert!(
                (rlon - lon).abs() < 1e-9 && (rlat - lat).abs() < 1e-9,
                "EPSG:{code} inverse ({gx:.6},{gy:.6}): got ({rlon:.9},{rlat:.9}), expected ({lon},{lat})"
            );
        }
    }

    // Edge case: |q| slightly above qp from rounding clamps cleanly to +-90 deg;
    // far beyond poles returns NaN.
    let CrsModel::CylindricalEqualArea(cea) = CrsModel::from_epsg(6933).unwrap() else {
        unreachable!()
    };
    let y_pole = cea.a * cea.qp / (2.0 * cea.k0);
    let (_, lat_clamp) = cea.inverse(0.0, y_pole * (1.0 + 1e-11));
    assert!((lat_clamp.to_degrees() - 90.0).abs() < 1e-6);
    let (nan_lon, nan_lat) = cea.inverse(0.0, y_pole * 1.01);
    assert!(nan_lon.is_nan() && nan_lat.is_nan());

    // Fast path: axis-aligned EPSG:6933 uses Separable1D in CoordPlan::build
    let crs6933 = CrsModel::from_epsg(6933).unwrap();
    let p2c = Affine2D {
        a: 34_735_040.0 / 1024.0,
        b: 0.0,
        c: -17_367_520.0,
        d: 0.0,
        e: -14_629_080.0 / 512.0,
        f: 7_314_540.0,
    };
    let c2p = p2c.invert().unwrap();
    let plan = CoordPlan::<256>::build(&crs6933, &c2p, None, 1024, 512, TileCoord { z: 2, x: 3, y: 1 })
        .expect("dateline tile should intersect global EASE-2");
    assert!(matches!(plan, CoordPlan::Separable1D { .. }));
}

#[test]
fn west_antimeridian_neg190_geographic_raster_wraps_eastern_hemisphere_tiles() {
    // Geographic raster spanning [-190, -170] x [-20, 0] (100x100 px)
    let p2c = Affine2D {
        a: 0.2,
        b: 0.0,
        c: -190.0,
        d: 0.0,
        e: -0.2,
        f: 0.0,
    };
    let c2p = p2c.invert().unwrap();
    assert_eq!(CrsModel::Geographic.geographic_neg_180_max_lon(&p2c, 100), Some(-170.0));
    // WebMercator z=4, x=15, y=8 covers lon [+157.5, +180], lat [-21.94, 0] -> overlaps +170..+180 (-190..-180)
    let plan = CoordPlan::<256>::build(
        &CrsModel::Geographic,
        &c2p,
        None,
        100,
        100,
        TileCoord { z: 4, x: 15, y: 8 },
    )
    .expect("z=4/15/8 (+157.5..+180) must intersect [-190, -170] geographic COG");
    assert!(matches!(plan, CoordPlan::Separable1D { .. }));
}
