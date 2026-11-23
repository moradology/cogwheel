//! Every entry of the generated EPSG table checked against PROJ.
//!
//! `data/epsg_reference.csv` holds five points inside each code's area of use,
//! projected by PROJ from the CRS's own geographic CRS. Regenerate both the
//! table and the reference with `benches/gen_epsg_table.py`.

use std::collections::HashMap;

use cogzilla_proj::{CrsModel, epsg::unit_metres, epsg_table::EPSG_TABLE, lookup_epsg};
use cogzilla_tiff::GeoKeys;

const REFERENCE: &str = include_str!("data/epsg_reference.csv");

/// One millimetre.
const TOLERANCE_M: f64 = 1e-3;

#[test]
fn table_is_sorted_and_unique() {
    assert!(EPSG_TABLE.windows(2).all(|w| w[0].code < w[1].code));
    assert!(EPSG_TABLE.len() > 3000);
}

#[test]
fn every_entry_matches_proj() {
    let mut points_per_code: HashMap<u16, u32> = HashMap::new();
    let mut worst = (0.0_f64, 0u16);
    for line in REFERENCE.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let code: u16 = f[0].parse().unwrap();
        let [lon, lat, x, y]: [f64; 4] = std::array::from_fn(|i| f[i + 1].parse().unwrap());

        let def = lookup_epsg(code).unwrap_or_else(|| panic!("EPSG:{code} has reference points but no entry"));
        // Go through the same door a file would.
        let keys = GeoKeys { projected_cs_type: Some(code), ..Default::default() };
        let crs = CrsModel::from_geo_keys(&keys).unwrap_or_else(|e| panic!("EPSG:{code}: {e}"));
        let unit_m = unit_metres(def.unit);

        let (fx, fy) = crs.from_lonlat_deg(lon, lat);
        let err_m = (fx - x).hypot(fy - y) * unit_m;
        assert!(
            err_m < TOLERANCE_M,
            "EPSG:{code}: forward({lon}, {lat}) = ({fx}, {fy}), PROJ says ({x}, {y}); off by {err_m} m"
        );
        if err_m > worst.0 {
            worst = (err_m, code);
        }

        let (ilon, ilat) = crs.to_lonlat_deg(x, y);
        let dlon = (ilon - lon + 540.0).rem_euclid(360.0) - 180.0;
        let ground_m = (dlon * lat.to_radians().cos()).hypot(ilat - lat) * 111_320.0;
        assert!(
            ground_m < TOLERANCE_M,
            "EPSG:{code}: inverse({x}, {y}) = ({ilon}, {ilat}), expected ({lon}, {lat}); off by {ground_m} m"
        );
        *points_per_code.entry(code).or_default() += 1;
    }
    for def in EPSG_TABLE {
        assert!(
            points_per_code.get(&def.code).copied().unwrap_or(0) >= 3,
            "EPSG:{} has fewer than three reference points",
            def.code
        );
    }
    println!("largest forward error: {:.3e} m (EPSG:{})", worst.0, worst.1);
}
