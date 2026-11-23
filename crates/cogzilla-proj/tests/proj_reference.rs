//! Every supported projection checked against coordinates computed by PROJ.
//!
//! `data/proj_reference_data.rs` holds (lon, lat, x, y) quadruples produced by PROJ
//! through rasterio for each case, from the projection's own geographic CRS so
//! that only the projection math is compared, not a datum shift. The GeoKeys
//! for the user-defined cases use the keys GDAL writes for that projection.
//! `benches/projection_oracle.py` covers the same ground end to end, through
//! real GeoTIFFs and a running server.

use cogzilla_proj::CrsModel;
use cogzilla_tiff::GeoKeys;

include!("data/proj_reference_data.rs");

/// One millimetre, in metres. Coordinates in feet are converted before comparing.
const TOLERANCE_M: f64 = 1e-3;

#[test]
fn forward_and_inverse_match_proj() {
    let mut checked = 0;
    for case in cases() {
        let crs = CrsModel::from_geo_keys(&case.keys)
            .unwrap_or_else(|e| panic!("{}: rejected: {e}", case.name));
        let unit_m = match case.keys.proj_linear_units {
            Some(9003) => 1200.0 / 3937.0,
            Some(9002) => 0.3048,
            _ => 1.0,
        };
        for &(lon, lat, x, y) in case.points {
            let (fx, fy) = crs.from_lonlat_deg(lon, lat);
            let err_m = (fx - x).hypot(fy - y) * unit_m;
            assert!(
                err_m < TOLERANCE_M,
                "{}: forward({lon}, {lat}) = ({fx}, {fy}), PROJ says ({x}, {y}); off by {err_m} m",
                case.name
            );

            let (ilon, ilat) = crs.to_lonlat_deg(x, y);
            let dlon = (ilon - lon + 540.0).rem_euclid(360.0) - 180.0;
            // A degree of longitude shrinks towards the poles; compare on the ground.
            let ground_m = (dlon * lat.to_radians().cos()).hypot(ilat - lat) * 111_320.0;
            assert!(
                ground_m < TOLERANCE_M,
                "{}: inverse({x}, {y}) = ({ilon}, {ilat}), expected ({lon}, {lat}); off by {ground_m} m",
                case.name
            );
            checked += 1;
        }
    }
    assert!(checked > 300, "only {checked} reference points were checked");
}
