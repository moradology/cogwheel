pub mod crs;
pub mod ellipsoid;
pub mod epsg;
pub mod epsg_table;
pub mod mercator;

pub use crs::{CoordPlan, CrsModel, GRID_DIM, ProjError};
pub use epsg::{EpsgDef, lookup as lookup_epsg};
pub use ellipsoid::{
    AlbersEqualArea, CylindricalEqualArea, Ellipsoid, EqualEarth, Equirectangular,
    HelmertTransform, LambertAzimuthalEqualArea, LambertConformalConic, Mercator,
    ObliqueStereographic, PolarStereographic, Sinusoidal, SwissObliqueMercator, TransverseMercator,
};
pub use mercator::{ORIGIN_SHIFT, TileCoord, WEB_MERCATOR_R, mercator_to_lonlat_deg, tile_bounds_3857};

/// Symmetric longitude normalisation for geographic (`EPSG:4326` / CRS84) coordinates.
///
/// Accepts finite longitudes within ±2 full turns (`[-720°, +720°]`) and wraps them
/// symmetrically into `[-180°, +180°]` (preserving `-180.0` and `+180.0`). Returns `None`
/// for non-finite or out-of-range values (`|lon| > 720°`), identically in both directions.
#[inline]
pub fn normalize_geographic_lon(lon: f64) -> Option<f64> {
    if !lon.is_finite() || lon.abs() > 720.0 {
        return None;
    }
    let mut w = lon;
    while w > 180.0 {
        w -= 360.0;
    }
    while w < -180.0 {
        w += 360.0;
    }
    Some(w)
}

#[cfg(test)]
mod lon_norm_tests {
    use super::normalize_geographic_lon;

    #[test]
    fn normalize_geographic_lon_is_symmetric_across_two_turns() {
        assert_eq!(normalize_geographic_lon(179.5), Some(179.5));
        assert_eq!(normalize_geographic_lon(180.5), Some(-179.5));
        assert_eq!(normalize_geographic_lon(-180.5), Some(179.5));
        assert_eq!(normalize_geographic_lon(181.5), Some(-178.5));
        assert_eq!(normalize_geographic_lon(-540.5), Some(179.5));
        assert_eq!(normalize_geographic_lon(540.5), Some(-179.5));
        assert_eq!(normalize_geographic_lon(180.0), Some(180.0));
        assert_eq!(normalize_geographic_lon(-180.0), Some(-180.0));
        assert_eq!(normalize_geographic_lon(720.0), Some(0.0));
        assert_eq!(normalize_geographic_lon(-720.0), Some(0.0));
        assert_eq!(normalize_geographic_lon(900.5), None);
        assert_eq!(normalize_geographic_lon(-900.5), None);
        assert_eq!(normalize_geographic_lon(f64::NAN), None);
    }
}
