use std::f64::consts::FRAC_PI_2;

/// Spherical Web Mercator (EPSG:3857) semi-major radius in meters.
pub const WEB_MERCATOR_R: f64 = 6_378_137.0;
/// Half world circumference in Web Mercator meters (`pi * R`).
pub const ORIGIN_SHIFT: f64 = std::f64::consts::PI * WEB_MERCATOR_R;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

/// Web Mercator (`EPSG:3857`) bounding box of an XYZ tile: `(min_x, min_y, max_x, max_y)`.
#[inline(always)]
pub fn tile_bounds_3857(tile: TileCoord) -> (f64, f64, f64, f64) {
    let n = (1u64 << tile.z) as f64;
    let span = (2.0 * ORIGIN_SHIFT) / n;
    let min_x = -ORIGIN_SHIFT + (tile.x as f64) * span;
    let max_x = min_x + span;
    let max_y = ORIGIN_SHIFT - (tile.y as f64) * span;
    let min_y = max_y - span;
    (min_x, min_y, max_x, max_y)
}

/// Inverse Web Mercator: `(mx, my)` in meters -> `(lon_rad, lat_rad)`.
#[inline(always)]
pub fn mercator_to_lonlat_rad(mx: f64, my: f64) -> (f64, f64) {
    let lon = mx / WEB_MERCATOR_R;
    let lat = 2.0 * (my / WEB_MERCATOR_R).exp().atan() - FRAC_PI_2;
    (lon, lat)
}

/// Inverse Web Mercator: `(mx, my)` in meters -> `(lon_deg, lat_deg)`.
#[inline(always)]
pub fn mercator_to_lonlat_deg(mx: f64, my: f64) -> (f64, f64) {
    let (lon, lat) = mercator_to_lonlat_rad(mx, my);
    (lon.to_degrees(), lat.to_degrees())
}
