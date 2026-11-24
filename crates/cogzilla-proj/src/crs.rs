use cogzilla_tiff::{Affine2D, GeoKeys};
use thiserror::Error;

use crate::ellipsoid::{
    AlbersEqualArea, Ellipsoid, LambertAzimuthalEqualArea, LambertConformalConic,
    PolarStereographic, Sinusoidal, TransverseMercator,
};
use crate::mercator::{
    ORIGIN_SHIFT, TileCoord, lonlat_deg_to_mercator, mercator_to_lonlat_deg,
    mercator_to_lonlat_rad, tile_bounds_3857,
};

#[derive(Debug, Error)]
pub enum ProjError {
    #[error("unsupported CRS: {reason} (GeoKeys: projected={projected:?}, geographic={geographic:?}, coord_trans={coord_trans:?})")]
    UnsupportedCrs {
        reason: String,
        projected: Option<u16>,
        geographic: Option<u16>,
        coord_trans: Option<u16>,
    },
}

/// Closed-form source CRS model built once per COG from its GeoTIFF GeoKeys.
#[derive(Debug, Clone)]
pub enum CrsModel {
    Epsg3857,
    Geographic,
    TransverseMercator(TransverseMercator),
    Albers(AlbersEqualArea),
    LambertConformalConic(LambertConformalConic),
    PolarStereographic(PolarStereographic),
    Laea(LambertAzimuthalEqualArea),
    Sinusoidal(Sinusoidal),
}

impl CrsModel {
    /// Construct a `CrsModel` from parsed GeoTIFF `GeoKeys`.
    ///
    /// Anything this crate cannot place correctly is an error. Coordinates are
    /// treated as WGS84 with no datum shift, so a CRS is only accepted when its
    /// datum is WGS84 or one that agrees with it to about two metres (NAD83,
    /// ETRS89, GDA94 and similar). A file on NAD27 or OSGB36 would otherwise be
    /// drawn tens to hundreds of metres out of place with no warning.
    pub fn from_geo_keys(gk: &GeoKeys) -> Result<Self, ProjError> {
        let unsupported = |reason: &str| ProjError::UnsupportedCrs {
            reason: reason.to_string(),
            projected: gk.projected_cs_type,
            geographic: gk.geographic_type,
            coord_trans: gk.proj_coord_trans,
        };

        if matches!(gk.geog_angular_units, Some(u) if u != 9102) {
            return Err(unsupported("angular units other than degrees"));
        }
        if !matches!(gk.prime_meridian, None | Some(8901 | 32767))
            || matches!(gk.prime_meridian_long, Some(lon) if !lon.is_finite() || lon.abs() > 1e-9)
        {
            return Err(unsupported("prime meridian other than Greenwich"));
        }

        // 1. ProjectedCSTypeGeoKey EPSG codes. A code fixes the datum, projection
        //    and units by definition, unless ProjLinearUnitsGeoKey (3076) explicitly
        //    overrides the linear unit with a conflicting code.
        match gk.projected_cs_type {
            Some(3857 | 3785) => {
                if matches!(gk.proj_linear_units, Some(u) if u != 9001 && u != 32767) {
                    return Err(unsupported("EPSG:3857 with non-metre linear units override"));
                }
                return Ok(Self::Epsg3857);
            }
            // User-defined: described by the parameter keys below.
            Some(32767) | None => {}
            Some(code) => {
                return match crate::epsg::lookup(code) {
                    Some(def) => {
                        let expected_unit = match def.unit {
                            0 => 9001,
                            1 => 9002,
                            2 => 9003,
                            _ => 0,
                        };
                        if matches!(gk.proj_linear_units, Some(u) if u != expected_unit && u != 32767)
                        {
                            return Err(unsupported(
                                "projected CRS linear units override conflicts with EPSG definition",
                            ));
                        }
                        Ok(def.to_model())
                    }
                    // Parameter keys beside an unknown code cannot vouch for its datum.
                    None => Err(unsupported("projected CRS code is not in the supported list")),
                };
            }
        }

        let sphere_ok = matches!(gk.proj_coord_trans, Some(10 | 24));
        let ellip = Self::wgs84_compatible_ellipsoid(gk, sphere_ok).map_err(unsupported)?;

        // 2. User-defined projection parameters via ProjCoordTransGeoKey.
        if let Some(ct) = gk.proj_coord_trans {
            let unit = match gk.proj_linear_units {
                None | Some(9001) => 1.0,
                Some(9002) => 0.3048,
                Some(9003) => 1200.0 / 3937.0,
                Some(_) => return Err(unsupported("linear units other than metres or feet")),
            };
            let missing = |what: &str| unsupported(&format!("{what} is missing"));
            let require_finite = |v: f64, what: &str| -> Result<f64, ProjError> {
                if v.is_finite() {
                    Ok(v)
                } else {
                    Err(unsupported(&format!("{what} must be finite")))
                }
            };
            let require_lat = |v: f64, open_poles: bool, what: &str| -> Result<f64, ProjError> {
                let valid = if open_poles {
                    v.is_finite() && v.abs() < 90.0 - 1e-9
                } else {
                    v.is_finite() && v.abs() <= 90.0
                };
                if valid {
                    Ok(v)
                } else {
                    Err(unsupported(&format!("{what} out of range")))
                }
            };
            let require_scale = |v: f64| -> Result<f64, ProjError> {
                if v.is_finite() && v > 0.0 {
                    Ok(v)
                } else {
                    Err(unsupported("scale factor must be finite and positive"))
                }
            };
            let lon0 = gk.false_origin_long.or(gk.nat_origin_long).or(gk.center_long);
            let lat0 = gk.false_origin_lat.or(gk.nat_origin_lat).or(gk.center_lat);
            let fe = require_finite(
                gk.false_origin_easting
                    .or(gk.false_easting)
                    .or(gk.center_easting)
                    .unwrap_or(0.0),
                "false easting",
            )?;
            let fn_m = require_finite(
                gk.false_origin_northing
                    .or(gk.false_northing)
                    .or(gk.center_northing)
                    .unwrap_or(0.0),
                "false northing",
            )?;
            return match ct {
                // CT_TransverseMercator
                1 => {
                    let lon0 = require_finite(
                        lon0.ok_or_else(|| missing("central meridian"))?,
                        "central meridian",
                    )?;
                    let lat0 = require_lat(lat0.unwrap_or(0.0), false, "latitude of origin")?;
                    let k0 = require_scale(gk.scale_at_nat_origin.unwrap_or(1.0))?;
                    let mut tm = TransverseMercator::new(ellip, lon0, lat0, k0, fe, fn_m);
                    tm.unit = unit;
                    Ok(Self::TransverseMercator(tm))
                }
                // CT_LambertConfConic_2SP
                8 => {
                    let p1 = require_lat(
                        gk.std_parallel_1.ok_or_else(|| missing("standard parallel"))?,
                        true,
                        "standard parallel 1",
                    )?;
                    let p2 = require_lat(
                        gk.std_parallel_2.unwrap_or(p1),
                        true,
                        "standard parallel 2",
                    )?;
                    if (p1 + p2).abs() < 1e-9 {
                        return Err(unsupported(
                            "LCC standard parallels are symmetric across the equator",
                        ));
                    }
                    let lat0 = require_lat(
                        lat0.ok_or_else(|| missing("latitude of origin"))?,
                        true,
                        "latitude of origin",
                    )?;
                    let lon0 = require_finite(
                        lon0.ok_or_else(|| missing("central meridian"))?,
                        "central meridian",
                    )?;
                    let mut lcc =
                        LambertConformalConic::new_2sp(ellip, p1, p2, lat0, lon0, fe, fn_m);
                    lcc.unit = unit;
                    Ok(Self::LambertConformalConic(lcc))
                }
                // CT_LambertConfConic_1SP (CT_LambertConfConic_Helmert)
                9 => {
                    let lat0 = require_lat(
                        lat0.ok_or_else(|| missing("latitude of origin"))?,
                        true,
                        "latitude of origin",
                    )?;
                    if lat0.abs() < 1e-9 {
                        return Err(unsupported("LCC 1SP latitude of origin cannot be zero"));
                    }
                    let lon0 = require_finite(
                        lon0.ok_or_else(|| missing("central meridian"))?,
                        "central meridian",
                    )?;
                    let k0 = require_scale(gk.scale_at_nat_origin.unwrap_or(1.0))?;
                    let mut lcc = LambertConformalConic::new_1sp(ellip, lat0, lon0, k0, fe, fn_m);
                    lcc.unit = unit;
                    Ok(Self::LambertConformalConic(lcc))
                }
                // CT_AlbersEqualArea
                11 => {
                    let p1 = require_lat(
                        gk.std_parallel_1.ok_or_else(|| missing("standard parallel"))?,
                        false,
                        "standard parallel 1",
                    )?;
                    let p2 = require_lat(
                        gk.std_parallel_2.unwrap_or(p1),
                        false,
                        "standard parallel 2",
                    )?;
                    if (p1 + p2).abs() < 1e-9 {
                        return Err(unsupported(
                            "Albers standard parallels are symmetric across the equator",
                        ));
                    }
                    let lat0 = require_lat(
                        lat0.ok_or_else(|| missing("latitude of origin"))?,
                        false,
                        "latitude of origin",
                    )?;
                    let lon0 = require_finite(
                        lon0.ok_or_else(|| missing("central meridian"))?,
                        "central meridian",
                    )?;
                    let mut alb = AlbersEqualArea::new(ellip, p1, p2, lat0, lon0, fe, fn_m);
                    alb.unit = unit;
                    Ok(Self::Albers(alb))
                }
                // CT_PolarStereographic. GDAL writes the latitude of true scale as
                // the natural origin latitude and the central meridian as the
                // "straight vertical pole longitude".
                15 => {
                    let lat_ts = require_lat(
                        gk.nat_origin_lat
                            .or(gk.std_parallel_1)
                            .ok_or_else(|| missing("latitude of true scale"))?,
                        false,
                        "latitude of true scale",
                    )?;
                    let lon0 = require_finite(
                        gk.straight_vert_pole_long
                            .or(gk.nat_origin_long)
                            .or(gk.center_long)
                            .ok_or_else(|| missing("central meridian"))?,
                        "central meridian",
                    )?;
                    let k0 = require_scale(gk.scale_at_nat_origin.unwrap_or(1.0))?;
                    let mut ps = if lat_ts.abs() > 90.0 - 1e-9 {
                        PolarStereographic::new_pole_scale(ellip, lat_ts < 0.0, k0, lon0, fe, fn_m)
                    } else if (k0 - 1.0).abs() > 1e-12 {
                        return Err(unsupported(
                            "polar stereographic with both a latitude of true scale and a scale factor",
                        ));
                    } else {
                        PolarStereographic::new(ellip, lat_ts, lon0, fe, fn_m)
                    };
                    ps.unit = unit;
                    Ok(Self::PolarStereographic(ps))
                }
                // CT_LambertAzimEqualArea
                10 => {
                    let lat0 = require_lat(
                        gk.center_lat
                            .or(gk.nat_origin_lat)
                            .ok_or_else(|| missing("latitude of origin"))?,
                        false,
                        "latitude of origin",
                    )?;
                    let lon0 = require_finite(
                        gk.center_long
                            .or(gk.nat_origin_long)
                            .ok_or_else(|| missing("central meridian"))?,
                        "central meridian",
                    )?;
                    let mut p = LambertAzimuthalEqualArea::new(ellip, lat0, lon0, fe, fn_m);
                    p.unit = unit;
                    Ok(Self::Laea(p))
                }
                // CT_Sinusoidal
                24 => {
                    let lon0 = require_finite(
                        gk.center_long
                            .or(gk.nat_origin_long)
                            .ok_or_else(|| missing("central meridian"))?,
                        "central meridian",
                    )?;
                    let mut p = Sinusoidal::new(ellip, lon0, fe, fn_m);
                    p.unit = unit;
                    Ok(Self::Sinusoidal(p))
                }
                _ => Err(unsupported("projection method is not implemented")),
            };
        }

        // 3. Geographic CRS in degrees.
        if gk.model_type == Some(2) || (gk.model_type.is_none() && gk.geographic_type.is_some()) {
            return Ok(Self::Geographic);
        }

        Err(unsupported("no recognizable CRS definition"))
    }

    /// The ellipsoid for a user-defined or geographic CRS, or the reason its
    /// datum cannot be treated as WGS84.
    fn wgs84_compatible_ellipsoid(
        gk: &GeoKeys,
        sphere_ok: bool,
    ) -> Result<Ellipsoid, &'static str> {
        use crate::epsg_table::{WGS84_LIKE_DATUM, WGS84_LIKE_GCS};
        const DATUM: &str = "datum is not WGS84 or one equivalent to it within about 2 m";
        const ELLIPSOID: &str = "ellipsoid is not WGS84 or GRS80, so the datum shift is unknown";
        match gk.geographic_type {
            Some(4326) => return Ok(Ellipsoid::WGS84),
            Some(32767) | None => {}
            Some(code) if WGS84_LIKE_GCS.binary_search(&code).is_ok() => {
                return Ok(Ellipsoid::GRS80);
            }
            Some(_) => return Err(DATUM),
        }
        if matches!(gk.geodetic_datum, Some(d) if d != 32767 && WGS84_LIKE_DATUM.binary_search(&d).is_err())
        {
            return Err(DATUM);
        }
        // A sphere is stored as equal axes or as zero inverse flattening.
        let inv_f = match (gk.semi_major_axis, gk.semi_minor_axis, gk.inv_flattening) {
            (_, _, Some(inv_f)) => Some(inv_f),
            (Some(a), Some(b), None) if a > 0.0 && b > 0.0 => {
                Some(if (a - b).abs() < 1e-6 { 0.0 } else { a / (a - b) })
            }
            _ => None,
        };
        match (gk.semi_major_axis, inv_f, gk.ellipsoid) {
            (Some(a), Some(inv_f), _) => {
                // WGS84 and GRS80 differ only past the ninth digit of 1/f.
                if (a - 6_378_137.0).abs() < 0.5 && (inv_f - 298.257_22).abs() < 1e-3 {
                    Ok(Ellipsoid::new(a, inv_f))
                } else if sphere_ok && inv_f == 0.0 && (6.35e6..6.40e6).contains(&a) {
                    // Equal-area world grids (MODIS sinusoidal, EASE-Grid) are defined
                    // on a sphere whose latitude and longitude are taken as WGS84's.
                    Ok(Ellipsoid::sphere(a))
                } else {
                    Err(ELLIPSOID)
                }
            }
            (Some(_), None, _) | (None, Some(_), _) => Err("ellipsoid is incompletely specified"),
            (None, None, Some(7030) | None) => Ok(Ellipsoid::WGS84),
            (None, None, Some(7019)) => Ok(Ellipsoid::GRS80),
            (None, None, Some(_)) => Err(ELLIPSOID),
        }
    }

    /// Map a Web Mercator (`EPSG:3857`) point `(mx, my)` into this CRS's `(x, y)`.
    #[inline(always)]
    pub fn from_3857(&self, mx: f64, my: f64) -> (f64, f64) {
        match self {
            Self::Epsg3857 => (mx, my),
            Self::Geographic => mercator_to_lonlat_deg(mx, my),
            Self::TransverseMercator(tm) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                tm.forward(lon, lat)
            }
            Self::Albers(alb) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                alb.forward(lon, lat)
            }
            Self::LambertConformalConic(lcc) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                lcc.forward(lon, lat)
            }
            Self::PolarStereographic(ps) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                ps.forward(lon, lat)
            }
            Self::Laea(p) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                p.forward(lon, lat)
            }
            Self::Sinusoidal(p) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                p.forward(lon, lat)
            }
        }
    }

    /// Estimate the target pixel resolution `(rx, ry)` in source CRS units at the center of an XYZ tile,
    /// used to select the optimal COG overview IFD. Falls back to interior quadrant probes if the
    /// tile center falls outside the projection domain (e.g. UTM at low zoom).
    pub fn target_resolution_for_tile(&self, tile: TileCoord, tile_size: u32) -> (f64, f64) {
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
        let dx = (max_x - min_x) / (tile_size as f64);
        let dy = (max_y - min_y) / (tile_size as f64);

        let probe = |fx: f64, fy: f64| -> (f64, f64) {
            let cx = min_x + fx * (max_x - min_x);
            let cy = min_y + fy * (max_y - min_y);
            let (x0, y0) = self.from_3857(cx, cy);
            let (xx, yx) = self.from_3857(cx + dx, cy);
            let (xy, yy) = self.from_3857(cx, cy - dy);
            let rx = ((xx - x0).powi(2) + (yx - y0).powi(2)).sqrt();
            let ry = ((xy - x0).powi(2) + (yy - y0).powi(2)).sqrt();
            (rx, ry)
        };

        let (rx, ry) = probe(0.5, 0.5);
        if rx.is_finite() && ry.is_finite() && rx > 0.0 && ry > 0.0 {
            return (rx, ry);
        }
        for &(fx, fy) in &[(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
            let (rx, ry) = probe(fx, fy);
            if rx.is_finite() && ry.is_finite() && rx > 0.0 && ry > 0.0 {
                return (rx, ry);
            }
        }
        (rx, ry)
    }

    /// Map a WGS84 `(lon_deg, lat_deg)` point directly into this CRS's `(x, y)`.
    #[inline(always)]
    pub fn from_lonlat_deg(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        match self {
            Self::Epsg3857 => lonlat_deg_to_mercator(lon_deg, lat_deg),
            Self::Geographic => (lon_deg, lat_deg),
            Self::TransverseMercator(tm) => tm.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::Albers(alb) => alb.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::LambertConformalConic(lcc) => {
                lcc.forward(lon_deg.to_radians(), lat_deg.to_radians())
            }
            Self::PolarStereographic(ps) => ps.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::Laea(p) => p.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::Sinusoidal(p) => p.forward(lon_deg.to_radians(), lat_deg.to_radians()),
        }
    }

    /// Inverse map a point `(x, y)` in this CRS back to WGS84 `(lon_deg, lat_deg)`,
    /// normalizing longitude into `[-180, 180]`.
    pub fn to_lonlat_deg(&self, x: f64, y: f64) -> (f64, f64) {
        let (lon, lat) = match self {
            Self::Epsg3857 => mercator_to_lonlat_deg(x, y),
            Self::Geographic => (x, y),
            Self::TransverseMercator(tm) => {
                let (lon, lat) = tm.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::Albers(alb) => {
                let (lon, lat) = alb.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::LambertConformalConic(lcc) => {
                let (lon, lat) = lcc.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::PolarStereographic(ps) => {
                let (lon, lat) = ps.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::Laea(p) => {
                let (lon, lat) = p.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::Sinusoidal(p) => {
                let (lon, lat) = p.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
        };
        let norm_lon = if lon.is_finite() && (lon < -180.0 || lon > 180.0) {
            ((lon + 180.0).rem_euclid(360.0)) - 180.0
        } else {
            lon
        };
        (norm_lon, lat)
    }

    /// Exact WGS84 geographic bounding box `[min_lon, min_lat, max_lon, max_lat]` by sampling
    /// 32 points along the 4 edges of the raster plus checking whether either pole lies inside.
    pub fn wgs84_bounds(&self, pixel_to_crs: &Affine2D, width: u32, height: u32) -> [f64; 4] {
        let mut min_lon = f64::INFINITY;
        let mut min_lat = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let w = width as f64;
        let h = height as f64;
        let steps = 8;

        let mut visit = |col: f64, row: f64| {
            let (sx, sy) = pixel_to_crs.forward(col, row);
            let (lon, lat) = self.to_lonlat_deg(sx, sy);
            if lon.is_finite() && lat.is_finite() {
                min_lon = min_lon.min(lon);
                max_lon = max_lon.max(lon);
                min_lat = min_lat.min(lat);
                max_lat = max_lat.max(lat);
            }
        };

        for i in 0..=steps {
            let t = (i as f64) / (steps as f64);
            visit(t * w, 0.0);
            visit(t * w, h);
            visit(0.0, t * h);
            visit(w, t * h);
        }

        // Check whether North or South Pole lies inside the raster footprint (polar CRSes).
        // Skip for Web Mercator (clamps ±90° to ±85.0511°) and Geographic (rectangular lon/lat).
        if !matches!(self, Self::Epsg3857 | Self::Geographic) {
            if let Some(crs_to_pixel) = pixel_to_crs.invert() {
                for &pole_lat in &[90.0_f64, -90.0_f64] {
                    let (px_crs, py_crs) = self.from_lonlat_deg(0.0, pole_lat);
                    if px_crs.is_finite() && py_crs.is_finite() {
                        let (col, row) = crs_to_pixel.forward(px_crs, py_crs);
                        if col >= 0.0 && col <= w && row >= 0.0 && row <= h {
                            min_lon = -180.0;
                            max_lon = 180.0;
                            if pole_lat > 0.0 {
                                max_lat = 90.0;
                            } else {
                                min_lat = -90.0;
                            }
                        }
                    }
                }
            }
        }

        [min_lon, min_lat, max_lon, max_lat]
    }

    /// Estimate TiTiler-compatible `(minzoom, maxzoom)` in WebMercatorQuad for a raster.
    pub fn web_mercator_zoom_range(
        &self,
        pixel_to_crs: &Affine2D,
        width: u32,
        height: u32,
        overview_factor: f64,
    ) -> (u8, u8) {
        let [w, s, e, n] = self.wgs84_bounds(pixel_to_crs, width, height);
        let (mx0, my0) = lonlat_deg_to_mercator(w, s);
        let (mx1, my1) = lonlat_deg_to_mercator(e, n);
        let span_m = (mx1 - mx0).abs().max((my1 - my0).abs()).max(1.0);
        let px_m = span_m / (width.max(height) as f64);
        let world_px0 = (2.0 * ORIGIN_SHIFT) / 256.0;
        let maxzoom = (world_px0 / px_m).log2().ceil().clamp(0.0, 22.0) as u8;
        let minzoom_f = (world_px0 / (px_m * overview_factor.max(1.0))).log2().floor();
        let minzoom = minzoom_f.clamp(0.0, maxzoom as f64) as u8;
        (minzoom, maxzoom)
    }
}

/// Control grid dimension: `17x17` points = `16x16` bilinear cells across any tile.
pub const GRID_DIM: usize = 17;

/// Precomputed coordinate mapping from output XYZ tile pixels -> source IFD pixel-center space
/// `(u = col - 0.5, v = row - 0.5)` so `(0.0, 0.0)` samples 100% of source pixel `(0, 0)`.
pub enum CoordPlan<const N: usize> {
    /// Separable 1D mapping (`EPSG:3857` and `EPSG:4326` with axis-aligned GeoTransform).
    /// Vertical weight and source row pointers are invariant across every output scanline!
    Separable1D {
        u: Box<[f32; N]>,
        v: Box<[f32; N]>,
        src_min_col: i32,
        src_max_col: i32,
        src_min_row: i32,
        src_max_row: i32,
        all_in_bounds: bool,
    },
    /// 17x17 exact closed-form control grid (z >= 9: UTM, Albers, LCC, Polar Stereo).
    Grid2D {
        grid_u: [f32; GRID_DIM * GRID_DIM],
        grid_v: [f32; GRID_DIM * GRID_DIM],
        src_min_col: i32,
        src_max_col: i32,
        src_min_row: i32,
        src_max_row: i32,
        all_in_bounds: bool,
    },
    /// Exact per-pixel closed-form coordinates for low-zoom (z <= 8) projected tiles where
    /// wide tile spans (~100s–1000s km) exceed linear grid curvature or cross projection boundaries.
    Exact2D {
        u: Box<[f32]>,
        v: Box<[f32]>,
        src_min_col: i32,
        src_max_col: i32,
        src_min_row: i32,
        src_max_row: i32,
        all_in_bounds: bool,
    },
}

impl<const N: usize> CoordPlan<N> {
    /// Build a `CoordPlan` for output XYZ `tile` into an IFD with `crs_to_pixel` affine.
    /// Returns `None` if the tile lies completely outside `[0..ifd_w, 0..ifd_h]`.
    pub fn build(
        crs: &CrsModel,
        crs_to_pixel: &Affine2D,
        ifd_w: u32,
        ifd_h: u32,
        tile: TileCoord,
    ) -> Option<Self> {
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
        let step_x = (max_x - min_x) / (N as f64);
        let step_y = (max_y - min_y) / (N as f64);

        if crs_to_pixel.is_axis_aligned()
            && matches!(crs, CrsModel::Epsg3857 | CrsModel::Geographic)
        {
            let mut u = Box::new([0.0f32; N]);
            let mut v = Box::new([0.0f32; N]);

            let mut min_u = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;

            match crs {
                CrsModel::Epsg3857 => {
                    for px in 0..N {
                        let mx = min_x + (px as f64 + 0.5) * step_x;
                        let uu = (crs_to_pixel.a * mx + crs_to_pixel.c - 0.5) as f32;
                        u[px] = uu;
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                    }
                    for py in 0..N {
                        let my = max_y - (py as f64 + 0.5) * step_y;
                        let vv = (crs_to_pixel.e * my + crs_to_pixel.f - 0.5) as f32;
                        v[py] = vv;
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    }
                }
                CrsModel::Geographic => {
                    for px in 0..N {
                        let mx = min_x + (px as f64 + 0.5) * step_x;
                        let (lon, _) = mercator_to_lonlat_deg(mx, 0.0);
                        let uu = (crs_to_pixel.a * lon + crs_to_pixel.c - 0.5) as f32;
                        u[px] = uu;
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                    }
                    for py in 0..N {
                        let my = max_y - (py as f64 + 0.5) * step_y;
                        let (_, lat) = mercator_to_lonlat_deg(0.0, my);
                        let vv = (crs_to_pixel.e * lat + crs_to_pixel.f - 0.5) as f32;
                        v[py] = vv;
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    }
                }
                _ => unreachable!(),
            }

            // Include +/- 1 pixel margin for bilinear/cubic neighbor lookups across block seams.
            // Use saturating arithmetic so extreme low-zoom coordinates on high-res COGs
            // (|u| > i32::MAX) never panic in debug or wrap to i32::MIN in release.
            let src_min_col = (min_u.floor() as i32).saturating_sub(1).max(0);
            let src_max_col = (max_u.ceil() as i32).saturating_add(1).min(ifd_w as i32 - 1);
            let src_min_row = (min_v.floor() as i32).saturating_sub(1).max(0);
            let src_max_row = (max_v.ceil() as i32).saturating_add(1).min(ifd_h as i32 - 1);

            if max_u < -0.5 || min_u > (ifd_w as f32 - 0.5) || max_v < -0.5 || min_v > (ifd_h as f32 - 0.5) {
                return None;
            }
            if src_min_col > src_max_col || src_min_row > src_max_row {
                return None;
            }

            let all_in_bounds = min_u >= 0.0
                && min_v >= 0.0
                && (max_u.floor() as i32).saturating_add(1) < ifd_w as i32
                && (max_v.floor() as i32).saturating_add(1) < ifd_h as i32;

            Some(Self::Separable1D {
                u,
                v,
                src_min_col,
                src_max_col,
                src_min_row,
                src_max_row,
                all_in_bounds,
            })
        } else if tile.z <= 8 || (tile.x as u64) + 1 >= (1u64 << tile.z.min(30)) {
            // Low-zoom projected tile (z=0..8): adaptive error-bounded quadtree subdivision.
            // Subdivides wherever projection curvature exceeds 0.05 source pixels or hits a
            // domain boundary (poles / antimeridian), skips empty out-of-bounds cells in O(1),
            // and linearly interpolates smooth cells — 50–100× fewer trig calls than per-pixel
            // evaluation with < 0.05 px worst-case error.
            let mut u = vec![f32::NAN; N * N].into_boxed_slice();
            let mut v = vec![f32::NAN; N * N].into_boxed_slice();
            let mut min_u = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            let mut any_in = false;
            let fw = ifd_w as f32;
            let fh = ifd_h as f32;

            let project_px = |px: usize, py: usize| -> (f32, f32) {
                let mx = min_x + (px as f64 + 0.5) * step_x;
                let my = max_y - (py as f64 + 0.5) * step_y;
                let (sx, sy) = crs.from_3857(mx, my);
                if !sx.is_finite() || !sy.is_finite() {
                    return (f32::NAN, f32::NAN);
                }
                let (col, row) = crs_to_pixel.forward(sx, sy);
                ((col - 0.5) as f32, (row - 0.5) as f32)
            };

            #[allow(clippy::too_many_arguments)]
            fn fill_cell<const N: usize, F: Fn(usize, usize) -> (f32, f32)>(
                x0: usize,
                y0: usize,
                x1: usize,
                y1: usize,
                c00: (f32, f32),
                c10: (f32, f32),
                c01: (f32, f32),
                c11: (f32, f32),
                fw: f32,
                fh: f32,
                project_px: &F,
                u: &mut [f32],
                v: &mut [f32],
                any_in: &mut bool,
                min_u: &mut f32,
                max_u: &mut f32,
                min_v: &mut f32,
                max_v: &mut f32,
            ) {
                let w = x1 - x0;
                let h = y1 - y0;
                if w <= 2 || h <= 2 {
                    for py in y0..=y1 {
                        let row_off = py * N;
                        for px in x0..=x1 {
                            let (uu, vv) = project_px(px, py);
                            u[row_off + px] = uu;
                            v[row_off + px] = vv;
                            if uu >= -1.0 && uu <= fw && vv >= -1.0 && vv <= fh {
                                *any_in = true;
                                *min_u = (*min_u).min(uu);
                                *max_u = (*max_u).max(uu);
                                *min_v = (*min_v).min(vv);
                                *max_v = (*max_v).max(vv);
                            }
                        }
                    }
                    return;
                }

                let all_finite = c00.0.is_finite()
                    && c00.1.is_finite()
                    && c10.0.is_finite()
                    && c10.1.is_finite()
                    && c01.0.is_finite()
                    && c01.1.is_finite()
                    && c11.0.is_finite()
                    && c11.1.is_finite();

                let xm = (x0 + x1) / 2;
                let ym = (y0 + y1) / 2;

                if all_finite {
                    let tx = (xm - x0) as f32 / (w as f32);
                    let ty = (ym - y0) as f32 / (h as f32);
                    let interp = |fx: f32, fy: f32| -> (f32, f32) {
                        let top_u = c00.0 + (c10.0 - c00.0) * fx;
                        let bot_u = c01.0 + (c11.0 - c01.0) * fx;
                        let top_v = c00.1 + (c10.1 - c00.1) * fx;
                        let bot_v = c01.1 + (c11.1 - c01.1) * fx;
                        (top_u + (bot_u - top_u) * fy, top_v + (bot_v - top_v) * fy)
                    };

                    let cm = project_px(xm, ym);
                    if cm.0.is_finite() && cm.1.is_finite() {
                        let est_m = interp(tx, ty);
                        let err_m = (cm.0 - est_m.0).abs().max((cm.1 - est_m.1).abs());

                        // Fast empty-cell reject when all 5 sample points lie strictly outside on
                        // the same side and curvature is bounded.
                        let margin = 2.0 + err_m * 2.0;
                        let all_left = c00.0 < -margin
                            && c10.0 < -margin
                            && c01.0 < -margin
                            && c11.0 < -margin
                            && cm.0 < -margin;
                        let all_right = c00.0 > fw + margin
                            && c10.0 > fw + margin
                            && c01.0 > fw + margin
                            && c11.0 > fw + margin
                            && cm.0 > fw + margin;
                        let all_above = c00.1 < -margin
                            && c10.1 < -margin
                            && c01.1 < -margin
                            && c11.1 < -margin
                            && cm.1 < -margin;
                        let all_below = c00.1 > fh + margin
                            && c10.1 > fh + margin
                            && c01.1 > fh + margin
                            && c11.1 > fh + margin
                            && cm.1 > fh + margin;
                        if (all_left || all_right || all_above || all_below) && err_m < 4.0 {
                            return;
                        }

                        // Smooth-cell fast fill when center + 2 edge midpoints agree within 0.05 px
                        if err_m <= 0.05 {
                            let ct = project_px(xm, y0);
                            let cl = project_px(x0, ym);
                            let est_t = interp(tx, 0.0);
                            let est_l = interp(0.0, ty);
                            let err_t = (ct.0 - est_t.0).abs().max((ct.1 - est_t.1).abs());
                            let err_l = (cl.0 - est_l.0).abs().max((cl.1 - est_l.1).abs());
                            if ct.0.is_finite()
                                && cl.0.is_finite()
                                && err_t <= 0.05
                                && err_l <= 0.05
                            {
                                let inv_w = 1.0f32 / (w as f32);
                                let inv_h = 1.0f32 / (h as f32);
                                for py in y0..=y1 {
                                    let fy = (py - y0) as f32 * inv_h;
                                    let lu = c00.0 + (c01.0 - c00.0) * fy;
                                    let ru = c10.0 + (c11.0 - c10.0) * fy;
                                    let lv = c00.1 + (c01.1 - c00.1) * fy;
                                    let rv = c10.1 + (c11.1 - c10.1) * fy;
                                    let du = (ru - lu) * inv_w;
                                    let dv = (rv - lv) * inv_w;
                                    let row_off = py * N;
                                    let mut uu = lu;
                                    let mut vv = lv;
                                    for px in x0..=x1 {
                                        u[row_off + px] = uu;
                                        v[row_off + px] = vv;
                                        if uu >= -1.0 && uu <= fw && vv >= -1.0 && vv <= fh {
                                            *any_in = true;
                                            *min_u = (*min_u).min(uu);
                                            *max_u = (*max_u).max(uu);
                                            *min_v = (*min_v).min(vv);
                                            *max_v = (*max_v).max(vv);
                                        }
                                        uu += du;
                                        vv += dv;
                                    }
                                }
                                return;
                            }
                        }
                    }
                }

                // Subdivide into 4 non-overlapping quadrants
                let q00_10 = project_px(xm, y0);
                let q00_01 = project_px(x0, ym);
                let q00_11 = project_px(xm, ym);
                fill_cell::<N, F>(
                    x0, y0, xm, ym, c00, q00_10, q00_01, q00_11, fw, fh, project_px, u, v, any_in,
                    min_u, max_u, min_v, max_v,
                );

                if xm + 1 <= x1 {
                    let q10_00 = project_px(xm + 1, y0);
                    let q10_01 = project_px(xm + 1, ym);
                    let q10_11 = project_px(x1, ym);
                    fill_cell::<N, F>(
                        xm + 1, y0, x1, ym, q10_00, c10, q10_01, q10_11, fw, fh, project_px, u, v,
                        any_in, min_u, max_u, min_v, max_v,
                    );
                }

                if ym + 1 <= y1 {
                    let q01_00 = project_px(x0, ym + 1);
                    let q01_10 = project_px(xm, ym + 1);
                    let q01_11 = project_px(xm, y1);
                    fill_cell::<N, F>(
                        x0, ym + 1, xm, y1, q01_00, q01_10, c01, q01_11, fw, fh, project_px, u, v,
                        any_in, min_u, max_u, min_v, max_v,
                    );
                }

                if xm + 1 <= x1 && ym + 1 <= y1 {
                    let q11_00 = project_px(xm + 1, ym + 1);
                    let q11_10 = project_px(x1, ym + 1);
                    let q11_01 = project_px(xm + 1, y1);
                    fill_cell::<N, F>(
                        xm + 1, ym + 1, x1, y1, q11_00, q11_10, q11_01, c11, fw, fh, project_px, u,
                        v, any_in, min_u, max_u, min_v, max_v,
                    );
                }
            }

            // Tile the N×N output into 16×16 top-level cells so even global z=0 tiles start small
            let step = 16usize.min(N);
            let mut y0 = 0usize;
            while y0 < N {
                let y1 = (y0 + step - 1).min(N - 1);
                let mut x0 = 0usize;
                while x0 < N {
                    let x1 = (x0 + step - 1).min(N - 1);
                    let c00 = project_px(x0, y0);
                    let c10 = project_px(x1, y0);
                    let c01 = project_px(x0, y1);
                    let c11 = project_px(x1, y1);
                    fill_cell::<N, _>(
                        x0, y0, x1, y1, c00, c10, c01, c11, fw, fh, &project_px, &mut u, &mut v,
                        &mut any_in, &mut min_u, &mut max_u, &mut min_v, &mut max_v,
                    );
                    x0 += step;
                }
                y0 += step;
            }

            if !any_in {
                return None;
            }
            let src_min_col = (min_u.floor() as i32 - 1).max(0);
            let src_max_col = (max_u.ceil() as i32 + 1).min(ifd_w as i32 - 1);
            let src_min_row = (min_v.floor() as i32 - 1).max(0);
            let src_max_row = (max_v.ceil() as i32 + 1).min(ifd_h as i32 - 1);
            if src_min_col > src_max_col || src_min_row > src_max_row {
                return None;
            }

            Some(Self::Exact2D {
                u,
                v,
                src_min_col,
                src_max_col,
                src_min_row,
                src_max_row,
                all_in_bounds: false,
            })
        } else {
            // 17x17 control grid spanning pixel centers [0.5 .. N - 0.5] across 16 intervals
            let mut grid_u = [0.0f32; GRID_DIM * GRID_DIM];
            let mut grid_v = [0.0f32; GRID_DIM * GRID_DIM];
            let mut min_u = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;

            let cell_px = (N as f64) / ((GRID_DIM - 1) as f64);

            for gy in 0..GRID_DIM {
                let py_f = (gy as f64) * cell_px;
                let my = max_y - (py_f + 0.5) * step_y;
                for gx in 0..GRID_DIM {
                    let px_f = (gx as f64) * cell_px;
                    let mx = min_x + (px_f + 0.5) * step_x;
                    let (sx, sy) = crs.from_3857(mx, my);
                    let (col, row) = crs_to_pixel.forward(sx, sy);
                    let uu = (col - 0.5) as f32;
                    let vv = (row - 0.5) as f32;
                    let idx = gy * GRID_DIM + gx;
                    grid_u[idx] = uu;
                    grid_v[idx] = vv;
                    if uu.is_finite() && vv.is_finite() {
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    }
                }
            }

            if !min_u.is_finite()
                || max_u < -0.5
                || min_u > (ifd_w as f32 - 0.5)
                || max_v < -0.5
                || min_v > (ifd_h as f32 - 0.5)
            {
                return None;
            }

            let src_min_col = (min_u.floor() as i32).saturating_sub(1).max(0);
            let src_max_col = (max_u.ceil() as i32).saturating_add(1).min(ifd_w as i32 - 1);
            let src_min_row = (min_v.floor() as i32).saturating_sub(1).max(0);
            let src_max_row = (max_v.ceil() as i32).saturating_add(1).min(ifd_h as i32 - 1);

            if src_min_col > src_max_col || src_min_row > src_max_row {
                return None;
            }

            let all_finite = grid_u.iter().all(|x| x.is_finite()) && grid_v.iter().all(|y| y.is_finite());
            let all_in_bounds = all_finite
                && min_u >= 0.0
                && min_v >= 0.0
                && (max_u.floor() as i32).saturating_add(1) < ifd_w as i32
                && (max_v.floor() as i32).saturating_add(1) < ifd_h as i32;

            Some(Self::Grid2D {
                grid_u,
                grid_v,
                src_min_col,
                src_max_col,
                src_min_row,
                src_max_row,
                all_in_bounds,
            })
        }
    }

    /// True iff EVERY output pixel's bilinear footprint lies strictly inside `[0..ifd_w, 0..ifd_h]`.
    #[inline(always)]
    pub fn all_in_bounds(&self) -> bool {
        match self {
            Self::Separable1D { all_in_bounds, .. }
            | Self::Grid2D { all_in_bounds, .. }
            | Self::Exact2D { all_in_bounds, .. } => *all_in_bounds,
        }
    }

    #[inline(always)]
    pub fn src_pixel_bounds(&self) -> (i32, i32, i32, i32) {
        match self {
            Self::Separable1D {
                src_min_col,
                src_max_col,
                src_min_row,
                src_max_row,
                ..
            }
            | Self::Grid2D {
                src_min_col,
                src_max_col,
                src_min_row,
                src_max_row,
                ..
            }
            | Self::Exact2D {
                src_min_col,
                src_max_col,
                src_min_row,
                src_max_row,
                ..
            } => (*src_min_col, *src_max_col, *src_min_row, *src_max_row),
        }
    }

    /// Inclusive internal COG tile coordinate range `(tx_min..=tx_max, ty_min..=ty_max)`.
    #[inline(always)]
    pub fn src_tile_range(
        &self,
        tile_w: u32,
        tile_h: u32,
        tiles_across: u32,
        tiles_down: u32,
    ) -> (u32, u32, u32, u32) {
        let (c0, c1, r0, r1) = self.src_pixel_bounds();
        let tx0 = (c0.max(0) as u32) / tile_w;
        let tx1 = ((c1.max(0) as u32) / tile_w).min(tiles_across.saturating_sub(1));
        let ty0 = (r0.max(0) as u32) / tile_h;
        let ty1 = ((r1.max(0) as u32) / tile_h).min(tiles_down.saturating_sub(1));
        (tx0, tx1, ty0, ty1)
    }
}

#[cfg(test)]
mod tests {
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
            let plan = CoordPlan::<256>::build(&crs, &c2p, w, h, tile)
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
            ("an EPSG code outside the list", keys(|k| k.projected_cs_type = Some(27700))),
            ("NAD27 UTM", keys(|k| k.projected_cs_type = Some(26710))),
            ("NAD27 CONUS Albers", keys(|k| k.projected_cs_type = Some(5069))),
            ("Swiss LV95 (oblique Mercator)", keys(|k| k.projected_cs_type = Some(2056))),
            (
                "an unknown code even with parameter keys beside it",
                keys(|k| {
                    k.projected_cs_type = Some(26710);
                    k.proj_coord_trans = Some(1);
                    k.nat_origin_long = Some(-123.0);
                }),
            ),
            (
                "NAD27 geographic",
                keys(|k| {
                    k.model_type = Some(2);
                    k.geographic_type = Some(4267);
                }),
            ),
            (
                "a user-defined projection on a WGS84 geographic CRS that is not implemented",
                keys(|k| {
                    k.model_type = Some(1);
                    k.projected_cs_type = Some(32767);
                    k.proj_coord_trans = Some(7); // Mercator
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
        let mb = CrsModel::Epsg3857.wgs84_bounds(&merc_aff, 200, 200);
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
        let gb = CrsModel::Geographic.wgs84_bounds(&geo_aff, 100, 100);
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
}
