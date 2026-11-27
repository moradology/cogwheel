use cogzilla_raster::Affine2D;
use cogzilla_tiff::GeoKeys;
use thiserror::Error;

use crate::ellipsoid::{
    AlbersEqualArea, CylindricalEqualArea, Ellipsoid, EqualEarth, Equirectangular,
    HelmertTransform, LambertAzimuthalEqualArea, LambertConformalConic, Mercator,
    ObliqueStereographic, PolarStereographic, Sinusoidal, SwissObliqueMercator, TransverseMercator,
};
pub use crate::plan::{CoordPlan, GRID_DIM};
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
    /// WGS84 Ellipsoidal World Mercator (`EPSG:3395`).
    WorldMercator,
    Geographic,
    TransverseMercator(TransverseMercator),
    Albers(AlbersEqualArea),
    LambertConformalConic(LambertConformalConic),
    PolarStereographic(PolarStereographic),
    Laea(LambertAzimuthalEqualArea),
    Sinusoidal(Sinusoidal),
    CylindricalEqualArea(CylindricalEqualArea),
    EqualEarth(EqualEarth),
    Equirectangular(Equirectangular),
    Mercator(Mercator),
    ObliqueStereographic(ObliqueStereographic),
    SwissObliqueMercator(SwissObliqueMercator),
    /// Any CRS defined on a non-WGS84 datum, bridged to WGS84 via a 7-parameter Helmert shift.
    DatumShifted {
        inner: Box<CrsModel>,
        helmert: HelmertTransform,
    },
}

/// Well-known projected EPSG codes outside the WGS84-like table:
/// Equirectangular, Mercator variants, Oblique Stereographic (Dutch RD New),
/// Swiss Oblique Mercator (LV95/LV03), British National Grid (OSGB36), and NAD27 UTM.
fn well_known_special_projected_epsg(code: u16) -> Option<CrsModel> {
    match code {
        // Equirectangular / Plate Carrée (WGS84)
        32662 | 32663 | 4087 => Some(CrsModel::Equirectangular(Equirectangular::new(
            Ellipsoid::WGS84,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ))),
        // WGS84 / PDC Mercator (lon0 = 150°E)
        3832 => Some(CrsModel::Mercator(Mercator::new_1sp(
            Ellipsoid::WGS84,
            150.0,
            1.0,
            0.0,
            0.0,
        ))),
        // WGS84 / Mercator 41 (lat_ts = -41°, lon0 = 100°E)
        3994 => Some(CrsModel::Mercator(Mercator::new_2sp(
            Ellipsoid::WGS84,
            -41.0,
            100.0,
            0.0,
            0.0,
        ))),
        // OSGB36 / British National Grid (Airy 1830 TM + EPSG:1314 Helmert)
        27700 => Some(CrsModel::DatumShifted {
            inner: Box::new(CrsModel::TransverseMercator(TransverseMercator::new(
                Ellipsoid::AIRY_1830,
                -2.0,
                49.0,
                0.999_601_271_7,
                400_000.0,
                -100_000.0,
            ))),
            helmert: HelmertTransform::OSGB36_TO_WGS84,
        }),
        // Amersfoort / RD New (Dutch Oblique Stereographic + EPSG:15934 Helmert)
        28992 => Some(CrsModel::DatumShifted {
            inner: Box::new(CrsModel::ObliqueStereographic(ObliqueStereographic::new(
                Ellipsoid::BESSEL_1841,
                52.156_160_555_555_55,
                5.387_638_888_888_89,
                0.999_907_9,
                155_000.0,
                463_000.0,
            ))),
            helmert: HelmertTransform::AMERSFOORT_TO_WGS84,
        }),
        // CH1903+ / LV95 (Swiss Oblique Mercator + EPSG:1676 Helmert)
        2056 => Some(CrsModel::DatumShifted {
            inner: Box::new(CrsModel::SwissObliqueMercator(SwissObliqueMercator::new(
                Ellipsoid::BESSEL_1841,
                46.952_405_555_555_56,
                7.439_583_333_333_333,
                2_600_000.0,
                1_200_000.0,
            ))),
            helmert: HelmertTransform::CH1903_TO_WGS84,
        }),
        // CH1903 / LV03 (Swiss Oblique Mercator + EPSG:1676 Helmert)
        21781 => Some(CrsModel::DatumShifted {
            inner: Box::new(CrsModel::SwissObliqueMercator(SwissObliqueMercator::new(
                Ellipsoid::BESSEL_1841,
                46.952_405_555_555_56,
                7.439_583_333_333_333,
                600_000.0,
                200_000.0,
            ))),
            helmert: HelmertTransform::CH1903_TO_WGS84,
        }),
        // NAD27 / UTM zones 1N–23N
        26701..=26723 => {
            let zone = (code - 26700) as u8;
            Some(CrsModel::DatumShifted {
                inner: Box::new(CrsModel::TransverseMercator(TransverseMercator::utm(
                    Ellipsoid::CLARKE_1866,
                    zone,
                    false,
                ))),
                helmert: HelmertTransform::NAD27_TO_WGS84,
            })
        }
        _ => None,
    }
}

/// Well-known geographic (degree-based) EPSG codes on non-WGS84 datums supported via Helmert shifts.
fn well_known_special_geographic_epsg(code: u16) -> Option<CrsModel> {
    match code {
        // OSGB36 Geographic
        4277 => Some(CrsModel::DatumShifted {
            inner: Box::new(CrsModel::Geographic),
            helmert: HelmertTransform::OSGB36_TO_WGS84,
        }),
        // Amersfoort Geographic
        4289 => Some(CrsModel::DatumShifted {
            inner: Box::new(CrsModel::Geographic),
            helmert: HelmertTransform::AMERSFOORT_TO_WGS84,
        }),
        // CH1903 / CH1903+ Geographic
        4149 | 4150 => Some(CrsModel::DatumShifted {
            inner: Box::new(CrsModel::Geographic),
            helmert: HelmertTransform::CH1903_TO_WGS84,
        }),
        // NAD27 Geographic (Clarke 1866 + EPSG:1173 CONUS Helmert)
        4267 => Some(CrsModel::DatumShifted {
            inner: Box::new(CrsModel::Geographic),
            helmert: HelmertTransform::NAD27_TO_WGS84,
        }),
        _ => None,
    }
}

impl CrsModel {
    /// Returns `true` for geographic (degree-based) CRSes, including datum-shifted geographic
    /// CRSes such as NAD27 (`EPSG:4267`), OSGB36 (`EPSG:4277`), Amersfoort (`EPSG:4289`), and
    /// CH1903 (`EPSG:4149`/`4150`).
    #[inline]
    pub fn is_geographic(&self) -> bool {
        matches!(self.unshifted(), Self::Geographic)
    }

    /// Returns `true` for degree-based CRSes (currently identical to [`Self::is_geographic`]).
    #[inline]
    pub fn is_degree_based(&self) -> bool {
        self.is_geographic()
    }

    /// Returns `true` for Web Mercator (`EPSG:3857`), including any datum-shifted wrapper.
    #[inline]
    pub fn is_web_mercator(&self) -> bool {
        matches!(self.unshifted(), Self::Epsg3857)
    }

    /// Unwrap any outer `DatumShifted` wrapper to inspect the underlying projection model.
    #[inline]
    pub(crate) fn unshifted(&self) -> &Self {
        match self {
            Self::DatumShifted { inner, .. } => inner.unshifted(),
            other => other,
        }
    }

    /// Construct a `CrsModel` directly from an EPSG code (supporting `4326`, `3857`, all 270+
    /// WGS84-compatible geographic codes, 3,260+ projected EPSG codes, plus Equirectangular,
    /// Mercator variants, Oblique Stereographic, Swiss LV95/LV03, OSGB36 BNG, and NAD27).
    pub fn from_epsg(code: u16) -> Result<Self, ProjError> {
        if code == 4326 || crate::epsg_table::WGS84_LIKE_GCS.binary_search(&code).is_ok() {
            return Ok(Self::Geographic);
        }
        if matches!(code, 3857 | 3785) {
            return Ok(Self::Epsg3857);
        }
        if code == 3395 {
            return Ok(Self::WorldMercator);
        }
        if let Some(model) = well_known_special_projected_epsg(code)
            .or_else(|| well_known_special_geographic_epsg(code))
        {
            return Ok(model);
        }
        if let Some(def) = crate::epsg::lookup(code) {
            return Ok(def.to_model());
        }
        Err(ProjError::UnsupportedCrs {
            reason: format!("EPSG:{code} is not in the supported list"),
            projected: Some(code),
            geographic: None,
            coord_trans: None,
        })
    }

    /// Linear unit conversion factor to metres (`1.0` for metre-based and geographic CRSes;
    /// `0.3048` / `1200/3937` for international/US survey foot projected CRSes).
    #[inline]
    pub fn linear_unit_metres(&self) -> f64 {
        match self {
            Self::TransverseMercator(p) => p.unit,
            Self::Albers(p) => p.unit,
            Self::LambertConformalConic(p) => p.unit,
            Self::PolarStereographic(p) => p.unit,
            Self::Laea(p) => p.unit,
            Self::Sinusoidal(p) => p.unit,
            Self::CylindricalEqualArea(p) => p.unit,
            Self::EqualEarth(p) => p.unit,
            Self::Equirectangular(p) => p.unit,
            Self::Mercator(p) => p.unit,
            Self::ObliqueStereographic(p) => p.unit,
            Self::SwissObliqueMercator(p) => p.unit,
            Self::DatumShifted { inner, .. } => inner.linear_unit_metres(),
            Self::Epsg3857 | Self::WorldMercator | Self::Geographic => 1.0,
        }
    }

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
            Some(3395) => {
                if matches!(gk.proj_linear_units, Some(u) if u != 9001 && u != 32767) {
                    return Err(unsupported("EPSG:3395 with non-metre linear units override"));
                }
                return Ok(Self::WorldMercator);
            }
            // User-defined: described by the parameter keys below.
            Some(32767) | None => {}
            Some(code) => {
                if let Some(model) = well_known_special_projected_epsg(code) {
                    if matches!(gk.proj_linear_units, Some(u) if u != 9001 && u != 32767) {
                        return Err(unsupported(
                            "projected CRS linear units override conflicts with EPSG definition",
                        ));
                    }
                    return Ok(model);
                }
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

        // Geographic CRS on a supported shifted datum (e.g. NAD27 4267, OSGB36 4277, Amersfoort 4289, CH1903 4149/4150).
        if gk.proj_coord_trans.is_none()
            && (gk.model_type == Some(2) || (gk.model_type.is_none() && gk.geographic_type.is_some()))
            && let Some(gcs) = gk.geographic_type
                && let Some(model) = well_known_special_geographic_epsg(gcs)
            {
                return Ok(model);
            }

        let sphere_ok = matches!(gk.proj_coord_trans, Some(7 | 10 | 17 | 24 | 28));
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
                // CT_CylindricalEqualArea
                28 => {
                    let lat_ts = require_lat(
                        gk.std_parallel_1
                            .or(gk.nat_origin_lat)
                            .unwrap_or(0.0),
                        true,
                        "standard parallel",
                    )?;
                    let lon0 = require_finite(
                        gk.nat_origin_long
                            .or(gk.center_long)
                            .unwrap_or(0.0),
                        "central meridian",
                    )?;
                    let mut p = CylindricalEqualArea::new(ellip, lat_ts, lon0, fe, fn_m);
                    p.unit = unit;
                    Ok(Self::CylindricalEqualArea(p))
                }
                // CT_Mercator (1SP / 2SP)
                7 => {
                    let lon0 = require_finite(lon0.unwrap_or(0.0), "central meridian")?;
                    let lat_ts_opt = gk.std_parallel_1.or_else(|| {
                        gk.nat_origin_lat.filter(|v| v.abs() > 1e-9)
                    });
                    let mut p = if let Some(lat_ts_raw) = lat_ts_opt {
                        let lat_ts = require_lat(lat_ts_raw, true, "standard parallel")?;
                        Mercator::new_2sp(ellip, lat_ts, lon0, fe, fn_m)
                    } else {
                        let k0 = require_scale(gk.scale_at_nat_origin.unwrap_or(1.0))?;
                        Mercator::new_1sp(ellip, lon0, k0, fe, fn_m)
                    };
                    p.unit = unit;
                    Ok(Self::Mercator(p))
                }
                // CT_ObliqueStereographic
                16 => {
                    let lat0 = require_lat(
                        lat0.ok_or_else(|| missing("latitude of origin"))?,
                        true,
                        "latitude of origin",
                    )?;
                    let lon0 = require_finite(
                        lon0.ok_or_else(|| missing("central meridian"))?,
                        "central meridian",
                    )?;
                    let k0 = require_scale(gk.scale_at_nat_origin.unwrap_or(1.0))?;
                    let mut p = ObliqueStereographic::new(ellip, lat0, lon0, k0, fe, fn_m);
                    p.unit = unit;
                    Ok(Self::ObliqueStereographic(p))
                }
                // CT_Equirectangular
                17 => {
                    let lat_ts = require_lat(
                        gk.std_parallel_1.unwrap_or(0.0),
                        true,
                        "standard parallel",
                    )?;
                    let lat0 = require_lat(lat0.unwrap_or(0.0), false, "latitude of origin")?;
                    let lon0 = require_finite(lon0.unwrap_or(0.0), "central meridian")?;
                    let mut p = Equirectangular::new(ellip, lat_ts, lat0, lon0, fe, fn_m);
                    p.unit = unit;
                    Ok(Self::Equirectangular(p))
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
            Self::WorldMercator => {
                let (lon, lat) = mercator_to_lonlat_deg(mx, my);
                world_mercator_forward(lon, lat)
            }
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
            Self::CylindricalEqualArea(p) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                p.forward(lon, lat)
            }
            Self::EqualEarth(p) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                p.forward(lon, lat)
            }
            Self::Equirectangular(p) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                p.forward(lon, lat)
            }
            Self::Mercator(p) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                p.forward(lon, lat)
            }
            Self::ObliqueStereographic(p) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                p.forward(lon, lat)
            }
            Self::SwissObliqueMercator(p) => {
                let (lon, lat) = mercator_to_lonlat_rad(mx, my);
                p.forward(lon, lat)
            }
            Self::DatumShifted { inner, helmert } => {
                let (lon, lat) = mercator_to_lonlat_deg(mx, my);
                let (dlon, dlat) = helmert.wgs84_to_datum_deg(lon, lat);
                inner.from_lonlat_deg(dlon, dlat)
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
            Self::WorldMercator => world_mercator_forward(lon_deg, lat_deg),
            Self::Geographic => (lon_deg, lat_deg),
            Self::TransverseMercator(tm) => tm.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::Albers(alb) => alb.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::LambertConformalConic(lcc) => {
                lcc.forward(lon_deg.to_radians(), lat_deg.to_radians())
            }
            Self::PolarStereographic(ps) => ps.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::Laea(p) => p.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::Sinusoidal(p) => p.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::CylindricalEqualArea(p) => {
                p.forward(lon_deg.to_radians(), lat_deg.to_radians())
            }
            Self::EqualEarth(p) => p.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::Equirectangular(p) => p.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::Mercator(p) => p.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::ObliqueStereographic(p) => p.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::SwissObliqueMercator(p) => p.forward(lon_deg.to_radians(), lat_deg.to_radians()),
            Self::DatumShifted { inner, helmert } => {
                let (dlon, dlat) = helmert.wgs84_to_datum_deg(lon_deg, lat_deg);
                inner.from_lonlat_deg(dlon, dlat)
            }
        }
    }

    /// Inverse map a point `(x, y)` in this CRS back to WGS84 `(lon_deg, lat_deg)`,
    /// normalizing longitude into `[-180, 180]`.
    pub fn to_lonlat_deg(&self, x: f64, y: f64) -> (f64, f64) {
        let (lon, lat) = match self {
            Self::Epsg3857 => mercator_to_lonlat_deg(x, y),
            Self::WorldMercator => world_mercator_inverse(x, y),
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
            Self::CylindricalEqualArea(p) => {
                let (lon, lat) = p.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::EqualEarth(p) => {
                let (lon, lat) = p.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::Equirectangular(p) => {
                let (lon, lat) = p.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::Mercator(p) => {
                let (lon, lat) = p.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::ObliqueStereographic(p) => {
                let (lon, lat) = p.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::SwissObliqueMercator(p) => {
                let (lon, lat) = p.inverse(x, y);
                (lon.to_degrees(), lat.to_degrees())
            }
            Self::DatumShifted { inner, helmert } => {
                let (dlon, dlat) = inner.to_lonlat_deg(x, y);
                helmert.datum_to_wgs84_deg(dlon, dlat)
            }
        };
        let norm_lon = if lon.is_finite() && !(-180.0..=180.0).contains(&lon) {
            ((lon + 180.0).rem_euclid(360.0)) - 180.0
        } else {
            lon
        };
        (norm_lon, lat)
    }

    /// Returns `Some(min_lon)` when this is a Geographic CRS whose raster extent uses `0..360`
    /// longitudes (including pixel-centered NOAA GFS / ECMWF ERA5 grids with `-dx/2` western
    /// overhang), so longitudes `< min_lon` should be wrapped by `+360.0` while longitudes in
    /// `[min_lon, 0.0)` remain directly in column 0.
    pub fn geographic_0_360_min_lon(&self, pixel_to_crs: &Affine2D, width: u32) -> Option<f64> {
        if !self.is_geographic() {
            return None;
        }
        let (x0, _) = pixel_to_crs.forward(0.0, 0.0);
        let (x1, _) = pixel_to_crs.forward(width as f64, 0.0);
        let lo = x0.min(x1);
        let hi = x0.max(x1);
        let px_w = ((hi - lo) / (width.max(1) as f64)).max(1e-6);
        (lo >= -px_w && hi > 180.0 + 1e-6).then_some(lo)
    }

    /// Returns `Some(max_lon)` when this is a Geographic CRS whose raster extent uses negative
    /// west-antimeridian longitudes (e.g. `[-190, -170]`), so longitudes `> max_lon` (e.g. `+175°E`)
    /// should be wrapped by `-360.0` into the raster's domain.
    pub fn geographic_neg_180_max_lon(&self, pixel_to_crs: &Affine2D, width: u32) -> Option<f64> {
        if !self.is_geographic() {
            return None;
        }
        let (x0, _) = pixel_to_crs.forward(0.0, 0.0);
        let (x1, _) = pixel_to_crs.forward(width as f64, 0.0);
        let lo = x0.min(x1);
        let hi = x0.max(x1);
        let px_w = ((hi - lo) / (width.max(1) as f64)).max(1e-6);
        (lo < -180.0 - 1e-6 && hi <= px_w).then_some(hi)
    }

    /// Returns `true` when this is a Geographic CRS whose raster extent uses `0..360` longitudes
    /// (including pixel-centered NOAA GFS / ECMWF ERA5 grids with `-dx/2` western overhang).
    pub fn is_geographic_0_360(&self, pixel_to_crs: &Affine2D, width: u32) -> bool {
        self.geographic_0_360_min_lon(pixel_to_crs, width).is_some()
    }

    /// Exact WGS84 geographic bounding box `[min_lon, min_lat, max_lon, max_lat]`, using a
    /// nonlinear `GcpPoly` when present or delegating directly to `wgs84_bounds` otherwise.
    pub fn wgs84_bounds_with_gcp(
        &self,
        pixel_to_crs: &Affine2D,
        gcp_poly: Option<&cogzilla_tiff::GcpPoly>,
        width: u32,
        height: u32,
    ) -> [f64; 4] {
        let Some(poly) = gcp_poly else {
            return self.wgs84_bounds(pixel_to_crs, width, height);
        };
        let w = width as f64;
        let h = height as f64;
        let mut min_lon = f64::INFINITY;
        let mut min_lat = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let steps = 16;
        for i in 0..=steps {
            let t = (i as f64) / (steps as f64);
            for (col, row) in [(t * w, 0.0), (t * w, h), (0.0, t * h), (w, t * h)] {
                let (sx, sy) = poly.pixel_to_crs(col, row);
                let (lon, lat) = self.to_lonlat_deg(sx, sy);
                if lon.is_finite() && lat.is_finite() {
                    min_lon = min_lon.min(lon);
                    max_lon = max_lon.max(lon);
                    min_lat = min_lat.min(lat);
                    max_lat = max_lat.max(lat);
                }
            }
        }
        if !min_lon.is_finite() || !min_lat.is_finite() {
            return self.wgs84_bounds(pixel_to_crs, width, height);
        }
        [min_lon, min_lat, max_lon, max_lat]
    }

    /// Exact WGS84 geographic bounding box `[min_lon, min_lat, max_lon, max_lat]` by sampling
    /// 32 points along the 4 edges of the raster plus checking whether either pole lies inside.
    pub fn wgs84_bounds(&self, pixel_to_crs: &Affine2D, width: u32, height: u32) -> [f64; 4] {
        let w = width as f64;
        let h = height as f64;
        if matches!(self, Self::Geographic) {
            let mut min_lon = f64::INFINITY;
            let mut min_lat = f64::INFINITY;
            let mut max_lon = f64::NEG_INFINITY;
            let mut max_lat = f64::NEG_INFINITY;
            for &(col, row) in &[(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
                let (lon, lat) = pixel_to_crs.forward(col, row);
                if lon.is_finite() && lat.is_finite() {
                    min_lon = min_lon.min(lon);
                    max_lon = max_lon.max(lon);
                    min_lat = min_lat.min(lat);
                    max_lat = max_lat.max(lat);
                }
            }
            if !min_lon.is_finite() || !min_lat.is_finite() {
                return [-180.0, -85.05112878, 180.0, 85.05112878];
            }
            // Global 0..360° grids (e.g. NOAA GFS / ECMWF ERA5) normalize to [-180, 180],
            // while regional antimeridian-crossing files (e.g. 170..190 or -190..-170) keep
            // their native geographic bounds so map clients don't zoom out to the whole world.
            if self.is_geographic_0_360(pixel_to_crs, width) && max_lon - min_lon >= 359.0 {
                return [-180.0, min_lat, 180.0, max_lat];
            }
            return [min_lon, min_lat, max_lon, max_lat];
        }
        let mut min_lon = f64::INFINITY;
        let mut min_lat = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let mut crosses_antimeridian = false;
        let steps = 8;

        let sample_pt = |col: f64, row: f64| -> Option<(f64, f64)> {
            let (sx, sy) = pixel_to_crs.forward(col, row);
            let (lon, lat) = self.to_lonlat_deg(sx, sy);
            if lon.is_finite() && lat.is_finite() {
                Some((lon, lat))
            } else {
                None
            }
        };

        let mut visit_edge = |pts: &[(f64, f64)]| {
            let mut prev_lon: Option<f64> = None;
            for &(col, row) in pts {
                if let Some((lon, lat)) = sample_pt(col, row) {
                    if let Some(pl) = prev_lon
                        && (lon - pl).abs() > 180.0 {
                            crosses_antimeridian = true;
                        }
                    prev_lon = Some(lon);
                    min_lon = min_lon.min(lon);
                    max_lon = max_lon.max(lon);
                    min_lat = min_lat.min(lat);
                    max_lat = max_lat.max(lat);
                }
            }
        };

        let top: Vec<(f64, f64)> = (0..=steps).map(|i| (i as f64 / steps as f64 * w, 0.0)).collect();
        let bot: Vec<(f64, f64)> = (0..=steps).map(|i| (i as f64 / steps as f64 * w, h)).collect();
        let left: Vec<(f64, f64)> = (0..=steps).map(|i| (0.0, i as f64 / steps as f64 * h)).collect();
        let right: Vec<(f64, f64)> = (0..=steps).map(|i| (w, i as f64 / steps as f64 * h)).collect();
        visit_edge(&top);
        visit_edge(&bot);
        visit_edge(&left);
        visit_edge(&right);

        if crosses_antimeridian {
            min_lon = -180.0;
            max_lon = 180.0;
        }

        // Check whether North or South Pole lies inside the raster footprint.
        // For polar/azimuthal projections, enclosing a pole means the raster spans all longitudes
        // [-180, 180]; for cylindrical/pseudocylindrical projections (CEA, EqualEarth), poles are
        // horizontal lines, so only extend min_lat / max_lat without expanding longitude.
        if !matches!(self.unshifted(), Self::Epsg3857 | Self::Geographic | Self::WorldMercator | Self::Mercator(_)) {
            let is_line_pole = matches!(
                self.unshifted(),
                Self::CylindricalEqualArea(_) | Self::EqualEarth(_) | Self::Equirectangular(_)
            );
            if let Some(crs_to_pixel) = pixel_to_crs.invert() {
                for &pole_lat in &[90.0_f64, -90.0_f64] {
                    let (px_crs, py_crs) = self.from_lonlat_deg(0.0, pole_lat);
                    if px_crs.is_finite() && py_crs.is_finite() {
                        let (col, row) = crs_to_pixel.forward(px_crs, py_crs);
                        if col >= 0.0 && col <= w && row >= 0.0 && row <= h {
                            if !is_line_pole {
                                min_lon = -180.0;
                                max_lon = 180.0;
                            }
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

        // Fallback if all border samples fell outside a lobe-bounded projection (e.g. Sinusoidal)
        if !min_lon.is_finite() || !min_lat.is_finite() {
            for iy in 1..=3 {
                for ix in 1..=3 {
                    if let Some((lon, lat)) = sample_pt(w * ix as f64 / 4.0, h * iy as f64 / 4.0) {
                        min_lon = min_lon.min(lon);
                        max_lon = max_lon.max(lon);
                        min_lat = min_lat.min(lat);
                        max_lat = max_lat.max(lat);
                    }
                }
            }
        }
        if !min_lon.is_finite() || !min_lat.is_finite() {
            return [-180.0, -85.05112878, 180.0, 85.05112878];
        }
        if self.is_geographic_0_360(pixel_to_crs, width) {
            if max_lon - min_lon >= 359.0 {
                return [-180.0, min_lat, 180.0, max_lat];
            } else if min_lon >= 180.0 {
                return [min_lon - 360.0, min_lat, max_lon - 360.0, max_lat];
            } else if max_lon > 180.0 {
                return [min_lon, min_lat, max_lon - 360.0, max_lat];
            }
        }

        [min_lon, min_lat, max_lon, max_lat]
    }

    /// Estimate TiTiler-compatible `(minzoom, maxzoom)` in `WebMercatorQuad` for a raster,
    /// matching `rio-tiler`'s `SpatialMixin` (`GDALSuggestedWarpOutput2` diagonal resolution,
    /// `get_maximum_overview_level(..., minsize=256)`, and `morecantile.zoom_for_res("auto")`).
    pub fn web_mercator_zoom_range(
        &self,
        pixel_to_crs: &Affine2D,
        width: u32,
        height: u32,
        _overview_factor: f64,
    ) -> (u8, u8) {
        const CELL0_MERC: f64 = 156_543.033_928_040_97;
        const MAX_Z_MERC: u8 = 24;

        let wf = (width.max(1)) as f64;
        let hf = (height.max(1)) as f64;

        let to_merc = |col: f64, row: f64| -> Option<(f64, f64)> {
            let (sx, sy) = pixel_to_crs.forward(col, row);
            let (mx, my) = match self {
                Self::Epsg3857 => (sx, sy),
                _ => {
                    let (lon, lat) = self.to_lonlat_deg(sx, sy);
                    if !lon.is_finite() || !lat.is_finite() {
                        return None;
                    }
                    lonlat_deg_to_mercator(lon, lat)
                }
            };
            if mx.is_finite() && my.is_finite() {
                Some((mx, my))
            } else {
                None
            }
        };

        let (res, warp_w, warp_h) = if matches!(self, Self::Epsg3857) {
            let dx = pixel_to_crs.a.hypot(pixel_to_crs.d);
            let dy = pixel_to_crs.b.hypot(pixel_to_crs.e);
            (dx.max(dy).max(1e-6), wf, hf)
        } else {
            // Sample 21 points along each of the 4 edges (matching GDAL / rasterio transform_bounds)
            // to compute warped Web Mercator extent and diagonal resolution.
            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            let mut all_ok = true;
            let mut crosses_antimeridian = false;
            let mut prev_x: Option<f64> = None;

            for edge in 0..4 {
                prev_x = None;
                for i in 0..=20 {
                    let t = i as f64 / 20.0;
                    let (col, row) = match edge {
                        0 => (t * wf, 0.0),
                        1 => (t * wf, hf),
                        2 => (0.0, t * hf),
                        _ => (wf, t * hf),
                    };
                    if let Some((mx, my)) = to_merc(col, row) {
                        if let Some(px) = prev_x
                            && (mx - px).abs() > ORIGIN_SHIFT {
                                crosses_antimeridian = true;
                            }
                        prev_x = Some(mx);
                        min_x = min_x.min(mx);
                        max_x = max_x.max(mx);
                        min_y = min_y.min(my);
                        max_y = max_y.max(my);
                    } else {
                        all_ok = false;
                    }
                }
            }
            let _ = prev_x;

            let ul = to_merc(0.0, 0.0);
            let lr = to_merc(wf, hf);

            if all_ok
                && !crosses_antimeridian
                && let (Some((ux, uy)), Some((lx, ly))) = (ul, lr)
                && (lx - ux).abs() < ORIGIN_SHIFT
                && (lx - ux).hypot(ly - uy) > 1e-3
            {
                let res = (lx - ux).hypot(ly - uy) / wf.hypot(hf);
                let ww = ((max_x - min_x) / res + 0.5).floor().max(1.0);
                let wh = ((max_y - min_y) / res + 0.5).floor().max(1.0);
                (res, ww, wh)
            } else {
                // Check if a pole lies inside the raster (e.g. Antarctic EPSG:3031 or Arctic UPS).
                // At/near a pole, a local center Jacobian explodes because longitude swings 90°-180°
                // across a few pixels; instead match GDAL's warped-extent diagonal resolution.
                let mut pole_inside = false;
                if !matches!(
                    self.unshifted(),
                    Self::Epsg3857
                        | Self::Geographic
                        | Self::WorldMercator
                        | Self::CylindricalEqualArea(_)
                        | Self::EqualEarth(_)
                        | Self::Equirectangular(_)
                        | Self::Mercator(_)
                )
                    && let Some(crs_to_pixel) = pixel_to_crs.invert() {
                        for &pole_lat in &[90.0_f64, -90.0_f64] {
                            let (px_crs, py_crs) = self.from_lonlat_deg(0.0, pole_lat);
                            if px_crs.is_finite() && py_crs.is_finite() {
                                let (col, row) = crs_to_pixel.forward(px_crs, py_crs);
                                if col >= 0.0 && col <= wf && row >= 0.0 && row <= hf {
                                    pole_inside = true;
                                    min_x = -ORIGIN_SHIFT;
                                    max_x = ORIGIN_SHIFT;
                                    if pole_lat > 0.0 {
                                        max_y = ORIGIN_SHIFT;
                                    } else {
                                        min_y = -ORIGIN_SHIFT;
                                    }
                                }
                            }
                        }
                    }
                if pole_inside
                    && min_x.is_finite()
                    && max_x.is_finite()
                    && min_y.is_finite()
                    && max_y.is_finite()
                {
                    let span_x = (max_x - min_x).max(1.0);
                    let span_y = (max_y - min_y).max(1.0);
                    let res = span_x.hypot(span_y) / wf.hypot(hf);
                    let ww = (span_x / res + 0.5).floor().max(1.0);
                    let wh = (span_y / res + 0.5).floor().max(1.0);
                    (res, ww, wh)
                } else {
                    // Safe fallback for antimeridian-crossing / partially out-of-domain rasters:
                    // local 1-pixel Jacobian at the raster center.
                    let cx = wf * 0.5;
                    let cy = hf * 0.5;
                    let step_px = (wf.min(hf) * 0.01).clamp(1.0, 64.0);
                    let px_m = if let (Some((mx0, my0)), Some((mxx, myx)), Some((mxy, myy))) = (
                        to_merc(cx, cy),
                        to_merc(cx + step_px, cy),
                        to_merc(cx, cy + step_px),
                    ) {
                        let wrap_dx = |dx: f64| -> f64 {
                            if dx > ORIGIN_SHIFT {
                                dx - 2.0 * ORIGIN_SHIFT
                            } else if dx < -ORIGIN_SHIFT {
                                dx + 2.0 * ORIGIN_SHIFT
                            } else {
                                dx
                            }
                        };
                        let dx_m = wrap_dx(mxx - mx0).hypot(myx - my0) / step_px;
                        let dy_m = wrap_dx(mxy - mx0).hypot(myy - my0) / step_px;
                        let local = dx_m.max(dy_m);
                        if local.is_finite() && local > 1e-3 {
                            local
                        } else {
                            let [w, s, e, n] = self.wgs84_bounds(pixel_to_crs, width, height);
                            let (a0, b0) = lonlat_deg_to_mercator(w, s);
                            let (a1, b1) = lonlat_deg_to_mercator(e, n);
                            (a1 - a0).abs().max((b1 - b0).abs()).max(1.0) / wf.max(hf)
                        }
                    } else {
                        let [w, s, e, n] = self.wgs84_bounds(pixel_to_crs, width, height);
                        let (a0, b0) = lonlat_deg_to_mercator(w, s);
                        let (a1, b1) = lonlat_deg_to_mercator(e, n);
                        (a1 - a0).abs().max((b1 - b0).abs()).max(1.0) / wf.max(hf)
                    };
                    (px_m, wf, hf)
                }
            }
        };

        let ov_level = max_overview_level(warp_w, warp_h, 256.0);
        let maxzoom = zoom_for_res(res, CELL0_MERC, MAX_Z_MERC);
        let minzoom = zoom_for_res(res * f64::from(1u32 << ov_level), CELL0_MERC, MAX_Z_MERC)
            .min(maxzoom);
        (minzoom, maxzoom)
    }

    /// Estimate TiTiler-compatible `(minzoom, maxzoom)` for OGC `WorldCRS84Quad`
    /// (`cellSize(z=0) = 180° / 256 = 0.703125°`, `maxzoom = 23`), matching `rio-tiler`
    /// and `morecantile.zoom_for_res("auto")` while guarding polar `1/cos(lat)` singularities.
    pub fn world_crs84_zoom_range(
        &self,
        pixel_to_crs: &Affine2D,
        width: u32,
        height: u32,
        _overview_factor: f64,
    ) -> (u8, u8) {
        const CELL0_CRS84: f64 = 180.0 / 256.0;
        const MAX_Z_CRS84: u8 = 23;

        let wf = (width.max(1)) as f64;
        let hf = (height.max(1)) as f64;

        let (res, warp_w, warp_h) = if matches!(self, Self::Geographic) {
            let dx = pixel_to_crs.a.hypot(pixel_to_crs.d);
            let dy = pixel_to_crs.b.hypot(pixel_to_crs.e);
            (dx.max(dy).max(1e-8), wf, hf)
        } else {
            let to_ll = |col: f64, row: f64| -> Option<(f64, f64)> {
                let (sx, sy) = pixel_to_crs.forward(col, row);
                let (lon, lat) = self.to_lonlat_deg(sx, sy);
                if lon.is_finite() && lat.is_finite() {
                    Some((lon, lat))
                } else {
                    None
                }
            };

            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            let mut all_ok = true;
            let mut crosses_antimeridian = false;
            let mut max_abs_lat = 0.0_f64;

            for edge in 0..4 {
                let mut prev_lon: Option<f64> = None;
                for i in 0..=20 {
                    let t = i as f64 / 20.0;
                    let (col, row) = match edge {
                        0 => (t * wf, 0.0),
                        1 => (t * wf, hf),
                        2 => (0.0, t * hf),
                        _ => (wf, t * hf),
                    };
                    if let Some((lon, lat)) = to_ll(col, row) {
                        if let Some(pl) = prev_lon
                            && (lon - pl).abs() > 180.0 {
                                crosses_antimeridian = true;
                            }
                        prev_lon = Some(lon);
                        max_abs_lat = max_abs_lat.max(lat.abs());
                        min_x = min_x.min(lon);
                        max_x = max_x.max(lon);
                        min_y = min_y.min(lat);
                        max_y = max_y.max(lat);
                    } else {
                        all_ok = false;
                    }
                }
            }

            let ul = to_ll(0.0, 0.0);
            let lr = to_ll(wf, hf);
            // Use warped diagonal resolution for mid-latitude rasters; near poles (>80°)
            // or across the antimeridian, fall back to the cos(lat)-scaled local Jacobian
            // to avoid the 1/cos(lat) longitude blow-up.
            if all_ok
                && !crosses_antimeridian
                && max_abs_lat < 80.0
                && let (Some((ux, uy)), Some((lx, ly))) = (ul, lr)
                && (lx - ux).abs() < 180.0
                && (lx - ux).hypot(ly - uy) > 1e-8
            {
                let res = (lx - ux).hypot(ly - uy) / wf.hypot(hf);
                let ww = ((max_x - min_x) / res + 0.5).floor().max(1.0);
                let wh = ((max_y - min_y) / res + 0.5).floor().max(1.0);
                (res, ww, wh)
            } else {
                let cx = wf * 0.5;
                let cy = hf * 0.5;
                let step_px = (wf.min(hf) * 0.01).clamp(1.0, 64.0);
                let px_deg = if let (Some((lon0, lat0)), Some((lonx, latx)), Some((lony, laty))) = (
                    to_ll(cx, cy),
                    to_ll(cx + step_px, cy),
                    to_ll(cx, cy + step_px),
                ) {
                    let wrap_dlon = |d: f64| -> f64 {
                        if d > 180.0 {
                            d - 360.0
                        } else if d < -180.0 {
                            d + 360.0
                        } else {
                            d
                        }
                    };
                    let cos_lat = lat0.to_radians().cos().abs();
                    let dx = (wrap_dlon(lonx - lon0) * cos_lat).hypot(latx - lat0) / step_px;
                    let dy = (wrap_dlon(lony - lon0) * cos_lat).hypot(laty - lat0) / step_px;
                    let local = dx.max(dy);
                    if local.is_finite() && local > 1e-8 {
                        local
                    } else {
                        let [w, s, e, n] = self.wgs84_bounds(pixel_to_crs, width, height);
                        (n - s).abs().max((e - w).abs() * cos_lat).max(1e-6) / wf.max(hf)
                    }
                } else {
                    let [w, s, e, n] = self.wgs84_bounds(pixel_to_crs, width, height);
                    (e - w).abs().max((n - s).abs()).max(1e-6) / wf.max(hf)
                };
                (px_deg, wf, hf)
            }
        };

        let ov_level = max_overview_level(warp_w, warp_h, 256.0);
        let maxzoom = zoom_for_res(res, CELL0_CRS84, MAX_Z_CRS84);
        let minzoom = zoom_for_res(res * f64::from(1u32 << ov_level), CELL0_CRS84, MAX_Z_CRS84)
            .min(maxzoom);
        (minzoom, maxzoom)
    }
}

/// Maximum virtual overview level whose smallest dimension stays above `minsize` pixels,
/// matching `rio_tiler.io.base.get_maximum_overview_level`.
#[inline]
fn max_overview_level(width: f64, height: f64, minsize: f64) -> u32 {
    let mut level = 0u32;
    while level < 30 && (width.min(height) / f64::from(1u32 << level)).floor() > minsize {
        level += 1;
    }
    level
}

/// Closest TMS zoom level for a target pixel resolution `res` in TMS units, matching
/// GDAL's COG driver and `morecantile.TileMatrixSet.zoom_for_res(..., zoom_level_strategy="auto")`.
#[inline]
fn zoom_for_res(res: f64, cell0: f64, max_z: u8) -> u8 {
    if !res.is_finite() || res <= 0.0 {
        return max_z;
    }
    let cell = |z: u8| cell0 / (1u64 << z) as f64;
    let mut zoom = 0u8;
    for z in 0..=max_z {
        zoom = z;
        let cz = cell(z);
        if res > cz || (res - cz).abs() / cz <= 1e-8 {
            break;
        }
    }
    let cz = cell(zoom);
    if zoom > 0 && (res - cz).abs() / cz > 1e-8 && (cell(zoom - 1) / res) < (res / cz) {
        zoom -= 1;
    }
    zoom
}

/// WGS84 ellipsoidal World Mercator (`EPSG:3395`) forward `(lon_deg, lat_deg) -> (x, y)`.
#[inline]
fn world_mercator_forward(lon_deg: f64, lat_deg: f64) -> (f64, f64) {
    const A: f64 = 6_378_137.0;
    const E: f64 = 0.081_819_190_842_621_49;
    let lat = lat_deg.clamp(-85.084_059_050_110_45, 85.084_059_050_110_45).to_radians();
    let lon = lon_deg.to_radians();
    let x = A * lon;
    let con = E * lat.sin();
    let ts = (std::f64::consts::FRAC_PI_4 + 0.5 * lat).tan()
        * ((1.0 - con) / (1.0 + con)).powf(0.5 * E);
    let y = A * ts.ln();
    (x, y)
}

/// WGS84 ellipsoidal World Mercator (`EPSG:3395`) inverse `(x, y) -> (lon_deg, lat_deg)`.
#[inline]
fn world_mercator_inverse(x: f64, y: f64) -> (f64, f64) {
    const A: f64 = 6_378_137.0;
    const E: f64 = 0.081_819_190_842_621_49;
    let lon_deg = (x / A).to_degrees();
    let ts = (-y / A).exp();
    let mut phi = std::f64::consts::FRAC_PI_2 - 2.0 * ts.atan();
    for _ in 0..8 {
        let con = E * phi.sin();
        let next = std::f64::consts::FRAC_PI_2
            - 2.0 * (ts * ((1.0 - con) / (1.0 + con)).powf(0.5 * E)).atan();
        if (next - phi).abs() < 1e-13 {
            phi = next;
            break;
        }
        phi = next;
    }
    (lon_deg, phi.to_degrees())
}

#[cfg(test)]
mod tests;
