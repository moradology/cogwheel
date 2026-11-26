//! Lookup of projected EPSG codes in the generated table.

use crate::crs::CrsModel;
use crate::ellipsoid::{
    AlbersEqualArea, CylindricalEqualArea, Ellipsoid, EqualEarth, LambertAzimuthalEqualArea,
    LambertConformalConic, PolarStereographic, Sinusoidal, TransverseMercator,
};
use crate::epsg_table::EPSG_TABLE;

/// Projection method of a table entry. The meaning of `EpsgDef::p` depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// `[lat0, lon0, k0, x0, y0]`
    Tm,
    /// `[lat1, lat2, lat0, lon0, x0, y0]`
    Lcc2sp,
    /// `[lat0, lon0, k0, x0, y0]`
    Lcc1sp,
    /// `[lat1, lat2, lat0, lon0, x0, y0]`
    Albers,
    /// `[lat_ts, lon0, x0, y0]`
    PolarTs,
    /// `[hemisphere (+1 or -1), k0, lon0, x0, y0]`
    PolarK,
    /// `[lat0, lon0, x0, y0]`
    Laea,
    /// `[lon0, x0, y0]`
    Sinusoidal,
    /// `[lat_ts, lon0, x0, y0]`
    Cea,
    /// `[lon0, x0, y0]`
    EqualEarth,
}

/// One projected CRS. Angles are degrees; `x0`/`y0` are in the CRS's own units.
#[derive(Debug, Clone, Copy)]
pub struct EpsgDef {
    pub code: u16,
    pub method: Method,
    /// 0 metres, 1 international feet, 2 US survey feet.
    pub unit: u8,
    /// Semi-major axis and inverse flattening (0 for a sphere).
    pub a: f64,
    pub rf: f64,
    pub p: [f64; 6],
}

/// Metres per unit for a table unit id.
pub fn unit_metres(unit: u8) -> f64 {
    match unit {
        1 => 0.3048,
        2 => 1200.0 / 3937.0,
        _ => 1.0,
    }
}

impl EpsgDef {
    pub fn to_model(&self) -> CrsModel {
        let ellip = Ellipsoid::new(self.a, self.rf);
        let unit = unit_metres(self.unit);
        let p = &self.p;
        match self.method {
            Method::Tm => {
                let mut m = TransverseMercator::new(ellip, p[1], p[0], p[2], p[3], p[4]);
                m.unit = unit;
                CrsModel::TransverseMercator(m)
            }
            Method::Lcc2sp => {
                let mut m = LambertConformalConic::new_2sp(ellip, p[0], p[1], p[2], p[3], p[4], p[5]);
                m.unit = unit;
                CrsModel::LambertConformalConic(m)
            }
            Method::Lcc1sp => {
                let mut m = LambertConformalConic::new_1sp(ellip, p[0], p[1], p[2], p[3], p[4]);
                m.unit = unit;
                CrsModel::LambertConformalConic(m)
            }
            Method::Albers => {
                let mut m = AlbersEqualArea::new(ellip, p[0], p[1], p[2], p[3], p[4], p[5]);
                m.unit = unit;
                CrsModel::Albers(m)
            }
            Method::PolarTs => {
                let mut m = PolarStereographic::new(ellip, p[0], p[1], p[2], p[3]);
                m.unit = unit;
                CrsModel::PolarStereographic(m)
            }
            Method::PolarK => {
                let mut m =
                    PolarStereographic::new_pole_scale(ellip, p[0] < 0.0, p[1], p[2], p[3], p[4]);
                m.unit = unit;
                CrsModel::PolarStereographic(m)
            }
            Method::Laea => {
                let mut m = LambertAzimuthalEqualArea::new(ellip, p[0], p[1], p[2], p[3]);
                m.unit = unit;
                CrsModel::Laea(m)
            }
            Method::Sinusoidal => {
                let mut m = Sinusoidal::new(ellip, p[0], p[1], p[2]);
                m.unit = unit;
                CrsModel::Sinusoidal(m)
            }
            Method::Cea => {
                let mut m = CylindricalEqualArea::new(ellip, p[0], p[1], p[2], p[3]);
                m.unit = unit;
                CrsModel::CylindricalEqualArea(m)
            }
            Method::EqualEarth => {
                let mut m = EqualEarth::new(ellip, p[0], p[1], p[2]);
                m.unit = unit;
                CrsModel::EqualEarth(m)
            }
        }
    }
}

/// The table entry for a projected EPSG code, if cogzilla can place it.
pub fn lookup(code: u16) -> Option<&'static EpsgDef> {
    EPSG_TABLE
        .binary_search_by_key(&code, |d| d.code)
        .ok()
        .map(|i| &EPSG_TABLE[i])
}
