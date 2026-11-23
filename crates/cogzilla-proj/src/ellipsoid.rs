/// Ellipsoid parameters for closed-form geodetic projections.
#[derive(Debug, Clone, Copy)]
pub struct Ellipsoid {
    pub a: f64,
    pub f: f64,
    pub e: f64,
    pub e2: f64,
}

impl Ellipsoid {
    pub const WGS84: Self = Self::from_a_inv_f(6_378_137.0, 298.257_223_563);
    pub const GRS80: Self = Self::from_a_inv_f(6_378_137.0, 298.257_222_101);

    pub const fn from_a_inv_f(a: f64, inv_f: f64) -> Self {
        let f = 1.0 / inv_f;
        let e2 = f * (2.0 - f);
        // const sqrt isn't stable in all contexts, compute via Newton iterations at compile time
        let mut x = 0.0818191908426215_f64;
        x = 0.5 * (x + e2 / x);
        x = 0.5 * (x + e2 / x);
        x = 0.5 * (x + e2 / x);
        x = 0.5 * (x + e2 / x);
        Self { a, f, e: x, e2 }
    }

    pub fn new(a: f64, inv_f: f64) -> Self {
        let f = 1.0 / inv_f;
        let e2 = f * (2.0 - f);
        Self {
            a,
            f,
            e: e2.sqrt(),
            e2,
        }
    }
}

/// Wrap a longitude difference into `[-pi, pi]`, so a projection whose central
/// meridian is near the antimeridian still sees nearby points as nearby.
#[inline(always)]
fn wrap_dlon(dlon: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    if dlon > PI {
        dlon - TAU
    } else if dlon < -PI {
        dlon + TAU
    } else {
        dlon
    }
}

/// Krüger 4th-order Transverse Mercator (exact to < 5 nm across UTM zones, matching PROJ).
#[derive(Debug, Clone, Copy)]
pub struct TransverseMercator {
    pub lon0: f64,
    pub k0: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    pub e: f64,
    pub rect_a: f64,
    pub alpha: [f64; 4],
    pub beta: [f64; 4],
    pub delta: [f64; 4],
    pub m0: f64,
    /// Metres per CRS linear unit (1.0 for metres). False origins are in CRS units.
    pub unit: f64,
}

impl TransverseMercator {
    pub fn new(
        ellip: Ellipsoid,
        lon0_deg: f64,
        lat0_deg: f64,
        k0: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let n = ellip.f / (2.0 - ellip.f);
        let n2 = n * n;
        let n3 = n2 * n;
        let n4 = n2 * n2;

        let rect_a = (ellip.a / (1.0 + n)) * (1.0 + n2 / 4.0 + n4 / 64.0);
        let alpha = [
            0.5 * n - (2.0 / 3.0) * n2 + (5.0 / 16.0) * n3 + (41.0 / 180.0) * n4,
            (13.0 / 48.0) * n2 - (3.0 / 5.0) * n3 + (557.0 / 1440.0) * n4,
            (61.0 / 240.0) * n3 - (103.0 / 140.0) * n4,
            (49561.0 / 161280.0) * n4,
        ];
        let beta = [
            0.5 * n - (2.0 / 3.0) * n2 + (37.0 / 96.0) * n3 - (1.0 / 360.0) * n4,
            (1.0 / 48.0) * n2 + (1.0 / 15.0) * n3 - (437.0 / 1440.0) * n4,
            (17.0 / 480.0) * n3 - (37.0 / 840.0) * n4,
            (4397.0 / 161280.0) * n4,
        ];
        let delta = [
            2.0 * n - (2.0 / 3.0) * n2 - 2.0 * n3 + (116.0 / 45.0) * n4,
            (7.0 / 3.0) * n2 - (8.0 / 5.0) * n3 - (227.0 / 45.0) * n4,
            (56.0 / 15.0) * n3 - (136.0 / 35.0) * n4,
            (4279.0 / 630.0) * n4,
        ];

        let mut tm = Self {
            lon0: lon0_deg.to_radians(),
            k0,
            false_easting,
            false_northing,
            e: ellip.e,
            rect_a,
            alpha,
            beta,
            delta,
            m0: 0.0,
            unit: 1.0,
        };
        if lat0_deg != 0.0 {
            let (_, n0) = tm.forward_raw(tm.lon0, lat0_deg.to_radians());
            tm.m0 = n0;
        }
        tm
    }

    pub fn utm(ellip: Ellipsoid, zone: u8, south: bool) -> Self {
        let lon0_deg = (zone as f64) * 6.0 - 183.0;
        let fn_m = if south { 10_000_000.0 } else { 0.0 };
        Self::new(ellip, lon0_deg, 0.0, 0.9996, 500_000.0, fn_m)
    }

    #[inline(always)]
    fn forward_raw(&self, lon: f64, lat: f64) -> (f64, f64) {
        let dlon = wrap_dlon(lon - self.lon0);
        // Outside valid Transverse Mercator domain (~75 deg from central meridian)
        if dlon.abs() > 1.32 || lat.abs() > 1.56 {
            return (f64::NAN, f64::NAN);
        }
        let sin_lat = lat.sin();
        let t = (lat.tan().asinh() - self.e * (self.e * sin_lat).atanh()).sinh();
        let xi_p = t.atan2(dlon.cos());
        let cos_dlon = dlon.cos();
        let eta_p = (dlon.sin() / (t * t + cos_dlon * cos_dlon).sqrt()).asinh();

        let mut xi = xi_p;
        let mut eta = eta_p;
        for (idx, &a) in self.alpha.iter().enumerate() {
            let j = (2 * (idx + 1)) as f64;
            let (s_xi, c_xi) = (j * xi_p).sin_cos();
            let s_eta = (j * eta_p).sinh();
            let c_eta = (j * eta_p).cosh();
            xi += a * s_xi * c_eta;
            eta += a * c_xi * s_eta;
        }
        (self.k0 * self.rect_a * eta, self.k0 * self.rect_a * xi)
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        let (x, y) = self.forward_raw(lon, lat);
        (
            self.false_easting + x / self.unit,
            self.false_northing + (y - self.m0) / self.unit,
        )
    }

    /// Exact Krüger 4th-order inverse `(E, N)` -> `(lon_rad, lat_rad)`.
    pub fn inverse(&self, easting: f64, northing: f64) -> (f64, f64) {
        let xi = ((northing - self.false_northing) * self.unit + self.m0) / (self.k0 * self.rect_a);
        let eta = (easting - self.false_easting) * self.unit / (self.k0 * self.rect_a);
        let mut xi_p = xi;
        let mut eta_p = eta;
        for (idx, &b) in self.beta.iter().enumerate() {
            let j = (2 * (idx + 1)) as f64;
            let (s_xi, c_xi) = (j * xi).sin_cos();
            let s_eta = (j * eta).sinh();
            let c_eta = (j * eta).cosh();
            xi_p -= b * s_xi * c_eta;
            eta_p -= b * c_xi * s_eta;
        }
        let chi = (xi_p.sin() / eta_p.cosh()).asin();
        let mut lat = chi;
        for (idx, &d) in self.delta.iter().enumerate() {
            let j = (2 * (idx + 1)) as f64;
            lat += d * (j * chi).sin();
        }
        let lon = self.lon0 + eta_p.sinh().atan2(xi_p.cos());
        (lon, lat)
    }
}

/// Ellipsoidal Albers Equal-Area Conic projection (e.g. EPSG:5070 CONUS Albers).
#[derive(Debug, Clone, Copy)]
pub struct AlbersEqualArea {
    pub a: f64,
    pub e: f64,
    pub e2: f64,
    pub lon0: f64,
    pub n: f64,
    pub c: f64,
    pub rho0: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    /// Metres per CRS linear unit (1.0 for metres). False origins are in CRS units.
    pub unit: f64,
}

impl AlbersEqualArea {
    #[inline(always)]
    fn q_fn(e: f64, e2: f64, sin_lat: f64) -> f64 {
        let es = e * sin_lat;
        (1.0 - e2) * (sin_lat / (1.0 - es * es) - (1.0 / (2.0 * e)) * ((1.0 - es) / (1.0 + es)).ln())
    }

    #[inline(always)]
    fn m_fn(e2: f64, lat: f64) -> f64 {
        let (s, c) = lat.sin_cos();
        c / (1.0 - e2 * s * s).sqrt()
    }

    pub fn new(
        ellip: Ellipsoid,
        lat1_deg: f64,
        lat2_deg: f64,
        lat0_deg: f64,
        lon0_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let lat1 = lat1_deg.to_radians();
        let lat2 = lat2_deg.to_radians();
        let lat0 = lat0_deg.to_radians();
        let lon0 = lon0_deg.to_radians();

        let m1 = Self::m_fn(ellip.e2, lat1);
        let m2 = Self::m_fn(ellip.e2, lat2);
        let q1 = Self::q_fn(ellip.e, ellip.e2, lat1.sin());
        let q2 = Self::q_fn(ellip.e, ellip.e2, lat2.sin());
        let q0 = Self::q_fn(ellip.e, ellip.e2, lat0.sin());

        let n = if (lat1 - lat2).abs() < 1e-10 {
            lat1.sin()
        } else {
            (m1 * m1 - m2 * m2) / (q2 - q1)
        };
        let c = m1 * m1 + n * q1;
        let rho0 = ellip.a * (c - n * q0).max(0.0).sqrt() / n;

        Self {
            a: ellip.a,
            e: ellip.e,
            e2: ellip.e2,
            lon0,
            n,
            c,
            rho0,
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        let q = Self::q_fn(self.e, self.e2, lat.sin());
        let rho = self.a * (self.c - self.n * q).max(0.0).sqrt() / self.n;
        let theta = self.n * wrap_dlon(lon - self.lon0);
        let (st, ct) = theta.sin_cos();
        (
            self.false_easting + rho * st / self.unit,
            self.false_northing + (self.rho0 - rho * ct) / self.unit,
        )
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> (f64, f64) {
        let dx = (easting - self.false_easting) * self.unit;
        let dy = self.rho0 - (northing - self.false_northing) * self.unit;
        let sign = if self.n >= 0.0 { 1.0 } else { -1.0 };
        let rho = (dx * dx + dy * dy).sqrt();
        let theta = (sign * dx).atan2(sign * dy);
        let lon = self.lon0 + theta / self.n;
        let q = (self.c - (rho * self.n / self.a).powi(2)) / self.n;
        let mut lat = (q / 2.0).clamp(-1.0, 1.0).asin();
        for _ in 0..6 {
            let s = lat.sin();
            let c = lat.cos();
            if c.abs() < 1e-12 {
                break;
            }
            let es = self.e * s;
            let one_m_e2s2 = 1.0 - es * es;
            let fq = Self::q_fn(self.e, self.e2, s) - q;
            let dfq = (1.0 - self.e2) * 2.0 * c / (one_m_e2s2 * one_m_e2s2);
            lat -= fq / dfq;
        }
        (lon, lat)
    }
}

/// Ellipsoidal Lambert Conformal Conic (1SP and 2SP, e.g. EPSG:2154 Lambert-93).
#[derive(Debug, Clone, Copy)]
pub struct LambertConformalConic {
    pub a: f64,
    pub e: f64,
    pub lon0: f64,
    pub n: f64,
    pub f_const: f64,
    pub rho0: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    /// Metres per CRS linear unit (1.0 for metres). False origins are in CRS units.
    pub unit: f64,
}

impl LambertConformalConic {
    #[inline(always)]
    fn t_fn(e: f64, lat: f64) -> f64 {
        let sin_lat = lat.sin();
        let es = e * sin_lat;
        (std::f64::consts::FRAC_PI_4 - 0.5 * lat).tan()
            / ((1.0 - es) / (1.0 + es)).powf(0.5 * e)
    }

    #[inline(always)]
    fn m_fn(e2: f64, lat: f64) -> f64 {
        let (s, c) = lat.sin_cos();
        c / (1.0 - e2 * s * s).sqrt()
    }

    pub fn new_2sp(
        ellip: Ellipsoid,
        lat1_deg: f64,
        lat2_deg: f64,
        lat0_deg: f64,
        lon0_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let lat1 = lat1_deg.to_radians();
        let lat2 = lat2_deg.to_radians();
        let lat0 = lat0_deg.to_radians();
        let lon0 = lon0_deg.to_radians();

        let m1 = Self::m_fn(ellip.e2, lat1);
        let m2 = Self::m_fn(ellip.e2, lat2);
        let t1 = Self::t_fn(ellip.e, lat1);
        let t2 = Self::t_fn(ellip.e, lat2);
        let t0 = Self::t_fn(ellip.e, lat0);

        let n = if (lat1 - lat2).abs() < 1e-10 {
            lat1.sin()
        } else {
            (m1.ln() - m2.ln()) / (t1.ln() - t2.ln())
        };
        let f_const = m1 / (n * t1.powf(n));
        let rho0 = ellip.a * f_const * t0.powf(n);

        Self {
            a: ellip.a,
            e: ellip.e,
            lon0,
            n,
            f_const,
            rho0,
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    /// One standard parallel at the latitude of origin, with scale factor `k0`
    /// there (EPSG method 9801, GeoTIFF `CT_LambertConfConic_1SP`).
    pub fn new_1sp(
        ellip: Ellipsoid,
        lat0_deg: f64,
        lon0_deg: f64,
        k0: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let mut lcc =
            Self::new_2sp(ellip, lat0_deg, lat0_deg, lat0_deg, lon0_deg, false_easting, false_northing);
        lcc.f_const *= k0;
        lcc.rho0 *= k0;
        lcc
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        let t = Self::t_fn(self.e, lat);
        let rho = self.a * self.f_const * t.powf(self.n);
        let theta = self.n * wrap_dlon(lon - self.lon0);
        let (st, ct) = theta.sin_cos();
        (
            self.false_easting + rho * st / self.unit,
            self.false_northing + (self.rho0 - rho * ct) / self.unit,
        )
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> (f64, f64) {
        let dx = (easting - self.false_easting) * self.unit;
        let dy = self.rho0 - (northing - self.false_northing) * self.unit;
        let sign = if self.n >= 0.0 { 1.0 } else { -1.0 };
        let rho = sign * (dx * dx + dy * dy).sqrt();
        let theta = (sign * dx).atan2(sign * dy);
        let lon = self.lon0 + theta / self.n;
        let t = (rho / (self.a * self.f_const)).max(1e-15).powf(1.0 / self.n);
        let mut lat = std::f64::consts::FRAC_PI_2 - 2.0 * t.atan();
        for _ in 0..7 {
            let es = self.e * lat.sin();
            lat = std::f64::consts::FRAC_PI_2
                - 2.0 * (t * ((1.0 - es) / (1.0 + es)).powf(0.5 * self.e)).atan();
        }
        (lon, lat)
    }
}

/// Ellipsoidal Polar Stereographic (e.g. EPSG:3413 Arctic, EPSG:3031 Antarctic).
#[derive(Debug, Clone, Copy)]
pub struct PolarStereographic {
    pub a: f64,
    pub e: f64,
    pub lon0: f64,
    pub south: bool,
    pub scale_factor: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    /// Metres per CRS linear unit (1.0 for metres). False origins are in CRS units.
    pub unit: f64,
}

impl PolarStereographic {
    pub fn new(
        ellip: Ellipsoid,
        lat_ts_deg: f64,
        lon0_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let south = lat_ts_deg < 0.0;
        let lat_ts = lat_ts_deg.abs().to_radians();
        let sin_ts = lat_ts.sin();
        let m_c = lat_ts.cos() / (1.0 - ellip.e2 * sin_ts * sin_ts).sqrt();
        let t_c = (std::f64::consts::FRAC_PI_4 - 0.5 * lat_ts).tan()
            / ((1.0 - ellip.e * sin_ts) / (1.0 + ellip.e * sin_ts)).powf(0.5 * ellip.e);
        let scale_factor = ellip.a * m_c / t_c;

        Self {
            a: ellip.a,
            e: ellip.e,
            lon0: lon0_deg.to_radians(),
            south,
            scale_factor,
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    /// Scale factor `k0` given at the pole instead of a latitude of true scale
    /// (EPSG method 9810, "variant A", as used by UPS).
    pub fn new_pole_scale(
        ellip: Ellipsoid,
        south: bool,
        k0: f64,
        lon0_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let e = ellip.e;
        let scale_factor =
            2.0 * ellip.a * k0 / ((1.0 + e).powf(1.0 + e) * (1.0 - e).powf(1.0 - e)).sqrt();
        Self {
            a: ellip.a,
            e,
            lon0: lon0_deg.to_radians(),
            south,
            scale_factor,
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        let sign = if self.south { -1.0 } else { 1.0 };
        let phi = sign * lat;
        let sin_p = phi.sin();
        let t = (std::f64::consts::FRAC_PI_4 - 0.5 * phi).tan()
            / ((1.0 - self.e * sin_p) / (1.0 + self.e * sin_p)).powf(0.5 * self.e);
        let rho = self.scale_factor * t;
        let dlon = lon - self.lon0;
        let (s, c) = dlon.sin_cos();
        let x = self.false_easting + rho * s / self.unit;
        let y = self.false_northing + if self.south { rho * c } else { -rho * c } / self.unit;
        (x, y)
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> (f64, f64) {
        let dx = (easting - self.false_easting) * self.unit;
        let dy = (northing - self.false_northing) * self.unit;
        let rho = (dx * dx + dy * dy).sqrt();
        let t = rho / self.scale_factor;
        let mut phi = std::f64::consts::FRAC_PI_2 - 2.0 * t.atan();
        for _ in 0..7 {
            let es = self.e * phi.sin();
            phi = std::f64::consts::FRAC_PI_2
                - 2.0 * (t * ((1.0 - es) / (1.0 + es)).powf(0.5 * self.e)).atan();
        }
        let lon = self.lon0 + if self.south { dx.atan2(dy) } else { dx.atan2(-dy) };
        let lat = if self.south { -phi } else { phi };
        (lon, lat)
    }
}
