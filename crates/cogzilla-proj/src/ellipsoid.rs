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
    pub const AIRY_1830: Self = Self::from_a_inv_f(6_377_563.396, 299.324_964_6);
    pub const BESSEL_1841: Self = Self::from_a_inv_f(6_377_397.155, 299.152_812_8);
    pub const CLARKE_1866: Self = Self::from_a_inv_f(6_378_206.4, 294.978_698_213_898_2);

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

    /// A sphere of radius `r`.
    pub fn sphere(r: f64) -> Self {
        Self { a: r, f: 0.0, e: 0.0, e2: 0.0 }
    }

    pub fn new(a: f64, inv_f: f64) -> Self {
        if inv_f == 0.0 {
            return Self::sphere(a);
        }
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
            // Some national grids put the origin at a pole, outside the range
            // forward() accepts, so the origin's northing skips the domain check.
            let (_, n0) = tm.forward_unchecked(0.0, lat0_deg.to_radians());
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
        self.forward_unchecked(dlon, lat)
    }

    /// The series itself, with no domain check. `dlon` is relative to `lon0`.
    #[inline(always)]
    fn forward_unchecked(&self, dlon: f64, lat: f64) -> (f64, f64) {
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
        if e < 1e-12 {
            return 2.0 * sin_lat;
        }
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

        let raw_n = if (lat1 - lat2).abs() < 1e-10 {
            lat1.sin()
        } else {
            (m1 * m1 - m2 * m2) / (q2 - q1)
        };
        let n = if raw_n.abs() < 1e-10 { 1e-10 } else { raw_n };
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
        let dlon = theta / self.n;
        if !dlon.is_finite() || dlon.abs() > std::f64::consts::PI + 1e-9 {
            return (f64::NAN, f64::NAN);
        }
        let lon = self.lon0 + dlon;
        let q = (self.c - (rho * self.n / self.a).powi(2)) / self.n;
        let qp = Self::q_fn(self.e, self.e2, 1.0);
        if !q.is_finite() || q.abs() > qp + 1e-12 {
            return (f64::NAN, f64::NAN);
        }
        let mut lat = (q / qp).clamp(-1.0, 1.0).asin();
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
            lat = (lat - fq / dfq).clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
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

        let n_raw = if (lat1 - lat2).abs() < 1e-10 {
            lat1.sin()
        } else {
            (m1.ln() - m2.ln()) / (t1.ln() - t2.ln())
        };
        let n = if n_raw.is_finite() && n_raw.abs() >= 1e-12 {
            n_raw
        } else {
            1e-12
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
        // Reject the opposite pole where tan(pi/4 - lat/2) overflows to ~1e16 in f64 instead of inf.
        if (self.n > 0.0 && lat <= -std::f64::consts::FRAC_PI_2 + 1e-12)
            || (self.n < 0.0 && lat >= std::f64::consts::FRAC_PI_2 - 1e-12)
        {
            return (f64::NAN, f64::NAN);
        }
        // At the same-hemisphere pole, tan(pi/4 - lat/2) is 0 (north) or finite ~1.63e16 (south
        // in f64 instead of +inf), so clamp rho directly to the cone apex 0.0.
        let rho = if (self.n > 0.0 && lat >= std::f64::consts::FRAC_PI_2 - 1e-12)
            || (self.n < 0.0 && lat <= -std::f64::consts::FRAC_PI_2 + 1e-12)
        {
            0.0
        } else {
            let t = Self::t_fn(self.e, lat);
            self.a * self.f_const * t.powf(self.n)
        };
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
        let dlon = theta / self.n;
        if !dlon.is_finite() || dlon.abs() > std::f64::consts::PI + 1e-9 {
            return (f64::NAN, f64::NAN);
        }
        let lon = self.lon0 + dlon;
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
        // Reject the opposite pole where tan(pi/4 - phi/2) overflows to ~1e16 in f64 instead of inf.
        if phi <= -std::f64::consts::FRAC_PI_2 + 1e-12 {
            return (f64::NAN, f64::NAN);
        }
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

/// Authalic `q` (Snyder 3-12). On a sphere it reduces to `2 sin(lat)`.
#[inline(always)]
fn authalic_q(e: f64, e2: f64, sin_lat: f64) -> f64 {
    if e < 1e-12 {
        return 2.0 * sin_lat;
    }
    let es = e * sin_lat;
    (1.0 - e2) * (sin_lat / (1.0 - es * es) - (1.0 / (2.0 * e)) * ((1.0 - es) / (1.0 + es)).ln())
}

/// Latitude whose authalic `q` is `q`, by Newton iteration from the spherical answer.
#[inline(always)]
fn lat_from_authalic_q(e: f64, e2: f64, q: f64, qp: f64) -> f64 {
    let mut lat = (q / qp).clamp(-1.0, 1.0).asin();
    if e < 1e-12 {
        return lat;
    }
    for _ in 0..6 {
        let (s, c) = lat.sin_cos();
        if c.abs() < 1e-12 {
            break;
        }
        let one_m = 1.0 - e2 * s * s;
        let step = (authalic_q(e, e2, s) - q) * one_m * one_m / ((1.0 - e2) * 2.0 * c);
        lat -= step;
        if step.abs() < 1e-15 {
            break;
        }
    }
    lat
}

/// Lambert Azimuthal Equal-Area, ellipsoidal or spherical, in its oblique,
/// equatorial and polar aspects (EPSG method 9820; e.g. EPSG:3035 for Europe,
/// EASE-Grid 2.0 at the poles).
#[derive(Debug, Clone, Copy)]
pub struct LambertAzimuthalEqualArea {
    pub a: f64,
    pub e: f64,
    pub e2: f64,
    pub lon0: f64,
    pub lat0: f64,
    /// +1 north polar, -1 south polar, 0 oblique or equatorial.
    pub pole: i8,
    pub qp: f64,
    pub rq: f64,
    pub sin_b1: f64,
    pub cos_b1: f64,
    pub d: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    /// Metres per CRS linear unit (1.0 for metres). False origins are in CRS units.
    pub unit: f64,
}

impl LambertAzimuthalEqualArea {
    pub fn new(
        ellip: Ellipsoid,
        lat0_deg: f64,
        lon0_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let lat0 = lat0_deg.to_radians();
        let pole = if lat0_deg >= 90.0 - 1e-9 {
            1
        } else if lat0_deg <= -90.0 + 1e-9 {
            -1
        } else {
            0
        };
        let qp = authalic_q(ellip.e, ellip.e2, 1.0);
        let rq = (0.5 * qp).sqrt();
        let (sin0, cos0) = lat0.sin_cos();
        let sin_b1 = authalic_q(ellip.e, ellip.e2, sin0) / qp;
        let cos_b1 = (1.0 - sin_b1 * sin_b1).max(0.0).sqrt();
        let d = if pole != 0 {
            1.0
        } else {
            cos0 / ((1.0 - ellip.e2 * sin0 * sin0).sqrt() * rq * cos_b1)
        };
        Self {
            a: ellip.a,
            e: ellip.e,
            e2: ellip.e2,
            lon0: lon0_deg.to_radians(),
            lat0,
            pole,
            qp,
            rq,
            sin_b1,
            cos_b1,
            d,
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        let dlon = wrap_dlon(lon - self.lon0);
        let (sin_l, cos_l) = dlon.sin_cos();
        let q = authalic_q(self.e, self.e2, lat.sin());
        let (x, y) = match self.pole {
            1 => {
                let rho = self.a * (self.qp - q).max(0.0).sqrt();
                (rho * sin_l, -rho * cos_l)
            }
            -1 => {
                let rho = self.a * (self.qp + q).max(0.0).sqrt();
                (rho * sin_l, rho * cos_l)
            }
            _ => {
                let sin_b = (q / self.qp).clamp(-1.0, 1.0);
                let cos_b = (1.0 - sin_b * sin_b).sqrt();
                let denom = 1.0 + self.sin_b1 * sin_b + self.cos_b1 * cos_b * cos_l;
                // The antipode of the projection centre has no image.
                if denom < 1e-12 {
                    return (f64::NAN, f64::NAN);
                }
                let b = self.a * self.rq * (2.0 / denom).sqrt();
                (
                    b * self.d * cos_b * sin_l,
                    (b / self.d) * (self.cos_b1 * sin_b - self.sin_b1 * cos_b * cos_l),
                )
            }
        };
        (
            self.false_easting + x / self.unit,
            self.false_northing + y / self.unit,
        )
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> (f64, f64) {
        let x = (easting - self.false_easting) * self.unit / self.a;
        let y = (northing - self.false_northing) * self.unit / self.a;
        if self.pole != 0 {
            let rho2 = x * x + y * y;
            // Beyond the opposite pole there is nothing to invert.
            if rho2 > 2.0 * self.qp * (1.0 + 1e-12) {
                return (f64::NAN, f64::NAN);
            }
            let q = (self.pole as f64) * (self.qp - rho2);
            let lat = lat_from_authalic_q(self.e, self.e2, q, self.qp);
            let lon = if self.pole == 1 { x.atan2(-y) } else { x.atan2(y) };
            return (self.lon0 + lon, lat);
        }
        let xd = x / self.d;
        let yd = y * self.d;
        let rho = xd.hypot(yd);
        if rho < 1e-15 {
            return (self.lon0, self.lat0);
        }
        let half = 0.5 * rho / self.rq;
        if half > 1.0 {
            return (f64::NAN, f64::NAN);
        }
        let ce = 2.0 * half.asin();
        let (sin_ce, cos_ce) = ce.sin_cos();
        let sin_b = cos_ce * self.sin_b1 + yd * sin_ce * self.cos_b1 / rho;
        let lon = (xd * sin_ce).atan2(rho * self.cos_b1 * cos_ce - yd * self.sin_b1 * sin_ce);
        let lat = lat_from_authalic_q(self.e, self.e2, self.qp * sin_b, self.qp);
        (self.lon0 + lon, lat)
    }
}

/// Sinusoidal (Sanson-Flamsteed), ellipsoidal or spherical. MODIS and VIIRS
/// land products use the spherical form.
#[derive(Debug, Clone, Copy)]
pub struct Sinusoidal {
    pub a: f64,
    pub e2: f64,
    pub lon0: f64,
    /// Rectifying radius: meridian distance is `rect_a * mu`.
    pub rect_a: f64,
    /// Series taking geodetic latitude to rectifying latitude, and back.
    pub to_mu: [f64; 4],
    pub from_mu: [f64; 4],
    pub false_easting: f64,
    pub false_northing: f64,
    /// Metres per CRS linear unit (1.0 for metres). False origins are in CRS units.
    pub unit: f64,
}

impl Sinusoidal {
    pub fn new(ellip: Ellipsoid, lon0_deg: f64, false_easting: f64, false_northing: f64) -> Self {
        let n = ellip.f / (2.0 - ellip.f);
        let (n2, n3, n4) = (n * n, n * n * n, n * n * n * n);
        Self {
            a: ellip.a,
            e2: ellip.e2,
            lon0: lon0_deg.to_radians(),
            rect_a: (ellip.a / (1.0 + n)) * (1.0 + n2 / 4.0 + n4 / 64.0),
            to_mu: [
                -1.5 * n + (9.0 / 16.0) * n3,
                (15.0 / 16.0) * n2 - (15.0 / 32.0) * n4,
                -(35.0 / 48.0) * n3,
                (315.0 / 512.0) * n4,
            ],
            from_mu: [
                1.5 * n - (27.0 / 32.0) * n3,
                (21.0 / 16.0) * n2 - (55.0 / 32.0) * n4,
                (151.0 / 96.0) * n3,
                (1097.0 / 512.0) * n4,
            ],
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        let dlon = wrap_dlon(lon - self.lon0);
        let (s, c) = lat.sin_cos();
        let mut mu = lat;
        for (i, k) in self.to_mu.iter().enumerate() {
            mu += k * (2.0 * (i as f64 + 1.0) * lat).sin();
        }
        let x = self.a * dlon * c / (1.0 - self.e2 * s * s).sqrt();
        let y = self.rect_a * mu;
        (
            self.false_easting + x / self.unit,
            self.false_northing + y / self.unit,
        )
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> (f64, f64) {
        let x = (easting - self.false_easting) * self.unit;
        let y = (northing - self.false_northing) * self.unit;
        let mu = y / self.rect_a;
        if mu.abs() > std::f64::consts::FRAC_PI_2 + 1e-12 {
            return (f64::NAN, f64::NAN);
        }
        let mut lat = mu;
        for (i, k) in self.from_mu.iter().enumerate() {
            lat += k * (2.0 * (i as f64 + 1.0) * mu).sin();
        }
        let (s, c) = lat.sin_cos();
        if c.abs() < 1e-12 {
            // The Sinusoidal lobe has zero width at ±90°; any nonzero x lies outside Earth.
            if x.abs() > 1e-6 {
                return (f64::NAN, f64::NAN);
            }
            return (self.lon0, lat);
        }
        let dlon = x * (1.0 - self.e2 * s * s).sqrt() / (self.a * c);
        // Outside the projection's lobe: the grid cell covers no part of the Earth.
        if dlon.abs() > std::f64::consts::PI + 1e-9 {
            return (f64::NAN, f64::NAN);
        }
        (self.lon0 + dlon, lat)
    }
}

/// Lambert Cylindrical Equal Area (EPSG method 9835 ellipsoidal / 9834 spherical).
///
/// Used by NSIDC EASE-Grid 2.0 Global (`EPSG:6933`, WGS84, `lat_ts = 30°`) and legacy
/// NSIDC EASE-Grid Global (`EPSG:3410`, sphere `R = 6,371,228 m`, `lat_ts = 30°`).
/// Completely separable in `(lon, lat)`: `x = f(lon)`, `y = g(lat)`.
#[derive(Debug, Clone, Copy)]
pub struct CylindricalEqualArea {
    pub a: f64,
    pub e: f64,
    pub e2: f64,
    pub lon0: f64,
    /// `k0 = cos(lat_ts) / sqrt(1 - e^2 sin^2(lat_ts))`
    pub k0: f64,
    /// Authalic `q` at the pole (`2.0` on a sphere).
    pub qp: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    /// Metres per CRS linear unit (1.0 for metres). False origins are in CRS units.
    pub unit: f64,
}

impl CylindricalEqualArea {
    pub fn new(
        ellip: Ellipsoid,
        lat_ts_deg: f64,
        lon0_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let lat_ts = lat_ts_deg.to_radians();
        let (sin_ts, cos_ts) = lat_ts.sin_cos();
        let k0 = cos_ts / (1.0 - ellip.e2 * sin_ts * sin_ts).sqrt();
        let qp = authalic_q(ellip.e, ellip.e2, 1.0);
        Self {
            a: ellip.a,
            e: ellip.e,
            e2: ellip.e2,
            lon0: lon0_deg.to_radians(),
            k0,
            qp,
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    #[inline(always)]
    pub fn forward_x(&self, lon: f64) -> f64 {
        let dlon = wrap_dlon(lon - self.lon0);
        self.false_easting + (self.a * self.k0 * dlon) / self.unit
    }

    #[inline(always)]
    pub fn forward_y(&self, lat: f64) -> f64 {
        let q = authalic_q(self.e, self.e2, lat.sin());
        self.false_northing + (self.a * q / (2.0 * self.k0)) / self.unit
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        (self.forward_x(lon), self.forward_y(lat))
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> (f64, f64) {
        let x = (easting - self.false_easting) * self.unit;
        let y = (northing - self.false_northing) * self.unit;
        let dlon = x / (self.a * self.k0);
        if dlon.abs() > std::f64::consts::PI + 1e-9 {
            return (f64::NAN, f64::NAN);
        }
        let q = 2.0 * y * self.k0 / self.a;
        if q.abs() > self.qp * (1.0 + 1e-9) {
            return (f64::NAN, f64::NAN);
        }
        let lat = lat_from_authalic_q(self.e, self.e2, q.clamp(-self.qp, self.qp), self.qp);
        (self.lon0 + dlon, lat)
    }
}

/// Equal Earth pseudocylindrical equal-area projection (Šavrič et al., 2018; EPSG method 1078).
///
/// Covers `EPSG:8857` (Greenwich), `EPSG:8858` (Americas), and `EPSG:8859` (Asia-Pacific).
#[derive(Debug, Clone, Copy)]
pub struct EqualEarth {
    pub a: f64,
    pub e: f64,
    pub e2: f64,
    pub lon0: f64,
    pub qp: f64,
    /// Authalic sphere radius `a * sqrt(qp / 2)`.
    pub rq: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    /// Metres per CRS linear unit (1.0 for metres). False origins are in CRS units.
    pub unit: f64,
}

impl EqualEarth {
    const A1: f64 = 1.340264;
    const A2: f64 = -0.081106;
    const A3: f64 = 0.000893;
    const A4: f64 = 0.003796;
    /// `sqrt(3) / 2`
    const M: f64 = 0.866_025_403_784_438_6;

    pub fn new(ellip: Ellipsoid, lon0_deg: f64, false_easting: f64, false_northing: f64) -> Self {
        let qp = authalic_q(ellip.e, ellip.e2, 1.0);
        let rq = ellip.a * (0.5 * qp).sqrt();
        Self {
            a: ellip.a,
            e: ellip.e,
            e2: ellip.e2,
            lon0: lon0_deg.to_radians(),
            qp,
            rq,
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        let dlon = wrap_dlon(lon - self.lon0);
        let sin_b = (authalic_q(self.e, self.e2, lat.sin()) / self.qp).clamp(-1.0, 1.0);
        let theta = (Self::M * sin_b).asin();
        let t2 = theta * theta;
        let t6 = t2 * t2 * t2;
        let t8 = t6 * t2;
        let denom = Self::M * (Self::A1 + 3.0 * Self::A2 * t2 + 7.0 * Self::A3 * t6 + 9.0 * Self::A4 * t8);
        let x = self.rq * dlon * theta.cos() / denom;
        let y = self.rq * theta * (Self::A1 + Self::A2 * t2 + Self::A3 * t6 + Self::A4 * t8);
        (
            self.false_easting + x / self.unit,
            self.false_northing + y / self.unit,
        )
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> (f64, f64) {
        let x = (easting - self.false_easting) * self.unit / self.rq;
        let y = (northing - self.false_northing) * self.unit / self.rq;
        // Maximum |y| at the poles (theta = asin(sqrt(3)/2) = pi/3).
        const THETA_MAX: f64 = std::f64::consts::FRAC_PI_3;
        let tm2 = THETA_MAX * THETA_MAX;
        let tm6 = tm2 * tm2 * tm2;
        let y_max = THETA_MAX * (Self::A1 + Self::A2 * tm2 + Self::A3 * tm6 + Self::A4 * tm6 * tm2);
        if y.abs() > y_max * (1.0 + 1e-9) {
            return (f64::NAN, f64::NAN);
        }
        let y_clamped = y.clamp(-y_max, y_max);
        let mut theta = (y_clamped / Self::A1).clamp(-THETA_MAX, THETA_MAX);
        for _ in 0..10 {
            let t2 = theta * theta;
            let t6 = t2 * t2 * t2;
            let t8 = t6 * t2;
            let f = theta * (Self::A1 + Self::A2 * t2 + Self::A3 * t6 + Self::A4 * t8) - y_clamped;
            let df = Self::A1 + 3.0 * Self::A2 * t2 + 7.0 * Self::A3 * t6 + 9.0 * Self::A4 * t8;
            let dt = f / df;
            theta -= dt;
            if dt.abs() < 1e-13 {
                break;
            }
        }
        theta = theta.clamp(-THETA_MAX, THETA_MAX);
        let t2 = theta * theta;
        let t6 = t2 * t2 * t2;
        let t8 = t6 * t2;
        let df = Self::A1 + 3.0 * Self::A2 * t2 + 7.0 * Self::A3 * t6 + 9.0 * Self::A4 * t8;
        let dlon = x * (Self::M * df) / theta.cos();
        if dlon.abs() > std::f64::consts::PI + 1e-9 {
            return (f64::NAN, f64::NAN);
        }
        let sin_b = (theta.sin() / Self::M).clamp(-1.0, 1.0);
        let lat = lat_from_authalic_q(self.e, self.e2, self.qp * sin_b, self.qp);
        (self.lon0 + dlon, lat)
    }
}

/// Solve `asinh(tan(lat)) - e * atanh(e * sin(lat)) = psi` for geodetic latitude `lat` (rad).
#[inline(always)]
fn inverse_isometric_lat(psi: f64, e: f64) -> f64 {
    let mut lat = psi.sinh().atan();
    if e == 0.0 {
        return lat;
    }
    let e2 = e * e;
    for _ in 0..6 {
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let f = lat.tan().asinh() - e * (e * sin_lat).atanh() - psi;
        let df = (1.0 - e2) / (cos_lat * (1.0 - e2 * sin_lat * sin_lat));
        let step = f / df;
        lat -= step;
        if step.abs() < 1e-13 {
            break;
        }
    }
    lat
}

/// Equirectangular / Equidistant Cylindrical projection (PROJ `+proj=eqc`, GeoTIFF `CT_Equirectangular = 17`).
///
/// Covers `EPSG:32662` (Plate Carrée), `EPSG:32663`, `EPSG:4087` (World Equidistant Cylindrical),
/// and user-defined Equirectangular CRSes.
#[derive(Debug, Clone, Copy)]
pub struct Equirectangular {
    pub a: f64,
    pub lon0: f64,
    pub lat0: f64,
    pub cos_lat_ts: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    pub unit: f64,
}

impl Equirectangular {
    pub fn new(
        ellip: Ellipsoid,
        lat_ts_deg: f64,
        lat0_deg: f64,
        lon0_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        Self {
            a: ellip.a,
            lon0: lon0_deg.to_radians(),
            lat0: lat0_deg.to_radians(),
            cos_lat_ts: lat_ts_deg.to_radians().cos(),
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    #[inline(always)]
    pub fn forward_x(&self, lon: f64) -> f64 {
        (self.a * wrap_dlon(lon - self.lon0) * self.cos_lat_ts) / self.unit + self.false_easting
    }

    #[inline(always)]
    pub fn forward_y(&self, lat: f64) -> f64 {
        (self.a * (lat - self.lat0)) / self.unit + self.false_northing
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        (self.forward_x(lon), self.forward_y(lat))
    }

    #[inline(always)]
    pub fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let dlon = (x - self.false_easting) * self.unit / (self.a * self.cos_lat_ts);
        let lat = self.lat0 + (y - self.false_northing) * self.unit / self.a;
        if dlon.abs() > std::f64::consts::PI + 1e-9 || lat.abs() > std::f64::consts::FRAC_PI_2 + 1e-9 {
            return (f64::NAN, f64::NAN);
        }
        (self.lon0 + dlon, lat.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2))
    }
}

/// Ellipsoidal / Spherical Mercator 1SP & 2SP projection (PROJ `+proj=merc`, GeoTIFF `CT_Mercator = 7`).
///
/// Covers `EPSG:3395`, `EPSG:3832` (PDC Mercator), `EPSG:3994` (Mercator 41), and user-defined Mercator CRSes.
#[derive(Debug, Clone, Copy)]
pub struct Mercator {
    pub e: f64,
    pub lon0: f64,
    /// `a * k0` (metres)
    pub ak: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    pub unit: f64,
}

impl Mercator {
    pub fn new_1sp(
        ellip: Ellipsoid,
        lon0_deg: f64,
        k0: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        Self {
            e: ellip.e,
            lon0: lon0_deg.to_radians(),
            ak: ellip.a * k0,
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    pub fn new_2sp(
        ellip: Ellipsoid,
        lat_ts_deg: f64,
        lon0_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let lat_ts = lat_ts_deg.to_radians();
        let sin_ts = lat_ts.sin();
        let k = lat_ts.cos() / (1.0 - ellip.e2 * sin_ts * sin_ts).sqrt();
        Self::new_1sp(ellip, lon0_deg, k, false_easting, false_northing)
    }

    #[inline(always)]
    pub fn forward_x(&self, lon: f64) -> f64 {
        (self.ak * wrap_dlon(lon - self.lon0)) / self.unit + self.false_easting
    }

    #[inline(always)]
    pub fn forward_y(&self, lat: f64) -> f64 {
        if lat.abs() >= std::f64::consts::FRAC_PI_2 - 1e-9 {
            return f64::NAN;
        }
        let psi = lat.tan().asinh() - self.e * (self.e * lat.sin()).atanh();
        (self.ak * psi) / self.unit + self.false_northing
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        (self.forward_x(lon), self.forward_y(lat))
    }

    #[inline(always)]
    pub fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let dlon = (x - self.false_easting) * self.unit / self.ak;
        if dlon.abs() > std::f64::consts::PI + 1e-9 {
            return (f64::NAN, f64::NAN);
        }
        let psi = (y - self.false_northing) * self.unit / self.ak;
        let lat = inverse_isometric_lat(psi, self.e);
        (self.lon0 + dlon, lat)
    }
}

/// Oblique Stereographic double projection (EPSG method 9809, PROJ `+proj=sterea`, GeoTIFF `CT_ObliqueStereographic = 16`).
///
/// Covers Dutch RD New (`EPSG:28992`), New Brunswick (`EPSG:2036`), Romania Stereo 70 (`EPSG:31700`), etc.
#[derive(Debug, Clone, Copy)]
pub struct ObliqueStereographic {
    pub e: f64,
    pub lon0: f64,
    pub k0: f64,
    pub r: f64,
    pub n: f64,
    pub c: f64,
    pub ln_c: f64,
    pub chi0: f64,
    pub sin_chi0: f64,
    pub cos_chi0: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    pub unit: f64,
}

impl ObliqueStereographic {
    pub fn new(
        ellip: Ellipsoid,
        lat0_deg: f64,
        lon0_deg: f64,
        k0: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let lat0 = lat0_deg.to_radians();
        let lon0 = lon0_deg.to_radians();
        let e = ellip.e;
        let e2 = ellip.e2;
        let sin0 = lat0.sin();
        let cos0 = lat0.cos();
        let one_minus_e2_sin2 = 1.0 - e2 * sin0 * sin0;
        let rho0 = ellip.a * (1.0 - e2) / one_minus_e2_sin2.powf(1.5);
        let nu0 = ellip.a / one_minus_e2_sin2.sqrt();
        let r = (rho0 * nu0).sqrt();
        let n = (1.0 + e2 * cos0.powi(4) / (1.0 - e2)).sqrt();
        let s1 = (1.0 + sin0) / (1.0 - sin0);
        let s2 = (1.0 - e * sin0) / (1.0 + e * sin0);
        let w1 = (s1 * s2.powf(e)).powf(n);
        let sin_chi0_init = (w1 - 1.0) / (w1 + 1.0);
        let c = (n + sin0) * (1.0 - sin_chi0_init) / ((n - sin0) * (1.0 + sin_chi0_init));
        let w2 = c * w1;
        let chi0 = 2.0 * w2.sqrt().atan() - std::f64::consts::FRAC_PI_2;
        Self {
            e,
            lon0,
            k0,
            r,
            n,
            c,
            ln_c: c.ln(),
            chi0,
            sin_chi0: chi0.sin(),
            cos_chi0: chi0.cos(),
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        if lat.abs() >= std::f64::consts::FRAC_PI_2 - 1e-9 {
            return (f64::NAN, f64::NAN);
        }
        let sin_lat = lat.sin();
        let sa = (1.0 + sin_lat) / (1.0 - sin_lat);
        let sb = (1.0 - self.e * sin_lat) / (1.0 + self.e * sin_lat);
        let w = self.c * (sa * sb.powf(self.e)).powf(self.n);
        let chi = 2.0 * w.sqrt().atan() - std::f64::consts::FRAC_PI_2;
        let lam = self.n * wrap_dlon(lon - self.lon0);
        let (sin_chi, cos_chi) = chi.sin_cos();
        let (sin_lam, cos_lam) = lam.sin_cos();
        let b = 1.0 + sin_chi * self.sin_chi0 + cos_chi * self.cos_chi0 * cos_lam;
        if b < 1e-12 {
            return (f64::NAN, f64::NAN);
        }
        let scale = 2.0 * self.r * self.k0 / (b * self.unit);
        let easting = self.false_easting + scale * cos_chi * sin_lam;
        let northing = self.false_northing
            + scale * (sin_chi * self.cos_chi0 - cos_chi * self.sin_chi0 * cos_lam);
        (easting, northing)
    }

    #[inline(always)]
    pub fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = (x - self.false_easting) * self.unit;
        let dy = (y - self.false_northing) * self.unit;
        let rho = dx.hypot(dy);
        let (chi, lam) = if rho < 1e-12 {
            (self.chi0, 0.0)
        } else {
            let ce = 2.0 * (rho / (2.0 * self.r * self.k0)).atan();
            let (sin_ce, cos_ce) = ce.sin_cos();
            let chi = (cos_ce * self.sin_chi0 + dy * sin_ce * self.cos_chi0 / rho)
                .clamp(-1.0, 1.0)
                .asin();
            let lam = (dx * sin_ce).atan2(rho * self.cos_chi0 * cos_ce - dy * self.sin_chi0 * sin_ce);
            (chi, lam)
        };
        let lon = self.lon0 + lam / self.n;
        let psi_chi = 0.5 * ((1.0 + chi.sin()) / (1.0 - chi.sin())).ln();
        let iso = (psi_chi - 0.5 * self.ln_c) / self.n;
        let lat = inverse_isometric_lat(iso, self.e);
        (lon, lat)
    }
}

/// Swiss Oblique Cylindrical / Mercator projection (EPSG method 9815, PROJ `+proj=somerc`).
///
/// Covers Swiss CH1903+ / LV95 (`EPSG:2056`) and CH1903 / LV03 (`EPSG:21781`).
#[derive(Debug, Clone, Copy)]
pub struct SwissObliqueMercator {
    pub e: f64,
    pub lon0: f64,
    pub alpha: f64,
    pub b0: f64,
    pub sin_b0: f64,
    pub cos_b0: f64,
    pub k: f64,
    pub r: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    pub unit: f64,
}

impl SwissObliqueMercator {
    pub fn new(
        ellip: Ellipsoid,
        lat0_deg: f64,
        lon0_deg: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let lat0 = lat0_deg.to_radians();
        let lon0 = lon0_deg.to_radians();
        let e = ellip.e;
        let e2 = ellip.e2;
        let sin0 = lat0.sin();
        let cos0 = lat0.cos();
        let alpha = (1.0 + (e2 / (1.0 - e2)) * cos0.powi(4)).sqrt();
        let b0 = (sin0 / alpha).asin();
        let k = (std::f64::consts::FRAC_PI_4 + 0.5 * b0).tan().ln()
            - alpha * (std::f64::consts::FRAC_PI_4 + 0.5 * lat0).tan().ln()
            + alpha * e * 0.5 * ((1.0 + e * sin0) / (1.0 - e * sin0)).ln();
        let r = ellip.a * (1.0 - e2).sqrt() / (1.0 - e2 * sin0 * sin0);
        Self {
            e,
            lon0,
            alpha,
            b0,
            sin_b0: b0.sin(),
            cos_b0: b0.cos(),
            k,
            r,
            false_easting,
            false_northing,
            unit: 1.0,
        }
    }

    #[inline(always)]
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        if lat.abs() >= std::f64::consts::FRAC_PI_2 - 1e-9 {
            return (f64::NAN, f64::NAN);
        }
        let sin_lat = lat.sin();
        let s = self.alpha * (std::f64::consts::FRAC_PI_4 + 0.5 * lat).tan().ln()
            - self.alpha * self.e * 0.5 * ((1.0 + self.e * sin_lat) / (1.0 - self.e * sin_lat)).ln()
            + self.k;
        let b = 2.0 * (s.exp().atan() - std::f64::consts::FRAC_PI_4);
        let l = self.alpha * wrap_dlon(lon - self.lon0);
        let (sin_l, cos_l) = l.sin_cos();
        let (sin_b, cos_b) = b.sin_cos();
        let l_rot = sin_l.atan2(self.sin_b0 * (sin_b / cos_b) + self.cos_b0 * cos_l);
        let b_rot = (self.cos_b0 * sin_b - self.sin_b0 * cos_b * cos_l)
            .clamp(-1.0, 1.0)
            .asin();
        let sin_br = b_rot.sin();
        if (1.0 - sin_br.abs()) < 1e-12 {
            return (f64::NAN, f64::NAN);
        }
        let easting = self.false_easting + (self.r * l_rot) / self.unit;
        let northing = self.false_northing
            + (0.5 * self.r * ((1.0 + sin_br) / (1.0 - sin_br)).ln()) / self.unit;
        (easting, northing)
    }

    #[inline(always)]
    pub fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let l_rot = (x - self.false_easting) * self.unit / self.r;
        let b_rot = 2.0 * (((y - self.false_northing) * self.unit / self.r).exp().atan()
            - std::f64::consts::FRAC_PI_4);
        let (sin_br, cos_br) = b_rot.sin_cos();
        let (sin_lr, cos_lr) = l_rot.sin_cos();
        let b = (self.cos_b0 * sin_br + self.sin_b0 * cos_br * cos_lr)
            .clamp(-1.0, 1.0)
            .asin();
        let l = sin_lr.atan2(self.cos_b0 * cos_lr - self.sin_b0 * (sin_br / cos_br));
        let lon = self.lon0 + l / self.alpha;
        let iso = ((std::f64::consts::FRAC_PI_4 + 0.5 * b).tan().ln() - self.k) / self.alpha;
        let lat = inverse_isometric_lat(iso, self.e);
        (lon, lat)
    }
}

/// 7-parameter Position Vector Helmert datum shift between a local ellipsoid datum and WGS84
/// (EPSG method 9606).
#[derive(Debug, Clone, Copy)]
pub struct HelmertTransform {
    pub local_ellip: Ellipsoid,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Rotation angles in radians (Position Vector convention, local → WGS84).
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
    /// Scale factor `1 + ds_ppm * 1e-6`.
    pub scale: f64,
}

impl HelmertTransform {
    pub const fn position_vector(
        local_ellip: Ellipsoid,
        dx: f64,
        dy: f64,
        dz: f64,
        rx_arcsec: f64,
        ry_arcsec: f64,
        rz_arcsec: f64,
        ds_ppm: f64,
    ) -> Self {
        const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3600.0);
        Self {
            local_ellip,
            dx,
            dy,
            dz,
            rx: rx_arcsec * ARCSEC_TO_RAD,
            ry: ry_arcsec * ARCSEC_TO_RAD,
            rz: rz_arcsec * ARCSEC_TO_RAD,
            scale: 1.0 + ds_ppm * 1e-6,
        }
    }

    /// OSGB36 (Airy 1830) → WGS84 (EPSG:1314, British National Grid `EPSG:27700`).
    pub const OSGB36_TO_WGS84: Self = Self::position_vector(
        Ellipsoid::AIRY_1830,
        446.448,
        -125.157,
        542.060,
        0.1502,
        0.2470,
        0.8421,
        -20.4894,
    );

    /// Amersfoort (Bessel 1841) → WGS84 (EPSG:15934, Dutch RD New `EPSG:28992`).
    pub const AMERSFOORT_TO_WGS84: Self = Self::position_vector(
        Ellipsoid::BESSEL_1841,
        565.4171,
        50.3319,
        465.5524,
        -0.398957,
        0.343988,
        -1.87740,
        4.0725,
    );

    /// CH1903 / CH1903+ (Bessel 1841) → WGS84 (EPSG:1676, Swiss LV95 `EPSG:2056` & LV03 `EPSG:21781`).
    pub const CH1903_TO_WGS84: Self = Self::position_vector(
        Ellipsoid::BESSEL_1841,
        674.374,
        15.056,
        405.346,
        0.0,
        0.0,
        0.0,
        0.0,
    );

    /// NAD27 CONUS (Clarke 1866) → WGS84 (EPSG:1173, `EPSG:4267` & NAD27 UTM `EPSG:26701..=26723`).
    pub const NAD27_TO_WGS84: Self = Self::position_vector(
        Ellipsoid::CLARKE_1866,
        -8.0,
        160.0,
        176.0,
        0.0,
        0.0,
        0.0,
        0.0,
    );

    #[inline(always)]
    fn geodetic_to_ecef(ellip: Ellipsoid, lon: f64, lat: f64) -> (f64, f64, f64) {
        let (sin_lat, cos_lat) = lat.sin_cos();
        let (sin_lon, cos_lon) = lon.sin_cos();
        let nu = ellip.a / (1.0 - ellip.e2 * sin_lat * sin_lat).sqrt();
        (
            nu * cos_lat * cos_lon,
            nu * cos_lat * sin_lon,
            nu * (1.0 - ellip.e2) * sin_lat,
        )
    }

    #[inline(always)]
    fn ecef_to_geodetic(ellip: Ellipsoid, x: f64, y: f64, z: f64) -> (f64, f64) {
        let lon = y.atan2(x);
        let p = x.hypot(y);
        if p < 1e-12 {
            let lat = if z >= 0.0 {
                std::f64::consts::FRAC_PI_2
            } else {
                -std::f64::consts::FRAC_PI_2
            };
            return (lon, lat);
        }
        let mut lat = (z / (p * (1.0 - ellip.e2))).atan();
        for _ in 0..5 {
            let sin_lat = lat.sin();
            let nu = ellip.a / (1.0 - ellip.e2 * sin_lat * sin_lat).sqrt();
            let next = ((z + ellip.e2 * nu * sin_lat) / p).atan();
            if (next - lat).abs() < 1e-13 {
                lat = next;
                break;
            }
            lat = next;
        }
        (lon, lat)
    }

    /// Transform `(lon_deg, lat_deg)` from local datum to WGS84.
    #[inline(always)]
    pub fn datum_to_wgs84_deg(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        if !lon_deg.is_finite() || !lat_deg.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        let (x, y, z) = Self::geodetic_to_ecef(
            self.local_ellip,
            lon_deg.to_radians(),
            lat_deg.to_radians(),
        );
        let xw = self.dx + self.scale * (x - self.rz * y + self.ry * z);
        let yw = self.dy + self.scale * (self.rz * x + y - self.rx * z);
        let zw = self.dz + self.scale * (-self.ry * x + self.rx * y + z);
        let (lon, lat) = Self::ecef_to_geodetic(Ellipsoid::WGS84, xw, yw, zw);
        (lon.to_degrees(), lat.to_degrees())
    }

    /// Transform `(lon_deg, lat_deg)` from WGS84 to local datum (exact algebraic inverse of `datum_to_wgs84_deg`).
    #[inline(always)]
    pub fn wgs84_to_datum_deg(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        if !lon_deg.is_finite() || !lat_deg.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        let (xw, yw, zw) =
            Self::geodetic_to_ecef(Ellipsoid::WGS84, lon_deg.to_radians(), lat_deg.to_radians());
        let vx = (xw - self.dx) / self.scale;
        let vy = (yw - self.dy) / self.scale;
        let vz = (zw - self.dz) / self.scale;
        // Exact inverse of R = [[1, -rz, ry], [rz, 1, -rx], [-ry, rx, 1]]
        // det(R) = 1 + rx^2 + ry^2 + rz^2
        let (rx, ry, rz) = (self.rx, self.ry, self.rz);
        let inv_det = 1.0 / (1.0 + rx * rx + ry * ry + rz * rz);
        let x = ((1.0 + rx * rx) * vx + (rx * ry + rz) * vy + (rx * rz - ry) * vz) * inv_det;
        let y = ((rx * ry - rz) * vx + (1.0 + ry * ry) * vy + (ry * rz + rx) * vz) * inv_det;
        let z = ((rx * rz + ry) * vx + (ry * rz - rx) * vy + (1.0 + rz * rz) * vz) * inv_det;
        let (lon, lat) = Self::ecef_to_geodetic(self.local_ellip, x, y, z);
        (lon.to_degrees(), lat.to_degrees())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn albers_inverse_returns_nan_inside_polar_hole() {
        // EPSG:3338 (NAD83 / Alaska Albers): lat1=55, lat2=65, lat0=50, lon0=-154
        let aea = AlbersEqualArea::new(Ellipsoid::GRS80, 55.0, 65.0, 50.0, -154.0, 0.0, 0.0);
        // Point well inside the polar hole (rho < rho_pole ~ 965 km)
        let (lon, lat) = aea.inverse(0.0, 3_935_540.0);
        assert!(lon.is_nan() && lat.is_nan(), "expected (NaN, NaN) inside Albers polar hole, got ({lon}, {lat})");
        // Valid point in Alaska round-trips accurately
        let (e, n) = aea.forward((-150.0_f64).to_radians(), 61.0_f64.to_radians());
        let (lon2, lat2) = aea.inverse(e, n);
        assert!((lon2.to_degrees() + 150.0).abs() < 1e-7);
        assert!((lat2.to_degrees() - 61.0).abs() < 1e-7);
    }

    #[test]
    fn opposite_poles_and_conic_wedge_return_nan() {
        // North Polar Stereographic (EPSG:3413) at South Pole must be NaN (not ~2e23 m)
        let nps = PolarStereographic::new(Ellipsoid::WGS84, 70.0, -45.0, 0.0, 0.0);
        let (x, y) = nps.forward(0.0, -std::f64::consts::FRAC_PI_2);
        assert!(x.is_nan() && y.is_nan(), "North Polar Stereo at -90 must be NaN, got ({x}, {y})");

        // South Polar Stereographic (EPSG:3031) at North Pole must be NaN
        let sps = PolarStereographic::new(Ellipsoid::WGS84, -71.0, 0.0, 0.0, 0.0);
        let (x, y) = sps.forward(0.0, std::f64::consts::FRAC_PI_2);
        assert!(x.is_nan() && y.is_nan(), "South Polar Stereo at +90 must be NaN, got ({x}, {y})");

        // Canadian NAD83 LCC (EPSG:3978): South Pole forward must be NaN, and unmapped cone wedge
        // inverse at (500_000, 15_000_000) must be NaN (not ghost Eurasia)
        let lcc = LambertConformalConic::new_2sp(
            Ellipsoid::GRS80,
            49.0,
            77.0,
            49.0,
            -95.0,
            0.0,
            0.0,
        );
        let (fx, fy) = lcc.forward(0.0, -std::f64::consts::FRAC_PI_2);
        assert!(fx.is_nan() && fy.is_nan(), "LCC at -90 must be NaN, got ({fx}, {fy})");
        let (ilo, ila) = lcc.inverse(500_000.0, 15_000_000.0);
        assert!(ilo.is_nan() && ila.is_nan(), "LCC unmapped cone wedge must be NaN, got ({ilo}, {ila})");

        // Albers Equal Area (EPSG:5070): unmapped cone wedge north of apex must be NaN
        let aea = AlbersEqualArea::new(Ellipsoid::GRS80, 29.5, 45.5, 23.0, -96.0, 0.0, 0.0);
        let (alo, ala) = aea.inverse(500_000.0, 10_000_000.0);
        assert!(alo.is_nan() && ala.is_nan(), "Albers unmapped cone wedge must be NaN, got ({alo}, {ala})");

        // Audit #70 #1: Sinusoidal out-of-lobe polar coordinates (x != 0 at y = +y_pole) must be NaN
        let sinu = Sinusoidal::new(Ellipsoid::sphere(6_371_007.181), 0.0, 0.0, 0.0);
        let y_pole = 6_371_007.181 * std::f64::consts::FRAC_PI_2;
        let (slo, sla) = sinu.inverse(-1_111_950.0, y_pole);
        assert!(slo.is_nan() && sla.is_nan(), "Sinusoidal off-lobe pole must be NaN, got ({slo}, {sla})");
        let (plo, pla) = sinu.inverse(0.0, y_pole);
        assert!(plo.abs() < 1e-12 && (pla - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        // Audit #70 #6: Southern Hemisphere LCC at South Pole must land on cone apex (rho == 0)
        let s_lcc = LambertConformalConic::new_2sp(
            Ellipsoid::WGS84,
            -5.0,
            -7.0,
            -6.0,
            0.0,
            0.0,
            0.0,
        );
        let (sx, sy) = s_lcc.forward(0.5, -std::f64::consts::FRAC_PI_2);
        assert!(sx.abs() < 1e-6, "Southern LCC South Pole X must be 0, got {sx}");
        assert!((sy - s_lcc.rho0).abs() < 1e-6, "Southern LCC South Pole Y must equal rho0, got {sy} vs {}", s_lcc.rho0);
    }

    #[test]
    fn oblique_stereographic_matches_epsg_gn7_2_amersfoort() {
        // EPSG Guidance Note 7-2 §1.3.13 worked example (EPSG:28992 Amersfoort / RD New):
        // lat = 53° N, lon = 6° E (on Bessel 1841) => E = 196105.283 m, N = 557057.739 m.
        let rd = ObliqueStereographic::new(
            Ellipsoid::BESSEL_1841,
            52.156_160_555_555_55,
            5.387_638_888_888_89,
            0.999_907_9,
            155_000.0,
            463_000.0,
        );
        let (e, n) = rd.forward(6.0_f64.to_radians(), 53.0_f64.to_radians());
        assert!((e - 196_105.283).abs() < 1e-2, "RD New easting {e}");
        assert!((n - 557_057.739).abs() < 1e-2, "RD New northing {n}");
        let (lon, lat) = rd.inverse(e, n);
        assert!((lon.to_degrees() - 6.0).abs() < 1e-9);
        assert!((lat.to_degrees() - 53.0).abs() < 1e-9);
    }

    #[test]
    fn swiss_oblique_mercator_matches_swisstopo_origin_and_roundtrips() {
        let lv95 = SwissObliqueMercator::new(
            Ellipsoid::BESSEL_1841,
            46.952_405_555_555_56,
            7.439_583_333_333_333,
            2_600_000.0,
            1_200_000.0,
        );
        // Bern origin maps to exact false easting/northing.
        let (e0, n0) = lv95.forward(
            7.439_583_333_333_333_f64.to_radians(),
            46.952_405_555_555_56_f64.to_radians(),
        );
        assert!((e0 - 2_600_000.0).abs() < 1e-3, "LV95 origin E={e0}");
        assert!((n0 - 1_200_000.0).abs() < 1e-3, "LV95 origin N={n0}");
        // Zurich (CH1903: 8.54°E, 47.37°N) round-trips to sub-nanometre.
        let (ez, nz) = lv95.forward(8.54_f64.to_radians(), 47.37_f64.to_radians());
        let (lon, lat) = lv95.inverse(ez, nz);
        assert!((lon.to_degrees() - 8.54).abs() < 1e-9);
        assert!((lat.to_degrees() - 47.37).abs() < 1e-9);
    }

    #[test]
    fn osgb36_bng_and_helmert_roundtrip() {
        // Caister Water Tower (OS Guide to Coordinate Systems):
        // OSGB36: 52°39'27.2531"N, 1°43'04.5177"E => BNG E = 651409.903 m, N = 313177.270 m
        let bng_tm = TransverseMercator::new(
            Ellipsoid::AIRY_1830,
            -2.0,
            49.0,
            0.999_601_271_7,
            400_000.0,
            -100_000.0,
        );
        let lat_osgb: f64 = 52.0 + 39.0 / 60.0 + 27.2531 / 3600.0;
        let lon_osgb: f64 = 1.0 + 43.0 / 60.0 + 4.5177 / 3600.0;
        let (e, n) = bng_tm.forward(lon_osgb.to_radians(), lat_osgb.to_radians());
        assert!((e - 651_409.903).abs() < 1e-2, "BNG E={e}");
        assert!((n - 313_177.270).abs() < 1e-2, "BNG N={n}");

        // Full EPSG:27700 via Helmert from WGS84 (52.657977°N, 1.716038°E) lands within ~2 m.
        let (dlon, dlat) = HelmertTransform::OSGB36_TO_WGS84.wgs84_to_datum_deg(1.716_038, 52.657_977);
        let (ew, nw) = bng_tm.forward(dlon.to_radians(), dlat.to_radians());
        assert!((ew - 651_409.9).abs() < 5.0, "WGS84->BNG E={ew}");
        assert!((nw - 313_177.3).abs() < 5.0, "WGS84->BNG N={nw}");

        // Exact algebraic inverse on all 4 built-in Helmert transforms.
        for (h, lon, lat) in [
            (HelmertTransform::OSGB36_TO_WGS84, -0.1276, 51.5072),
            (HelmertTransform::AMERSFOORT_TO_WGS84, 4.9041, 52.3676),
            (HelmertTransform::CH1903_TO_WGS84, 7.4474, 46.9480),
            (HelmertTransform::NAD27_TO_WGS84, -122.4194, 37.7749),
        ] {
            let (dlon, dlat) = h.wgs84_to_datum_deg(lon, lat);
            let (lon2, lat2) = h.datum_to_wgs84_deg(dlon, dlat);
            // 2D-only (h=0) round-trip across different ellipsoids has ~1 mm height-projection residual.
            assert!((lon2 - lon).abs() < 1e-7, "Helmert lon round-trip: {lon} -> {lon2}");
            assert!((lat2 - lat).abs() < 1e-7, "Helmert lat round-trip: {lat} -> {lat2}");
        }
    }

    #[test]
    fn equirectangular_and_mercator_roundtrip() {
        let eqc = Equirectangular::new(Ellipsoid::WGS84, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (ex, ey) = eqc.forward((-122.4_f64).to_radians(), 37.8_f64.to_radians());
        let (elo, ela) = eqc.inverse(ex, ey);
        assert!((elo.to_degrees() + 122.4).abs() < 1e-10);
        assert!((ela.to_degrees() - 37.8).abs() < 1e-10);

        let m2 = Mercator::new_2sp(Ellipsoid::WGS84, -41.0, 100.0, 0.0, 0.0);
        let (mx, my) = m2.forward(145.0_f64.to_radians(), (-37.8_f64).to_radians());
        let (mlo, mla) = m2.inverse(mx, my);
        assert!((mlo.to_degrees() - 145.0).abs() < 1e-10);
        assert!((mla.to_degrees() + 37.8).abs() < 1e-10);
    }
}
