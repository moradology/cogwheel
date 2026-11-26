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
        let lon = self.lon0 + theta / self.n;
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
}
