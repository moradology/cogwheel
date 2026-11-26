//! Exact analytical (`exactextract`-style) coverage-weighted & ellipsoidal area-weighted
//! zonal statistics on native raster pixel grids.
//!
//! For every pixel `(c, r)` in the native raster window `[c0..c1] × [r0..r1]`:
//! - **Exact coverage fraction** `cov ∈ [0, 1]` is computed analytically via Green's theorem
//!   edge-walking across grid cells (`O(V + perimeter_cells)` geometry work) followed by a
//!   single linear row scanline prefix sum (`O(win_w * win_h)`). Holes subtract, MultiPolygon
//!   parts add, and each cell is clamped to `[0, 1]`.
//! - **Ellipsoidal ground area** `area_px` (WGS84, m²) uses a closed-form authalic band area
//!   per row (`O(win_h)`) for Geographic (`EPSG:4326`) and Web Mercator (`EPSG:3857`) rasters,
//!   and a coarse `5×5` Jacobian lattice for other projected CRSs with a `< 1e-6` variation
//!   short-circuit so small UTM/Albers/LAEA polygons run at zero extra cost.

use std::cell::RefCell;
use std::sync::Arc;

use cogzilla_io::OpenedCog;
use cogzilla_proj::CrsModel;
use cogzilla_tiff::{Affine2D, PlanarConfig};

use crate::engine::{EngineError, TileEngine};
use crate::titiler::{PolygonMask, StatsOptions, compute_f32_band_stats};

/// WGS84 ellipsoid constants.
const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257_223_563;
const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);
const WGS84_B2: f64 = WGS84_A * WGS84_A * (1.0 - WGS84_E2);

/// Reusable thread-local scratch buffers so repeated zonal statistics calls avoid re-allocating.
struct ZonalScratch {
    area: Vec<f64>,
    run_delta: Vec<f64>,
    t_breaks: Vec<f64>,
}

impl ZonalScratch {
    fn new() -> Self {
        Self {
            area: Vec::new(),
            run_delta: Vec::new(),
            t_breaks: Vec::with_capacity(64),
        }
    }
}

thread_local! {
    static ZONAL_SCRATCH: RefCell<ZonalScratch> = RefCell::new(ZonalScratch::new());
}

/// WGS84 authalic zone integral `q(phi)` where `phi` is geodetic latitude in radians.
/// The ellipsoidal surface area between longitudes `[lam0, lam1]` and latitudes `[phi0, phi1]` is:
/// `A = (b^2 / 2) * |lam1 - lam0| * |q(phi1) - q(phi0)|`.
#[inline]
pub fn wgs84_authalic_q(phi_rad: f64) -> f64 {
    let e = WGS84_E2.sqrt();
    let s = phi_rad.sin();
    let es = e * s;
    s / (1.0 - es * es) + ((1.0 + es) / (1.0 - es)).ln() / (2.0 * e)
}

/// Exact WGS84 ellipsoidal area (m²) of a geographic cell `[lon0, lon1] × [lat0, lat1]` in degrees.
#[inline]
pub fn wgs84_geographic_cell_area_m2(lon0_deg: f64, lon1_deg: f64, lat0_deg: f64, lat1_deg: f64) -> f64 {
    let dlam = (lon1_deg - lon0_deg).abs().to_radians();
    let phi0 = lat0_deg.clamp(-90.0, 90.0).to_radians();
    let phi1 = lat1_deg.clamp(-90.0, 90.0).to_radians();
    0.5 * WGS84_B2 * dlam * (wgs84_authalic_q(phi1) - wgs84_authalic_q(phi0)).abs()
}

/// Minimum per-pixel weight to accumulate into weighted sums (`count`, `sum`, `area`, `mean`).
const COV_EPS: f64 = 1e-12;

/// Minimum per-pixel weight for pixel-set statistics (`valid_pixels`, `masked_pixels`,
/// `valid_percent`, `min`, `max`, `unique`, `majority`, `minority`, categorical keys) so
/// `1e-9..1e-8 px²` CRS round-trip residue on aligned boundaries never admits neighbor pixels.
const PIXEL_SET_EPS: f64 = 1e-6;

/// Ordered finite `f64` wrapper for exact categorical keying (normalizing `-0.0` to `+0.0`).
#[derive(Debug, Clone, Copy)]
struct OrderedF64(f64);

impl OrderedF64 {
    #[inline]
    fn new(v: f64) -> Self {
        Self(if v == 0.0 { 0.0 } else { v })
    }
}

impl PartialEq for OrderedF64 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for OrderedF64 {}
impl PartialOrd for OrderedF64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF64 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[inline]
fn format_category_key(v: f64) -> String {
    let v = if v == 0.0 { 0.0 } else { v };
    if v.fract() == 0.0 && v.abs() < 1e15 {
        (v as i64).to_string()
    } else {
        v.to_string()
    }
}

/// Semantic geometry dimensionality for zonal statistics (`area` vs `length` vs point weights).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZonalGeomKind {
    Polygon,
    Line,
    Point,
}

/// WGS84 ellipsoidal arc length (metres) of a short segment between two pixel coordinates
/// `(px0, py0)` and `(px1, py1)` using WGS84 meridional and prime-vertical radii of curvature.
#[inline]
fn wgs84_segment_length_m(
    crs: &CrsModel,
    p2c: &Affine2D,
    px0: f64,
    py0: f64,
    px1: f64,
    py1: f64,
) -> f64 {
    let (sx0, sy0) = p2c.forward(px0, py0);
    let (sx1, sy1) = p2c.forward(px1, py1);
    let (lo0, la0) = crs.to_lonlat_deg(sx0, sy0);
    let (lo1, la1) = crs.to_lonlat_deg(sx1, sy1);
    if !(lo0.is_finite() && la0.is_finite() && lo1.is_finite() && la1.is_finite()) {
        return 0.0;
    }
    let mut dlo = lo1 - lo0;
    if dlo > 180.0 {
        dlo -= 360.0;
    } else if dlo < -180.0 {
        dlo += 360.0;
    }
    let dla = la1 - la0;
    let phi = (0.5 * (la0 + la1)).clamp(-89.9999, 89.9999).to_radians();
    let s = phi.sin();
    let denom = 1.0 - WGS84_E2 * s * s;
    let rm = WGS84_A * (1.0 - WGS84_E2) / (denom * denom.sqrt());
    let rn = WGS84_A / denom.sqrt();
    let dx = dlo.to_radians() * rn * phi.cos().abs();
    let dy = dla.to_radians() * rm;
    dx.hypot(dy)
}

#[inline]
fn snap_near_int(v: f64) -> f64 {
    let r = v.round();
    if (v - r).abs() < 1e-6 { r } else { v }
}

/// Local WGS84 ellipsoidal ground area (m²) of 1 pixel centered at `(col + 0.5, row + 0.5)` via
/// central finite-difference Jacobian of `(col, row) -> (lon, lat)` and WGS84 principal radii of
/// curvature `R_M * R_N * cos(phi)`.
fn wgs84_projected_pixel_area_m2(crs: &CrsModel, p2c: &Affine2D, col: f64, row: f64) -> f64 {
    const EPS: f64 = 1e-3;
    let cx = col + 0.5;
    let cy = row + 0.5;
    let (sxc, syc) = p2c.forward(cx, cy);
    let (_, lac) = crs.to_lonlat_deg(sxc, syc);
    let (sxm, sym) = p2c.forward(cx - EPS, cy);
    let (sxp, syp) = p2c.forward(cx + EPS, cy);
    let (sxn, syn) = p2c.forward(cx, cy - EPS);
    let (sxs, sys) = p2c.forward(cx, cy + EPS);
    let (lom, lam) = crs.to_lonlat_deg(sxm, sym);
    let (lop, lap) = crs.to_lonlat_deg(sxp, syp);
    let (lon, lan) = crs.to_lonlat_deg(sxn, syn);
    let (los, las) = crs.to_lonlat_deg(sxs, sys);
    if !(lac.is_finite()
        && lom.is_finite()
        && lam.is_finite()
        && lop.is_finite()
        && lap.is_finite()
        && lon.is_finite()
        && lan.is_finite()
        && los.is_finite()
        && las.is_finite())
    {
        return 0.0;
    }
    let wrap_dlon = |d: f64| -> f64 {
        if d > 180.0 {
            d - 360.0
        } else if d < -180.0 {
            d + 360.0
        } else {
            d
        }
    };
    let inv_2eps = 0.5 / EPS;
    let dlo_dc = wrap_dlon(lop - lom).to_radians() * inv_2eps;
    let dla_dc = (lap - lam).to_radians() * inv_2eps;
    let dlo_dr = wrap_dlon(los - lon).to_radians() * inv_2eps;
    let dla_dr = (las - lan).to_radians() * inv_2eps;
    let det = (dlo_dc * dla_dr - dlo_dr * dla_dc).abs();
    let phi = lac.clamp(-89.9999, 89.9999).to_radians();
    let s = phi.sin();
    let denom = 1.0 - WGS84_E2 * s * s;
    let rm = WGS84_A * (1.0 - WGS84_E2) / (denom * denom.sqrt());
    let rn = WGS84_A / denom.sqrt();
    det * rm * rn * phi.cos().abs()
}

/// Ellipsoidal pixel ground-area model over a native pixel window `[c0..c0+win_w) × [r0..r0+win_h)`.
pub enum PixelAreaModel {
    /// Constant area per pixel (variation `< 1e-6` across the window, e.g. small UTM polygon).
    /// Allows skipping per-pixel area weighting in mean/std/quantiles (zero overhead).
    Constant(f64),
    /// Per-row exact area (m²) for Geographic and Web Mercator rasters (`O(win_h)`).
    PerRow(Vec<f64>),
    /// Coarse `N × N` bilinearly interpolated lattice for large projected polygons.
    Lattice {
        nx: usize,
        ny: usize,
        vals: Vec<f64>,
        win_w: usize,
        win_h: usize,
    },
}

impl PixelAreaModel {
    pub fn build(
        crs: &CrsModel,
        p2c: &Affine2D,
        c0: usize,
        r0: usize,
        win_w: usize,
        win_h: usize,
    ) -> Self {
        if win_w == 0 || win_h == 0 {
            return Self::Constant(0.0);
        }
        // Fast exact O(win_h) closed-form authalic band area for axis-aligned Geographic & EPSG:3857.
        if p2c.is_axis_aligned() {
            match crs {
                CrsModel::Geographic => {
                    let dlon = p2c.a.abs();
                    let mut rows = Vec::with_capacity(win_h);
                    let mut min_a = f64::INFINITY;
                    let mut max_a = 0.0_f64;
                    for dy in 0..win_h {
                        let r = (r0 + dy) as f64;
                        let (_, lat0) = p2c.forward(c0 as f64, r);
                        let (_, lat1) = p2c.forward(c0 as f64, r + 1.0);
                        let a = wgs84_geographic_cell_area_m2(0.0, dlon, lat0, lat1);
                        min_a = min_a.min(a);
                        max_a = max_a.max(a);
                        rows.push(a);
                    }
                    if max_a > 0.0 && (max_a - min_a) / max_a < 1e-6 {
                        return Self::Constant(0.5 * (min_a + max_a));
                    }
                    return Self::PerRow(rows);
                }
                CrsModel::Epsg3857 => {
                    let (mx0, _) = p2c.forward(c0 as f64, r0 as f64);
                    let (mx1, _) = p2c.forward((c0 + 1) as f64, r0 as f64);
                    let (lon0, _) = cogzilla_proj::mercator_to_lonlat_deg(mx0, 0.0);
                    let (lon1, _) = cogzilla_proj::mercator_to_lonlat_deg(mx1, 0.0);
                    let dlon = (lon1 - lon0).abs();
                    let mut rows = Vec::with_capacity(win_h);
                    let mut min_a = f64::INFINITY;
                    let mut max_a = 0.0_f64;
                    for dy in 0..win_h {
                        let r = (r0 + dy) as f64;
                        let (_, my0) = p2c.forward(c0 as f64, r);
                        let (_, my1) = p2c.forward(c0 as f64, r + 1.0);
                        let (_, lat0) = cogzilla_proj::mercator_to_lonlat_deg(0.0, my0);
                        let (_, lat1) = cogzilla_proj::mercator_to_lonlat_deg(0.0, my1);
                        let a = wgs84_geographic_cell_area_m2(0.0, dlon, lat0, lat1);
                        min_a = min_a.min(a);
                        max_a = max_a.max(a);
                        rows.push(a);
                    }
                    if max_a > 0.0 && (max_a - min_a) / max_a < 1e-6 {
                        return Self::Constant(0.5 * (min_a + max_a));
                    }
                    return Self::PerRow(rows);
                }
                _ => {}
            }
        }

        // Fast exact constant area for WGS84/GRS80 equal-area projections (Albers, LAEA, CEA,
        // Equal Earth, Sinusoidal): every pixel has identical ellipsoidal ground area |det(p2c)| * unit^2.
        let equal_area_ellipsoidal = match crs {
            CrsModel::Albers(p) => p.e2 > 0.0,
            CrsModel::Laea(p) => p.e2 > 0.0,
            CrsModel::CylindricalEqualArea(p) => p.e2 > 0.0,
            CrsModel::EqualEarth(p) => p.e2 > 0.0,
            CrsModel::Sinusoidal(p) => p.e2 > 0.0,
            _ => false,
        };
        if equal_area_ellipsoidal {
            let unit = crs.linear_unit_metres();
            let det = (p2c.a * p2c.e - p2c.b * p2c.d).abs();
            return Self::Constant(det * unit * unit);
        }

        // General projected / rotated CRS: evaluate exact WGS84 Jacobian on a 5×5 lattice of
        // in-bounds pixel centers [c0 .. c0 + win_w - 1] × [r0 .. r0 + win_h - 1].
        let nx = if win_w <= 1 { 1 } else { 5 };
        let ny = if win_h <= 1 { 1 } else { 5 };
        let mut vals = Vec::with_capacity(nx * ny);
        let mut min_a = f64::INFINITY;
        let mut max_a = 0.0_f64;
        let mut sum_a = 0.0_f64;
        let mut valid_nodes = 0usize;
        for iy in 0..ny {
            let fy = if ny == 1 { 0.0 } else { iy as f64 / (ny - 1) as f64 };
            let row = r0 as f64 + fy * ((win_h - 1) as f64);
            for ix in 0..nx {
                let fx = if nx == 1 { 0.0 } else { ix as f64 / (nx - 1) as f64 };
                let col = c0 as f64 + fx * ((win_w - 1) as f64);
                let a = wgs84_projected_pixel_area_m2(crs, p2c, col, row);
                if a > 0.0 {
                    min_a = min_a.min(a);
                    max_a = max_a.max(a);
                    sum_a += a;
                    valid_nodes += 1;
                }
                vals.push(a);
            }
        }
        if valid_nodes == 0 {
            return Self::Constant(0.0);
        }
        let mean_a = sum_a / (valid_nodes as f64);
        for v in &mut vals {
            if *v <= 0.0 {
                *v = mean_a;
            }
        }
        if max_a > 0.0 && (max_a - min_a) / max_a < 1e-6 {
            return Self::Constant(mean_a);
        }
        Self::Lattice {
            nx,
            ny,
            vals,
            win_w,
            win_h,
        }
    }

    #[inline(always)]
    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    #[inline(always)]
    pub fn area_at(&self, dx: usize, dy: usize) -> f64 {
        match self {
            Self::Constant(a) => *a,
            Self::PerRow(rows) => rows[dy],
            Self::Lattice {
                nx,
                ny,
                vals,
                win_w,
                win_h,
            } => {
                let fx = if *nx <= 1 {
                    0.0
                } else {
                    ((dx as f64 + 0.5) / (*win_w as f64)).clamp(0.0, 1.0) * ((*nx - 1) as f64)
                };
                let fy = if *ny <= 1 {
                    0.0
                } else {
                    ((dy as f64 + 0.5) / (*win_h as f64)).clamp(0.0, 1.0) * ((*ny - 1) as f64)
                };
                let x0 = (fx.floor() as usize).min(nx.saturating_sub(1));
                let x1 = (x0 + 1).min(nx.saturating_sub(1));
                let y0 = (fy.floor() as usize).min(ny.saturating_sub(1));
                let y1 = (y0 + 1).min(ny.saturating_sub(1));
                let tx = fx - x0 as f64;
                let ty = fy - y0 as f64;
                let v00 = vals[y0 * nx + x0];
                let v10 = vals[y0 * nx + x1];
                let v01 = vals[y1 * nx + x0];
                let v11 = vals[y1 * nx + x1];
                let top = v00 + (v10 - v00) * tx;
                let bot = v01 + (v11 - v01) * tx;
                top + (bot - top) * ty
            }
        }
    }
}

/// Compute uniform `360°` longitude shifts (`0.0`, `+360.0`, `-360.0`) for a whole ring so that
/// every overlapping shift intersects a geographic raster's native `[x_lo, x_hi]` longitude span
/// without tearing or inverting rings that cross `0°` or `±180°`.
fn geo_ring_lon_shifts(
    ring: &[(f64, f64)],
    ifd: &cogzilla_tiff::IfdMeta,
    is_geo_dataset: bool,
) -> Vec<f64> {
    if !is_geo_dataset || ring.is_empty() {
        return vec![0.0];
    }
    let a = ifd.pixel_to_crs.forward(0.0, 0.0).0;
    let b = ifd.pixel_to_crs.forward(ifd.width as f64, 0.0).0;
    let (x_lo, x_hi) = if a <= b { (a, b) } else { (b, a) };
    let mut r_lo = f64::INFINITY;
    let mut r_hi = f64::NEG_INFINITY;
    for &(lon, _) in ring {
        if lon.is_finite() {
            r_lo = r_lo.min(lon);
            r_hi = r_hi.max(lon);
        }
    }
    if !r_lo.is_finite() || !r_hi.is_finite() {
        return vec![0.0];
    }
    let mut shifts = Vec::with_capacity(2);
    for shift in [0.0_f64, 360.0, -360.0] {
        if r_lo + shift < x_hi && r_hi + shift > x_lo {
            shifts.push(shift);
        }
    }
    if shifts.is_empty() {
        shifts.push(0.0);
    }
    shifts
}

/// Shift a single point's longitude by `±360°` if needed to land inside a geographic raster's
/// native `[x_lo, x_hi]` span.
fn geo_point_lon_shift(lon: f64, ifd: &cogzilla_tiff::IfdMeta, is_geo_dataset: bool) -> f64 {
    if !is_geo_dataset || !lon.is_finite() {
        return lon;
    }
    let a = ifd.pixel_to_crs.forward(0.0, 0.0).0;
    let b = ifd.pixel_to_crs.forward(ifd.width as f64, 0.0).0;
    let (x_lo, x_hi) = if a <= b { (a, b) } else { (b, a) };
    if lon < x_lo && lon + 360.0 <= x_hi {
        lon + 360.0
    } else if lon > x_hi && lon - 360.0 >= x_lo {
        lon - 360.0
    } else {
        lon
    }
}

/// Densify a ring in WGS84 `(lon, lat)` degrees before transforming into a non-geographic dataset CRS,
/// so straight GeoJSON edges do not introduce boundary projection error.
/// Tolerance: `0.01°` (~1.1 km) for normal polygons (`<= 2048` vertices), relaxed to `0.1°` for
/// high-vertex polygons (`> 2048` vertices), capped at 64 subdivisions per edge.
#[allow(clippy::too_many_arguments)]
pub fn densify_ring_lonlat_to_pixel(
    ring: &[(f64, f64)],
    crs: &CrsModel,
    crs_to_pixel: &Affine2D,
    gcp_poly: Option<&cogzilla_tiff::GcpPoly>,
    is_geo_dataset: bool,
    densify_lonlat: bool,
    lon_shift_deg: f64,
    total_vertices_hint: usize,
) -> Vec<(f64, f64)> {
    if ring.len() < 2 {
        return Vec::new();
    }
    let map_pt = |lon: f64, lat: f64| -> (f64, f64) {
        let adj_lon = lon + lon_shift_deg;
        let (sx, sy) = if is_geo_dataset {
            (adj_lon, lat)
        } else {
            crs.from_lonlat_deg(adj_lon, lat)
        };
        let (col, row) = if let Some(poly) = gcp_poly {
            poly.crs_to_pixel(sx, sy)
        } else {
            crs_to_pixel.forward(sx, sy)
        };
        // Snap vertices within 1e-6 px of an integer grid line onto it before clipping so
        // CRS round-trip noise (e.g. UTM -> WGS84 -> UTM ~1e-9 px) never bleeds into neighbors.
        let snap_grid = |v: f64| -> f64 {
            let r = v.round();
            if (v - r).abs() < 1e-6 { r } else { v }
        };
        (snap_grid(col), snap_grid(row))
    };
    if !densify_lonlat {
        return ring.iter().map(|&(lo, la)| map_pt(lo, la)).collect();
    }
    let max_step_deg = if total_vertices_hint > 2048 { 0.1 } else { 0.01 };
    let mut out = Vec::with_capacity(ring.len());
    for i in 0..ring.len() {
        let (x0, y0) = ring[i];
        out.push(map_pt(x0, y0));
        if i + 1 < ring.len() {
            let (x1, y1) = ring[i + 1];
            let dist = (x1 - x0).hypot(y1 - y0);
            if dist > max_step_deg {
                let segs = ((dist / max_step_deg).ceil() as usize).clamp(1, 64);
                for k in 1..segs {
                    let t = k as f64 / segs as f64;
                    out.push(map_pt(x0 + t * (x1 - x0), y0 + t * (y1 - y0)));
                }
            }
        }
    }
    out
}

/// Compute signed area of a closed ring in pixel coordinates (`> 0` for CCW in standard Cartesian).
#[inline]
fn ring_signed_area(ring: &[(f64, f64)]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..ring.len() {
        let (x0, y0) = ring[i];
        let (x1, y1) = ring[(i + 1) % ring.len()];
        sum += x0 * y1 - x1 * y0;
    }
    0.5 * sum
}

/// Traverse one directed edge `(x0, y0) -> (x1, y1)` across the pixel grid `[c0..c1) × [r0..r1)`
/// and accumulate exact Green's-theorem per-cell area and scanline run deltas.
fn accumulate_edge_exact(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    c0: i64,
    r0: i64,
    win_w: usize,
    win_h: usize,
    area: &mut [f64],
    run_delta: &mut [f64],
    t_breaks: &mut Vec<f64>,
) {
    let dy_total = y1 - y0;
    if dy_total == 0.0 || !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
        return;
    }
    let r1 = r0 + win_h as i64;
    let c1 = c0 + win_w as i64;

    // Quick vertical reject if edge is completely outside [r0, r1].
    let y_min = y0.min(y1);
    let y_max = y0.max(y1);
    if y_max <= r0 as f64 || y_min >= r1 as f64 {
        return;
    }

    // Clip segment to y ∈ [r0, r1].
    let mut t_start = 0.0_f64;
    let mut t_end = 1.0_f64;
    if dy_total > 0.0 {
        if y0 < r0 as f64 {
            t_start = t_start.max((r0 as f64 - y0) / dy_total);
        }
        if y1 > r1 as f64 {
            t_end = t_end.min((r1 as f64 - y0) / dy_total);
        }
    } else {
        if y0 > r1 as f64 {
            t_start = t_start.max((r1 as f64 - y0) / dy_total);
        }
        if y1 < r0 as f64 {
            t_end = t_end.min((r0 as f64 - y0) / dy_total);
        }
    }
    if t_end <= t_start {
        return;
    }

    let dx_total = x1 - x0;
    let sx0 = x0 + t_start * dx_total;
    let sy0 = y0 + t_start * dy_total;
    let sx1 = x0 + t_end * dx_total;
    let sy1 = y0 + t_end * dy_total;

    t_breaks.clear();
    t_breaks.push(t_start);
    t_breaks.push(t_end);

    // Collect all horizontal grid-line crossings y = k strictly inside (t_start, t_end).
    let ry_lo = sy0.min(sy1).ceil() as i64;
    let ry_hi = sy0.max(sy1).floor() as i64;
    for ry in ry_lo.max(r0)..=ry_hi.min(r1) {
        let t = (ry as f64 - y0) / dy_total;
        if t > t_start && t < t_end {
            t_breaks.push(t);
        }
    }

    // Collect all vertical grid-line crossings x = k strictly inside (t_start, t_end) and within [c0, c1].
    if dx_total != 0.0 {
        let cx_lo = sx0.min(sx1).ceil() as i64;
        let cx_hi = sx0.max(sx1).floor() as i64;
        for cx in cx_lo.max(c0)..=cx_hi.min(c1) {
            let t = (cx as f64 - x0) / dx_total;
            if t > t_start && t < t_end {
                t_breaks.push(t);
            }
        }
    }

    t_breaks.sort_unstable_by(|a, b| a.total_cmp(b));

    let stride_delta = win_w + 1;
    for k in 0..t_breaks.len() - 1 {
        let ta = t_breaks[k];
        let tb = t_breaks[k + 1];
        if tb - ta <= 1e-15 {
            continue;
        }
        let ax = x0 + ta * dx_total;
        let ay = y0 + ta * dy_total;
        let bx = x0 + tb * dx_total;
        let by = y0 + tb * dy_total;
        let seg_dy = by - ay;
        if seg_dy == 0.0 {
            continue;
        }
        let mid_y = 0.5 * (ay + by);
        let r = mid_y.floor() as i64;
        if r < r0 || r >= r1 {
            continue;
        }
        let row_idx = (r - r0) as usize;
        let mid_x = 0.5 * (ax + bx);
        let c = mid_x.floor() as i64;
        if c < c0 {
            // Segment lies strictly to the left of the window: contributes -seg_dy to every cell in this row.
            run_delta[row_idx * stride_delta] += -seg_dy;
        } else if c < c1 {
            let col_idx = (c - c0) as usize;
            area[row_idx * win_w + col_idx] += (mid_x - (c as f64 + 1.0)) * seg_dy;
            run_delta[row_idx * stride_delta + col_idx + 1] += -seg_dy;
        }
        // If c >= c1, segment lies strictly to the right of the window: zero contribution to [c0, c1).
    }
}

/// Rasterize exact analytical coverage fractions `∈ [0.0, 1.0]` for a collection of polygons
/// (each polygon = `[outer_ring, hole1, ...]`, coordinates in native pixel space), optional
/// polylines (weighted by clipped segment length per cell, clamped to `1.0`), and optional
/// points (weight `1.0` at the containing pixel) over window `[c0..c0+win_w) × [r0..r0+win_h)`.
pub fn rasterize_exact_coverage(
    polygons_px: &[Vec<Vec<(f64, f64)>>],
    lines_px: &[Vec<(f64, f64)>],
    points_px: &[(f64, f64)],
    c0: usize,
    r0: usize,
    win_w: usize,
    win_h: usize,
) -> Vec<f64> {
    let n_px = win_w * win_h;
    let mut cov = vec![0.0_f64; n_px];
    if n_px == 0 {
        return cov;
    }
    let c0_i = c0 as i64;
    let r0_i = r0 as i64;
    let stride_delta = win_w + 1;

    ZONAL_SCRATCH.with(|cell| {
        let mut st = cell.borrow_mut();
        st.area.resize(n_px, 0.0);
        st.run_delta.resize(win_h * stride_delta, 0.0);

        for poly in polygons_px {
            st.area[..n_px].fill(0.0);
            st.run_delta[..win_h * stride_delta].fill(0.0);

            for (ring_idx, ring) in poly.iter().enumerate() {
                if ring.len() < 3 {
                    continue;
                }
                let sa = ring_signed_area(ring);
                if sa.abs() < 1e-18 {
                    continue;
                }
                // Outer ring (ring_idx == 0) must be positively oriented (CCW: sa > 0);
                // Holes (ring_idx > 0) must be negatively oriented (CW: sa < 0) so they subtract.
                let need_reverse = if ring_idx == 0 { sa < 0.0 } else { sa > 0.0 };
                let n = ring.len();
                let ZonalScratch {
                    area,
                    run_delta,
                    t_breaks,
                } = &mut *st;
                if need_reverse {
                    for i in 0..n {
                        let (x0, y0) = ring[(i + 1) % n];
                        let (x1, y1) = ring[i];
                        accumulate_edge_exact(
                            x0, y0, x1, y1, c0_i, r0_i, win_w, win_h, area, run_delta, t_breaks,
                        );
                    }
                } else {
                    for i in 0..n {
                        let (x0, y0) = ring[i];
                        let (x1, y1) = ring[(i + 1) % n];
                        accumulate_edge_exact(
                            x0, y0, x1, y1, c0_i, r0_i, win_w, win_h, area, run_delta, t_breaks,
                        );
                    }
                }
            }

            // Single linear scanline pass per row: combine prefix run with boundary cell area.
            for row in 0..win_h {
                let d_off = row * stride_delta;
                let a_off = row * win_w;
                let mut prefix = 0.0_f64;
                for col in 0..win_w {
                    prefix += st.run_delta[d_off + col];
                    let v = (prefix + st.area[a_off + col]).clamp(0.0, 1.0);
                    if v > 1e-12 {
                        cov[a_off + col] = (cov[a_off + col] + v).min(1.0);
                    }
                }
            }
        }

        // LineString / MultiLineString: weight each touched cell by its segment length inside the cell (clamped to 1.0).
        for line in lines_px {
            for seg in line.windows(2) {
                let (x0, y0) = seg[0];
                let (x1, y1) = seg[1];
                if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
                    continue;
                }
                let dx = x1 - x0;
                let dy = y1 - y0;
                let len = dx.hypot(dy);
                if len <= 1e-15 {
                    continue;
                }
                st.t_breaks.clear();
                st.t_breaks.push(0.0);
                st.t_breaks.push(1.0);
                if dx != 0.0 {
                    let lo = x0.min(x1).ceil() as i64;
                    let hi = x0.max(x1).floor() as i64;
                    for cx in lo.max(c0_i)..=hi.min(c0_i + win_w as i64) {
                        let t = (cx as f64 - x0) / dx;
                        if t > 0.0 && t < 1.0 {
                            st.t_breaks.push(t);
                        }
                    }
                }
                if dy != 0.0 {
                    let lo = y0.min(y1).ceil() as i64;
                    let hi = y0.max(y1).floor() as i64;
                    for ry in lo.max(r0_i)..=hi.min(r0_i + win_h as i64) {
                        let t = (ry as f64 - y0) / dy;
                        if t > 0.0 && t < 1.0 {
                            st.t_breaks.push(t);
                        }
                    }
                }
                st.t_breaks.sort_unstable_by(|a, b| a.total_cmp(b));
                for k in 0..st.t_breaks.len() - 1 {
                    let ta = st.t_breaks[k];
                    let tb = st.t_breaks[k + 1];
                    if tb - ta <= 1e-15 {
                        continue;
                    }
                    let mid_t = 0.5 * (ta + tb);
                    let mx = x0 + mid_t * dx;
                    let my = y0 + mid_t * dy;
                    let c = mx.floor() as i64;
                    let r = my.floor() as i64;
                    if c >= c0_i && c < c0_i + win_w as i64 && r >= r0_i && r < r0_i + win_h as i64 {
                        let idx = (r - r0_i) as usize * win_w + (c - c0_i) as usize;
                        let seg_len = (tb - ta) * len;
                        if polygons_px.is_empty() {
                            cov[idx] += seg_len;
                        } else {
                            cov[idx] = (cov[idx] + seg_len).min(1.0);
                        }
                    }
                }
            }
        }
        // Shrink oversized thread-local scratch buffers so blocking pool threads never retain
        // hundreds of MiB after a large 4096×4096 polygon request.
        const MAX_RETAINED_ELEMS: usize = 512 * 512;
        if st.area.capacity() > MAX_RETAINED_ELEMS {
            st.area = Vec::new();
        }
        if st.run_delta.capacity() > MAX_RETAINED_ELEMS {
            st.run_delta = Vec::new();
        }
        if st.t_breaks.capacity() > 4096 {
            st.t_breaks = Vec::new();
        }
    });

    // Point / MultiPoint: weight 1.0 per point at its containing pixel.
    for &(px, py) in points_px {
        if px.is_finite() && py.is_finite() {
            let c = px.floor() as i64;
            let r = py.floor() as i64;
            if c >= c0_i && c < c0_i + win_w as i64 && r >= r0_i && r < r0_i + win_h as i64 {
                let idx = (r - r0_i) as usize * win_w + (c - c0_i) as usize;
                if polygons_px.is_empty() && lines_px.is_empty() {
                    cov[idx] += 1.0;
                } else {
                    cov[idx] = 1.0;
                }
            }
        }
    }

    cov
}

/// Compute per-cell WGS84 geodesic segment lengths (metres) for `LineString` / `MultiLineString`
/// features over `[c0..c0+win_w) × [r0..r0+win_h)`.
fn rasterize_line_geodesic_lengths_m(
    lines_px: &[Vec<(f64, f64)>],
    crs: &CrsModel,
    p2c: &Affine2D,
    c0: usize,
    r0: usize,
    win_w: usize,
    win_h: usize,
) -> Vec<f64> {
    let mut len_m = vec![0.0_f64; win_w * win_h];
    if win_w == 0 || win_h == 0 || lines_px.is_empty() {
        return len_m;
    }
    let c0_i = c0 as i64;
    let r0_i = r0 as i64;
    let mut t_breaks: Vec<f64> = Vec::with_capacity(64);
    for line in lines_px {
        for seg in line.windows(2) {
            let (x0, y0) = seg[0];
            let (x1, y1) = seg[1];
            if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
                continue;
            }
            let dx = x1 - x0;
            let dy = y1 - y0;
            if dx.hypot(dy) <= 1e-15 {
                continue;
            }
            t_breaks.clear();
            t_breaks.push(0.0);
            t_breaks.push(1.0);
            if dx != 0.0 {
                let lo = x0.min(x1).ceil() as i64;
                let hi = x0.max(x1).floor() as i64;
                for cx in lo.max(c0_i)..=hi.min(c0_i + win_w as i64) {
                    let t = (cx as f64 - x0) / dx;
                    if t > 0.0 && t < 1.0 {
                        t_breaks.push(t);
                    }
                }
            }
            if dy != 0.0 {
                let lo = y0.min(y1).ceil() as i64;
                let hi = y0.max(y1).floor() as i64;
                for ry in lo.max(r0_i)..=hi.min(r0_i + win_h as i64) {
                    let t = (ry as f64 - y0) / dy;
                    if t > 0.0 && t < 1.0 {
                        t_breaks.push(t);
                    }
                }
            }
            t_breaks.sort_unstable_by(|a, b| a.total_cmp(b));
            for k in 0..t_breaks.len() - 1 {
                let ta = t_breaks[k];
                let tb = t_breaks[k + 1];
                if tb - ta <= 1e-15 {
                    continue;
                }
                let mid_t = 0.5 * (ta + tb);
                let mx = x0 + mid_t * dx;
                let my = y0 + mid_t * dy;
                let c = mx.floor() as i64;
                let r = my.floor() as i64;
                if c >= c0_i && c < c0_i + win_w as i64 && r >= r0_i && r < r0_i + win_h as i64 {
                    let idx = (r - r0_i) as usize * win_w + (c - c0_i) as usize;
                    let ax = x0 + ta * dx;
                    let ay = y0 + ta * dy;
                    let bx = x0 + tb * dx;
                    let by = y0 + tb * dy;
                    len_m[idx] += wgs84_segment_length_m(crs, p2c, ax, ay, bx, by);
                }
            }
        }
    }
    len_m
}

/// Compute exact coverage-weighted & ellipsoidal area-weighted band statistics over a native window.
///
/// Semantics:
/// - `count`: sum of pixel coverage fractions (equals true area in pixels).
/// - `sum`: coverage-weighted sum of valid pixel values (`sum(cov_i * v_i)`).
/// - `area`: ellipsoidal ground area (m²) covered by valid data (`sum(cov_i * area_px_i)`).
/// - `mean`, `std`, `median`, `percentiles`, `histogram`: weighted by `w_i = cov_i * area_px_i`
///   when `area_weighted == true` (short-circuited to `cov_i` when `area_model.is_constant()`),
///   or `w_i = cov_i` when `area_weighted == false`.
/// - `min`, `max`, `unique`, `majority`, `minority`: over valid pixels with `cov > 0`.
/// - `valid_pixels`, `masked_pixels`, `valid_percent`: counts of pixels with `cov > 0`.
#[allow(clippy::too_many_arguments)]
pub fn compute_weighted_band_stats(
    pixels: &[f32],
    cov: &[f64],
    win_w: usize,
    win_h: usize,
    area_model: &PixelAreaModel,
    area_weighted: bool,
    description: &str,
    opts: &StatsOptions,
) -> serde_json::Value {
    compute_weighted_band_stats_with_geom(
        pixels,
        cov,
        win_w,
        win_h,
        area_model,
        area_weighted,
        ZonalGeomKind::Polygon,
        None,
        None,
        description,
        opts,
    )
}

/// Exact shoelace area (in squared native IFD pixels) of a single polygon ring.
fn ring_area_px(ring: &[(f64, f64)]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..ring.len() {
        let (x0, y0) = ring[i];
        let (x1, y1) = ring[(i + 1) % ring.len()];
        sum += x0 * y1 - x1 * y0;
    }
    0.5 * sum.abs()
}

/// Total unclipped area (in squared native IFD pixels) of a polygon / MultiPolygon in IFD pixel
/// coordinates, including any portion extending outside the raster extent.
fn polygon_total_area_px(polys_px: &[Vec<Vec<(f64, f64)>>]) -> f64 {
    let mut total = 0.0_f64;
    for poly in polys_px {
        if let Some(outer) = poly.first() {
            let mut a = ring_area_px(outer);
            for hole in poly.iter().skip(1) {
                a -= ring_area_px(hole);
            }
            total += a.max(0.0);
        }
    }
    total
}

#[inline(always)]
fn f32_to_sortable_u32(v: f32) -> u32 {
    let bits = if v == 0.0 { 0u32 } else { v.to_bits() };
    let mask = ((bits as i32) >> 31) as u32 | 0x8000_0000;
    bits ^ mask
}

#[inline(always)]
fn sortable_u32_to_f32(k: u32) -> f32 {
    let mask = (((k ^ 0x8000_0000) as i32) >> 31) as u32 | 0x8000_0000;
    f32::from_bits(k ^ mask)
}

/// 4-pass 8-bit LSD radix sort keyed by an order-preserving `u32` (skips uniform byte passes).
#[inline]
fn radix_sort_by_u32<T: Copy + Default>(data: &mut [T], key: impl Fn(&T) -> u32) {
    let n = data.len();
    if n < 1024 {
        data.sort_unstable_by_key(|x| key(x));
        return;
    }
    let mut tmp = vec![T::default(); n];
    let mut in_tmp = false;
    for shift in [0u32, 8, 16, 24] {
        let (src, dst): (&[T], &mut [T]) = if in_tmp {
            (&tmp, data)
        } else {
            (data, &mut tmp)
        };
        let mut counts = [0usize; 256];
        for item in src {
            counts[((key(item) >> shift) & 0xFF) as usize] += 1;
        }
        if counts[((key(&src[0]) >> shift) & 0xFF) as usize] == n {
            continue;
        }
        let mut sum = 0usize;
        for c in &mut counts {
            let v = *c;
            *c = sum;
            sum += v;
        }
        for &item in src {
            let b = ((key(&item) >> shift) & 0xFF) as usize;
            let pos = counts[b];
            counts[b] = pos + 1;
            dst[pos] = item;
        }
        in_tmp = !in_tmp;
    }
    if in_tmp {
        data.copy_from_slice(&tmp);
    }
}

fn radix_sort_u32(data: &mut [u32]) {
    radix_sort_by_u32(data, |&k| k);
}

fn radix_sort_u32_f64(data: &mut [(u32, f64)]) {
    radix_sort_by_u32(data, |&(k, _)| k);
}

fn radix_sort_u32_f64_f64(data: &mut [(u32, f64, f64)]) {
    radix_sort_by_u32(data, |&(k, _, _)| k);
}

#[allow(clippy::too_many_arguments)]
pub fn compute_weighted_band_stats_with_geom(
    pixels: &[f32],
    cov: &[f64],
    win_w: usize,
    win_h: usize,
    area_model: &PixelAreaModel,
    area_weighted: bool,
    geom_kind: ZonalGeomKind,
    line_len_m: Option<&[f64]>,
    total_geom_area_px: Option<f64>,
    description: &str,
    opts: &StatsOptions,
) -> serde_json::Value {
    let use_area_w = geom_kind == ZonalGeomKind::Polygon
        && area_weighted
        && !area_model.is_constant();
    let const_poly_area = (geom_kind == ZonalGeomKind::Polygon && area_model.is_constant())
        .then(|| area_model.area_at(0, 0));
    let w_eq_c = !use_area_w && !(geom_kind == ZonalGeomKind::Line && area_weighted && line_len_m.is_some());
    let is_categorical =
        opts.categorical || opts.categories.as_ref().is_some_and(|c| !c.is_empty());

    let mut valid_pixels = 0usize;
    let mut masked_pixels = 0usize;
    let mut valid_pixels_all = 0usize;
    let mut masked_pixels_all = 0usize;
    let mut count_cov = 0.0_f64;
    let mut sum_cov = 0.0_f64;
    let mut area_valid_m2 = 0.0_f64;
    let mut length_valid_m = 0.0_f64;
    let mut w_total = 0.0_f64;
    let mut w_sum = 0.0_f64;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut min_v_all = f64::INFINITY;
    let mut max_v_all = f64::NEG_INFINITY;

    // Compact value storage sized to the exact number of covered pixels (`cov > COV_EPS`),
    // avoiding huge transient over-allocation on sparse lines/points spanning large windows.
    let est_cap = cov.iter().filter(|&&c| c > COV_EPS).count();
    let mut unit_keys: Vec<u32> = if w_eq_c {
        Vec::with_capacity(est_cap)
    } else {
        Vec::new()
    };
    let mut cov_vals: Vec<(u32, f64)> = Vec::new();
    let mut weighted_vals: Vec<(u32, f64, f64)> = if !w_eq_c {
        Vec::with_capacity(est_cap)
    } else {
        Vec::new()
    };
    let mut all_unit = w_eq_c;

    // Exact f64 key -> `(coverage_sum, area_m2_sum)` for categorical stats (T2 + T4)
    let mut cat_map: std::collections::BTreeMap<OrderedF64, (f64, f64)> =
        std::collections::BTreeMap::new();
    let mut cat_map_all: std::collections::BTreeMap<OrderedF64, (f64, f64)> =
        std::collections::BTreeMap::new();

    if w_eq_c {
        let total = win_w * win_h;
        for i in 0..total {
            let c = cov[i];
            if c <= COV_EPS {
                continue;
            }
            let in_pixel_set = c > PIXEL_SET_EPS;
            let v = pixels[i];
            if !v.is_finite() {
                masked_pixels_all += 1;
                if in_pixel_set {
                    masked_pixels += 1;
                }
                continue;
            }
            let vf = v as f64;
            let key_u32 = f32_to_sortable_u32(v);
            if geom_kind == ZonalGeomKind::Polygon && const_poly_area.is_none() {
                let dy = i / win_w;
                let dx = i - dy * win_w;
                area_valid_m2 += c * area_model.area_at(dx, dy);
            } else if geom_kind == ZonalGeomKind::Line {
                length_valid_m += line_len_m.map_or(c, |buf| buf[i]);
            }
            valid_pixels_all += 1;
            count_cov += c;
            sum_cov += c * vf;
            if vf < min_v_all {
                min_v_all = vf;
            }
            if vf > max_v_all {
                max_v_all = vf;
            }
            if in_pixel_set {
                valid_pixels += 1;
                if vf < min_v {
                    min_v = vf;
                }
                if vf > max_v {
                    max_v = vf;
                }
            }
            if all_unit {
                if c == 1.0 {
                    unit_keys.push(key_u32);
                } else {
                    all_unit = false;
                    cov_vals.reserve(est_cap.max(unit_keys.len() + 1));
                    for &uk in &unit_keys {
                        cov_vals.push((uk, 1.0));
                    }
                    unit_keys.clear();
                    unit_keys.shrink_to_fit();
                    cov_vals.push((key_u32, c));
                }
            } else {
                cov_vals.push((key_u32, c));
            }
            if is_categorical {
                let cell_area_m2 = if let Some(a_px) = const_poly_area {
                    c * a_px
                } else if geom_kind == ZonalGeomKind::Polygon {
                    let dy = i / win_w;
                    let dx = i - dy * win_w;
                    c * area_model.area_at(dx, dy)
                } else {
                    0.0
                };
                let key = OrderedF64::new(vf);
                if in_pixel_set {
                    let entry = cat_map.entry(key).or_insert((0.0, 0.0));
                    entry.0 += c;
                    entry.1 += cell_area_m2;
                } else if valid_pixels == 0 {
                    let entry = cat_map_all.entry(key).or_insert((0.0, 0.0));
                    entry.0 += c;
                    entry.1 += cell_area_m2;
                }
            }
        }
        w_total = count_cov;
        w_sum = sum_cov;
        if let Some(a_px) = const_poly_area {
            area_valid_m2 = count_cov * a_px;
        }
    } else {
        for dy in 0..win_h {
            let row_off = dy * win_w;
            for dx in 0..win_w {
                let i = row_off + dx;
                let c = cov[i];
                if c <= COV_EPS {
                    continue;
                }
                let in_pixel_set = c > PIXEL_SET_EPS;
                let v = pixels[i];
                if !v.is_finite() {
                    masked_pixels_all += 1;
                    if in_pixel_set {
                        masked_pixels += 1;
                    }
                    continue;
                }
                let vf = v as f64;
                let mut cell_area_m2 = 0.0_f64;
                let w = match geom_kind {
                    ZonalGeomKind::Polygon => {
                        let a_px = area_model.area_at(dx, dy);
                        cell_area_m2 = c * a_px;
                        area_valid_m2 += cell_area_m2;
                        if use_area_w { cell_area_m2 } else { c }
                    }
                    ZonalGeomKind::Line => {
                        let lm = line_len_m.map_or(c, |buf| buf[i]);
                        length_valid_m += lm;
                        if area_weighted && lm > 0.0 { lm } else { c }
                    }
                    ZonalGeomKind::Point => c,
                };
                valid_pixels_all += 1;
                count_cov += c;
                sum_cov += c * vf;
                w_total += w;
                w_sum += w * vf;
                if vf < min_v_all {
                    min_v_all = vf;
                }
                if vf > max_v_all {
                    max_v_all = vf;
                }
                if in_pixel_set {
                    valid_pixels += 1;
                    if vf < min_v {
                        min_v = vf;
                    }
                    if vf > max_v {
                        max_v = vf;
                    }
                }
                weighted_vals.push((f32_to_sortable_u32(v), w, c));
                if is_categorical {
                    let key = OrderedF64::new(vf);
                    if in_pixel_set {
                        let entry = cat_map.entry(key).or_insert((0.0, 0.0));
                        entry.0 += c;
                        entry.1 += cell_area_m2;
                    } else if valid_pixels == 0 {
                        let entry = cat_map_all.entry(key).or_insert((0.0, 0.0));
                        entry.0 += c;
                        entry.1 += cell_area_m2;
                    }
                }
            }
        }
    }

    // If a microscopic sub-pixel polygon had all pixels with `COV_EPS < c <= PIXEL_SET_EPS`,
    // fall back to the `COV_EPS` pixel set so valid_pixels/min/max are still populated.
    let use_set_eps = valid_pixels > 0;
    if !use_set_eps {
        valid_pixels = valid_pixels_all;
        masked_pixels = masked_pixels_all;
        min_v = min_v_all;
        max_v = max_v_all;
        if cat_map.is_empty() {
            cat_map = cat_map_all;
        }
    }

    if valid_pixels == 0 || w_total <= 0.0 {
        let mut empty = compute_f32_band_stats(&[], description, opts);
        if let Some(obj) = empty.as_object_mut() {
            let masked_out = total_geom_area_px
                .filter(|&a| a > COV_EPS)
                .unwrap_or(masked_pixels as f64);
            obj.insert("masked_pixels".into(), serde_json::json!(masked_out));
            match geom_kind {
                ZonalGeomKind::Polygon => {
                    obj.insert("area".into(), serde_json::json!(0.0));
                }
                ZonalGeomKind::Line => {
                    obj.insert("area".into(), serde_json::Value::Null);
                    obj.insert("length".into(), serde_json::json!(0.0));
                }
                ZonalGeomKind::Point => {
                    obj.insert("area".into(), serde_json::Value::Null);
                }
            }
        }
        return empty;
    }

    let mean = w_sum / w_total;
    let var_acc = if all_unit {
        let mut acc = 0.0_f64;
        for &k in &unit_keys {
            let d = sortable_u32_to_f32(k) as f64 - mean;
            acc += d * d;
        }
        acc
    } else if w_eq_c {
        let mut acc = 0.0_f64;
        for &(k, c) in &cov_vals {
            let d = sortable_u32_to_f32(k) as f64 - mean;
            acc += c * d * d;
        }
        acc
    } else {
        let mut acc = 0.0_f64;
        for &(k, w, _) in &weighted_vals {
            let d = sortable_u32_to_f32(k) as f64 - mean;
            acc += w * d * d;
        }
        acc
    };
    let std = (var_acc / w_total).max(0.0).sqrt();

    let (valid_percent, masked_pixels_out) = if let Some(poly_area) = total_geom_area_px
        && poly_area > COV_EPS
    {
        let diff = (poly_area - count_cov).max(0.0);
        if masked_pixels == 0 && diff <= 1e-6 * poly_area.max(1.0) {
            (100.0, 0.0)
        } else {
            (((count_cov / poly_area) * 100.0).clamp(0.0, 100.0), diff)
        }
    } else {
        let total_touched = valid_pixels + masked_pixels;
        let vp = if total_touched > 0 {
            ((valid_pixels as f64 / total_touched as f64) * 100.0).min(100.0)
        } else {
            0.0
        };
        (vp, masked_pixels as f64)
    };

    // Radix-sort once for weighted quantiles, unique count, majority/minority, and histogram.
    if all_unit {
        radix_sort_u32(&mut unit_keys);
    } else if w_eq_c {
        radix_sort_u32_f64(&mut cov_vals);
    } else {
        radix_sort_u32_f64_f64(&mut weighted_vals);
    }

    // Weighted quantile helper (linear interpolation on weighted midpoint CDF).
    let weighted_quantile = |pct: f64| -> f64 {
        let q = (pct / 100.0).clamp(0.0, 1.0);
        if all_unit {
            let n = unit_keys.len();
            if n == 1 {
                return sortable_u32_to_f32(unit_keys[0]) as f64;
            }
            let pos = q * (n as f64) - 0.5;
            if pos <= 0.0 {
                return sortable_u32_to_f32(unit_keys[0]) as f64;
            }
            if pos >= (n - 1) as f64 {
                return sortable_u32_to_f32(unit_keys[n - 1]) as f64;
            }
            let lo = pos.floor() as usize;
            let frac = pos - (lo as f64);
            let v0 = sortable_u32_to_f32(unit_keys[lo]) as f64;
            let v1 = sortable_u32_to_f32(unit_keys[lo + 1]) as f64;
            return v0 + frac * (v1 - v0);
        }
        let target = q * w_total;
        let mut cum = 0.0_f64;
        let mut prev_mid = 0.0_f64;
        if w_eq_c {
            if cov_vals.len() == 1 {
                return sortable_u32_to_f32(cov_vals[0].0) as f64;
            }
            let mut prev_v = sortable_u32_to_f32(cov_vals[0].0) as f64;
            for (idx, &(k, w)) in cov_vals.iter().enumerate() {
                let v = sortable_u32_to_f32(k) as f64;
                let mid = cum + 0.5 * w;
                if idx == 0 {
                    if target <= mid {
                        return v;
                    }
                } else if target <= mid {
                    let span = (mid - prev_mid).max(1e-30);
                    let frac = ((target - prev_mid) / span).clamp(0.0, 1.0);
                    return prev_v + frac * (v - prev_v);
                }
                cum += w;
                prev_mid = mid;
                prev_v = v;
            }
            sortable_u32_to_f32(cov_vals.last().unwrap().0) as f64
        } else {
            if weighted_vals.len() == 1 {
                return sortable_u32_to_f32(weighted_vals[0].0) as f64;
            }
            let mut prev_v = sortable_u32_to_f32(weighted_vals[0].0) as f64;
            for (idx, &(k, w, _)) in weighted_vals.iter().enumerate() {
                let v = sortable_u32_to_f32(k) as f64;
                let mid = cum + 0.5 * w;
                if idx == 0 {
                    if target <= mid {
                        return v;
                    }
                } else if target <= mid {
                    let span = (mid - prev_mid).max(1e-30);
                    let frac = ((target - prev_mid) / span).clamp(0.0, 1.0);
                    return prev_v + frac * (v - prev_v);
                }
                cum += w;
                prev_mid = mid;
                prev_v = v;
            }
            sortable_u32_to_f32(weighted_vals.last().unwrap().0) as f64
        }
    };

    let median = weighted_quantile(50.0);

    // Unique count, majority, minority over pixels in the pixel set (weighted by coverage `c`).
    let set_thresh = if use_set_eps { PIXEL_SET_EPS } else { COV_EPS };
    let mut unique_count = 0usize;
    let mut maj_val = min_v;
    let mut maj_w = -1.0_f64;
    let mut min_val_freq = min_v;
    let mut min_w = f64::INFINITY;
    if all_unit {
        let mut idx = 0;
        while idx < unit_keys.len() {
            let k = unit_keys[idx];
            let start = idx;
            idx += 1;
            while idx < unit_keys.len() && unit_keys[idx] == k {
                idx += 1;
            }
            let gc = (idx - start) as f64;
            let v = sortable_u32_to_f32(k) as f64;
            unique_count += 1;
            if gc > maj_w {
                maj_w = gc;
                maj_val = v;
            }
            if gc < min_w {
                min_w = gc;
                min_val_freq = v;
            }
        }
    } else if w_eq_c {
        let mut idx = 0;
        while idx < cov_vals.len() {
            let k = cov_vals[idx].0;
            let mut gc = 0.0_f64;
            let mut any_in_set = false;
            while idx < cov_vals.len() && cov_vals[idx].0 == k {
                let c = cov_vals[idx].1;
                if c > set_thresh {
                    gc += c;
                    any_in_set = true;
                }
                idx += 1;
            }
            if !any_in_set {
                continue;
            }
            let v = sortable_u32_to_f32(k) as f64;
            unique_count += 1;
            if gc > maj_w {
                maj_w = gc;
                maj_val = v;
            }
            if gc < min_w {
                min_w = gc;
                min_val_freq = v;
            }
        }
    } else {
        let mut idx = 0;
        while idx < weighted_vals.len() {
            let k = weighted_vals[idx].0;
            let mut gc = 0.0_f64;
            let mut any_in_set = false;
            while idx < weighted_vals.len() && weighted_vals[idx].0 == k {
                let c = weighted_vals[idx].2;
                if c > set_thresh {
                    gc += c;
                    any_in_set = true;
                }
                idx += 1;
            }
            if !any_in_set {
                continue;
            }
            let v = sortable_u32_to_f32(k) as f64;
            unique_count += 1;
            if gc > maj_w {
                maj_w = gc;
                maj_val = v;
            }
            if gc < min_w {
                min_w = gc;
                min_val_freq = v;
            }
        }
    }

    let count_out = snap_near_int(count_cov);
    let mut obj = serde_json::Map::new();
    obj.insert("min".into(), serde_json::json!(min_v));
    obj.insert("max".into(), serde_json::json!(max_v));
    obj.insert("mean".into(), serde_json::json!(mean));
    obj.insert("count".into(), serde_json::json!(count_out));
    obj.insert("sum".into(), serde_json::json!(sum_cov));
    obj.insert("std".into(), serde_json::json!(std));
    obj.insert("median".into(), serde_json::json!(median));
    obj.insert("majority".into(), serde_json::json!(maj_val));
    obj.insert("minority".into(), serde_json::json!(min_val_freq));
    obj.insert("unique".into(), serde_json::json!(unique_count as f64));
    if is_categorical {
        // T2 + T4: categorical histogram and `categories` always report exact coverage counts
        // summing to `count` (preserving float keys on float rasters), and `category_areas`
        // reports per-class ground area in m² (summing to `area`) for polygons.
        let mut cat_counts: Vec<f64> = Vec::new();
        let mut cat_vals: Vec<f64> = Vec::new();
        let mut cats_obj = serde_json::Map::new();
        let mut cat_areas_obj = serde_json::Map::new();
        if let Some(ref cats) = opts.categories
            && !cats.is_empty()
        {
            for &c_val in cats {
                let key = OrderedF64::new(c_val);
                let (cov_c, area_c) = cat_map.get(&key).copied().unwrap_or((0.0, 0.0));
                let cc = snap_near_int(cov_c);
                let kstr = format_category_key(key.0);
                cat_counts.push(cc);
                cat_vals.push(key.0);
                cats_obj.insert(kstr.clone(), serde_json::json!(cc));
                if geom_kind == ZonalGeomKind::Polygon {
                    cat_areas_obj.insert(kstr, serde_json::json!(area_c));
                }
            }
        } else {
            for (key, (cov_c, area_c)) in cat_map {
                let cc = snap_near_int(cov_c);
                let kstr = format_category_key(key.0);
                cat_counts.push(cc);
                cat_vals.push(key.0);
                cats_obj.insert(kstr.clone(), serde_json::json!(cc));
                if geom_kind == ZonalGeomKind::Polygon {
                    cat_areas_obj.insert(kstr, serde_json::json!(area_c));
                }
            }
        }
        obj.insert("histogram".into(), serde_json::json!([cat_counts, cat_vals]));
        obj.insert("categories".into(), serde_json::Value::Object(cats_obj));
        if geom_kind == ZonalGeomKind::Polygon {
            obj.insert("category_areas".into(), serde_json::Value::Object(cat_areas_obj));
        }
    } else {
        // T4: numeric histogram always reports coverage counts summing to `count`.
        let (h_lo, h_hi) = opts.histogram_range.unwrap_or((min_v, max_v));
        let edges: Vec<f64> = if let Some(ref custom_edges) = opts.histogram_edges
            && custom_edges.len() >= 2
        {
            custom_edges.clone()
        } else {
            let nbins = opts.histogram_bins.clamp(1, 1000);
            let degenerate = (h_hi - h_lo).abs() < 1e-12;
            let span = if degenerate { 1.0 } else { h_hi - h_lo };
            (0..=nbins)
                .map(|k| {
                    if k == nbins {
                        if degenerate { h_lo + 1.0 } else { h_hi }
                    } else {
                        h_lo + span * (k as f64) / (nbins as f64)
                    }
                })
                .collect()
        };
        let nbins = edges.len() - 1;
        let mut hist_w = vec![0.0_f64; nbins];
        if all_unit {
            let mut start = unit_keys
                .partition_point(|&k| (sortable_u32_to_f32(k) as f64) < edges[0]);
            for bin in 0..nbins {
                let end = if bin + 1 < nbins {
                    let hi = edges[bin + 1];
                    start
                        + unit_keys[start..]
                            .partition_point(|&k| (sortable_u32_to_f32(k) as f64) < hi)
                } else {
                    let hi = edges[nbins];
                    start
                        + unit_keys[start..]
                            .partition_point(|&k| (sortable_u32_to_f32(k) as f64) <= hi)
                };
                hist_w[bin] = (end - start) as f64;
                start = end;
            }
        } else if w_eq_c {
            let mut bin = 0usize;
            for &(k, c) in &cov_vals {
                let vf = sortable_u32_to_f32(k) as f64;
                if vf < edges[0] {
                    continue;
                }
                if vf > edges[nbins] {
                    break;
                }
                while bin + 1 < nbins && vf >= edges[bin + 1] {
                    bin += 1;
                }
                hist_w[bin] += c;
            }
        } else {
            let mut bin = 0usize;
            for &(k, _, c) in &weighted_vals {
                let vf = sortable_u32_to_f32(k) as f64;
                if vf < edges[0] {
                    continue;
                }
                if vf > edges[nbins] {
                    break;
                }
                while bin + 1 < nbins && vf >= edges[bin + 1] {
                    bin += 1;
                }
                hist_w[bin] += c;
            }
        }
        obj.insert("histogram".into(), serde_json::json!([hist_w, edges]));
    }
    obj.insert("valid_percent".into(), serde_json::json!(valid_percent));
    obj.insert("masked_pixels".into(), serde_json::json!(masked_pixels_out));
    obj.insert("valid_pixels".into(), serde_json::json!(valid_pixels as f64));
    match geom_kind {
        ZonalGeomKind::Polygon => {
            obj.insert("area".into(), serde_json::json!(area_valid_m2));
        }
        ZonalGeomKind::Line => {
            obj.insert("area".into(), serde_json::Value::Null);
            obj.insert("length".into(), serde_json::json!(length_valid_m));
        }
        ZonalGeomKind::Point => {
            obj.insert("area".into(), serde_json::Value::Null);
        }
    }

    let pcts: Vec<f64> = if opts.percentiles.is_empty() {
        vec![2.0, 98.0]
    } else {
        opts.percentiles.clone()
    };
    for p in pcts {
        let key = if p.fract() == 0.0 {
            format!("percentile_{}", p as i64)
        } else {
            format!("percentile_{p}")
        };
        obj.insert(key, serde_json::json!(weighted_quantile(p)));
    }
    if !description.is_empty() {
        obj.insert("description".into(), serde_json::json!(description));
    }
    serde_json::Value::Object(obj)
}

/// Compute exact analytical coverage-weighted & ellipsoidal area-weighted zonal statistics
/// directly on the COG's native pixel grid (zero resampling).
#[allow(clippy::too_many_arguments)]
pub async fn compute_native_exact_polygon_stats(
    engine: &TileEngine,
    cog: &Arc<OpenedCog>,
    crs: &CrsModel,
    pm: &PolygonMask,
    max_size: u32,
    unscale: bool,
    nodata_override: Option<f32>,
    opts: &StatsOptions,
) -> Result<serde_json::Value, EngineError> {
    let full = cog.index.full_res();
    let count = full.samples_per_pixel;
    let alpha_idx = cogzilla_codec::alpha_channel_index(count as usize, &full.extra_samples);
    let sel_bands: Vec<u16> = if let Some(ref exprs) = opts.expressions {
        let mut needed: Vec<u16> = Vec::new();
        for (_, prog) in exprs {
            for &b in &prog.used_bands {
                if !needed.contains(&b) {
                    needed.push(b);
                }
            }
        }
        needed.sort_unstable();
        needed
    } else {
        match &opts.bidx {
            Some(b) if !b.is_empty() => b.clone(),
            _ => (1..=count)
                .filter(|&b| Some((b - 1) as usize) != alpha_idx)
                .collect(),
        }
    };

    let geom_kind = if !pm.polygons.is_empty() {
        ZonalGeomKind::Polygon
    } else if !pm.lines.is_empty() {
        ZonalGeomKind::Line
    } else {
        ZonalGeomKind::Point
    };

    let empty_result = || -> serde_json::Value {
        let mut out = serde_json::Map::new();
        let insert_empty_geom_fields = |o: &mut serde_json::Map<String, serde_json::Value>| match geom_kind {
            ZonalGeomKind::Polygon => {
                o.insert("area".into(), serde_json::json!(0.0));
            }
            ZonalGeomKind::Line => {
                o.insert("area".into(), serde_json::Value::Null);
                o.insert("length".into(), serde_json::json!(0.0));
            }
            ZonalGeomKind::Point => {
                o.insert("area".into(), serde_json::Value::Null);
            }
        };
        if let Some(ref exprs) = opts.expressions {
            for (i, (desc, _)) in exprs.iter().enumerate() {
                let mut e = compute_f32_band_stats(&[], desc, opts);
                if let Some(o) = e.as_object_mut() {
                    insert_empty_geom_fields(o);
                }
                out.insert(format!("b{}", i + 1), e);
            }
        } else {
            for &b1 in &sel_bands {
                let desc = cog
                    .index
                    .band_descriptions
                    .get((b1 - 1) as usize)
                    .map(String::as_str)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .unwrap_or_else(|| format!("b{b1}"));
                let mut e = compute_f32_band_stats(&[], &desc, opts);
                if let Some(o) = e.as_object_mut() {
                    insert_empty_geom_fields(o);
                }
                out.insert(format!("b{b1}"), e);
            }
        }
        serde_json::Value::Object(out)
    };

    let total_verts: usize = pm
        .polygons
        .iter()
        .flat_map(|p| p.iter())
        .map(|r| r.len())
        .sum::<usize>()
        + pm.lines.iter().map(|l| l.len()).sum::<usize>()
        + pm.points.len();
    let is_geo_dataset = matches!(crs, CrsModel::Geographic);

    let densify_lonlat = !is_geo_dataset && !pm.coord_crs_projected;

    let project_to_ifd = |ifd: &cogzilla_tiff::IfdMeta| -> Result<
        (
            Vec<Vec<Vec<(f64, f64)>>>,
            f64,
            Vec<Vec<(f64, f64)>>,
            Vec<(f64, f64)>,
            usize,
            usize,
            usize,
            usize,
        ),
        EngineError,
    > {
        let c2p = ifd.pixel_to_crs.invert().ok_or_else(|| {
            EngineError::BadRequest("dataset has singular affine transform".into())
        })?;
        let mut min_c = f64::INFINITY;
        let mut max_c = f64::NEG_INFINITY;
        let mut min_r = f64::INFINITY;
        let mut max_r = f64::NEG_INFINITY;

        let mut polys_px = Vec::with_capacity(pm.polygons.len());
        let mut poly_area_px = 0.0_f64;
        for poly in &pm.polygons {
            let shifts = poly
                .first()
                .map(|outer| geo_ring_lon_shifts(outer, ifd, is_geo_dataset))
                .unwrap_or_else(|| vec![0.0]);
            for (si, shift) in shifts.into_iter().enumerate() {
                let mut rings_px = Vec::with_capacity(poly.len());
                for ring in poly {
                    let rp = densify_ring_lonlat_to_pixel(
                        ring,
                        crs,
                        &c2p,
                        ifd.gcp_poly.as_deref(),
                        is_geo_dataset,
                        densify_lonlat,
                        shift,
                        total_verts,
                    );
                    for &(c, r) in &rp {
                        if c.is_finite() && r.is_finite() {
                            min_c = min_c.min(c);
                            max_c = max_c.max(c);
                            min_r = min_r.min(r);
                            max_r = max_r.max(r);
                        }
                    }
                    rings_px.push(rp);
                }
                if si == 0 {
                    poly_area_px += polygon_total_area_px(std::slice::from_ref(&rings_px));
                }
                polys_px.push(rings_px);
            }
        }

        let mut lines_px = Vec::with_capacity(pm.lines.len());
        for line in &pm.lines {
            for shift in geo_ring_lon_shifts(line, ifd, is_geo_dataset) {
                let lp = densify_ring_lonlat_to_pixel(
                    line,
                    crs,
                    &c2p,
                    ifd.gcp_poly.as_deref(),
                    is_geo_dataset,
                    densify_lonlat,
                    shift,
                    total_verts,
                );
                for &(c, r) in &lp {
                    if c.is_finite() && r.is_finite() {
                        min_c = min_c.min(c);
                        max_c = max_c.max(c + 1.0);
                        min_r = min_r.min(r);
                        max_r = max_r.max(r + 1.0);
                    }
                }
                lines_px.push(lp);
            }
        }

        let mut pts_px = Vec::with_capacity(pm.points.len());
        for &(lon, lat) in &pm.points {
            let adj_lon = geo_point_lon_shift(lon, ifd, is_geo_dataset);
            let (sx, sy) = if is_geo_dataset {
                (adj_lon, lat)
            } else {
                crs.from_lonlat_deg(adj_lon, lat)
            };
            let (c, r) = ifd.crs_to_pixel_pt(sx, sy);
            if c.is_finite() && r.is_finite() {
                min_c = min_c.min(c);
                max_c = max_c.max(c + 1.0);
                min_r = min_r.min(r);
                max_r = max_r.max(r + 1.0);
                pts_px.push((c, r));
            }
        }

        if !min_c.is_finite() || !min_r.is_finite() {
            return Ok((polys_px, poly_area_px, lines_px, pts_px, 0, 0, 0, 0));
        }
        let c0 = min_c.floor().clamp(0.0, ifd.width as f64) as usize;
        let c1 = max_c.ceil().clamp(c0 as f64, ifd.width as f64) as usize;
        let r0 = min_r.floor().clamp(0.0, ifd.height as f64) as usize;
        let r1 = max_r.ceil().clamp(r0 as f64, ifd.height as f64) as usize;
        Ok((polys_px, poly_area_px, lines_px, pts_px, c0, c1, r0, r1))
    };

    // Start on full-resolution IFD 0; only step down to an overview if explicit max_size > 0 requests it.
    let mut ifd_idx = 0usize;
    let (
        mut polys_px,
        mut total_poly_area_px_val,
        mut lines_px,
        mut pts_px,
        mut c0,
        mut c1,
        mut r0,
        mut r1,
    ) = project_to_ifd(full)?;
    if c1 <= c0 || r1 <= r0 {
        return Ok(empty_result());
    }
    if opts.max_size_explicit && max_size > 0 && (c1 - c0).max(r1 - r0) > max_size as usize {
        for (i, ov) in cog.index.ifds.iter().enumerate().skip(1) {
            let (pp, pa, lp, ptp, oc0, oc1, or0, or1) = project_to_ifd(ov)?;
            ifd_idx = i;
            polys_px = pp;
            total_poly_area_px_val = pa;
            lines_px = lp;
            pts_px = ptp;
            c0 = oc0;
            c1 = oc1;
            r0 = or0;
            r1 = or1;
            if (c1 - c0).max(r1 - r0) <= max_size as usize {
                break;
            }
        }
    }
    if c1 <= c0 || r1 <= r0 {
        return Ok(empty_result());
    }

    let ifd = &cog.index.ifds[ifd_idx];
    let win_w = c1 - c0;
    let win_h = r1 - r0;
    let n_px = win_w.saturating_mul(win_h);
    const MAX_EXACT_WINDOW_PIXELS: usize = 16_777_216; // 4096 x 4096
    if n_px > MAX_EXACT_WINDOW_PIXELS {
        return Err(EngineError::BadRequest(format!(
            "polygon window ({win_w}x{win_h} = {n_px} px) exceeds maximum exact statistics window ({MAX_EXACT_WINDOW_PIXELS} px); pass max_size to downsample"
        )));
    }

    let tw = ifd.tile_width as usize;
    let th = ifd.tile_height as usize;
    let tx0 = c0 / tw;
    let tx1 = (c1 - 1) / tw;
    let ty0 = r0 / th;
    let ty1 = (r1 - 1) / th;
    let blocks_x = tx1 - tx0 + 1;
    let blocks_y = ty1 - ty0 + 1;
    const MAX_EXACT_BLOCKS: usize = 1024;
    let total_blocks = match ifd.planar_config {
        PlanarConfig::Chunky => blocks_x.saturating_mul(blocks_y),
        PlanarConfig::Planar => sel_bands
            .len()
            .saturating_mul(blocks_x.saturating_mul(blocks_y)),
    };
    if total_blocks > MAX_EXACT_BLOCKS {
        return Err(EngineError::BadRequest(format!(
            "polygon window requires {total_blocks} blocks (max {MAX_EXACT_BLOCKS}); pass max_size to downsample",
        )));
    }

    // Validate requested bands against the selected IFD's actual band count (guards against
    // out-of-range expression bands like `b99` or heterogeneous overview band counts).
    for &b1 in &sel_bands {
        if b1 == 0 || b1 > ifd.samples_per_pixel {
            return Err(EngineError::BadRequest(format!(
                "band index b{b1} out of range for {}-band dataset",
                ifd.samples_per_pixel
            )));
        }
    }

    let stream_chunky =
        matches!(ifd.planar_config, PlanarConfig::Chunky) && opts.expressions.is_none();
    const MAX_EXACT_TOTAL_SAMPLES: usize = 67_108_864; // 256 MiB across retained bands
    if !stream_chunky {
        let total_samples = n_px.saturating_mul(sel_bands.len());
        if total_samples > MAX_EXACT_TOTAL_SAMPLES {
            return Err(EngineError::BadRequest(format!(
                "polygon window requires {total_samples} band-samples across {} bands (max {MAX_EXACT_TOTAL_SAMPLES}); pass bidx or max_size to narrow",
                sel_bands.len()
            )));
        }
    }

    // Fetch intersecting data blocks in parallel (get_blocks already applies internal mask IFDs).
    let eff_nodata = nodata_override.or(cog.index.nodata);

    let mut band_bufs: Vec<(u16, Vec<f32>)> = if stream_chunky {
        Vec::new()
    } else {
        Vec::with_capacity(sel_bands.len())
    };
    let mut streamed_chunky_tiles = None;
    match ifd.planar_config {
        PlanarConfig::Chunky => {
            let mut indices = Vec::with_capacity(blocks_x * blocks_y);
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    indices.push(Some(ifd.tile_index(tx as u32, ty as u32, 0) as u32));
                }
            }
            let tiles = engine.store.get_blocks(cog, ifd_idx, &indices).await?;
            if stream_chunky {
                streamed_chunky_tiles = Some(tiles);
            } else {
                for &b1 in &sel_bands {
                    let b0 = (b1 - 1) as usize;
                    let sc = if unscale {
                        cog.index.scales.get(b0).copied().unwrap_or(1.0)
                    } else {
                        1.0
                    };
                    let off = if unscale {
                        cog.index.offsets.get(b0).copied().unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    let mut pixels = vec![f32::NAN; n_px];
                    for (bi, tile_opt) in tiles.iter().enumerate() {
                        let Some(tile) = tile_opt else { continue };
                        let Some(band) = tile.bands.get(b0) else { continue };
                        let ty = ty0 + bi / blocks_x;
                        let tx = tx0 + bi % blocks_x;
                        let tile_x0 = tx * tw;
                        let tile_y0 = ty * th;
                        let rx0 = c0.max(tile_x0);
                        let rx1 = c1.min(tile_x0 + tw);
                        let ry0 = r0.max(tile_y0);
                        let ry1 = r1.min(tile_y0 + th);
                        for row in ry0..ry1 {
                            let ly = row - tile_y0;
                            let dst_row = (row - r0) * win_w;
                            for col in rx0..rx1 {
                                let lx = col - tile_x0;
                                let src_i = ly * tw + lx;
                                let mut v = band.data.get_f32(src_i);
                                if !v.is_finite() {
                                    continue;
                                }
                                if let Some(nd) = eff_nodata
                                    && (nd.is_nan() && v.is_nan() || v == nd)
                                {
                                    continue;
                                }
                                if unscale {
                                    v = v * sc + off;
                                }
                                pixels[dst_row + (col - c0)] = v;
                            }
                        }
                    }
                    band_bufs.push((b1, pixels));
                }
            }
        }
        PlanarConfig::Planar => {
            for &b1 in &sel_bands {
                let b0 = (b1 - 1) as usize;
                let mut indices = Vec::with_capacity(blocks_x * blocks_y);
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        indices.push(Some(ifd.tile_index(tx as u32, ty as u32, b0 as u16) as u32));
                    }
                }
                let tiles = engine.store.get_blocks(cog, ifd_idx, &indices).await?;
                let sc = if unscale {
                    cog.index.scales.get(b0).copied().unwrap_or(1.0)
                } else {
                    1.0
                };
                let off = if unscale {
                    cog.index.offsets.get(b0).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                let mut pixels = vec![f32::NAN; n_px];
                for (bi, tile_opt) in tiles.iter().enumerate() {
                    let Some(tile) = tile_opt else { continue };
                    let Some(band) = tile.bands.first() else { continue };
                    let ty = ty0 + bi / blocks_x;
                    let tx = tx0 + bi % blocks_x;
                    let tile_x0 = tx * tw;
                    let tile_y0 = ty * th;
                    let rx0 = c0.max(tile_x0);
                    let rx1 = c1.min(tile_x0 + tw);
                    let ry0 = r0.max(tile_y0);
                    let ry1 = r1.min(tile_y0 + th);
                    for row in ry0..ry1 {
                        let ly = row - tile_y0;
                        let dst_row = (row - r0) * win_w;
                        for col in rx0..rx1 {
                            let lx = col - tile_x0;
                            let src_i = ly * tw + lx;
                            let mut v = band.data.get_f32(src_i);
                            if !v.is_finite() {
                                continue;
                            }
                            if let Some(nd) = eff_nodata
                                && (nd.is_nan() && v.is_nan() || v == nd)
                            {
                                continue;
                            }
                            if unscale {
                                v = v * sc + off;
                            }
                            pixels[dst_row + (col - c0)] = v;
                        }
                    }
                }
                band_bufs.push((b1, pixels));
            }
        }
    }

    // Compute exact coverage, ellipsoidal area model (or geodesic line lengths), and weighted statistics.
    let crs_owned = crs.clone();
    let p2c = ifd.pixel_to_crs;
    let opts_owned = opts.clone();
    let descs_owned = cog.index.band_descriptions.clone();
    let scales_owned = if stream_chunky && unscale {
        cog.index.scales.clone()
    } else {
        Vec::new()
    };
    let offsets_owned = if stream_chunky && unscale {
        cog.index.offsets.clone()
    } else {
        Vec::new()
    };
    let geom_kind = if !polys_px.is_empty() {
        ZonalGeomKind::Polygon
    } else if !lines_px.is_empty() {
        ZonalGeomKind::Line
    } else {
        ZonalGeomKind::Point
    };
    let compute_cpu = move || -> serde_json::Value {
        let total_poly_area_px =
            (geom_kind == ZonalGeomKind::Polygon).then_some(total_poly_area_px_val);
        let cov = rasterize_exact_coverage(&polys_px, &lines_px, &pts_px, c0, r0, win_w, win_h);
        let line_len_m = if geom_kind == ZonalGeomKind::Line {
            Some(rasterize_line_geodesic_lengths_m(
                &lines_px, &crs_owned, &p2c, c0, r0, win_w, win_h,
            ))
        } else {
            None
        };
        let area_model = if geom_kind == ZonalGeomKind::Polygon {
            PixelAreaModel::build(&crs_owned, &p2c, c0, r0, win_w, win_h)
        } else {
            PixelAreaModel::Constant(0.0)
        };
        let mut out = serde_json::Map::new();
        if let Some(ref exprs) = opts_owned.expressions {
            for (ei, (desc, prog)) in exprs.iter().enumerate() {
                let mut expr_px = vec![f32::NAN; n_px];
                let slot_bufs: Vec<Option<&[f32]>> = prog
                    .used_bands
                    .iter()
                    .map(|&b1| {
                        band_bufs
                            .iter()
                            .find(|(k, _)| *k == b1)
                            .map(|(_, buf)| buf.as_slice())
                    })
                    .collect();
                let allow_nan = prog.ops.contains(&cogzilla_kernel::expr_vm::Op::Where);
                let mut b_vals = vec![0.0_f32; slot_bufs.len()];
                for i in 0..n_px {
                    if cov[i] <= COV_EPS {
                        continue;
                    }
                    let mut all_valid = true;
                    for (slot, buf_opt) in slot_bufs.iter().enumerate() {
                        let v = buf_opt.map_or(f32::NAN, |buf| buf[i]);
                        if !v.is_finite() {
                            if allow_nan {
                                b_vals[slot] = f32::NAN;
                                continue;
                            }
                            all_valid = false;
                            break;
                        }
                        b_vals[slot] = v;
                    }
                    if all_valid {
                        let r = prog.eval(&b_vals);
                        if r.is_finite() {
                            expr_px[i] = r;
                        }
                    }
                }
                out.insert(
                    format!("b{}", ei + 1),
                    compute_weighted_band_stats_with_geom(
                        &expr_px,
                        &cov,
                        win_w,
                        win_h,
                        &area_model,
                        opts_owned.area_weighted,
                        geom_kind,
                        line_len_m.as_deref(),
                        total_poly_area_px,
                        desc,
                        &opts_owned,
                    ),
                );
            }
        } else if let Some(ref tiles) = streamed_chunky_tiles {
            let mut pixels = vec![f32::NAN; n_px];
            for (bi_sel, &b1) in sel_bands.iter().enumerate() {
                if bi_sel > 0 {
                    pixels.fill(f32::NAN);
                }
                let b0 = (b1 - 1) as usize;
                let sc = if unscale {
                    scales_owned.get(b0).copied().unwrap_or(1.0)
                } else {
                    1.0
                };
                let off = if unscale {
                    offsets_owned.get(b0).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                for (bi, tile_opt) in tiles.iter().enumerate() {
                    let Some(tile) = tile_opt else { continue };
                    let Some(band) = tile.bands.get(b0) else { continue };
                    let ty = ty0 + bi / blocks_x;
                    let tx = tx0 + bi % blocks_x;
                    let tile_x0 = tx * tw;
                    let tile_y0 = ty * th;
                    let rx0 = c0.max(tile_x0);
                    let rx1 = c1.min(tile_x0 + tw);
                    let ry0 = r0.max(tile_y0);
                    let ry1 = r1.min(tile_y0 + th);
                    for row in ry0..ry1 {
                        let ly = row - tile_y0;
                        let dst_row = (row - r0) * win_w;
                        for col in rx0..rx1 {
                            let lx = col - tile_x0;
                            let src_i = ly * tw + lx;
                            let mut v = band.data.get_f32(src_i);
                            if !v.is_finite() {
                                continue;
                            }
                            if let Some(nd) = eff_nodata
                                && (nd.is_nan() && v.is_nan() || v == nd)
                            {
                                continue;
                            }
                            if unscale {
                                v = v * sc + off;
                            }
                            pixels[dst_row + (col - c0)] = v;
                        }
                    }
                }
                let desc = descs_owned
                    .get(b0)
                    .map(String::as_str)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .unwrap_or_else(|| format!("b{b1}"));
                out.insert(
                    format!("b{b1}"),
                    compute_weighted_band_stats_with_geom(
                        &pixels,
                        &cov,
                        win_w,
                        win_h,
                        &area_model,
                        opts_owned.area_weighted,
                        geom_kind,
                        line_len_m.as_deref(),
                        total_poly_area_px,
                        &desc,
                        &opts_owned,
                    ),
                );
            }
        } else {
            for (b1, pixels) in &band_bufs {
                let desc = descs_owned
                    .get((*b1 - 1) as usize)
                    .map(String::as_str)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .unwrap_or_else(|| format!("b{b1}"));
                out.insert(
                    format!("b{b1}"),
                    compute_weighted_band_stats_with_geom(
                        pixels,
                        &cov,
                        win_w,
                        win_h,
                        &area_model,
                        opts_owned.area_weighted,
                        geom_kind,
                        line_len_m.as_deref(),
                        total_poly_area_px,
                        &desc,
                        &opts_owned,
                    ),
                );
            }
        }
        serde_json::Value::Object(out)
    };

    if n_px >= 65_536 {
        Ok(engine.store.run_heavy_cpu(compute_cpu).await?)
    } else {
        Ok(compute_cpu())
    }
}

/// Compute exact analytical coverage-weighted & ellipsoidal area-weighted expression statistics
/// across 1 or more same-grid STAC assets (`cogs[0]` = `b1`, `cogs[1]` = `b2`, ...) on their
/// native pixel grid. Returns `Ok(None)` if `cogs` have heterogeneous grids (so the caller can
/// fall back to reprojected bbox rendering).
#[allow(clippy::too_many_arguments)]
pub async fn compute_native_exact_multi_cog_expr_stats(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    pm: &PolygonMask,
    max_size: u32,
    unscale: bool,
    nodata_override: Option<f32>,
    exprs: &[(String, cogzilla_kernel::ExprProgram)],
    opts: &StatsOptions,
) -> Result<Option<serde_json::Value>, EngineError> {
    if cogs.is_empty() {
        return Err(EngineError::BadRequest("no assets provided".into()));
    }
    let primary = &cogs[0];
    let crs = engine.get_or_build_crs(primary)?;
    if cogs.len() == 1 {
        let mut opts_e = opts.clone();
        opts_e.expressions = Some(exprs.to_vec());
        return Ok(Some(
            compute_native_exact_polygon_stats(
                engine,
                primary,
                &crs,
                pm,
                max_size,
                unscale,
                nodata_override,
                &opts_e,
            )
            .await?,
        ));
    }

    let is_geo = matches!(crs, CrsModel::Geographic);
    let densify_lonlat = !is_geo && !pm.coord_crs_projected;
    let total_verts = pm.vertex_count();

    let project_to_ifd = |ifd: &cogzilla_tiff::IfdMeta| {
        let mut polys_px: Vec<Vec<Vec<(f64, f64)>>> = Vec::with_capacity(pm.polygons.len());
        let mut poly_area_px = 0.0_f64;
        for poly in &pm.polygons {
            let shifts = poly
                .first()
                .map(|outer| geo_ring_lon_shifts(outer, ifd, is_geo))
                .unwrap_or_else(|| vec![0.0]);
            for (si, shift) in shifts.into_iter().enumerate() {
                let mut rings_px = Vec::with_capacity(poly.len());
                for ring in poly {
                    rings_px.push(densify_ring_lonlat_to_pixel(
                        ring,
                        &crs,
                        &ifd.crs_to_pixel,
                        ifd.gcp_poly.as_deref(),
                        is_geo,
                        densify_lonlat,
                        shift,
                        total_verts,
                    ));
                }
                if si == 0 {
                    poly_area_px += polygon_total_area_px(std::slice::from_ref(&rings_px));
                }
                polys_px.push(rings_px);
            }
        }
        let mut lines_px: Vec<Vec<(f64, f64)>> = Vec::with_capacity(pm.lines.len());
        for line in &pm.lines {
            for shift in geo_ring_lon_shifts(line, ifd, is_geo) {
                lines_px.push(densify_ring_lonlat_to_pixel(
                    line,
                    &crs,
                    &ifd.crs_to_pixel,
                    ifd.gcp_poly.as_deref(),
                    is_geo,
                    densify_lonlat,
                    shift,
                    total_verts,
                ));
            }
        }
        let mut pts_px: Vec<(f64, f64)> = Vec::with_capacity(pm.points.len());
        for &(lo, la) in &pm.points {
            let adj_lo = geo_point_lon_shift(lo, ifd, is_geo);
            let (sx, sy) = if is_geo { (adj_lo, la) } else { crs.from_lonlat_deg(adj_lo, la) };
            let (c, r) = ifd.crs_to_pixel_pt(sx, sy);
            pts_px.push((c, r));
        }

        let mut min_c = f64::INFINITY;
        let mut max_c = f64::NEG_INFINITY;
        let mut min_r = f64::INFINITY;
        let mut max_r = f64::NEG_INFINITY;
        for poly in &polys_px {
            if let Some(outer) = poly.first() {
                for &(c, r) in outer {
                    if c.is_finite() && r.is_finite() {
                        min_c = min_c.min(c);
                        max_c = max_c.max(c);
                        min_r = min_r.min(r);
                        max_r = max_r.max(r);
                    }
                }
            }
        }
        for line in &lines_px {
            for &(c, r) in line {
                if c.is_finite() && r.is_finite() {
                    min_c = min_c.min(c);
                    max_c = max_c.max(c + 1.0);
                    min_r = min_r.min(r);
                    max_r = max_r.max(r + 1.0);
                }
            }
        }
        for &(c, r) in &pts_px {
            if c.is_finite() && r.is_finite() {
                min_c = min_c.min(c);
                max_c = max_c.max(c + 1.0);
                min_r = min_r.min(r);
                max_r = max_r.max(r + 1.0);
            }
        }
        if !min_c.is_finite() || !min_r.is_finite() {
            return (polys_px, poly_area_px, lines_px, pts_px, 0usize, 0usize, 0usize, 0usize);
        }
        let c0 = min_c.floor().clamp(0.0, ifd.width as f64) as usize;
        let c1 = max_c.ceil().clamp(c0 as f64, ifd.width as f64) as usize;
        let r0 = min_r.floor().clamp(0.0, ifd.height as f64) as usize;
        let r1 = max_r.ceil().clamp(r0 as f64, ifd.height as f64) as usize;
        (polys_px, poly_area_px, lines_px, pts_px, c0, c1, r0, r1)
    };

    let geom_kind = if !pm.polygons.is_empty() {
        ZonalGeomKind::Polygon
    } else if !pm.lines.is_empty() {
        ZonalGeomKind::Line
    } else {
        ZonalGeomKind::Point
    };

    let empty_expr_result = || -> serde_json::Value {
        let mut out = serde_json::Map::new();
        for (ei, (desc, _)) in exprs.iter().enumerate() {
            let mut empty = compute_f32_band_stats(&[], desc, opts);
            if let Some(o) = empty.as_object_mut() {
                match geom_kind {
                    ZonalGeomKind::Polygon => {
                        o.insert("area".into(), serde_json::json!(0.0));
                    }
                    ZonalGeomKind::Line => {
                        o.insert("area".into(), serde_json::Value::Null);
                        o.insert("length".into(), serde_json::json!(0.0));
                    }
                    ZonalGeomKind::Point => {
                        o.insert("area".into(), serde_json::Value::Null);
                    }
                }
            }
            out.insert(format!("b{}", ei + 1), empty);
        }
        serde_json::Value::Object(out)
    };

    // Start on full-resolution IFD 0; only step down to an overview if explicit max_size > 0
    // requests it (matching `compute_native_exact_polygon_stats`).
    let mut ifd_idx = 0usize;
    if crate::engine::validate_multi_cog_grid(cogs, 0).is_err() {
        return Ok(None);
    }
    let (
        mut polys_px,
        mut total_poly_area_px_val,
        mut lines_px,
        mut pts_px,
        mut c0,
        mut c1,
        mut r0,
        mut r1,
    ) = project_to_ifd(&primary.index.ifds[0]);
    if c1 <= c0 || r1 <= r0 {
        return Ok(Some(empty_expr_result()));
    }
    if opts.max_size_explicit && max_size > 0 && (c1 - c0).max(r1 - r0) > max_size as usize {
        for (i, ov) in primary.index.ifds.iter().enumerate().skip(1) {
            if crate::engine::validate_multi_cog_grid(cogs, i).is_err() {
                break;
            }
            let (pp, pa, lp, ptp, oc0, oc1, or0, or1) = project_to_ifd(ov);
            ifd_idx = i;
            polys_px = pp;
            total_poly_area_px_val = pa;
            lines_px = lp;
            pts_px = ptp;
            c0 = oc0;
            c1 = oc1;
            r0 = or0;
            r1 = or1;
            if (c1 - c0).max(r1 - r0) <= max_size as usize {
                break;
            }
        }
    }
    if c1 <= c0 || r1 <= r0 {
        return Ok(Some(empty_expr_result()));
    }
    let ifd = &primary.index.ifds[ifd_idx];
    let win_w = c1 - c0;
    let win_h = r1 - r0;
    let n_px = win_w.saturating_mul(win_h);
    const MAX_EXACT_WINDOW_PX: usize = 4096 * 4096;
    if n_px > MAX_EXACT_WINDOW_PX {
        return Err(EngineError::BadRequest(format!(
            "polygon window {win_w}x{win_h} ({n_px} px) exceeds max {MAX_EXACT_WINDOW_PX} px; pass a smaller max_size"
        )));
    }

    let mut ref_slots: Vec<u16> = Vec::new();
    for (_, prog) in exprs {
        for &b1 in &prog.used_bands {
            if !ref_slots.contains(&b1) {
                ref_slots.push(b1);
            }
        }
    }

    let tw = ifd.tile_width as usize;
    let th = ifd.tile_height as usize;
    let tx0 = c0 / tw;
    let tx1 = (c1 - 1) / tw;
    let ty0 = r0 / th;
    let ty1 = (r1 - 1) / th;
    let blocks_x = tx1 - tx0 + 1;
    let blocks_y = ty1 - ty0 + 1;
    const MAX_EXACT_BLOCKS: usize = 1024;
    let total_blocks = ref_slots
        .len()
        .max(1)
        .saturating_mul(blocks_x.saturating_mul(blocks_y));
    if total_blocks > MAX_EXACT_BLOCKS {
        return Err(EngineError::BadRequest(format!(
            "polygon window requires {total_blocks} blocks (max {MAX_EXACT_BLOCKS}); pass max_size to downsample",
        )));
    }
    const MAX_EXACT_TOTAL_SAMPLES: usize = 67_108_864; // 256 MiB across retained bands
    let total_samples = n_px.saturating_mul(ref_slots.len());
    if total_samples > MAX_EXACT_TOTAL_SAMPLES {
        return Err(EngineError::BadRequest(format!(
            "polygon window requires {total_samples} band-samples across {} bands (max {MAX_EXACT_TOTAL_SAMPLES}); pass max_size to narrow",
            ref_slots.len()
        )));
    }

    let mut indices = Vec::with_capacity(blocks_x * blocks_y);
    for ty in ty0..=ty1 {
        for tx in tx0..=tx1 {
            indices.push(Some(ifd.tile_index(tx as u32, ty as u32, 0) as u32));
        }
    }

    let mut band_bufs: Vec<(u16, Vec<f32>)> = Vec::with_capacity(ref_slots.len());
    for &b1 in &ref_slots {
        let slot = (b1.saturating_sub(1) as usize).min(cogs.len().saturating_sub(1));
        let asset_cog = &cogs[slot];
        let asset_ifd = &asset_cog.index.ifds[ifd_idx];
        let asset_tw = asset_ifd.tile_width as usize;
        let asset_th = asset_ifd.tile_height as usize;
        let eff_nodata = nodata_override.or_else(|| asset_cog.index.nodata.map(|v| v as f32));
        let sc = if unscale {
            asset_cog.index.scales.first().copied().unwrap_or(1.0)
        } else {
            1.0
        };
        let off = if unscale {
            asset_cog.index.offsets.first().copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let tiles = engine.store.get_blocks(asset_cog, ifd_idx, &indices).await?;
        let mut pixels = vec![f32::NAN; n_px];
        for (bi, tile_opt) in tiles.iter().enumerate() {
            let Some(tile) = tile_opt else { continue };
            let Some(band) = tile.bands.first() else { continue };
            let ty = ty0 + bi / blocks_x;
            let tx = tx0 + bi % blocks_x;
            let tile_x0 = tx * asset_tw;
            let tile_y0 = ty * asset_th;
            let rx0 = c0.max(tile_x0);
            let rx1 = c1.min(tile_x0 + asset_tw);
            let ry0 = r0.max(tile_y0);
            let ry1 = r1.min(tile_y0 + asset_th);
            for row in ry0..ry1 {
                let ly = row - tile_y0;
                let dst_row = (row - r0) * win_w;
                for col in rx0..rx1 {
                    let lx = col - tile_x0;
                    let mut v = band.data.get_f32(ly * asset_tw + lx);
                    if !v.is_finite() {
                        continue;
                    }
                    if let Some(nd) = eff_nodata
                        && (nd.is_nan() && v.is_nan() || v == nd)
                    {
                        continue;
                    }
                    if unscale {
                        v = v * sc + off;
                    }
                    pixels[dst_row + (col - c0)] = v;
                }
            }
        }
        band_bufs.push((b1, pixels));
    }

    let crs_owned = crs.clone();
    let p2c = ifd.pixel_to_crs;
    let opts_owned = opts.clone();
    let exprs_owned = exprs.to_vec();
    let compute_cpu = move || -> serde_json::Value {
        let total_poly_area_px =
            (geom_kind == ZonalGeomKind::Polygon).then_some(total_poly_area_px_val);
        let cov = rasterize_exact_coverage(&polys_px, &lines_px, &pts_px, c0, r0, win_w, win_h);
        let line_len_m = if geom_kind == ZonalGeomKind::Line {
            Some(rasterize_line_geodesic_lengths_m(
                &lines_px, &crs_owned, &p2c, c0, r0, win_w, win_h,
            ))
        } else {
            None
        };
        let area_model = if geom_kind == ZonalGeomKind::Polygon {
            PixelAreaModel::build(&crs_owned, &p2c, c0, r0, win_w, win_h)
        } else {
            PixelAreaModel::Constant(0.0)
        };
        let mut out = serde_json::Map::new();
        for (ei, (desc, prog)) in exprs_owned.iter().enumerate() {
            let mut expr_px = vec![f32::NAN; n_px];
            let slot_bufs: Vec<Option<&[f32]>> = prog
                .used_bands
                .iter()
                .map(|&b1| {
                    band_bufs
                        .iter()
                        .find(|(k, _)| *k == b1)
                        .map(|(_, buf)| buf.as_slice())
                })
                .collect();
            let allow_nan = prog.ops.contains(&cogzilla_kernel::expr_vm::Op::Where);
            let mut b_vals = vec![0.0_f32; slot_bufs.len()];
            for i in 0..n_px {
                if cov[i] <= COV_EPS {
                    continue;
                }
                let mut all_valid = true;
                for (slot, buf_opt) in slot_bufs.iter().enumerate() {
                    let v = buf_opt.map_or(f32::NAN, |buf| buf[i]);
                    if !v.is_finite() {
                        if allow_nan {
                            b_vals[slot] = f32::NAN;
                            continue;
                        }
                        all_valid = false;
                        break;
                    }
                    b_vals[slot] = v;
                }
                if all_valid {
                    let r = prog.eval(&b_vals);
                    if r.is_finite() {
                        expr_px[i] = r;
                    }
                }
            }
            out.insert(
                format!("b{}", ei + 1),
                compute_weighted_band_stats_with_geom(
                    &expr_px,
                    &cov,
                    win_w,
                    win_h,
                    &area_model,
                    opts_owned.area_weighted,
                    geom_kind,
                    line_len_m.as_deref(),
                    total_poly_area_px,
                    desc,
                    &opts_owned,
                ),
            );
        }
        serde_json::Value::Object(out)
    };

    let val = if n_px >= 65_536 {
        engine.store.run_heavy_cpu(compute_cpu).await?
    } else {
        compute_cpu()
    };
    Ok(Some(val))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum_cov(cov: &[f64]) -> f64 {
        cov.iter().sum()
    }

    #[test]
    fn exact_coverage_matches_analytic_area_to_1e_6_on_all_shapes() {
        // 1. 3x3 grid-aligned (true = 9.0)
        let poly = vec![vec![vec![(2.0, 2.0), (5.0, 2.0), (5.0, 5.0), (2.0, 5.0), (2.0, 2.0)]]];
        let cov = rasterize_exact_coverage(&poly, &[], &[], 0, 0, 10, 10);
        assert!((sum_cov(&cov) - 9.0).abs() < 1e-12);

        // 2. 3x3 shifted half a pixel (true = 9.0)
        let poly = vec![vec![vec![(2.5, 2.5), (5.5, 2.5), (5.5, 5.5), (2.5, 5.5), (2.5, 2.5)]]];
        let cov = rasterize_exact_coverage(&poly, &[], &[], 0, 0, 10, 10);
        assert!((sum_cov(&cov) - 9.0).abs() < 1e-12);

        // 3. 3x3 grown 0.1 px per side -> 3.2 x 3.2 (true = 10.24)
        let poly = vec![vec![vec![(1.9, 1.9), (5.1, 1.9), (5.1, 5.1), (1.9, 5.1), (1.9, 1.9)]]];
        let cov = rasterize_exact_coverage(&poly, &[], &[], 0, 0, 10, 10);
        assert!((sum_cov(&cov) - 10.24).abs() < 1e-12);

        // 4. Sliver 0.2 px wide by 100 long (true = 20.0)
        let poly = vec![vec![vec![(5.3, 1.0), (5.5, 1.0), (5.5, 101.0), (5.3, 101.0), (5.3, 1.0)]]];
        let cov = rasterize_exact_coverage(&poly, &[], &[], 0, 0, 10, 105);
        assert!((sum_cov(&cov) - 20.0).abs() < 1e-12);

        // 5. Right triangle 100x100 (true = 5000.0)
        let poly = vec![vec![vec![(1.3, 2.7), (101.3, 2.7), (101.3, 102.7), (1.3, 2.7)]]];
        let cov = rasterize_exact_coverage(&poly, &[], &[], 0, 0, 110, 110);
        assert!((sum_cov(&cov) - 5000.0).abs() < 1e-9);

        // 6. 50x50 square (true = 2500.0)
        let poly = vec![vec![vec![(3.7, 4.2), (53.7, 4.2), (53.7, 54.2), (3.7, 54.2), (3.7, 4.2)]]];
        let cov = rasterize_exact_coverage(&poly, &[], &[], 0, 0, 60, 60);
        assert!((sum_cov(&cov) - 2500.0).abs() < 1e-10);
    }

    #[test]
    fn adjacent_polygons_holes_multipolygon_and_subpixel() {
        // Adjacent polygons split along a diagonal line inside cells -> sum equals union with no double count
        let left = vec![vec![vec![(1.2, 1.3), (4.7, 1.3), (4.7, 6.8), (1.2, 1.3)]]];
        let right = vec![vec![vec![(1.2, 1.3), (4.7, 6.8), (1.2, 6.8), (1.2, 1.3)]]];
        let union = vec![vec![vec![(1.2, 1.3), (4.7, 1.3), (4.7, 6.8), (1.2, 6.8), (1.2, 1.3)]]];
        let cl = rasterize_exact_coverage(&left, &[], &[], 0, 0, 10, 10);
        let cr = rasterize_exact_coverage(&right, &[], &[], 0, 0, 10, 10);
        let cu = rasterize_exact_coverage(&union, &[], &[], 0, 0, 10, 10);
        for i in 0..cu.len() {
            assert!((cl[i] + cr[i] - cu[i]).abs() < 1e-12, "cell {i} double-count mismatch");
        }

        // Polygon with hole: 6x6 outer (36) minus 2x2 hole (4) = 32.0
        let with_hole = vec![vec![
            vec![(1.0, 1.0), (7.0, 1.0), (7.0, 7.0), (1.0, 7.0), (1.0, 1.0)],
            vec![(3.0, 3.0), (5.0, 3.0), (5.0, 5.0), (3.0, 5.0), (3.0, 3.0)],
        ]];
        let ch = rasterize_exact_coverage(&with_hole, &[], &[], 0, 0, 10, 10);
        assert!((sum_cov(&ch) - 32.0).abs() < 1e-12);

        // MultiPolygon: two disjoint 2x2 squares = 8.0
        let mp = vec![
            vec![vec![(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5), (0.5, 0.5)]],
            vec![vec![(5.2, 5.2), (7.2, 5.2), (7.2, 7.2), (5.2, 7.2), (5.2, 5.2)]],
        ];
        let cm = rasterize_exact_coverage(&mp, &[], &[], 0, 0, 10, 10);
        assert!((sum_cov(&cm) - 8.0).abs() < 1e-12);

        // Sub-pixel polygon (0.2 x 0.3 = 0.06) inside pixel (4, 4) returns fractional weight & pixel value
        let tiny = vec![vec![vec![(4.2, 4.3), (4.4, 4.3), (4.4, 4.6), (4.2, 4.6), (4.2, 4.3)]]];
        let ct = rasterize_exact_coverage(&tiny, &[], &[], 4, 4, 1, 1);
        assert!((ct[0] - 0.06).abs() < 1e-12);
        let st = compute_weighted_band_stats(
            &[42.5_f32],
            &ct,
            1,
            1,
            &PixelAreaModel::Constant(900.0),
            true,
            "b1",
            &StatsOptions::default(),
        );
        assert!((st["count"].as_f64().unwrap() - 0.06).abs() < 1e-12);
        assert!((st["mean"].as_f64().unwrap() - 42.5).abs() < 1e-6);
        assert!((st["area"].as_f64().unwrap() - 54.0).abs() < 1e-9);
    }

    #[test]
    fn ellipsoidal_area_and_40n_60n_cos_weighted_mean() {
        // 1. 1° x 1° box at equator (0..1N) and at 60N (59.5..60.5N) on WGS84
        // Exact reference values verified against pyproj.Geod(ellps="WGS84"):
        // 0..1N = 12308.463926 km^2, 59.5..60.5N = 6216.722658 km^2
        let eq_km2 = wgs84_geographic_cell_area_m2(0.0, 1.0, 0.0, 1.0) / 1e6;
        let hi_km2 = wgs84_geographic_cell_area_m2(0.0, 1.0, 59.5, 60.5) / 1e6;
        assert!(
            (eq_km2 - 12_308.4639).abs() < 1e-3,
            "equator 1x1 deg area expected 12308.4639 km2, got {eq_km2}"
        );
        assert!(
            (hi_km2 - 6_216.7227).abs() < 1e-3,
            "60N 1x1 deg area expected 6216.7227 km2, got {hi_km2}"
        );

        // 2. Synthetic EPSG:4326 raster over 40N..60N (0.05° resolution -> 400 rows x 20 cols)
        // whose pixel value equals its center latitude.
        let win_w = 20usize;
        let win_h = 400usize;
        let step = 20.0 / win_h as f64;
        let p2c = Affine2D {
            a: step,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: -step,
            f: 60.0,
        };
        let mut pixels = vec![0.0_f32; win_w * win_h];
        for r in 0..win_h {
            let lat_center = 60.0 - (r as f64 + 0.5) * step;
            for c in 0..win_w {
                pixels[r * win_w + c] = lat_center as f32;
            }
        }
        let cov = vec![1.0_f64; win_w * win_h];
        let area_model = PixelAreaModel::build(&CrsModel::Geographic, &p2c, 0, 0, win_w, win_h);
        let st_weighted = compute_weighted_band_stats(
            &pixels,
            &cov,
            win_w,
            win_h,
            &area_model,
            true,
            "lat",
            &StatsOptions::default(),
        );
        let st_unweighted = compute_weighted_band_stats(
            &pixels,
            &cov,
            win_w,
            win_h,
            &area_model,
            false,
            "lat",
            &StatsOptions::default(),
        );

        // Exact analytic WGS84 ellipsoidal mean latitude over 40..60N:
        // Compare against exact sum(lat_center * band_area) / sum(band_area) (~49.306° vs 50.0° unweighted)
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for r in 0..win_h {
            let lat_top = 60.0 - r as f64 * step;
            let lat_bot = lat_top - step;
            let a = wgs84_geographic_cell_area_m2(0.0, step, lat_bot, lat_top);
            num += (0.5 * (lat_top + lat_bot)) * a;
            den += a;
        }
        let expected_weighted = num / den;
        let got_weighted = st_weighted["mean"].as_f64().unwrap();
        let got_unweighted = st_unweighted["mean"].as_f64().unwrap();
        assert!(
            ((got_weighted - expected_weighted) / expected_weighted).abs() < 1e-6,
            "weighted mean {got_weighted} vs analytic {expected_weighted}"
        );
        assert!(
            (got_unweighted - 50.0).abs() < 1e-5,
            "unweighted mean expected 50.0, got {got_unweighted}"
        );
        assert!(
            (got_weighted - 49.306).abs() < 0.02,
            "40N..60N ellipsoidal mean should be ~49.306°, got {got_weighted}"
        );

        // 3. Small UTM polygon: PixelAreaModel short-circuits to Constant so area_weighted true == false
        let utm = CrsModel::from_epsg(32610).unwrap();
        let p2c_utm = Affine2D {
            a: 30.0,
            b: 0.0,
            c: 500_000.0,
            d: 0.0,
            e: -30.0,
            f: 4_200_000.0,
        };
        let utm_model = PixelAreaModel::build(&utm, &p2c_utm, 0, 0, 10, 10);
        assert!(utm_model.is_constant(), "small UTM window must short-circuit to Constant");
    }

    #[test]
    fn q1_q2_q3_zero_cov_categorical_and_point_line_semantics() {
        // Q1: Grid-aligned 3x3 square surrounded by extreme outlier neighbors (-999, +999)
        // with floating-point edge-touch perturbation (1e-15).
        // valid_pixels must be 9 (not 15) and min/max must reflect only the 9 interior pixels.
        let win = 5usize;
        let mut pixels = vec![999.0f32; win * win];
        pixels[0] = -999.0;
        for r in 1..4 {
            for c in 1..4 {
                pixels[r * win + c] = (r * 10 + c) as f32; // 11..33
            }
        }
        let ring = vec![
            (1.0 - 1e-15, 1.0 - 1e-15),
            (4.0 + 1e-15, 1.0 - 1e-15),
            (4.0 + 1e-15, 4.0 + 1e-15),
            (1.0 - 1e-15, 4.0 + 1e-15),
            (1.0 - 1e-15, 1.0 - 1e-15),
        ];
        let cov = rasterize_exact_coverage(&[vec![ring]], &[], &[], 0, 0, win, win);
        let opts = StatsOptions::default();
        let st = compute_weighted_band_stats(
            &pixels,
            &cov,
            win,
            win,
            &PixelAreaModel::Constant(900.0),
            true,
            "b1",
            &opts,
        );
        assert_eq!(st["valid_pixels"].as_f64().unwrap(), 9.0);
        assert_eq!(st["count"].as_f64().unwrap(), 9.0);
        assert_eq!(st["min"].as_f64().unwrap(), 11.0);
        assert_eq!(st["max"].as_f64().unwrap(), 33.0);

        // Q2: categorical=true returns [[counts], [values]] histogram + categories map
        let cat_px = vec![1.0f32; 1600];
        let cat_cov = vec![1.0f64; 1600];
        let cat_opts = StatsOptions {
            categorical: true,
            ..Default::default()
        };
        let st_cat = compute_weighted_band_stats(
            &cat_px,
            &cat_cov,
            40,
            40,
            &PixelAreaModel::Constant(900.0),
            true,
            "b1",
            &cat_opts,
        );
        assert_eq!(st_cat["histogram"], serde_json::json!([[1600.0], [1.0]]));
        assert_eq!(st_cat["categories"]["1"], serde_json::json!(1600.0));

        let cat_opts2 = StatsOptions {
            categorical: true,
            categories: Some(vec![1.0, 2.0]),
            ..Default::default()
        };
        let st_cat2 = compute_weighted_band_stats(
            &cat_px,
            &cat_cov,
            40,
            40,
            &PixelAreaModel::Constant(900.0),
            true,
            "b1",
            &cat_opts2,
        );
        assert_eq!(st_cat2["histogram"], serde_json::json!([[1600.0, 0.0], [1.0, 2.0]]));

        // Q3: Point -> count=1, area=null; 100-px diagonal LineString -> count=141.421..., area=null, length in m
        let utm = CrsModel::from_epsg(32610).unwrap();
        let p2c = Affine2D {
            a: 30.0,
            b: 0.0,
            c: 500_000.0,
            d: 0.0,
            e: -30.0,
            f: 4_200_000.0,
        };
        let pt_cov = rasterize_exact_coverage(&[], &[], &[(2.5, 2.5)], 0, 0, 5, 5);
        let st_pt = compute_weighted_band_stats_with_geom(
            &vec![42.0f32; 25],
            &pt_cov,
            5,
            5,
            &PixelAreaModel::Constant(0.0),
            true,
            ZonalGeomKind::Point,
            None,
            None,
            "b1",
            &opts,
        );
        assert_eq!(st_pt["count"].as_f64().unwrap(), 1.0);
        assert!(st_pt["area"].is_null(), "Point must have area=null, got {:?}", st_pt["area"]);

        let diag = vec![(0.0, 0.0), (100.0, 100.0)];
        let line_cov = rasterize_exact_coverage(&[], &[diag.clone()], &[], 0, 0, 100, 100);
        let line_len_m = rasterize_line_geodesic_lengths_m(&[diag], &utm, &p2c, 0, 0, 100, 100);
        let st_line = compute_weighted_band_stats_with_geom(
            &vec![5.0f32; 10_000],
            &line_cov,
            100,
            100,
            &PixelAreaModel::Constant(0.0),
            true,
            ZonalGeomKind::Line,
            Some(&line_len_m),
            None,
            "b1",
            &opts,
        );
        let expected_px = 100.0 * std::f64::consts::SQRT_2;
        assert!(
            (st_line["count"].as_f64().unwrap() - expected_px).abs() < 1e-6,
            "100-px diagonal count expected {expected_px}, got {}",
            st_line["count"]
        );
        assert!(st_line["area"].is_null(), "LineString must have area=null");
        let len_m = st_line["length"].as_f64().unwrap();
        let expected_m = expected_px * 30.0;
        assert!(
            (len_m - expected_m).abs() / expected_m < 2e-3,
            "100-px 30m UTM diagonal geodesic length expected ~{expected_m} m, got {len_m} m"
        );
    }

    #[test]
    fn radix_sort_and_unit_fast_path_match_general_weighted_stats() {
        // Verify order-preserving f32 <-> u32 bijection across negative, -0.0, +0.0, subnormals, positive.
        let specials = [-1e20f32, -10.5, -1.0, -1e-40, -0.0, 0.0, 1e-40, 1.0, 10.5, 1e20];
        let mut keys: Vec<u32> = specials.iter().map(|&v| f32_to_sortable_u32(v)).collect();
        for w in keys.windows(2) {
            assert!(w[0] <= w[1], "sortable u32 order broken: {:?}", keys);
        }
        assert_eq!(f32_to_sortable_u32(-0.0), f32_to_sortable_u32(0.0));
        for (&orig, &k) in specials.iter().zip(&keys) {
            let rt = sortable_u32_to_f32(k);
            assert_eq!(rt, orig);
        }

        // Verify radix sort on 2500 elements (> 1024 threshold) for u32 and (u32, f64).
        let n = 2500usize;
        let pixels: Vec<f32> = (0..n)
            .map(|i| (((i * 7919 + 104729) % 2000) as f32 - 1000.0) * 0.013)
            .collect();
        let cov_unit = vec![1.0f64; n];
        let mut cov_frac = vec![1.0f64; n];
        cov_frac[0] = 0.25;
        cov_frac[n - 1] = 0.75;

        let opts = StatsOptions::default();
        // Unit-weight fast path (all c == 1.0, Constant area) vs general variable-area path
        // (same constant value per row via RowArea -> exercises Path 3).
        let st_unit = compute_weighted_band_stats(
            &pixels,
            &cov_unit,
            50,
            50,
            &PixelAreaModel::Constant(900.0),
            true,
            "b1",
            &opts,
        );
        let st_general = compute_weighted_band_stats(
            &pixels,
            &cov_unit,
            50,
            50,
            &PixelAreaModel::PerRow(vec![900.0; 50]),
            true,
            "b1",
            &opts,
        );
        for k in [
            "min",
            "max",
            "mean",
            "count",
            "sum",
            "std",
            "median",
            "majority",
            "minority",
            "unique",
            "percentile_2",
            "percentile_98",
            "area",
        ] {
            let a = st_unit[k].as_f64().unwrap();
            let b = st_general[k].as_f64().unwrap();
            assert!((a - b).abs() <= 1e-6 * a.abs().max(1.0), "mismatch on {k}: {a} vs {b}");
        }
        assert_eq!(st_unit["histogram"], st_general["histogram"]);

        // Fractional-coverage Path 2 vs Path 3
        let st_cov = compute_weighted_band_stats(
            &pixels,
            &cov_frac,
            50,
            50,
            &PixelAreaModel::Constant(900.0),
            true,
            "b1",
            &opts,
        );
        let st_cov_gen = compute_weighted_band_stats(
            &pixels,
            &cov_frac,
            50,
            50,
            &PixelAreaModel::PerRow(vec![900.0; 50]),
            true,
            "b1",
            &opts,
        );
        for k in ["mean", "std", "median", "percentile_2", "percentile_98", "unique"] {
            let a = st_cov[k].as_f64().unwrap();
            let b = st_cov_gen[k].as_f64().unwrap();
            assert!((a - b).abs() <= 1e-6 * a.abs().max(1.0), "cov mismatch on {k}: {a} vs {b}");
        }
        assert_eq!(st_cov["histogram"], st_cov_gen["histogram"]);

        // Verify histogram upper edge pinning never drops max_v due to float rounding (Fix #5)
        let mut opts_h = StatsOptions::default();
        opts_h.histogram_bins = 3;
        opts_h.histogram_range = Some((0.3, 1.0));
        let st_h1 = compute_weighted_band_stats(
            &[0.3f32, 0.5, 1.0, 1.0],
            &[1.0; 4],
            4,
            1,
            &PixelAreaModel::Constant(1.0),
            false,
            "b1",
            &opts_h,
        );
        let h1_sum: f64 = st_h1["histogram"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .sum();
        assert_eq!(h1_sum, 4.0, "max_v=1.0 dropped from histogram: {:?}", st_h1["histogram"]);

        let st_h2 = compute_weighted_band_stats(
            &[-4.440528e-8_f32, 27.136889_f32],
            &[1.0; 2],
            2,
            1,
            &PixelAreaModel::Constant(1.0),
            false,
            "b1",
            &StatsOptions::default(),
        );
        let h2_sum: f64 = st_h2["histogram"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .sum();
        assert_eq!(h2_sum, 2.0, "max_v dropped on default 10 bins: {:?}", st_h2["histogram"]);

        keys.clear();
    }
}
