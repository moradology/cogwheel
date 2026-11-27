//! Precomputed tile coordinate plans (`CoordPlan`) mapping output XYZ/TMS pixels to source IFD space.

use cogzilla_raster::Affine2D;

use crate::crs::CrsModel;
use crate::mercator::{TileCoord, mercator_to_lonlat_deg, mercator_to_lonlat_rad, tile_bounds_3857};

/// Control grid dimension: `17x17` points = `16x16` bilinear cells across any tile.
pub const GRID_DIM: usize = 17;

/// Precomputed coordinate mapping from output XYZ tile pixels -> source IFD pixel-center space
/// `(u = col - 0.5, v = row - 0.5)` so `(0.0, 0.0)` samples 100% of source pixel `(0, 0)`.
#[allow(clippy::large_enum_variant)] // Grid2D keeps 17x17 f32 control grids stack-inline on the tile hot path
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
    /// Build a coordinate plan for a tile, using a nonlinear `GcpPoly` when present or delegating
    /// directly to the zero-overhead `build()` path when `gcp_poly` is `None`.
    pub fn build_with_gcp(
        crs: &CrsModel,
        crs_to_pixel: &Affine2D,
        gcp_poly: Option<&cogzilla_tiff::GcpPoly>,
        ifd_w: u32,
        ifd_h: u32,
        tile: TileCoord,
    ) -> Option<Self> {
        let Some(poly) = gcp_poly else {
            return Self::build(crs, crs_to_pixel, ifd_w, ifd_h, tile);
        };
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
        let step_x = (max_x - min_x) / (N as f64);
        let step_y = (max_y - min_y) / (N as f64);
        let mut u = vec![f32::NAN; N * N].into_boxed_slice();
        let mut v = vec![f32::NAN; N * N].into_boxed_slice();
        let mut min_u = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut any_in = false;
        let fw = ifd_w as f32;
        let fh = ifd_h as f32;
        for py in 0..N {
            let my = max_y - (py as f64 + 0.5) * step_y;
            let row_off = py * N;
            for px in 0..N {
                let mx = min_x + (px as f64 + 0.5) * step_x;
                let (sx, sy) = crs.from_3857(mx, my);
                if sx.is_finite() && sy.is_finite() {
                    let (col, row) = poly.crs_to_pixel(sx, sy);
                    let uu = (col - 0.5) as f32;
                    let vv = (row - 0.5) as f32;
                    u[row_off + px] = uu;
                    v[row_off + px] = vv;
                    if uu >= -1.0 && uu <= fw && vv >= -1.0 && vv <= fh {
                        any_in = true;
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    }
                }
            }
        }
        if !any_in {
            return None;
        }
        let src_min_col = (min_u.floor() as i32).max(0);
        let src_max_col = (max_u.floor() as i32).saturating_add(1).min(ifd_w as i32 - 1);
        let src_min_row = (min_v.floor() as i32).max(0);
        let src_max_row = (max_v.floor() as i32).saturating_add(1).min(ifd_h as i32 - 1);
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
    }

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

        let branch_lon0: Option<f64> = match crs.unshifted() {
            CrsModel::Sinusoidal(p) => Some(p.lon0),
            CrsModel::EqualEarth(p) => Some(p.lon0),
            CrsModel::Albers(p) => Some(p.lon0),
            CrsModel::LambertConformalConic(p) => Some(p.lon0),
            CrsModel::CylindricalEqualArea(p) => Some(p.lon0),
            CrsModel::Equirectangular(p) => Some(p.lon0),
            CrsModel::Mercator(p) => Some(p.lon0),
            _ => None,
        };
        let straddles_branch_cut = branch_lon0.is_some_and(|lon0| {
            let (lon_w, _) = mercator_to_lonlat_rad(min_x + 0.5 * step_x, max_y);
            let (lon_e, _) = mercator_to_lonlat_rad(max_x - 0.5 * step_x, max_y);
            let wrap = |d: f64| {
                (d + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
            };
            wrap(lon_w - lon0) > 0.0 && wrap(lon_e - lon0) < 0.0
        });

        if crs_to_pixel.is_axis_aligned()
            && !straddles_branch_cut
            && matches!(
                crs,
                CrsModel::Epsg3857
                    | CrsModel::WorldMercator
                    | CrsModel::Geographic
                    | CrsModel::CylindricalEqualArea(_)
                    | CrsModel::Equirectangular(_)
                    | CrsModel::Mercator(_)
            )
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
                CrsModel::WorldMercator => {
                    for px in 0..N {
                        let mx = min_x + (px as f64 + 0.5) * step_x;
                        let uu = (crs_to_pixel.a * mx + crs_to_pixel.c - 0.5) as f32;
                        u[px] = uu;
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                    }
                    for py in 0..N {
                        let my = max_y - (py as f64 + 0.5) * step_y;
                        let (_, sy) = crs.from_3857(0.0, my);
                        let vv = (crs_to_pixel.e * sy + crs_to_pixel.f - 0.5) as f32;
                        v[py] = vv;
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    }
                }
                CrsModel::Geographic => {
                    let (wrap_lo, wrap_hi) = if crs_to_pixel.a.abs() > 1e-18 {
                        let x0 = -crs_to_pixel.c / crs_to_pixel.a;
                        let x1 = (ifd_w as f64 - crs_to_pixel.c) / crs_to_pixel.a;
                        let lo = x0.min(x1);
                        let hi = x0.max(x1);
                        let px_w = ((hi - lo) / (ifd_w.max(1) as f64)).max(1e-6);
                        (
                            (lo >= -px_w && hi > 180.0 + 1e-6).then_some(lo),
                            (lo < -180.0 - 1e-6 && hi <= px_w).then_some(hi),
                        )
                    } else {
                        (None, None)
                    };
                    for px in 0..N {
                        let mx = min_x + (px as f64 + 0.5) * step_x;
                        let (mut lon, _) = mercator_to_lonlat_deg(mx, 0.0);
                        if let Some(lo) = wrap_lo
                            && lon < lo
                        {
                            lon += 360.0;
                        } else if let Some(hi) = wrap_hi
                            && lon > hi
                        {
                            lon -= 360.0;
                        }
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
                CrsModel::CylindricalEqualArea(cea) => {
                    for px in 0..N {
                        let mx = min_x + (px as f64 + 0.5) * step_x;
                        let (lon, _) = mercator_to_lonlat_rad(mx, 0.0);
                        let sx = cea.forward_x(lon);
                        let uu = (crs_to_pixel.a * sx + crs_to_pixel.c - 0.5) as f32;
                        u[px] = uu;
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                    }
                    for py in 0..N {
                        let my = max_y - (py as f64 + 0.5) * step_y;
                        let (_, lat) = mercator_to_lonlat_rad(0.0, my);
                        let sy = cea.forward_y(lat);
                        let vv = (crs_to_pixel.e * sy + crs_to_pixel.f - 0.5) as f32;
                        v[py] = vv;
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    }
                }
                CrsModel::Equirectangular(eqc) => {
                    for px in 0..N {
                        let mx = min_x + (px as f64 + 0.5) * step_x;
                        let (lon, _) = mercator_to_lonlat_rad(mx, 0.0);
                        let sx = eqc.forward_x(lon);
                        let uu = (crs_to_pixel.a * sx + crs_to_pixel.c - 0.5) as f32;
                        u[px] = uu;
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                    }
                    for py in 0..N {
                        let my = max_y - (py as f64 + 0.5) * step_y;
                        let (_, lat) = mercator_to_lonlat_rad(0.0, my);
                        let sy = eqc.forward_y(lat);
                        let vv = (crs_to_pixel.e * sy + crs_to_pixel.f - 0.5) as f32;
                        v[py] = vv;
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    }
                }
                CrsModel::Mercator(merc) => {
                    for px in 0..N {
                        let mx = min_x + (px as f64 + 0.5) * step_x;
                        let (lon, _) = mercator_to_lonlat_rad(mx, 0.0);
                        let sx = merc.forward_x(lon);
                        let uu = (crs_to_pixel.a * sx + crs_to_pixel.c - 0.5) as f32;
                        u[px] = uu;
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                    }
                    for py in 0..N {
                        let my = max_y - (py as f64 + 0.5) * step_y;
                        let (_, lat) = mercator_to_lonlat_rad(0.0, my);
                        let sy = merc.forward_y(lat);
                        let vv = (crs_to_pixel.e * sy + crs_to_pixel.f - 0.5) as f32;
                        v[py] = vv;
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    }
                }
                _ => unreachable!(),
            }

            // Exact bilinear pixel footprint `[floor(min), floor(max) + 1]`.
            // Use saturating arithmetic so extreme low-zoom coordinates on high-res COGs
            // (|u| > i32::MAX) never panic in debug or wrap to i32::MIN in release.
            let src_min_col = (min_u.floor() as i32).max(0);
            let src_max_col = (max_u.floor() as i32).saturating_add(1).min(ifd_w as i32 - 1);
            let src_min_row = (min_v.floor() as i32).max(0);
            let src_max_row = (max_v.floor() as i32).saturating_add(1).min(ifd_h as i32 - 1);

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
        } else {
            // For z >= 9 (and not on the eastern antimeridian tile column or an interior
            // `lon0 ± π` projection branch cut), try the fast 17x17 bilinear control grid first.
            // If all 289 control points are finite, return `Grid2D` (or `None` if strictly outside
            // the raster); if any control point hits a projection domain boundary, fall through
            // to adaptive `Exact2D` so boundary 16x16 blocks are never dropped or smeared.
            let near_singularity = match crs.unshifted() {
                CrsModel::Laea(p) if p.pole == 0 => {
                    let (lon_c, lat_c) =
                        mercator_to_lonlat_rad(0.5 * (min_x + max_x), 0.5 * (min_y + max_y));
                    let cos_dist = p.lat0.sin() * lat_c.sin()
                        + p.lat0.cos() * lat_c.cos() * (lon_c - p.lon0).cos();
                    1.0 + cos_dist < 0.02
                }
                CrsModel::ObliqueStereographic(p) => {
                    let (lon_c, lat_c) =
                        mercator_to_lonlat_rad(0.5 * (min_x + max_x), 0.5 * (min_y + max_y));
                    let cos_dist = p.sin_chi0 * lat_c.sin()
                        + p.cos_chi0 * lat_c.cos() * (lon_c - p.lon0).cos();
                    1.0 + cos_dist < 0.05
                }
                CrsModel::SwissObliqueMercator(p) => {
                    let (lon_c, lat_c) =
                        mercator_to_lonlat_rad(0.5 * (min_x + max_x), 0.5 * (min_y + max_y));
                    let sin_br = p.cos_b0 * lat_c.sin()
                        - p.sin_b0 * lat_c.cos() * (lon_c - p.lon0).cos();
                    sin_br.abs() > 0.98
                }
                _ => false,
            };
            let try_grid_2d = tile.z > 8
                && ((tile.x as u64) + 1 < (1u64 << tile.z.min(30)))
                && !straddles_branch_cut
                && !near_singularity;
            if try_grid_2d {
                let mut grid_u = [0.0f32; GRID_DIM * GRID_DIM];
                let mut grid_v = [0.0f32; GRID_DIM * GRID_DIM];
                let mut min_u = f32::INFINITY;
                let mut max_u = f32::NEG_INFINITY;
                let mut min_v = f32::INFINITY;
                let mut max_v = f32::NEG_INFINITY;
                let mut all_finite = true;

                let cell_px = (N as f64) / ((GRID_DIM - 1) as f64);
                for gy in 0..GRID_DIM {
                    let py_f = (gy as f64) * cell_px;
                    let my = max_y - (py_f + 0.5) * step_y;
                    for gx in 0..GRID_DIM {
                        let px_f = (gx as f64) * cell_px;
                        let mx = min_x + (px_f + 0.5) * step_x;
                        let (sx, sy) = crs.from_3857(mx, my);
                        let (uu, vv) = if sx.is_finite() && sy.is_finite() {
                            let (col, row) = crs_to_pixel.forward(sx, sy);
                            ((col - 0.5) as f32, (row - 0.5) as f32)
                        } else {
                            (f32::NAN, f32::NAN)
                        };
                        grid_u[gy * GRID_DIM + gx] = uu;
                        grid_v[gy * GRID_DIM + gx] = vv;
                        if !uu.is_finite() || !vv.is_finite() {
                            all_finite = false;
                        }
                    }
                }

                if all_finite {
                    // Compute exact min/max over the rendered pixel centers `0..=N-1` (NOT the
                    // unrendered `px = N` / `py = N` right/bottom control-grid boundary nodes,
                    // which lie 1 full pixel past the last rendered pixel and can spuriously
                    // cross a 512-px COG block boundary). Because bilinear interpolation on each
                    // grid cell is convex, its extrema over `[0, N-1] x [0, N-1]` occur at the
                    // interior/top/left grid nodes `0..GRID_DIM-1` and the interpolated
                    // `px = N-1` right edge, `py = N-1` bottom edge, and `(N-1, N-1)` corner.
                    let last = GRID_DIM - 1;
                    let t_edge = ((cell_px - 1.0) / cell_px) as f32;
                    let mut track = |uu: f32, vv: f32| {
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    };
                    for gy in 0..last {
                        for gx in 0..last {
                            track(grid_u[gy * GRID_DIM + gx], grid_v[gy * GRID_DIM + gx]);
                        }
                        // Rightmost rendered column `px = N - 1`
                        let i0 = gy * GRID_DIM + (last - 1);
                        let i1 = gy * GRID_DIM + last;
                        track(
                            grid_u[i0] + t_edge * (grid_u[i1] - grid_u[i0]),
                            grid_v[i0] + t_edge * (grid_v[i1] - grid_v[i0]),
                        );
                    }
                    for gx in 0..last {
                        // Bottommost rendered row `py = N - 1`
                        let i0 = (last - 1) * GRID_DIM + gx;
                        let i1 = last * GRID_DIM + gx;
                        track(
                            grid_u[i0] + t_edge * (grid_u[i1] - grid_u[i0]),
                            grid_v[i0] + t_edge * (grid_v[i1] - grid_v[i0]),
                        );
                    }
                    // Bottom-right rendered pixel `(N - 1, N - 1)`
                    let c00 = (last - 1) * GRID_DIM + (last - 1);
                    let c10 = (last - 1) * GRID_DIM + last;
                    let c01 = last * GRID_DIM + (last - 1);
                    let c11 = last * GRID_DIM + last;
                    let u_top = grid_u[c00] + t_edge * (grid_u[c10] - grid_u[c00]);
                    let u_bot = grid_u[c01] + t_edge * (grid_u[c11] - grid_u[c01]);
                    let v_top = grid_v[c00] + t_edge * (grid_v[c10] - grid_v[c00]);
                    let v_bot = grid_v[c01] + t_edge * (grid_v[c11] - grid_v[c01]);
                    track(
                        u_top + t_edge * (u_bot - u_top),
                        v_top + t_edge * (v_bot - v_top),
                    );

                    if !min_u.is_finite()
                        || max_u < -0.5
                        || min_u > (ifd_w as f32 - 0.5)
                        || max_v < -0.5
                        || min_v > (ifd_h as f32 - 0.5)
                    {
                        return None;
                    }

                    let src_min_col = (min_u.floor() as i32).max(0);
                    let src_max_col =
                        (max_u.floor() as i32).saturating_add(1).min(ifd_w as i32 - 1);
                    let src_min_row = (min_v.floor() as i32).max(0);
                    let src_max_row =
                        (max_v.floor() as i32).saturating_add(1).min(ifd_h as i32 - 1);

                    if src_min_col > src_max_col || src_min_row > src_max_row {
                        return None;
                    }

                    let all_in_bounds = min_u >= 0.0
                        && min_v >= 0.0
                        && (max_u.floor() as i32).saturating_add(1) < ifd_w as i32
                        && (max_v.floor() as i32).saturating_add(1) < ifd_h as i32;

                    return Some(Self::Grid2D {
                        grid_u,
                        grid_v,
                        src_min_col,
                        src_max_col,
                        src_min_row,
                        src_max_row,
                        all_in_bounds,
                    });
                }
            }

            // Low-zoom projected tile (z=0..8) or z>=9 tile straddling a projection boundary:
            // adaptive error-bounded quadtree subdivision.
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

                if xm < x1 {
                    let q10_00 = project_px(xm + 1, y0);
                    let q10_01 = project_px(xm + 1, ym);
                    let q10_11 = project_px(x1, ym);
                    fill_cell::<N, F>(
                        xm + 1, y0, x1, ym, q10_00, c10, q10_01, q10_11, fw, fh, project_px, u, v,
                        any_in, min_u, max_u, min_v, max_v,
                    );
                }

                if ym < y1 {
                    let q01_00 = project_px(x0, ym + 1);
                    let q01_10 = project_px(xm, ym + 1);
                    let q01_11 = project_px(xm, y1);
                    fill_cell::<N, F>(
                        x0, ym + 1, xm, y1, q01_00, q01_10, c01, q01_11, fw, fh, project_px, u, v,
                        any_in, min_u, max_u, min_v, max_v,
                    );
                }

                if xm < x1 && ym < y1 {
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
            let src_min_col = (min_u.floor() as i32).max(0);
            let src_max_col = (max_u.floor() as i32).saturating_add(1).min(ifd_w as i32 - 1);
            let src_min_row = (min_v.floor() as i32).max(0);
            let src_max_row = (max_v.floor() as i32).saturating_add(1).min(ifd_h as i32 - 1);
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
