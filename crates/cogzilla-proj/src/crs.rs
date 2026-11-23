use cogzilla_tiff::{Affine2D, GeoKeys};
use thiserror::Error;

use crate::ellipsoid::{
    AlbersEqualArea, Ellipsoid, LambertConformalConic, PolarStereographic, TransverseMercator,
};
use crate::mercator::{
    ORIGIN_SHIFT, TileCoord, lonlat_deg_to_mercator, mercator_to_lonlat_deg,
    mercator_to_lonlat_rad, tile_bounds_3857,
};

#[derive(Debug, Error)]
pub enum ProjError {
    #[error("unsupported or unrecognized CRS in GeoKeys: projected={projected:?}, geographic={geographic:?}, coord_trans={coord_trans:?}")]
    UnsupportedCrs {
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
}

impl CrsModel {
    /// Construct a `CrsModel` from parsed GeoTIFF `GeoKeys`.
    pub fn from_geo_keys(gk: &GeoKeys) -> Result<Self, ProjError> {
        let ellip = match (gk.semi_major_axis, gk.inv_flattening) {
            (Some(a), Some(inv_f)) if a > 0.0 && inv_f > 0.0 => Ellipsoid::new(a, inv_f),
            _ => match gk.geographic_type {
                Some(4269) => Ellipsoid::GRS80,
                _ => Ellipsoid::WGS84,
            },
        };

        // 1. Check well-known ProjectedCSTypeGeoKey EPSG codes first
        if let Some(code) = gk.projected_cs_type {
            match code {
                3857 | 3785 => return Ok(Self::Epsg3857),
                // WGS84 UTM North zones 1..60
                32601..=32660 => {
                    let zone = (code - 32600) as u8;
                    return Ok(Self::TransverseMercator(TransverseMercator::utm(
                        Ellipsoid::WGS84,
                        zone,
                        false,
                    )));
                }
                // WGS84 UTM South zones 1..60
                32701..=32760 => {
                    let zone = (code - 32700) as u8;
                    return Ok(Self::TransverseMercator(TransverseMercator::utm(
                        Ellipsoid::WGS84,
                        zone,
                        true,
                    )));
                }
                // NAD83 UTM North zones 1..23
                26901..=26923 => {
                    let zone = (code - 26900) as u8;
                    return Ok(Self::TransverseMercator(TransverseMercator::utm(
                        Ellipsoid::GRS80,
                        zone,
                        false,
                    )));
                }
                // NAD83 CONUS Albers (EPSG:5070)
                5070 | 5069 => {
                    return Ok(Self::Albers(AlbersEqualArea::new(
                        Ellipsoid::GRS80,
                        29.5,
                        45.5,
                        23.0,
                        -96.0,
                        0.0,
                        0.0,
                    )));
                }
                // California Albers (EPSG:3310)
                3310 => {
                    return Ok(Self::Albers(AlbersEqualArea::new(
                        Ellipsoid::GRS80,
                        34.0,
                        40.5,
                        0.0,
                        -120.0,
                        0.0,
                        -4_000_000.0,
                    )));
                }
                // RGF93 Lambert-93 (EPSG:2154)
                2154 => {
                    return Ok(Self::LambertConformalConic(LambertConformalConic::new_2sp(
                        Ellipsoid::GRS80,
                        44.0,
                        49.0,
                        46.5,
                        3.0,
                        700_000.0,
                        6_600_000.0,
                    )));
                }
                // NSIDC Sea Ice Polar Stereographic North (EPSG:3413)
                3413 => {
                    return Ok(Self::PolarStereographic(PolarStereographic::new(
                        Ellipsoid::WGS84,
                        70.0,
                        -45.0,
                        0.0,
                        0.0,
                    )));
                }
                // Antarctic Polar Stereographic (EPSG:3031)
                3031 => {
                    return Ok(Self::PolarStereographic(PolarStereographic::new(
                        Ellipsoid::WGS84,
                        -71.0,
                        0.0,
                        0.0,
                        0.0,
                    )));
                }
                32767 => {
                    // User-defined projected CRS: fall through to ProjCoordTransGeoKey
                }
                _ => {}
            }
        }

        // 2. User-defined projection parameters via ProjCoordTransGeoKey
        if let Some(ct) = gk.proj_coord_trans {
            let fe = gk.false_easting.unwrap_or(0.0);
            let fn_m = gk.false_northing.unwrap_or(0.0);
            let lon0 = gk.nat_origin_long.or(gk.center_long).unwrap_or(0.0);
            let lat0 = gk.nat_origin_lat.or(gk.center_lat).unwrap_or(0.0);
            match ct {
                1 => {
                    let k0 = gk.scale_at_nat_origin.unwrap_or(1.0);
                    return Ok(Self::TransverseMercator(TransverseMercator::new(
                        ellip, lon0, lat0, k0, fe, fn_m,
                    )));
                }
                8 | 9 => {
                    let p1 = gk.std_parallel_1.unwrap_or(lat0);
                    let p2 = gk.std_parallel_2.unwrap_or(p1);
                    return Ok(Self::LambertConformalConic(LambertConformalConic::new_2sp(
                        ellip, p1, p2, lat0, lon0, fe, fn_m,
                    )));
                }
                11 => {
                    let p1 = gk.std_parallel_1.unwrap_or(29.5);
                    let p2 = gk.std_parallel_2.unwrap_or(45.5);
                    return Ok(Self::Albers(AlbersEqualArea::new(
                        ellip, p1, p2, lat0, lon0, fe, fn_m,
                    )));
                }
                15 => {
                    let lat_ts = gk.std_parallel_1.or(gk.nat_origin_lat).unwrap_or(70.0);
                    return Ok(Self::PolarStereographic(PolarStereographic::new(
                        ellip, lat_ts, lon0, fe, fn_m,
                    )));
                }
                _ => {}
            }
        }

        // 3. Geographic CRS (EPSG:4326, NAD83 4269, etc.)
        if gk.model_type == Some(2) || matches!(gk.geographic_type, Some(4326 | 4269 | 4258)) {
            return Ok(Self::Geographic);
        }

        Err(ProjError::UnsupportedCrs {
            projected: gk.projected_cs_type,
            geographic: gk.geographic_type,
            coord_trans: gk.proj_coord_trans,
        })
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
        }
    }

    /// Estimate the target pixel resolution `(rx, ry)` in source CRS units at the center of an XYZ tile,
    /// used to select the optimal COG overview IFD.
    pub fn target_resolution_for_tile(&self, tile: TileCoord, tile_size: u32) -> (f64, f64) {
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
        let cx = 0.5 * (min_x + max_x);
        let cy = 0.5 * (min_y + max_y);
        let dx = (max_x - min_x) / (tile_size as f64);
        let dy = (max_y - min_y) / (tile_size as f64);

        let (x0, y0) = self.from_3857(cx, cy);
        let (xx, yx) = self.from_3857(cx + dx, cy);
        let (xy, yy) = self.from_3857(cx, cy - dy);

        let rx = ((xx - x0).powi(2) + (yx - y0).powi(2)).sqrt();
        let ry = ((xy - x0).powi(2) + (yy - y0).powi(2)).sqrt();
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
        }
    }

    /// Inverse map a point `(x, y)` in this CRS back to WGS84 `(lon_deg, lat_deg)`.
    pub fn to_lonlat_deg(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
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
        }
    }

    /// Exact WGS84 geographic bounding box `[min_lon, min_lat, max_lon, max_lat]` by sampling
    /// 32 points along the 4 edges of the raster.
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

            // Include +/- 1 pixel margin for bilinear neighbors
            let src_min_col = (min_u.floor() as i32).max(0);
            let src_max_col = ((max_u.ceil() as i32) + 1).min(ifd_w as i32 - 1);
            let src_min_row = (min_v.floor() as i32).max(0);
            let src_max_row = ((max_v.ceil() as i32) + 1).min(ifd_h as i32 - 1);

            if max_u < -0.5 || min_u > (ifd_w as f32 - 0.5) || max_v < -0.5 || min_v > (ifd_h as f32 - 0.5) {
                return None;
            }
            if src_min_col > src_max_col || src_min_row > src_max_row {
                return None;
            }

            let all_in_bounds = min_u >= 0.0
                && min_v >= 0.0
                && (max_u.floor() as i32 + 1) < ifd_w as i32
                && (max_v.floor() as i32 + 1) < ifd_h as i32;

            Some(Self::Separable1D {
                u,
                v,
                src_min_col,
                src_max_col,
                src_min_row,
                src_max_row,
                all_in_bounds,
            })
        } else if tile.z <= 8 {
            // Low-zoom projected tile (z=0..8): evaluate exact closed-form per pixel so even
            // global/continental views of UTM/Albers/Polar scenes have zero curvature distortion.
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
                    if !sx.is_finite() || !sy.is_finite() {
                        continue;
                    }
                    let (col, row) = crs_to_pixel.forward(sx, sy);
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

            let src_min_col = (min_u.floor() as i32 - 1).max(0);
            let src_max_col = (max_u.ceil() as i32 + 1).min(ifd_w as i32 - 1);
            let src_min_row = (min_v.floor() as i32 - 1).max(0);
            let src_max_row = (max_v.ceil() as i32 + 1).min(ifd_h as i32 - 1);

            if src_min_col > src_max_col || src_min_row > src_max_row {
                return None;
            }

            let all_finite = grid_u.iter().all(|x| x.is_finite()) && grid_v.iter().all(|y| y.is_finite());
            let all_in_bounds = all_finite
                && min_u >= 0.0
                && min_v >= 0.0
                && (max_u.floor() as i32 + 1) < ifd_w as i32
                && (max_v.floor() as i32 + 1) < ifd_h as i32;

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
