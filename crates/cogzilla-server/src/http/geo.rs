//! CRS parsing, bounding-box reprojection, and tile-footprint intersection helpers.

use cogzilla_proj::mercator::lonlat_deg_to_mercator;
use cogzilla_proj::{TileCoord, mercator_to_lonlat_deg};

use crate::catalog::SceneCatalog;
use crate::engine::{EngineError, TileEngine};

use super::web_mercator_buffered_3857_bbox;

/// Parse a TiTiler CRS string (`EPSG:32610`, `EPSG:900913`, `CRS84`, `http://.../EPSG/0/32610`,
/// `urn:ogc:def:crs:EPSG::32610`, etc.) into `(CrsModel, epsg_code)` using our 3,260+ EPSG table.
pub(crate) fn parse_coord_crs_with_epsg(raw: &str) -> Result<(cogzilla_proj::CrsModel, u16), String> {
    let raw_trimmed = raw.trim();
    let trimmed = raw_trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(raw_trimmed)
        .trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower == "crs84"
        || lower == "crs:84"
        || lower.ends_with(":crs84")
        || lower.ends_with("/crs84")
    {
        return Ok((cogzilla_proj::CrsModel::Geographic, 4326));
    }
    if lower == "esri:102100" || lower == "esri:102113" {
        return Ok((cogzilla_proj::CrsModel::Epsg3857, 3857));
    }
    // Common PROJ4 strings (both `%2Bproj=...` and raw `+proj=...` where `+` was decoded as space)
    if lower.contains("proj=") || lower.contains("init=epsg:") {
        let tokens: Vec<&str> = lower
            .split_whitespace()
            .map(|t| t.trim_start_matches('+'))
            .collect();
        if let Some(digits) = tokens.iter().find_map(|tok| tok.strip_prefix("init=epsg:")) {
            if digits == "900913" {
                return Ok((cogzilla_proj::CrsModel::Epsg3857, 3857));
            }
            if let Ok(code) = digits.parse::<u16>() {
                let model = cogzilla_proj::CrsModel::from_epsg(code)
                    .map_err(|e| format!("unsupported CRS '{raw}': {e}"))?;
                return Ok((model, code));
            }
        }
        let proj_name = tokens
            .iter()
            .find_map(|tok| tok.strip_prefix("proj="))
            .unwrap_or("");
        match proj_name {
            "longlat" | "latlong" => return Ok((cogzilla_proj::CrsModel::Geographic, 4326)),
            "merc" => return Ok((cogzilla_proj::CrsModel::Epsg3857, 3857)),
            "utm" => {
                if let Some(zone) = tokens
                    .iter()
                    .find_map(|tok| tok.strip_prefix("zone="))
                    .and_then(|z| z.parse::<u16>().ok())
                    .filter(|&z| (1..=60).contains(&z))
                {
                    let is_south = tokens.contains(&"south");
                    let code = if is_south { 32700 + zone } else { 32600 + zone };
                    let model = cogzilla_proj::CrsModel::from_epsg(code)
                        .map_err(|e| format!("unsupported CRS '{raw}': {e}"))?;
                    return Ok((model, code));
                }
            }
            _ => {}
        }
    }
    let digits = if let Some(rest) = lower.strip_prefix("epsg:") {
        rest.trim()
    } else if let Some(pos) = lower.find("epsg:") {
        // e.g. urn:ogc:def:crs:epsg::32610 or urn:ogc:def:crs:epsg:6.3:32610
        lower[pos + 5..].rsplit(':').next().unwrap_or("").trim()
    } else if let Some(pos) = lower.find("/epsg/") {
        // e.g. http://www.opengis.net/def/crs/EPSG/0/32610
        lower[pos + 6..].rsplit('/').next().unwrap_or("").trim()
    } else {
        return Err(format!(
            "unsupported CRS '{raw}' (expected EPSG:<code>, CRS84, or PROJ string)"
        ));
    };
    if digits == "900913" {
        return Ok((cogzilla_proj::CrsModel::Epsg3857, 3857));
    }
    if digits == "4979" {
        return Ok((cogzilla_proj::CrsModel::Geographic, 4979));
    }
    let Ok(code) = digits.parse::<u16>() else {
        return Err(format!("unsupported CRS '{raw}' (expected EPSG:<code> or CRS84)"));
    };
    let model = cogzilla_proj::CrsModel::from_epsg(code)
        .map_err(|e| format!("unsupported CRS '{raw}': {e}"))?;
    Ok((model, code))
}

pub(super) fn dataset_native_crs_and_bounds(
    engine: &TileEngine,
    cog: &cogzilla_io::OpenedCog,
) -> Result<(cogzilla_proj::CrsModel, Option<u16>, [f64; 4]), EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let full = cog.index.full_res();
    let gk = &cog.index.geo_keys;
    let std_proj_epsg = gk.projected_cs_type.filter(|&c| c > 0 && c != 32767);
    let is_user_defined_projected = std_proj_epsg.is_none()
        && (gk.model_type == Some(1)
            || gk.projected_cs_type == Some(32767)
            || gk.proj_coord_trans.is_some());
    let native_epsg = if let Some(epsg) = std_proj_epsg {
        Some(epsg)
    } else if is_user_defined_projected {
        None
    } else if let Some(epsg) = gk.geographic_type.filter(|&c| c > 0 && c != 32767) {
        Some(epsg)
    } else {
        Some(4326)
    };
    let (x00, y00) = full.pixel_to_crs.forward(0.0, 0.0);
    let (x10, y10) = full.pixel_to_crs.forward(full.width as f64, 0.0);
    let (x01, y01) = full.pixel_to_crs.forward(0.0, full.height as f64);
    let (x11, y11) = full.pixel_to_crs.forward(full.width as f64, full.height as f64);
    let native_bounds = [
        x00.min(x10).min(x01).min(x11),
        y00.min(y10).min(y01).min(y11),
        x00.max(x10).max(x01).max(x11),
        y00.max(y10).max(y01).max(y11),
    ];
    Ok((crs, native_epsg, native_bounds))
}

/// Transform a WGS84 `[min_lon, min_lat, max_lon, max_lat]` bbox into `dst` CRS by densifying
/// along the 4 edges plus an interior lattice so curved/limited-domain projections (e.g. UTM,
/// NZTM, LAEA on global COGs) preserve their full projected envelope.
pub(super) fn reproject_wgs84_bounds(
    wgs84: [f64; 4],
    dst: &cogzilla_proj::CrsModel,
) -> Result<[f64; 4], EngineError> {
    if matches!(dst, cogzilla_proj::CrsModel::Geographic) {
        return Ok(wgs84);
    }
    let [w, mut s, e, mut n] = wgs84;
    // TransverseMercator::forward_raw rejects |lat| > 1.56 rad (~89.38°); clamp polar edges
    // to [-89.35°, 89.35°] so global COGs don't drop the 67.5°..89.3° band, while rejecting
    // bboxes wholly outside the TM domain (s >= n after clamping).
    if matches!(dst, cogzilla_proj::CrsModel::TransverseMercator(_)) {
        const MAX_TM_LAT: f64 = 89.35;
        s = s.clamp(-MAX_TM_LAT, MAX_TM_LAT);
        n = n.clamp(-MAX_TM_LAT, MAX_TM_LAT);
        if s >= n {
            return Err(EngineError::BadRequest(
                "bounds could not be projected to target CRS".into(),
            ));
        }
    }
    let antimeridian = e < w;
    let crosses_180 = antimeridian || e > 180.0;
    let lon_span = if antimeridian { (e + 360.0) - w } else { e - w };
    let wrap_lon = |lo: f64| if lo > 180.0 { lo - 360.0 } else { lo };
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut accum = |lo: f64, la: f64| {
        let (x, y) = dst.from_lonlat_deg(lo, la);
        if x.is_finite() && y.is_finite() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    };
    const SAMPLES: usize = 16;
    for i in 0..=SAMPLES {
        let t = i as f64 / SAMPLES as f64;
        let lon = wrap_lon(w + lon_span * t);
        let lat = s + (n - s) * t;
        accum(lon, s);
        accum(lon, n);
        accum(wrap_lon(w), lat);
        accum(wrap_lon(e), lat);
        if crosses_180 {
            accum(180.0, lat);
            accum(-180.0, lat);
        }
    }
    const INTERIOR: usize = 8;
    for iy in 1..INTERIOR {
        let lat = s + (n - s) * (iy as f64 / INTERIOR as f64);
        for ix in 1..INTERIOR {
            let lon = wrap_lon(w + lon_span * (ix as f64 / INTERIOR as f64));
            accum(lon, lat);
        }
    }
    if !min_x.is_finite() || min_x >= max_x || min_y >= max_y {
        return Err(EngineError::BadRequest(
            "bounds could not be projected to target CRS".into(),
        ));
    }
    Ok([min_x, min_y, max_x, max_y])
}

/// Validate a target rendering bbox and unwrap antimeridian-crossing geographic bboxes (`w > e`)
/// into `[w, s, e + 360, n]` so aspect ratio, `ModelPixelScale`, and pixel stepping are positive.
pub(super) fn normalize_render_bbox(
    mut bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
) -> Result<[f64; 4], EngineError> {
    if !bbox.iter().all(|v| v.is_finite()) || bbox[1] >= bbox[3] {
        return Err(EngineError::BadRequest(format!(
            "invalid bbox: [{}, {}, {}, {}]",
            bbox[0], bbox[1], bbox[2], bbox[3]
        )));
    }
    if dst_crs.is_geographic() && bbox[0] > bbox[2] {
        bbox[2] += 360.0;
    }
    if bbox[0] >= bbox[2] {
        return Err(EngineError::BadRequest(format!(
            "invalid bbox: [{}, {}, {}, {}]",
            bbox[0], bbox[1], bbox[2], bbox[3]
        )));
    }
    Ok(bbox)
}

/// Return the standard EPSG code for a COG's GeoKeys, or `None` for user-defined projected CRSes
/// (`ProjectedCSTypeGeoKey == 32767` or `ModelType == 1` / `ProjCoordTrans` without a standard EPSG).
pub(super) fn cog_standard_epsg(gk: &cogzilla_tiff::GeoKeys) -> Option<u16> {
    if let Some(epsg) = gk.projected_cs_type.filter(|&c| c > 0 && c != 32767) {
        return Some(epsg);
    }
    if gk.model_type == Some(1)
        || gk.projected_cs_type == Some(32767)
        || gk.proj_coord_trans.is_some()
    {
        return None;
    }
    Some(
        gk.geographic_type
            .filter(|&c| c > 0 && c != 32767)
            .unwrap_or(4326),
    )
}

/// Return the COG's native `(standard_epsg_opt, [minx, miny, maxx, maxy])` in its own CRS.
pub(super) fn cog_native_epsg_and_bounds(cog: &cogzilla_io::OpenedCog) -> (Option<u16>, [f64; 4]) {
    let full = cog.index.full_res();
    let epsg = cog_standard_epsg(&cog.index.geo_keys);
    let w = full.width as f64;
    let h = full.height as f64;
    let corners = [
        full.pixel_to_crs.forward(0.0, 0.0),
        full.pixel_to_crs.forward(w, 0.0),
        full.pixel_to_crs.forward(0.0, h),
        full.pixel_to_crs.forward(w, h),
    ];
    let min_x = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
    let max_x = corners.iter().map(|c| c.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let max_y = corners.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
    (epsg, [min_x, min_y, max_x, max_y])
}

pub(super) fn coord_to_lonlat(
    x: f64,
    y: f64,
    coord_crs: Option<&cogzilla_proj::CrsModel>,
) -> Result<(f64, f64), EngineError> {
    let (lon, lat) = match coord_crs {
        None | Some(cogzilla_proj::CrsModel::Geographic) => (x, y),
        Some(cogzilla_proj::CrsModel::Epsg3857) => mercator_to_lonlat_deg(x, y),
        Some(crs) => crs.to_lonlat_deg(x, y),
    };
    if !lon.is_finite() || !lat.is_finite() {
        return Err(EngineError::BadRequest(
            "point coordinates could not be transformed to WGS84".into(),
        ));
    }
    let Some(norm_lon) = cogzilla_proj::normalize_geographic_lon(lon) else {
        return Err(EngineError::NotFound(format!(
            "point ({x}, {y}) is outside dataset bounds"
        )));
    };
    Ok((norm_lon, lat))
}

pub(super) fn bbox_to_wgs84(
    bbox: [f64; 4],
    coord_crs: Option<&cogzilla_proj::CrsModel>,
) -> Result<[f64; 4], EngineError> {
    let Some(crs) = coord_crs else {
        return Ok(bbox);
    };
    if matches!(crs, cogzilla_proj::CrsModel::Geographic) {
        return Ok(bbox);
    }
    let [x0, y0, x1, y1] = bbox;
    let mx = 0.5 * (x0 + x1);
    let my = 0.5 * (y0 + y1);
    let pts = [
        (x0, y0),
        (x1, y0),
        (x0, y1),
        (x1, y1),
        (mx, y0),
        (mx, y1),
        (x0, my),
        (x1, my),
    ];
    let (center_lon, center_lat) = crs.to_lonlat_deg(mx, my);
    let mut ref_lon = if center_lon.is_finite() && center_lat.is_finite() {
        Some(center_lon)
    } else {
        None
    };
    let mut min_lon = f64::INFINITY;
    let mut min_lat = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    for (x, y) in pts {
        let (lo, la) = crs.to_lonlat_deg(x, y);
        if lo.is_finite() && la.is_finite() {
            let r = *ref_lon.get_or_insert(lo);
            let mut delta = lo - r;
            if delta > 180.0 {
                delta -= 360.0;
            } else if delta < -180.0 {
                delta += 360.0;
            }
            let unwrapped = r + delta;
            min_lon = min_lon.min(unwrapped);
            max_lon = max_lon.max(unwrapped);
            min_lat = min_lat.min(la);
            max_lat = max_lat.max(la);
        }
    }
    if min_lon < -180.0 {
        min_lon += 360.0;
        max_lon += 360.0;
    }
    // Polar-centered projected bboxes (e.g. EPSG:3413 / EPSG:3031) contain a pole in their interior:
    // if either pole projects inside [x0..=x1, y0..=y1], span full [-180, 180] and extend to ±90°.
    // Exclude cylindrical / pseudocylindrical projections where poles are horizontal lines (not points).
    let is_line_pole = crs.is_geographic()
        || crs.is_web_mercator()
        || matches!(
            crs,
            cogzilla_proj::CrsModel::WorldMercator
                | cogzilla_proj::CrsModel::CylindricalEqualArea(_)
                | cogzilla_proj::CrsModel::EqualEarth(_)
        );
    if !is_line_pole {
        let (bx0, bx1) = (x0.min(x1), x0.max(x1));
        let (by0, by1) = (y0.min(y1), y0.max(y1));
        for pole_lat in [90.0_f64, -90.0_f64] {
            let (px, py) = crs.from_lonlat_deg(0.0, pole_lat);
            if px.is_finite()
                && py.is_finite()
                && px >= bx0
                && px <= bx1
                && py >= by0
                && py <= by1
            {
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
    if !min_lon.is_finite() || min_lon >= max_lon || min_lat >= max_lat {
        return Err(EngineError::BadRequest(
            "bbox coordinates could not be transformed to WGS84".into(),
        ));
    }
    Ok([min_lon, min_lat, max_lon, max_lat])
}

/// Query a MosaicJSON's WebMercator quadkey index for a WGS84 bbox, splitting antimeridian-crossing
/// bboxes (`e > 180` or `w > e`) into eastern and western hemisphere halves deduplicated in order.
pub(super) fn query_mosaicjson_wgs84_bbox(
    mj: &crate::catalog::MosaicJsonCatalog,
    wgs_bbox: [f64; 4],
) -> Vec<String> {
    let [w, s, e, n] = wgs_bbox;
    let query_half = |lo0: f64, lo1: f64| {
        let (mx0, my0) = lonlat_deg_to_mercator(lo0, s);
        let (mx1, my1) = lonlat_deg_to_mercator(lo1, n);
        mj.query_bbox_3857(mx0.min(mx1), my0.min(my1), mx0.max(mx1), my0.max(my1))
    };
    if e > 180.0 || w > e {
        let e_wrapped = if e > 180.0 { e - 360.0 } else { e };
        let mut urls = query_half(w.clamp(-180.0, 180.0), 180.0);
        for u in query_half(-180.0, e_wrapped.clamp(-180.0, 180.0)) {
            if !urls.contains(&u) {
                urls.push(u);
            }
        }
        urls
    } else {
        query_half(w, e)
    }
}

/// Check whether two WGS84 bounding boxes `[w, s, e, n]` intersect (handling antimeridian-crossing
/// bboxes where `w > e` or `e > 180`).
#[inline]
pub(super) fn wgs84_bboxes_intersect(a: [f64; 4], b: [f64; 4]) -> bool {
    let [aw, as_, ae, an] = a;
    let [bw, bs, be, bn] = b;
    if !(aw.is_finite()
        && as_.is_finite()
        && ae.is_finite()
        && an.is_finite()
        && bw.is_finite()
        && bs.is_finite()
        && be.is_finite()
        && bn.is_finite())
    {
        return false;
    }
    if an < bs || as_ > bn {
        return false;
    }
    // Split any longitude interval [w, e] into 1 or 2 normalized segments in [-180, 180].
    let split_lon = |w: f64, e: f64| -> [(f64, f64); 2] {
        if e - w >= 360.0 {
            return [(-180.0, 180.0), (f64::NAN, f64::NAN)];
        }
        if w > e {
            // RFC 7946 antimeridian-crossing [w..180] U [-180..e]
            [(w.clamp(-180.0, 180.0), 180.0), (-180.0, e.clamp(-180.0, 180.0))]
        } else if w < -180.0 && e <= -180.0 {
            [((w + 360.0).max(-180.0), (e + 360.0).min(180.0)), (f64::NAN, f64::NAN)]
        } else if w < -180.0 {
            [(-180.0, e.min(180.0)), ((w + 360.0).max(-180.0), 180.0)]
        } else if e > 180.0 && w >= 180.0 {
            [((w - 360.0).max(-180.0), (e - 360.0).min(180.0)), (f64::NAN, f64::NAN)]
        } else if e > 180.0 {
            [(w.max(-180.0), 180.0), (-180.0, (e - 360.0).min(180.0))]
        } else {
            [(w, e), (f64::NAN, f64::NAN)]
        }
    };
    let segs_a = split_lon(aw, ae);
    let segs_b = split_lon(bw, be);
    for (w1, e1) in segs_a {
        if !w1.is_finite() {
            continue;
        }
        for (w2, e2) in segs_b {
            if !w2.is_finite() {
                continue;
            }
            if e1 >= w2 && w1 <= e2 {
                return true;
            }
        }
    }
    false
}

/// Exact geometric test: returns `true` iff at least one point of the tile box `tb` in `dst_crs`
/// maps onto `[0..full.width, 0..full.height]` in `src_crs`, OR a corner/center of the dataset
/// maps inside `tb`. Failed / non-finite reprojections are treated as outside (never as inside).
pub(crate) fn tile_box_intersects_cog(
    src_crs: &cogzilla_proj::CrsModel,
    full: &cogzilla_tiff::IfdMeta,
    dst_crs: &cogzilla_proj::CrsModel,
    tb: [f64; 4],
) -> bool {
    let [tx0, ty0, tx1, ty1] = tb;
    let wf = full.width as f64;
    let hf = full.height as f64;
    let wrap_lo = src_crs.geographic_0_360_min_lon(&full.pixel_to_crs, full.width);
    let wrap_hi = src_crs.geographic_neg_180_max_lon(&full.pixel_to_crs, full.width);

    let dst_pt_in_src = |dx: f64, dy: f64| -> bool {
        let (mut sx, sy) = match dst_crs {
            cogzilla_proj::CrsModel::Epsg3857 => src_crs.from_3857(dx, dy),
            cogzilla_proj::CrsModel::Geographic => {
                let lon = if dx > 180.0 {
                    dx - 360.0
                } else if dx < -180.0 {
                    dx + 360.0
                } else {
                    dx
                };
                src_crs.from_lonlat_deg(lon, dy)
            }
            other => {
                let (lon, lat) = other.to_lonlat_deg(dx, dy);
                if lon.is_finite() && lat.is_finite() {
                    src_crs.from_lonlat_deg(lon, lat)
                } else {
                    return false;
                }
            }
        };
        if !sx.is_finite() || !sy.is_finite() {
            return false;
        }
        if let Some(lo) = wrap_lo
            && sx < lo
        {
            sx += 360.0;
        } else if let Some(hi) = wrap_hi
            && sx > hi
        {
            sx -= 360.0;
        }
        let (col, row) = full.crs_to_pixel_pt(sx, sy);
        col.is_finite()
            && row.is_finite()
            && col >= -0.5
            && col <= wf + 0.5
            && row >= -0.5
            && row <= hf + 0.5
    };

    // 1. O(1) fast-accept: tile center + 4 corners (hits immediately on 99%+ of in-bounds tiles)
    let cx = 0.5 * (tx0 + tx1);
    let cy = 0.5 * (ty0 + ty1);
    for (dx, dy) in [(cx, cy), (tx0, ty0), (tx1, ty0), (tx0, ty1), (tx1, ty1)] {
        if dst_pt_in_src(dx, dy) {
            return true;
        }
    }

    // 2. Densified tile outline (16 steps per edge) + 3x3 interior grid
    const STEPS: usize = 16;
    for i in 1..STEPS {
        let t = (i as f64) / (STEPS as f64);
        let x = tx0 + t * (tx1 - tx0);
        let y = ty0 + t * (ty1 - ty0);
        if dst_pt_in_src(x, ty0)
            || dst_pt_in_src(x, ty1)
            || dst_pt_in_src(tx0, y)
            || dst_pt_in_src(tx1, y)
        {
            return true;
        }
    }
    for iy in 1..=3 {
        let y = ty0 + (iy as f64 / 4.0) * (ty1 - ty0);
        for ix in 1..=3 {
            let x = tx0 + (ix as f64 / 4.0) * (tx1 - tx0);
            if dst_pt_in_src(x, y) {
                return true;
            }
        }
    }

    // 3. Dataset anchor points (center + 4 corners + 4 edge midpoints) projected into dst_crs
    //    (catches small datasets strictly contained inside a large low-zoom tile).
    let anchors = [
        (0.5 * wf, 0.5 * hf),
        (0.0, 0.0),
        (wf, 0.0),
        (0.0, hf),
        (wf, hf),
        (0.5 * wf, 0.0),
        (0.5 * wf, hf),
        (0.0, 0.5 * hf),
        (wf, 0.5 * hf),
    ];
    for (col, row) in anchors {
        let (sx, sy) = full.pixel_to_crs_pt(col, row);
        let (lon, lat) = src_crs.to_lonlat_deg(sx, sy);
        if !lon.is_finite() || !lat.is_finite() {
            continue;
        }
        let (dx, dy) = dst_crs.from_lonlat_deg(lon, lat);
        if dx.is_finite() && dy.is_finite() && dx >= tx0 && dx <= tx1 && dy >= ty0 && dy <= ty1 {
            return true;
        }
    }
    false
}

/// Check whether a requested tile intersects the dataset's footprint (used to return
/// semantically exact `404 Not Found` on out-of-bounds `/cog/tiles` and `/stac/tiles` requests
/// across all 13 TileMatrixSets). Uses an O(1) WGS84 fast-reject on Mercator/Geographic grids
/// and an exact densified-outline footprint check whenever the cheap test is inconclusive.
pub(super) fn tile_intersects_cog(
    engine: &TileEngine,
    cog: &cogzilla_io::OpenedCog,
    tms: &str,
    tile: TileCoord,
    buffer: f32,
    tile_size: u32,
) -> Result<bool, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let full = cog.index.full_res();
    let [cw, cs, ce, cn] = crs.wgs84_bounds_with_gcp(
        &full.pixel_to_crs,
        full.gcp_poly.as_deref(),
        full.width,
        full.height,
    );
    if let Some(tms_def) = crate::tms::lookup_tms(tms) {
        let Some(tb) = tms_def.tile_bounds(tile.z, tile.x, tile.y, buffer, tile_size) else {
            return Ok(false);
        };
        let dst_crs = tms_def.crs_model()?;
        // Stage 1: O(1) cheap WGS84 fast-reject on WebMercatorQuad & WorldCRS84Quad
        if cw.is_finite() && cs.is_finite() && ce.is_finite() && cn.is_finite() {
            match dst_crs {
                cogzilla_proj::CrsModel::Epsg3857 => {
                    let (lon0, lat0) = mercator_to_lonlat_deg(tb[0], tb[1]);
                    let (lon1, lat1) = mercator_to_lonlat_deg(tb[2], tb[3]);
                    let tw = [lon0.min(lon1), lat0.min(lat1), lon0.max(lon1), lat0.max(lat1)];
                    if !wgs84_bboxes_intersect(tw, [cw, cs, ce, cn]) {
                        return Ok(false);
                    }
                    // For axis-aligned Mercator/Geographic/CEA rasters without nonlinear GCPs, the WGS84 box IS exact.
                    if full.gcp_poly.is_none()
                        && full.pixel_to_crs.is_axis_aligned()
                        && (crs.is_web_mercator()
                            || crs.is_geographic()
                            || matches!(crs, cogzilla_proj::CrsModel::CylindricalEqualArea(_)))
                    {
                        return Ok(true);
                    }
                }
                cogzilla_proj::CrsModel::Geographic => {
                    if !wgs84_bboxes_intersect(tb, [cw, cs, ce, cn]) {
                        return Ok(false);
                    }
                    if full.gcp_poly.is_none()
                        && full.pixel_to_crs.is_axis_aligned()
                        && (crs.is_web_mercator()
                            || crs.is_geographic()
                            || matches!(crs, cogzilla_proj::CrsModel::CylindricalEqualArea(_)))
                    {
                        return Ok(true);
                    }
                }
                _ => {}
            }
        }
        // Stage 2: Exact densified-outline footprint test in source CRS (treats failed / infinite
        // reprojections as outside; never depends on the TileMatrixSet).
        return Ok(tile_box_intersects_cog(&crs, full, &dst_crs, tb));
    }
    let tb = web_mercator_buffered_3857_bbox(tile.z, tile.x, tile.y, buffer, tile_size);
    if cw.is_finite() && cs.is_finite() && ce.is_finite() && cn.is_finite() {
        let (lon0, lat0) = mercator_to_lonlat_deg(tb[0], tb[1]);
        let (lon1, lat1) = mercator_to_lonlat_deg(tb[2], tb[3]);
        if !wgs84_bboxes_intersect(
            [lon0.min(lon1), lat0.min(lat1), lon0.max(lon1), lat0.max(lat1)],
            [cw, cs, ce, cn],
        ) {
            return Ok(false);
        }
    }
    Ok(tile_box_intersects_cog(
        &crs,
        full,
        &cogzilla_proj::CrsModel::Epsg3857,
        tb,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_proj::CrsModel;

    #[test]
    fn datum_shifted_geographic_epsg_codes_behave_as_geographic() {
        // NAD27 (4267), OSGB36 (4277), Amersfoort (4289), CH1903 (4149), CH1903+ (4150).
        for epsg in [4326u16, 4267, 4277, 4289, 4149, 4150] {
            let crs = CrsModel::from_epsg(epsg).unwrap_or_else(|e| panic!("EPSG:{epsg}: {e}"));
            assert!(crs.is_geographic(), "EPSG:{epsg} must be geographic");
            assert!(crs.is_degree_based(), "EPSG:{epsg} must be degree-based");
            assert!(!crs.is_web_mercator(), "EPSG:{epsg} must not be web mercator");

            // Antimeridian bbox unwrap: 170..-170 -> 170..190 (previously 400 on 4267/4277/4289/4149/4150).
            let unwrapped = super::super::resolve_input_bbox([170.0, 0.0, -170.0, 10.0], Some(&crs))
                .unwrap_or_else(|e| panic!("EPSG:{epsg} antimeridian resolve_input_bbox failed: {e}"));
            assert_eq!(unwrapped, [170.0, 0.0, 190.0, 10.0], "EPSG:{epsg} antimeridian unwrap");

            // ±2-turn longitude normalization.
            let norm = super::super::resolve_input_bbox([530.0, 0.0, 550.0, 10.0], Some(&crs))
                .unwrap_or_else(|e| panic!("EPSG:{epsg} longitude norm failed: {e}"));
            assert_eq!(norm, [170.0, 0.0, 190.0, 10.0], "EPSG:{epsg} lon normalization");

            // normalize_render_bbox antimeridian unwrap.
            let rendered = normalize_render_bbox([170.0, 0.0, -170.0, 10.0], &crs)
                .unwrap_or_else(|e| panic!("EPSG:{epsg} normalize_render_bbox failed: {e}"));
            assert_eq!(rendered, [170.0, 0.0, 190.0, 10.0], "EPSG:{epsg} render bbox unwrap");

            // Point coordinate transformation succeeds (and preserves datum shift for non-WGS84).
            let (lon, lat) = coord_to_lonlat(-100.0, 40.0, Some(&crs))
                .unwrap_or_else(|e| panic!("EPSG:{epsg} coord_to_lonlat failed: {e}"));
            assert!((lon - (-100.0)).abs() < 0.05 && (lat - 40.0).abs() < 0.05, "EPSG:{epsg}: ({lon}, {lat})");
            if epsg != 4326 {
                assert!(
                    (lon - (-100.0)).abs() > 1e-5 || (lat - 40.0).abs() > 1e-5,
                    "EPSG:{epsg} must preserve non-zero datum shift, got ({lon}, {lat})"
                );
            }
        }

        // Zonal densify_ring_lonlat_to_pixel must preserve Helmert datum shift on datum-shifted geographic CRSes.
        let nad27 = CrsModel::from_epsg(4267).unwrap();
        let identity = cogzilla_tiff::Affine2D {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 1.0,
            f: 0.0,
        };
        let ring = vec![(-100.0, 40.0), (-99.0, 40.0), (-99.0, 41.0), (-100.0, 40.0)];
        let shifted = crate::zonal::densify_ring_lonlat_to_pixel(
            &ring,
            &nad27,
            &identity,
            None,
            nad27.is_geographic(),
            false,
            0.0,
            ring.len(),
        );
        let (sx, sy) = shifted[0];
        let (exp_x, exp_y) = nad27.from_lonlat_deg(-100.0, 40.0);
        assert!(
            (sx - exp_x).abs() < 1e-9 && (sy - exp_y).abs() < 1e-9,
            "NAD27 zonal ring must apply Helmert shift: got ({sx}, {sy}), expected ({exp_x}, {exp_y})"
        );
        assert!(
            (sx - (-100.0)).abs() > 1e-5 || (sy - 40.0).abs() > 1e-5,
            "NAD27 zonal ring shift must be non-zero"
        );

        // Projected CRSes (e.g. EPSG:3857) must still reject minx >= maxx.
        let merc = CrsModel::from_epsg(3857).unwrap();
        assert!(!merc.is_geographic());
        assert!(merc.is_web_mercator());
        assert!(super::super::resolve_input_bbox([170.0, 0.0, -170.0, 10.0], Some(&merc)).is_err());
        assert!(normalize_render_bbox([170.0, 0.0, -170.0, 10.0], &merc).is_err());
    }
}
