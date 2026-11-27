//! GeoJSON geometry parsing, CRS extraction, and ray-casting / ALL_TOUCHED polygon masking.

use crate::engine::EngineError;

/// Simple ray-casting / ALL_TOUCHED rasterizer for GeoJSON Feature masking
/// (`POST /*/feature`, `POST /*/statistics`), supporting Polygon, MultiPolygon,
/// LineString, MultiLineString, Point, MultiPoint, and GeometryCollection.
#[derive(Clone)]
pub struct PolygonMask {
    /// List of polygons; each polygon is `[outer_ring, hole1, ...]`, each ring is `Vec<(lon, lat)>`.
    pub polygons: Vec<Vec<Vec<(f64, f64)>>>,
    /// List of polylines (`LineString` / `MultiLineString`) in `(lon, lat)`.
    pub lines: Vec<Vec<(f64, f64)>>,
    /// List of points (`Point` / `MultiPoint`) in `(lon, lat)`.
    pub points: Vec<(f64, f64)>,
    /// Precomputed `[min_lon, min_lat, max_lon, max_lat]` for fast bounding-box rejection.
    pub bbox: [f64; 4],
    /// Raw `[min_x, min_y, max_x, max_y]` in the incoming GeoJSON `coord_crs` (exact before WGS84 round-trip).
    pub raw_bbox: [f64; 4],
    /// True when the incoming GeoJSON coordinates were provided in a projected `coord_crs`.
    pub coord_crs_projected: bool,
}

/// Extract a legacy GeoJSON 2008 `"crs"` member (e.g.
/// `{"type": "name", "properties": {"name": "EPSG:32610"}}` or
/// `{"type": "EPSG", "properties": {"code": 32610}}`) from a GeoJSON object.
///
/// Checks `val["crs"]`, and if `val` is a `Feature` without its own `"crs"`, also checks
/// `val["geometry"]["crs"]`. Returns `Ok(None)` when absent or `null`.
pub fn extract_geojson_crs(
    val: &serde_json::Value,
) -> Result<Option<cogzilla_proj::CrsModel>, EngineError> {
    let crs_val = val
        .get("crs")
        .filter(|v| !v.is_null())
        .or_else(|| {
            if val.get("type").and_then(|v| v.as_str()) == Some("Feature") {
                val.get("geometry")
                    .and_then(|g| g.get("crs"))
                    .filter(|v| !v.is_null())
            } else {
                None
            }
        });
    let Some(crs_obj) = crs_val else {
        return Ok(None);
    };
    if let Some(s) = crs_obj.as_str() {
        let (model, _) = crate::http::parse_coord_crs_with_epsg(s)
            .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON crs: {e}")))?;
        return Ok(Some(model));
    }
    let crs_type = crs_obj
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let props = crs_obj.get("properties");
    if crs_type == "name" {
        let name = props
            .and_then(|p| p.get("name"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                EngineError::BadRequest(
                    "invalid GeoJSON crs: type 'name' requires properties.name string".into(),
                )
            })?;
        let (model, _) = crate::http::parse_coord_crs_with_epsg(name)
            .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON crs: {e}")))?;
        return Ok(Some(model));
    }
    if crs_type == "epsg" {
        let code_val = props.and_then(|p| p.get("code"));
        let raw = if let Some(n) = code_val.and_then(|v| v.as_u64()) {
            format!("EPSG:{n}")
        } else if let Some(s) = code_val.and_then(|v| v.as_str()) {
            if s.to_ascii_lowercase().starts_with("epsg:") {
                s.to_string()
            } else {
                format!("EPSG:{s}")
            }
        } else {
            return Err(EngineError::BadRequest(
                "invalid GeoJSON crs: type 'EPSG' requires properties.code".into(),
            ));
        };
        let (model, _) = crate::http::parse_coord_crs_with_epsg(&raw)
            .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON crs: {e}")))?;
        return Ok(Some(model));
    }
    Err(EngineError::BadRequest("unsupported GeoJSON crs member (expected type 'name' or 'EPSG'; pass ?coord_crs=EPSG:<code>)".to_string()))
}

impl PolygonMask {
    /// Empty mask (e.g. RFC 7946 §3.2 Feature with `"geometry": null`), producing zero-count
    /// null statistics identical to a feature outside the dataset.
    pub fn empty() -> Self {
        Self {
            polygons: Vec::new(),
            lines: Vec::new(),
            points: Vec::new(),
            bbox: [0.0; 4],
            raw_bbox: [0.0; 4],
            coord_crs_projected: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.polygons.is_empty() && self.lines.is_empty() && self.points.is_empty()
    }

    pub fn from_geojson(val: &serde_json::Value) -> Result<(Self, [f64; 4]), EngineError> {
        Self::from_geojson_in_crs(val, None)
    }

    pub fn from_geojson_in_crs(
        val: &serde_json::Value,
        coord_crs: Option<&cogzilla_proj::CrsModel>,
    ) -> Result<(Self, [f64; 4]), EngineError> {
        let owned_geojson_crs = if coord_crs.is_none() {
            extract_geojson_crs(val)?
        } else {
            None
        };
        let effective_crs = coord_crs.or(owned_geojson_crs.as_ref());
        let mut geometries: Vec<&serde_json::Value> = Vec::new();
        match val.get("type").and_then(|v| v.as_str()) {
            Some("FeatureCollection") => {
                let feats = val
                    .get("features")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        EngineError::BadRequest("GeoJSON FeatureCollection missing features".into())
                    })?;
                if feats.is_empty() {
                    return Err(EngineError::BadRequest(
                        "GeoJSON FeatureCollection has no features".into(),
                    ));
                }
                if feats.len() > 64 {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON FeatureCollection has {} features (max 64)",
                        feats.len()
                    )));
                }
                for f in feats {
                    let g = f.get("geometry").ok_or_else(|| {
                        EngineError::BadRequest("GeoJSON Feature missing geometry".into())
                    })?;
                    geometries.push(g);
                }
            }
            Some("Feature") => {
                let g = val.get("geometry").ok_or_else(|| {
                    EngineError::BadRequest("GeoJSON Feature missing geometry".into())
                })?;
                geometries.push(g);
            }
            _ => {
                geometries.push(val);
            }
        }

        const MAX_GEOJSON_VERTICES: usize = 65_536;
        const MAX_GEOJSON_RINGS: usize = 4_096;
        const MAX_GEOM_COLLECTION_DEPTH: usize = 8;
        const MAX_LEAF_GEOMETRIES: usize = 64;

        // Iteratively flatten any GeometryCollection nodes with bounded depth and leaf count.
        let mut flat_geoms: Vec<&serde_json::Value> = Vec::with_capacity(geometries.len());
        let mut stack: Vec<(&serde_json::Value, usize)> =
            geometries.into_iter().rev().map(|g| (g, 0)).collect();
        while let Some((g, depth)) = stack.pop() {
            if g.get("type").and_then(|v| v.as_str()) == Some("GeometryCollection") {
                if depth >= MAX_GEOM_COLLECTION_DEPTH {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON GeometryCollection exceeds max nesting depth {MAX_GEOM_COLLECTION_DEPTH}"
                    )));
                }
                let children = g
                    .get("geometries")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        EngineError::BadRequest(
                            "GeoJSON GeometryCollection missing geometries array".into(),
                        )
                    })?;
                if children.len().saturating_add(stack.len()).saturating_add(flat_geoms.len())
                    > MAX_LEAF_GEOMETRIES
                {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON exceeds max {MAX_LEAF_GEOMETRIES} geometries"
                    )));
                }
                for child in children.iter().rev() {
                    stack.push((child, depth + 1));
                }
            } else {
                flat_geoms.push(g);
                if flat_geoms.len() > MAX_LEAF_GEOMETRIES {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON exceeds max {MAX_LEAF_GEOMETRIES} geometries"
                    )));
                }
            }
        }

        let mut total_vertices = 0usize;
        let mut total_rings = 0usize;
        let raw_bounds = std::cell::Cell::new([
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ]);

        let parse_coord = |pt: &serde_json::Value| -> Result<(f64, f64), EngineError> {
            let p = pt
                .as_array()
                .ok_or_else(|| EngineError::BadRequest("invalid GeoJSON coord".into()))?;
            if p.len() < 2 {
                return Err(EngineError::BadRequest("coord needs [lon, lat]".into()));
            }
            let (Some(x), Some(y)) = (p[0].as_f64(), p[1].as_f64()) else {
                return Err(EngineError::BadRequest(
                    "GeoJSON coordinates must be numeric".into(),
                ));
            };
            if !x.is_finite() || !y.is_finite() {
                return Err(EngineError::BadRequest(
                    "GeoJSON coordinates must be finite".into(),
                ));
            }
            let mut rb = raw_bounds.get();
            rb[0] = rb[0].min(x);
            rb[1] = rb[1].min(y);
            rb[2] = rb[2].max(x);
            rb[3] = rb[3].max(y);
            raw_bounds.set(rb);
            if let Some(crs) = effective_crs {
                let (lo, la) = crs.to_lonlat_deg(x, y);
                if !lo.is_finite() || !la.is_finite() {
                    return Err(EngineError::BadRequest(
                        "reprojected GeoJSON coordinates are non-finite".into(),
                    ));
                }
                Ok((lo, la))
            } else {
                if x.abs() > 360.0 || y.abs() > 90.0 {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON coordinate [{x}, {y}] is outside WGS84 lon/lat bounds ([-360, 360] x [-90, 90]); \
                         if coordinates are in a projected CRS, pass ?coord_crs=EPSG:<code> or include a GeoJSON 'crs' member"
                    )));
                }
                Ok((x, y))
            }
        };

        let parse_ring = |arr: &serde_json::Value,
                          min_len: usize,
                          total_rings: &mut usize,
                          total_vertices: &mut usize|
         -> Result<Vec<(f64, f64)>, EngineError> {
            let list = arr
                .as_array()
                .ok_or_else(|| EngineError::BadRequest("invalid GeoJSON coordinate array".into()))?;
            if list.len() < min_len {
                return Err(EngineError::BadRequest(format!(
                    "GeoJSON geometry ring/line must have at least {min_len} vertices"
                )));
            }
            *total_rings += 1;
            if *total_rings > MAX_GEOJSON_RINGS {
                return Err(EngineError::BadRequest(format!(
                    "GeoJSON geometry exceeds max {MAX_GEOJSON_RINGS} rings/lines"
                )));
            }
            *total_vertices = total_vertices.saturating_add(list.len());
            if *total_vertices > MAX_GEOJSON_VERTICES {
                return Err(EngineError::BadRequest(format!(
                    "GeoJSON geometry exceeds max {MAX_GEOJSON_VERTICES} vertices"
                )));
            }
            let mut ring = Vec::with_capacity(list.len());
            for pt in list {
                ring.push(parse_coord(pt)?);
            }
            Ok(ring)
        };

        let mut polygons = Vec::new();
        let mut lines = Vec::new();
        let mut points = Vec::new();
        for geom in flat_geoms {
            let gtype = geom
                .get("type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| EngineError::BadRequest("GeoJSON missing geometry.type".into()))?;
            let coords = geom
                .get("coordinates")
                .ok_or_else(|| EngineError::BadRequest("GeoJSON missing coordinates".into()))?;
            match gtype {
                "Polygon" => {
                    let rings = coords
                        .as_array()
                        .ok_or_else(|| EngineError::BadRequest("invalid Polygon coords".into()))?;
                    let mut poly = Vec::new();
                    for r in rings {
                        poly.push(parse_ring(r, 3, &mut total_rings, &mut total_vertices)?);
                    }
                    polygons.push(poly);
                }
                "MultiPolygon" => {
                    let polys = coords.as_array().ok_or_else(|| {
                        EngineError::BadRequest("invalid MultiPolygon coords".into())
                    })?;
                    for p in polys {
                        let rings = p
                            .as_array()
                            .ok_or_else(|| EngineError::BadRequest("invalid polygon".into()))?;
                        let mut poly = Vec::new();
                        for r in rings {
                            poly.push(parse_ring(r, 3, &mut total_rings, &mut total_vertices)?);
                        }
                        polygons.push(poly);
                    }
                }
                "LineString" => {
                    lines.push(parse_ring(coords, 2, &mut total_rings, &mut total_vertices)?);
                }
                "MultiLineString" => {
                    let ls = coords.as_array().ok_or_else(|| {
                        EngineError::BadRequest("invalid MultiLineString coords".into())
                    })?;
                    for l in ls {
                        lines.push(parse_ring(l, 2, &mut total_rings, &mut total_vertices)?);
                    }
                }
                "Point" => {
                    total_vertices = total_vertices.saturating_add(1);
                    if total_vertices > MAX_GEOJSON_VERTICES {
                        return Err(EngineError::BadRequest(format!(
                            "GeoJSON geometry exceeds max {MAX_GEOJSON_VERTICES} vertices"
                        )));
                    }
                    points.push(parse_coord(coords)?);
                }
                "MultiPoint" => {
                    let pts = coords.as_array().ok_or_else(|| {
                        EngineError::BadRequest("invalid MultiPoint coords".into())
                    })?;
                    total_vertices = total_vertices.saturating_add(pts.len());
                    if total_vertices > MAX_GEOJSON_VERTICES {
                        return Err(EngineError::BadRequest(format!(
                            "GeoJSON geometry exceeds max {MAX_GEOJSON_VERTICES} vertices"
                        )));
                    }
                    for pt in pts {
                        points.push(parse_coord(pt)?);
                    }
                }
                other => {
                    return Err(EngineError::BadRequest(format!(
                        "unsupported GeoJSON geometry type '{other}'"
                    )));
                }
            }
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for poly in &polygons {
            if let Some(outer) = poly.first() {
                for &(x, y) in outer {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }
        for line in &lines {
            for &(x, y) in line {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
        for &(x, y) in &points {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let has_only_polygons = lines.is_empty() && points.is_empty();
        if !min_x.is_finite()
            || !min_y.is_finite()
            || !max_x.is_finite()
            || !max_y.is_finite()
            || (has_only_polygons && (min_x >= max_x || min_y >= max_y))
        {
            return Err(EngineError::BadRequest(
                "empty or degenerate GeoJSON geometry".into(),
            ));
        }
        let bbox = [min_x, min_y, max_x, max_y];
        let rb = raw_bounds.get();
        let raw_bbox = if rb.iter().all(|v| v.is_finite()) {
            rb
        } else {
            bbox
        };
        let coord_crs_projected =
            effective_crs.is_some_and(|c| !c.is_geographic());
        Ok((Self { polygons, lines, points, bbox, raw_bbox, coord_crs_projected }, bbox))
    }

    /// Compute the exact tight bounding box of this feature's vertices directly in `dst_crs`
    /// (avoiding the area expansion caused by reprojecting an axis-aligned WGS84 envelope).
    pub fn bbox_in_crs(&self, dst_crs: &cogzilla_proj::CrsModel) -> Option<[f64; 4]> {
        let mut x0 = f64::INFINITY;
        let mut y0 = f64::INFINITY;
        let mut x1 = f64::NEG_INFINITY;
        let mut y1 = f64::NEG_INFINITY;
        let mut visit = |lon: f64, lat: f64| {
            let (x, y) = dst_crs.from_lonlat_deg(lon, lat);
            if x.is_finite() && y.is_finite() {
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        };
        for poly in &self.polygons {
            if let Some(outer) = poly.first() {
                for &(lo, la) in outer {
                    visit(lo, la);
                }
            }
        }
        for line in &self.lines {
            for &(lo, la) in line {
                visit(lo, la);
            }
        }
        for &(lo, la) in &self.points {
            visit(lo, la);
        }
        if x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite() && x0 < x1 && y0 < y1 {
            Some([x0, y0, x1, y1])
        } else {
            None
        }
    }

    fn ring_contains(ring: &[(f64, f64)], x: f64, y: f64) -> bool {
        let mut inside = false;
        let n = ring.len();
        if n < 3 {
            return false;
        }
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = ring[i];
            let (xj, yj) = ring[j];
            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Total number of geometry vertices across all rings, lines, and points.
    pub fn vertex_count(&self) -> usize {
        let poly_v: usize = self
            .polygons
            .iter()
            .flat_map(|rings| rings.iter())
            .map(|ring| ring.len())
            .sum();
        let line_v: usize = self.lines.iter().map(|l| l.len()).sum();
        poly_v + line_v + self.points.len()
    }

    /// Guard against `O(pixels * vertices)` point-in-polygon CPU exhaustion.
    pub fn check_work_budget(&self, pixels: usize) -> Result<(), EngineError> {
        const MAX_POLYGON_WORK: usize = 256_000_000;
        let verts = self.vertex_count();
        if pixels.saturating_mul(verts) > MAX_POLYGON_WORK {
            return Err(EngineError::BadRequest(format!(
                "polygon complexity ({verts} vertices x {pixels} pixels) exceeds work budget ({MAX_POLYGON_WORK})"
            )));
        }
        Ok(())
    }

    #[inline(always)]
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        if lon < self.bbox[0] || lon > self.bbox[2] || lat < self.bbox[1] || lat > self.bbox[3] {
            return false;
        }
        for poly in &self.polygons {
            if let Some(outer) = poly.first()
                && Self::ring_contains(outer, lon, lat) {
                    let in_hole = poly[1..].iter().any(|h| Self::ring_contains(h, lon, lat));
                    if !in_hole {
                        return true;
                    }
                }
        }
        false
    }

    /// Rasterize the polygon onto a `(w, h)` grid covering `bbox` in `dst_crs` using GDAL's
    /// `ALL_TOUCHED=TRUE` semantics: a pixel is `true` if its center is inside the polygon OR
    /// any polygon ring edge intersects the pixel cell `[px, px+1] x [py, py+1]`.
    pub fn rasterize_all_touched(
        &self,
        w: usize,
        h: usize,
        bbox: [f64; 4],
        dst_crs: &cogzilla_proj::CrsModel,
    ) -> Vec<bool> {
        let mut mask = vec![false; w * h];
        if w == 0 || h == 0 {
            return mask;
        }
        let dlx = (bbox[2] - bbox[0]) / (w as f64);
        let dly = (bbox[3] - bbox[1]) / (h as f64);
        if !dlx.is_finite() || !dly.is_finite() || dlx <= 0.0 || dly <= 0.0 {
            return mask;
        }
        let wrap_hi = self.bbox[2] > 180.0;
        let wrap_lo = self.bbox[0] < -180.0;
        if !self.polygons.is_empty() {
            let is_geo = matches!(dst_crs, cogzilla_proj::CrsModel::Geographic);
            let is_merc = matches!(
                dst_crs,
                cogzilla_proj::CrsModel::Epsg3857 | cogzilla_proj::CrsModel::WorldMercator
            );
            let project_pt = |lon: f64, lat: f64, lon_shift: f64| -> (f64, f64) {
                let (ax, ay) = if lon_shift != 0.0 && is_geo {
                    (lon + lon_shift, lat)
                } else {
                    dst_crs.from_lonlat_deg(lon + lon_shift, lat)
                };
                ((ax - bbox[0]) / dlx, (bbox[3] - ay) / dly)
            };
            let mut polys_px: Vec<Vec<Vec<(f64, f64)>>> = Vec::with_capacity(self.polygons.len());
            let mut push_poly = |poly: &Vec<Vec<(f64, f64)>>, lon_shift: f64| {
                let mut rings_px: Vec<Vec<(f64, f64)>> = Vec::with_capacity(poly.len());
                for (ring_idx, ring) in poly.iter().enumerate() {
                    let mut rpx: Vec<(f64, f64)> = Vec::with_capacity(ring.len());
                    if is_geo {
                        for &(lon, lat) in ring {
                            let (px, py) = project_pt(lon, lat, lon_shift);
                            if px.is_finite() && py.is_finite() {
                                rpx.push((px, py));
                            }
                        }
                    } else if ring.len() >= 3 {
                        // Adaptive midpoint densification for curved projected CRSs (Polar Stereo,
                        // LCC, LAEA, Albers, wide UTM) so straight lon/lat chords stay within 0.15 px.
                        fn push_densified(
                            a: (f64, f64),
                            pa: (f64, f64),
                            b: (f64, f64),
                            pb: (f64, f64),
                            depth: u8,
                            project: &impl Fn(f64, f64) -> (f64, f64),
                            out: &mut Vec<(f64, f64)>,
                        ) {
                            if depth < 6 {
                                let mid = (0.5 * (a.0 + b.0), 0.5 * (a.1 + b.1));
                                let pm = project(mid.0, mid.1);
                                if pm.0.is_finite() && pm.1.is_finite() {
                                    let lin_x = 0.5 * (pa.0 + pb.0);
                                    let lin_y = 0.5 * (pa.1 + pb.1);
                                    let err2 = (pm.0 - lin_x).powi(2) + (pm.1 - lin_y).powi(2);
                                    if err2 > 0.0225 {
                                        push_densified(a, pa, mid, pm, depth + 1, project, out);
                                        out.push(pm);
                                        push_densified(mid, pm, b, pb, depth + 1, project, out);
                                    }
                                }
                            }
                        }
                        let proj = |lon: f64, lat: f64| project_pt(lon, lat, lon_shift);
                        for i in 0..ring.len() {
                            let a = ring[i];
                            let b = ring[(i + 1) % ring.len()];
                            let pa = proj(a.0, a.1);
                            let pb = proj(b.0, b.1);
                            if pa.0.is_finite() && pa.1.is_finite() {
                                rpx.push(pa);
                                if pb.0.is_finite() && pb.1.is_finite() {
                                    push_densified(a, pa, b, pb, 0, &proj, &mut rpx);
                                }
                            }
                        }
                    }
                    if rpx.len() >= 3 {
                        rings_px.push(rpx);
                    } else if ring_idx == 0 {
                        // Never promote a hole ring to exterior when the outer ring is clipped away.
                        return;
                    }
                }
                if !rings_px.is_empty() {
                    polys_px.push(rings_px);
                }
            };
            for poly in &self.polygons {
                push_poly(poly, 0.0);
                if is_geo || is_merc {
                    if (is_geo && bbox[2] > 180.0) || wrap_lo {
                        push_poly(poly, 360.0);
                    }
                    if (is_geo && bbox[0] < -180.0) || wrap_hi {
                        push_poly(poly, -360.0);
                    }
                }
            }
            if !polys_px.is_empty() {
                let cov = crate::zonal::rasterize_exact_coverage(&polys_px, &[], &[], 0, 0, w, h);
                for (m, &c) in mask.iter_mut().zip(cov.iter()) {
                    if c > 0.0 {
                        *m = true;
                    }
                }
            }
        }
        let mut mark_segment = |ax: f64, ay: f64, bx: f64, by: f64| {
            let x0 = (ax - bbox[0]) / dlx;
            let y0 = (bbox[3] - ay) / dly;
            let x1 = (bx - bbox[0]) / dlx;
            let y1 = (bbox[3] - by) / dly;
            if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
                return;
            }
            let dx = x1 - x0;
            let dy = y1 - y0;
            let mut t0 = 0.0f64;
            let mut t1 = 1.0f64;
            for (p, q) in [
                (-dx, x0),
                (dx, w as f64 - x0),
                (-dy, y0),
                (dy, h as f64 - y0),
            ] {
                if p.abs() < 1e-15 {
                    if q < 0.0 {
                        return;
                    }
                } else {
                    let r = q / p;
                    if p < 0.0 {
                        if r > t1 {
                            return;
                        }
                        if r > t0 {
                            t0 = r;
                        }
                    } else {
                        if r < t0 {
                            return;
                        }
                        if r < t1 {
                            t1 = r;
                        }
                    }
                }
            }
            let sx = (x0 + t0 * dx).clamp(0.0, w as f64);
            let sy = (y0 + t0 * dy).clamp(0.0, h as f64);
            let ex = (x0 + t1 * dx).clamp(0.0, w as f64);
            let ey = (y0 + t1 * dy).clamp(0.0, h as f64);
            let mut ix = (sx.floor() as isize).clamp(0, w as isize - 1);
            let mut iy = (sy.floor() as isize).clamp(0, h as isize - 1);
            let end_x = (ex.floor() as isize).clamp(0, w as isize - 1);
            let end_y = (ey.floor() as isize).clamp(0, h as isize - 1);
            mask[iy as usize * w + ix as usize] = true;
            let sdx = ex - sx;
            let sdy = ey - sy;
            let step_x: isize = match end_x.cmp(&ix) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
            };
            let step_y: isize = match end_y.cmp(&iy) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
            };
            let mut t_max_x = if step_x > 0 {
                ((ix + 1) as f64 - sx) / sdx
            } else if step_x < 0 {
                (ix as f64 - sx) / sdx
            } else {
                f64::INFINITY
            };
            let mut t_max_y = if step_y > 0 {
                ((iy + 1) as f64 - sy) / sdy
            } else if step_y < 0 {
                (iy as f64 - sy) / sdy
            } else {
                f64::INFINITY
            };
            let t_delta_x = if step_x != 0 {
                (1.0 / sdx).abs()
            } else {
                f64::INFINITY
            };
            let t_delta_y = if step_y != 0 {
                (1.0 / sdy).abs()
            } else {
                f64::INFINITY
            };
            let max_steps = (w + h) * 2 + 4;
            for _ in 0..max_steps {
                if ix == end_x && iy == end_y {
                    break;
                }
                if t_max_x < t_max_y {
                    ix += step_x;
                    t_max_x += t_delta_x;
                } else if t_max_y < t_max_x {
                    iy += step_y;
                    t_max_y += t_delta_y;
                } else {
                    ix += step_x;
                    iy += step_y;
                    t_max_x += t_delta_x;
                    t_max_y += t_delta_y;
                }
                if ix < 0 || ix >= w as isize || iy < 0 || iy >= h as isize {
                    break;
                }
                mask[iy as usize * w + ix as usize] = true;
            }
        };
        for line in &self.lines {
            for seg in line.windows(2) {
                let (ax, ay) = dst_crs.from_lonlat_deg(seg[0].0, seg[0].1);
                let (bx, by) = dst_crs.from_lonlat_deg(seg[1].0, seg[1].1);
                mark_segment(ax, ay, bx, by);
            }
        }
        for &(lon, lat) in &self.points {
            let (px_crs, py_crs) = dst_crs.from_lonlat_deg(lon, lat);
            let fx = (px_crs - bbox[0]) / dlx;
            let fy = (bbox[3] - py_crs) / dly;
            if fx.is_finite() && fy.is_finite() {
                let ix = fx.floor() as isize;
                let iy = fy.floor() as isize;
                if ix >= 0 && ix < w as isize && iy >= 0 && iy < h as isize {
                    mask[iy as usize * w + ix as usize] = true;
                }
            }
        }
        mask
    }

    /// Rasterize the polygon into a compact 1-bit-per-pixel `Vec<u32>` mask (64× smaller than
    /// `Exact2D` `f32` coords) and return `(mask_bits, covered_pixel_count)`.
    pub fn rasterize_all_touched_bits(
        &self,
        w: usize,
        h: usize,
        bbox: [f64; 4],
        dst_crs: &cogzilla_proj::CrsModel,
    ) -> (Vec<u32>, usize) {
        let bools = self.rasterize_all_touched(w, h, bbox, dst_crs);
        let n = bools.len();
        let n_words = n.div_ceil(32);
        let mut words = vec![0u32; n_words];
        let pack_chunk = |word_start: usize, dst: &mut [u32]| -> usize {
            let mut covered = 0usize;
            for (i, w_out) in dst.iter_mut().enumerate() {
                let base = (word_start + i) * 32;
                let end = (base + 32).min(n);
                let mut acc = 0u32;
                for (bit, &b) in bools[base..end].iter().enumerate() {
                    if b {
                        acc |= 1u32 << bit;
                    }
                }
                *w_out = acc;
                covered += acc.count_ones() as usize;
            }
            covered
        };
        let covered = if n_words >= 4096 {
            let threads = std::thread::available_parallelism()
                .map(|p| p.get().clamp(2, 8))
                .unwrap_or(4);
            let words_per = n_words.div_ceil(threads);
            std::thread::scope(|s| {
                let handles: Vec<_> = words
                    .chunks_mut(words_per)
                    .enumerate()
                    .map(|(ci, chunk)| {
                        let w0 = ci * words_per;
                        s.spawn(move || pack_chunk(w0, chunk))
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap_or(0)).sum()
            })
        } else {
            pack_chunk(0, &mut words)
        };
        (words, covered)
    }
}
