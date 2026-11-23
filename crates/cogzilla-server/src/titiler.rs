use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_io::OpenedCog;
use cogzilla_kernel::jpeg_fast::{encode_jpeg_rgb, encode_jpeg_rgba};
use cogzilla_kernel::png_fast::{encode_filtered_png, filter_sub_rgb_row, filter_sub_rgba_row};
use cogzilla_kernel::webp_fast::{encode_webp_rgb, encode_webp_rgba};
use cogzilla_kernel::{
    BandSource, ColorFormula, ColormapLut, OutputFormat, ResampleMode, fast_db_from_power,
};
use cogzilla_tiff::PlanarConfig;

use crate::engine::{BandRequest, EngineError, TileEngine};

/// TiTiler-compatible `/cog/info` JSON payload built directly from our `CogIndex`.
pub fn build_cog_info(engine: &TileEngine, cog: &OpenedCog) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let full = cog.index.full_res();
    let bounds = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
    let coarsest = cog.index.ifds.last().unwrap();
    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
        .max(full.height as f64 / coarsest.height.max(1) as f64);
    let (minzoom, maxzoom) =
        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor);

    let count = full.samples_per_pixel as usize;
    let band_metadata: Vec<_> = (1..=count)
        .map(|i| serde_json::json!([format!("b{i}"), {}]))
        .collect();
    let band_descriptions: Vec<_> = (1..=count)
        .map(|i| serde_json::json!([format!("b{i}"), ""]))
        .collect();
    let colorinterp: Vec<&str> = match count {
        3 => vec!["red", "green", "blue"],
        4 => vec!["red", "green", "blue", "alpha"],
        _ => vec!["gray"; count],
    };
    let overviews: Vec<u32> = cog
        .index
        .ifds
        .iter()
        .skip(1)
        .map(|ifd| (full.width as f64 / ifd.width.max(1) as f64).round() as u32)
        .collect();

    let (nodata_type, nodata_value) = match cog.index.nodata {
        Some(v) if v.is_nan() => ("Nodata", serde_json::json!("nan")),
        Some(v) => ("Nodata", serde_json::json!(v)),
        None => ("None", serde_json::Value::Null),
    };

    let (x0, y0) = full.pixel_to_crs.forward(0.0, 0.0);
    let (x1, y1) = full
        .pixel_to_crs
        .forward(full.width as f64, full.height as f64);
    let native_bounds = [x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)];

    Ok(serde_json::json!({
        "bounds": native_bounds,
        "wgs84_bounds": bounds,
        "minzoom": minzoom,
        "maxzoom": maxzoom,
        "band_metadata": band_metadata,
        "band_descriptions": band_descriptions,
        "dtype": full.sample_type.gdal_name(),
        "nodata_type": nodata_type,
        "nodata_value": nodata_value,
        "colorinterp": colorinterp,
        "driver": "GTiff",
        "count": count,
        "width": full.width,
        "height": full.height,
        "overviews": overviews,
    }))
}

/// TiTiler-compatible `/cog/info.geojson` Feature wrapping the raster footprint polygon.
pub fn build_cog_info_geojson(
    engine: &TileEngine,
    cog: &OpenedCog,
) -> Result<serde_json::Value, EngineError> {
    let info = build_cog_info(engine, cog)?;
    let b = info["wgs84_bounds"].as_array().unwrap();
    let (w, s, e, n) = (
        b[0].as_f64().unwrap(),
        b[1].as_f64().unwrap(),
        b[2].as_f64().unwrap(),
        b[3].as_f64().unwrap(),
    );
    Ok(serde_json::json!({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
        },
        "properties": info,
    }))
}

/// TiTiler-compatible TileJSON 2.2.0 (`/cog/{tms}/tilejson.json`).
pub fn build_tilejson(
    engine: &TileEngine,
    cog: &OpenedCog,
    host: &str,
    tms: &str,
    raw_query: &str,
) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let full = cog.index.full_res();
    let bounds = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
    let coarsest = cog.index.ifds.last().unwrap();
    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
        .max(full.height as f64 / coarsest.height.max(1) as f64);
    let (minzoom, maxzoom) =
        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor);
    let center = [
        0.5 * (bounds[0] + bounds[2]),
        0.5 * (bounds[1] + bounds[3]),
        minzoom as f64,
    ];
    let tile_url = if raw_query.is_empty() {
        format!("http://{host}/cog/tiles/{tms}/{{z}}/{{x}}/{{y}}")
    } else {
        format!("http://{host}/cog/tiles/{tms}/{{z}}/{{x}}/{{y}}?{raw_query}")
    };
    Ok(serde_json::json!({
        "tilejson": "3.0.0",
        "name": cog.uri,
        "version": "1.0.0",
        "scheme": "xyz",
        "tiles": [tile_url],
        "minzoom": minzoom,
        "maxzoom": maxzoom,
        "bounds": bounds,
        "center": center,
    }))
}

/// TiTiler-compatible `/cog/point/{lon},{lat}` exact point query (fetches only 1 internal tile!).
pub async fn query_point(
    engine: &TileEngine,
    cog: &OpenedCog,
    lon: f64,
    lat: f64,
) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let ifd = cog.index.full_res();
    let (sx, sy) = crs.from_lonlat_deg(lon, lat);
    let (col_f, row_f) = ifd.crs_to_pixel.forward(sx, sy);
    let col = col_f.floor() as i64;
    let row = row_f.floor() as i64;

    let count = ifd.samples_per_pixel as usize;
    let band_names: Vec<String> = (1..=count).map(|i| format!("b{i}")).collect();

    if col < 0 || row < 0 || col >= ifd.width as i64 || row >= ifd.height as i64 {
        return Err(EngineError::NotFound(format!(
            "point ({lon}, {lat}) is outside dataset bounds"
        )));
    }
    let tx = (col as u32) / ifd.tile_width;
    let ty = (row as u32) / ifd.tile_height;
    let lx = (col as u32) % ifd.tile_width;
    let ly = (row as u32) % ifd.tile_height;
    let pidx = (ly as usize) * (ifd.tile_width as usize) + (lx as usize);

    let mut values: Vec<serde_json::Value> = Vec::with_capacity(count);
    match ifd.planar_config {
        PlanarConfig::Chunky => {
            let idx = ifd.tile_index(tx, ty, 0) as u32;
            let tiles = engine.store.get_blocks(cog, 0, &[Some(idx)]).await?;
            for b in 0..count {
                let v = tiles[0]
                    .as_ref()
                    .and_then(|t| t.bands.get(b))
                    .map(|band| band.data.get_f32(pidx))
                    .unwrap_or(f32::NAN);
                values.push(if v.is_nan() {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(v)
                });
            }
        }
        PlanarConfig::Planar => {
            let indices: Vec<Option<u32>> = (0..count)
                .map(|b| Some(ifd.tile_index(tx, ty, b as u16) as u32))
                .collect();
            let tiles = engine.store.get_blocks(cog, 0, &indices).await?;
            for b in 0..count {
                let v = tiles[b]
                    .as_ref()
                    .and_then(|t| t.bands.first())
                    .map(|band| band.data.get_f32(pidx))
                    .unwrap_or(f32::NAN);
                values.push(if v.is_nan() {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(v)
                });
            }
        }
    }

    Ok(serde_json::json!({
        "coordinates": [lon, lat],
        "values": values,
        "band_names": band_names,
    }))
}

/// TiTiler-compatible `/cog/statistics` computed from the coarsest overview <= `max_size`.
/// When `poly_mask` is `Some`, clips statistics strictly to pixels inside the posted GeoJSON polygon.
pub async fn compute_statistics(
    engine: &TileEngine,
    cog: &OpenedCog,
    max_size: u32,
    poly_mask: Option<&PolygonMask>,
    unscale: bool,
) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let mut ifd_idx = cog.index.ifds.len() - 1;
    for (i, ifd) in cog.index.ifds.iter().enumerate() {
        if ifd.width.max(ifd.height) <= max_size {
            ifd_idx = i;
            break;
        }
    }
    let ifd = &cog.index.ifds[ifd_idx];
    let count = ifd.samples_per_pixel as usize;
    let total_pixels = (ifd.width as usize) * (ifd.height as usize);
    let mut mask_total_pixels = 0usize;

    let mut per_band: Vec<Vec<f32>> = vec![Vec::with_capacity(total_pixels); count];
    match ifd.planar_config {
        PlanarConfig::Chunky => {
            let mut indices = Vec::new();
            for ty in 0..ifd.tiles_down {
                for tx in 0..ifd.tiles_across {
                    indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                }
            }
            let tiles = engine.store.get_blocks(cog, ifd_idx, &indices).await?;
            for (ti, tile_opt) in tiles.iter().enumerate() {
                let Some(tile) = tile_opt else { continue };
                let tx = (ti as u32) % ifd.tiles_across;
                let ty = (ti as u32) / ifd.tiles_across;
                let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
                let y_max = (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;
                let tw = ifd.tile_width as usize;
                for ly in 0..y_max {
                    let row_pix = (ty * ifd.tile_height) as usize + ly;
                    for lx in 0..x_max {
                        let col_pix = (tx * ifd.tile_width) as usize + lx;
                        if let Some(pm) = poly_mask {
                            let (sx, sy) = ifd.pixel_to_crs.forward(col_pix as f64 + 0.5, row_pix as f64 + 0.5);
                            let (lon, lat) = crs.to_lonlat_deg(sx, sy);
                            if !pm.contains(lon, lat) {
                                continue;
                            }
                            mask_total_pixels += 1;
                        }
                        let pidx = ly * tw + lx;
                        for b in 0..count {
                            let mut v = tile.bands[b].data.get_f32(pidx);
                            if v.is_finite() {
                                if unscale {
                                    v = v * cog.index.scales[b] + cog.index.offsets[b];
                                }
                                per_band[b].push(v);
                            }
                        }
                    }
                }
            }
        }
        PlanarConfig::Planar => {
            for b in 0..count {
                let mut indices = Vec::new();
                for ty in 0..ifd.tiles_down {
                    for tx in 0..ifd.tiles_across {
                        indices.push(Some(ifd.tile_index(tx, ty, b as u16) as u32));
                    }
                }
                let tiles = engine.store.get_blocks(cog, ifd_idx, &indices).await?;
                for (ti, tile_opt) in tiles.iter().enumerate() {
                    let Some(tile) = tile_opt else { continue };
                    let band = &tile.bands[0];
                    if band.stats.all_nodata {
                        continue;
                    }
                    let tx = (ti as u32) % ifd.tiles_across;
                    let ty = (ti as u32) / ifd.tiles_across;
                    let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
                    let y_max = (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;
                    let tw = ifd.tile_width as usize;
                    for ly in 0..y_max {
                        let row_pix = (ty * ifd.tile_height) as usize + ly;
                        for lx in 0..x_max {
                            let col_pix = (tx * ifd.tile_width) as usize + lx;
                            if let Some(pm) = poly_mask {
                                let (sx, sy) = ifd.pixel_to_crs.forward(col_pix as f64 + 0.5, row_pix as f64 + 0.5);
                                let (lon, lat) = crs.to_lonlat_deg(sx, sy);
                                if !pm.contains(lon, lat) {
                                    continue;
                                }
                                if b == 0 {
                                    mask_total_pixels += 1;
                                }
                            }
                            let mut v = band.data.get_f32(ly * tw + lx);
                            if v.is_finite() {
                                if unscale {
                                    v = v * cog.index.scales[b] + cog.index.offsets[b];
                                }
                                per_band[b].push(v);
                            }
                        }
                    }
                }
            }
        }
    }

    let denom_pixels = if poly_mask.is_some() {
        mask_total_pixels.max(1)
    } else {
        total_pixels
    };

    let mut out = serde_json::Map::new();
    for (b, vals) in per_band.iter_mut().enumerate() {
        let valid_px = vals.len();
        let masked_px = denom_pixels.saturating_sub(valid_px);
        let valid_pct = if denom_pixels > 0 {
            (valid_px as f64 / denom_pixels as f64) * 100.0
        } else {
            0.0
        };
        if valid_px == 0 {
            out.insert(
                format!("b{}", b + 1),
                serde_json::json!({
                    "min": null, "max": null, "mean": null, "count": 0, "sum": 0.0,
                    "std": null, "median": null, "valid_percent": 0.0,
                    "masked_pixels": masked_px, "valid_pixels": 0,
                    "percentile_2": null, "percentile_98": null,
                    "histogram": [[], []],
                }),
            );
            continue;
        }
        vals.sort_unstable_by(|a, b| a.total_cmp(b));
        let min = vals[0];
        let max = vals[valid_px - 1];
        let median = vals[valid_px / 2];
        let p2 = vals[((valid_px as f64) * 0.02) as usize];
        let p98 = vals[(((valid_px as f64) * 0.98) as usize).min(valid_px - 1)];
        let sum: f64 = vals.iter().map(|&v| v as f64).sum();
        let mean = sum / (valid_px as f64);
        let var: f64 = vals
            .iter()
            .map(|&v| {
                let d = (v as f64) - mean;
                d * d
            })
            .sum::<f64>()
            / (valid_px as f64);
        let std = var.sqrt();

        // 10-bin histogram matching TiTiler's default `bins=10`
        let mut counts = [0u64; 10];
        let span = (max - min).max(1e-12);
        for &v in vals.iter() {
            let bin = (((v - min) / span) * 10.0).floor().clamp(0.0, 9.0) as usize;
            counts[bin] += 1;
        }
        let edges: Vec<f32> = (0..=10).map(|i| min + span * (i as f32 / 10.0)).collect();

        out.insert(
            format!("b{}", b + 1),
            serde_json::json!({
                "min": min,
                "max": max,
                "mean": mean,
                "count": valid_px,
                "sum": sum,
                "std": std,
                "median": median,
                "valid_percent": valid_pct,
                "masked_pixels": masked_px,
                "valid_pixels": valid_px,
                "percentile_2": p2,
                "percentile_98": p98,
                "histogram": [counts, edges],
            }),
        );
    }

    Ok(serde_json::Value::Object(out))
}

/// Simple ray-casting point-in-polygon check for GeoJSON Feature crop masking (`POST /cog/feature`).
pub struct PolygonMask {
    /// List of polygons; each polygon is `[outer_ring, hole1, ...]`, each ring is `Vec<(lon, lat)>`.
    pub polygons: Vec<Vec<Vec<(f64, f64)>>>,
}

impl PolygonMask {
    pub fn from_geojson(val: &serde_json::Value) -> Result<(Self, [f64; 4]), EngineError> {
        let geom = if val.get("type").and_then(|v| v.as_str()) == Some("Feature") {
            val.get("geometry")
                .ok_or_else(|| EngineError::BadRequest("GeoJSON Feature missing geometry".into()))?
        } else {
            val
        };
        let gtype = geom
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| EngineError::BadRequest("GeoJSON missing geometry.type".into()))?;
        let coords = geom
            .get("coordinates")
            .ok_or_else(|| EngineError::BadRequest("GeoJSON missing coordinates".into()))?;

        let parse_ring = |arr: &serde_json::Value| -> Result<Vec<(f64, f64)>, EngineError> {
            let list = arr
                .as_array()
                .ok_or_else(|| EngineError::BadRequest("invalid GeoJSON ring".into()))?;
            let mut ring = Vec::with_capacity(list.len());
            for pt in list {
                let p = pt
                    .as_array()
                    .ok_or_else(|| EngineError::BadRequest("invalid GeoJSON coord".into()))?;
                if p.len() < 2 {
                    return Err(EngineError::BadRequest("coord needs [lon, lat]".into()));
                }
                ring.push((
                    p[0].as_f64().unwrap_or(0.0),
                    p[1].as_f64().unwrap_or(0.0),
                ));
            }
            Ok(ring)
        };

        let mut polygons = Vec::new();
        match gtype {
            "Polygon" => {
                let rings = coords
                    .as_array()
                    .ok_or_else(|| EngineError::BadRequest("invalid Polygon coords".into()))?;
                let mut poly = Vec::new();
                for r in rings {
                    poly.push(parse_ring(r)?);
                }
                polygons.push(poly);
            }
            "MultiPolygon" => {
                let polys = coords
                    .as_array()
                    .ok_or_else(|| EngineError::BadRequest("invalid MultiPolygon coords".into()))?;
                for p in polys {
                    let rings = p
                        .as_array()
                        .ok_or_else(|| EngineError::BadRequest("invalid polygon".into()))?;
                    let mut poly = Vec::new();
                    for r in rings {
                        poly.push(parse_ring(r)?);
                    }
                    polygons.push(poly);
                }
            }
            other => {
                return Err(EngineError::BadRequest(format!(
                    "unsupported GeoJSON geometry type '{other}' (expected Polygon or MultiPolygon)"
                )));
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
        if !min_x.is_finite() {
            return Err(EngineError::BadRequest("empty GeoJSON polygon".into()));
        }
        Ok((Self { polygons }, [min_x, min_y, max_x, max_y]))
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

    #[inline(always)]
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        for poly in &self.polygons {
            if let Some(outer) = poly.first() {
                if Self::ring_contains(outer, lon, lat) {
                    let in_hole = poly[1..].iter().any(|h| Self::ring_contains(h, lon, lat));
                    if !in_hole {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Arbitrary-dimensions WGS84 bounding-box renderer backing `/cog/preview`, `/cog/bbox`,
/// `POST /cog/feature`, `WorldCRS84Quad` tiles, and custom post-processing (`color_formula`,
/// custom JSON `?colormap=`, `?return_mask=false`, `?nodata=`, `?buffer=`).
#[allow(clippy::too_many_arguments)]
pub async fn render_wgs84_bbox(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4], // [min_lon, min_lat, max_lon, max_lat]
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&ColormapLut>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
) -> Result<Bytes, EngineError> {
    let w = width.clamp(1, 4096) as usize;
    let h = height.clamp(1, 4096) as usize;
    let primary = &cogs[0];
    let crs = engine.get_or_build_crs(primary)?;

    // Estimate target CRS resolution at bbox center to pick optimal overview IFD
    let cx = 0.5 * (bbox[0] + bbox[2]);
    let cy = 0.5 * (bbox[1] + bbox[3]);
    let dlx = (bbox[2] - bbox[0]) / (w as f64);
    let dly = (bbox[3] - bbox[1]) / (h as f64);
    let (s0x, s0y) = crs.from_lonlat_deg(cx, cy);
    let (s1x, s1y) = crs.from_lonlat_deg(cx + dlx, cy);
    let (s2x, s2y) = crs.from_lonlat_deg(cx, cy + dly);
    let target_rx = ((s1x - s0x).powi(2) + (s1y - s0y).powi(2)).sqrt();
    let target_ry = ((s2x - s0x).powi(2) + (s2y - s0y).powi(2)).sqrt();
    let ifd_idx = primary.index.select_ifd(target_rx, target_ry);
    let ifd = &primary.index.ifds[ifd_idx];

    // Precompute source pixel coordinates `(u, v)` and find intersecting internal tile bounds
    let mut u_map = vec![f32::NAN; w * h];
    let mut v_map = vec![f32::NAN; w * h];
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;

    for py in 0..h {
        let lat = bbox[3] - (py as f64 + 0.5) * dly;
        let row_off = py * w;
        for px in 0..w {
            let lon = bbox[0] + (px as f64 + 0.5) * dlx;
            if let Some(pm) = poly_mask {
                if !pm.contains(lon, lat) {
                    continue;
                }
            }
            let (sx, sy) = crs.from_lonlat_deg(lon, lat);
            if !sx.is_finite() || !sy.is_finite() {
                continue;
            }
            let (col, row) = ifd.crs_to_pixel.forward(sx, sy);
            let uu = (col - 0.5) as f32;
            let vv = (row - 0.5) as f32;
            if uu >= -1.0 && uu <= ifd.width as f32 && vv >= -1.0 && vv <= ifd.height as f32 {
                u_map[row_off + px] = uu;
                v_map[row_off + px] = vv;
                min_u = min_u.min(uu);
                max_u = max_u.max(uu);
                min_v = min_v.min(vv);
                max_v = max_v.max(vv);
            }
        }
    }

    // Empty check
    if !min_u.is_finite() {
        let raw = vec![0u8; h * (1 + w * 4)];
        return Ok(match format {
            OutputFormat::Png => encode_filtered_png(w as u32, h as u32, 6, &raw),
            OutputFormat::Webp { quality } => {
                encode_webp_rgba(w as u32, h as u32, &vec![0u8; w * h * 4], quality)
            }
            OutputFormat::Jpeg { quality } => {
                encode_jpeg_rgb(w as u32, h as u32, &vec![0u8; w * h * 3], quality)
            }
            OutputFormat::RawF32 => {
                let mut b = BytesMut::with_capacity(w * h * 4);
                for _ in 0..(w * h) {
                    b.put_slice(&f32::NAN.to_le_bytes());
                }
                b.freeze()
            }
            OutputFormat::RawRgba => Bytes::from(vec![0u8; w * h * 4]),
        });
    }

    let c0 = (min_u.floor() as i32 - 1).max(0) as u32;
    let c1 = (max_u.ceil() as i32 + 1).min(ifd.width as i32 - 1).max(0) as u32;
    let r0 = (min_v.floor() as i32 - 1).max(0) as u32;
    let r1 = (max_v.ceil() as i32 + 1).min(ifd.height as i32 - 1).max(0) as u32;
    let tx0 = c0 / ifd.tile_width;
    let tx1 = (c1 / ifd.tile_width).min(ifd.tiles_across.saturating_sub(1));
    let ty0 = r0 / ifd.tile_height;
    let ty1 = (r1 / ifd.tile_height).min(ifd.tiles_down.saturating_sub(1));
    let grid_w = tx1 - tx0 + 1;
    let grid_h = ty1 - ty0 + 1;

    // Fetch needed bands (single COG or multi-COG)
    let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
    for ty in ty0..=ty1 {
        for tx in tx0..=tx1 {
            indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
        }
    }
    let mut cog_tiles = Vec::with_capacity(cogs.len());
    for c in cogs {
        cog_tiles.push(engine.store.get_blocks(c, ifd_idx, &indices).await?);
    }

    let sample = |src: &BandSource<'_>, u: f32, v: f32| -> f32 {
        if !u.is_finite() || !v.is_finite() {
            return f32::NAN;
        }
        if resample == ResampleMode::Nearest {
            return src.sample_checked((u + 0.5).floor() as i32, (v + 0.5).floor() as i32);
        }
        if resample == ResampleMode::Cubic {
            return cogzilla_kernel::sample_cubic(src, u, v);
        }
        let x0_f = u.floor();
        let y0_f = v.floor();
        let fx = u - x0_f;
        let fy = v - y0_f;
        let x0 = x0_f as i32;
        let y0 = y0_f as i32;
        let v00 = src.sample_checked(x0, y0);
        let v10 = src.sample_checked(x0 + 1, y0);
        let v01 = src.sample_checked(x0, y0 + 1);
        let v11 = src.sample_checked(x0 + 1, y0 + 1);
        let (mut sum, mut wsum) = (0.0f32, 0.0f32);
        for (val, wt) in [
            (v00, (1.0 - fx) * (1.0 - fy)),
            (v10, fx * (1.0 - fy)),
            (v01, (1.0 - fx) * fy),
            (v11, fx * fy),
        ] {
            if !val.is_nan() {
                sum += val * wt;
                wsum += wt;
            }
        }
        if wsum > 1e-6 { sum / wsum } else { f32::NAN }
    };

    // Helper to build BandSource for 1-based band index across single or multi-COG
    let get_band_refs = |b_1based: u16, asset_slot: usize| -> Vec<Option<&cogzilla_codec::DecodedBand>> {
        if cogs.len() > 1 {
            let slot = asset_slot.min(cogs.len() - 1);
            cog_tiles[slot]
                .iter()
                .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                .collect()
        } else {
            let b0 = b_1based.saturating_sub(1) as usize;
            cog_tiles[0]
                .iter()
                .map(|o| o.as_ref().and_then(|t| t.bands.get(b0)))
                .collect()
        }
    };

    let mut rgba = vec![0u8; w * h * 4];
    match bands {
        BandRequest::Single {
            band_1based,
            rescale,
            colormap,
        } => {
            let refs = get_band_refs(*band_1based, 0);
            let src = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h,
                &refs,
            )
            .with_nodata_override(nodata_override);
            let lut = custom_lut.unwrap_or_else(|| colormap.lut());
            let inv = 255.0 / (rescale.1 - rescale.0).max(1e-12);
            for i in 0..(w * h) {
                let v = sample(&src, u_map[i], v_map[i]);
                if !v.is_nan() {
                    let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                    let c = lut.rgb[idx];
                    rgba[i * 4] = c[0];
                    rgba[i * 4 + 1] = c[1];
                    rgba[i * 4 + 2] = c[2];
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::Terrain {
            band_1based,
            encoding,
        } => {
            let refs = get_band_refs(*band_1based, 0);
            let src = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h,
                &refs,
            )
            .with_nodata_override(nodata_override);
            for i in 0..(w * h) {
                let v = sample(&src, u_map[i], v_map[i]);
                if !v.is_nan() {
                    let [r, g, b] = match encoding {
                        cogzilla_kernel::TerrainEncoding::Terrarium => {
                            cogzilla_kernel::encode_terrarium_rgb(v)
                        }
                        cogzilla_kernel::TerrainEncoding::TerrainRgb => {
                            cogzilla_kernel::encode_terrain_rgb(v)
                        }
                    };
                    rgba[i * 4] = r;
                    rgba[i * 4 + 1] = g;
                    rgba[i * 4 + 2] = b;
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::Rgb {
            bands_1based,
            rescale_min,
            rescale_max,
        } => {
            let r0 = get_band_refs(bands_1based[0], 0);
            let r1 = get_band_refs(bands_1based[1], 1);
            let r2 = get_band_refs(bands_1based[2], 2);
            let s0 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r0,
            );
            let s1 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r1,
            );
            let s2 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r2,
            );
            let inv = [
                255.0 / (rescale_max[0] - rescale_min[0]).max(1e-12),
                255.0 / (rescale_max[1] - rescale_min[1]).max(1e-12),
                255.0 / (rescale_max[2] - rescale_min[2]).max(1e-12),
            ];
            for i in 0..(w * h) {
                let r = sample(&s0, u_map[i], v_map[i]);
                let g = sample(&s1, u_map[i], v_map[i]);
                let b = sample(&s2, u_map[i], v_map[i]);
                if !r.is_nan() && !g.is_nan() && !b.is_nan() {
                    rgba[i * 4] = ((r - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 1] = ((g - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 2] = ((b - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::SarRgb {
            bands_1based,
            rescale_min,
            rescale_max,
        } => {
            let r0 = get_band_refs(bands_1based[0], 0);
            let r1 = get_band_refs(bands_1based[1], 1);
            let s0 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r0,
            );
            let s1 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r1,
            );
            let inv = [
                255.0 / (rescale_max[0] - rescale_min[0]).max(1e-12),
                255.0 / (rescale_max[1] - rescale_min[1]).max(1e-12),
                255.0 / (rescale_max[2] - rescale_min[2]).max(1e-12),
            ];
            for i in 0..(w * h) {
                let vv = fast_db_from_power(sample(&s0, u_map[i], v_map[i]));
                let vh = fast_db_from_power(sample(&s1, u_map[i], v_map[i]));
                if !vv.is_nan() && !vh.is_nan() {
                    rgba[i * 4] = ((vv - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 1] = ((vh - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 2] = (((vv - vh) - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::CustomExpr {
            prog,
            rescale,
            colormap,
        } => {
            let ref_vecs: Vec<_> = prog
                .used_bands
                .iter()
                .enumerate()
                .map(|(slot, &b)| get_band_refs(b, slot))
                .collect();
            let srcs: Vec<_> = ref_vecs
                .iter()
                .map(|r| {
                    BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                        grid_h, r,
                    )
                })
                .collect();
            let lut = colormap.lut();
            let inv = 255.0 / (rescale.1 - rescale.0).max(1e-12);
            let mut vals = [0.0f32; 16];
            let nb = srcs.len().min(16);
            for i in 0..(w * h) {
                for k in 0..nb {
                    vals[k] = sample(&srcs[k], u_map[i], v_map[i]);
                }
                let v = prog.eval(&vals[..nb]);
                if !v.is_nan() {
                    let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                    let c = lut.rgb[idx];
                    rgba[i * 4] = c[0];
                    rgba[i * 4 + 1] = c[1];
                    rgba[i * 4 + 2] = c[2];
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::Spectral1 {
            expr,
            band_1based,
            rescale,
            colormap,
        } => {
            let refs = get_band_refs(*band_1based, 0);
            let src = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h,
                &refs,
            )
            .with_nodata_override(nodata_override);
            let lut = custom_lut.unwrap_or_else(|| colormap.lut());
            let inv = 255.0 / (rescale.1 - rescale.0).max(1e-12);
            for i in 0..(w * h) {
                let v = expr.eval1(sample(&src, u_map[i], v_map[i]));
                if !v.is_nan() {
                    let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                    let c = lut.rgb[idx];
                    rgba[i * 4] = c[0];
                    rgba[i * 4 + 1] = c[1];
                    rgba[i * 4 + 2] = c[2];
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::Spectral2 {
            expr,
            bands_1based,
            rescale,
            colormap,
        } => {
            let r0 = get_band_refs(bands_1based[0], 0);
            let r1 = get_band_refs(bands_1based[1], 1);
            let s0 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r0,
            );
            let s1 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r1,
            );
            let lut = colormap.lut();
            let inv = 255.0 / (rescale.1 - rescale.0).max(1e-12);
            for i in 0..(w * h) {
                let a = sample(&s0, u_map[i], v_map[i]);
                let b = sample(&s1, u_map[i], v_map[i]);
                let v = if a.is_nan() || b.is_nan() {
                    f32::NAN
                } else {
                    expr.eval2(a, b)
                };
                if !v.is_nan() {
                    let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                    let c = lut.rgb[idx];
                    rgba[i * 4] = c[0];
                    rgba[i * 4 + 1] = c[1];
                    rgba[i * 4 + 2] = c[2];
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::Spectral3 {
            expr,
            bands_1based,
            rescale,
            colormap,
        } => {
            let r0 = get_band_refs(bands_1based[0], 0);
            let r1 = get_band_refs(bands_1based[1], 1);
            let r2 = get_band_refs(bands_1based[2], 2);
            let s0 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r0,
            );
            let s1 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r1,
            );
            let s2 = BandSource::new(
                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, &r2,
            );
            let lut = colormap.lut();
            let inv = 255.0 / (rescale.1 - rescale.0).max(1e-12);
            for i in 0..(w * h) {
                let a = sample(&s0, u_map[i], v_map[i]);
                let b = sample(&s1, u_map[i], v_map[i]);
                let c = sample(&s2, u_map[i], v_map[i]);
                let v = if a.is_nan() || b.is_nan() || c.is_nan() {
                    f32::NAN
                } else {
                    expr.eval3(a, b, c)
                };
                if !v.is_nan() {
                    let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                    let col = lut.rgb[idx];
                    rgba[i * 4] = col[0];
                    rgba[i * 4 + 1] = col[1];
                    rgba[i * 4 + 2] = col[2];
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
    }

    if let Some(cf) = color_formula {
        cf.apply_buffer_in_place(&mut rgba, 4);
    }

    match format {
        OutputFormat::Png => {
            if !return_mask {
                let mut rgb = vec![0u8; w * h * 3];
                for i in 0..(w * h) {
                    if rgba[i * 4 + 3] != 0 {
                        rgb[i * 3] = rgba[i * 4];
                        rgb[i * 3 + 1] = rgba[i * 4 + 1];
                        rgb[i * 3 + 2] = rgba[i * 4 + 2];
                    }
                }
                let stride = 1 + w * 3;
                let mut filtered = vec![0u8; h * stride];
                for py in 0..h {
                    filter_sub_rgb_row(
                        &rgb[py * w * 3..(py + 1) * w * 3],
                        &mut filtered[py * stride..(py + 1) * stride],
                    );
                }
                Ok(encode_filtered_png(w as u32, h as u32, 2, &filtered))
            } else {
                let stride = 1 + w * 4;
                let mut filtered = vec![0u8; h * stride];
                for py in 0..h {
                    filter_sub_rgba_row(
                        &rgba[py * w * 4..(py + 1) * w * 4],
                        &mut filtered[py * stride..(py + 1) * stride],
                    );
                }
                Ok(encode_filtered_png(w as u32, h as u32, 6, &filtered))
            }
        }
        OutputFormat::Webp { quality } => {
            if !return_mask {
                let mut rgb = vec![0u8; w * h * 3];
                for i in 0..(w * h) {
                    if rgba[i * 4 + 3] != 0 {
                        rgb[i * 3] = rgba[i * 4];
                        rgb[i * 3 + 1] = rgba[i * 4 + 1];
                        rgb[i * 3 + 2] = rgba[i * 4 + 2];
                    }
                }
                Ok(encode_webp_rgb(w as u32, h as u32, &rgb, quality))
            } else {
                Ok(encode_webp_rgba(w as u32, h as u32, &rgba, quality))
            }
        }
        OutputFormat::Jpeg { quality } => Ok(encode_jpeg_rgba(w as u32, h as u32, &rgba, quality)),
        OutputFormat::RawF32 | OutputFormat::RawRgba => Ok(Bytes::from(rgba)),
    }
}

/// Interactive MapLibre HTML viewer (`/cog/{tms}/map.html`), matching TiTiler's built-in viewer.
pub fn build_map_html(tilejson_url: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>cogzilla Map Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <style>body {{ margin: 0; padding: 0; }} #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}</style>
</head>
<body>
  <div id="map"></div>
  <script>
    fetch({tilejson_url:?}).then(r => r.json()).then(tj => {{
      const map = new maplibregl.Map({{
        container: 'map',
        style: {{
          version: 8,
          sources: {{
            osm: {{ type: 'raster', tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'], tileSize: 256 }},
            cog: {{ type: 'raster', tiles: tj.tiles, tileSize: 256, bounds: tj.bounds }}
          }},
          layers: [
            {{ id: 'osm', type: 'raster', source: 'osm' }},
            {{ id: 'cog', type: 'raster', source: 'cog' }}
          ]
        }},
        center: [tj.center[0], tj.center[1]],
        zoom: tj.center[2]
      }});
      map.fitBounds([[tj.bounds[0], tj.bounds[1]], [tj.bounds[2], tj.bounds[3]]], {{ padding: 40 }});
    }});
  </script>
</body>
</html>"#
    )
}
